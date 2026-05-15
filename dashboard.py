# ============================================================
# dashboard.py — ClassWatch Web Dashboard & API
# Fixes: JWT token correctly sent on all requests,
#        session start/stop lifecycle, GPU status,
#        analytics endpoints, camera deferred to session start
# ============================================================

import os
import threading
import webbrowser
import queue
import cv2
import statistics
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory, g
from flask_socketio import SocketIO
from flask_cors import CORS

from database import (db, init_db, Institution, School, Class,
                       User, Session as DbSession, AttentionSnapshot, Alert)
from auth import auth_bp, require_auth, require_role
from config import (
    WEB_PORT, WEB_HOST, SECRET_KEY, DATABASE_URL,
    STREAM_PASSWORD, DISTRACTION_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT,
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ── Flask ─────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config.update(
    SECRET_KEY                     = SECRET_KEY,
    SQLALCHEMY_DATABASE_URI        = DATABASE_URL,
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    SQLALCHEMY_ENGINE_OPTIONS      = {"pool_pre_ping": True, "pool_recycle": 300},
)
CORS(app, origins="*", supports_credentials=True,
     allow_headers=["Authorization", "Content-Type"],
     expose_headers=["Authorization"])
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
app.register_blueprint(auth_bp)
@app.route("/api/debug/token", methods=["GET", "POST"])
def debug_token():
    auth = request.headers.get("Authorization", "MISSING")
    return jsonify({"auth_header": auth, "all_headers": dict(request.headers)})

# ── MJPEG ─────────────────────────────────────────────────────
_frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=1)
_last_jpeg:   bytes = b""
_frame_lock   = threading.Lock()
_cam_enabled  = True

# ── Session control ───────────────────────────────────────────
_session_running = False
_session_lock    = threading.Lock()

# ── Resolution control ────────────────────────────────────────
RESOLUTIONS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
_target_resolution = (FRAME_WIDTH, FRAME_HEIGHT)
_current_res_key   = "720p"
_resolution_lock   = threading.Lock()

# ── GPU status ────────────────────────────────────────────────
_gpu_device = "cpu"

# ── Live state ────────────────────────────────────────────────
_live = {
    "session_db_id":   None,
    "pct":             0.0,
    "avg_pct":         0.0,
    "attentive":       0,
    "total":           0,
    "fps":             0.0,
    "start_time":      "—",
    "timeline":        [],
    "distraction_log": [],
    "peak_pct":        0.0,
    "peak_time":       "—",
    "low_pct":         100.0,
    "low_time":        "—",
    "alert_count":     0,
}
_live_lock           = threading.Lock()
_last_was_distracted = False
_stop_event_ref      = None
_snapshot_counter    = 0
_teacher_id_ref      = 1
_class_id_ref        = None


# ── Helpers called from main.py ───────────────────────────────

def get_session_running() -> bool:
    with _session_lock:
        return _session_running

def get_target_resolution() -> tuple:
    with _resolution_lock:
        return _target_resolution

def notify_gpu_status(device: str):
    global _gpu_device
    _gpu_device = device

def register_shutdown(event):
    global _stop_event_ref
    _stop_event_ref = event


# ── SPA ───────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path):
    full = os.path.join(FRONTEND_DIR, path)
    if path and os.path.exists(full):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── MJPEG stream ──────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    if STREAM_PASSWORD and request.args.get("pwd") != STREAM_PASSWORD:
        return Response("Unauthorized", status=401)

    def generate():
        global _last_jpeg
        while True:
            if not _cam_enabled:
                import time; time.sleep(0.1); continue
            try:
                jpeg = _frame_queue.get(timeout=5.0)
                with _frame_lock:
                    _last_jpeg = jpeg
            except queue.Empty:
                with _frame_lock:
                    jpeg = _last_jpeg
                if not jpeg:
                    continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Camera control ────────────────────────────────────────────

@app.route("/api/camera/toggle", methods=["POST"])
@require_auth
def toggle_camera():
    global _cam_enabled
    data = request.get_json(silent=True) or {}
    _cam_enabled = bool(data.get("enabled", True))
    return jsonify({"ok": True, "enabled": _cam_enabled})


@app.route("/api/camera/resolution", methods=["POST"])
@require_auth
def set_resolution():
    global _target_resolution, _current_res_key
    data = request.get_json(silent=True) or {}
    key  = data.get("resolution", "720p")
    if key not in RESOLUTIONS:
        return jsonify({"error": f"Unknown resolution '{key}'"}), 400
    with _resolution_lock:
        _target_resolution = RESOLUTIONS[key]
        _current_res_key   = key
    socketio.emit("resolution_changed", {"resolution": key})
    return jsonify({"ok": True, "resolution": key, "size": list(RESOLUTIONS[key])})


# ── GPU status ────────────────────────────────────────────────

@app.route("/api/system/status", methods=["GET"])
@require_auth
def system_status():
    return jsonify({
        "gpu_device":   _gpu_device,
        "gpu_label":    "CUDA (NVIDIA)" if _gpu_device == "cuda" else "CPU",
        "resolution":   _current_res_key,
        "cam_enabled":  _cam_enabled,
        "session_active": _session_running,
    })


# ── Session lifecycle ─────────────────────────────────────────

@app.route("/api/session/start", methods=["POST"])
@require_auth
def session_start():
    global _session_running, _last_was_distracted, _snapshot_counter

    with _session_lock:
        if _session_running:
            return jsonify({"error": "Session already running"}), 409
        _session_running = True

    with _live_lock:
        _live.update({
            "timeline": [], "distraction_log": [], "alert_count": 0,
            "peak_pct": 0.0, "peak_time": "—",
            "low_pct": 100.0, "low_time": "—",
            "pct": 0.0, "avg_pct": 0.0, "attentive": 0, "total": 0,
        })

    _last_was_distracted = False
    _snapshot_counter    = 0

    data      = request.get_json(silent=True) or {}
    class_id  = data.get("class_id") or _class_id_ref
    teacher_id = g.current_user.id

    with app.app_context():
        s = DbSession(teacher_id=teacher_id, class_id=class_id, is_active=True)
        db.session.add(s)
        db.session.commit()
        sid = s.id

    with _live_lock:
        _live["session_db_id"] = sid

    socketio.emit("session_started", {"session_id": sid})
    return jsonify({"ok": True, "session_id": sid})


@app.route("/api/session/stop", methods=["POST"])
@require_auth
def session_stop():
    global _session_running

    with _session_lock:
        if not _session_running:
            return jsonify({"error": "No active session"}), 409
        _session_running = False

    with _live_lock:
        sid = _live.get("session_db_id")

    if sid:
        _finalize_session(sid)

    return jsonify({"ok": True})


def _finalize_session(sid: int):
    with _live_lock:
        tl     = list(_live.get("timeline", []))
        avg    = _live.get("avg_pct")
        peak   = _live.get("peak_pct")
        low    = _live.get("low_pct")
        alerts = _live.get("alert_count", 0)

    # Stability = 100 - stdev of timeline pcts
    vals      = [t["pct"] for t in tl]
    stability = round(max(0.0, 100.0 - (statistics.stdev(vals) if len(vals) >= 2 else 0.0)), 1)

    with app.app_context():
        s = db.session.get(DbSession, sid)
        if not s:
            return
        s.ended_at           = datetime.utcnow()
        s.is_active          = False
        s.avg_attention      = avg
        s.peak_attention     = peak
        s.low_attention      = low
        s.stability_score    = stability
        s.distraction_events = alerts
        if s.started_at:
            s.duration_seconds = int((s.ended_at - s.started_at).total_seconds())
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()

    summary = {
        "session_id":         sid,
        "average_attention":  avg or 0,
        "max_attention":      peak or 0,
        "min_attention":      low or 100,
        "stability_score":    stability,
        "distraction_events": alerts,
    }
    socketio.emit("session_summary", summary)


@app.route("/api/sessions/active", methods=["GET"])
@require_auth
def active_session():
    with _session_lock:
        running = _session_running
    with _live_lock:
        sid = _live.get("session_db_id")
    if not running:
        return jsonify({"active": False})
    s = db.session.get(DbSession, sid) if sid else None
    return jsonify({"active": True, "session": s.to_dict() if s else None})


# ── Sessions history ──────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
@require_auth
def list_sessions():
    user = g.current_user
    q    = DbSession.query
    if user.role == "teacher":
        q = q.filter_by(teacher_id=user.id)
    elif user.role == "school_admin":
        class_ids = [c.id for c in Class.query.filter_by(school_id=user.school_id).all()]
        q = q.filter(DbSession.class_id.in_(class_ids))

    page     = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    pag      = q.order_by(DbSession.started_at.desc()).paginate(
                    page=page, per_page=per_page, error_out=False)
    return jsonify({
        "sessions": [s.to_dict() for s in pag.items],
        "total":    pag.total,
        "pages":    pag.pages,
        "page":     page,
    })


@app.route("/api/sessions/<int:sid>", methods=["GET"])
@require_auth
def get_session(sid):
    s = db.session.get(DbSession, sid)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"session": s.to_dict(include_snapshots=True)})


# ── Admin APIs ────────────────────────────────────────────────

@app.route("/api/schools", methods=["GET"])
@require_auth
def list_schools():
    user = g.current_user
    q    = School.query.filter_by(institution_id=user.institution_id, is_active=True)
    if user.role == "school_admin":
        q = q.filter_by(id=user.school_id)
    return jsonify({"schools": [s.to_dict() for s in q.all()]})


@app.route("/api/schools", methods=["POST"])
@require_auth
@require_role("institution_admin")
def create_school():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    school = School(name=name, institution_id=g.current_user.institution_id)
    db.session.add(school)
    db.session.commit()
    return jsonify({"school": school.to_dict()}), 201


@app.route("/api/classes", methods=["GET"])
@require_auth
def list_classes():
    user = g.current_user
    if user.role == "institution_admin":
        school_ids = [s.id for s in School.query.filter_by(
            institution_id=user.institution_id, is_active=True).all()]
    else:
        school_ids = [user.school_id] if user.school_id else []
    classes = Class.query.filter(
        Class.school_id.in_(school_ids), Class.is_active == True).all()
    return jsonify({"classes": [c.to_dict() for c in classes]})


@app.route("/api/classes", methods=["POST"])
@require_auth
@require_role("institution_admin", "school_admin")
def create_class():
    data      = request.get_json(silent=True) or {}
    name      = (data.get("name") or "").strip()
    subject   = (data.get("subject") or "").strip()
    school_id = data.get("school_id") or g.current_user.school_id
    if not name or not school_id:
        return jsonify({"error": "name and school_id required"}), 400
    cls = Class(name=name, subject=subject, school_id=school_id)
    db.session.add(cls)
    db.session.commit()
    return jsonify({"class": cls.to_dict()}), 201


# ── Analytics ─────────────────────────────────────────────────

@app.route("/api/analytics/overview", methods=["GET"])
@require_auth
def analytics_overview():
    from sqlalchemy import func
    q = db.session.query(
        func.count(DbSession.id).label("total_sessions"),
        func.avg(DbSession.avg_attention).label("avg_attention"),
        func.sum(DbSession.distraction_events).label("total_distraction_events"),
        func.avg(DbSession.duration_seconds).label("avg_duration"),
    )
    if g.current_user.role == "teacher":
        q = q.filter(DbSession.teacher_id == g.current_user.id)
    row = q.one()
    return jsonify({
        "total_sessions":           row.total_sessions or 0,
        "avg_attention":            round(float(row.avg_attention or 0), 1),
        "total_distraction_events": int(row.total_distraction_events or 0),
        "avg_duration_seconds":     int(row.avg_duration or 0),
    })


@app.route("/api/alerts", methods=["GET"])
@require_auth
def list_alerts():
    limit = request.args.get("limit", 50, type=int)
    sid   = request.args.get("session_id", type=int)
    q     = Alert.query
    if sid:
        q = q.filter_by(session_id=sid)
    alerts = q.order_by(Alert.timestamp.desc()).limit(limit).all()
    return jsonify({"alerts": [a.to_dict() for a in alerts]})


# ── Shutdown ──────────────────────────────────────────────────

@app.route("/api/shutdown", methods=["POST"])
@require_auth
def shutdown():
    if _stop_event_ref is not None:
        _stop_event_ref.set()
    return jsonify({"ok": True})


# ── SocketIO ──────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    with _session_lock:
        running = _session_running
    with _live_lock:
        snap = dict(_live)
    snap["session_running"] = running
    snap["gpu_device"]      = _gpu_device
    socketio.emit("state_sync", snap, to=request.sid)


# ── Internal: push_frame / update_live ───────────────────────

def push_frame(frame):
    if frame is None or frame.size == 0 or not _cam_enabled:
        return
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return
    jpeg = buf.tobytes()
    try:
        _frame_queue.get_nowait()
    except queue.Empty:
        pass
    _frame_queue.put_nowait(jpeg)


def update_live(attentive: int, total: int, avg_pct: float,
                session_start: str, fps: float = 0.0):
    global _last_was_distracted, _snapshot_counter

    # Do NOT update if session not running
    with _session_lock:
        if not _session_running:
            return

    pct     = round(attentive / total * 100, 1) if total > 0 else 0.0
    now     = datetime.now().strftime("%H:%M:%S")
    is_dist = pct < DISTRACTION_THRESHOLD

    with _live_lock:
        _live.update({
            "pct":        pct,
            "avg_pct":    avg_pct,
            "attentive":  attentive,
            "total":      total,
            "fps":        round(fps, 1),
            "start_time": session_start,
        })
        _live["timeline"].append({"time": now, "pct": pct})
        if len(_live["timeline"]) > 60:
            _live["timeline"].pop(0)
        if pct > _live["peak_pct"]:
            _live["peak_pct"] = pct; _live["peak_time"] = now
        if total > 0 and pct < _live["low_pct"]:
            _live["low_pct"]  = pct; _live["low_time"]  = now

        if is_dist and not _last_was_distracted:
            _live["distraction_log"].append({"time": now, "pct": pct})
            _live["alert_count"] += 1
            sid = _live["session_db_id"]
            if sid:
                _write_alert(sid, pct, now)

        payload = {
            "pct":             pct,
            "avg_pct":         avg_pct,
            "attentive":       attentive,
            "total":           total,
            "fps":             round(fps, 1),
            "start_time":      session_start,
            "timeline":        _live["timeline"][-30:],
            "distraction_log": _live["distraction_log"],
            "alert_count":     _live["alert_count"],
        }

    _last_was_distracted = is_dist
    socketio.emit("attention_update", payload)

    _snapshot_counter += 1
    if _snapshot_counter % 5 == 0:
        with _live_lock:
            sid = _live["session_db_id"]
        if sid:
            _write_snapshot(sid, attentive, total, pct, fps)


def _write_snapshot(sid, attentive, total, pct, fps):
    with app.app_context():
        snap = AttentionSnapshot(
            session_id=sid, attentive_count=attentive,
            total_count=total, attention_pct=pct, fps=fps,
        )
        db.session.add(snap)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()


def _write_alert(sid, pct, time_str):
    with app.app_context():
        alert = Alert(
            session_id=sid, pct=pct, alert_type="low_attention",
            message=f"Attention dropped to {pct}% at {time_str}",
        )
        db.session.add(alert)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()


def start_web_dashboard(teacher_id: int = 1, class_id=None):
    global _teacher_id_ref, _class_id_ref
    _teacher_id_ref = teacher_id
    _class_id_ref   = class_id

    with app.app_context():
        init_db(app)

    def _run():
        socketio.run(app, host=WEB_HOST, port=WEB_PORT,
                     debug=False, use_reloader=False, log_output=False)

    threading.Thread(target=_run, daemon=True).start()
    threading.Timer(1.4, lambda: webbrowser.open(f"http://localhost:{WEB_PORT}")).start()


def run_final_dashboard(stats: dict):
    with _live_lock:
        sid = _live.get("session_db_id")
    if sid:
        _finalize_session(sid)