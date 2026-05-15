# ============================================================
# dashboard.py — ClassWatch Web Dashboard & API
# New: /api/sessions/<id>/export/csv
#      /api/sessions/<id>/export/pdf
#      /api/sessions/export/csv  (all sessions)
# ============================================================

import os
import io
import csv
import statistics
import threading
import webbrowser
import queue
import cv2
from datetime import datetime
from flask import (Flask, request, jsonify, Response,
                   send_from_directory, g, make_response)
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

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config.update(
    SECRET_KEY                     = SECRET_KEY,
    SQLALCHEMY_DATABASE_URI        = DATABASE_URL,
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    SQLALCHEMY_ENGINE_OPTIONS      = {
        "pool_pre_ping":    True,
        "pool_recycle":     300,
        # SQLite: allow access from multiple threads (main loop + Flask threads)
        "connect_args":     {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    },
)
CORS(app, origins="*", supports_credentials=True,
     allow_headers=["Authorization", "Content-Type"],
     expose_headers=["Authorization", "Content-Disposition"])
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
app.register_blueprint(auth_bp)

# ── MJPEG ─────────────────────────────────────────────────────
_frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=1)
_last_jpeg:   bytes = b""
_frame_lock   = threading.Lock()
_cam_enabled  = True

# ── Session control ───────────────────────────────────────────
_session_running = False
_session_lock    = threading.Lock()

# ── Resolution ────────────────────────────────────────────────
RESOLUTIONS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
_target_resolution = (FRAME_WIDTH, FRAME_HEIGHT)
_current_res_key   = "720p"
_resolution_lock   = threading.Lock()

# ── GPU ───────────────────────────────────────────────────────
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


# ── Helpers for main.py ───────────────────────────────────────

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

@app.route("/app")
def app_redirect():
    from flask import redirect
    return redirect("/")


@app.route("/landing")
def landing():
    return send_from_directory(FRONTEND_DIR, "landing.html")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path):
    # Don't intercept API routes or socket.io
    if path.startswith(("api/", "video_feed", "socket.io")):
        from flask import abort
        abort(404)
    full = os.path.join(FRONTEND_DIR, path)
    # Serve any file that actually exists (landing.html, assets, etc.)
    if path and os.path.exists(full) and os.path.isfile(full):
        return send_from_directory(FRONTEND_DIR, path)
    # Everything else → SPA
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── MJPEG ─────────────────────────────────────────────────────

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
    return jsonify({"ok": True, "resolution": key})


@app.route("/api/system/status", methods=["GET"])
@require_auth
def system_status():
    return jsonify({
        "gpu_device":     _gpu_device,
        "gpu_label":      "CUDA (NVIDIA)" if _gpu_device == "cuda" else "CPU",
        "resolution":     _current_res_key,
        "cam_enabled":    _cam_enabled,
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

    data       = request.get_json(silent=True) or {}
    class_id   = data.get("class_id") or _class_id_ref
    teacher_id = g.current_user.id

    # Already inside a Flask request context — no need for app.app_context()
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
        _finalize_session(sid, inside_request=True)

    return jsonify({"ok": True})


def _finalize_session(sid: int, inside_request: bool = False):
    with _live_lock:
        tl     = list(_live.get("timeline", []))
        avg    = _live.get("avg_pct")
        peak   = _live.get("peak_pct")
        low    = _live.get("low_pct")
        alerts = _live.get("alert_count", 0)

    vals      = [t["pct"] for t in tl]
    stability = round(max(0.0, 100.0 - (statistics.stdev(vals) if len(vals) >= 2 else 0.0)), 1)

    def _do_finalize():
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

    if inside_request:
        _do_finalize()
    else:
        with app.app_context():
            _do_finalize()

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


# ── Export: CSV ───────────────────────────────────────────────

@app.route("/api/sessions/<int:sid>/export/csv", methods=["GET"])
@require_auth
def export_session_csv(sid):
    s = db.session.get(DbSession, sid)
    if not s:
        return jsonify({"error": "Not found"}), 404

    output = io.StringIO()
    writer = csv.writer(output)

    # Header block
    writer.writerow(["ClassWatch Session Report"])
    writer.writerow(["Session ID", sid])
    writer.writerow(["Teacher",    s.teacher.name if s.teacher else "—"])
    writer.writerow(["Class",      s.class_.name  if s.class_  else "—"])
    writer.writerow(["Started",    s.started_at.strftime("%Y-%m-%d %H:%M:%S") if s.started_at else "—"])
    writer.writerow(["Ended",      s.ended_at.strftime("%Y-%m-%d %H:%M:%S")   if s.ended_at  else "—"])
    writer.writerow(["Duration",   f"{s.duration_seconds}s" if s.duration_seconds else "—"])
    writer.writerow([])
    writer.writerow(["Summary"])
    writer.writerow(["Avg Attention",      f"{s.avg_attention}%"   if s.avg_attention   is not None else "—"])
    writer.writerow(["Peak Attention",     f"{s.peak_attention}%"  if s.peak_attention  is not None else "—"])
    writer.writerow(["Lowest Attention",   f"{s.low_attention}%"   if s.low_attention   is not None else "—"])
    writer.writerow(["Stability Score",    f"{s.stability_score}"  if s.stability_score is not None else "—"])
    writer.writerow(["Distraction Events", s.distraction_events])
    writer.writerow([])
    writer.writerow(["Timeline"])
    writer.writerow(["Timestamp", "Attentive Count", "Total Count", "Attention %", "FPS"])
    for snap in s.snapshots:
        writer.writerow([
            snap.timestamp.strftime("%H:%M:%S"),
            snap.attentive_count,
            snap.total_count,
            snap.attention_pct,
            snap.fps,
        ])

    output.seek(0)
    resp = make_response(output.getvalue())
    resp.headers["Content-Type"]        = "text/csv"
    resp.headers["Content-Disposition"] = f"attachment; filename=classwatch_session_{sid}.csv"
    return resp


@app.route("/api/sessions/export/csv", methods=["GET"])
@require_auth
def export_all_csv():
    """Export all sessions for this user as CSV."""
    user = g.current_user
    q    = DbSession.query.filter_by(is_active=False)
    if user.role == "teacher":
        q = q.filter_by(teacher_id=user.id)
    sessions = q.order_by(DbSession.started_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ClassWatch — All Sessions Export"])
    writer.writerow(["Exported", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")])
    writer.writerow([])
    writer.writerow(["ID", "Teacher", "Class", "Date", "Duration (s)",
                     "Avg Attention %", "Peak %", "Low %",
                     "Stability", "Distraction Events"])
    for s in sessions:
        writer.writerow([
            s.id,
            s.teacher.name if s.teacher else "—",
            s.class_.name  if s.class_  else "—",
            s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "—",
            s.duration_seconds or 0,
            s.avg_attention    or 0,
            s.peak_attention   or 0,
            s.low_attention    or 0,
            s.stability_score  or 0,
            s.distraction_events or 0,
        ])

    output.seek(0)
    resp = make_response(output.getvalue())
    resp.headers["Content-Type"]        = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=classwatch_all_sessions.csv"
    return resp


# ── Export: PDF (HTML rendered as PDF via browser print) ──────
# We generate a styled HTML report that the browser prints as PDF.
# No server-side PDF library needed.

@app.route("/api/sessions/<int:sid>/export/pdf", methods=["GET"])
@require_auth
def export_session_pdf(sid):
    s = db.session.get(DbSession, sid)
    if not s:
        return jsonify({"error": "Not found"}), 404

    snaps    = s.snapshots
    timeline = [(snap.timestamp.strftime("%H:%M:%S"), snap.attention_pct)
                for snap in snaps]
    # Build sparkline data as JS array
    tl_labels = [f'"{t[0]}"' for t in timeline]
    tl_data   = [str(round(t[1], 1)) for t in timeline]

    avg   = s.avg_attention   or 0
    peak  = s.peak_attention  or 0
    low   = s.low_attention   or 0
    stab  = s.stability_score or 0
    alts  = s.distraction_events or 0
    dur   = f"{s.duration_seconds // 60}m {s.duration_seconds % 60}s" if s.duration_seconds else "—"
    grade = "Excellent" if avg >= 80 else "Good" if avg >= 65 else "Fair" if avg >= 50 else "Needs Attention"
    grade_col = "#00e5a0" if avg >= 80 else "#3b82f6" if avg >= 65 else "#f5a623" if avg >= 50 else "#ff3b5c"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<title>ClassWatch Report — Session #{sid}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'DM Sans',sans-serif;background:#fff;color:#111;font-size:14px;padding:40px;max-width:900px;margin:0 auto}}
  .header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:36px;padding-bottom:20px;border-bottom:2px solid #f0f0f0}}
  .logo{{display:flex;align-items:center;gap:10px}}
  .logo-mark{{width:36px;height:36px;border-radius:9px;background:linear-gradient(135deg,#00e5a0,#3b82f6);display:grid;place-items:center}}
  .logo-mark svg{{width:18px;height:18px}}
  .logo-text{{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;letter-spacing:-.3px}}
  .report-title{{text-align:right}}
  .report-title h2{{font-family:'Syne',sans-serif;font-size:22px;font-weight:700;letter-spacing:-.4px}}
  .report-title p{{font-size:12px;color:#888;margin-top:4px}}
  .meta-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px}}
  .meta-item{{background:#f8f8f8;border-radius:10px;padding:14px 16px}}
  .meta-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:5px}}
  .meta-value{{font-family:'Syne',sans-serif;font-size:15px;font-weight:600}}
  .kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:28px}}
  .kpi{{border:1px solid #f0f0f0;border-radius:12px;padding:16px;text-align:center;position:relative;overflow:hidden}}
  .kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--c,#e0e0e0);border-radius:12px 12px 0 0}}
  .kpi-val{{font-family:'Syne',sans-serif;font-size:26px;font-weight:700;line-height:1;margin-bottom:5px}}
  .kpi-lbl{{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.7px}}
  .grade-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;background:{grade_col}22;color:{grade_col};border:1px solid {grade_col}44;margin-top:8px}}
  .section-title{{font-family:'Syne',sans-serif;font-size:16px;font-weight:700;letter-spacing:-.2px;margin-bottom:14px}}
  .chart-wrap{{background:#f8f8f8;border-radius:12px;padding:20px;margin-bottom:24px}}
  .alerts-table{{width:100%;border-collapse:collapse;margin-bottom:24px}}
  .alerts-table th{{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:#888;text-align:left;padding:0 12px 10px;border-bottom:1px solid #f0f0f0;font-weight:500}}
  .alerts-table td{{padding:10px 12px;border-bottom:1px solid #f8f8f8;font-size:13px}}
  .alerts-table tr:last-child td{{border-bottom:none}}
  .footer{{margin-top:36px;padding-top:16px;border-top:1px solid #f0f0f0;display:flex;justify-content:space-between;font-size:11px;color:#aaa}}
  @media print{{
    body{{padding:20px}}
    button{{display:none}}
    .no-print{{display:none}}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="logo">
    <div class="logo-mark"><svg viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></div>
    <span class="logo-text">ClassWatch</span>
  </div>
  <div class="report-title">
    <h2>Session Report #{sid}</h2>
    <p>Generated {datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC")}</p>
  </div>
</div>

<div class="meta-grid">
  <div class="meta-item"><div class="meta-label">Teacher</div><div class="meta-value">{s.teacher.name if s.teacher else "—"}</div></div>
  <div class="meta-item"><div class="meta-label">Class</div><div class="meta-value">{s.class_.name if s.class_ else "—"}</div></div>
  <div class="meta-item"><div class="meta-label">Date</div><div class="meta-value">{s.started_at.strftime("%b %d, %Y") if s.started_at else "—"}</div></div>
  <div class="meta-item"><div class="meta-label">Started</div><div class="meta-value">{s.started_at.strftime("%H:%M:%S") if s.started_at else "—"}</div></div>
  <div class="meta-item"><div class="meta-label">Ended</div><div class="meta-value">{s.ended_at.strftime("%H:%M:%S") if s.ended_at else "—"}</div></div>
  <div class="meta-item"><div class="meta-label">Duration</div><div class="meta-value">{dur}</div></div>
</div>

<div class="kpi-row">
  <div class="kpi" style="--c:{grade_col}">
    <div class="kpi-val" style="color:{grade_col}">{avg}%</div>
    <div class="kpi-lbl">Avg Attention</div>
    <div class="grade-badge">{grade}</div>
  </div>
  <div class="kpi" style="--c:#00e5a0">
    <div class="kpi-val" style="color:#00e5a0">{peak}%</div>
    <div class="kpi-lbl">Peak</div>
  </div>
  <div class="kpi" style="--c:#ff3b5c">
    <div class="kpi-val" style="color:#ff3b5c">{low}%</div>
    <div class="kpi-lbl">Lowest</div>
  </div>
  <div class="kpi" style="--c:#3b82f6">
    <div class="kpi-val" style="color:#3b82f6">{stab}</div>
    <div class="kpi-lbl">Stability / 100</div>
  </div>
  <div class="kpi" style="--c:#f5a623">
    <div class="kpi-val" style="color:#f5a623">{alts}</div>
    <div class="kpi-lbl">Alert Events</div>
  </div>
</div>

<div class="section-title">Attention Timeline</div>
<div class="chart-wrap">
  <canvas id="tlChart" height="120"></canvas>
</div>

<div class="section-title">Distraction Events</div>
{"".join([f'<table class="alerts-table"><thead><tr><th>Time</th><th>Attention %</th><th>Drop from 50%</th></tr></thead><tbody>' +
"".join([f'<tr><td>{a.timestamp.strftime("%H:%M:%S")}</td><td style="color:#ff3b5c;font-weight:600">{a.pct}%</td><td>{round(50-a.pct,1)}% below threshold</td></tr>' for a in s.alerts]) +
'</tbody></table>' if s.alerts else '<p style="color:#aaa;font-size:13px;padding:12px 0">No distraction events recorded — class maintained focus throughout.</p>'])}

<div class="footer">
  <span>ClassWatch v2.0 — AI-Powered Classroom Attention Analysis</span>
  <span>Session #{sid} · {s.started_at.strftime("%Y-%m-%d") if s.started_at else ""}</span>
</div>

<script>
const ctx = document.getElementById('tlChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: [{",".join(tl_labels)}],
    datasets: [{{
      data: [{",".join(tl_data)}],
      borderColor: '#00e5a0',
      backgroundColor: 'rgba(0,229,160,0.08)',
      borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true
    }}]
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 10, font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }},
      y: {{ min: 0, max: 100, ticks: {{ callback: v => v+'%', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}
    }}
  }}
}});
window.onload = () => setTimeout(() => window.print(), 800);
</script>
</body>
</html>"""

    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


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


# ── Internal push/update ──────────────────────────────────────

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

    with _session_lock:
        if not _session_running:
            return

    pct     = round(attentive / total * 100, 1) if total > 0 else 0.0
    now     = datetime.now().strftime("%H:%M:%S")
    is_dist = pct < DISTRACTION_THRESHOLD

    with _live_lock:
        _live.update({
            "pct": pct, "avg_pct": avg_pct,
            "attentive": attentive, "total": total,
            "fps": round(fps, 1), "start_time": session_start,
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
            "pct": pct, "avg_pct": avg_pct,
            "attentive": attentive, "total": total,
            "fps": round(fps, 1), "start_time": session_start,
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
    """Write attention snapshot — called from main loop thread."""
    try:
        with app.app_context():
            snap = AttentionSnapshot(
                session_id=sid, attentive_count=attentive,
                total_count=total, attention_pct=pct, fps=fps,
            )
            db.session.add(snap)
            db.session.commit()
            db.session.expunge_all()
    except Exception:
        try:
            with app.app_context():
                db.session.rollback()
        except Exception:
            pass


def _write_alert(sid, pct, time_str):
    """Write alert — called from main loop thread."""
    try:
        with app.app_context():
            alert = Alert(
                session_id=sid, pct=pct, alert_type="low_attention",
                message=f"Attention dropped to {pct}% at {time_str}",
            )
            db.session.add(alert)
            db.session.commit()
            db.session.expunge_all()
    except Exception:
        try:
            with app.app_context():
                db.session.rollback()
        except Exception:
            pass


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
    threading.Timer(1.4, lambda: webbrowser.open(f"http://localhost:{WEB_PORT}/landing")).start()


def run_final_dashboard(stats: dict):
    with _live_lock:
        sid = _live.get("session_db_id")
    if sid:
        _finalize_session(sid)