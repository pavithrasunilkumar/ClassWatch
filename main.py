# ============================================================
# main.py — ClassWatch Entry Point
# • Camera ONLY opens when session starts from UI
# • CUDA GPU auto-detected and forced for YOLO
# • No counting when session not running
# • No terminal output
# ============================================================

import cv2
import sys
import signal
import threading
import time
import numpy as np
from datetime import datetime

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONF, YOLO_DEVICE, YOLO_MAX_DET,
    SMOOTHING_WINDOW, YAW_THRESHOLD, PITCH_THRESHOLD,
    PRIVACY_ENABLED,
)
from dashboard import (
    start_web_dashboard, update_live, push_frame,
    register_shutdown, run_final_dashboard,
    get_session_running, get_target_resolution,
    notify_gpu_status,
)
from privacy   import blur_faces
from utils     import FPSCounter, draw_person_box, draw_hud, ensure_dirs
from analytics import compute_statistics

import torch
from ultralytics import YOLO
import mediapipe as mp

# ── GPU detection ─────────────────────────────────────────────
_device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load YOLO ─────────────────────────────────────────────────
_model = YOLO(YOLO_MODEL)
_model.to(_device)

# ── MediaPipe (CPU — runs fast enough; GPU via OpenCV handled in privacy.py) ──
_mp_fd    = mp.solutions.face_detection
_mp_fm    = mp.solutions.face_mesh
_face_det = _mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.3)
_face_mesh = _mp_fm.FaceMesh(
    refine_landmarks=True, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)


def _smooth(history: list, window: int) -> str:
    recent = history[-window:]
    return max(set(recent), key=recent.count)


def get_attention_state(frame, box):
    x1, y1, x2, y2 = box
    pad = 15
    h_f, w_f = frame.shape[:2]
    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
    cx2 = min(w_f, x2 + pad); cy2 = min(h_f, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "Distracted"
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    fd  = _face_det.process(rgb)
    if not fd.detections:
        return "Distracted"
    det = fd.detections[0]
    bb  = det.location_data.relative_bounding_box
    ch, cw = crop.shape[:2]
    fx1 = max(0, int(bb.xmin * cw))
    fy1 = max(0, int(bb.ymin * ch))
    fx2 = min(cw, int((bb.xmin + bb.width) * cw))
    fy2 = min(ch, int((bb.ymin + bb.height) * ch))
    face = crop[fy1:fy2, fx1:fx2]
    if face.size == 0:
        return "Distracted"
    mr = _face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    if not mr.multi_face_landmarks:
        return "Distracted"
    lms = mr.multi_face_landmarks[0]
    fh, fw = face.shape[:2]
    p2d, p3d = [], []
    for idx, lm in enumerate(lms.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            px, py = int(lm.x * fw), int(lm.y * fh)
            p2d.append([px, py])
            p3d.append([px, py, lm.z])
    if len(p2d) < 6:
        return "Distracted"
    p2d  = np.array(p2d, dtype=np.float64)
    p3d  = np.array(p3d, dtype=np.float64)
    cam  = np.array([[fw, 0, fw/2],[0, fw, fh/2],[0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(p3d, p2d, cam, dist)
    if not ok:
        return "Distracted"
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0] * 360
    yaw   = angles[1] * 360
    if abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD:
        return "Attentive"
    return "Distracted"


# ── Shutdown ──────────────────────────────────────────────────
_stop_event = threading.Event()

def _signal_handler(sig, frame):
    _stop_event.set()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Main ──────────────────────────────────────────────────────
def main():
    ensure_dirs("data", "outputs")

    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--teacher-id", type=int, default=1)
    parser.add_argument("--class-id",   type=int, default=None)
    args, _ = parser.parse_known_args()

    register_shutdown(_stop_event)

    # Tell dashboard what GPU we have so UI can show it
    notify_gpu_status(_device)

    # Start web dashboard (does NOT open camera)
    start_web_dashboard(teacher_id=args.teacher_id, class_id=args.class_id)

    cap             = None
    current_w       = FRAME_WIDTH
    current_h       = FRAME_HEIGHT
    fps_counter     = None
    student_history: dict = {}
    _sum_pct        = 0.0
    _frame_idx      = 0
    session_start   = None
    was_running     = False

    while not _stop_event.is_set():

        session_running = get_session_running()

        # ── Session just stopped ───────────────────────────────
        if was_running and not session_running:
            was_running     = False
            session_start   = None
            student_history = {}
            _sum_pct        = 0.0
            _frame_idx      = 0
            fps_counter     = None
            # Release camera when session ends
            if cap is not None:
                cap.release()
                cap = None
            continue

        # ── Not running — idle ─────────────────────────────────
        if not session_running:
            time.sleep(0.05)
            continue

        # ── Session just started ───────────────────────────────
        if not was_running:
            was_running   = True
            session_start = datetime.now().strftime("%H:%M:%S")
            fps_counter   = FPSCounter(window=30)

            target_w, target_h = get_target_resolution()
            current_w, current_h = target_w, target_h

            # Open camera now
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                # Signal back that camera failed — stop session
                _stop_event.set()
                break
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  current_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_h)

        # ── Dynamic resolution change ──────────────────────────
        target_w, target_h = get_target_resolution()
        if (target_w, target_h) != (current_w, current_h):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            current_w, current_h = target_w, target_h
            student_history = {}
            _sum_pct        = 0.0
            _frame_idx      = 0

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame      = cv2.resize(frame, (current_w, current_h))
        _frame_idx += 1

        # ── YOLO detect + track ───────────────────────────────
        results = _model.track(
            frame, persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONF,
            max_det=YOLO_MAX_DET,
            device=_device,
            verbose=False,
        )

        attentive  = 0
        total      = 0
        blur_dets: list[dict] = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != 0 or box.id is None:
                    continue
                tid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                total += 1
                blur_dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "track_id": tid})
                raw      = get_attention_state(frame, (x1, y1, x2, y2))
                student_history.setdefault(tid, []).append(raw)
                smoothed = _smooth(student_history[tid], SMOOTHING_WINDOW)
                if smoothed == "Attentive":
                    attentive += 1
                draw_person_box(frame, x1, y1, x2, y2, tid, smoothed)

        if PRIVACY_ENABLED and blur_dets:
            blur_faces(frame, blur_dets)

        fps = fps_counter.tick()
        draw_hud(frame, attentive, total, fps, PRIVACY_ENABLED)

        # Only count attention when people are actually visible
        if total > 0:
            _sum_pct += (attentive / total * 100)
        avg_pct = round(_sum_pct / max(_frame_idx, 1), 1)

        update_live(attentive, total, avg_pct, session_start, fps)
        push_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            _stop_event.set()

    # ── Cleanup ───────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    from config import LOG_PATH
    stats = compute_statistics(LOG_PATH) or {}
    run_final_dashboard(stats)


if __name__ == "__main__":
    main()