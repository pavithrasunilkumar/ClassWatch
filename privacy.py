# ============================================================
# privacy.py — Privacy Layer
# • MediaPipe face detection for accurate face region
# • CUDA-accelerated Gaussian blur if opencv-contrib + CUDA
# • Falls back to CPU GaussianBlur gracefully
# • No face images stored
# ============================================================

import cv2
import numpy as np
import contextlib
import io

# ── CUDA blur availability check ──────────────────────────────
_CUDA_BLUR = False
try:
    _test = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (61, 61), 25)
    _CUDA_BLUR = True
except (cv2.error, AttributeError):
    _CUDA_BLUR = False

# ── Blur filter (built once, reused) ──────────────────────────
BLUR_KERNEL = (61, 61)
BLUR_SIGMA  = 25

if _CUDA_BLUR:
    _cuda_filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC3, cv2.CV_8UC3, BLUR_KERNEL, BLUR_SIGMA
    )

def _gaussian_blur(roi: np.ndarray) -> np.ndarray:
    """Blur a BGR image region — GPU if available, else CPU."""
    if roi.size == 0:
        return roi
    if _CUDA_BLUR:
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(roi)
            gpu_dst = _cuda_filter.apply(gpu_src)
            return gpu_dst.download()
        except Exception:
            pass
    return cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)


# ── MediaPipe face detection ──────────────────────────────────
_USE_MP      = False
_face_det    = None
_mp_checked  = False


def _ensure_mediapipe():
    global _USE_MP, _face_det, _mp_checked
    if _mp_checked:
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            import mediapipe as mp
        _mp_fd    = mp.solutions.face_detection
        _face_det = _mp_fd.FaceDetection(
            model_selection=1, min_detection_confidence=0.4)
        _USE_MP = True
    except Exception:
        _USE_MP   = False
        _face_det = None
    finally:
        _mp_checked = True


# ── Smoothing ─────────────────────────────────────────────────
BLUR_PAD        = 10
SMOOTHING_ALPHA = 0.8
_face_cache: dict = {}


def _cache_key(det: dict) -> str:
    tid = det.get("track_id")
    if tid is not None:
        return str(tid)
    return f"{int(det.get('x1',0))}:{int(det.get('y1',0))}:{int(det.get('x2',0))}:{int(det.get('y2',0))}"


def _smooth_box(prev, curr):
    if prev is None:
        return curr
    return tuple(prev[i] * SMOOTHING_ALPHA + curr[i] * (1.0 - SMOOTHING_ALPHA) for i in range(4))


# ── Main entry point ──────────────────────────────────────────

def blur_faces(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Blur face regions in each person bounding box.

    Strategy:
      1. Crop person region.
      2. Run MediaPipe face detection → accurate face bbox.
      3. Apply CUDA (or CPU) Gaussian blur to that region.
      4. If no face found, blur upper 30% of person bbox.
    """
    if frame is None or frame.size == 0:
        return frame

    _ensure_mediapipe()
    h_f, w_f = frame.shape[:2]

    for det in detections:
        key = _cache_key(det)
        x1  = max(0, min(int(det.get("x1", 0)), w_f - 1))
        y1  = max(0, min(int(det.get("y1", 0)), h_f - 1))
        x2  = max(0, min(int(det.get("x2", 0)), w_f - 1))
        y2  = max(0, min(int(det.get("y2", 0)), h_f - 1))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        face_found    = False
        cached_rel    = _face_cache.get(key)

        if _USE_MP:
            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = _face_det.process(rgb)

            if results.detections:
                for face in results.detections:
                    bb  = face.location_data.relative_bounding_box
                    ch, cw = crop.shape[:2]

                    fx1 = max(0, int(bb.xmin * cw) - BLUR_PAD)
                    fy1 = max(0, int(bb.ymin * ch) - BLUR_PAD)
                    fx2 = min(cw, int((bb.xmin + bb.width)  * cw) + BLUR_PAD)
                    fy2 = min(ch, int((bb.ymin + bb.height) * ch) + BLUR_PAD)

                    if fx2 - fx1 < 5 or fy2 - fy1 < 5:
                        continue

                    curr_rel = (fx1/cw, fy1/ch, fx2/cw, fy2/ch)
                    smooth   = _smooth_box(cached_rel, curr_rel)
                    _face_cache[key] = smooth

                    sfx1 = max(0, min(int(smooth[0]*cw), cw-1))
                    sfy1 = max(0, min(int(smooth[1]*ch), ch-1))
                    sfx2 = max(0, min(int(smooth[2]*cw), cw))
                    sfy2 = max(0, min(int(smooth[3]*ch), ch))

                    if sfx2 - sfx1 < 5 or sfy2 - sfy1 < 5:
                        continue

                    roi = crop[sfy1:sfy2, sfx1:sfx2].copy()
                    crop[sfy1:sfy2, sfx1:sfx2] = _gaussian_blur(roi)
                    face_found = True
                    break

        # Use cached box if no new detection
        if not face_found and cached_rel is not None:
            ch, cw = crop.shape[:2]
            fx1 = max(0, min(int(cached_rel[0]*cw), cw-1))
            fy1 = max(0, min(int(cached_rel[1]*ch), ch-1))
            fx2 = max(0, min(int(cached_rel[2]*cw), cw))
            fy2 = max(0, min(int(cached_rel[3]*ch), ch))
            if fx2-fx1 >= 5 and fy2-fy1 >= 5:
                roi = crop[fy1:fy2, fx1:fx2].copy()
                crop[fy1:fy2, fx1:fx2] = _gaussian_blur(roi)
                face_found = True

        # Hard fallback: blur top 30%
        if not face_found:
            ch, cw = crop.shape[:2]
            fy2    = int(ch * 0.30)
            if fy2 > 4:
                roi = crop[0:fy2, 0:cw].copy()
                crop[0:fy2, 0:cw] = _gaussian_blur(roi)

        frame[y1:y2, x1:x2] = crop

    return frame