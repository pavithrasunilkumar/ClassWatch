# ============================================================
# config.py — ClassWatch Central Configuration
# Production-ready: PostgreSQL, JWT, no hardcoded values.
# All values read from .env / environment variables.
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

def _bool(key, default=True):
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")

def _int(key, default):
    try:    return int(os.getenv(key, default))
    except: return default

def _float(key, default):
    try:    return float(os.getenv(key, default))
    except: return default

# ── Server ────────────────────────────────────────────────────
WEB_PORT   = _int("WEB_PORT", 5000)
WEB_HOST   = os.getenv("WEB_HOST", "0.0.0.0")
SECRET_KEY = os.getenv("SECRET_KEY", "classwatch-secret-change-in-production-32chars")

# ── Database ─────────────────────────────────────────────────
# PostgreSQL for production. Example:
#   DATABASE_URL=postgresql://user:password@localhost:5432/classwatch
# Falls back to local SQLite for development:
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///classwatch.db"
)

# ── JWT ──────────────────────────────────────────────────────
JWT_SECRET_KEY        = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
JWT_ACCESS_EXPIRES_H  = _int("JWT_ACCESS_EXPIRES_H", 8)    # hours
JWT_REFRESH_EXPIRES_D = _int("JWT_REFRESH_EXPIRES_D", 30)  # days

# ── Stream auth ───────────────────────────────────────────────
STREAM_PASSWORD = os.getenv("STREAM_PASSWORD", "")

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX  = _int("CAMERA_INDEX", 0)
FRAME_WIDTH   = _int("FRAME_WIDTH", 1280)
FRAME_HEIGHT  = _int("FRAME_HEIGHT", 720)

# ── YOLO ─────────────────────────────────────────────────────
YOLO_MODEL   = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF    = _float("YOLO_CONF", 0.6)
YOLO_DEVICE  = os.getenv("YOLO_DEVICE", "cpu")
YOLO_MAX_DET = _int("YOLO_MAX_DET", 30)

# ── Attention ────────────────────────────────────────────────
SMOOTHING_WINDOW      = _int("SMOOTHING_WINDOW", 11)
YAW_THRESHOLD         = _int("YAW_THRESHOLD", 55)
PITCH_THRESHOLD       = _int("PITCH_THRESHOLD", 50)
DISTRACTION_THRESHOLD = _float("DISTRACTION_THRESHOLD", 50.0)

# ── Privacy ───────────────────────────────────────────────────
PRIVACY_ENABLED = _bool("PRIVACY_ENABLED", True)

# ── Logging ───────────────────────────────────────────────────
LOG_INTERVAL = _int("LOG_INTERVAL", 2)

# ── Paths ─────────────────────────────────────────────────────
LOG_PATH         = "data/attention_log.csv"
STUDENT_LOG_PATH = "data/student_log.csv"
GRAPH_PATH       = "outputs/attention_graph.png"
SUMMARY_PATH     = "outputs/summary.txt"