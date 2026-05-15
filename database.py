# ============================================================
# database.py — ClassWatch SQLAlchemy Models
# Hierarchy: Institution → Schools → Classes → Teachers
# Roles: institution_admin | school_admin | teacher | student
# ============================================================

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# ─────────────────────────────────────────────────────────────
# Hierarchy models
# ─────────────────────────────────────────────────────────────

class Institution(db.Model):
    __tablename__ = "institutions"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(200), nullable=False, unique=True)
    slug       = db.Column(db.String(80), nullable=False, unique=True)
    logo_url   = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active  = db.Column(db.Boolean, default=True)

    schools = db.relationship("School", back_populates="institution", cascade="all, delete-orphan")
    users   = db.relationship("User", back_populates="institution")

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "slug": self.slug,
            "logo_url": self.logo_url,
            "school_count": len(self.schools),
            "created_at": self.created_at.isoformat(),
        }


class School(db.Model):
    __tablename__ = "schools"

    id             = db.Column(db.Integer, primary_key=True)
    institution_id = db.Column(db.Integer, db.ForeignKey("institutions.id"), nullable=False)
    name           = db.Column(db.String(200), nullable=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    is_active      = db.Column(db.Boolean, default=True)

    institution = db.relationship("Institution", back_populates="schools")
    classes     = db.relationship("Class", back_populates="school", cascade="all, delete-orphan")
    users       = db.relationship("User", back_populates="school")

    def to_dict(self):
        return {
            "id": self.id, "name": self.name,
            "institution_id": self.institution_id,
            "institution_name": self.institution.name if self.institution else None,
            "class_count": len(self.classes),
            "created_at": self.created_at.isoformat(),
        }


class Class(db.Model):
    __tablename__ = "classes"

    id         = db.Column(db.Integer, primary_key=True)
    school_id  = db.Column(db.Integer, db.ForeignKey("schools.id"), nullable=False)
    name       = db.Column(db.String(200), nullable=False)
    subject    = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active  = db.Column(db.Boolean, default=True)

    school   = db.relationship("School", back_populates="classes")
    sessions = db.relationship("Session", back_populates="class_", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "subject": self.subject,
            "school_id": self.school_id,
            "school_name": self.school.name if self.school else None,
            "created_at": self.created_at.isoformat(),
        }


# ─────────────────────────────────────────────────────────────
# User model (all roles in one table, role column gates access)
# ─────────────────────────────────────────────────────────────

ROLES = ("institution_admin", "school_admin", "teacher", "student")


class User(db.Model):
    __tablename__ = "users"

    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(255), nullable=False, unique=True, index=True)
    name           = db.Column(db.String(200), nullable=False)
    password_hash  = db.Column(db.String(512), nullable=False)
    role           = db.Column(db.String(30), nullable=False, default="teacher")
    institution_id = db.Column(db.Integer, db.ForeignKey("institutions.id"), nullable=True)
    school_id      = db.Column(db.Integer, db.ForeignKey("schools.id"), nullable=True)
    is_active      = db.Column(db.Boolean, default=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    last_login     = db.Column(db.DateTime, nullable=True)

    institution = db.relationship("Institution", back_populates="users")
    school      = db.relationship("School", back_populates="users")
    sessions    = db.relationship("Session", back_populates="teacher")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id, "email": self.email, "name": self.name,
            "role": self.role,
            "institution_id": self.institution_id,
            "institution_name": self.institution.name if self.institution else None,
            "school_id": self.school_id,
            "school_name": self.school.name if self.school else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat(),
        }


# ─────────────────────────────────────────────────────────────
# Session — one per camera/class recording
# ─────────────────────────────────────────────────────────────

class Session(db.Model):
    __tablename__ = "sessions"

    id             = db.Column(db.Integer, primary_key=True)
    teacher_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    class_id       = db.Column(db.Integer, db.ForeignKey("classes.id"), nullable=True)
    started_at     = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at       = db.Column(db.DateTime, nullable=True)
    is_active      = db.Column(db.Boolean, default=True)

    # Summary stats (filled on session end)
    avg_attention      = db.Column(db.Float, nullable=True)
    peak_attention     = db.Column(db.Float, nullable=True)
    low_attention      = db.Column(db.Float, nullable=True)
    stability_score    = db.Column(db.Float, nullable=True)
    distraction_events = db.Column(db.Integer, default=0)
    total_students     = db.Column(db.Integer, default=0)
    duration_seconds   = db.Column(db.Integer, nullable=True)

    teacher    = db.relationship("User", back_populates="sessions")
    class_     = db.relationship("Class", back_populates="sessions")
    snapshots  = db.relationship("AttentionSnapshot", back_populates="session",
                                  cascade="all, delete-orphan")
    alerts     = db.relationship("Alert", back_populates="session",
                                  cascade="all, delete-orphan")

    def to_dict(self, include_snapshots=False):
        d = {
            "id": self.id,
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher.name if self.teacher else None,
            "class_id": self.class_id,
            "class_name": self.class_.name if self.class_ else None,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "is_active": self.is_active,
            "avg_attention": self.avg_attention,
            "peak_attention": self.peak_attention,
            "low_attention": self.low_attention,
            "stability_score": self.stability_score,
            "distraction_events": self.distraction_events,
            "total_students": self.total_students,
            "duration_seconds": self.duration_seconds,
        }
        if include_snapshots:
            d["snapshots"] = [s.to_dict() for s in self.snapshots]
        return d


# ─────────────────────────────────────────────────────────────
# AttentionSnapshot — time-series data points (1 per frame-tick)
# ─────────────────────────────────────────────────────────────

class AttentionSnapshot(db.Model):
    __tablename__ = "attention_snapshots"

    id              = db.Column(db.Integer, primary_key=True)
    session_id      = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow)
    attentive_count = db.Column(db.Integer, default=0)
    total_count     = db.Column(db.Integer, default=0)
    attention_pct   = db.Column(db.Float, default=0.0)
    fps             = db.Column(db.Float, default=0.0)

    session = db.relationship("Session", back_populates="snapshots")

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "attentive": self.attentive_count,
            "total": self.total_count,
            "pct": self.attention_pct,
        }


# ─────────────────────────────────────────────────────────────
# Alert — threshold breach events
# ─────────────────────────────────────────────────────────────

class Alert(db.Model):
    __tablename__ = "alerts"

    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(50), default="low_attention")
    pct        = db.Column(db.Float, nullable=False)
    message    = db.Column(db.String(300), nullable=True)

    session = db.relationship("Session", back_populates="alerts")

    def to_dict(self):
        return {
            "id": self.id, "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type, "pct": self.pct,
            "message": self.message,
        }


# ─────────────────────────────────────────────────────────────
# DB init helper
# ─────────────────────────────────────────────────────────────

def init_db(app):
    """Call once at app startup to create tables and seed a superadmin."""
    with app.app_context():
        db.create_all()
        _seed_superadmin()


def _seed_superadmin():
    """Create a default institution_admin if none exists."""
    import os
    if User.query.filter_by(role="institution_admin").first():
        return
    inst = Institution(name="ClassWatch Demo", slug="classwatch-demo")
    db.session.add(inst)
    db.session.flush()

    admin = User(
        email=os.getenv("ADMIN_EMAIL", "admin@classwatch.io"),
        name="Super Admin",
        role="institution_admin",
        institution_id=inst.id,
        is_active=True,
    )
    admin.set_password(os.getenv("ADMIN_PASSWORD", "ClassWatch@2025"))
    db.session.add(admin)
    db.session.commit()
