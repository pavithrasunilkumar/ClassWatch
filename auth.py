# ============================================================
# auth.py — ClassWatch JWT Authentication
# ============================================================

import jwt
import functools
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, g
from database import db, User, Institution
from config import JWT_SECRET_KEY, JWT_ACCESS_EXPIRES_H, JWT_REFRESH_EXPIRES_D

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ── Token helpers ─────────────────────────────────────────────

def _make_access_token(user: User) -> str:
    payload = {
        "sub":   str(user.id),   # PyJWT v2 requires sub as string
        "email": user.email,
        "role":  user.role,
        "inst":  user.institution_id,
        "sch":   user.school_id,
        "exp":   datetime.utcnow() + timedelta(hours=JWT_ACCESS_EXPIRES_H),
        "type":  "access",
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def _make_refresh_token(user: User) -> str:
    payload = {
        "sub":  str(user.id),   # PyJWT v2 requires sub as string
        "exp":  datetime.utcnow() + timedelta(days=JWT_REFRESH_EXPIRES_D),
        "type": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])


# ── Auth middleware ───────────────────────────────────────────

def require_auth(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # Support ?token= query param for file download links (PDF/CSV)
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            qtoken = request.args.get("token", "")
            if qtoken:
                header = "Bearer " + qtoken
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing authorization token"}), 401
        token = header[7:].strip()
        try:
            payload = decode_token(token)
            if payload.get("type") != "access":
                raise ValueError("Wrong token type")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401

        user = db.session.get(User, int(payload["sub"]))
        if not user or not user.is_active:
            return jsonify({"error": "User not found or inactive"}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return wrapper


def require_role(*roles):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not hasattr(g, "current_user"):
                return jsonify({"error": "Not authenticated"}), 401
            if g.current_user.role not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ── Routes ───────────────────────────────────────────────────

@auth_bp.route("/login", methods=["POST"])
def login():
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = User.query.filter_by(email=email, is_active=True).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    user.last_login = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "access_token":  _make_access_token(user),
        "refresh_token": _make_refresh_token(user),
        "user":          user.to_dict(),
    })


@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    data  = request.get_json(silent=True) or {}
    token = data.get("refresh_token") or ""
    try:
        payload = decode_token(token)
        if payload.get("type") != "refresh":
            raise ValueError()
    except Exception:
        return jsonify({"error": "Invalid or expired refresh token"}), 401

    user = db.session.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        return jsonify({"error": "User not found"}), 401

    return jsonify({"access_token": _make_access_token(user)})


@auth_bp.route("/me", methods=["GET"])
@require_auth
def me():
    return jsonify({"user": g.current_user.to_dict()})


@auth_bp.route("/logout", methods=["POST"])
@require_auth
def logout():
    return jsonify({"ok": True})


# ── Public self-registration ──────────────────────────────────
# Creates account with role=teacher, is_active=False.
# Admin must activate. No auth required.

@auth_bp.route("/signup", methods=["POST"])
def signup():
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    name     = (data.get("name") or "").strip()
    password = data.get("password") or ""
    role     = data.get("role") or "teacher"
    plan     = data.get("plan") or "starter"   # stored for reference

    if role not in ("teacher", "school_admin", "student"):
        role = "teacher"

    if not all([email, name, password]):
        return jsonify({"error": "name, email and password required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    inst = Institution.query.first()

    user = User(
        email=email, name=name, role=role,
        institution_id=inst.id if inst else None,
        is_active=True,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": f"Account created on the {plan.title()} plan. You can now sign in.",
        "user":    user.to_dict(),
    }), 201


# ── Admin user management ─────────────────────────────────────

@auth_bp.route("/register", methods=["POST"])
@require_auth
@require_role("institution_admin", "school_admin")
def register():
    data      = request.get_json(silent=True) or {}
    caller    = g.current_user
    email     = (data.get("email") or "").strip().lower()
    name      = (data.get("name") or "").strip()
    password  = data.get("password") or ""
    role      = data.get("role") or "teacher"

    if not all([email, name, password]):
        return jsonify({"error": "email, name, password required"}), 400

    ROLE_RANK = {"institution_admin": 4, "school_admin": 3, "teacher": 2, "student": 1}
    if ROLE_RANK.get(role, 0) >= ROLE_RANK.get(caller.role, 0):
        return jsonify({"error": "Cannot create user with equal or higher role"}), 403

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(
        email=email, name=name, role=role,
        institution_id=caller.institution_id,
        school_id=data.get("school_id") or caller.school_id,
        is_active=True,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"user": user.to_dict()}), 201


@auth_bp.route("/users", methods=["GET"])
@require_auth
@require_role("institution_admin", "school_admin")
def list_users():
    caller = g.current_user
    q = User.query.filter_by(institution_id=caller.institution_id)
    if caller.role == "school_admin":
        q = q.filter_by(school_id=caller.school_id)
    users = q.order_by(User.created_at.desc()).all()
    return jsonify({"users": [u.to_dict() for u in users]})


@auth_bp.route("/users/<int:uid>", methods=["PATCH"])
@require_auth
@require_role("institution_admin", "school_admin")
def update_user(uid):
    user = db.session.get(User, uid)
    if not user:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json(silent=True) or {}
    for field in ("name", "school_id", "is_active"):
        if field in data:
            setattr(user, field, data[field])
    if "password" in data:
        user.set_password(data["password"])
    db.session.commit()
    return jsonify({"user": user.to_dict()})


@auth_bp.route("/users/<int:uid>", methods=["DELETE"])
@require_auth
@require_role("institution_admin")
def delete_user(uid):
    user = db.session.get(User, uid)
    if not user:
        return jsonify({"error": "User not found"}), 404
    user.is_active = False
    db.session.commit()
    return jsonify({"ok": True})