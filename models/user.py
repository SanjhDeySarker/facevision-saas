from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone  # Fixed: Added timezone for utcnow deprecation

db = SQLAlchemy()

class User(db.Model):
    """
    User model for authentication (SQLite table: 'users').
    Stores hashed passwords; supports JWT identity.
    """
    __tablename__ = 'users'  # Explicit table name (plural convention)

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  # Fixed: Timezone-aware UTC
    is_active = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        """Hash and set password (secure storage)."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify plain password against hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """Serialize user to JSON-friendly dict (excludes password)."""
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active
        }