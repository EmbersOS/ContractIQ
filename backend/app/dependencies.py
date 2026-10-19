"""Global FastAPI dependencies (re-exports for convenience)."""
from __future__ import annotations

from app.auth.dependencies import get_current_user, require_role
from app.db.session import get_db
from app.storage.local import LocalStorageBackend
from app.config import settings

_storage_instance: LocalStorageBackend | None = None


def get_storage() -> LocalStorageBackend:
    """Return a singleton LocalStorageBackend."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = LocalStorageBackend(base_dir=settings.UPLOAD_DIR)
    return _storage_instance


__all__ = ["get_db", "get_current_user", "require_role", "get_storage"]

