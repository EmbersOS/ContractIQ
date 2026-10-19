"""SQLAlchemy declarative base and UUID helper."""
from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase


def new_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Shared declarative base for all models."""
    pass

