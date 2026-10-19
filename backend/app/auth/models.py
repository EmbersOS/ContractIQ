"""User SQLAlchemy model."""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from app.db.base import Base, new_uuid


class User(Base):
    __tablename__ = "users"

    id: str = sa.Column(sa.String(36), primary_key=True, default=new_uuid)
    email: str = sa.Column(sa.String(255), nullable=False, unique=True, index=True)
    password_hash: str = sa.Column(sa.String(255), nullable=False)
    role: str = sa.Column(sa.String(50), nullable=False, default="viewer")
    created_at = sa.Column(sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)

    contracts = relationship("Contract", back_populates="owner", cascade="all, delete-orphan")
    review_events = relationship("ReviewEvent", back_populates="reviewer")

