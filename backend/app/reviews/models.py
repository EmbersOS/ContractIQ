"""ReviewEvent SQLAlchemy model — append-only audit log."""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from app.db.base import Base, new_uuid


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id: str = sa.Column(sa.String(36), primary_key=True, default=new_uuid)
    reviewer_id: str | None = sa.Column(
        sa.String(36),
        sa.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    target_id: str = sa.Column(sa.String(36), nullable=False, index=True)
    target_type: str = sa.Column(sa.String(50), nullable=False)
    action: str = sa.Column(sa.String(100), nullable=False)
    notes: str | None = sa.Column(sa.Text(), nullable=True)
    payload: dict | None = sa.Column(sa.JSON(), nullable=True)
    created_at = sa.Column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )

    reviewer = relationship("User", back_populates="review_events")

