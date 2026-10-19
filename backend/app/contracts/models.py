"""Contract SQLAlchemy model."""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from app.db.base import Base, new_uuid


class Contract(Base):
    __tablename__ = "contracts"

    id: str = sa.Column(sa.String(36), primary_key=True, default=new_uuid)
    owner_id: str = sa.Column(sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: str = sa.Column(sa.String(500), nullable=False)
    original_filename: str = sa.Column(sa.String(500), nullable=False)
    storage_path: str = sa.Column(sa.String(1000), nullable=False)
    mime_type: str = sa.Column(sa.String(100), nullable=False)
    file_size_bytes: int = sa.Column(sa.BigInteger(), nullable=False)
    extracted_text: str | None = sa.Column(sa.Text(), nullable=True)
    clause_candidates = sa.Column(sa.JSON(), nullable=True)
    status: str = sa.Column(sa.String(50), nullable=False, default="PENDING")
    error_message: str | None = sa.Column(sa.Text(), nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    updated_at = sa.Column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
    )

    owner = relationship("User", back_populates="contracts")

