"""SQLAlchemy models for Clause Predictions and Clause Embeddings."""
from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

from app.db.base import Base, new_uuid


class ClausePrediction(Base):
    __tablename__ = "clause_predictions"

    id: str = sa.Column(sa.String(36), primary_key=True, default=new_uuid)
    contract_id: str = sa.Column(
        sa.String(36),
        sa.ForeignKey("contracts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    clause_type: str = sa.Column(sa.String(100), nullable=False, index=True)
    span_text: str = sa.Column(sa.Text(), nullable=False)
    char_start: int = sa.Column(sa.Integer(), nullable=False)
    char_end: int = sa.Column(sa.Integer(), nullable=False)
    confidence: float = sa.Column(sa.Float(), nullable=False, default=1.0)
    prediction_source: str = sa.Column(sa.String(50), nullable=False, default="classifier")
    model_version: str = sa.Column(sa.String(50), nullable=False, default="1.0.0")
    created_at = sa.Column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )

    embedding = relationship(
        "ClauseEmbedding",
        back_populates="clause_prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )
    risk_findings = relationship(
        "RiskFinding",
        back_populates="clause_prediction",
        cascade="all, delete-orphan",
    )


class ClauseEmbedding(Base):
    __tablename__ = "clause_embeddings"

    id: str = sa.Column(sa.String(36), primary_key=True, default=new_uuid)
    clause_id: str = sa.Column(
        sa.String(36),
        sa.ForeignKey("clause_predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    embedding = sa.Column(Vector(384), nullable=False)
    model_name: str = sa.Column(sa.String(100), nullable=False, default="all-MiniLM-L6-v2")

    clause_prediction = relationship("ClausePrediction", back_populates="embedding")
