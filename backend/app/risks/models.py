"""SQLAlchemy model for deterministic clause risk findings."""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from app.db.base import Base, new_uuid


class RiskFinding(Base):
    __tablename__ = "risk_findings"
    __table_args__ = (sa.UniqueConstraint("clause_id", "rule_id", name="uq_risk_findings_clause_rule"),)

    id: str = sa.Column(sa.String(36), primary_key=True, default=new_uuid)
    contract_id: str = sa.Column(
        sa.String(36), sa.ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clause_id: str = sa.Column(
        sa.String(36), sa.ForeignKey("clause_predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rule_id: str = sa.Column(sa.String(100), nullable=False)
    risk_level: str = sa.Column(sa.String(20), nullable=False)
    explanation: str = sa.Column(sa.Text(), nullable=False)
    span_text: str = sa.Column(sa.Text(), nullable=False)
    char_start: int = sa.Column(sa.Integer(), nullable=False)
    char_end: int = sa.Column(sa.Integer(), nullable=False)
    created_at = sa.Column(sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)

    clause_prediction = relationship("ClausePrediction", back_populates="risk_findings")
