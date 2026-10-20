"""Add clause prediction and deterministic risk finding tables.

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "clause_predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("contract_id", sa.String(36), sa.ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("clause_type", sa.String(100), nullable=False),
        sa.Column("span_text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("prediction_source", sa.String(50), nullable=False, server_default="classifier"),
        sa.Column("model_version", sa.String(50), nullable=False, server_default="1.0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_clause_predictions_contract_id", "clause_predictions", ["contract_id"])
    op.create_index("ix_clause_predictions_clause_type", "clause_predictions", ["clause_type"])
    op.create_table(
        "clause_embeddings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("clause_id", sa.String(36), sa.ForeignKey("clause_predictions.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False, server_default="all-MiniLM-L6-v2"),
    )
    op.create_index("ix_clause_embeddings_clause_id", "clause_embeddings", ["clause_id"])
    op.create_table(
        "risk_findings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("contract_id", sa.String(36), sa.ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("clause_id", sa.String(36), sa.ForeignKey("clause_predictions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rule_id", sa.String(100), nullable=False),
        sa.Column("risk_level", sa.String(20), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("span_text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("clause_id", "rule_id", name="uq_risk_findings_clause_rule"),
    )
    op.create_index("ix_risk_findings_contract_id", "risk_findings", ["contract_id"])
    op.create_index("ix_risk_findings_clause_id", "risk_findings", ["clause_id"])


def downgrade() -> None:
    op.drop_index("ix_risk_findings_clause_id", table_name="risk_findings")
    op.drop_index("ix_risk_findings_contract_id", table_name="risk_findings")
    op.drop_table("risk_findings")
    op.drop_index("ix_clause_embeddings_clause_id", table_name="clause_embeddings")
    op.drop_table("clause_embeddings")
    op.drop_index("ix_clause_predictions_clause_type", table_name="clause_predictions")
    op.drop_index("ix_clause_predictions_contract_id", table_name="clause_predictions")
    op.drop_table("clause_predictions")
