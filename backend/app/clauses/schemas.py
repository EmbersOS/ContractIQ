"""Clause prediction API schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ClausePredictionRead(BaseModel):
    id: str
    contract_id: str
    clause_type: str
    span_text: str
    char_start: int
    char_end: int
    confidence: float
    prediction_source: str
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}
