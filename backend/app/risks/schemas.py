"""Risk finding API schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class RiskFindingRead(BaseModel):
    id: str
    contract_id: str
    clause_id: str
    rule_id: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    explanation: str
    span_text: str
    char_start: int
    char_end: int
    created_at: datetime

    model_config = {"from_attributes": True}
