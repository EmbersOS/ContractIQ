"""Review Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator

_VALID_TARGET_TYPES = {"contract", "clause", "finding"}


class ReviewRequest(BaseModel):
    target_id: str
    target_type: str
    action: str
    notes: str | None = None
    payload: dict[str, Any] | None = None

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        if v not in _VALID_TARGET_TYPES:
            raise ValueError(f"target_type must be one of {_VALID_TARGET_TYPES}")
        return v


class ReviewEventRead(BaseModel):
    id: str
    reviewer_id: str | None
    target_id: str
    target_type: str
    action: str
    notes: str | None = None
    payload: dict[str, Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
