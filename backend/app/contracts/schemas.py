"""Contract Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PageOffset(BaseModel):
    page_num: int
    char_start: int
    char_end: int


class ContractRead(BaseModel):
    id: str
    owner_id: str
    title: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    extracted_text: str | None = None
    clause_candidates: list[Any] | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContractCreate(BaseModel):
    title: str


class ContractStatus(BaseModel):
    id: str
    status: str
    error_message: str | None = None

