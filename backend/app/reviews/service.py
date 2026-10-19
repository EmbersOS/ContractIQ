"""Review service — insert-only, no updates or deletes."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.reviews.models import ReviewEvent
from app.reviews.schemas import ReviewRequest


_VALID_TARGET_TYPES = {"contract", "clause", "finding"}


def create_review_event(
    db: Session,
    body: ReviewRequest,
    reviewer_id: str,
) -> ReviewEvent:
    """Insert a new review event. Raises ValueError for invalid target_type."""
    if body.target_type not in _VALID_TARGET_TYPES:
        raise ValueError(f"target_type must be one of {_VALID_TARGET_TYPES}")

    event = ReviewEvent(
        reviewer_id=reviewer_id,
        target_id=body.target_id,
        target_type=body.target_type,
        action=body.action,
        notes=body.notes,
        payload=body.payload,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

