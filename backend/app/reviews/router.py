"""Reviews router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.db.session import get_db
from app.reviews import service as review_service
from app.reviews.models import ReviewEvent
from app.reviews.schemas import ReviewEventRead, ReviewRequest

router = APIRouter(tags=["reviews"])

_VALID_TARGET_TYPES = {"contract", "clause", "finding"}


@router.post("/reviews", response_model=ReviewEventRead, status_code=status.HTTP_201_CREATED)
def create_review(
    body: ReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a review event."""
    try:
        if body.target_type not in _VALID_TARGET_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"target_type must be one of {_VALID_TARGET_TYPES}",
            )
        event = review_service.create_review_event(db, body, current_user.id)
        return event
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/reviews", response_model=list[ReviewEventRead])
def list_reviews(
    action: str | None = None,
    target_type: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List review events, optionally filtered by action or target_type."""
    try:
        query = db.query(ReviewEvent)
        if action is not None:
            query = query.filter(ReviewEvent.action == action)
        if target_type is not None:
            query = query.filter(ReviewEvent.target_type == target_type)
        events = query.order_by(ReviewEvent.created_at.asc()).offset(skip).limit(limit).all()
        return events
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/contracts/{contract_id}/audit", response_model=list[ReviewEventRead])
def get_contract_audit(
    contract_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all review events for a specific contract in chronological order."""
    try:
        from app.contracts.models import Contract

        contract = db.query(Contract).filter(Contract.id == contract_id).first()
        if contract is not None and contract.owner_id != current_user.id and current_user.role not in ("reviewer", "admin"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        events = (
            db.query(ReviewEvent)
            .filter(ReviewEvent.target_id == contract_id, ReviewEvent.target_type == "contract")
            .order_by(ReviewEvent.created_at.asc())
            .all()
        )
        return events
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

