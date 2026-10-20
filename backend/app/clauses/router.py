"""Detected clause prediction read endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.clauses.models import ClausePrediction
from app.clauses.schemas import ClausePredictionRead
from app.contracts.models import Contract
from app.db.session import get_db

router = APIRouter(prefix="/contracts", tags=["clauses"])


@router.get("/{contract_id}/clauses", response_model=list[ClausePredictionRead])
def list_clauses(
    contract_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List detected clauses for an accessible contract, ordered by source position."""
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if contract is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")
    if contract.owner_id != current_user.id and current_user.role not in ("reviewer", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return (
        db.query(ClausePrediction)
        .filter(ClausePrediction.contract_id == contract_id)
        .order_by(ClausePrediction.char_start.asc(), ClausePrediction.char_end.asc())
        .all()
    )
