"""Risk finding read endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.contracts.models import Contract
from app.db.session import get_db
from app.risks.models import RiskFinding
from app.risks.schemas import RiskFindingRead

router = APIRouter(prefix="/contracts", tags=["risks"])


@router.get("/{contract_id}/risks", response_model=list[RiskFindingRead])
def list_risk_findings(
    contract_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List deterministic risk findings for an accessible contract."""
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if contract is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")
    if contract.owner_id != current_user.id and current_user.role not in ("reviewer", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return (
        db.query(RiskFinding)
        .filter(RiskFinding.contract_id == contract_id)
        .order_by(RiskFinding.created_at.asc())
        .all()
    )
