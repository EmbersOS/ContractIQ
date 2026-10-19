"""Contract router: upload, list, get, delete."""
from __future__ import annotations

import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.config import settings
from app.contracts.models import Contract
from app.contracts.schemas import ContractRead
from app.contracts.service import process_contract
from app.db.base import new_uuid
from app.db.session import get_db
from app.storage.base import StorageBackend
from app.dependencies import get_storage

router = APIRouter(prefix="/contracts", tags=["contracts"])

# Allowed MIME types
_ALLOWED_MIMES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
# Magic bytes: PDF → b'%PDF', DOCX → PK zip header
_MAGIC: dict[bytes, str] = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _detect_mime(header: bytes) -> str | None:
    """Detect MIME type from first 4 bytes."""
    for magic, mime in _MAGIC.items():
        if header[: len(magic)] == magic:
            return mime
    return None


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=ContractRead)
async def upload_contract(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    storage: StorageBackend = Depends(get_storage),
):
    """Upload a contract (PDF or DOCX ≤ 20 MB). Returns 202 immediately."""
    try:
        max_bytes: int = settings.MAX_UPLOAD_BYTES

        # Fast reject if Content-Length header or UploadFile size reports oversized
        if file.size is not None and file.size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_bytes} bytes",
            )

        # Read up to max_bytes + 1 to prevent memory exhaustion DoS from huge uploads
        raw = await file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_bytes} bytes",
            )

        # Magic-byte MIME verification
        detected = _detect_mime(raw[:4])
        if detected is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Only PDF and DOCX are accepted.",
            )

        # Validate declared content_type against detected
        declared = (file.content_type or "").split(";")[0].strip()
        if declared and declared not in _ALLOWED_MIMES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Content-Type '{declared}' is not allowed.",
            )

        mime_type = detected

        # Persist to storage (using normalized forward slashes for storage keys)
        contract_id = new_uuid()
        ext = ".pdf" if mime_type == "application/pdf" else ".docx"
        storage_path = f"{current_user.id}/{contract_id}{ext}"
        storage.write_file(storage_path, raw)

        # Sanitize filename and create DB record
        raw_name = os.path.basename(file.filename) if file.filename else f"upload{ext}"
        filename = raw_name if raw_name else f"upload{ext}"
        contract = Contract(
            id=contract_id,
            owner_id=current_user.id,
            title=os.path.splitext(filename)[0],
            original_filename=filename,
            storage_path=storage_path,
            mime_type=mime_type,
            file_size_bytes=len(raw),
            status="PENDING",
        )
        db.add(contract)
        db.commit()
        db.refresh(contract)

        # Enqueue background processing
        background_tasks.add_task(process_contract, contract_id, storage, db)

        return contract

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("", response_model=list[ContractRead])
def list_contracts(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List contracts owned by the current user."""
    try:
        contracts = (
            db.query(Contract)
            .filter(Contract.owner_id == current_user.id)
            .order_by(Contract.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return contracts
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/{contract_id}", response_model=ContractRead)
def get_contract(
    contract_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single contract by ID."""
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).first()
        if contract is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")
        if contract.owner_id != current_user.id and current_user.role not in ("reviewer", "admin"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        return contract
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.delete("/{contract_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_contract(
    contract_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    storage: StorageBackend = Depends(get_storage),
):
    """Delete a contract. Only the owner or an admin may delete."""
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).first()
        if contract is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")
        if contract.owner_id != current_user.id and current_user.role != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        # Remove from storage
        try:
            storage.delete_file(contract.storage_path)
        except Exception:
            pass  # Don't fail if file already gone

        db.delete(contract)
        db.commit()
        return None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

