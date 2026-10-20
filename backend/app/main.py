"""FastAPI application factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.auth.router import router as auth_router
from app.clauses.router import router as clauses_router
from app.contracts.router import router as contracts_router
from app.reviews.router import router as reviews_router
from app.risks.router import router as risks_router
from app.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="ContractIQ API",
        description="Contract clause review platform — Stage 1",
        version="1.0.0",
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(auth_router)
    application.include_router(contracts_router)
    application.include_router(clauses_router)
    application.include_router(reviews_router)
    application.include_router(risks_router)

    @application.get("/health", tags=["health"])
    def health_check():
        return {"status": "ok"}

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    if frontend_dir.is_dir():
        application.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        @application.get("/", include_in_schema=False)
        def frontend_root():
            return RedirectResponse(url="/app/")

    return application


app = create_app()

