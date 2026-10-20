"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth.router import router as auth_router
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
    application.include_router(reviews_router)
    application.include_router(risks_router)

    @application.get("/health", tags=["health"])
    def health_check():
        return {"status": "ok"}

    return application


app = create_app()

