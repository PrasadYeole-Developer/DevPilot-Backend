"""Health check endpoint used to verify the service is running."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Simple liveness check — no external dependencies."""
    return {"status": "ok"}
