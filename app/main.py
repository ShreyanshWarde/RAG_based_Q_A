import logging

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="A grounded RAG API that answers questions strictly from uploaded documents.",
)

app.include_router(router)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
