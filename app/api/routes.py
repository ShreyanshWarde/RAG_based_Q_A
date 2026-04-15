import asyncio
import logging
from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, UploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.retrieval_service import RetrievalService
from app.utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_max_requests,
    window_seconds=settings.rate_limit_window_seconds,
)

# Lazy initialization - services are loaded only when first used
_document_processor = None
_retrieval_service = None


def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor (lazy initialization)."""
    global _document_processor
    if _document_processor is None:
        logger.info("Initializing DocumentProcessor...")
        _document_processor = DocumentProcessor()
        logger.info("DocumentProcessor initialized successfully")
    return _document_processor


def get_retrieval_service() -> RetrievalService:
    """Get or create the retrieval service (lazy initialization)."""
    global _retrieval_service
    if _retrieval_service is None:
        logger.info("Initializing RetrievalService...")
        _retrieval_service = RetrievalService()
        logger.info("RetrievalService initialized successfully")
    return _retrieval_service


def rate_limit_dependency(request: Request) -> None:
    client_host = request.client.host if request.client else "anonymous"
    try:
        rate_limiter.check(client_host)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc


@router.post(
    "/upload",
    response_model=UploadResponse,
    tags=["documents"],
    dependencies=[Depends(rate_limit_dependency)],
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    started_at = perf_counter()
    document_processor = get_document_processor()

    try:
        result = await asyncio.to_thread(
            document_processor.process_upload,
            file.filename or "uploaded_document",
            content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "Processed upload '%s' in %.2f ms (%s chunks)",
        result.source_file,
        (perf_counter() - started_at) * 1000,
        result.chunks_created,
    )
    return UploadResponse(
        message="Document processed successfully",
        chunks_created=result.chunks_created,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    dependencies=[Depends(rate_limit_dependency)],
)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    started_at = perf_counter()
    retrieval_service = get_retrieval_service()
    result = await asyncio.to_thread(retrieval_service.answer_question, payload.question)
    elapsed_ms = (perf_counter() - started_at) * 1000

    logger.info(
        "Query handled in %.2f ms | embedding=%.2f ms | retrieval=%.2f ms | generation=%.2f ms",
        elapsed_ms,
        result.embedding_time_ms,
        result.retrieval_time_ms,
        result.generation_time_ms,
    )
    return QueryResponse(answer=result.answer, retrieved_chunks=result.retrieved_chunks)
