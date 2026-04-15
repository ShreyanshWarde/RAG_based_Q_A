from dataclasses import dataclass
from time import perf_counter

from app.services.answer_generator import AnswerGenerator
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass(slots=True)
class QueryResult:
    answer: str
    retrieved_chunks: list[str]
    embedding_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float


class RetrievalService:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()

    def answer_question(self, question: str) -> QueryResult:
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_service.encode_query(question)
        embedding_time_ms = (perf_counter() - embedding_started_at) * 1000

        retrieval_started_at = perf_counter()
        results = self.vector_store.search(query_embedding)
        retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000

        chunks = [result["text"] for result in results]
        answer, generation_time_ms = self.answer_generator.generate(question, chunks)

        return QueryResult(
            answer=answer,
            retrieved_chunks=chunks,
            embedding_time_ms=embedding_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
        )
