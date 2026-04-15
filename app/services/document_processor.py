from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass(slots=True)
class ProcessedDocument:
    source_file: str
    chunks_created: int


class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def process_upload(self, original_name: str, file_bytes: bytes) -> ProcessedDocument:
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")
        if len(file_bytes) > self.settings.max_file_size_bytes:
            raise ValueError("Uploaded file exceeds the maximum allowed size.")

        suffix = Path(original_name).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError("Only PDF and TXT files are supported.")

        stored_path = self._save_raw_file(original_name, file_bytes)
        extracted_text = self._extract_text(stored_path)
        if not extracted_text.strip():
            raise ValueError("No readable text could be extracted from the uploaded file.")

        chunks = self._chunk_text(extracted_text)
        embeddings = self.embedding_service.encode(chunks)
        self.vector_store.add_embeddings(chunks=chunks, embeddings=embeddings, source_file=stored_path.name)
        return ProcessedDocument(source_file=stored_path.name, chunks_created=len(chunks))

    def _save_raw_file(self, original_name: str, file_bytes: bytes) -> Path:
        source = Path(original_name)
        safe_name = f"{source.stem}_{uuid4().hex}{source.suffix.lower()}"
        target_path = self.settings.raw_dir / safe_name
        target_path.write_bytes(file_bytes)
        return target_path

    def _extract_text(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _chunk_text(self, text: str) -> list[str]:
        normalized_text = " ".join(text.split())
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        if overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        chunks: list[str] = []
        start = 0
        step = chunk_size - overlap

        while start < len(normalized_text):
            chunk = normalized_text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        if not chunks:
            raise ValueError("No chunks could be created from the uploaded content.")

        return chunks
