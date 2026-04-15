import logging
import re
from time import perf_counter

from openai import OpenAI

from app.config import get_settings


logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        if self.settings.openai_api_key:
            self.client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )

    def generate(self, question: str, chunks: list[str]) -> tuple[str, float]:
        started_at = perf_counter()

        if not chunks:
            return "No relevant information found in documents", (perf_counter() - started_at) * 1000

        if self.client is None:
            answer = self._extractive_fallback(question, chunks)
            return answer, (perf_counter() - started_at) * 1000

        try:
            answer = self._llm_answer(question, chunks)
        except Exception as exc:  # pragma: no cover - network/provider failures are environment-specific.
            logger.warning("LLM generation failed, falling back to extractive mode: %s", exc)
            answer = self._extractive_fallback(question, chunks)

        return answer, (perf_counter() - started_at) * 1000

    def _llm_answer(self, question: str, chunks: list[str]) -> str:
        context = "\n\n".join(f"Chunk {index + 1}:\n{chunk}" for index, chunk in enumerate(chunks))
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided context. "
                        "If the answer is not explicitly supported by the context, "
                        "reply exactly with: No relevant information found in documents"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Context:\n{context}\n\n"
                        "Return a concise answer grounded only in the context."
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        final_answer = content.strip()
        return final_answer or "No relevant information found in documents"

    def _extractive_fallback(self, question: str, chunks: list[str]) -> str:
        keywords = {token for token in re.findall(r"\w+", question.lower()) if len(token) > 2}
        scored_sentences: list[tuple[int, str]] = []

        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk):
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                sentence_tokens = set(re.findall(r"\w+", cleaned.lower()))
                score = len(keywords & sentence_tokens)
                if score > 0:
                    scored_sentences.append((score, cleaned))

        if scored_sentences:
            top_sentences = [text for _, text in sorted(scored_sentences, reverse=True)[:3]]
            return " ".join(top_sentences)

        return chunks[0][:500].strip() or "No relevant information found in documents"
