# RAG QA System Explanation

## 1. Chunking Strategy

- **Chunk size = 500 characters** because it is large enough to preserve nearby meaning while still being small enough for precise retrieval. Smaller chunks improve search precision, but if they become too short they lose supporting context. Larger chunks keep more context, but retrieval becomes noisy because each match contains more unrelated text.
- **Overlap = 50 characters** because neighboring chunks otherwise cut sentences or ideas in half. The overlap preserves continuity across chunk boundaries so important phrases are less likely to disappear from retrieval.

This gives a practical balance between **context retention** and **retrieval precision**, which is exactly what the PRD asks for.

## 2. Retrieval Failure Case

Example failure case:

> Query: "What is the conclusion?"

This query is vague. If the source document never uses the exact word *conclusion*, retrieval may return chunks that are semantically similar but still not the real summary section. In this system, low-similarity results are filtered by a threshold, and the API returns:

```json
{
  "answer": "No relevant information found in documents"
}
```

That is safer than hallucinating an answer.

## 3. Metric Tracking

The system records these latency metrics in application logs during `/query`:

- Embedding Time
- Retrieval Time
- Generation Time
- Total Query Time

Example log:

```text
Query handled in 243.91 ms | embedding=119.44 ms | retrieval=13.20 ms | generation=111.27 ms
```

These timings make it easy to evaluate system performance without changing the response schema.

## 4. Anti-Hallucination Design

- The retriever returns only the top 3 chunks above a similarity threshold.
- If no chunk passes the threshold, the system returns **"No relevant information found in documents"**.
- The LLM prompt explicitly forbids external knowledge.
- If no API key is configured or the provider call fails, the application falls back to an extractive answer made only from the retrieved chunks.

## 5. Background Processing Choice

The upload route uses `asyncio.to_thread(...)` to move CPU-heavy extraction, embedding, and FAISS writing off the FastAPI event loop. This keeps the API responsive while still returning a synchronous success response with the created chunk count.
