from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    message: str
    chunks_created: int


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user question to answer from uploaded documents.",
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
