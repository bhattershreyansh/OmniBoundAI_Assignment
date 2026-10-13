from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class DocumentIn(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    source: Optional[str] = None
    brand: str = Field(..., description="The brand or list of brands discussed")

    @model_validator(mode="after")
    def validate_content_source(self) -> "DocumentIn":
        has_url = bool(self.url and self.url.strip())
        has_text = bool(self.text and self.text.strip())
        if not (has_url ^ has_text):
            raise ValueError("Document must provide either 'url' or 'text', but not both.")
        if has_url and not self.source:
            self.source = self.url.strip()
        return self


class IngestRequest(BaseModel):
    documents: List[DocumentIn] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    documents_stored: int
    chunks_stored: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question in plain English")


class Citation(BaseModel):
    source: str
    brand: str
    chunk_text: str


class TraceEntry(BaseModel):
    node: str
    detail: str


class AskResponse(BaseModel):
    answer: str
    found_in_corpus: bool
    citations: List[Citation]
    trace: List[TraceEntry]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    tenant_id: str
    doc_id: str
    chunk_index: int
    source: str
    brand: str


class HealthResponse(BaseModel):
    status: str
    vector_db: str
    llm: str
