# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)

class Source(BaseModel):
    title: str
    score: int
    answer_score: int
    tags: str
    preview: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    retrieved_chunks: int
    reranker: str = "none"
    vector_db: str
    llm_used: str
    embedding_model: str
    response_time_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    vectorstore_loaded: bool
    llm_ready: bool
    embedding_ready: bool
    aws_region: str
    llm_model: str
    embedding_model: str
    pinecone_vectors: int = 0
