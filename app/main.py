# app/main.py
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.models import QuestionRequest, QuestionResponse, HealthResponse
from app.rag_pipeline import PythonQARAG
from app.config import get_settings
from app.utils.pinecone_store import pinecone_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_pipeline = None
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    try:
        logger.info("🚀 Initializing RAG pipeline with Pinecone...")
        # Verify Pinecone connection
        stats = pinecone_manager.get_stats()
        logger.info(f"☁️  Pinecone connected: {stats['total_vectors']:,} vectors")
        
        rag_pipeline = PythonQARAG()
        logger.info("✅ RAG pipeline ready!")
    except Exception as e:
        logger.error(f"⚠️ Failed: {e}")
    yield
    logger.info("🛑 Shutting down...")

app = FastAPI(
    title="Python Q&A Assistant",
    description="RAG with Pinecone (Cloud) + AWS Bedrock (Qwen + Titan)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    stats = pinecone_manager.get_stats() if pinecone_manager.index else {}
    return HealthResponse(
        status="healthy" if rag_pipeline else "degraded",
        vectorstore_loaded=rag_pipeline is not None,
        llm_ready=rag_pipeline is not None,
        embedding_ready=rag_pipeline is not None,
        aws_region=settings.AWS_REGION,
        llm_model=settings.BEDROCK_LLM_MODEL_ID,
        embedding_model=settings.BEDROCK_EMBEDDING_MODEL_ID,
        pinecone_vectors=stats.get("total_vectors", 0)
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized."
        )
    try:
        result = rag_pipeline.ask(request.question)
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/")
async def root():
    return {
        "message": "Python Q&A (Pinecone Cloud + AWS Bedrock)",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.ENVIRONMENT == "development"
    )
