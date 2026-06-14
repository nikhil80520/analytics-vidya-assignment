# app/config.py
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # AWS
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_S3_BUCKET_NAME: str = ""
    
    # Kaggle
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    
    # Bedrock Models
    BEDROCK_LLM_MODEL_ID: str = "qwen.qwen3-coder-next"
    BEDROCK_EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    
    # Pinecone (Free Tier Cloud Vector DB)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "python-qa-index"
    PINECONE_CLOUD: str = "aws"           # aws, gcp, azure
    PINECONE_REGION: str = "us-east-1"    # Match your AWS region
    PINECONE_DIMENSION: int = 1536          # Titan v2 = 1024, Cohere = 1024
    PINECONE_METRIC: str = "cosine"       # cosine, euclidean, dotproduct
    
    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"
    
    # RAG Config
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    
    # Batch size for Pinecone upserts (free tier limit: 100 vectors/batch)
    PINECONE_BATCH_SIZE: int = 100
    
    model_config = SettingsConfigDict(env_file='.env')

@lru_cache()
def get_settings() -> Settings:
    return Settings()
