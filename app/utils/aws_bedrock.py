# app/utils/aws_bedrock.py
import boto3
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from app.config import get_settings

class BedrockManager:
    """Manages AWS Bedrock connections for LLMs and embeddings."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Configure the boto3 client
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.settings.AWS_REGION,
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY,
        )
        self.llm = None
        self.embeddings = None

    def get_chat_model(self) -> ChatBedrockConverse:
        """Returns the ChatBedrockConverse LLM instance."""
        if self.llm is None:
            self.llm = ChatBedrockConverse(
                client=self.client,
                model=self.settings.BEDROCK_LLM_MODEL_ID,
                temperature=self.settings.TEMPERATURE,
                max_tokens=self.settings.MAX_TOKENS
            )
        return self.llm

    def get_embeddings(self) -> BedrockEmbeddings:
        """Returns the BedrockEmbeddings instance."""
        if self.embeddings is None:
            self.embeddings = BedrockEmbeddings(
                client=self.client,
                model_id=self.settings.BEDROCK_EMBEDDING_MODEL_ID
            )
        return self.embeddings

# Singleton instance
bedrock_manager = BedrockManager()
