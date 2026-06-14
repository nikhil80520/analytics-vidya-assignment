# app/rag_pipeline.py
import time
from typing import List, Dict
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from app.config import get_settings
from app.utils.aws_bedrock import bedrock_manager
from app.utils.pinecone_store import pinecone_manager

RAG_SYSTEM_PROMPT = """You are an expert Python programming assistant powered by Qwen via AWS Bedrock. Your answers are based STRICTLY on the provided Stack Overflow context from Pinecone.

Guidelines:
1. Provide clear, accurate Python answers with code examples
2. Cite source question titles when possible
3. If context is insufficient, say so and provide general knowledge with a disclaimer
4. Be concise but thorough
5. Use proper markdown formatting for code blocks"""

RAG_USER_TEMPLATE = """Context from Stack Overflow (retrieved from Pinecone):
{context}

---

Question: {question}

Provide a well-structured Python answer:"""

class PythonQARAG:
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Pinecone retriever (no local FAISS needed!)
        print("☁️  Connecting to Pinecone cloud vector database...")
        self.retriever = pinecone_manager.get_retriever()
        
        # Initialize Qwen via Bedrock
        print("🤖 Initializing Qwen model...")
        self.llm = bedrock_manager.get_chat_model()
        
        # Build chain
        self.chain = self._build_chain()
        print("✅ RAG pipeline ready!")
    
    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('title', 'Unknown')
            formatted.append(f"[{i}] {source}\n{doc.page_content[:1500]}")
        return "\n\n---\n\n".join(formatted)
    
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_TEMPLATE)
        ])
        
        return (
            RunnableParallel({
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str) -> Dict:
        start_time = time.time()
        
        # Retrieve from Pinecone cloud
        docs = self.retriever.invoke(question)
        
        # Generate answer with Qwen
        answer = self.chain.invoke(question)
        
        response_time = (time.time() - start_time) * 1000
        
        sources = []
        for doc in docs:
            sources.append({
                "title": doc.metadata.get('title', 'Unknown'),
                "score": doc.metadata.get('question_score', 0),
                "answer_score": doc.metadata.get('answer_score', 0),
                "tags": doc.metadata.get('tags', ''),
                "preview": doc.page_content[:200] + "..."
            })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(docs),
            "vector_db": "pinecone",
            "llm_used": self.settings.BEDROCK_LLM_MODEL_ID,
            "embedding_model": self.settings.BEDROCK_EMBEDDING_MODEL_ID,
            "response_time_ms": round(response_time, 2)
        }
