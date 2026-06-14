# app/utils/pinecone_store.py
import time
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.aws_bedrock import bedrock_manager

class PineconeManager:
    """Manages Pinecone cloud vector database for RAG."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
        self.index = None
        self.vectorstore = None
        
    def create_index(self, dimension: Optional[int] = None) -> None:
        """Create Pinecone index if it doesn't exist."""
        dim = dimension or self.settings.PINECONE_DIMENSION
        index_name = self.settings.PINECONE_INDEX_NAME
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"🏗️  Creating Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dim,
                metric=self.settings.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=self.settings.PINECONE_CLOUD,
                    region=self.settings.PINECONE_REGION
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                print("⏳ Waiting for index to be ready...")
                time.sleep(2)
            print(f"✅ Index {index_name} created!")
        else:
            print(f"📦 Index {index_name} already exists")
    
    def get_index(self):
        """Get or create index connection."""
        if self.index is None:
            self.index = self.pc.Index(self.settings.PINECONE_INDEX_NAME)
        return self.index
    
    def get_vectorstore(self):
        """Get LangChain Pinecone vector store."""
        if self.vectorstore is None:
            embeddings = bedrock_manager.get_embeddings()
            self.vectorstore = PineconeVectorStore(
                index=self.get_index(),
                embedding=embeddings,
                text_key="text"
            )
        return self.vectorstore
    
    def upload_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Chunk documents and upload to Pinecone in batches.
        Run this ONCE locally to populate the index.
        """
        batch_size = batch_size or self.settings.PINECONE_BATCH_SIZE
        
        # Split into chunks
        print("✂️  Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📦 Created {len(chunks):,} chunks")
        
        # Get embeddings and vectorstore
        embeddings = bedrock_manager.get_embeddings()
        vectorstore = self.get_vectorstore()
        
        # Upload in parallel batches using ThreadPoolExecutor
        print(f"☁️  Uploading to Pinecone in parallel batches of {batch_size}...")
        
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import time

        with tqdm(total=len(batches), desc="Uploading Batches") as pbar:
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all batches
                futures = {executor.submit(vectorstore.add_documents, batch): batch for batch in batches}
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Batch generated an exception: {exc}")
                    
                    pbar.update(1)
                    time.sleep(0.1) # Small delay to prevent hitting Bedrock limits too hard
        
        print(f"🎉 Successfully uploaded {len(chunks):,} vectors to Pinecone!")
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Get retriever for RAG."""
        default_kwargs = {
            "k": self.settings.TOP_K_RETRIEVAL,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        vectorstore = self.get_vectorstore()
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=default_kwargs
        )
    
    def delete_index(self) -> None:
        """Delete the entire index (use with caution)."""
        self.pc.delete_index(self.settings.PINECONE_INDEX_NAME)
        print(f"🗑️  Deleted index: {self.settings.PINECONE_INDEX_NAME}")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        index = self.get_index()
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }

# Singleton
pinecone_manager = PineconeManager()
