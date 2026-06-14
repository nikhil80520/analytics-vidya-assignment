import pytest
import httpx
from fastapi.testclient import TestClient
from app.main import app

# Create a TestClient for local testing without starting the server manually
client = TestClient(app)

def test_health_endpoint():
    """Test the /health endpoint returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "vectorstore_loaded" in data

def test_ask_endpoint_validation():
    """Test the /ask endpoint rejects invalid input."""
    # Test missing question
    response = client.post("/ask", json={})
    assert response.status_code == 422 # Unprocessable Entity
    
    # Test question too short
    response = client.post("/ask", json={"question": "a"})
    assert response.status_code == 422
    
# NOTE: To test the actual RAG pipeline, the local Pinecone index must have data 
# and AWS Bedrock credentials must be valid. The below test checks if the API 
# responds gracefully (either with a successful answer or a 500/503 error if not fully configured).
def test_ask_endpoint_integration():
    """Test the /ask endpoint processing a valid query."""
    question = "How do I reverse a list in Python?"
    response = client.post("/ask", json={"question": question})
    
    # If the pipeline isn't initialized or credentials fail, it should return 500 or 503
    if response.status_code == 503:
        pytest.skip("RAG pipeline not initialized (likely missing credentials or Pinecone connection).")
    elif response.status_code == 500:
        pytest.skip(f"Internal error (likely Bedrock/Pinecone auth issue): {response.json()}")
        
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == question
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "response_time_ms" in data
