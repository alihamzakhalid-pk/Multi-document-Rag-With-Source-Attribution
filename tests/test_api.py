"""
Integration tests for the RAG API.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns valid response."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "documents_loaded" in data
        assert "total_chunks" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Multi-Document RAG System"
        assert "version" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints."""
    
    def test_list_documents_empty(self, client):
        """Test listing documents when empty."""
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_chunks" in data
    
    def test_upload_unsupported_file(self, client):
        """Test uploading unsupported file type."""
        files = {"file": ("test.xyz", b"content", "text/plain")}
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]
    
    def test_upload_txt_file(self, client):
        """Test uploading a TXT file."""
        content = b"This is a test document for the RAG system."
        files = {"file": ("test.txt", content, "text/plain")}
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_name"] == "test.txt"
        assert data["chunks_created"] > 0
    
    def test_delete_nonexistent_document(self, client):
        """Test deleting a document that doesn't exist."""
        response = client.delete("/api/v1/documents/nonexistent.pdf")
        
        assert response.status_code == 404


class TestQueryEndpoint:
    """Tests for query endpoint."""
    
    def test_query_empty_question(self, client):
        """Test query with empty question fails validation."""
        response = client.post(
            "/api/v1/query",
            json={"question": ""}
        )
        
        # Pydantic validation should fail
        assert response.status_code == 422
    
    def test_query_structure(self, client):
        """Test query response has correct structure."""
        # First upload a document
        content = b"The capital of France is Paris. Paris is a beautiful city."
        files = {"file": ("geography.txt", content, "text/plain")}
        client.post("/api/v1/documents/upload", files=files)
        
        # Skip if API key not configured (will get 503)
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the capital of France?"}
        )
        
        # Either success or API key not configured
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
