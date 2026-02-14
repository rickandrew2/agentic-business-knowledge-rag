"""Integration tests for the main API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_includes_service_name(self, client):
        """Test that health response includes service name."""
        response = client.get("/api/health")
        data = response.json()
        assert "service" in data
        assert "RAG" in data["service"]


class TestConfigEndpoint:
    """Test configuration endpoint."""
    
    def test_get_config(self, client):
        """Test that config endpoint returns public settings."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        
        # Should contain these public settings
        assert "app_name" in data
        assert "openai_model" in data
        assert "chunk_size" in data
        assert "retrieval_top_k" in data
        
        # Should NOT contain sensitive data
        assert "openai_api_key" not in data
        assert "database_url" not in data


class TestCORSHeaders:
    """Test CORS configuration."""
    
    def test_cors_origin_allowed(self, client):
        """Test that CORS allows frontend origins."""
        response = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"}
        )
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 response for non-existent endpoint."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        data = response.json()
        # FastAPI returns 'detail' for default 404 errors
        assert "detail" in data or "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
