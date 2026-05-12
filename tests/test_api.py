import pytest
from fastapi.testclient import TestClient

from config import load_config
from context_manager import ContextManager
from coordinator import Coordinator
from loader import ModelLoader
from main import app
from router import Router


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def config():
    return load_config()


class TestChatCompletions:
    def test_chat_completions_basic(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]

    def test_chat_completions_with_temperature(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "Write code"}],
                "temperature": 0.5,
                "max_tokens": 256,
                "stream": False,
            },
        )

        assert response.status_code == 200

    def test_chat_completions_empty_messages(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [],
                "stream": False,
            },
        )

        assert response.status_code == 400

    def test_chat_completions_multi_turn(self, client):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": messages,
                "stream": False,
            },
        )

        assert response.status_code == 200


class TestModels:
    def test_models_endpoint(self, client):
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_models_include_domains(self, client):
        response = client.get("/v1/models")
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]

        assert "code" in model_ids
        assert "math" in model_ids
        assert "chat" in model_ids
        assert "summarization" in model_ids


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "loaded_domain" in data
        assert "ram_used_gb" in data
        assert "ram_budget_gb" in data

    def test_health_coordinator_mode(self, client):
        response = client.get("/health")
        data = response.json()

        assert data["coordinator_mode"] in ["keyword", "llama"]


class TestAdmin:
    def test_swap_valid_domain(self, client):
        response = client.post(
            "/v1/admin/swap",
            json={"domain": "code"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "code"
        assert data["status"] == "ok"

    def test_swap_invalid_domain(self, client):
        response = client.post(
            "/v1/admin/swap",
            json={"domain": "invalid"},
        )

        assert response.status_code == 400

    def test_swap_missing_domain(self, client):
        response = client.post(
            "/v1/admin/swap",
            json={},
        )

        assert response.status_code == 400

    def test_health_after_swap(self, client):
        client.post(
            "/v1/admin/swap",
            json={"domain": "math"},
        )

        response = client.get("/health")
        data = response.json()
        assert data["loaded_domain"] == "math"


class TestErrorHandling:
    def test_invalid_model_file(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )
        assert response.status_code in [200, 503, 504]

    def test_request_count_increment(self, client):
        health1 = client.get("/health").json()
        count1 = health1["total_requests"]

        client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )

        health2 = client.get("/health").json()
        count2 = health2["total_requests"]

        assert count2 > count1
