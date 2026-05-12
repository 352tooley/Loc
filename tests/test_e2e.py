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


class TestE2EConversation:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset context before each test."""
        ctx = ContextManager()
        ctx.reset()

    def test_6_turn_conversation(self, client, config):
        """Test 6-turn conversation with domain swaps."""
        turns = [
            {
                "query": "Write a binary search function in Python",
                "expected_domain": "code",
                "should_swap": False,
            },
            {
                "query": "How would you implement this recursively?",
                "expected_domain": "code",
                "should_swap": False,
            },
            {
                "query": "Solve for x: 2x² + 3x + 1 = 0",
                "expected_domain": "math",
                "should_swap": True,
            },
            {
                "query": "What's the discriminant formula?",
                "expected_domain": "math",
                "should_swap": False,
            },
            {
                "query": "What do you think about machine learning?",
                "expected_domain": "chat",
                "should_swap": True,
            },
            {
                "query": "Summarize the key points of our conversation",
                "expected_domain": "summarization",
                "should_swap": True,
            },
        ]

        last_domain = None

        for i, turn in enumerate(turns, 1):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartpack",
                    "messages": [{"role": "user", "content": turn["query"]}],
                    "stream": False,
                },
            )

            assert response.status_code == 200, f"Turn {i} failed: {response.text}"

            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0

            choice = data["choices"][0]
            assert "message" in choice
            assert choice["message"]["role"] == "assistant"
            assert len(choice["message"]["content"]) > 0

            health = client.get("/health").json()
            current_domain = health["loaded_domain"]

            print(f"Turn {i}: {turn['query'][:50]}...")
            print(f"  → Domain: {current_domain}")
            print(f"  → Response: {choice['message']['content'][:100]}...")

            if last_domain:
                if turn["should_swap"]:
                    assert current_domain != last_domain, (
                        f"Turn {i}: Expected swap from {last_domain}, "
                        f"but stayed in {current_domain}"
                    )
                else:
                    assert current_domain == last_domain, (
                        f"Turn {i}: Did not expect swap, "
                        f"but changed from {last_domain} to {current_domain}"
                    )

            last_domain = current_domain

    def test_conversation_context_persistence(self, client):
        """Test that context persists across turns."""
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice"},
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

        health = client.get("/health").json()
        assert health["context_turns"] >= 2

    def test_no_crash_on_unknown_domain(self, client):
        """Test that unknown domains don't crash the server."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "xyzabc qwerty 12345"}],
                "stream": False,
            },
        )

        assert response.status_code in [200, 503, 504]

    def test_response_format_compliance(self, client):
        """Test that responses follow OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )

        assert response.status_code == 200

        data = response.json()

        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        choice = data["choices"][0]
        assert "message" in choice
        assert "finish_reason" in choice

        message = choice["message"]
        assert "role" in message
        assert "content" in message

        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_temperature_affects_response(self, client):
        """Test that temperature parameter is accepted."""
        response_cold = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "Say hello"}],
                "temperature": 0.1,
                "stream": False,
            },
        )

        response_hot = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "Say hello"}],
                "temperature": 0.9,
                "stream": False,
            },
        )

        assert response_cold.status_code == 200
        assert response_hot.status_code == 200

    def test_max_tokens_respected(self, client):
        """Test that max_tokens parameter is accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartpack",
                "messages": [{"role": "user", "content": "Tell me a long story"}],
                "max_tokens": 50,
                "stream": False,
            },
        )

        assert response.status_code == 200

    def test_sequential_code_queries(self, client):
        """Test sequential queries in same domain."""
        queries = [
            "Write a function to sort an array",
            "Optimize that function",
            "Add error handling to it",
        ]

        last_domain = None

        for query in queries:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartpack",
                    "messages": [{"role": "user", "content": query}],
                    "stream": False,
                },
            )

            assert response.status_code == 200

            health = client.get("/health").json()
            current_domain = health["loaded_domain"]

            if last_domain:
                assert current_domain == last_domain, f"Unexpected domain swap for: {query}"

            last_domain = current_domain
