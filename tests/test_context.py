import json
import pytest
from pathlib import Path

from context_manager import ContextManager


@pytest.fixture
def context_file(tmp_path):
    return str(tmp_path / "test_context.json")


@pytest.fixture
def context(context_file):
    return ContextManager(context_file)


class TestContextManager:
    def test_init_empty(self, context):
        assert context.history == []
        assert context.summary == ""

    def test_add_turn(self, context):
        context.add_turn("user", "Hello", "chat")
        assert len(context.history) == 1
        assert context.history[0]["role"] == "user"
        assert context.history[0]["content"] == "Hello"
        assert context.history[0]["domain"] == "chat"

    def test_get_recent(self, context):
        for i in range(5):
            context.add_turn("user", f"message {i}", "chat")

        recent = context.get_recent(3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "message 4"

    def test_get_recent_more_than_available(self, context):
        context.add_turn("user", "msg1", "chat")
        context.add_turn("user", "msg2", "chat")

        recent = context.get_recent(10)
        assert len(recent) == 2

    def test_set_summary(self, context):
        context.set_summary("Conversation about AI")
        assert context.get_summary() == "Conversation about AI"

    def test_should_compress_false(self, context):
        for i in range(5):
            context.add_turn("user", f"msg {i}", "chat")

        assert not context.should_compress()

    def test_should_compress_true(self, context):
        for i in range(6):
            context.add_turn("user", f"msg {i}", "chat")

        assert context.should_compress()

    def test_compress(self, context):
        for i in range(6):
            context.add_turn("user", f"msg {i}", "chat")

        context.compress()

        assert len(context.history) == 2
        assert context.history[0]["content"] == "msg 4"
        assert context.history[1]["content"] == "msg 5"
        assert context.summary != ""

    def test_persistence_save(self, context, context_file):
        context.add_turn("user", "Hello", "chat")
        context.set_summary("Test summary")

        assert Path(context_file).exists()

        with open(context_file) as f:
            data = json.load(f)

        assert len(data["history"]) == 1
        assert data["summary"] == "Test summary"

    def test_persistence_load(self, context_file):
        data = {
            "summary": "Loaded summary",
            "history": [{"role": "user", "content": "test", "domain": "chat", "timestamp": "2024-01-01T00:00:00"}],
        }

        with open(context_file, "w") as f:
            json.dump(data, f)

        loaded_context = ContextManager(context_file)
        assert loaded_context.summary == "Loaded summary"
        assert len(loaded_context.history) == 1
        assert loaded_context.history[0]["content"] == "test"

    def test_corrupted_context_recovery(self, context_file):
        with open(context_file, "w") as f:
            f.write("invalid json")

        context = ContextManager(context_file)
        assert context.history == []
        assert context.summary == ""

    def test_reset(self, context):
        context.add_turn("user", "msg", "chat")
        context.set_summary("summary")

        context.reset()

        assert context.history == []
        assert context.summary == ""

    def test_get_context_for_model(self, context):
        context.set_summary("Test summary")
        context.add_turn("user", "msg1", "chat")
        context.add_turn("assistant", "response", "chat")

        summary, recent = context.get_context_for_model()
        assert summary == "Test summary"
        assert len(recent) <= 3

    def test_format_history_for_prompt(self, context):
        context.add_turn("user", "question", "chat")
        context.add_turn("assistant", "answer", "chat")

        formatted = context.format_history_for_prompt()
        assert "User" in formatted or "user" in formatted
        assert "Assistant" in formatted or "assistant" in formatted

    def test_multiple_domains_in_history(self, context):
        context.add_turn("user", "code question", "code")
        context.add_turn("assistant", "code answer", "code")
        context.add_turn("user", "math question", "math")

        recent = context.get_recent(2)
        assert recent[0]["domain"] == "code"
        assert recent[1]["domain"] == "math"
