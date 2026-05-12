import json
import os
import pytest
from pathlib import Path

os.environ["SMARTPACK_USE_MOCK"] = "1"

import memory_agent
from context_manager import ContextManager


@pytest.fixture
def ctx(tmp_path):
    return ContextManager(str(tmp_path / "ctx.json"))


class TestMemoryIntegration:
    def test_bullets_compression_fires_at_turn_6(self, ctx):
        for i in range(6):
            ctx.add_turn("user", f"message number {i} about topic", "chat")
        assert ctx.should_compress()
        ctx.compress()
        assert len(ctx.history) == 2
        assert ctx.summary != ""

    def test_output_stays_under_60_tokens(self):
        turns = [
            {"role": "user", "content": "This is a long message about artificial intelligence and machine learning systems that process data efficiently in production environments with low latency requirements and high throughput."},
            {"role": "assistant", "content": "I understand you are asking about AI and ML systems in production. These systems need careful optimization for latency and throughput. The key metrics are response time and requests per second."},
            {"role": "user", "content": "Can you explain the difference between batch processing and real time streaming for these systems?"},
            {"role": "assistant", "content": "Batch processing groups data together for periodic processing while streaming handles data continuously as it arrives in real time."},
        ]
        result = memory_agent.bullets_compress(turns)
        tokens = memory_agent._estimate_tokens(result)
        assert tokens <= 60

    def test_retrieval_accuracy_logged_correctly(self, tmp_path, monkeypatch):
        import memory_agent as ma
        log_file = tmp_path / "memory_metrics.jsonl"
        monkeypatch.setattr(ma, "_METRICS_FILE", log_file)
        log_file.parent.mkdir(exist_ok=True)
        turns = [
            {"role": "user", "content": "What is Python programming language?"},
            {"role": "assistant", "content": "Python is a high level programming language known for readability."},
        ]
        compressed = ma.compress_turns(turns)
        accuracy = ma.check_retrieval_accuracy(compressed, turns)
        ma.log_compression(len(turns), 50, 20, accuracy)
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["event"] == "compression"
        assert 0.0 <= record["accuracy"] <= 1.0

    def test_fallback_triggers_on_compression_failure(self, monkeypatch):
        import memory_agent as ma

        def raise_error(turns):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(ma, "bullets_compress", raise_error)
        turns = [{"role": "user", "content": "hello world this is a test message"}]
        result = ma.compress_turns(turns)
        assert result != ""
        assert "hello" in result or len(result) > 0

    def test_context_persistence_across_restart(self, tmp_path):
        ctx_file = str(tmp_path / "persist.json")
        ctx1 = ContextManager(ctx_file)
        for i in range(6):
            ctx1.add_turn("user", f"persistent message {i}", "chat")
        ctx1.compress()

        ctx2 = ContextManager(ctx_file)
        assert len(ctx2.history) == 2
        assert ctx2.summary != ""

    def test_bullets_format(self):
        turns = [
            {"role": "user", "content": "There was a bug in the system. We fixed the error. Next steps are to test."},
            {"role": "assistant", "content": "The bug fix was deployed successfully."},
        ]
        result = memory_agent.bullets_compress(turns)
        assert result.strip() != ""

    def test_empty_turns_returns_empty(self):
        result = memory_agent.bullets_compress([])
        assert result == ""

    def test_naive_fallback_produces_output(self):
        turns = [
            {"role": "user", "content": "x" * 300},
            {"role": "assistant", "content": "y" * 300},
        ]
        result = memory_agent.naive_compress(turns)
        assert len(result) > 0
        assert len(result) <= 400 + 10
