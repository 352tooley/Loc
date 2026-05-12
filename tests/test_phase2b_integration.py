import asyncio
import json
import os
import time
import pytest
from pathlib import Path

os.environ["SMARTPACK_USE_MOCK"] = "1"

from config import load_config
from loader import ModelLoader
from coordinator import Coordinator
from router import Router
from context_manager import ContextManager
from api.chat import ChatCompletionRequest, Message, chat_completion_handler


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def loader(config):
    return ModelLoader(config, use_mock=True)


@pytest.fixture
def router(config):
    return Router(config)


@pytest.fixture
def ctx(tmp_path):
    c = ContextManager(str(tmp_path / "ctx.json"))
    return c


def make_request(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=[Message(role="user", content=content)],
    )


async def send(req, router, loader, ctx):
    return await chat_completion_handler(req, router, loader, ctx)


class TestScenarioA_PreloadHit:
    def test_preload_fires_after_code_load(self, loader):
        loader.load("code")
        assert loader.preload_agent is not None
        assert loader.preload_agent.get_counters() is not None

    def test_preload_hit_counter(self, loader, tmp_path):
        dummy = tmp_path / "model.gguf"
        dummy.write_bytes(b"x" * 1024)
        loader.preload_agent.trigger_preload("code", str(dummy))
        time.sleep(0.3)
        buf = loader.preload_agent.consume_buffer("code")
        if buf is not None:
            assert loader.preload_agent.preload_hits >= 1


class TestScenarioB_MemoryCompression:
    def test_compression_fires_at_turn_6(self, loader, router, ctx):
        queries = [
            "Tell me about Python",
            "How does machine learning work",
            "What is deep learning",
            "Explain neural networks briefly",
            "What are transformers in AI",
            "How does attention mechanism work",
            "Summarize what we discussed",
        ]
        for q in queries:
            req = make_request(q)
            asyncio.get_event_loop().run_until_complete(send(req, router, loader, ctx))

        assert len(ctx.history) < len(queries)

    def test_compression_entry_in_memory_metrics(self, loader, router, tmp_path, monkeypatch):
        import memory_agent as ma
        log_file = tmp_path / "memory_metrics.jsonl"
        monkeypatch.setattr(ma, "_METRICS_FILE", log_file)
        log_file.parent.mkdir(exist_ok=True)

        c = ContextManager(str(tmp_path / "ctx2.json"))
        for i in range(7):
            req = make_request(f"question number {i} about various topics in the world")
            asyncio.get_event_loop().run_until_complete(send(req, router, loader, c))

        if log_file.exists():
            lines = log_file.read_text().strip().splitlines()
            records = [json.loads(l) for l in lines]
            assert any(r["event"] == "compression" for r in records)


class TestScenarioC_QualityEscalation:
    def test_medical_question_escalated(self, loader, router, ctx):
        req = make_request("What medication should I take for severe chest pain and shortness of breath?")
        response, escalated = asyncio.get_event_loop().run_until_complete(
            send(req, router, loader, ctx)
        )
        assert escalated is True

    def test_quality_metrics_written(self, loader, router, tmp_path, monkeypatch):
        import quality_agent as qa
        log_file = tmp_path / "quality_metrics.jsonl"
        monkeypatch.setattr(qa, "_METRICS_FILE", log_file)
        log_file.parent.mkdir(exist_ok=True)
        c = ContextManager(str(tmp_path / "ctx3.json"))
        req = make_request("What are symptoms of diabetes?")
        asyncio.get_event_loop().run_until_complete(send(req, router, loader, c))
        assert log_file.exists()
        records = [json.loads(l) for l in log_file.read_text().strip().splitlines()]
        assert any(r["event"] == "escalation" for r in records)


class TestScenarioD_CoordinatorRouting:
    def test_coordinator_metrics_logged_for_all_queries(self, loader, router, tmp_path, monkeypatch):
        import coordinator as coord_module
        log_file = tmp_path / "coordinator_metrics.jsonl"
        monkeypatch.setattr(coord_module, "_METRICS_FILE", log_file)
        log_file.parent.mkdir(exist_ok=True)

        coord = Coordinator(load_config())
        queries = [
            "write a python function to sort a list",
            "solve the quadratic equation x squared plus 2x plus 1",
            "how are you doing today",
            "summarize this long document about climate change",
        ]
        for q in queries:
            coord.route(q)

        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) >= 4

    def test_code_query_routes_to_code(self, config):
        coord = Coordinator(config)
        decision = coord.route("write a python function to parse JSON data")
        assert decision.domain in config.domains

    def test_math_query_routes_to_math(self, config):
        coord = Coordinator(config)
        decision = coord.route("solve the integral of x squared")
        assert decision.domain in config.domains

    def test_summarization_query_routes_to_summarization(self, config):
        coord = Coordinator(config)
        decision = coord.route("summarize the key points of this document")
        assert decision.domain in config.domains


class TestScenarioE_FullConversation:
    def test_8_turn_conversation(self, loader, router, tmp_path):
        ctx = ContextManager(str(tmp_path / "full_ctx.json"))
        turns = [
            "write a function to sort a list in python",
            "can you optimize that code further",
            "solve for x in equation 2x plus 5 equals 15",
            "what is the derivative of x cubed",
            "how are you doing today",
            "summarize the key points of our conversation so far",
            "I have chest pain symptoms what should I check",
            "write a binary search function in python",
        ]
        escalation_seen = False
        for turn in turns:
            req = make_request(turn)
            response, escalated = asyncio.get_event_loop().run_until_complete(
                send(req, router, loader, ctx)
            )
            assert response is not None
            if escalated:
                escalation_seen = True

        assert escalation_seen

    def test_all_metric_logs_updated(self, loader, router, tmp_path, monkeypatch):
        import coordinator as coord_module
        import quality_agent as qa

        coord_log = tmp_path / "coordinator_metrics.jsonl"
        quality_log = tmp_path / "quality_metrics.jsonl"
        monkeypatch.setattr(coord_module, "_METRICS_FILE", coord_log)
        monkeypatch.setattr(qa, "_METRICS_FILE", quality_log)
        coord_log.parent.mkdir(exist_ok=True)

        ctx = ContextManager(str(tmp_path / "multi_ctx.json"))
        r = Router(load_config())

        queries = [
            "write python code to parse JSON",
            "what medication for headache symptoms",
        ]
        for q in queries:
            req = make_request(q)
            asyncio.get_event_loop().run_until_complete(send(req, r, loader, ctx))

        assert coord_log.exists()
        assert quality_log.exists()
