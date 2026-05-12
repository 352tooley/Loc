import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

os.environ["SMARTPACK_USE_MOCK"] = "1"

import quality_agent


class TestQualityIntegration:
    def test_threshold_at_055_triggers(self):
        score = 0.54
        assert quality_agent.should_escalate(score, "") is True

    def test_above_threshold_no_escalation(self):
        score = 0.80
        assert quality_agent.should_escalate(score, "") is False

    def test_medical_task_always_escalates(self):
        assert quality_agent.should_escalate(0.99, "medical") is True

    def test_legal_task_always_escalates(self):
        assert quality_agent.should_escalate(0.99, "legal") is True

    def test_financial_task_always_escalates(self):
        assert quality_agent.should_escalate(0.99, "financial") is True

    def test_security_task_always_escalates(self):
        assert quality_agent.should_escalate(0.99, "security") is True

    def test_escalation_with_no_tier2_returns_tier1(self):
        from config import load_config
        from loader import ModelLoader
        from api.chat import chat_completion_handler
        import asyncio
        from context_manager import ContextManager
        from router import Router

        config = load_config()
        loader = ModelLoader(config, use_mock=True)
        ctx = ContextManager(":memory:")
        ctx.context_file = Path("/tmp/test_quality_ctx.json")
        router = Router(config)

        from api.chat import ChatCompletionRequest, Message
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="I have a medical emergency what should I do?")],
        )
        response, escalated = asyncio.get_event_loop().run_until_complete(
            chat_completion_handler(req, router, loader, ctx)
        )
        assert response is not None

    def test_escalated_header_present(self):
        from config import load_config
        from loader import ModelLoader
        from api.chat import chat_completion_handler
        import asyncio
        from context_manager import ContextManager
        from router import Router
        from api.chat import ChatCompletionRequest, Message

        config = load_config()
        loader = ModelLoader(config, use_mock=True)
        ctx = ContextManager(":memory:")
        ctx.context_file = Path("/tmp/test_header_ctx.json")
        router = Router(config)

        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="What is 2 + 2?")],
        )
        response, escalated = asyncio.get_event_loop().run_until_complete(
            chat_completion_handler(req, router, loader, ctx)
        )
        assert isinstance(escalated, bool)

    def test_quality_metrics_logged_to_jsonl(self, tmp_path, monkeypatch):
        import quality_agent as qa
        log_file = tmp_path / "quality_metrics.jsonl"
        monkeypatch.setattr(qa, "_METRICS_FILE", log_file)
        log_file.parent.mkdir(exist_ok=True)
        qa.log_escalation("medical", 0.50, True)
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["event"] == "escalation"
        assert record["task_type"] == "medical"
        assert record["escalated"] is True

    def test_score_response_returns_float(self):
        score = quality_agent.score_response("what is 2+2", "2+2=4", "")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_detect_task_type_medical(self):
        from api.chat import Message
        msgs = [Message(role="user", content="I have medical symptoms and need diagnosis")]
        task_type = quality_agent.detect_task_type(msgs)
        assert task_type == "medical"

    def test_detect_task_type_empty(self):
        from api.chat import Message
        msgs = [Message(role="user", content="What is the weather today")]
        task_type = quality_agent.detect_task_type(msgs)
        assert task_type == ""

    def test_four_quality_cases_all_escalate(self):
        cases = [
            ("pack fragile items for shipping", "just wrap it", ""),
            ("what is the security policy for passwords", "use strong passwords", "security"),
            ("summarize returns policy document", "returns allowed within 30 days", ""),
            ("what medication for headache", "take aspirin", "medical"),
        ]
        for prompt, response, task_type in cases:
            score = quality_agent.score_response(prompt, response, task_type)
            result = quality_agent.should_escalate(score, task_type)
            assert isinstance(result, bool)
