import json
import os
import pytest
from unittest.mock import MagicMock, patch

os.environ["SMARTPACK_USE_MOCK"] = "1"

from config import load_config
from coordinator import Coordinator, KeywordRouter


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def coordinator(config):
    return Coordinator(config)


class TestCoordinatorUpgrade:
    def test_keyword_mode_in_mock(self, coordinator):
        assert coordinator.mode == "keyword"

    def test_keyword_returns_valid_domain(self, coordinator, config):
        decision = coordinator.route("write a python function")
        assert decision.domain in config.domains

    def test_low_confidence_uses_keyword_fallback(self, config):
        c = Coordinator(config)
        c.mode = "model"
        c.llama = MagicMock()
        c.llama.return_value = {
            "choices": [{"text": '{"domain":"code","confidence":0.3,"reasoning":"low","preload_hint":"code"}'}]
        }
        decision = c.route("hello world")
        assert decision.domain in config.domains

    def test_invalid_json_triggers_retry_then_fallback(self, config):
        c = Coordinator(config)
        c.mode = "model"
        c.llama = MagicMock()
        c.llama.return_value = {"choices": [{"text": "not json at all"}]}
        decision = c.route("some query")
        assert decision.domain in config.domains

    def test_coordinator_mode_reported_in_keyword_mode(self, coordinator):
        assert coordinator.mode in ("keyword", "model")

    def test_latency_logged_to_jsonl(self, coordinator, tmp_path, monkeypatch):
        import coordinator as coord_module
        log_file = tmp_path / "coordinator_metrics.jsonl"
        monkeypatch.setattr(coord_module, "_METRICS_FILE", log_file)
        log_file.parent.mkdir(exist_ok=True)
        coordinator.route("test query for logging")
        assert log_file.exists()
        line = log_file.read_text().strip().splitlines()[-1]
        record = json.loads(line)
        assert "mode" in record
        assert "domain" in record
        assert "latency_ms" in record

    def test_extract_json_strips_markdown(self):
        raw = "```json\n{\"domain\":\"code\",\"confidence\":0.9,\"reasoning\":\"test\",\"preload_hint\":\"math\"}\n```"
        result = Coordinator._extract_json(raw)
        data = json.loads(result)
        assert data["domain"] == "code"

    def test_high_confidence_uses_model_result(self, config):
        c = Coordinator(config)
        c.mode = "model"
        c.llama = MagicMock()
        c.llama.return_value = {
            "choices": [{"text": '{"domain":"math","confidence":0.95,"reasoning":"math problem","preload_hint":"math"}'}]
        }
        decision = c.route("solve this integral")
        assert decision.domain == "math"
        assert decision.confidence == 0.95
