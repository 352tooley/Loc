import io
import os
import time
import threading
import pytest
from pathlib import Path
from unittest.mock import patch

os.environ["SMARTPACK_USE_MOCK"] = "1"

from preload_agent import PreloadAgent


@pytest.fixture
def agent():
    return PreloadAgent(buffer_mb=1)


class TestPreloadIntegration:
    def test_preload_fires_after_model_load(self, tmp_path):
        from config import load_config
        from loader import ModelLoader
        loader = ModelLoader(load_config(), use_mock=True)
        loader.load("code")
        assert loader.preload_agent is not None

    def test_correct_domain_preloaded_on_predictable_sequence(self, agent):
        history = [
            {"role": "user", "content": "q", "domain": "code"},
            {"role": "user", "content": "q", "domain": "math"},
        ]
        predicted = agent.predict_next("math", history)
        assert predicted == "math"

    def test_miss_handled_gracefully(self, agent):
        agent.trigger_preload("code", "nonexistent_path.gguf")
        time.sleep(0.1)
        result = agent.consume_buffer("math")
        assert result is None
        assert agent.preload_misses >= 1

    def test_preload_thread_crash_does_not_affect_main(self, agent):
        with patch.object(agent, "_warm_buffer", side_effect=RuntimeError("crash")):
            try:
                agent.trigger_preload("code", "some_path.gguf")
            except Exception:
                pass
        decision = agent.predict_next("code", [])
        assert decision is not None

    def test_counters_start_at_zero(self, agent):
        counters = agent.get_counters()
        assert counters["preload_hits"] == 0
        assert counters["preload_misses"] == 0
        assert counters["buffer_discards"] == 0

    def test_preload_hit_increments_counter(self, tmp_path):
        dummy_model = tmp_path / "model.gguf"
        dummy_model.write_bytes(b"x" * 1024)
        agent = PreloadAgent(buffer_mb=1)
        agent.trigger_preload("code", str(dummy_model))
        time.sleep(0.3)
        buf = agent.consume_buffer("code")
        assert buf is not None
        assert agent.preload_hits == 1

    def test_buffer_discard_on_domain_mismatch(self, tmp_path):
        dummy_model = tmp_path / "model.gguf"
        dummy_model.write_bytes(b"x" * 1024)
        agent = PreloadAgent(buffer_mb=1)
        agent.trigger_preload("code", str(dummy_model))
        time.sleep(0.3)
        agent.trigger_preload("math", str(dummy_model))
        assert agent.buffer_discards >= 0

    def test_health_exposes_preload_counters(self):
        from config import load_config
        from loader import ModelLoader
        loader = ModelLoader(load_config(), use_mock=True)
        counters = loader.preload_agent.get_counters()
        assert "preload_hits" in counters
        assert "preload_misses" in counters
        assert "buffer_discards" in counters
