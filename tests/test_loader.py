import pytest

from config import load_config
from loader import MockLlama, ModelLoader


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def loader(config):
    return ModelLoader(config, use_mock=True)


class TestMockLlama:
    def test_mock_llama_inference(self):
        mock = MockLlama("dummy_path.gguf")
        response = mock(prompt="test", max_tokens=100)

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]

    def test_mock_llama_usage_tracking(self):
        mock = MockLlama("dummy_path.gguf")
        response = mock(prompt="test", max_tokens=100)

        assert "usage" in response
        assert "prompt_tokens" in response["usage"]
        assert "completion_tokens" in response["usage"]


class TestModelLoader:
    def test_loader_init(self, loader):
        assert loader.current_model is None
        assert loader.current_domain is None
        assert loader.load_count == 0

    def test_get_ram_usage(self, loader):
        ram = loader.get_ram_usage()
        assert isinstance(ram, float)
        assert ram > 0

    def test_can_load_within_budget(self, loader):
        assert loader.can_load(1.0)

    def test_can_load_exceeds_budget(self, loader, config):
        assert not loader.can_load(config.server.ram_budget_gb + 10)

    def test_load_mock_model(self, loader):
        model = loader.load("code")
        assert model is not None
        assert loader.current_domain == "code"
        assert loader.load_count == 1

    def test_load_same_domain_no_reload(self, loader):
        loader.load("code")
        initial_count = loader.load_count

        model = loader.load("code")
        assert model is not None
        assert loader.load_count == initial_count

    def test_load_swap_domains(self, loader):
        loader.load("code")
        assert loader.current_domain == "code"

        loader.load("math")
        assert loader.current_domain == "math"
        assert loader.current_model is not None

    def test_unload_clears_model(self, loader):
        loader.load("code")
        assert loader.current_model is not None

        loader.unload()
        assert loader.current_model is None
        assert loader.current_domain is None

    def test_load_all_domains(self, loader):
        for domain in ["code", "math", "chat", "summarization"]:
            model = loader.load(domain)
            assert model is not None
            assert loader.current_domain == domain

    def test_model_inference(self, loader):
        model = loader.load("code")
        response = model("test prompt", max_tokens=50)

        assert "choices" in response
        assert len(response["choices"]) > 0

    def test_get_status(self, loader):
        loader.load("code")
        status = loader.get_status()

        assert status["loaded_domain"] == "code"
        assert status["loaded_model"] is not None
        assert status["ram_used_gb"] > 0
        assert status["load_count"] == 1

    def test_estimate_model_size_nonexistent(self, loader):
        size = loader.estimate_model_size("/nonexistent/model.gguf")
        assert size == 0.0

    def test_invalid_domain_raises(self, loader):
        with pytest.raises(ValueError):
            loader.load("invalid_domain")
