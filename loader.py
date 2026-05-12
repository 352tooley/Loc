import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional

import psutil

from config import SmartPackConfig
from preload_agent import PreloadAgent

logger = logging.getLogger(__name__)


class MockLlama:
    """Mock Llama class for testing without real model files."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 1, verbose: bool = False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> dict:
        return {
            "choices": [
                {
                    "text": f"Mock response to: {prompt[:50]}...",
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": max_tokens,
                "total_tokens": len(prompt.split()) + max_tokens,
            },
        }


class ModelLoader:
    def __init__(self, config: SmartPackConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        self.current_model: Optional[object] = None
        self.current_domain: Optional[str] = None
        self.load_count = 0
        self.last_load_metrics: dict = {}
        self.last_unload_metrics: dict = {}
        self._process = psutil.Process()
        self.preload_agent = PreloadAgent()

    def get_ram_usage(self) -> float:
        """Get current RAM usage in GB."""
        return self._process.memory_info().rss / (1024**3)

    def estimate_model_size(self, model_path: str) -> float:
        """Estimate model size with 1.1x overhead."""
        try:
            size_bytes = Path(model_path).stat().st_size
            return (size_bytes / (1024**3)) * 1.1
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Could not estimate model size: {e}")
            return 0.0

    def can_load(self, estimated_size_gb: float) -> bool:
        """Check if model fits within RAM budget."""
        current_usage = self.get_ram_usage()
        if current_usage + estimated_size_gb > self.config.server.ram_budget_gb:
            logger.warning(
                f"Not enough RAM: {current_usage:.2f}GB + {estimated_size_gb:.2f}GB > {self.config.server.ram_budget_gb:.2f}GB"
            )
            return False
        return True

    def unload(self) -> None:
        """Unload current model and free RAM."""
        if self.current_model is not None:
            ram_before = self.get_ram_usage()
            unload_start = time.time()
            logger.info(f"Unloading model from domain: {self.current_domain}")
            del self.current_model
            self.current_model = None
            self.current_domain = None
            gc.collect()
            time.sleep(0.1)
            ram_after = self.get_ram_usage()
            self.last_unload_metrics = {
                "unload_time_seconds": time.time() - unload_start,
                "ram_before_gb": ram_before,
                "ram_after_gb": ram_after,
                "ram_freed_gb": max(0.0, ram_before - ram_after),
            }

    def load(self, domain: str) -> object:
        """Load model for specified domain, unloading previous if needed."""
        domain_config = self.config.domains.get(domain)
        if domain_config is None:
            raise ValueError(f"Unknown domain: {domain}")

        if self.current_domain == domain and self.current_model is not None:
            logger.debug(f"Model for domain {domain} already loaded")
            return self.current_model

        model_path = domain_config.model_path or self.config.domains[self.config.fallback_domain].model_path

        if not model_path and not self.use_mock:
            logger.warning(f"No model path for domain {domain} or fallback {self.config.fallback_domain}")
            raise ValueError(f"No model path available for domain {domain}")

        ram_before = self.get_ram_usage()
        load_start = time.time()

        self.unload()

        if model_path:
            estimated_size = self.estimate_model_size(model_path)
            if estimated_size > 0 and not self.can_load(estimated_size):
                raise MemoryError(f"Model {model_path} exceeds RAM budget")
        else:
            estimated_size = 0

        try:
            if self.use_mock:
                logger.info(f"Loading mock model for domain: {domain}")
                self.current_model = MockLlama(
                    model_path=model_path,
                    n_ctx=self.config.server.context_window,
                    n_threads=max(1, os.cpu_count() - 1 if os.cpu_count() else 1),
                    verbose=False,
                )
            else:
                from llama_cpp import Llama

                logger.info(f"Loading model for domain {domain} from {model_path}")
                n_ctx = self._effective_context_window()
                self.current_model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=max(1, os.cpu_count() - 1 if os.cpu_count() else 1),
                    n_batch=min(512, n_ctx),
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )

            self.current_domain = domain
            self.load_count += 1

            ram_after = self.get_ram_usage()
            load_time = time.time() - load_start
            self.last_load_metrics = {
                "domain": domain,
                "model_path": model_path,
                "estimated_size_gb": estimated_size,
                "load_time_seconds": load_time,
                "ram_before_gb": ram_before,
                "ram_after_gb": ram_after,
                "ram_delta_gb": ram_after - ram_before,
                "unload": self.last_unload_metrics,
            }

            logger.info(
                f"Loaded {domain_config.model_name} ({Path(model_path).name}) "
                f"in {load_time:.2f}s. RAM: {ram_before:.2f}GB → {ram_after:.2f}GB"
            )

            self._trigger_preload(domain)

            return self.current_model
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _effective_context_window(self) -> int:
        """Keep real model context inside a conservative RAM envelope."""
        configured = self.config.server.context_window
        if self.config.server.ram_budget_gb <= 4:
            return min(configured, 2048)
        if self.config.server.ram_budget_gb <= 6:
            return min(configured, 3072)
        return configured

    def _trigger_preload(self, current_domain: str) -> None:
        try:
            predicted = self.preload_agent.predict_next(current_domain, [])
            predicted_config = self.config.domains.get(predicted)
            if predicted_config and predicted_config.model_path:
                self.preload_agent.trigger_preload(predicted, predicted_config.model_path)
        except Exception as e:
            logger.warning(f"Preload trigger failed: {e}")

    def get_status(self) -> dict:
        """Get current loader status."""
        return {
            "loaded_domain": self.current_domain,
            "loaded_model": Path(self.config.domains[self.current_domain].model_path).name
            if self.current_domain
            else None,
            "ram_used_gb": self.get_ram_usage(),
            "load_count": self.load_count,
            "last_load_metrics": self.last_load_metrics,
        }
