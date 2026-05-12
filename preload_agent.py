import io
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_METRICS_FILE = Path("logs/preload_metrics.jsonl")
_BUFFER_MB = 50
_BUFFER_BYTES = _BUFFER_MB * 1024 * 1024


def _log_event(event: str, domain: str, latency_ms: float) -> None:
    try:
        _METRICS_FILE.parent.mkdir(exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "domain": domain,
            "latency_ms": round(latency_ms, 3),
        }
        with open(_METRICS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log preload metrics: {e}")


class PreloadAgent:
    def __init__(self, buffer_mb: int = _BUFFER_MB):
        self.buffer_mb = buffer_mb
        self.buffer_bytes = buffer_mb * 1024 * 1024
        self.preload_hits = 0
        self.preload_misses = 0
        self.buffer_discards = 0

        self._lock = threading.Lock()
        self._predicted_domain: Optional[str] = None
        self._buffer: Optional[io.BytesIO] = None
        self._buffer_domain: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def predict_next(self, current_domain: str, conversation_history: list[dict]) -> str:
        domain_sequence = [t.get("domain", current_domain) for t in conversation_history if t.get("role") == "user"]
        if len(domain_sequence) >= 2:
            return domain_sequence[-1]
        return current_domain

    def trigger_preload(self, predicted_domain: str, model_path: str) -> None:
        with self._lock:
            if self._buffer_domain == predicted_domain and self._buffer is not None:
                return
            if self._buffer is not None:
                self.buffer_discards += 1
                self._buffer = None
                self._buffer_domain = None

        t = threading.Thread(
            target=self._warm_buffer,
            args=(predicted_domain, model_path),
            daemon=True,
        )
        t.start()
        with self._lock:
            self._thread = t
            self._predicted_domain = predicted_domain

    def _warm_buffer(self, domain: str, model_path: str) -> None:
        t0 = time.time()
        try:
            path = Path(model_path)
            if not path.exists():
                logger.debug(f"Preload: model path not found: {model_path}")
                return
            buf = io.BytesIO()
            with open(path, "rb") as f:
                data = f.read(self.buffer_bytes)
            buf.write(data)
            buf.seek(0)
            with self._lock:
                if self._predicted_domain == domain:
                    self._buffer = buf
                    self._buffer_domain = domain
                else:
                    self.buffer_discards += 1
            latency_ms = (time.time() - t0) * 1000
            _log_event("preload_warm", domain, latency_ms)
        except Exception as e:
            logger.warning(f"Preload thread error for {domain}: {e}, ignoring")

    def consume_buffer(self, requested_domain: str) -> Optional[io.BytesIO]:
        with self._lock:
            if self._buffer_domain == requested_domain and self._buffer is not None:
                buf = self._buffer
                self._buffer = None
                self._buffer_domain = None
                self.preload_hits += 1
                _log_event("preload_hit", requested_domain, 0)
                return buf
            if self._buffer is not None:
                self.buffer_discards += 1
                self._buffer = None
                self._buffer_domain = None
            self.preload_misses += 1
            _log_event("preload_miss", requested_domain, 0)
            return None

    def get_counters(self) -> dict:
        return {
            "preload_hits": self.preload_hits,
            "preload_misses": self.preload_misses,
            "buffer_discards": self.buffer_discards,
        }
