import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import SmartPackConfig

logger = logging.getLogger(__name__)

_METRICS_FILE = Path("logs/coordinator_metrics.jsonl")


def _log_routing(mode: str, domain: str, confidence: float, latency_ms: float, used_fallback: bool) -> None:
    try:
        _METRICS_FILE.parent.mkdir(exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "routing",
            "mode": mode,
            "domain": domain,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 3),
            "used_fallback": used_fallback,
        }
        with open(_METRICS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log coordinator metrics: {e}")


class RoutingDecision:
    def __init__(self, domain: str, confidence: float, reasoning: str, preload_hint: str):
        self.domain = domain
        self.confidence = confidence
        self.reasoning = reasoning
        self.preload_hint = preload_hint

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "preload_hint": self.preload_hint,
        }


class KeywordRouter:
    def __init__(self, config: SmartPackConfig):
        self.config = config
        self.domain_keywords = {
            domain_name: domain_config.keywords
            for domain_name, domain_config in config.domains.items()
        }

    def route(
        self,
        query: str,
        summary: str = "",
        recent_history: str = "",
    ) -> RoutingDecision:
        scores = {}
        query_lower = query.lower()

        for domain, keywords in self.domain_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            scores[domain] = count

        max_score = max(scores.values()) if scores.values() else 0

        if max_score == 0:
            selected_domain = self.config.fallback_domain
            confidence = 0.0
        else:
            selected_domain = max(scores, key=scores.get)
            max_possible = max(len(keywords) for keywords in self.domain_keywords.values())
            confidence = min(max_score / max_possible, 1.0)

        return RoutingDecision(
            domain=selected_domain,
            confidence=confidence,
            reasoning=f"matched {max_score} keywords",
            preload_hint=selected_domain,
        )


class Coordinator:
    def __init__(self, config: SmartPackConfig):
        self.config = config
        self.llama = None
        self.keyword_router = KeywordRouter(config)
        self.mode = "keyword"
        self._init_llama()

    def _init_llama(self) -> None:
        if os.getenv("SMARTPACK_USE_MOCK", "").lower() in {"1", "true", "yes", "on"}:
            logger.info("SMARTPACK_USE_MOCK enabled, using keyword router")
            self.mode = "keyword"
            return

        if not self.config.coordinator.model_path:
            logger.info("Coordinator model path not set, using keyword router")
            self.mode = "keyword"
            return

        try:
            from llama_cpp import Llama

            logger.info(f"Loading coordinator model from {self.config.coordinator.model_path}")
            self.llama = Llama(
                model_path=self.config.coordinator.model_path,
                n_ctx=min(self.config.server.context_window, 2048),
                n_threads=1,
                n_batch=256,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            self.mode = "model"
            logger.info("Coordinator model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load coordinator model: {e}, falling back to keyword router")
            self.mode = "keyword"
            self.llama = None

    def route(
        self,
        query: str,
        summary: str = "",
        recent_history: str = "",
    ) -> RoutingDecision:
        t0 = time.time()

        if self.mode == "keyword":
            decision = self.keyword_router.route(query, summary, recent_history)
            _log_routing("keyword", decision.domain, decision.confidence,
                         (time.time() - t0) * 1000, False)
            return decision

        decision = self._llama_route(query, summary, recent_history)
        used_fallback = False

        if decision.confidence < self.config.swap_threshold_confidence:
            logger.info(
                f"Coordinator confidence {decision.confidence:.2f} < threshold "
                f"{self.config.swap_threshold_confidence}, using keyword fallback"
            )
            decision = self.keyword_router.route(query, summary, recent_history)
            used_fallback = True

        _log_routing(
            "keyword" if used_fallback else "model",
            decision.domain,
            decision.confidence,
            (time.time() - t0) * 1000,
            used_fallback,
        )
        return decision

    def _llama_route(
        self,
        query: str,
        summary: str = "",
        recent_history: str = "",
    ) -> RoutingDecision:
        last_3_turns = recent_history or summary or "none"
        prompt = (
            "You are a routing coordinator. Analyze the query and context.\n"
            "Return ONLY raw JSON, no other text:\n"
            '{"domain":"code","confidence":0.92,"reasoning":"algorithm request","preload_hint":"math"}\n'
            "Domains: code, math, chat, summarization\n"
            "confidence: 0.0-1.0\n"
            "reasoning: max 8 words\n\n"
            f"Recent context: {last_3_turns}\n"
            f"Query: {query}"
        )

        for attempt in range(2):
            try:
                response = self.llama(
                    prompt,
                    max_tokens=self.config.coordinator.max_tokens,
                    temperature=self.config.coordinator.temperature,
                    stop=["\n\n", "</s>"],
                )
                output = response["choices"][0]["text"].strip()
                return self._parse_response(output, fallback_query=query)
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.warning("Coordinator JSON parse failed, retrying")
                    continue
                logger.warning("Coordinator JSON parse failed after retry, using keyword fallback")
                self.mode = "keyword"
                return self.keyword_router.route(query, summary, recent_history)
            except Exception as e:
                logger.error(f"Coordinator LLM error: {e}, falling back to keyword router")
                self.mode = "keyword"
                return self.keyword_router.route(query, summary, recent_history)

        self.mode = "keyword"
        return self.keyword_router.route(query, summary, recent_history)

    def _parse_response(self, output: str, fallback_query: str = "") -> RoutingDecision:
        output = self._extract_json(output)
        data = json.loads(output)
        domain = data.get("domain", self.config.fallback_domain)
        confidence = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", ""))
        preload_hint = data.get("preload_hint", domain)

        if domain not in self.config.domains:
            logger.warning(f"Invalid domain from coordinator: {domain}, using fallback")
            domain = self.config.fallback_domain

        confidence = max(0.0, min(1.0, confidence))

        return RoutingDecision(
            domain=domain,
            confidence=confidence,
            reasoning=reasoning,
            preload_hint=preload_hint,
        )

    @staticmethod
    def _extract_json(output: str) -> str:
        cleaned = re.sub(r"```[a-z]*\n?", "", output).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        return match.group(0) if match else cleaned
