import json
import logging
import os
import re
from typing import Optional

from config import SmartPackConfig

logger = logging.getLogger(__name__)


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

        reasoning = f"matched {max_score} keywords"
        preload_hint = selected_domain

        return RoutingDecision(
            domain=selected_domain,
            confidence=confidence,
            reasoning=reasoning,
            preload_hint=preload_hint,
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
            self.mode = "llama"
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
        if self.mode == "keyword":
            return self.keyword_router.route(query, summary, recent_history)

        return self._llama_route(query, summary, recent_history)

    def _llama_route(
        self,
        query: str,
        summary: str = "",
        recent_history: str = "",
    ) -> RoutingDecision:
        prompt = f"""You are a routing coordinator for a local LLM server. Analyze the user query and conversation context, then return ONLY valid JSON. Do not add markdown, prose, or code fences.

Domains available: code, math, chat, summarization

Current conversation summary: {summary}
Last 3 turns: {recent_history}
Current query: {query}

Return exactly:
{{"domain":"code","confidence":0.92,"reasoning":"algorithm implementation request","preload_hint":"math"}}

Rules:
- domain must be one of: code, math, chat, summarization
- confidence must be 0.0 to 1.0
- preload_hint is your best guess at the NEXT likely domain (can be same)
- reasoning is max 10 words"""

        try:
            response = self.llama(
                prompt,
                max_tokens=self.config.coordinator.max_tokens,
                temperature=self.config.coordinator.temperature,
                stop=["\n\n", "</s>"],
            )
            output = response["choices"][0]["text"].strip()
            return self._parse_llama_response(output, fallback_query=query)
        except Exception as e:
            logger.error(f"Coordinator LLM error: {e}, falling back to keyword router")
            self.mode = "keyword"
            return self.keyword_router.route(query, summary, recent_history)

    def _parse_llama_response(self, output: str, fallback_query: str = "") -> RoutingDecision:
        try:
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
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse coordinator JSON: {e}, using keyword fallback")
            return self.keyword_router.route(fallback_query, "", "")

    @staticmethod
    def _extract_json(output: str) -> str:
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        return match.group(0) if match else output
