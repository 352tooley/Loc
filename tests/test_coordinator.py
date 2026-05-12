import pytest

from config import load_config
from coordinator import Coordinator, KeywordRouter


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def keyword_router(config):
    return KeywordRouter(config)


@pytest.fixture
def coordinator(config):
    return Coordinator(config)


class TestKeywordRouter:
    def test_code_routing(self, keyword_router):
        decision = keyword_router.route("Write a Python function")
        assert decision.domain == "code"
        assert decision.confidence > 0.3

    def test_math_routing(self, keyword_router):
        decision = keyword_router.route("Solve this equation: x² + 2x + 1 = 0")
        assert decision.domain == "math"
        assert decision.confidence > 0.3

    def test_chat_routing(self, keyword_router):
        decision = keyword_router.route("What do you think about this topic?")
        assert decision.domain == "chat"
        assert decision.confidence > 0.3

    def test_summarization_routing(self, keyword_router):
        decision = keyword_router.route("Summarize this document for me")
        assert decision.domain == "summarization"
        assert decision.confidence > 0.3

    def test_code_multiple_keywords(self, keyword_router):
        decision = keyword_router.route("Debug my Python code and implement this algorithm")
        assert decision.domain == "code"
        assert decision.confidence >= 0.4

    def test_fallback_on_no_match(self, keyword_router):
        decision = keyword_router.route("xyzabc qwerty")
        assert decision.domain == "chat"

    def test_routing_with_context(self, keyword_router):
        decision = keyword_router.route(
            "Follow-up question",
            summary="Previous conversation about coding",
            recent_history="User asked: implement a function",
        )
        assert decision.domain == "code"

    def test_decision_structure(self, keyword_router):
        decision = keyword_router.route("hello world")
        assert hasattr(decision, "domain")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "preload_hint")
        assert 0.0 <= decision.confidence <= 1.0

    def test_all_domains_detectable(self, keyword_router):
        test_queries = {
            "code": "implement a binary search in python",
            "math": "solve for x in this equation",
            "chat": "what do you think about that",
            "summarization": "summarize this article",
        }

        for expected_domain, query in test_queries.items():
            decision = keyword_router.route(query)
            assert decision.domain == expected_domain, f"Failed for {expected_domain}: got {decision.domain}"


class TestCoordinator:
    def test_coordinator_init(self, coordinator):
        assert coordinator.mode in ["keyword", "llama"]
        assert coordinator.keyword_router is not None

    def test_coordinator_fallback_mode(self, coordinator):
        decision = coordinator.route("What is 2+2?")
        assert decision.domain in ["code", "math", "chat", "summarization"]
        assert 0.0 <= decision.confidence <= 1.0

    def test_coordinator_with_context(self, coordinator):
        decision = coordinator.route(
            "Write a function",
            summary="We're discussing algorithms",
            recent_history="User: implement quicksort",
        )
        assert decision.domain in ["code", "math", "chat", "summarization"]

    def test_multiple_routes_consistent(self, coordinator):
        query = "Debug this code"
        decisions = [coordinator.route(query) for _ in range(3)]
        domains = [d.domain for d in decisions]
        assert all(d == domains[0] for d in domains), "Keyword router should be deterministic"
