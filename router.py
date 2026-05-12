import logging
from typing import Optional

from config import SmartPackConfig
from coordinator import Coordinator, RoutingDecision

logger = logging.getLogger(__name__)


class Router:
    def __init__(self, config: SmartPackConfig):
        self.config = config
        self.coordinator = Coordinator(config)
        self.last_domain: Optional[str] = None
        self.turns_since_swap = 0

    def route(
        self,
        query: str,
        summary: str = "",
        recent_history: str = "",
    ) -> tuple[str, RoutingDecision]:
        """Route query to appropriate domain."""
        decision = self.coordinator.route(query, summary, recent_history)

        if self._should_respect_min_turns(decision.domain):
            logger.info(
                f"Respecting min_turns_before_swap: staying with {self.last_domain} "
                f"(turn {self.turns_since_swap}/{self.config.min_turns_before_swap})"
            )
            domain = self.last_domain
        else:
            domain = decision.domain

        if domain != self.last_domain:
            logger.info(
                f"Domain swap: {self.last_domain} → {domain} "
                f"(confidence: {decision.confidence:.2f}, reason: {decision.reasoning})"
            )
            self.last_domain = domain
            self.turns_since_swap = 0
        else:
            self.turns_since_swap += 1

        return domain, decision

    def _should_respect_min_turns(self, new_domain: str) -> bool:
        """Check if we should stick with current domain."""
        if self.last_domain is None:
            return False

        if new_domain == self.last_domain:
            return False

        if self.turns_since_swap < self.config.min_turns_before_swap:
            return True

        return False

    def set_last_domain(self, domain: str) -> None:
        """Manually set last domain (for initialization)."""
        self.last_domain = domain
        self.turns_since_swap = 0
