import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(self, context_file: str = "context.json"):
        self.context_file = Path(context_file)
        self.history: list[dict] = []
        self.summary: str = ""
        self.load()

    def add_turn(self, role: str, content: str, domain: str = "unknown") -> None:
        """Add a turn to conversation history."""
        turn = {
            "role": role,
            "content": content,
            "domain": domain,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.history.append(turn)
        logger.debug(f"Added turn {len(self.history)}: {role} in {domain}")
        self.save()

    def get_recent(self, n: int = 3) -> list[dict]:
        """Get last n turns from history."""
        return self.history[-n:]

    def get_summary(self) -> str:
        """Get current conversation summary."""
        return self.summary

    def set_summary(self, summary: str) -> None:
        """Update conversation summary."""
        self.summary = summary
        logger.debug("Updated conversation summary")
        self.save()

    def get_context_for_model(self) -> tuple[str, list[dict]]:
        """Get summary and recent turns for model context."""
        recent = self.get_recent(3)
        summary = self.summary if self.summary else "Conversation just started."
        return summary, recent

    def should_compress(self) -> bool:
        """Check if history should be compressed (>6 turns)."""
        return len(self.history) >= 6

    def compress(self, new_summary: str) -> None:
        """Compress history, keeping only last 2 turns and updating summary."""
        logger.info(f"Compressing context: {len(self.history)} turns → last 2 + summary")
        self.history = self.history[-2:]
        self.summary = new_summary
        self.save()

    def save(self) -> None:
        """Persist context to disk."""
        try:
            data = {
                "summary": self.summary,
                "history": self.history,
            }
            with open(self.context_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved context to {self.context_file}")
        except Exception as e:
            logger.error(f"Failed to save context: {e}")

    def load(self) -> None:
        """Load context from disk."""
        if not self.context_file.exists():
            logger.debug(f"No context file found at {self.context_file}")
            return

        try:
            with open(self.context_file) as f:
                data = json.load(f)
            self.summary = data.get("summary", "")
            self.history = data.get("history", [])
            logger.info(f"Loaded context with {len(self.history)} turns and summary")
        except json.JSONDecodeError as e:
            logger.warning(f"Context file corrupted, resetting: {e}")
            self.history = []
            self.summary = ""
        except Exception as e:
            logger.error(f"Failed to load context: {e}")

    def reset(self) -> None:
        """Reset context completely."""
        logger.info("Resetting context")
        self.history = []
        self.summary = ""
        self.save()

    def format_history_for_prompt(self) -> str:
        """Format history for inclusion in model prompt."""
        if not self.history:
            return ""

        lines = []
        for turn in self.history[-3:]:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")[:100]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)
