import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_METRICS_FILE = Path("logs/quality_metrics.jsonl")
_THRESHOLD = 0.55
_ALWAYS_ESCALATE = {"medical", "legal", "financial", "security"}

_UNCERTAINTY_PHRASES = [
    "i'm not sure", "i don't know", "i cannot", "i can't", "unclear",
    "uncertain", "not certain", "may be", "might be", "possibly",
    "perhaps", "it depends", "i'm unable",
]

_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into",
    "your", "about", "please", "should",
}


def _estimate_tokens(text: str) -> int:
    words = len(text.strip().split()) if text.strip() else 0
    chars = len(text) // 4
    return max(words, chars)


def _meaningful_terms(text: str) -> list[str]:
    return [
        w for w in text.lower().replace(r"[^a-z0-9\s-]", "").split()
        if len(w) > 3 and w not in _STOP_WORDS
    ][:18]


def score_response(prompt: str, response: str, task_type: str = "") -> float:
    resp_lower = response.lower()

    uncertainty = sum(1 for p in _UNCERTAINTY_PHRASES if p in resp_lower)
    certainty = max(0.0, 1.0 - uncertainty * 0.18)

    tokens = _estimate_tokens(response)
    if tokens < 10:
        length_score = 0.2
    elif tokens < 30:
        length_score = 0.5
    elif tokens < 300:
        length_score = 0.85
    else:
        length_score = 0.75

    prompt_terms = _meaningful_terms(prompt)
    if prompt_terms:
        overlap = sum(1 for t in prompt_terms if t in resp_lower)
        overlap_score = overlap / len(prompt_terms)
    else:
        overlap_score = 0.6

    complexity_penalty = 0.08 if task_type in _ALWAYS_ESCALATE else 0.0
    task_fit = max(0.0, min(1.0, 0.35 + overlap_score * 0.65 - complexity_penalty))

    quality = min(1.0, (certainty * 0.5 + length_score * 0.25 + task_fit * 0.25))

    score = round(
        certainty * 0.22 + length_score * 0.16 + task_fit * 0.18 + quality * 0.44,
        4,
    )
    return max(0.0, min(1.0, score))


def should_escalate(quality_score: float, task_type: str = "") -> bool:
    if task_type in _ALWAYS_ESCALATE:
        return True
    return quality_score < _THRESHOLD


def log_escalation(task_type: str, tier1_quality: float, escalated: bool) -> None:
    try:
        _METRICS_FILE.parent.mkdir(exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "escalation",
            "task_type": task_type,
            "tier1_quality": round(tier1_quality, 4),
            "escalated": escalated,
        }
        with open(_METRICS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log quality metrics: {e}")


def detect_task_type(messages: list) -> str:
    text = " ".join(
        m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        for m in messages
    ).lower()

    if any(w in text for w in ["medical", "medicine", "medication", "symptom", "diagnosis", "drug", "dose", "illness", "disease", "chest pain", "shortness of breath", "headache", "pain", "treatment", "doctor"]):
        return "medical"
    if any(w in text for w in ["legal", "law", "contract", "attorney", "lawsuit", "court", "regulation"]):
        return "legal"
    if any(w in text for w in ["financial", "investment", "stock", "tax", "portfolio", "crypto", "money"]):
        return "financial"
    if any(w in text for w in ["security", "vulnerability", "exploit", "attack", "cve", "penetration"]):
        return "security"
    return ""
