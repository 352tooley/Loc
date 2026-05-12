import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_METRICS_FILE = Path("logs/memory_metrics.jsonl")
_MAX_TOKENS = 60
_MAX_BULLETS = 3
_MAX_WORDS_PER_BULLET = 15

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was", "with",
}

_ACTION_TERMS = {
    "blocker", "bug", "decision", "error", "failed", "fix",
    "metric", "next", "risk", "todo", "warning",
}


def _estimate_tokens(text: str) -> int:
    words = len(text.strip().split()) if text.strip() else 0
    chars = len(text) // 4
    return max(words, chars)


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9_\-\s]", " ", text.lower())
    return [t for t in cleaned.split() if len(t) > 1 and t not in _STOP_WORDS]


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [p.strip() for p in parts if p.strip()]


def _truncate_bullet_to_words(sentence: str, max_words: int = _MAX_WORDS_PER_BULLET) -> str:
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words])


def bullets_compress(turns: list[dict]) -> str:
    text = " ".join(t.get("content", "") for t in turns)
    sentences = _split_sentences(text)

    ranked = []
    for i, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        action_score = sum(1 for t in tokens if t in _ACTION_TERMS)
        keyword_score = len(set(tokens)) / 20.0
        ranked.append((action_score * 2 + keyword_score, i, sentence))

    ranked.sort(key=lambda x: (-x[0], x[1]))

    bullets: list[str] = []
    total_tokens = 0

    for _, _, sentence in ranked:
        if len(bullets) >= _MAX_BULLETS:
            break
        truncated = _truncate_bullet_to_words(sentence)
        line = f"- {truncated}"
        line_tokens = _estimate_tokens(line)
        if bullets and total_tokens + line_tokens > _MAX_TOKENS:
            continue
        bullets.append(line)
        total_tokens += line_tokens
        if total_tokens >= _MAX_TOKENS:
            break

    result = "\n".join(bullets)

    if _estimate_tokens(result) > _MAX_TOKENS:
        result = _truncate_to_token_limit(result, _MAX_TOKENS)

    return result


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    lines = text.split("\n")
    kept: list[str] = []
    total = 0
    for line in lines:
        t = _estimate_tokens(line)
        if kept and total + t > max_tokens:
            break
        kept.append(line)
        total += t
    return "\n".join(kept)


def naive_compress(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        content = t.get("content", "")
        parts.append(content[-200:] if len(content) > 200 else content)
    return " ".join(parts)


def compress_turns(turns: list[dict]) -> str:
    try:
        result = bullets_compress(turns)
        if not result.strip():
            raise ValueError("Empty compression result")
        return result
    except Exception as e:
        logger.warning(f"Bullets compression failed: {e}, using naive fallback")
        return naive_compress(turns)


def check_retrieval_accuracy(compressed: str, archived_turns: list[dict], n: int = 3) -> float:
    recent = archived_turns[-n:] if len(archived_turns) >= n else archived_turns
    if not recent:
        return 1.0

    hits = 0
    compressed_lower = compressed.lower()
    for turn in recent:
        content = turn.get("content", "")
        meaningful = [w for w in content.lower().split() if len(w) > 4 and w not in _STOP_WORDS]
        if not meaningful:
            hits += 1
            continue
        matched = sum(1 for w in meaningful[:5] if w in compressed_lower)
        if matched >= 1:
            hits += 1

    return hits / len(recent)


def log_compression(turns_count: int, tokens_before: int, tokens_after: int, accuracy: float) -> None:
    try:
        _METRICS_FILE.parent.mkdir(exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "compression",
            "turns_compressed": turns_count,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "accuracy": round(accuracy, 4),
        }
        with open(_METRICS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log memory metrics: {e}")
