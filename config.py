import json
import logging
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    ram_budget_gb: float = 6.0
    context_window: int = 4096


class CoordinatorConfig(BaseModel):
    model_path: str = ""
    fallback_mode: str = "keyword"
    max_tokens: int = 100
    temperature: float = 0.1


class PreloadConfig(BaseModel):
    enabled: bool = True
    strategy: str = "aggressive"
    buffer_mb: int = 50


class MemoryConfig(BaseModel):
    compression_strategy: str = "bullets"
    max_tokens: int = 60
    trigger_turns: int = 6


class QualityConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.55
    always_escalate_types: list[str] = ["medical", "legal", "financial", "security"]


class DomainConfig(BaseModel):
    model_path: str = ""
    model_name: str
    keywords: list[str]
    size_gb: float = 0.0
    escalation_model_path: str = ""


class SmartPackConfig(BaseModel):
    server: ServerConfig
    coordinator: CoordinatorConfig
    domains: Dict[str, DomainConfig]
    fallback_domain: str = "chat"
    swap_threshold_confidence: float = 0.6
    min_turns_before_swap: int = 2
    preload: PreloadConfig = PreloadConfig()
    memory: MemoryConfig = MemoryConfig()
    quality: QualityConfig = QualityConfig()

    @validator("domains")
    def validate_domains(cls, domains: Dict[str, DomainConfig]) -> Dict[str, DomainConfig]:
        if "code" not in domains or "math" not in domains or "chat" not in domains or "summarization" not in domains:
            raise ValueError("Must have code, math, chat, and summarization domains")
        return domains

    @validator("fallback_domain")
    def validate_fallback_domain(cls, fallback_domain: str, values: dict) -> str:
        if "domains" in values and fallback_domain not in values["domains"]:
            raise ValueError(f"Fallback domain must be one of: {list(values['domains'].keys())}")
        return fallback_domain


def load_config(config_path: Optional[str] = None) -> SmartPackConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, creating default")
        create_default_config(config_path)

    try:
        with open(config_path) as f:
            config_data = json.load(f)
        return SmartPackConfig(**config_data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def create_default_config(config_path: Path) -> None:
    default = {
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "ram_budget_gb": 6.0,
            "context_window": 4096,
        },
        "coordinator": {
            "model_path": "",
            "fallback_mode": "keyword",
            "max_tokens": 100,
            "temperature": 0.1,
        },
        "domains": {
            "code": {
                "model_path": "",
                "model_name": "Code Specialist",
                "keywords": ["function", "code", "debug", "implement", "class", "error", "python", "javascript", "api", "sql", "bug"],
                "size_gb": 0.0,
            },
            "math": {
                "model_path": "",
                "model_name": "Math Specialist",
                "keywords": ["solve", "calculate", "equation", "integral", "derivative", "proof", "matrix"],
                "size_gb": 0.0,
            },
            "chat": {
                "model_path": "",
                "model_name": "Chat Specialist",
                "keywords": ["feeling", "think", "opinion", "recommend", "advice", "help"],
                "size_gb": 0.0,
            },
            "summarization": {
                "model_path": "",
                "model_name": "Summarization Specialist",
                "keywords": ["summarize", "summary", "condense", "key points", "tldr"],
                "size_gb": 0.0,
            },
        },
        "fallback_domain": "chat",
        "swap_threshold_confidence": 0.6,
        "min_turns_before_swap": 2,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(default, f, indent=2)
    logger.info(f"Created default config at {config_path}")
