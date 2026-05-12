import logging
from pathlib import Path

import psutil
from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    loaded_domain: str | None
    loaded_model: str | None
    ram_used_gb: float
    ram_budget_gb: float
    ram_available_gb: float
    coordinator_mode: str
    total_requests: int
    context_turns: int


class SwapRequest(BaseModel):
    domain: str


async def health_handler(
    config,
    loader,
    coordinator,
    context_manager,
    request_count: int,
) -> HealthResponse:
    """Handle /health request."""
    loaded_domain = loader.current_domain
    loaded_model = None

    if loaded_domain:
        domain_config = config.domains.get(loaded_domain)
        if domain_config and domain_config.model_path:
            loaded_model = Path(domain_config.model_path).name

    process = psutil.Process()
    ram_available = psutil.virtual_memory().available / (1024**3)

    return HealthResponse(
        status="ok",
        loaded_domain=loaded_domain,
        loaded_model=loaded_model,
        ram_used_gb=loader.get_ram_usage(),
        ram_budget_gb=config.server.ram_budget_gb,
        ram_available_gb=ram_available,
        coordinator_mode=coordinator.mode,
        total_requests=request_count,
        context_turns=len(context_manager.history),
    )


async def swap_handler(
    domain: str,
    config,
    loader,
    router,
) -> dict:
    """Handle /v1/admin/swap request."""
    if domain not in config.domains:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")

    try:
        model = loader.load(domain)
        router.set_last_domain(domain)
        logger.info(f"Manually swapped to domain: {domain}")
        return {"status": "ok", "domain": domain}
    except Exception as e:
        logger.error(f"Failed to swap to domain {domain}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load domain: {e}")
