import logging
import os
import sys
from typing import AsyncGenerator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from api.admin import HealthResponse, health_handler, swap_handler
from api.chat import ChatCompletionRequest, ChatCompletionResponse, chat_completion_handler, stream_chat_completion
from api.models import ModelsResponse, models_handler
from config import load_config
from context_manager import ContextManager
from coordinator import Coordinator
from loader import ModelLoader
from router import Router

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    config = load_config()
    logger.info("Config loaded successfully")
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    raise

def _should_use_mock_models() -> bool:
    value = os.getenv("SMARTPACK_USE_MOCK")
    if value is not None:
        return value.lower() in {"1", "true", "yes", "on"}
    return "pytest" in sys.modules


loader = ModelLoader(config, use_mock=_should_use_mock_models())
coordinator = Coordinator(config)
router = Router(config)
context_manager = ContextManager()

request_count = 0

app = FastAPI(
    title="SmartPack",
    description="Local LLM inference server with domain routing",
    version="0.1.0",
)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_smartpack_domain: str | None = Header(None),
):
    """OpenAI-compatible chat completions endpoint."""
    global request_count
    request_count += 1

    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request, router, loader, context_manager),
                media_type="text/event-stream",
            )
        else:
            response, escalated = await chat_completion_handler(request, router, loader, context_manager)

            model_file = "mock"
            if loader.current_domain and loader.current_domain in config.domains:
                domain_config = config.domains[loader.current_domain]
                if domain_config.model_path:
                    from pathlib import Path
                    model_file = Path(domain_config.model_path).name

            headers = {
                "X-SmartPack-Domain": loader.current_domain or "unknown",
                "X-SmartPack-Model": model_file,
                "X-SmartPack-Escalated": "true" if escalated else "false",
            }
            return JSONResponse(
                content=response.dict(),
                headers=headers,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    """List available models."""
    return await models_handler(config)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return await health_handler(config, loader, coordinator, context_manager, request_count)


@app.post("/v1/admin/swap")
async def swap_model(request: dict) -> dict:
    """Force swap to specific domain."""
    domain = request.get("domain")
    if not domain:
        raise HTTPException(status_code=400, detail="Domain required")

    return await swap_handler(domain, config, loader, router)


@app.on_event("startup")
async def startup() -> None:
    """Startup event."""
    logger.info("SmartPack server starting")
    logger.info(f"Config: {config.server}")
    logger.info(f"Domains: {list(config.domains.keys())}")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Shutdown event."""
    logger.info("SmartPack server shutting down")
    loader.unload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )
