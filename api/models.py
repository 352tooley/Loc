import logging
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Model(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "smartpack"
    permission: list[dict] = []


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[Model]


async def models_handler(config) -> ModelsResponse:
    """Handle /v1/models request."""
    models = []

    for domain_name, domain_config in config.domains.items():
        model = Model(
            id=domain_name,
            owned_by="smartpack",
        )
        models.append(model)

    return ModelsResponse(data=models)
