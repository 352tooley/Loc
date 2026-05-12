import logging
import time
from typing import AsyncGenerator, Optional

from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict


class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]


async def chat_completion_handler(
    request: ChatCompletionRequest,
    router,
    loader,
    context_manager,
) -> ChatCompletionResponse:
    """Handle chat completion request."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1]
    query = last_message.content

    summary, recent = context_manager.get_context_for_model()
    recent_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    domain, decision = router.route(query, summary, recent_text)
    logger.info(f"Routed to domain: {domain}")

    try:
        model = loader.load(domain)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    prompt = _build_prompt(request.messages, context_manager)

    try:
        response = _run_model(
            model=model,
            prompt=prompt,
            messages=request.messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
        )
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=504, detail=f"Inference timeout or error: {e}")

    output_text = response["choices"][0]["text"].strip()

    context_manager.add_turn("user", query, domain)
    context_manager.add_turn("assistant", output_text, domain)

    if context_manager.should_compress():
        summary_prompt = f"Summarize this conversation in 1-2 sentences:\n{context_manager.format_history_for_prompt()}"
        try:
            summary_response = model(summary_prompt, max_tokens=150, temperature=0.1)
            new_summary = summary_response["choices"][0]["text"].strip()
            context_manager.compress(new_summary)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=domain,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=output_text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(output_text.split()),
            "total_tokens": len(prompt.split()) + len(output_text.split()),
        },
    )


async def stream_chat_completion(
    request: ChatCompletionRequest,
    router,
    loader,
    context_manager,
) -> AsyncGenerator[str, None]:
    """Stream chat completion response."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1]
    query = last_message.content

    summary, recent = context_manager.get_context_for_model()
    recent_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    domain, decision = router.route(query, summary, recent_text)
    logger.info(f"Streaming routed to domain: {domain}")

    try:
        model = loader.load(domain)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    prompt = _build_prompt(request.messages, context_manager)

    try:
        response = _run_model(
            model=model,
            prompt=prompt,
            messages=request.messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
        )
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=504, detail=f"Inference timeout or error: {e}")

    output_text = response["choices"][0]["text"].strip()

    import time
    import json

    chunk_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    for token in output_text.split():
        chunk_data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": domain,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token + " "},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": domain,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    context_manager.add_turn("user", query, domain)
    context_manager.add_turn("assistant", output_text, domain)

    if context_manager.should_compress():
        summary_prompt = f"Summarize this conversation in 1-2 sentences:\n{context_manager.format_history_for_prompt()}"
        try:
            summary_response = model(summary_prompt, max_tokens=150, temperature=0.1)
            new_summary = summary_response["choices"][0]["text"].strip()
            context_manager.compress(new_summary)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")


def _build_prompt(messages: list[Message], context_manager) -> str:
    """Build prompt from messages and context."""
    summary, recent = context_manager.get_context_for_model()

    prompt_parts = []

    if summary:
        prompt_parts.append(f"Summary: {summary}")

    prompt_parts.append("Recent conversation:")
    for msg in recent:
        prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")

    prompt_parts.append("Messages:")
    for msg in messages:
        prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")

    return "\n".join(prompt_parts)


def _run_model(
    model,
    prompt: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float,
) -> dict:
    """Run mock or real llama.cpp models and normalize to completion text."""
    if hasattr(model, "create_chat_completion"):
        chat_messages = _build_chat_messages(messages, prompt)
        start = time.time()
        response = model.create_chat_completion(
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        output_text = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(output_text.split()),
            "total_tokens": len(prompt.split()) + len(output_text.split()),
        }
        usage["latency_seconds"] = time.time() - start
        return {
            "choices": [{"text": output_text, "finish_reason": response["choices"][0].get("finish_reason", "stop")}],
            "usage": usage,
        }

    return model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _build_chat_messages(messages: list[Message], prompt: str) -> list[dict]:
    """Prefer model chat templates for GGUF instruct models."""
    system = (
        "You are SmartPack, a concise local assistant. Use the provided conversation "
        "context when relevant and answer the latest user message directly."
    )
    chat_messages = [{"role": "system", "content": system}]

    if len(messages) == 1:
        chat_messages.append({"role": "user", "content": prompt})
        return chat_messages

    for msg in messages:
        role = msg.role if msg.role in {"system", "user", "assistant"} else "user"
        chat_messages.append({"role": role, "content": msg.content})
    return chat_messages
