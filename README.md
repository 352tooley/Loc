# SmartPack Phase 1

A local LLM inference server that routes queries to specialist models based on domain, loads only one model into RAM at a time, and exposes an OpenAI-compatible API.

## Features

- **Domain-based Routing**: Automatically routes queries to code, math, chat, or summarization specialists
- **Memory Efficient**: Loads only one model at a time with configurable RAM budget
- **OpenAI Compatible**: Drop-in replacement API for chat completions
- **Conversation Context**: Maintains conversation history with automatic summarization
- **Model Coordination**: Uses a small coordinator model (or keyword fallback) for intelligent routing

## Requirements

- Python 3.10+
- 6+ GB RAM (configurable)
- ~1-2 GB per specialist model

## Quick Start

### 1. Setup

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Configure Models

Edit `config.json` and add your GGUF model paths:

```json
{
  "coordinator": {
    "model_path": "/path/to/coordinator-model.gguf"
  },
  "domains": {
    "code": {
      "model_path": "/path/to/code-specialist.gguf",
      "size_gb": 1.5
    },
    "math": {
      "model_path": "/path/to/math-specialist.gguf",
      "size_gb": 1.2
    },
    "chat": {
      "model_path": "/path/to/chat-specialist.gguf",
      "size_gb": 1.0
    },
    "summarization": {
      "model_path": "/path/to/summarization-specialist.gguf",
      "size_gb": 0.9
    }
  }
}
```

If a domain model path is empty, SmartPack will use the chat domain model as fallback.

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Start Server

```bash
# Run with mock models (for testing)
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8080
```

## API Endpoints

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smartpack",
    "messages": [
      {"role": "user", "content": "Write a binary search in Python"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": false
  }'
```

Response includes custom headers:
- `X-SmartPack-Domain`: Selected domain (code, math, chat, summarization)
- `X-SmartPack-Model`: Selected model file

### Streaming

Same endpoint with `"stream": true` returns Server-Sent Events (SSE) format.

### List Models

```bash
curl http://localhost:8080/v1/models
```

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "loaded_domain": "code",
  "loaded_model": "qwen2.5-coder-1.5b-q4.gguf",
  "ram_used_gb": 2.1,
  "ram_budget_gb": 6.0,
  "ram_available_gb": 3.9,
  "coordinator_mode": "keyword",
  "total_requests": 42,
  "context_turns": 8
}
```

### Force Domain Swap

```bash
curl -X POST http://localhost:8080/v1/admin/swap \
  -H "Content-Type: application/json" \
  -d '{"domain": "math"}'
```

## Configuration

### config.json

- **server.ram_budget_gb**: Maximum RAM for all models (default: 6.0)
- **server.context_window**: Context size for models (default: 4096)
- **coordinator.model_path**: Path to coordinator model (empty = keyword router)
- **coordinator.fallback_mode**: "keyword" for fallback behavior
- **swap_threshold_confidence**: Minimum confidence to swap domains (default: 0.6)
- **min_turns_before_swap**: Minimum turns before allowing domain swap (default: 2)

### Environment Variables

- **LOG_LEVEL**: Set logging level (default: INFO)

## How SmartPack Works

### Coordinator

On each turn, the coordinator receives:
- Current query
- Conversation summary
- Last 3 conversation turns

It returns a routing decision with:
- Selected domain (code, math, chat, summarization)
- Confidence score (0.0-1.0)
- Reasoning
- Preload hint for next likely domain

**Fallback**: If coordinator model fails or is empty, keyword router takes over.

### Keyword Router

Scores each domain by counting keyword matches:
- Code: "function", "debug", "implement", "python", etc.
- Math: "solve", "equation", "integral", "proof", etc.
- Chat: "feeling", "opinion", "advice", "what do you", etc.
- Summarization: "summarize", "summary", "tldr", etc.

### Model Loading

1. Router selects domain
2. Loader checks RAM budget
3. Unloads previous model (explicit del + gc.collect)
4. Loads new model via llama-cpp-python
5. Logs timing and RAM usage

### Context Management

- Full conversation history stored (one turn = {role, content, domain, timestamp})
- After 6 turns, automatically compress history:
  - Generate summary via current model
  - Keep last 2 turns + summary
  - Inject into new model's context on swap
- Context persists to `context.json` on disk

## Testing Without Models

SmartPack uses mock models by default for testing. Run tests immediately without model files:

```bash
pytest tests/ -v
```

To test with real models, update `loader.py` to use `use_mock=False` or set models in config.json.

## Adding a New Specialist Domain

1. Add domain to `config.json` domains section
2. Provide keywords list
3. Add model path
4. Coordinator will automatically route to it

Example:
```json
{
  "domains": {
    "music": {
      "model_path": "/path/to/music-model.gguf",
      "model_name": "Music Specialist",
      "keywords": ["compose", "chord", "melody", "beat", "rhythm"],
      "size_gb": 1.0
    }
  }
}
```

## Architecture

- **main.py**: FastAPI server and route handlers
- **config.py**: Configuration loading with Pydantic validation
- **coordinator.py**: Routing decision logic + keyword router
- **router.py**: Domain selection with swap throttling
- **loader.py**: Model lifecycle management
- **context_manager.py**: Conversation history + summarization
- **api/**: API endpoint handlers (chat, models, admin)
- **tests/**: Comprehensive test suite

## Troubleshooting

### "Model file not found"
Ensure model_path in config.json is absolute path to .gguf file.

### "Out of memory"
Reduce `ram_budget_gb` or use smaller quantized models.

### "Coordinator returns invalid JSON"
SmartPack automatically falls back to keyword router. Check logs: `LOG_LEVEL=DEBUG python main.py`

### Server crashes on startup
Check config.json syntax and verify all required fields exist.

## Performance Tips

- Use Q4 or Q5 quantized models for best speed/quality
- Set context_window to model's actual capacity
- Preload coordinator to stay in "llama" mode
- Use mmap when available (llama-cpp-python handles this)

## License

SmartPack Phase 1 - Inference routing server
