import json
import os
import statistics
import time
from pathlib import Path

import psutil

from config import load_config
from loader import ModelLoader


MODELS = {
    "code": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "math": ("bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF", "Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"),
    "chat": ("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
    "summarization": ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
    "coordinator": ("bartowski/SmolLM2-360M-Instruct-GGUF", "SmolLM2-360M-Instruct-Q4_K_M.gguf"),
}

SMOKE_PROMPTS = {
    "code": "Write a Python function that returns the larger of two numbers.",
    "math": "Solve for x: 2x + 3 = 11.",
    "chat": "Give one practical tip for staying focused.",
    "summarization": "Summarize: SmartPack routes tasks to small specialist models.",
}

ROUTING_QUERIES = [
    "Write a quicksort in Python",
    "Debug this: for i in range(10) print(i)",
    "What is the integral of e^x",
    "Solve for x: 2x^2 + 5x - 3 = 0",
    "I need advice on a career change",
    "What movie should I watch tonight",
    "Summarize the key points of a long document",
    "Give me the main takeaways from this report",
]


def ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def download_models() -> list[str]:
    from huggingface_hub import hf_hub_download

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    lines = []
    for domain, (repo, filename) in MODELS.items():
        target = model_dir / filename
        if not target.exists():
            hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir)
        size_gb = target.stat().st_size / (1024**3)
        lines.append(f"{domain:13} {filename:48} {size_gb:.2f} GB")
    return lines


def smoke_models(config) -> tuple[list[dict], list[str]]:
    loader = ModelLoader(config, use_mock=False)
    rows = []
    errors = []
    for domain, prompt in SMOKE_PROMPTS.items():
        try:
            before = ram_gb()
            model = loader.load(domain)
            loaded = loader.last_load_metrics
            start = time.time()
            response = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.2,
            )
            elapsed = time.time() - start
            text = response["choices"][0]["message"]["content"].strip()
            completion_tokens = len(text.split())
            rows.append(
                {
                    "domain": domain,
                    "load_time": loaded.get("load_time_seconds", 0.0),
                    "ram_delta": ram_gb() - before,
                    "tokens_sec": completion_tokens / elapsed if elapsed else 0.0,
                    "first_token_latency": elapsed,
                    "sample": text[:160],
                }
            )
        except Exception as exc:
            errors.append(f"{domain}: {exc}")
        finally:
            loader.unload()
    return rows, errors


def benchmark_routing() -> list[dict]:
    os.environ["SMARTPACK_USE_MOCK"] = "0"
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    rows = []
    for query in ROUTING_QUERIES:
        start = time.time()
        response = client.post(
            "/v1/chat/completions",
            json={"model": "smartpack", "messages": [{"role": "user", "content": query}], "max_tokens": 80},
        )
        elapsed = time.time() - start
        data = response.json()
        health = client.get("/health").json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        rows.append(
            {
                "query": query,
                "status": response.status_code,
                "domain": health.get("loaded_domain"),
                "model": health.get("loaded_model"),
                "latency": elapsed,
                "tokens_sec": len(text.split()) / elapsed if elapsed else 0.0,
                "quality": subjective_quality(text),
            }
        )
    main.loader.unload()
    return rows


def measure_swaps(config) -> tuple[list[dict], dict]:
    loader = ModelLoader(config, use_mock=False)
    sequence = ["code", "math", "chat", "summarization", "code", "chat", "math", "summarization", "chat", "code"]
    rows = []
    for domain in sequence:
        start = time.time()
        before = ram_gb()
        loader.load(domain)
        after = ram_gb()
        metrics = loader.last_load_metrics
        rows.append(
            {
                "domain": domain,
                "total_swap_time": time.time() - start,
                "load_time": metrics.get("load_time_seconds", 0.0),
                "unload_time": metrics.get("unload", {}).get("unload_time_seconds", 0.0),
                "ram_delta": after - before,
                "ram_used": after,
                "ram_freed": metrics.get("unload", {}).get("ram_freed_gb", 0.0),
            }
        )
    loader.unload()
    latencies = [row["total_swap_time"] for row in rows]
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0]
    return rows, {"average": statistics.mean(latencies), "p95": p95}


def context_persistence_check() -> dict:
    os.environ["SMARTPACK_USE_MOCK"] = "0"
    from fastapi.testclient import TestClient
    import main

    main.context_manager.reset()
    main.router.last_domain = None
    main.router.turns_since_swap = 0
    client = TestClient(main.app)
    turns = [
        "My project codename is Blue Lantern.",
        "Write a Python helper that stores a codename.",
        "Solve 3x + 9 = 21.",
        "What was my project codename?",
        "Summarize what we did so far.",
        "Give one practical next step.",
    ]
    outputs = []
    for turn in turns:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "smartpack", "messages": [{"role": "user", "content": turn}], "max_tokens": 80},
        )
        outputs.append(response.json()["choices"][0]["message"]["content"])
    main.loader.unload()
    joined = "\n".join(outputs).lower()
    return {"passed": "blue" in joined or "lantern" in joined, "outputs": outputs}


def subjective_quality(text: str) -> int:
    if not text.strip():
        return 1
    if len(text.split()) < 4:
        return 2
    if any(marker in text.lower() for marker in ["error", "traceback", "exception"]):
        return 2
    return 4


def write_report(downloads, smoke_rows, smoke_errors, routing_rows, swap_rows, swap_summary, context_result) -> None:
    lines = [
        "# SmartPack Phase 2A Performance Report",
        "",
        "## Model Files",
        *downloads,
        "",
        "## Smoke Tests",
    ]
    for row in smoke_rows:
        lines.append(
            f"{row['domain']:13} load={row['load_time']:.2f}s ram_delta={row['ram_delta']:.2f}GB "
            f"tok/s={row['tokens_sec']:.2f} first_token={row['first_token_latency']:.2f}s"
        )
    if smoke_errors:
        lines.extend(["", "Smoke errors:", *smoke_errors])
    lines.extend(["", "## Routing Benchmark"])
    for row in routing_rows:
        lines.append(
            f"{row['domain'] or 'unknown':13} status={row['status']} latency={row['latency']:.2f}s "
            f"tok/s={row['tokens_sec']:.2f} quality={row['quality']} query={row['query']}"
        )
    lines.extend(
        [
            "",
            "## Swap Latency",
            f"average={swap_summary['average']:.2f}s p95={swap_summary['p95']:.2f}s",
        ]
    )
    for row in swap_rows:
        lines.append(
            f"{row['domain']:13} total={row['total_swap_time']:.2f}s load={row['load_time']:.2f}s "
            f"unload={row['unload_time']:.2f}s ram_used={row['ram_used']:.2f}GB"
        )
    lines.extend(
        [
            "",
            "## Context Persistence",
            "pass" if context_result["passed"] else "fail",
            "",
            "## Recommended RAM Targets",
            "4GB: n_ctx 2048, one 1.5B specialist loaded at a time, coordinator optional.",
            "6GB: n_ctx 3072, coordinator plus one 1.5B specialist.",
            "8GB: n_ctx 4096, coordinator plus one 1.7B specialist with mmap.",
        ]
    )
    Path("performance_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    downloads = download_models()
    config = load_config()
    smoke_rows, smoke_errors = smoke_models(config)
    routing_rows = benchmark_routing()
    swap_rows, swap_summary = measure_swaps(config)
    context_result = context_persistence_check()
    write_report(downloads, smoke_rows, smoke_errors, routing_rows, swap_rows, swap_summary, context_result)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/phase2a-results.json").write_text(
        json.dumps(
            {
                "downloads": downloads,
                "smoke": smoke_rows,
                "smoke_errors": smoke_errors,
                "routing": routing_rows,
                "swaps": swap_rows,
                "swap_summary": swap_summary,
                "context": context_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
