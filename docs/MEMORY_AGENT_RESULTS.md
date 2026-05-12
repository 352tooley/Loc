# Phase 2 Memory Agent Experimental Results

This experiment validates a self-contained memory-agent prototype without adding production infrastructure. The code lives under `experiments/memory/` and uses deterministic token estimates, simple term vectors, and keyword scoring with no external runtime dependencies.

## Implemented Scope

- Rolling window with `activeContext` and compressed `archivedContext`.
- Compression strategies:
  - `naive`: first-sentence rolling summary up to a token budget.
  - `semantic`: sentence ranking against a term-vector centroid with a small recency boost.
  - `bullets`: action-oriented bullet extraction for decisions, risks, bugs, fixes, todos, and metrics.
- Retrieval modes:
  - `keyword`: normalized query-term frequency.
  - `semantic`: cosine similarity over deterministic bag-of-terms vectors.
  - `hybrid`: weighted keyword and semantic score.
- Metrics:
  - RAM usage.
  - Context size before and after compression.
  - Compression time.
  - Retrieval accuracy.
  - Retrieval latency before and after compression.

## How To Run

The replay harness writes the required result artifact and compares all three compression strategies:

```bash
cd /Users/ma/Loc
node --experimental-strip-types experiments/memory/replay-test.ts
```

If the runtime does not support TypeScript type stripping, run the same file with `ts-node` or compile with the local TypeScript toolchain and run the emitted JavaScript.

Expected output path:

```text
logs/memory-agent-results.json
```

## Methodology

The replay uses a deterministic synthetic SmartPack conversation dataset covering auth, billing, routing, inventory, UX, metrics, security, and support. It compares a full baseline context against a rolling active window plus compressed archives. Accuracy is measured by whether each query retrieves the expected original memory id directly or through an archive containing that source id.

## Notes

This is experimental validation only. It intentionally avoids persistence, background workers, embeddings services, vector databases, and production APIs. The prototype is suitable for comparing compression and retrieval behavior before deciding whether a production memory pipeline is warranted.
