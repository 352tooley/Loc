# Phase 2 Preload Agent Results

This is an experimental validation harness, not final orchestration.

## Scope

- Predicts five task intents: coding, reasoning, web research, chat, and summarization.
- Uses keyword analysis, task-shape signals, rolling trends, and recent task weighting.
- Preloads lightweight tokenizer state, embeddings, vector indexes, prompt templates, and small model metadata.
- Exercises aggressive, delayed, and confidence-based timing strategies.
- Writes benchmark output to `logs/preload-agent-results.json` when `preload-benchmark.ts` is run with log writing enabled.

## Files

- `experiments/preload/intent-predictor.ts`: standalone heuristic predictor with rolling and recency weighting.
- `experiments/preload/preload-cache.ts`: TTL cache with RAM estimates and least-recently-used eviction.
- `experiments/preload/preload-manager.ts`: preload policy runner for the three timing strategies.
- `experiments/preload/preload-metrics.ts`: metric event collector and JSON report writer.
- `experiments/preload/preload-benchmark.ts`: synthetic benchmark dataset and CLI entrypoint.

## Metrics Logged

The benchmark report includes:

- prediction accuracy
- preload duration
- RAM usage
- cold-start reduction
- average response latency
- per-strategy summaries for aggressive, delayed, and confidence-based timing

## How To Run

This experiment has no runtime package dependencies. In a TypeScript-capable environment, run:

```bash
ts-node experiments/preload/preload-benchmark.ts
```

The CLI writes:

```text
logs/preload-agent-results.json
```

If `ts-node` is unavailable, compile the files with the repo's TypeScript toolchain and execute the emitted `preload-benchmark.js` with Node.

## Validation Notes

The benchmark uses synthetic task examples and deterministic lightweight resources. The numbers are suitable for comparing preload strategies inside this experiment, but they should not be treated as production latency claims.
