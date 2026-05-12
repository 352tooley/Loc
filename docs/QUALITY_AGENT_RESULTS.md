# Quality Agent Phase 2 Results

This experiment validates a lightweight quality-agent loop without production routing or external dependencies. The implementation lives in `experiments/quality/` and uses deterministic heuristics so benchmark runs are reproducible.

## Implemented Files

- `confidence-engine.ts`: combines response length, uncertainty, contradiction, task-fit, and quality heuristics into a confidence score.
- `response-analyzer.ts`: measures word and sentence counts, scores response length, detects uncertainty phrases, and flags simple contradiction patterns.
- `escalation-router.ts`: supports `always-small-model`, `confidence-triggered`, `task-type`, and `retry-escalation` strategies.
- `quality-metrics.ts`: defines model tiers, cost and latency estimates, response quality scoring, and aggregate benchmark metrics.
- `benchmark-suite.ts`: runs a small deterministic benchmark across small, medium, and large model tiers and writes results to `logs/quality-agent-results.json` when executed.

## Validation Scope

The benchmark tracks:

- latency by model tier
- estimated token cost
- response quality score
- escalation frequency
- escalation count
- false escalations
- missed escalations
- quality improvement against the always-small baseline
- latency penalty against the always-small baseline

## Current Dataset

The experimental dataset includes four representative cases:

- fragile packing plan
- security policy review
- customer-support return summary
- medical safety response

The cases intentionally include weak small-model responses with uncertainty, short answers, or contradictions so the confidence and escalation paths can be exercised.

## Running

From the repo root, execute the TypeScript file with the project runner used by the active branch. When executed directly after TypeScript compilation, `benchmark-suite.ts` writes:

```text
logs/quality-agent-results.json
```

The log payload contains the benchmark timestamp, selected strategy, aggregate metrics, and per-case observations.

## Notes

This is experimental validation only. The heuristics are intentionally transparent and conservative, and they should not be treated as production policy without calibration against real model outputs and human labels.
