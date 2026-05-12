import { IntentPredictor } from "./intent-predictor.ts";
import type { IntentPrediction, TaskIntent } from "./intent-predictor.ts";
import { cacheKey, PreloadCache } from "./preload-cache.ts";
import type { PreloadResourceType } from "./preload-cache.ts";

export type PreloadTimingStrategy = "aggressive" | "delayed" | "confidence";

export interface PreloadManagerOptions {
  strategy: PreloadTimingStrategy;
  confidenceThreshold?: number;
  delayMs?: number;
  ttlMs?: number;
}

export interface PreloadRunResult {
  strategy: PreloadTimingStrategy;
  prediction: IntentPrediction;
  attempted: boolean;
  delayedMs: number;
  resourcesLoaded: number;
  cacheHits: number;
  preloadDurationMs: number;
  estimatedRamBytes: number;
  cacheKeys: string[];
}

const DEFAULT_OPTIONS: Required<PreloadManagerOptions> = {
  strategy: "confidence",
  confidenceThreshold: 0.68,
  delayMs: 35,
  ttlMs: 10 * 60 * 1000,
};

const RESOURCE_ORDER: PreloadResourceType[] = [
  "tokenizer_state",
  "prompt_template",
  "embeddings",
  "vector_index",
  "lightweight_model",
];

export class PreloadManager {
  private readonly predictor: IntentPredictor;
  private readonly cache: PreloadCache;
  private readonly options: PreloadManagerOptions;

  constructor(
    predictor = new IntentPredictor(),
    cache = new PreloadCache(),
    options: PreloadManagerOptions = DEFAULT_OPTIONS,
  ) {
    this.predictor = predictor;
    this.cache = cache;
    this.options = options;
  }

  async considerPreload(taskText: string): Promise<PreloadRunResult> {
    const options = { ...DEFAULT_OPTIONS, ...this.options };
    const prediction = this.predictor.predict(taskText);
    const shouldPreload =
      options.strategy === "aggressive" ||
      (options.strategy === "delayed" && prediction.confidence >= 0.45) ||
      (options.strategy === "confidence" && prediction.confidence >= options.confidenceThreshold);

    const startedAt = nowMs();
    let delayedMs = 0;
    let resourcesLoaded = 0;
    let cacheHits = 0;

    if (shouldPreload) {
      if (options.strategy === "delayed") {
        await sleep(options.delayMs);
        delayedMs = options.delayMs;
      }

      for (const type of RESOURCE_ORDER) {
        const key = cacheKey(prediction.intent, type);
        if (this.cache.hasFresh(key)) {
          cacheHits += 1;
          continue;
        }

        this.cache.set({
          key,
          intent: prediction.intent,
          type,
          payload: buildLightweightResource(prediction.intent, type),
          ttlMs: options.ttlMs,
        });
        resourcesLoaded += 1;
      }
    }

    const preloadDurationMs = nowMs() - startedAt;
    const snapshot = this.cache.snapshot();

    return {
      strategy: options.strategy,
      prediction,
      attempted: shouldPreload,
      delayedMs,
      resourcesLoaded,
      cacheHits,
      preloadDurationMs: round(preloadDurationMs, 3),
      estimatedRamBytes: snapshot.bytes,
      cacheKeys: this.cache.listKeys(),
    };
  }

  recordOutcome(taskText: string, actualIntent: TaskIntent, prediction: IntentPrediction): void {
    this.predictor.recordOutcome(taskText, actualIntent, prediction);
  }

  getCache(): PreloadCache {
    return this.cache;
  }
}

function buildLightweightResource(intent: TaskIntent, type: PreloadResourceType): unknown {
  switch (type) {
    case "tokenizer_state":
      return {
        intent,
        merges: [`${intent}:common`, `${intent}:rare`],
        vocabularyHints: intentTokens(intent),
      };
    case "embeddings":
      return syntheticEmbedding(intent, 64);
    case "vector_index":
      return intentTokens(intent).map((token, index) => ({
        token,
        vectorOffset: index * 64,
        norm: round(0.75 + index * 0.011, 3),
      }));
    case "prompt_template":
      return [
        `Intent: ${intent}`,
        "Use cached lightweight state only.",
        "Prefer short warm-start context over full orchestration.",
      ].join("\n");
    case "lightweight_model":
      return {
        intent,
        version: "experimental-linear-v1",
        weights: syntheticEmbedding(intent, 24),
      };
  }
}

function intentTokens(intent: TaskIntent): string[] {
  const tokens: Record<TaskIntent, string[]> = {
    coding: ["code", "test", "diff", "compile", "debug"],
    reasoning: ["reason", "compare", "why", "tradeoff", "evidence"],
    web_research: ["search", "source", "latest", "web", "citation"],
    chat: ["chat", "reply", "tone", "context", "short"],
    summarization: ["summary", "extract", "brief", "takeaway", "document"],
  };
  return tokens[intent];
}

function syntheticEmbedding(intent: TaskIntent, dimensions: number): Float32Array {
  const seed = intent.split("").reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const values = new Float32Array(dimensions);
  for (let index = 0; index < dimensions; index += 1) {
    values[index] = Math.sin(seed + index) * 0.5 + 0.5;
  }
  return values;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function nowMs(): number {
  const perf = (globalThis as { performance?: { now: () => number } }).performance;
  return perf ? perf.now() : Date.now();
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
