import { nowMs, round } from "./compression-engine.ts";
import { RetrievalEngine } from "./retrieval-engine.ts";
import type { CompressionResult } from "./compression-engine.ts";
import type { RetrievalMode, RetrievalQuery } from "./retrieval-engine.ts";

export interface LatencySummary {
  mode: RetrievalMode;
  beforeMs: number;
  afterMs: number;
  deltaMs: number;
}

export interface AccuracySummary {
  mode: RetrievalMode;
  before: number;
  after: number;
  delta: number;
}

export interface MemoryAgentMetrics {
  ramUsageMb: number;
  contextTokensBefore: number;
  contextTokensAfter: number;
  tokenReductionRatio: number;
  compressionTimeMs: number;
  retrievalAccuracy: AccuracySummary[];
  retrievalLatency: LatencySummary[];
}

export class MemoryMetrics {
  static ramUsageMb(): number {
    if (typeof process === "undefined" || !process.memoryUsage) {
      return 0;
    }
    return round(process.memoryUsage().rss / 1024 / 1024);
  }

  static retrievalLatency(
    engine: RetrievalEngine,
    queries: RetrievalQuery[],
    mode: RetrievalMode,
    rounds = 20,
  ): number {
    const started = nowMs();
    for (let roundIndex = 0; roundIndex < rounds; roundIndex += 1) {
      for (const query of queries) {
        engine.search(query.query, mode, 5);
      }
    }
    return round((nowMs() - started) / Math.max(1, rounds * queries.length));
  }

  static buildReport(params: {
    beforeEngine: RetrievalEngine;
    afterEngine: RetrievalEngine;
    queries: RetrievalQuery[];
    compressions: CompressionResult[];
    modes: RetrievalMode[];
  }): MemoryAgentMetrics {
    const contextTokensBefore = params.beforeEngine.contextTokens();
    const contextTokensAfter = params.afterEngine.contextTokens();
    const compressionTimeMs = params.compressions.reduce((sum, result) => sum + result.elapsedMs, 0);

    return {
      ramUsageMb: this.ramUsageMb(),
      contextTokensBefore,
      contextTokensAfter,
      tokenReductionRatio: round(
        contextTokensBefore === 0 ? 0 : 1 - contextTokensAfter / contextTokensBefore,
      ),
      compressionTimeMs: round(compressionTimeMs),
      retrievalAccuracy: params.modes.map((mode) => {
        const before = params.beforeEngine.evaluate(params.queries, mode);
        const after = params.afterEngine.evaluate(params.queries, mode);
        return { mode, before: round(before), after: round(after), delta: round(after - before) };
      }),
      retrievalLatency: params.modes.map((mode) => {
        const beforeMs = this.retrievalLatency(params.beforeEngine, params.queries, mode);
        const afterMs = this.retrievalLatency(params.afterEngine, params.queries, mode);
        return { mode, beforeMs, afterMs, deltaMs: round(afterMs - beforeMs) };
      }),
    };
  }
}
