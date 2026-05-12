declare const process: {
  argv: string[];
  stdout: { write: (message: string) => void };
  stderr: { write: (message: string) => void };
  exitCode?: number;
};

import { IntentPredictor } from "./intent-predictor.ts";
import type { TaskIntent } from "./intent-predictor.ts";
import { PreloadCache } from "./preload-cache.ts";
import { PreloadManager } from "./preload-manager.ts";
import type { PreloadTimingStrategy } from "./preload-manager.ts";
import { PreloadMetrics } from "./preload-metrics.ts";
import type { PreloadMetricReport } from "./preload-metrics.ts";

interface BenchmarkTask {
  id: string;
  text: string;
  actualIntent: TaskIntent;
}

interface BenchmarkOptions {
  writeLog?: boolean;
}

const TASKS: BenchmarkTask[] = [
  {
    id: "coding-1",
    actualIntent: "coding",
    text: "Implement the TypeScript cache and add tests for eviction behavior in the repo.",
  },
  {
    id: "reasoning-1",
    actualIntent: "reasoning",
    text: "Compare these two approaches and explain which tradeoff is better for latency.",
  },
  {
    id: "web-1",
    actualIntent: "web_research",
    text: "Search the web for the latest API pricing and include source links.",
  },
  {
    id: "chat-1",
    actualIntent: "chat",
    text: "Hi, can you help me word a quick reply?",
  },
  {
    id: "summary-1",
    actualIntent: "summarization",
    text: "Summarize this transcript into takeaways, decisions, and follow-up notes.",
  },
  {
    id: "coding-2",
    actualIntent: "coding",
    text: "Debug why the function fails to compile after the refactor and propose a patch.",
  },
  {
    id: "web-2",
    actualIntent: "web_research",
    text: "Look up current browser support and cite the sources from today if possible.",
  },
  {
    id: "reasoning-2",
    actualIntent: "reasoning",
    text: "Why would a confidence threshold reduce wasted preloads? Analyze failure cases.",
  },
  {
    id: "summary-2",
    actualIntent: "summarization",
    text: "Condense this long article into a brief outline and TL;DR.",
  },
  {
    id: "chat-2",
    actualIntent: "chat",
    text: "Thanks. Please make the tone warmer but still concise.",
  },
];

const STRATEGIES: PreloadTimingStrategy[] = ["aggressive", "delayed", "confidence"];

export async function runPreloadBenchmark(options: BenchmarkOptions = {}): Promise<PreloadMetricReport> {
  const metrics = new PreloadMetrics();

  for (const strategy of STRATEGIES) {
    const predictor = new IntentPredictor();
    const cache = new PreloadCache();
    const manager = new PreloadManager(predictor, cache, {
      strategy,
      confidenceThreshold: 0.66,
      delayMs: 25,
      ttlMs: 5 * 60 * 1000,
    });

    for (const task of TASKS) {
      const run = await manager.considerPreload(task.text);
      const predictionCorrect = run.prediction.intent === task.actualIntent;
      const coldStartReductionMs = estimateColdStartReduction(
        run.strategy,
        run.attempted,
        predictionCorrect,
        run.resourcesLoaded,
        run.cacheHits,
      );
      const responseLatencyMs = estimateResponseLatency(
        task.actualIntent,
        run.preloadDurationMs,
        coldStartReductionMs,
      );

      metrics.record({
        timestamp: new Date().toISOString(),
        strategy,
        taskId: task.id,
        actualIntent: task.actualIntent,
        predictedIntent: run.prediction.intent,
        confidence: run.prediction.confidence,
        predictionCorrect,
        preloadAttempted: run.attempted,
        preloadDurationMs: run.preloadDurationMs,
        ramUsageBytes: run.estimatedRamBytes,
        coldStartReductionMs,
        responseLatencyMs,
      });

      manager.recordOutcome(task.text, task.actualIntent, run.prediction);
    }
  }

  if (options.writeLog) {
    await metrics.writeJson();
  }

  return metrics.report();
}

function estimateColdStartReduction(
  strategy: PreloadTimingStrategy,
  attempted: boolean,
  predictionCorrect: boolean,
  resourcesLoaded: number,
  cacheHits: number,
): number {
  if (!attempted) {
    return 0;
  }
  const strategyFactor = strategy === "aggressive" ? 1.05 : strategy === "delayed" ? 0.82 : 0.96;
  const correctnessFactor = predictionCorrect ? 1 : -0.35;
  const resourceFactor = resourcesLoaded * 6 + cacheHits * 3;
  return round(resourceFactor * strategyFactor * correctnessFactor, 3);
}

function estimateResponseLatency(
  intent: TaskIntent,
  preloadDurationMs: number,
  coldStartReductionMs: number,
): number {
  const base: Record<TaskIntent, number> = {
    coding: 160,
    reasoning: 145,
    web_research: 190,
    chat: 90,
    summarization: 150,
  };
  return round(Math.max(35, base[intent] + preloadDurationMs * 0.25 - coldStartReductionMs), 3);
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runPreloadBenchmark({ writeLog: true })
    .then((report) => {
      process.stdout.write(`${JSON.stringify(report.summary, null, 2)}\n`);
    })
    .catch((error) => {
      process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
      process.exitCode = 1;
    });
}
