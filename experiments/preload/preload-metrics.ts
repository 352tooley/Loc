import fs from "node:fs/promises";
import path from "node:path";

declare const process: { cwd: () => string };

import type { PreloadTimingStrategy } from "./preload-manager.ts";
import type { TaskIntent } from "./intent-predictor.ts";

export interface PreloadMetricEvent {
  timestamp: string;
  strategy: PreloadTimingStrategy;
  taskId: string;
  actualIntent: TaskIntent;
  predictedIntent: TaskIntent;
  confidence: number;
  predictionCorrect: boolean;
  preloadAttempted: boolean;
  preloadDurationMs: number;
  ramUsageBytes: number;
  coldStartReductionMs: number;
  responseLatencyMs: number;
}

export interface PreloadMetricSummary {
  generatedAt: string;
  events: number;
  predictionAccuracy: number;
  averagePreloadDurationMs: number;
  averageRamUsageBytes: number;
  averageColdStartReductionMs: number;
  averageResponseLatencyMs: number;
  byStrategy: Record<PreloadTimingStrategy, StrategySummary>;
}

export interface StrategySummary {
  events: number;
  predictionAccuracy: number;
  averagePreloadDurationMs: number;
  averageRamUsageBytes: number;
  averageColdStartReductionMs: number;
  averageResponseLatencyMs: number;
}

export interface PreloadMetricReport {
  summary: PreloadMetricSummary;
  events: PreloadMetricEvent[];
}

export class PreloadMetrics {
  private readonly events: PreloadMetricEvent[] = [];

  record(event: PreloadMetricEvent): void {
    this.events.push(event);
  }

  all(): PreloadMetricEvent[] {
    return this.events.slice();
  }

  report(): PreloadMetricReport {
    return {
      summary: summarize(this.events),
      events: this.all(),
    };
  }

  async writeJson(logPath = path.join("logs", "preload-agent-results.json")): Promise<void> {
    const resolved = path.resolve(process.cwd(), logPath);
    await fs.mkdir(path.dirname(resolved), { recursive: true });
    await fs.writeFile(resolved, `${JSON.stringify(this.report(), null, 2)}\n`, "utf8");
  }
}

export function summarize(events: PreloadMetricEvent[]): PreloadMetricSummary {
  return {
    generatedAt: new Date().toISOString(),
    events: events.length,
    predictionAccuracy: average(events, (event) => event.predictionCorrect ? 1 : 0),
    averagePreloadDurationMs: average(events, (event) => event.preloadDurationMs),
    averageRamUsageBytes: average(events, (event) => event.ramUsageBytes),
    averageColdStartReductionMs: average(events, (event) => event.coldStartReductionMs),
    averageResponseLatencyMs: average(events, (event) => event.responseLatencyMs),
    byStrategy: {
      aggressive: summarizeStrategy(events, "aggressive"),
      delayed: summarizeStrategy(events, "delayed"),
      confidence: summarizeStrategy(events, "confidence"),
    },
  };
}

function summarizeStrategy(
  events: PreloadMetricEvent[],
  strategy: PreloadTimingStrategy,
): StrategySummary {
  const selected = events.filter((event) => event.strategy === strategy);
  return {
    events: selected.length,
    predictionAccuracy: average(selected, (event) => event.predictionCorrect ? 1 : 0),
    averagePreloadDurationMs: average(selected, (event) => event.preloadDurationMs),
    averageRamUsageBytes: average(selected, (event) => event.ramUsageBytes),
    averageColdStartReductionMs: average(selected, (event) => event.coldStartReductionMs),
    averageResponseLatencyMs: average(selected, (event) => event.responseLatencyMs),
  };
}

function average(events: PreloadMetricEvent[], select: (event: PreloadMetricEvent) => number): number {
  if (events.length === 0) {
    return 0;
  }
  return round(events.reduce((sum, event) => sum + select(event), 0) / events.length, 4);
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
