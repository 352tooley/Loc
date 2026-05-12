import {
  calculateBenchmarkMetrics,
  estimateTokenCost,
  MODEL_PROFILES,
  round,
} from "./quality-metrics.ts";
import type {
  BenchmarkObservation,
  ModelTier,
} from "./quality-metrics.ts";
import { analyzeResponse } from "./response-analyzer.ts";
import { routeEscalation, shouldEscalateForOracle } from "./escalation-router.ts";
import type { EscalationStrategy } from "./escalation-router.ts";
import fs from "node:fs";
import path from "node:path";

declare const process: { argv: string[]; cwd: () => string };

interface BenchmarkCase {
  id: string;
  taskType: string;
  prompt: string;
  smallResponse: string;
  mediumResponse: string;
  largeResponse: string;
}

interface BenchmarkRun {
  generatedAt: string;
  strategy: EscalationStrategy;
  metrics: ReturnType<typeof calculateBenchmarkMetrics>;
  observations: BenchmarkObservation[];
}

const CASES: BenchmarkCase[] = [
  {
    id: "pack-fragile",
    taskType: "planning",
    prompt: "Create a packing plan for fragile glassware with cost and breakage risk considerations.",
    smallResponse: "Wrap the glassware and put it in a box. It might be fine, but it depends on the trip.",
    mediumResponse:
      "Use double-wall boxes, wrap each item in paper and bubble wrap, add 2 inches of void fill, and label the carton fragile. This should reduce breakage risk while keeping material cost moderate.",
    largeResponse:
      "Use a double-wall carton sized so each wrapped glass has 2 inches of clearance. Wrap stems and bowls separately, place heavier pieces on the bottom, fill voids, and run a shake test before sealing. This adds modest material cost while lowering breakage risk.",
  },
  {
    id: "security-policy",
    taskType: "security",
    prompt: "Review whether the response safely handles API key leakage in logs.",
    smallResponse:
      "The logs never contain secrets except when debug mode includes them, so no action is required.",
    mediumResponse:
      "Mask API keys before writing logs, rotate any exposed keys, and restrict debug logs. The current answer is probably okay but should mention rotation.",
    largeResponse:
      "Treat API keys in logs as exposed credentials. Mask secrets before persistence, rotate affected keys, limit log retention, and add a regression test that asserts redaction for debug and error paths.",
  },
  {
    id: "returns-summary",
    taskType: "summary",
    prompt: "Summarize the return policy in clear customer-support language.",
    smallResponse:
      "Customers can return items within 30 days if unused. Include the order number and original packaging.",
    mediumResponse:
      "Customers can return unused items within 30 days with the order number and original packaging. Refunds are processed after inspection.",
    largeResponse:
      "Customers may return unused items within 30 days. Ask them to include the order number and original packaging, then explain that refunds are issued after inspection.",
  },
  {
    id: "medical-advice",
    taskType: "medical",
    prompt: "Answer a user asking whether chest pain can be ignored if it goes away.",
    smallResponse:
      "Chest pain can be ignored if it goes away and there are no other symptoms.",
    mediumResponse:
      "Chest pain can have many causes. If it is severe, returns, or comes with shortness of breath, seek urgent medical care.",
    largeResponse:
      "Do not dismiss chest pain solely because it improves. Seek urgent care for severe, recurring, or pressure-like pain, or pain with shortness of breath, sweating, nausea, or arm or jaw discomfort.",
  },
];

export function runBenchmark(strategy: EscalationStrategy = "confidence-triggered"): BenchmarkRun {
  const observations: BenchmarkObservation[] = [];
  const baseline: BenchmarkObservation[] = [];

  for (const item of CASES) {
    const smallObservation = observeCase(item, "small", false);
    baseline.push(smallObservation);

    const route = routeEscalation({
      prompt: item.prompt,
      response: item.smallResponse,
      taskType: item.taskType,
      model: "small",
      strategy,
      retryCount: item.id === "pack-fragile" ? 1 : 0,
    });
    observations.push(observeCase(item, route.selectedModel, route.escalated));
  }

  return {
    generatedAt: new Date().toISOString(),
    strategy,
    metrics: calculateBenchmarkMetrics(observations, baseline),
    observations,
  };
}

export function writeBenchmarkLog(run: BenchmarkRun, outputPath = "logs/quality-agent-results.json"): void {
  const absolutePath = path.join(process.cwd(), outputPath);
  const outputDirectory = absolutePath.replace(/\/[^/]+$/, "");

  if (!fs.existsSync(outputDirectory)) {
    fs.mkdirSync(outputDirectory, { recursive: true });
  }

  fs.writeFileSync(absolutePath, `${JSON.stringify(run, null, 2)}\n`, "utf8");
}

function observeCase(item: BenchmarkCase, model: ModelTier, escalated: boolean): BenchmarkObservation {
  const response = responseForModel(item, model);
  const analysis = analyzeResponse(response, { model });
  const latencyMs = MODEL_PROFILES[model].latencyMs + syntheticLatencyJitter(item.id, model);
  const qualityScore = round(analysis.quality.score, 4);

  return {
    caseId: item.id,
    taskType: item.taskType,
    model,
    latencyMs,
    tokenCost: estimateTokenCost(`${item.prompt}\n${response}`, model),
    qualityScore,
    escalated,
    shouldEscalate: shouldEscalateForOracle(item.taskType, analyzeResponse(item.smallResponse).quality.score),
  };
}

function responseForModel(item: BenchmarkCase, model: ModelTier): string {
  if (model === "large") return item.largeResponse;
  if (model === "medium") return item.mediumResponse;
  return item.smallResponse;
}

function syntheticLatencyJitter(caseId: string, model: ModelTier): number {
  const seed = caseId.split("").reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const tierOffset = model === "small" ? 15 : model === "medium" ? 35 : 60;
  return (seed % 90) + tierOffset;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  writeBenchmarkLog(runBenchmark());
}
