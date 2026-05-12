import { evaluateConfidence } from "./confidence-engine.ts";
import type { ConfidenceResult } from "./confidence-engine.ts";
import type { ModelTier } from "./quality-metrics.ts";

export type EscalationStrategy =
  | "always-small-model"
  | "confidence-triggered"
  | "task-type"
  | "retry-escalation";

export interface EscalationInput {
  prompt: string;
  response: string;
  taskType?: string;
  model: ModelTier;
  strategy: EscalationStrategy;
  retryCount?: number;
  confidenceThreshold?: number;
}

export interface EscalationDecision {
  selectedModel: ModelTier;
  escalated: boolean;
  strategy: EscalationStrategy;
  confidence: ConfidenceResult;
  reason: string;
}

const HIGH_RISK_TASKS = new Set(["legal", "medical", "financial", "security", "code-review"]);

export function routeEscalation(input: EscalationInput): EscalationDecision {
  const confidence = evaluateConfidence({
    prompt: input.prompt,
    response: input.response,
    taskType: input.taskType,
    model: input.model,
  });

  if (input.strategy === "always-small-model") {
    return decision(input, confidence, "small", "baseline small-model route");
  }

  if (input.strategy === "task-type" && input.taskType && HIGH_RISK_TASKS.has(input.taskType)) {
    return decision(input, confidence, "large", `task type requires large model: ${input.taskType}`);
  }

  if (input.strategy === "retry-escalation" && (input.retryCount || 0) > 0) {
    return decision(input, confidence, nextTier(input.model), "retry requested higher model tier");
  }

  const threshold = input.confidenceThreshold ?? 0.62;
  if (input.strategy === "confidence-triggered" && confidence.score < threshold) {
    const target = confidence.score < 0.45 ? "large" : "medium";
    return decision(input, confidence, target, `confidence ${confidence.score} below ${threshold}`);
  }

  return decision(input, confidence, input.model, "no escalation criteria met");
}

export function shouldEscalateForOracle(taskType: string, responseQualityScore: number): boolean {
  return HIGH_RISK_TASKS.has(taskType) || responseQualityScore < 0.68;
}

function decision(
  input: EscalationInput,
  confidence: ConfidenceResult,
  selectedModel: ModelTier,
  reason: string,
): EscalationDecision {
  return {
    selectedModel,
    escalated: selectedModel !== input.model,
    strategy: input.strategy,
    confidence,
    reason,
  };
}

function nextTier(model: ModelTier): ModelTier {
  if (model === "small") return "medium";
  if (model === "medium") return "large";
  return "large";
}
