export type ModelTier = "small" | "medium" | "large";

export interface ModelProfile {
  tier: ModelTier;
  latencyMs: number;
  tokenCostPer1k: number;
  qualityBias: number;
}

export interface ResponseQuality {
  score: number;
  lengthScore: number;
  uncertaintyPenalty: number;
  contradictionPenalty: number;
  specificityScore: number;
  issues: string[];
}

export interface BenchmarkObservation {
  caseId: string;
  taskType: string;
  model: ModelTier;
  latencyMs: number;
  tokenCost: number;
  qualityScore: number;
  escalated: boolean;
  shouldEscalate: boolean;
}

export interface BenchmarkMetrics {
  totalCases: number;
  averageLatencyMs: number;
  totalTokenCost: number;
  averageQuality: number;
  escalationFrequency: number;
  escalationCount: number;
  falseEscalations: number;
  missedEscalations: number;
  qualityImprovement: number;
  latencyPenaltyMs: number;
  byModel: Record<ModelTier, {
    count: number;
    averageLatencyMs: number;
    totalTokenCost: number;
    averageQuality: number;
  }>;
}

export const MODEL_PROFILES: Record<ModelTier, ModelProfile> = {
  small: {
    tier: "small",
    latencyMs: 450,
    tokenCostPer1k: 0.15,
    qualityBias: -0.08,
  },
  medium: {
    tier: "medium",
    latencyMs: 950,
    tokenCostPer1k: 0.65,
    qualityBias: 0.04,
  },
  large: {
    tier: "large",
    latencyMs: 1650,
    tokenCostPer1k: 2.25,
    qualityBias: 0.12,
  },
};

export function estimateTokenCount(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(words * 1.3));
}

export function estimateTokenCost(text: string, model: ModelTier): number {
  const tokenCount = estimateTokenCount(text);
  return round((tokenCount / 1000) * MODEL_PROFILES[model].tokenCostPer1k, 6);
}

export function evaluateResponseQuality(input: {
  response: string;
  model: ModelTier;
  uncertaintyCount: number;
  contradictionCount: number;
  lengthScore: number;
}): ResponseQuality {
  const specificityScore = calculateSpecificityScore(input.response);
  const uncertaintyPenalty = Math.min(0.3, input.uncertaintyCount * 0.08);
  const contradictionPenalty = Math.min(0.45, input.contradictionCount * 0.2);
  const modelBias = MODEL_PROFILES[input.model].qualityBias;
  const score = clamp(
    input.lengthScore * 0.28 +
      specificityScore * 0.32 +
      (1 - uncertaintyPenalty) * 0.18 +
      (1 - contradictionPenalty) * 0.22 +
      modelBias,
    0,
    1,
  );

  const issues: string[] = [];
  if (input.lengthScore < 0.45) issues.push("response length outside expected range");
  if (input.uncertaintyCount > 0) issues.push("uncertainty language detected");
  if (input.contradictionCount > 0) issues.push("possible contradiction detected");
  if (specificityScore < 0.45) issues.push("low specificity");

  return {
    score: round(score, 4),
    lengthScore: round(input.lengthScore, 4),
    uncertaintyPenalty: round(uncertaintyPenalty, 4),
    contradictionPenalty: round(contradictionPenalty, 4),
    specificityScore: round(specificityScore, 4),
    issues,
  };
}

export function calculateBenchmarkMetrics(
  observations: BenchmarkObservation[],
  smallModelBaseline: BenchmarkObservation[],
): BenchmarkMetrics {
  const totals = observations.reduce(
    (acc, observation) => {
      acc.latency += observation.latencyMs;
      acc.cost += observation.tokenCost;
      acc.quality += observation.qualityScore;
      if (observation.escalated) acc.escalations += 1;
      if (observation.escalated && !observation.shouldEscalate) acc.falseEscalations += 1;
      if (!observation.escalated && observation.shouldEscalate) acc.missedEscalations += 1;
      return acc;
    },
    { latency: 0, cost: 0, quality: 0, escalations: 0, falseEscalations: 0, missedEscalations: 0 },
  );

  const baselineLatency = average(smallModelBaseline.map((item) => item.latencyMs));
  const baselineQuality = average(smallModelBaseline.map((item) => item.qualityScore));
  const observedLatency = average(observations.map((item) => item.latencyMs));
  const observedQuality = average(observations.map((item) => item.qualityScore));

  return {
    totalCases: observations.length,
    averageLatencyMs: round(observedLatency, 2),
    totalTokenCost: round(totals.cost, 6),
    averageQuality: round(observedQuality, 4),
    escalationFrequency: round(safeDivide(totals.escalations, observations.length), 4),
    escalationCount: totals.escalations,
    falseEscalations: totals.falseEscalations,
    missedEscalations: totals.missedEscalations,
    qualityImprovement: round(observedQuality - baselineQuality, 4),
    latencyPenaltyMs: round(observedLatency - baselineLatency, 2),
    byModel: summarizeByModel(observations),
  };
}

function calculateSpecificityScore(response: string): number {
  const bullets = (response.match(/(^|\n)\s*(-|\*|\d+\.)\s+/g) || []).length;
  const numbers = (response.match(/\b\d+(\.\d+)?%?\b/g) || []).length;
  const concreteTerms = (response.match(/\b(because|therefore|for example|step|must|should|will)\b/gi) || []).length;
  return clamp(0.25 + bullets * 0.08 + numbers * 0.05 + concreteTerms * 0.04, 0, 1);
}

function summarizeByModel(observations: BenchmarkObservation[]): BenchmarkMetrics["byModel"] {
  const initial = {
    small: emptyModelSummary(),
    medium: emptyModelSummary(),
    large: emptyModelSummary(),
  };

  for (const observation of observations) {
    const summary = initial[observation.model];
    summary.count += 1;
    summary.averageLatencyMs += observation.latencyMs;
    summary.totalTokenCost += observation.tokenCost;
    summary.averageQuality += observation.qualityScore;
  }

  for (const tier of Object.keys(initial) as ModelTier[]) {
    const summary = initial[tier];
    summary.averageLatencyMs = round(safeDivide(summary.averageLatencyMs, summary.count), 2);
    summary.totalTokenCost = round(summary.totalTokenCost, 6);
    summary.averageQuality = round(safeDivide(summary.averageQuality, summary.count), 4);
  }

  return initial;
}

function emptyModelSummary(): BenchmarkMetrics["byModel"][ModelTier] {
  return {
    count: 0,
    averageLatencyMs: 0,
    totalTokenCost: 0,
    averageQuality: 0,
  };
}

function average(values: number[]): number {
  return safeDivide(values.reduce((sum, value) => sum + value, 0), values.length);
}

function safeDivide(value: number, divisor: number): number {
  return divisor === 0 ? 0 : value / divisor;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
