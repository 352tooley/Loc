import { analyzeResponse } from "./response-analyzer.ts";
import type { ResponseAnalysis } from "./response-analyzer.ts";
import { clamp, round } from "./quality-metrics.ts";
import type { ModelTier } from "./quality-metrics.ts";

export interface ConfidenceInput {
  prompt: string;
  response: string;
  taskType?: string;
  model?: ModelTier;
  expectedMinWords?: number;
  expectedMaxWords?: number;
}

export interface ConfidenceResult {
  score: number;
  band: "low" | "medium" | "high";
  analysis: ResponseAnalysis;
  factors: {
    length: number;
    certainty: number;
    consistency: number;
    taskFit: number;
    quality: number;
  };
  reasons: string[];
}

const COMPLEX_TASK_TYPES = new Set(["legal", "medical", "financial", "security", "architecture", "code-review"]);

export function evaluateConfidence(input: ConfidenceInput): ConfidenceResult {
  const model = input.model || "small";
  const analysis = analyzeResponse(input.response, {
    model,
    expectedMinWords: input.expectedMinWords,
    expectedMaxWords: input.expectedMaxWords,
  });

  const certainty = clamp(1 - analysis.uncertaintyPhrases.length * 0.18, 0, 1);
  const consistency = clamp(1 - analysis.contradictions.length * 0.28, 0, 1);
  const taskFit = scoreTaskFit(input.prompt, input.response, input.taskType);
  const quality = analysis.quality.score;
  const length = analysis.lengthScore;

  const score = round(
    clamp(
      length * 0.16 +
        certainty * 0.22 +
        consistency * 0.22 +
        taskFit * 0.18 +
        quality * 0.22,
      0,
      1,
    ),
    4,
  );

  return {
    score,
    band: score >= 0.78 ? "high" : score >= 0.55 ? "medium" : "low",
    analysis,
    factors: {
      length: round(length, 4),
      certainty: round(certainty, 4),
      consistency: round(consistency, 4),
      taskFit: round(taskFit, 4),
      quality: round(quality, 4),
    },
    reasons: buildReasons(score, input.taskType, analysis),
  };
}

function scoreTaskFit(prompt: string, response: string, taskType?: string): number {
  const promptTerms = extractMeaningfulTerms(prompt);
  const responseLower = response.toLowerCase();
  const overlap = promptTerms.filter((term) => responseLower.includes(term)).length;
  const overlapScore = promptTerms.length === 0 ? 0.6 : overlap / promptTerms.length;
  const complexityPenalty = taskType && COMPLEX_TASK_TYPES.has(taskType) ? 0.08 : 0;
  return clamp(0.35 + overlapScore * 0.65 - complexityPenalty, 0, 1);
}

function extractMeaningfulTerms(prompt: string): string[] {
  const stopWords = new Set([
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "about",
    "please",
    "should",
  ]);
  return prompt
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .split(/\s+/)
    .filter((term) => term.length > 3 && !stopWords.has(term))
    .slice(0, 18);
}

function buildReasons(score: number, taskType: string | undefined, analysis: ResponseAnalysis): string[] {
  const reasons: string[] = [];
  if (score < 0.55) reasons.push("confidence below experimental threshold");
  if (taskType && COMPLEX_TASK_TYPES.has(taskType)) reasons.push(`complex task type: ${taskType}`);
  if (analysis.lengthScore < 0.55) reasons.push("response length is weak for the expected range");
  if (analysis.uncertaintyPhrases.length > 0) reasons.push("uncertainty phrases found");
  if (analysis.contradictions.length > 0) reasons.push("possible contradictions found");
  if (analysis.quality.score < 0.6) reasons.push("quality heuristic below target");
  return reasons;
}
