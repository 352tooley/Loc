import { evaluateResponseQuality } from "./quality-metrics.ts";
import type { ModelTier, ResponseQuality } from "./quality-metrics.ts";

export interface ResponseAnalysis {
  responseLength: number;
  wordCount: number;
  sentenceCount: number;
  lengthScore: number;
  uncertaintyPhrases: string[];
  contradictions: string[];
  quality: ResponseQuality;
}

export interface ResponseAnalysisOptions {
  model?: ModelTier;
  expectedMinWords?: number;
  expectedMaxWords?: number;
}

const UNCERTAINTY_PATTERNS = [
  /\b(i think|i believe|probably|possibly|maybe|might|could be|unclear|not sure)\b/gi,
  /\b(as an ai|i cannot verify|it depends|hard to say|likely|appears to)\b/gi,
];

const CONTRADICTION_PATTERNS = [
  /\b(always|never)\b[\s\S]{0,120}\b(except|unless|sometimes)\b/gi,
  /\b(can|will|is)\b[\s\S]{0,80}\b(cannot|won't|is not|isn't)\b/gi,
  /\b(required|must)\b[\s\S]{0,100}\b(optional|not required|may skip)\b/gi,
  /\b(no|none|without)\b[\s\S]{0,100}\b(has|contains|includes|with)\b/gi,
];

export function analyzeResponse(
  response: string,
  options: ResponseAnalysisOptions = {},
): ResponseAnalysis {
  const model = options.model || "small";
  const wordCount = countWords(response);
  const sentenceCount = countSentences(response);
  const lengthScore = scoreResponseLength(
    wordCount,
    options.expectedMinWords || 40,
    options.expectedMaxWords || 220,
  );
  const uncertaintyPhrases = detectUncertaintyPhrases(response);
  const contradictions = detectContradictions(response);
  const quality = evaluateResponseQuality({
    response,
    model,
    uncertaintyCount: uncertaintyPhrases.length,
    contradictionCount: contradictions.length,
    lengthScore,
  });

  return {
    responseLength: response.length,
    wordCount,
    sentenceCount,
    lengthScore,
    uncertaintyPhrases,
    contradictions,
    quality,
  };
}

export function scoreResponseLength(wordCount: number, minWords = 40, maxWords = 220): number {
  if (wordCount <= 0) return 0;
  if (wordCount >= minWords && wordCount <= maxWords) return 1;
  if (wordCount < minWords) return Math.max(0, wordCount / minWords);

  const excessRatio = (wordCount - maxWords) / maxWords;
  return Math.max(0.15, 1 - excessRatio);
}

export function detectUncertaintyPhrases(response: string): string[] {
  return uniqueMatches(response, UNCERTAINTY_PATTERNS);
}

export function detectContradictions(response: string): string[] {
  const patternMatches = uniqueMatches(response, CONTRADICTION_PATTERNS);
  const sentenceConflicts = detectSimpleNegationConflicts(response);
  return [...patternMatches, ...sentenceConflicts];
}

function detectSimpleNegationConflicts(response: string): string[] {
  const sentences = splitSentences(response);
  const normalized = sentences.map((sentence) => sentence.toLowerCase().replace(/[^a-z0-9\s]/g, ""));
  const conflicts: string[] = [];

  for (let i = 0; i < normalized.length; i += 1) {
    for (let j = i + 1; j < normalized.length; j += 1) {
      const first = normalized[i];
      const second = normalized[j];
      if (hasSharedCoreTerms(first, second) && hasOppositePolarity(first, second)) {
        conflicts.push(`${sentences[i].trim()} / ${sentences[j].trim()}`);
      }
    }
  }

  return conflicts.slice(0, 5);
}

function hasSharedCoreTerms(first: string, second: string): boolean {
  const stopWords = new Set(["the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "are", "be"]);
  const firstTerms = first.split(/\s+/).filter((term) => term.length > 4 && !stopWords.has(term));
  const secondTerms = new Set(second.split(/\s+/).filter((term) => term.length > 4 && !stopWords.has(term)));
  return firstTerms.some((term) => secondTerms.has(term));
}

function hasOppositePolarity(first: string, second: string): boolean {
  const negative = /\b(no|not|never|cannot|can't|won't|without|disable|disabled|fail|failed)\b/;
  return negative.test(first) !== negative.test(second);
}

function uniqueMatches(response: string, patterns: RegExp[]): string[] {
  const matches = new Set<string>();
  for (const pattern of patterns) {
    const found = response.match(pattern) || [];
    for (const match of found) {
      matches.add(match.trim());
    }
  }
  return [...matches];
}

function countWords(response: string): number {
  return response.trim().split(/\s+/).filter(Boolean).length;
}

function countSentences(response: string): number {
  return splitSentences(response).length;
}

function splitSentences(response: string): string[] {
  return response.split(/[.!?]+/).map((sentence) => sentence.trim()).filter(Boolean);
}
