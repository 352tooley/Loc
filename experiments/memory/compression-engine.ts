export type CompressionStrategy = "naive" | "semantic" | "bullets";

export interface CompressionInput {
  id: string;
  text: string;
  timestamp?: number;
}

export interface CompressionResult {
  strategy: CompressionStrategy;
  sourceIds: string[];
  summary: string;
  originalTokens: number;
  compressedTokens: number;
  compressionRatio: number;
  elapsedMs: number;
}

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "for",
  "from",
  "has",
  "have",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "or",
  "that",
  "the",
  "this",
  "to",
  "was",
  "with",
]);

export function nowMs(): number {
  if (typeof process !== "undefined" && process.hrtime?.bigint) {
    return Number(process.hrtime.bigint()) / 1_000_000;
  }
  return Date.now();
}

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9_\-\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

export function estimateTokenCount(text: string): number {
  const roughWords = text.trim().split(/\s+/).filter(Boolean).length;
  const charTokens = Math.ceil(text.length / 4);
  return Math.max(roughWords, charTokens);
}

export function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

export function termVector(text: string): Map<string, number> {
  const vector = new Map<string, number>();
  for (const token of tokenize(text)) {
    vector.set(token, (vector.get(token) ?? 0) + 1);
  }
  return vector;
}

export function cosineSimilarity(left: Map<string, number>, right: Map<string, number>): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (const value of left.values()) {
    leftNorm += value * value;
  }
  for (const value of right.values()) {
    rightNorm += value * value;
  }
  for (const [term, value] of left.entries()) {
    dot += value * (right.get(term) ?? 0);
  }

  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }
  return dot / Math.sqrt(leftNorm * rightNorm);
}

export class CompressionEngine {
  compress(
    inputs: CompressionInput[],
    strategy: CompressionStrategy,
    targetRatio = 0.35,
  ): CompressionResult {
    const started = nowMs();
    const originalText = inputs.map((item) => item.text).join("\n");
    const originalTokens = estimateTokenCount(originalText);
    const targetTokens = Math.max(24, Math.floor(originalTokens * targetRatio));

    let summary: string;
    if (strategy === "semantic") {
      summary = this.semanticSummary(originalText, targetTokens);
    } else if (strategy === "bullets") {
      summary = this.bulletExtraction(originalText, targetTokens);
    } else {
      summary = this.naiveSummary(originalText, targetTokens);
    }

    const compressedTokens = estimateTokenCount(summary);
    return {
      strategy,
      sourceIds: inputs.map((item) => item.id),
      summary,
      originalTokens,
      compressedTokens,
      compressionRatio: originalTokens === 0 ? 1 : compressedTokens / originalTokens,
      elapsedMs: round(nowMs() - started),
    };
  }

  private naiveSummary(text: string, targetTokens: number): string {
    const sentences = splitSentences(text);
    const selected: string[] = [];
    let total = 0;

    for (const sentence of sentences) {
      const tokens = estimateTokenCount(sentence);
      if (selected.length > 0 && total + tokens > targetTokens) {
        break;
      }
      selected.push(sentence);
      total += tokens;
    }

    return selected.join(" ");
  }

  private semanticSummary(text: string, targetTokens: number): string {
    const sentences = splitSentences(text);
    const centroid = termVector(text);
    const ranked = sentences
      .map((sentence, index) => {
        const similarity = cosineSimilarity(termVector(sentence), centroid);
        const recencyBoost = sentences.length === 0 ? 0 : index / sentences.length / 5;
        return { sentence, index, score: similarity + recencyBoost };
      })
      .sort((left, right) => right.score - left.score);

    const selected: typeof ranked = [];
    let total = 0;
    for (const candidate of ranked) {
      const tokens = estimateTokenCount(candidate.sentence);
      if (selected.length > 0 && total + tokens > targetTokens) {
        continue;
      }
      selected.push(candidate);
      total += tokens;
      if (total >= targetTokens) {
        break;
      }
    }

    return selected.sort((left, right) => left.index - right.index).map((item) => item.sentence).join(" ");
  }

  private bulletExtraction(text: string, targetTokens: number): string {
    const actionTerms = new Set([
      "blocker",
      "bug",
      "decision",
      "error",
      "failed",
      "fix",
      "metric",
      "next",
      "risk",
      "todo",
      "warning",
    ]);
    const sentences = splitSentences(text);
    const ranked = sentences
      .map((sentence, index) => {
        const tokens = tokenize(sentence);
        const actionScore = tokens.filter((token) => actionTerms.has(token)).length;
        const keywordScore = new Set(tokens).size / 20;
        return { sentence, index, score: actionScore * 2 + keywordScore };
      })
      .sort((left, right) => right.score - left.score || left.index - right.index);

    const bullets: string[] = [];
    let total = 0;
    for (const candidate of ranked) {
      const line = `- ${candidate.sentence}`;
      const tokens = estimateTokenCount(line);
      if (bullets.length > 0 && total + tokens > targetTokens) {
        continue;
      }
      bullets.push(line);
      total += tokens;
      if (total >= targetTokens) {
        break;
      }
    }

    return bullets.join("\n");
  }
}

export function round(value: number, places = 3): number {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}
