import {
  cosineSimilarity,
  estimateTokenCount,
  termVector,
  tokenize,
} from "./compression-engine.ts";
import type { ArchivedContext, MemoryEntry } from "./rolling-window.ts";

export type RetrievalMode = "keyword" | "semantic" | "hybrid";

export interface RetrievalDocument {
  id: string;
  text: string;
  tokenCount: number;
  source: "active" | "archive" | "baseline";
  sourceIds?: string[];
}

export interface RetrievalResult extends RetrievalDocument {
  score: number;
}

export interface RetrievalQuery {
  query: string;
  expectedIds: string[];
}

export class RetrievalEngine {
  private documents: RetrievalDocument[];

  constructor(documents: RetrievalDocument[]) {
    this.documents = documents;
  }

  static fromWindow(activeContext: MemoryEntry[], archivedContext: ArchivedContext[]): RetrievalEngine {
    return new RetrievalEngine([
      ...activeContext.map((item) => ({
        id: item.id,
        text: item.text,
        tokenCount: item.tokenCount,
        source: "active" as const,
      })),
      ...archivedContext.map((item) => ({
        id: item.id,
        text: item.text,
        tokenCount: item.tokenCount,
        source: "archive" as const,
        sourceIds: item.sourceIds,
      })),
    ]);
  }

  search(query: string, mode: RetrievalMode, limit = 5): RetrievalResult[] {
    const queryTerms = tokenize(query);
    const queryVector = termVector(query);
    return this.documents
      .map((document) => ({
        ...document,
        score: this.score(document.text, queryTerms, queryVector, mode),
      }))
      .filter((document) => document.score > 0)
      .sort((left, right) => right.score - left.score || left.tokenCount - right.tokenCount)
      .slice(0, limit);
  }

  evaluate(queries: RetrievalQuery[], mode: RetrievalMode, limit = 5): number {
    if (queries.length === 0) {
      return 0;
    }

    const hits = queries.filter((query) => {
      const expected = new Set(query.expectedIds);
      return this.search(query.query, mode, limit).some((result) => {
        if (expected.has(result.id)) {
          return true;
        }
        return result.sourceIds?.some((sourceId) => expected.has(sourceId)) ?? false;
      });
    }).length;

    return hits / queries.length;
  }

  contextTokens(): number {
    return this.documents.reduce((sum, document) => sum + estimateTokenCount(document.text), 0);
  }

  private score(
    text: string,
    queryTerms: string[],
    queryVector: Map<string, number>,
    mode: RetrievalMode,
  ): number {
    const keyword = keywordScore(text, queryTerms);
    const semantic = cosineSimilarity(termVector(text), queryVector);

    if (mode === "keyword") {
      return keyword;
    }
    if (mode === "semantic") {
      return semantic;
    }
    return keyword * 0.55 + semantic * 0.45;
  }
}

export function keywordScore(text: string, queryTerms: string[]): number {
  if (queryTerms.length === 0) {
    return 0;
  }

  const terms = tokenize(text);
  const frequencies = new Map<string, number>();
  for (const term of terms) {
    frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
  }

  const matched = queryTerms.reduce((sum, term) => sum + Math.min(frequencies.get(term) ?? 0, 3), 0);
  return matched / queryTerms.length;
}
