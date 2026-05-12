import {
  CompressionEngine,
  estimateTokenCount,
} from "./compression-engine.ts";
import type { CompressionResult, CompressionStrategy } from "./compression-engine.ts";

export interface MemoryEntry {
  id: string;
  text: string;
  timestamp: number;
  tokenCount: number;
  tags?: string[];
}

export interface ArchivedContext {
  id: string;
  text: string;
  timestamp: number;
  tokenCount: number;
  strategy: CompressionStrategy;
  sourceIds: string[];
  compression: CompressionResult;
}

export interface RollingWindowOptions {
  maxActiveTokens: number;
  archiveBatchSize?: number;
  compressionStrategy?: CompressionStrategy;
  compressionRatio?: number;
}

export interface WindowSnapshot {
  activeContext: MemoryEntry[];
  archivedContext: ArchivedContext[];
  activeTokens: number;
  archivedTokens: number;
  totalTokens: number;
}

export class RollingMemoryWindow {
  readonly activeContext: MemoryEntry[] = [];
  readonly archivedContext: ArchivedContext[] = [];

  private readonly engine: CompressionEngine;
  private readonly maxActiveTokens: number;
  private readonly archiveBatchSize: number;
  private readonly compressionStrategy: CompressionStrategy;
  private readonly compressionRatio: number;
  private archiveSequence = 0;

  constructor(options: RollingWindowOptions, engine = new CompressionEngine()) {
    this.engine = engine;
    this.maxActiveTokens = options.maxActiveTokens;
    this.archiveBatchSize = options.archiveBatchSize ?? 4;
    this.compressionStrategy = options.compressionStrategy ?? "semantic";
    this.compressionRatio = options.compressionRatio ?? 0.35;
  }

  add(text: string, tags: string[] = [], timestamp = Date.now(), id?: string): MemoryEntry {
    const entry: MemoryEntry = {
      id: id ?? `m-${timestamp}-${this.activeContext.length + this.archivedContext.length}`,
      text,
      timestamp,
      tokenCount: estimateTokenCount(text),
      tags,
    };

    this.activeContext.push(entry);
    this.enforceLimit();
    return entry;
  }

  snapshot(): WindowSnapshot {
    const activeTokens = this.activeContext.reduce((sum, item) => sum + item.tokenCount, 0);
    const archivedTokens = this.archivedContext.reduce((sum, item) => sum + item.tokenCount, 0);
    return {
      activeContext: [...this.activeContext],
      archivedContext: [...this.archivedContext],
      activeTokens,
      archivedTokens,
      totalTokens: activeTokens + archivedTokens,
    };
  }

  private enforceLimit(): void {
    while (this.activeTokens() > this.maxActiveTokens && this.activeContext.length > 1) {
      const batch = this.activeContext.splice(0, this.archiveBatchSize);
      const compression = this.engine.compress(batch, this.compressionStrategy, this.compressionRatio);
      this.archivedContext.push({
        id: `archive-${++this.archiveSequence}`,
        text: compression.summary,
        timestamp: batch[batch.length - 1].timestamp,
        tokenCount: compression.compressedTokens,
        strategy: this.compressionStrategy,
        sourceIds: compression.sourceIds,
        compression,
      });
    }
  }

  private activeTokens(): number {
    return this.activeContext.reduce((sum, item) => sum + item.tokenCount, 0);
  }
}
