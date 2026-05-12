import type { TaskIntent } from "./intent-predictor.ts";

export type PreloadResourceType =
  | "tokenizer_state"
  | "embeddings"
  | "vector_index"
  | "prompt_template"
  | "lightweight_model";

export interface PreloadResource {
  key: string;
  intent: TaskIntent;
  type: PreloadResourceType;
  payload: unknown;
  bytes: number;
  createdAt: number;
  lastAccessedAt: number;
  ttlMs: number;
}

export interface CacheSnapshot {
  entries: number;
  bytes: number;
  maxBytes: number;
  byType: Record<PreloadResourceType, number>;
  byIntent: Record<TaskIntent, number>;
}

const RESOURCE_TYPES: PreloadResourceType[] = [
  "tokenizer_state",
  "embeddings",
  "vector_index",
  "prompt_template",
  "lightweight_model",
];

const INTENTS: TaskIntent[] = [
  "coding",
  "reasoning",
  "web_research",
  "chat",
  "summarization",
];

export class PreloadCache {
  private entries = new Map<string, PreloadResource>();
  private totalBytes = 0;
  private readonly maxBytes: number;

  constructor(maxBytes = 28 * 1024 * 1024) {
    this.maxBytes = maxBytes;
  }

  get(key: string): PreloadResource | undefined {
    const entry = this.entries.get(key);
    if (!entry) {
      return undefined;
    }
    if (Date.now() - entry.createdAt > entry.ttlMs) {
      this.delete(key);
      return undefined;
    }
    entry.lastAccessedAt = Date.now();
    return entry;
  }

  set(resource: Omit<PreloadResource, "bytes" | "createdAt" | "lastAccessedAt"> & { bytes?: number }): PreloadResource {
    const now = Date.now();
    const entry: PreloadResource = {
      ...resource,
      bytes: resource.bytes ?? estimateBytes(resource.payload),
      createdAt: now,
      lastAccessedAt: now,
    };

    this.delete(entry.key);
    this.entries.set(entry.key, entry);
    this.totalBytes += entry.bytes;
    this.evictToLimit();
    return entry;
  }

  delete(key: string): boolean {
    const entry = this.entries.get(key);
    if (!entry) {
      return false;
    }
    this.entries.delete(key);
    this.totalBytes -= entry.bytes;
    return true;
  }

  hasFresh(key: string): boolean {
    return this.get(key) !== undefined;
  }

  clearExpired(now = Date.now()): number {
    let removed = 0;
    for (const entry of this.entries.values()) {
      if (now - entry.createdAt > entry.ttlMs) {
        this.delete(entry.key);
        removed += 1;
      }
    }
    return removed;
  }

  snapshot(): CacheSnapshot {
    this.clearExpired();
    const byType = zeroResourceTypes();
    const byIntent = zeroIntents();

    for (const entry of this.entries.values()) {
      byType[entry.type] += 1;
      byIntent[entry.intent] += 1;
    }

    return {
      entries: this.entries.size,
      bytes: this.totalBytes,
      maxBytes: this.maxBytes,
      byType,
      byIntent,
    };
  }

  listKeys(): string[] {
    this.clearExpired();
    return Array.from(this.entries.keys()).sort();
  }

  private evictToLimit(): void {
    while (this.totalBytes > this.maxBytes && this.entries.size > 0) {
      const oldest = Array.from(this.entries.values())
        .sort((a, b) => a.lastAccessedAt - b.lastAccessedAt)[0];
      if (!oldest) {
        return;
      }
      this.delete(oldest.key);
    }
  }
}

export function cacheKey(intent: TaskIntent, type: PreloadResourceType): string {
  return `${intent}:${type}`;
}

function estimateBytes(payload: unknown): number {
  if (typeof payload === "string") {
    return payload.length * 2;
  }
  if (payload instanceof Float32Array) {
    return payload.byteLength;
  }
  if (Array.isArray(payload)) {
    return payload.length * 16;
  }
  return JSON.stringify(payload).length * 2;
}

function zeroResourceTypes(): Record<PreloadResourceType, number> {
  return RESOURCE_TYPES.reduce((acc, type) => {
    acc[type] = 0;
    return acc;
  }, {} as Record<PreloadResourceType, number>);
}

function zeroIntents(): Record<TaskIntent, number> {
  return INTENTS.reduce((acc, intent) => {
    acc[intent] = 0;
    return acc;
  }, {} as Record<TaskIntent, number>);
}
