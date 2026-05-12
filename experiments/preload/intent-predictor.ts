export type TaskIntent =
  | "coding"
  | "reasoning"
  | "web_research"
  | "chat"
  | "summarization";

export interface TaskRecord {
  text: string;
  actualIntent: TaskIntent;
  predictedIntent?: TaskIntent;
  confidence?: number;
  timestamp: number;
}

export interface IntentPrediction {
  intent: TaskIntent;
  confidence: number;
  scores: Record<TaskIntent, number>;
  signals: string[];
}

const INTENTS: TaskIntent[] = [
  "coding",
  "reasoning",
  "web_research",
  "chat",
  "summarization",
];

const KEYWORDS: Record<TaskIntent, string[]> = {
  coding: [
    "bug",
    "code",
    "compile",
    "debug",
    "function",
    "implement",
    "merge",
    "pr",
    "refactor",
    "repo",
    "test",
    "typescript",
  ],
  reasoning: [
    "analyze",
    "compare",
    "decide",
    "evaluate",
    "explain",
    "infer",
    "logic",
    "plan",
    "prove",
    "reason",
    "tradeoff",
    "why",
  ],
  web_research: [
    "browse",
    "current",
    "latest",
    "link",
    "look up",
    "news",
    "price",
    "research",
    "search",
    "source",
    "today",
    "web",
  ],
  chat: [
    "hello",
    "hi",
    "thanks",
    "thank you",
    "quick question",
    "can you",
    "could you",
    "please",
    "help",
    "idea",
  ],
  summarization: [
    "brief",
    "digest",
    "extract",
    "recap",
    "shorten",
    "summarize",
    "summary",
    "tl;dr",
    "takeaways",
    "notes",
    "outline",
    "condense",
  ],
};

const RECENT_WINDOW = 8;
const ROLLING_WINDOW_MS = 30 * 60 * 1000;

export class IntentPredictor {
  private history: TaskRecord[];

  constructor(history: TaskRecord[] = []) {
    this.history = history.slice(-64);
  }

  predict(text: string, observedHistory: TaskRecord[] = this.history): IntentPrediction {
    const normalized = normalize(text);
    const scores = emptyScores(0.02);
    const signals: string[] = [];

    for (const intent of INTENTS) {
      for (const keyword of KEYWORDS[intent]) {
        const hits = countKeywordHits(normalized, keyword);
        if (hits > 0) {
          const weight = keyword.includes(" ") ? 1.45 : 1;
          scores[intent] += hits * weight;
          signals.push(`${intent}:keyword:${keyword}`);
        }
      }
    }

    this.applyShapeSignals(normalized, scores, signals);
    this.applyRollingTrends(observedHistory, scores, signals);
    this.applyRecentTaskWeighting(observedHistory, scores, signals);

    const ranked = INTENTS.map((intent) => ({ intent, score: scores[intent] }))
      .sort((a, b) => b.score - a.score);
    const total = ranked.reduce((sum, entry) => sum + Math.max(entry.score, 0), 0) || 1;
    const best = ranked[0] ?? { intent: "chat" as TaskIntent, score: 0 };
    const second = ranked[1] ?? { intent: "chat" as TaskIntent, score: 0 };
    const margin = Math.max(0, best.score - second.score);
    const probability = best.score / total;
    const confidence = clamp(0.35 + probability * 0.45 + margin * 0.06, 0.05, 0.98);

    return {
      intent: best.intent,
      confidence: round(confidence, 4),
      scores: roundScores(scores),
      signals: signals.slice(0, 16),
    };
  }

  recordOutcome(text: string, actualIntent: TaskIntent, predicted?: IntentPrediction): void {
    const record: TaskRecord = {
      text,
      actualIntent,
      timestamp: Date.now(),
    };
    if (predicted) {
      record.predictedIntent = predicted.intent;
      record.confidence = predicted.confidence;
    }
    this.history.push(record);
    this.history = this.history.slice(-64);
  }

  getHistory(): TaskRecord[] {
    return this.history.slice();
  }

  private applyShapeSignals(
    text: string,
    scores: Record<TaskIntent, number>,
    signals: string[],
  ): void {
    if (/```|\/\*|\b(class|interface|const|let|def|import|export)\b/.test(text)) {
      scores.coding += 1.3;
      signals.push("coding:shape:code-block-or-symbols");
    }
    if (/\?/.test(text) && /\b(why|how|should|would|could|explain)\b/.test(text)) {
      scores.reasoning += 0.7;
      signals.push("reasoning:shape:question");
    }
    if (/\bhttps?:\/\/|\bwww\./.test(text)) {
      scores.web_research += 1.1;
      signals.push("web_research:shape:url");
    }
    if (text.length > 900 || /\b(attached|transcript|document|article)\b/.test(text)) {
      scores.summarization += 0.8;
      signals.push("summarization:shape:long-or-document");
    }
    if (text.length < 80 && /\b(hi|hello|thanks|please)\b/.test(text)) {
      scores.chat += 0.8;
      signals.push("chat:shape:short-conversation");
    }
  }

  private applyRollingTrends(
    history: TaskRecord[],
    scores: Record<TaskIntent, number>,
    signals: string[],
  ): void {
    const now = Date.now();
    const recent = history.filter((item) => now - item.timestamp <= ROLLING_WINDOW_MS);
    if (recent.length === 0) {
      return;
    }

    const trendScores = emptyScores(0);
    for (const item of recent) {
      const ageRatio = (now - item.timestamp) / ROLLING_WINDOW_MS;
      trendScores[item.actualIntent] += 0.35 * (1 - ageRatio);
    }

    for (const intent of INTENTS) {
      if (trendScores[intent] > 0) {
        scores[intent] += trendScores[intent];
        signals.push(`${intent}:rolling:${round(trendScores[intent], 3)}`);
      }
    }
  }

  private applyRecentTaskWeighting(
    history: TaskRecord[],
    scores: Record<TaskIntent, number>,
    signals: string[],
  ): void {
    const recent = history.slice(-RECENT_WINDOW);
    recent.forEach((item, index) => {
      const recency = (index + 1) / recent.length;
      const correctnessBonus = item.predictedIntent === item.actualIntent ? 1 : 0.55;
      const confidence = item.confidence ?? 0.5;
      const boost = 0.08 + recency * correctnessBonus * confidence * 0.24;
      scores[item.actualIntent] += boost;
      if (index >= Math.max(0, recent.length - 3)) {
        signals.push(`${item.actualIntent}:recent:${round(boost, 3)}`);
      }
    });
  }
}

export function createIntentPredictor(history: TaskRecord[] = []): IntentPredictor {
  return new IntentPredictor(history);
}

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

function countKeywordHits(text: string, keyword: string): number {
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const boundary = keyword.includes(" ") ? escaped : `\\b${escaped}\\b`;
  return text.match(new RegExp(boundary, "g"))?.length ?? 0;
}

function emptyScores(value: number): Record<TaskIntent, number> {
  return {
    coding: value,
    reasoning: value,
    web_research: value,
    chat: value,
    summarization: value,
  };
}

function roundScores(scores: Record<TaskIntent, number>): Record<TaskIntent, number> {
  return {
    coding: round(scores.coding, 4),
    reasoning: round(scores.reasoning, 4),
    web_research: round(scores.web_research, 4),
    chat: round(scores.chat, 4),
    summarization: round(scores.summarization, 4),
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
