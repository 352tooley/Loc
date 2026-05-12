import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryMetrics } from "./memory-metrics.ts";
import { RetrievalEngine } from "./retrieval-engine.ts";
import { RollingMemoryWindow } from "./rolling-window.ts";
import type { CompressionStrategy } from "./compression-engine.ts";
import type { RetrievalDocument, RetrievalMode, RetrievalQuery } from "./retrieval-engine.ts";

interface ReplayItem {
  id: string;
  text: string;
  tags: string[];
}

const replayDataset: ReplayItem[] = [
  {
    id: "turn-001",
    tags: ["auth", "decision"],
    text: "Decision: SmartPack will keep OAuth refresh tokens encrypted at rest. The auth worker rotates keys every seven days and logs key ids for audit review.",
  },
  {
    id: "turn-002",
    tags: ["billing", "bug"],
    text: "Bug: invoice export failed when a customer had more than fifty package adjustments. Fix requires paginating adjustments before generating CSV rows.",
  },
  {
    id: "turn-003",
    tags: ["routing"],
    text: "Routing note: carrier selection uses destination zip, promised delivery date, cold-chain requirements, and historical damage rate.",
  },
  {
    id: "turn-004",
    tags: ["inventory", "risk"],
    text: "Risk: inventory reconciliation can double count quarantined lots when a warehouse sends duplicate ASN updates during shift change.",
  },
  {
    id: "turn-005",
    tags: ["ux"],
    text: "UX decision: the packing station shows scan status, remaining items, and exception reason in one dense panel for repeat operator workflows.",
  },
  {
    id: "turn-006",
    tags: ["metrics"],
    text: "Metric: preload success is measured by first reply latency, context hit rate, stale context rate, and manual correction count.",
  },
  {
    id: "turn-007",
    tags: ["security"],
    text: "Security warning: admin impersonation must emit an immutable audit event with actor id, target id, reason, and expiration time.",
  },
  {
    id: "turn-008",
    tags: ["support", "todo"],
    text: "Todo: support needs a saved reply for damaged gel packs that explains replacement timing, evidence requirements, and refund escalation.",
  },
  {
    id: "turn-009",
    tags: ["routing", "metric"],
    text: "Metric: route optimizer accuracy is validated by comparing recommended carrier against final carrier on delivered shipments with no manual override.",
  },
  {
    id: "turn-010",
    tags: ["inventory", "fix"],
    text: "Fix: quarantine counts should subtract released lots only after the release event is acknowledged by the warehouse integration.",
  },
  {
    id: "turn-011",
    tags: ["billing", "decision"],
    text: "Decision: billing adjustments below five dollars are grouped into a monthly small-balance line item to reduce invoice noise.",
  },
  {
    id: "turn-012",
    tags: ["auth", "risk"],
    text: "Risk: stale OAuth scopes may remain active if the account linking job retries after a provider timeout without reloading consent state.",
  },
];

const queries: RetrievalQuery[] = [
  { query: "OAuth refresh token rotation audit key ids", expectedIds: ["turn-001"] },
  { query: "invoice export failed package adjustments CSV pagination", expectedIds: ["turn-002"] },
  { query: "quarantined lots duplicate ASN inventory reconciliation", expectedIds: ["turn-004"] },
  { query: "first reply latency context hit stale context correction", expectedIds: ["turn-006"] },
  { query: "admin impersonation immutable audit event reason expiration", expectedIds: ["turn-007"] },
  { query: "route optimizer recommended carrier final carrier delivered shipments", expectedIds: ["turn-009"] },
  { query: "warehouse release event acknowledged quarantine counts", expectedIds: ["turn-010"] },
  { query: "monthly small-balance billing adjustments invoice noise", expectedIds: ["turn-011"] },
];

export function runReplay(strategy: CompressionStrategy = "semantic"): object {
  const window = new RollingMemoryWindow({
    maxActiveTokens: 190,
    archiveBatchSize: 3,
    compressionStrategy: strategy,
    compressionRatio: 0.42,
  });

  for (const item of replayDataset) {
    window.add(item.text, item.tags, Number(item.id.replace("turn-", "")), item.id);
  }

  const snapshot = window.snapshot();
  const beforeDocs: RetrievalDocument[] = replayDataset.map((item) => ({
    id: item.id,
    text: item.text,
    tokenCount: item.text.length,
    source: "baseline",
  }));
  const beforeEngine = new RetrievalEngine(beforeDocs);
  const afterEngine = RetrievalEngine.fromWindow(snapshot.activeContext, snapshot.archivedContext);
  const modes: RetrievalMode[] = ["keyword", "semantic", "hybrid"];
  const compressions = snapshot.archivedContext.map((item) => item.compression);
  const metrics = MemoryMetrics.buildReport({ beforeEngine, afterEngine, queries, compressions, modes });

  return {
    generatedAt: new Date().toISOString(),
    experiment: "phase-2-memory-agent",
    strategy,
    window: {
      activeItems: snapshot.activeContext.length,
      archivedItems: snapshot.archivedContext.length,
      activeTokens: snapshot.activeTokens,
      archivedTokens: snapshot.archivedTokens,
      totalTokens: snapshot.totalTokens,
    },
    archivedContext: snapshot.archivedContext.map((item) => ({
      id: item.id,
      strategy: item.strategy,
      sourceIds: item.sourceIds,
      tokens: item.tokenCount,
      text: item.text,
    })),
    metrics,
  };
}

export function runReplaySuite(): object {
  const strategies: CompressionStrategy[] = ["naive", "semantic", "bullets"];
  return {
    generatedAt: new Date().toISOString(),
    experiment: "phase-2-memory-agent",
    resultPath: "logs/memory-agent-results.json",
    runs: strategies.map((strategy) => runReplay(strategy)),
  };
}

export function writeReplayResults(outputPath = "logs/memory-agent-results.json"): object {
  const report = runReplaySuite();
  const resolved = path.resolve(process.cwd(), outputPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  return report;
}

const isMain = process.argv[1] === fileURLToPath(import.meta.url);

if (isMain) {
  const report = writeReplayResults();
  console.log(JSON.stringify(report, null, 2));
}
