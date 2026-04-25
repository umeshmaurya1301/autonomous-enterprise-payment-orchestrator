"use client";
import { cn } from "@/lib/utils";
import type { AEPOObservation, ObsHistory } from "@/lib/types";
import { Tooltip, SignalTooltip } from "@/components/ui/Tooltip";

interface SignalConfig {
  key: keyof AEPOObservation;
  label: string;
  max: number;
  thresholds: { warn: number; crit: number };
  format: (v: number) => string;
  tooltip: {
    title: string;
    range: string;
    description: string;
    warnAt?: string;
    critAt?: string;
    tip?: string;
  };
}

const SIGNALS: SignalConfig[] = [
  {
    key: "channel",
    label: "Payment Channel",
    max: 2,
    thresholds: { warn: 99, crit: 99 },
    format: (v) => ["P2P", "P2M", "AutoPay"][Math.round(v)] ?? "?",
    tooltip: {
      title: "Payment Channel",
      range: "0=P2P, 1=P2M, 2=AutoPay",
      description: "Simulated payment rail type for the current transaction. P2P = person-to-person, P2M = person-to-merchant, AutoPay = automated recurring.",
      tip: "Channel affects which routing strategy is most rewarded.",
    },
  },
  {
    key: "risk_score",
    label: "Risk Score [sim]",
    max: 100,
    thresholds: { warn: 60, crit: 80 },
    format: (v) => `${Math.round(v)} / 100`,
    tooltip: {
      title: "Fraud Risk Score",
      range: "0 – 100",
      description: "Simulated fraud probability for the current transaction. Computed from adversary behavior and system state.",
      warnAt: "> 60 — elevated fraud signal",
      critAt: "> 80 — Reject/Challenge strongly preferred",
      tip: "Approving when score > 80 gives -0.3 reward.",
    },
  },
  {
    key: "adversary_threat_level",
    label: "Adversary Threat [sim]",
    max: 10,
    thresholds: { warn: 5, crit: 8 },
    format: (v) => `${v.toFixed(1)} / 10`,
    tooltip: {
      title: "Adversary Threat Level",
      range: "0 – 10",
      description: "Adversary escalation in the AEPO multi-agent sim. Rises as the agent performs well — the env 'pushes back' to test robustness.",
      warnAt: "> 5 — adversary actively probing",
      critAt: "> 8 — maximum escalation",
      tip: "Curriculum Level auto-increments as the agent overcomes escalating threats.",
    },
  },
  {
    key: "system_entropy",
    label: "System Entropy [sim]",
    max: 100,
    thresholds: { warn: 50, crit: 70 },
    format: (v) => `${Math.round(v)} / 100`,
    tooltip: {
      title: "System Entropy Index",
      range: "0 – 100",
      description: "Chaos index measuring system instability. Above 70 triggers a latency spike in the next step — the API latency sim jumps sharply.",
      warnAt: "> 50 — instability building",
      critAt: "> 70 — latency spike incoming next step",
      tip: "Use Throttle routing to reduce entropy when this rises.",
    },
  },
  {
    key: "kafka_lag",
    label: "Kafka Lag [sim]",
    max: 10000,
    thresholds: { warn: 3000, crit: 4000 },
    format: (v) => `${(v / 1000).toFixed(1)}k msgs`,
    tooltip: {
      title: "Kafka Consumer Lag",
      range: "0 – 10,000 messages",
      description: "Simulated message queue backlog. Above 3000 causes latency compounding. Above 4000 is the crash threshold — the env enters cascade phase.",
      warnAt: "> 3000 — latency compounding active",
      critAt: "> 4000 — CRASH THRESHOLD, cascade phase",
      tip: "CircuitBreaker routing reduces lag fastest but costs throughput penalty.",
    },
  },
  {
    key: "api_latency",
    label: "API Latency [sim]",
    max: 5000,
    thresholds: { warn: 500, crit: 800 },
    format: (v) => `${Math.round(v)} ms`,
    tooltip: {
      title: "Downstream API Latency",
      range: "0 – 5,000 ms",
      description: "Raw simulated latency for the current step. Spikes occur during entropy > 70 or kafka cascade phases. This feeds into the EMA for Rolling P99.",
      warnAt: "> 500ms — approaching SLA limit",
      critAt: "> 800ms — SLA breach contributing to P99",
      tip: "This is instantaneous. Rolling P99 is the actual penalized metric.",
    },
  },
  {
    key: "rolling_p99",
    label: "Rolling P99 [ema]",
    max: 5000,
    thresholds: { warn: 500, crit: 800 },
    format: (v) => `${Math.round(v)} ms`,
    tooltip: {
      title: "Rolling P99 Latency (EMA)",
      range: "0 – 5,000 ms",
      description: "Exponential moving average of latency: 0.8 × prev + 0.2 × api_latency. This is the SLA training signal. The true P99 from a 20-step ring buffer is in info['true_p99'].",
      warnAt: "> 500ms — warming toward breach",
      critAt: "> 800ms — SLA BREACH, -1.0 reward penalty per step",
      tip: "EMA means a single spike doesn't immediately breach SLA — sustained high latency does.",
    },
  },
  {
    key: "db_connection_pool",
    label: "DB Pool Util [sim]",
    max: 100,
    thresholds: { warn: 65, crit: 80 },
    format: (v) => `${Math.round(v)} %`,
    tooltip: {
      title: "DB Connection Pool Utilization",
      range: "0 – 100%",
      description: "Simulated database connection pool usage. Above 80% triggers retry overhead penalty. DB Retry Policy setting affects how this is handled.",
      warnAt: "> 65% — monitor closely",
      critAt: "> 80% — ExponentialBackoff retry policy adds -0.10 reward",
      tip: "Use FailFast policy when DB pool is healthy; switch to ExpBackoff only above 80%.",
    },
  },
  {
    key: "bank_api_status",
    label: "Bank Sim Status",
    max: 2,
    thresholds: { warn: 0.9, crit: 1.9 },
    format: (v) => {
      const s = Math.round(v);
      return s === 0 ? "Healthy (sim)" : s === 1 ? "Degraded (sim)" : "Unknown (sim)";
    },
    tooltip: {
      title: "Bank API Status (Simulation)",
      range: "0=Healthy, 1=Degraded, 2=Unknown",
      description: "Purely simulated bank connection status. No real bank API is called — this is an env signal that affects settlement policy reward bonuses.",
      tip: "When Degraded, DeferredAsync settlement policy gives +0.05 reward over StandardSync.",
    },
  },
  {
    key: "merchant_tier",
    label: "Merchant Tier [sim]",
    max: 1,
    thresholds: { warn: 99, crit: 99 },
    format: (v) => (v >= 0.5 ? "Enterprise (1)" : "Small (0)"),
    tooltip: {
      title: "Merchant Tier",
      range: "0=Small, 1=Enterprise",
      description: "Simulated merchant classification. Enterprise merchants get a +0.02 reward bonus when App Priority is set to 'UPI'. Small merchants prefer 'Balanced'.",
      tip: "Match App Priority to tier for a small but consistent bonus each step.",
    },
  },
];

// Inline SVG sparkline — no recharts needed for this tiny component
function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const W = 52, H = 16;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - ((v - min) / range) * H;
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width={W} height={H} className="shrink-0 opacity-70">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// Delta arrow — shows change from last observation
function Delta({ current, prev, max }: { current: number; prev: number | null; max: number }) {
  if (prev === null) return null;
  const diff = current - prev;
  if (Math.abs(diff) < max * 0.005) return null; // ignore tiny jitter
  const up = diff > 0;
  const pct = Math.abs((diff / max) * 100);
  const color = up ? "text-red-400" : "text-green-400";
  return (
    <span className={cn("text-[9px] font-mono", color)}>
      {up ? "▲" : "▼"} {pct < 1 ? pct.toFixed(1) : Math.round(pct)}%
    </span>
  );
}

interface ObservationGridProps {
  obs: AEPOObservation;
  prevObs: AEPOObservation | null;
  obsHistory: ObsHistory;
  phase?: string;
  curriculumLevel?: number;
}

export function ObservationGrid({ obs, prevObs, obsHistory, phase = "—", curriculumLevel = 0 }: ObservationGridProps) {
  return (
    <div className="flex flex-col gap-3 bg-[#161b27] border border-[#1e2535] rounded-xl p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">
          Observation Space · 10 Signals
        </h3>
        <div className="flex items-center gap-2">
          <PhaseTag phase={phase} />
          <Tooltip content={<SignalTooltip title="Curriculum Level" range="0–2" description="Auto-promoted by env when recent mean reward exceeds threshold. Level 0=Easy, 1=Medium, 2=Hard internal conditions." />} side="left">
            <span className="text-[10px] text-purple-400 font-mono cursor-help">L{curriculumLevel}</span>
          </Tooltip>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        {SIGNALS.map((sig) => {
          const raw = obs[sig.key] as number;
          const prev = prevObs ? (prevObs[sig.key] as number) : null;
          const norm = Math.min(1, Math.max(0, raw / sig.max));
          const isCrit = raw >= sig.thresholds.crit;
          const isWarn = !isCrit && raw >= sig.thresholds.warn;
          const barColor = isCrit ? "#ef4444" : isWarn ? "#f59e0b" : "#3b82f6";
          const sparkColor = isCrit ? "#ef4444" : isWarn ? "#f59e0b" : "#3b82f6";

          return (
            <Tooltip key={sig.key} content={<SignalTooltip {...sig.tooltip} />} side="right">
              <div className="flex flex-col gap-1.5 w-full cursor-help">
                <div className="flex items-center justify-between gap-1">
                  <span className={cn("text-[11px] font-mono truncate", isCrit ? "text-red-400" : isWarn ? "text-yellow-400" : "text-slate-400")}>
                    {sig.label}
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <Delta current={raw} prev={prev} max={sig.max} />
                    <span className="text-[11px] font-mono font-semibold" style={{ color: barColor }}>
                      {sig.format(raw)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1 bg-[#0f1117] rounded-full overflow-hidden">
                    <div
                      className={cn("h-full rounded-full transition-all duration-500", isCrit && "animate-pulse")}
                      style={{ width: `${norm * 100}%`, backgroundColor: barColor }}
                    />
                  </div>
                  <Sparkline values={obsHistory[sig.key] ?? []} color={sparkColor} />
                </div>
              </div>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
}

function PhaseTag({ phase }: { phase: string }) {
  const color =
    phase === "cascade" ? "bg-red-500/15 text-red-400 border-red-500/30" :
    phase === "spike"   ? "bg-orange-500/15 text-orange-400 border-orange-500/30" :
    phase === "normal"  ? "bg-green-500/10 text-green-400 border-green-500/20" :
                          "bg-slate-700/30 text-slate-500 border-slate-600/20";
  return (
    <Tooltip
      content={<SignalTooltip
        title="Simulation Phase"
        range="normal | spike | cascade"
        description="Current phase of the AEPO environment. Phases drive different reward dynamics and signal distributions."
        tip="Cascade phase = Kafka > 4000 + adversary escalation. Spike phase = entropy surge or latency jump."
      />}
      side="left"
    >
      <span className={cn("text-[10px] font-mono px-2 py-0.5 rounded border cursor-help", color)}>
        {phase}
      </span>
    </Tooltip>
  );
}
