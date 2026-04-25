"use client";
import { useRef, useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  getRiskDecisionLabel,
  getCryptoVerifyLabel,
  getInfraRoutingLabel,
  getDbRetryLabel,
  getSettlementLabel,
} from "@/lib/utils";
import type { ActionLogEntry } from "@/lib/types";

interface LiveActionFeedProps {
  entries: ActionLogEntry[];
  currentStep: number;
  lastReward: number | null;
  currentPhase: string;
  curriculumLevel: number;
}

function rewardColor(r: number): string {
  if (r > 0.5) return "text-green-400";
  if (r > 0) return "text-emerald-400";
  if (r > -0.1) return "text-yellow-400";
  return "text-red-400";
}

function decisionBadge(d: number): string {
  if (d === 0) return "bg-green-500/15 text-green-400 border-green-500/30";
  if (d === 1) return "bg-red-500/15 text-red-400 border-red-500/30";
  return "bg-yellow-500/15 text-yellow-400 border-yellow-500/30";
}

function phaseBadge(phase: string): string {
  if (phase === "cascade") return "text-red-400";
  if (phase === "spike")   return "text-orange-400";
  return "text-slate-600";
}

function RewardBreakdown({ breakdown }: { breakdown: Record<string, number> }) {
  const entries = Object.entries(breakdown).filter(([, v]) => v !== 0);
  if (entries.length === 0) return null;
  return (
    <div className="mt-1.5 pt-1.5 border-t border-[#1e2535] grid grid-cols-2 gap-x-3 gap-y-0.5">
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center justify-between gap-1">
          <span className="text-[9px] text-slate-600 font-mono truncate">{k}</span>
          <span className={cn("text-[9px] font-mono font-semibold", v >= 0 ? "text-emerald-500" : "text-red-500")}>
            {v >= 0 ? "+" : ""}{v.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function LiveActionFeed({
  entries,
  currentStep,
  lastReward,
  currentPhase,
  curriculumLevel,
}: LiveActionFeedProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [entries.length]);

  return (
    <div className="flex flex-col h-full bg-[#161b27] border border-[#1e2535] rounded-none overflow-hidden">
      {/* Header */}
      <div className="flex flex-col gap-1.5 px-4 py-3 border-b border-[#1e2535]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-slate-400 font-mono">
            <Activity className="w-3.5 h-3.5 text-blue-400" />
            <span>Step Log</span>
          </div>
          <div className="flex items-center gap-2 text-[11px] font-mono">
            <span className="text-slate-600">step</span>
            <span className="text-blue-400 font-semibold">{currentStep}</span>
          </div>
        </div>
        {/* Phase + curriculum row */}
        <div className="flex items-center justify-between text-[10px] font-mono">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-600">phase</span>
            <span className={cn("font-semibold", phaseBadge(currentPhase))}>{currentPhase || "—"}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-600">curriculum</span>
            <span className="text-purple-400 font-semibold">L{curriculumLevel}</span>
            {lastReward !== null && (
              <span className={cn("font-semibold", rewardColor(lastReward))}>
                {lastReward >= 0 ? "+" : ""}{lastReward.toFixed(3)}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Entry list */}
      <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
        {entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600 text-xs font-mono">
            <span>No steps yet</span>
            <span className="text-[10px] text-slate-700">Reset and start an episode</span>
          </div>
        ) : (
          entries.map((entry, i) => {
            const expanded = expandedId === entry.id;
            return (
              <div
                key={entry.id}
                onClick={() => setExpandedId(expanded ? null : entry.id)}
                className={cn(
                  "flex flex-col gap-1 p-2.5 rounded-lg border cursor-pointer transition-all duration-150",
                  i === 0
                    ? "bg-blue-500/5 border-blue-500/20 animate-fade-in"
                    : "bg-[#0f1117]/60 border-[#1e2535] hover:border-[#2a3448]"
                )}
              >
                {/* Row 1: step + decision + reward */}
                <div className="flex items-center justify-between gap-1">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className="text-slate-600 text-[10px] font-mono shrink-0">#{entry.step}</span>
                    <span className={cn("text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border shrink-0", decisionBadge(entry.action.risk_decision))}>
                      {getRiskDecisionLabel(entry.action.risk_decision)}
                    </span>
                    <span className="text-[10px] font-mono text-slate-500 truncate">
                      {getCryptoVerifyLabel(entry.action.crypto_verify)}
                    </span>
                  </div>
                  <span className={cn("text-[11px] font-mono font-semibold shrink-0", rewardColor(entry.reward))}>
                    {entry.reward >= 0 ? "+" : ""}{entry.reward.toFixed(3)}
                  </span>
                </div>

                {/* Row 2: infra + phase + time */}
                <div className="flex items-center justify-between text-[10px] font-mono text-slate-600">
                  <span>{getInfraRoutingLabel(entry.action.infra_routing)} · {getDbRetryLabel(entry.action.db_retry_policy)}</span>
                  <div className="flex items-center gap-1.5">
                    {entry.phase && entry.phase !== "normal" && (
                      <span className={phaseBadge(entry.phase)}>{entry.phase}</span>
                    )}
                    <span>{entry.timestamp}</span>
                  </div>
                </div>

                {/* Expanded: reward breakdown */}
                {expanded && <RewardBreakdown breakdown={entry.rewardBreakdown} />}
              </div>
            );
          })
        )}
      </div>

      <div className="px-4 py-2 border-t border-[#1e2535] text-[10px] text-slate-700 font-mono">
        Click any entry to expand reward breakdown
      </div>
    </div>
  );
}
