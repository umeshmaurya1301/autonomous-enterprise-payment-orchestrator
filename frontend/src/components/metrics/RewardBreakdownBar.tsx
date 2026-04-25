"use client";
import { cn } from "@/lib/utils";
import { Tooltip } from "@/components/ui/Tooltip";
import { InfoBadge } from "@/components/ui/InfoBadge";
import { EmptyPanel } from "@/components/ui/EmptyPanel";
import { BarChart2 } from "lucide-react";

interface RewardBreakdownBarProps {
  breakdown: Record<string, number>;
  totalReward: number | null;
}

const COMPONENT_COLORS: Record<string, string> = {
  fraud_reward:      "#22c55e",
  sla_penalty:       "#ef4444",
  adversary_penalty: "#f97316",
  entropy_penalty:   "#a855f7",
  db_retry_penalty:  "#f59e0b",
  throughput_bonus:  "#06b6d4",
  priority_bonus:    "#3b82f6",
  settlement_bonus:  "#10b981",
};

const COMPONENT_DESCRIPTIONS: Record<string, string> = {
  fraud_reward:      "+0.8 for correct risk decision. -0.3 for mismatch (approve on high-risk or reject on low-risk).",
  sla_penalty:       "-1.0 per step when rolling P99 > 800ms. Most impactful penalty.",
  adversary_penalty: "Grows each step as adversary threat level rises. Peaks at -0.5 on max escalation.",
  entropy_penalty:   "Triggered when system entropy > 70. Cascades into latency spike next step.",
  db_retry_penalty:  "-0.10 when ExpBackoff policy is active and DB pool > 80%.",
  throughput_bonus:  "+0.05 bonus for normal routing when kafka lag is low.",
  priority_bonus:    "+0.02 when app_priority matches merchant tier.",
  settlement_bonus:  "+0.05 when DeferredAsync policy is used during bank sim degradation.",
};

function getColor(key: string): string {
  return COMPONENT_COLORS[key] ?? "#64748b";
}

export function RewardBreakdownBar({ breakdown, totalReward }: RewardBreakdownBarProps) {
  const entries = Object.entries(breakdown).filter(([, v]) => Math.abs(v) > 0.001);

  return (
    <div className="flex flex-col gap-2 bg-[#161b27] border border-[#1e2535] rounded-xl p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">Reward Breakdown</h3>
          <InfoBadge
            title="Reward Breakdown"
            lines={[
              "Shows how the last step's total reward was composed from individual components.",
              "Each bar = one reward/penalty signal from the env. Hover a bar to see its definition.",
              "Populated automatically after every POST /step. Fires with manual step or Auto Run.",
            ]}
            side="top"
          />
        </div>
        {totalReward !== null && (
          <span className={cn("text-xs font-mono font-semibold", totalReward >= 0 ? "text-green-400" : "text-red-400")}>
            total {totalReward >= 0 ? "+" : ""}{totalReward.toFixed(3)}
          </span>
        )}
      </div>

      {entries.length === 0 ? (
        <EmptyPanel
          icon={<BarChart2 className="w-8 h-8" />}
          title="No reward data yet"
          description="Each POST /step returns a reward_breakdown dict showing what contributed to the score."
          hint="↓ Send an action via the Control Panel below"
          height="h-28"
        />
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.sort(([, a], [, b]) => Math.abs(b) - Math.abs(a)).map(([key, value]) => {
            const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)));
            const norm = Math.abs(value) / maxAbs;
            const color = getColor(key);
            const desc = COMPONENT_DESCRIPTIONS[key] ?? "Reward signal from env step.";
            return (
              <Tooltip key={key} side="left" content={
                <div className="font-mono text-[11px] flex flex-col gap-1">
                  <span className="text-slate-200 font-semibold">{key}</span>
                  <span style={{ color }}>{value >= 0 ? "+" : ""}{value.toFixed(4)}</span>
                  <span className="text-slate-500 text-[10px] leading-relaxed">{desc}</span>
                </div>
              }>
                <div className="flex items-center gap-2 cursor-help">
                  <span className="text-[10px] font-mono text-slate-500 w-36 truncate">{key}</span>
                  <div className="flex-1 h-1.5 bg-[#0f1117] rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-400"
                      style={{ width: `${norm * 100}%`, backgroundColor: color }} />
                  </div>
                  <span className="text-[10px] font-mono w-14 text-right" style={{ color }}>
                    {value >= 0 ? "+" : ""}{value.toFixed(3)}
                  </span>
                </div>
              </Tooltip>
            );
          })}
        </div>
      )}
    </div>
  );
}
