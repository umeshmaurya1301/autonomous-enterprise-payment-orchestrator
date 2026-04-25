"use client";
import { Trophy, RotateCcw, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import type { EpisodeStats, TaskDifficulty } from "@/lib/types";

const THRESHOLDS: Record<TaskDifficulty, number> = {
  easy: 0.75,
  medium: 0.45,
  hard: 0.30,
};

interface EpisodeDoneOverlayProps {
  stats: EpisodeStats;
  onDismiss: () => void;
  onRestart: (task: TaskDifficulty) => void;
}

export function EpisodeDoneOverlay({ stats, onDismiss, onRestart }: EpisodeDoneOverlayProps) {
  const avgReward = stats.steps > 0 ? stats.totalReward / stats.steps : 0;
  const threshold = THRESHOLDS[stats.task];
  const passed = avgReward >= threshold;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="bg-[#161b27] border border-[#1e2535] rounded-2xl p-8 w-full max-w-md shadow-2xl flex flex-col gap-6">

        {/* Title */}
        <div className="flex items-center gap-3">
          <div className={cn(
            "p-2.5 rounded-xl border",
            passed ? "bg-green-500/10 border-green-500/30" : "bg-red-500/10 border-red-500/30"
          )}>
            <Trophy className={cn("w-5 h-5", passed ? "text-green-400" : "text-red-400")} />
          </div>
          <div>
            <div className="text-base font-mono font-bold text-slate-200">Episode Complete</div>
            <div className="text-xs font-mono text-slate-500">RL Simulation · AEPO Environment</div>
          </div>
          <div className={cn(
            "ml-auto px-3 py-1 rounded-full border text-xs font-mono font-semibold",
            passed ? "bg-green-500/15 border-green-500/30 text-green-400" : "bg-red-500/10 border-red-500/30 text-red-400"
          )}>
            {passed ? "PASS" : "FAIL"}
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 gap-3">
          <StatBox label="Task" value={stats.task.toUpperCase()} accent="text-blue-400" />
          <StatBox label="Steps" value={String(stats.steps)} accent="text-cyan-400" />
          <StatBox label="Total Reward" value={stats.totalReward.toFixed(3)} accent={stats.totalReward >= 0 ? "text-green-400" : "text-red-400"} />
          <StatBox label="Avg Reward/Step" value={avgReward.toFixed(3)} accent={avgReward >= threshold ? "text-green-400" : "text-red-400"} />
          <StatBox label="Final Phase" value={stats.finalPhase} accent={stats.finalPhase === "cascade" ? "text-red-400" : stats.finalPhase === "spike" ? "text-orange-400" : "text-slate-300"} />
          <StatBox label="Curriculum" value={`Level ${stats.curriculumLevel}`} accent="text-purple-400" />
        </div>

        {/* Threshold bar */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-500">
            <span>Avg Reward vs Threshold ({threshold})</span>
            <RewardDelta avg={avgReward} threshold={threshold} />
          </div>
          <div className="h-2 bg-[#0f1117] rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-700", passed ? "bg-green-400" : "bg-red-400")}
              style={{ width: `${Math.min(100, (avgReward / threshold) * 100)}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] font-mono text-slate-700">
            <span>0</span>
            <span className="text-slate-500">threshold {threshold}</span>
            <span>1.0</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => { onRestart(stats.task); onDismiss(); }}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono font-semibold hover:bg-blue-500/20 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Restart Same Task
          </button>
          <button
            onClick={onDismiss}
            className="px-4 py-2.5 rounded-lg bg-[#0f1117] border border-[#1e2535] text-slate-500 text-xs font-mono hover:text-slate-300 hover:border-[#2a3448] transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="bg-[#0f1117] border border-[#1e2535] rounded-lg px-3 py-2.5 flex flex-col gap-0.5">
      <span className="text-[10px] text-slate-600 font-mono uppercase">{label}</span>
      <span className={cn("text-sm font-mono font-semibold", accent)}>{value}</span>
    </div>
  );
}

function RewardDelta({ avg, threshold }: { avg: number; threshold: number }) {
  const delta = avg - threshold;
  if (Math.abs(delta) < 0.001)
    return <span className="flex items-center gap-1 text-slate-400"><Minus className="w-3 h-3" />{delta.toFixed(3)}</span>;
  if (delta > 0)
    return <span className="flex items-center gap-1 text-green-400"><TrendingUp className="w-3 h-3" />+{delta.toFixed(3)}</span>;
  return <span className="flex items-center gap-1 text-red-400"><TrendingDown className="w-3 h-3" />{delta.toFixed(3)}</span>;
}
