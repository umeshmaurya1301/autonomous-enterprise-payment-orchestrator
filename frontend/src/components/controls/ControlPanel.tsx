"use client";
import { useState } from "react";
import { RotateCcw, Play, Pause, Send, ChevronDown, Download, Zap, ShieldX, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import { Tooltip, SignalTooltip } from "@/components/ui/Tooltip";
import type { AEPOAction, TaskDifficulty, ActionLogEntry } from "@/lib/types";

interface ControlPanelProps {
  onReset: (task: TaskDifficulty) => void;
  onStep: (action: AEPOAction) => void;
  isResetting: boolean;
  isRunning: boolean;
  onToggleAutoRun: () => void;
  episodeActive: boolean;
  autoRunInterval: number;
  onIntervalChange: (ms: number) => void;
  actionLog: ActionLogEntry[];
}

const TASK_OPTIONS: { value: TaskDifficulty; label: string; color: string }[] = [
  { value: "easy",   label: "Easy",   color: "text-green-400" },
  { value: "medium", label: "Medium", color: "text-yellow-400" },
  { value: "hard",   label: "Hard",   color: "text-red-400" },
];

const SPEED_OPTIONS = [
  { label: "Slow",   ms: 1200 },
  { label: "Normal", ms: 600  },
  { label: "Fast",   ms: 200  },
  { label: "Turbo",  ms: 50   },
];

const SCENARIO_PRESETS: { label: string; icon: React.ReactNode; action: AEPOAction; description: string; color: string }[] = [
  {
    label: "Kafka Crisis",
    icon: <Zap className="w-3 h-3" />,
    description: "CircuitBreaker + FailFast — sheds kafka lag fastest",
    color: "border-red-500/40 text-red-400 bg-red-500/10 hover:bg-red-500/20",
    action: { risk_decision: 1, crypto_verify: 1, infra_routing: 2, db_retry_policy: 0, settlement_policy: 1, app_priority: 2 },
  },
  {
    label: "Fraud Spike",
    icon: <ShieldX className="w-3 h-3" />,
    description: "Reject + FullVerify — max fraud defense",
    color: "border-orange-500/40 text-orange-400 bg-orange-500/10 hover:bg-orange-500/20",
    action: { risk_decision: 1, crypto_verify: 0, infra_routing: 0, db_retry_policy: 0, settlement_policy: 0, app_priority: 2 },
  },
  {
    label: "SLA Recovery",
    icon: <Activity className="w-3 h-3" />,
    description: "Approve + Throttle — reduces P99 without full CB",
    color: "border-cyan-500/40 text-cyan-400 bg-cyan-500/10 hover:bg-cyan-500/20",
    action: { risk_decision: 0, crypto_verify: 1, infra_routing: 1, db_retry_policy: 0, settlement_policy: 1, app_priority: 2 },
  },
];

const ACTION_FIELDS: {
  key: keyof AEPOAction;
  label: string;
  options: { value: number; label: string }[];
  tooltip: string;
}[] = [
  {
    key: "risk_decision",
    label: "Risk Decision",
    tooltip: "Core reward driver. Approve on low-risk = +0.8. Reject on high-risk = +0.8. Mismatch = -0.3.",
    options: [{ value: 0, label: "0 · Approve" }, { value: 1, label: "1 · Reject" }, { value: 2, label: "2 · Challenge" }],
  },
  {
    key: "crypto_verify",
    label: "Crypto Verify",
    tooltip: "FullVerify adds latency but is safe. SkipVerify is faster but gives -0.3 if risk_score > 50.",
    options: [{ value: 0, label: "0 · FullVerify" }, { value: 1, label: "1 · SkipVerify" }],
  },
  {
    key: "infra_routing",
    label: "Infra Routing",
    tooltip: "Normal = default throughput. Throttle = reduces kafka lag slowly. CircuitBreaker = fastest lag reduction, -0.1 throughput penalty.",
    options: [{ value: 0, label: "0 · Normal" }, { value: 1, label: "1 · Throttle" }, { value: 2, label: "2 · CircuitBreaker" }],
  },
  {
    key: "db_retry_policy",
    label: "DB Retry",
    tooltip: "FailFast = default, no penalty. ExpBackoff = -0.10 penalty but stabilizes DB pool when usage > 80%.",
    options: [{ value: 0, label: "0 · FailFast" }, { value: 1, label: "1 · ExpBackoff" }],
  },
  {
    key: "settlement_policy",
    label: "Settlement",
    tooltip: "StandardSync = default. DeferredAsync = +0.05 bonus when bank sim status is Degraded.",
    options: [{ value: 0, label: "0 · StandardSync" }, { value: 1, label: "1 · DeferredAsync" }],
  },
  {
    key: "app_priority",
    label: "App Priority",
    tooltip: "Match to merchant tier for +0.02/step bonus. Enterprise → UPI. Small → Balanced.",
    options: [{ value: 0, label: "0 · UPI" }, { value: 1, label: "1 · Credit" }, { value: 2, label: "2 · Balanced" }],
  },
];

const DEFAULT_ACTION: AEPOAction = {
  risk_decision: 0, crypto_verify: 0, infra_routing: 0,
  db_retry_policy: 0, settlement_policy: 0, app_priority: 2,
};

function exportCSV(log: ActionLogEntry[]) {
  if (log.length === 0) return;
  const headers = ["step", "timestamp", "phase", "reward", "risk_decision", "crypto_verify", "infra_routing", "db_retry_policy", "settlement_policy", "app_priority"];
  const rows = log.map((e) => [
    e.step, e.timestamp, e.phase, e.reward.toFixed(4),
    e.action.risk_decision, e.action.crypto_verify, e.action.infra_routing,
    e.action.db_retry_policy, e.action.settlement_policy, e.action.app_priority,
  ].join(","));
  const csv = [headers.join(","), ...rows.reverse()].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `aepo_episode_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function ControlPanel({
  onReset, onStep, isResetting, isRunning, onToggleAutoRun,
  episodeActive, autoRunInterval, onIntervalChange, actionLog,
}: ControlPanelProps) {
  const [task, setTask] = useState<TaskDifficulty>("easy");
  const [action, setAction] = useState<AEPOAction>(DEFAULT_ACTION);

  return (
    <div className="flex flex-col gap-4 bg-[#161b27] border border-[#1e2535] rounded-xl p-5">
      <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">
        Control Panel · Manual Override
      </h3>

      {/* Row 1: Reset + Auto Run + Speed */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative">
          <select
            value={task}
            onChange={(e) => setTask(e.target.value as TaskDifficulty)}
            className="appearance-none bg-[#0f1117] border border-[#1e2535] text-slate-300 text-xs font-mono px-3 py-2 pr-7 rounded-lg focus:outline-none focus:border-blue-500 cursor-pointer"
          >
            {TASK_OPTIONS.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-500 pointer-events-none" />
        </div>

        <button
          onClick={() => onReset(task)}
          disabled={isResetting}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono font-semibold transition-all duration-200 border",
            isResetting
              ? "bg-slate-700/40 border-slate-600/30 text-slate-500 cursor-not-allowed"
              : "bg-orange-500/10 border-orange-500/30 text-orange-400 hover:bg-orange-500/20"
          )}
        >
          <RotateCcw className={cn("w-3.5 h-3.5", isResetting && "animate-spin")} />
          {isResetting ? "Resetting…" : "Reset"}
        </button>

        <button
          onClick={onToggleAutoRun}
          disabled={!episodeActive}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono font-semibold transition-all duration-200 border",
            !episodeActive
              ? "bg-slate-700/30 border-slate-600/20 text-slate-600 cursor-not-allowed"
              : isRunning
              ? "bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20"
              : "bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20"
          )}
        >
          {isRunning ? <><Pause className="w-3.5 h-3.5" />Stop</> : <><Play className="w-3.5 h-3.5" />Auto Run</>}
        </button>

        {/* Speed selector */}
        <div className="flex items-center gap-1.5 bg-[#0f1117] border border-[#1e2535] rounded-lg px-2 py-1.5">
          <span className="text-[10px] text-slate-600 font-mono">Speed:</span>
          {SPEED_OPTIONS.map((opt) => (
            <Tooltip key={opt.label} content={`Fire action every ${opt.ms}ms`} side="top">
              <button
                onClick={() => onIntervalChange(opt.ms)}
                className={cn(
                  "px-2 py-0.5 rounded text-[10px] font-mono font-semibold transition-colors",
                  autoRunInterval === opt.ms
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "text-slate-600 hover:text-slate-400"
                )}
              >
                {opt.label}
              </button>
            </Tooltip>
          ))}
        </div>

        {/* Export CSV */}
        <Tooltip content="Download full episode step log as CSV" side="top">
          <button
            onClick={() => exportCSV(actionLog)}
            disabled={actionLog.length === 0}
            className={cn(
              "flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-mono border transition-colors ml-auto",
              actionLog.length === 0
                ? "border-slate-700/30 text-slate-700 cursor-not-allowed"
                : "border-slate-600/30 text-slate-500 hover:text-slate-300 hover:border-slate-500/50"
            )}
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </button>
        </Tooltip>
      </div>

      {/* Row 2: Scenario Presets */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-slate-600 font-mono uppercase shrink-0">Stress presets:</span>
        {SCENARIO_PRESETS.map((preset) => (
          <Tooltip key={preset.label} content={preset.description} side="top">
            <button
              onClick={() => { setAction(preset.action); onStep(preset.action); }}
              disabled={!episodeActive || isRunning}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-mono font-semibold border transition-all",
                (!episodeActive || isRunning)
                  ? "border-slate-700/20 text-slate-700 bg-transparent cursor-not-allowed"
                  : preset.color
              )}
            >
              {preset.icon}
              {preset.label}
            </button>
          </Tooltip>
        ))}
      </div>

      <div className="border-t border-[#1e2535]" />

      {/* 6-dim action form */}
      <div className="grid grid-cols-3 gap-3">
        {ACTION_FIELDS.map((field) => (
          <Tooltip key={field.key} content={field.tooltip} side="top">
            <div className="flex flex-col gap-1 w-full cursor-help">
              <label className="text-[10px] text-slate-500 font-mono uppercase">{field.label}</label>
              <div className="relative">
                <select
                  value={action[field.key]}
                  onChange={(e) => setAction((prev) => ({ ...prev, [field.key]: Number(e.target.value) }))}
                  className="w-full appearance-none bg-[#0f1117] border border-[#1e2535] text-slate-300 text-[11px] font-mono px-2.5 py-1.5 pr-6 rounded-lg focus:outline-none focus:border-blue-500 cursor-pointer"
                >
                  {field.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-600 pointer-events-none" />
              </div>
            </div>
          </Tooltip>
        ))}
      </div>

      {/* Send button + preview */}
      <button
        onClick={() => onStep(action)}
        disabled={!episodeActive || isRunning}
        className={cn(
          "flex items-center justify-center gap-2 w-full py-2.5 rounded-lg text-xs font-mono font-semibold transition-all duration-200 border",
          !episodeActive || isRunning
            ? "bg-slate-700/30 border-slate-600/20 text-slate-600 cursor-not-allowed"
            : "bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20"
        )}
      >
        <Send className="w-3.5 h-3.5" />
        Send Action · POST /step
      </button>

      <div className="bg-[#0f1117] rounded-lg px-3 py-2 border border-[#1e2535]">
        <span className="text-[10px] text-slate-600 font-mono">
          payload: <span className="text-slate-400">
            [{action.risk_decision},{action.crypto_verify},{action.infra_routing},
            {action.db_retry_policy},{action.settlement_policy},{action.app_priority}]
          </span>
        </span>
      </div>
    </div>
  );
}
