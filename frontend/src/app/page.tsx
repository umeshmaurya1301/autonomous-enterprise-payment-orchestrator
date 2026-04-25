"use client";
import { useState, useEffect, useRef } from "react";
import { Zap, Circle, Wifi, WifiOff, Brain, Layers, Hash } from "lucide-react";
import { useAEPO } from "@/hooks/useAEPO";
import { RiskTriad } from "@/components/metrics/RiskTriad";
import { ObservationGrid } from "@/components/metrics/ObservationGrid";
import { RewardChart } from "@/components/metrics/RewardChart";
import { QTableHeatmap } from "@/components/metrics/QTableHeatmap";
import { InfraChart, type InfraPoint } from "@/components/metrics/InfraChart";
import { RewardBreakdownBar } from "@/components/metrics/RewardBreakdownBar";
import { CurriculumProgress } from "@/components/metrics/CurriculumProgress";
import { EpisodeHistory } from "@/components/feed/EpisodeHistory";
import { LiveActionFeed } from "@/components/feed/LiveActionFeed";
import { ControlPanel } from "@/components/controls/ControlPanel";
import { ToastContainer } from "@/components/ui/ToastNotification";
import { EpisodeDoneOverlay } from "@/components/ui/EpisodeDoneOverlay";
import { KafkaCrisisAlert } from "@/components/ui/KafkaCrisisAlert";
import { LiveClock } from "@/components/ui/LiveClock";
import { EmptyPanel } from "@/components/ui/EmptyPanel";
import { InfoBadge } from "@/components/ui/InfoBadge";
import { GlossaryPanel } from "@/components/ui/GlossaryPanel";
import { ShieldAlert, Server, Timer, Grid3x3, BarChart2, Table2, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";

const MAX_INFRA = 80;

export default function DashboardPage() {
  const {
    observation, prevObservation, obsHistory,
    rewardHistory, phaseHistory,
    actionLog, notifications, episodeHistory,
    isRunning, episodeActive, episodeDone, episodeStats,
    currentStep, cumulativeReward, isResetting,
    lastReward, lastRewardBreakdown,
    currentTask, currentPhase, curriculumLevel, episodeCount,
    autoRunInterval,
    reset, step, toggleAutoRun, dismissNotification, dismissDone,
    setAutoRunInterval,
  } = useAEPO();

  const [infraHistory, setInfraHistory] = useState<InfraPoint[]>([]);
  const infraStepRef = useRef(0);
  const [glossaryOpen, setGlossaryOpen] = useState(false);

  useEffect(() => {
    if (!observation) return;
    infraStepRef.current += 1;
    setInfraHistory((prev) => [
      ...prev.slice(-MAX_INFRA + 1),
      { step: infraStepRef.current, kafkaLag: observation.kafka_lag / 100, p99: observation.rolling_p99, dbPool: observation.db_connection_pool },
    ]);
  }, [observation]);

  const kafkaLag = observation?.kafka_lag ?? 0;
  const kafkaCritical = kafkaLag > 4000;
  const kafkaWarn = !kafkaCritical && kafkaLag > 3000;
  const slaBreach = observation && observation.rolling_p99 > 800;

  // "Apply CircuitBreaker" quick fix for kafka crisis
  const handleKafkaFix = () => {
    if (!episodeActive) return;
    step({ risk_decision: 1, crypto_verify: 1, infra_routing: 2, db_retry_policy: 0, settlement_policy: 1, app_priority: 2 });
  };

  const handleReset = (task: Parameters<typeof reset>[0]) => {
    infraStepRef.current = 0;
    setInfraHistory([]);
    reset(task);
  };

  return (
    <div className={cn(
      "min-h-screen bg-[#0f1117] flex flex-col transition-all duration-500",
      kafkaCritical && "ring-1 ring-inset ring-red-500/20"
    )}>

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#1e2535] bg-[#0f1117]/95 backdrop-blur sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-blue-400" />
            <span className="text-sm font-mono font-bold text-slate-200 tracking-wide">AEPO</span>
            <span className="text-xs font-mono text-slate-500">RL Simulation Dashboard</span>
          </div>
          <div className="h-4 w-px bg-[#1e2535]" />
          <StatusPill active={episodeActive} />
        </div>
        <div className="flex items-center gap-4 text-[11px] font-mono">
          {kafkaCritical && (
            <span className="flex items-center gap-1.5 text-red-400 animate-pulse font-semibold">
              <Circle className="w-2 h-2 fill-red-400" />KAFKA CRASH
            </span>
          )}
          {kafkaWarn && (
            <span className="flex items-center gap-1.5 text-orange-400">
              <Circle className="w-2 h-2 fill-orange-400" />KAFKA WARNING
            </span>
          )}
          {slaBreach && (
            <span className="flex items-center gap-1.5 text-red-400 animate-pulse">
              <Circle className="w-2 h-2 fill-red-400" />SLA BREACH
            </span>
          )}
          <LiveClock />
          <button
            onClick={() => setGlossaryOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-[#1e2535] text-slate-400 hover:text-blue-400 hover:border-blue-500/40 hover:bg-blue-500/5 transition-all"
            title="Open Metrics Guide (explain numbers)"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span className="text-[10px]">Guide</span>
          </button>
        </div>
      </header>

      {/* ── Kafka Crisis Alert Banner ── */}
      <KafkaCrisisAlert
        kafkaLag={kafkaLag}
        onFix={episodeActive ? handleKafkaFix : undefined}
      />

      {/* ── Episode Summary Bar ── */}
      <div className="flex flex-col gap-1.5 px-6 py-2 border-b border-[#1e2535] bg-[#0d1019]">
        <div className="flex items-center gap-5 text-[11px] font-mono overflow-x-auto">
          <EpisodeStat icon={<Layers className="w-3 h-3" />} label="Task" value={currentTask.toUpperCase()} color={taskColor(currentTask)} />
          <EpisodeStat icon={<Hash className="w-3 h-3" />} label="Episode" value={String(episodeCount)} color="text-slate-300" />
          <EpisodeStat icon={<Zap className="w-3 h-3" />} label="Step" value={String(currentStep)} color="text-blue-400" />
          <EpisodeStat label="Phase" value={currentPhase || "—"} color={phaseColor(currentPhase)} />
          <EpisodeStat label="Cum. Reward" value={cumulativeReward.toFixed(3)} color={cumulativeReward >= 0 ? "text-green-400" : "text-red-400"} />
          <EpisodeStat label="Avg/Step" value={currentStep > 0 ? (cumulativeReward / currentStep).toFixed(3) : "—"} color="text-slate-400" />
          <span className="ml-auto text-slate-700 shrink-0">localhost:7860</span>
        </div>
        <CurriculumProgress level={curriculumLevel} task={currentTask} />
      </div>

      {/* ── Main Grid ── */}
      <main className="flex-1 grid grid-cols-[1fr_288px] gap-0 overflow-hidden">
        <div className="flex flex-col gap-4 p-4 overflow-y-auto">

          {/* Row 1: Risk Triad */}
          {observation ? <RiskTriad obs={observation} /> : (
            <div className="grid grid-cols-3 gap-4">
              <EmptyPanel icon={<ShieldAlert className="w-8 h-8" />} title="Fraud Risk Gauge" description="Shows risk_score (0–100) and adversary_threat_level (0–10) as a live arc gauge. Pulses red when risk > 80." hint="↓ Reset below to populate" height="h-44" />
              <EmptyPanel icon={<Server className="w-8 h-8" />} title="Infra Health Gauge" description="Shows kafka_lag (0–10k msgs) as a live arc. Triggers crash alert and red banner above 4,000." hint="↓ Reset below to populate" height="h-44" />
              <EmptyPanel icon={<Timer className="w-8 h-8" />} title="SLA Compliance Gauge" description="Tracks rolling P99 latency (EMA). SLA threshold is 800ms — breaching it adds -1.0 reward penalty per step." hint="↓ Reset below to populate" height="h-44" />
            </div>
          )}

          {/* Row 2: Observation Grid + Charts */}
          <div className="grid grid-cols-[1fr_1.2fr] gap-4">
            {observation
              ? <ObservationGrid obs={observation} prevObs={prevObservation} obsHistory={obsHistory} phase={currentPhase} curriculumLevel={curriculumLevel} />
              : <EmptyPanel icon={<Grid3x3 className="w-8 h-8" />} title="10-Dim Observation Space" description="All 10 env signals shown as labeled progress bars with delta arrows (▲▼ vs previous step) and 25-step sparklines. Hover any signal for its definition and thresholds." hint="↓ Reset below to populate" height="h-64" />
            }
            <div className="flex flex-col gap-4">
              <RewardChart data={rewardHistory} cumulativeReward={cumulativeReward} phases={phaseHistory} />
              <InfraChart data={infraHistory} />
            </div>
          </div>

          {/* Row 3: Reward Breakdown + Q-Table */}
          <div className="grid grid-cols-2 gap-4">
            <RewardBreakdownBar breakdown={lastRewardBreakdown} totalReward={lastReward} />
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">Q-Table</span>
                <InfoBadge title="Q-Table Heatmap" lines={["See the heatmap panel below for full explanation."]} side="right" />
              </div>
              <QTableHeatmap />
            </div>
          </div>

          {/* Row 4: Episode History */}
          <EpisodeHistory entries={episodeHistory} />

          {/* Row 5: Control Panel */}
          <ControlPanel
            onReset={handleReset}
            onStep={step}
            isResetting={isResetting}
            isRunning={isRunning}
            onToggleAutoRun={toggleAutoRun}
            episodeActive={episodeActive}
            autoRunInterval={autoRunInterval}
            onIntervalChange={setAutoRunInterval}
            actionLog={actionLog}
          />
        </div>

        {/* Right: Step Log */}
        <div className="border-l border-[#1e2535] overflow-hidden h-full">
          <LiveActionFeed
            entries={actionLog}
            currentStep={currentStep}
            lastReward={lastReward}
            currentPhase={currentPhase}
            curriculumLevel={curriculumLevel}
          />
        </div>
      </main>

      <ToastContainer notifications={notifications} onDismiss={dismissNotification} />

      {episodeDone && episodeStats && (
        <EpisodeDoneOverlay
          stats={episodeStats}
          onDismiss={dismissDone}
          onRestart={(task) => handleReset(task)}
        />
      )}

      <GlossaryPanel open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
    </div>
  );
}

function StatusPill({ active }: { active: boolean }) {
  return (
    <div className={cn(
      "flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-mono font-semibold",
      active ? "bg-green-500/10 border-green-500/30 text-green-400" : "bg-slate-700/30 border-slate-600/30 text-slate-500"
    )}>
      {active ? <><Wifi className="w-3 h-3" />RUNNING</> : <><WifiOff className="w-3 h-3" />IDLE</>}
    </div>
  );
}

function EpisodeStat({ icon, label, value, color }: { icon?: React.ReactNode; label: string; value: string; color: string }) {
  return (
    <div className="flex items-center gap-1.5 shrink-0">
      {icon && <span className="text-slate-600">{icon}</span>}
      <span className="text-slate-600">{label}</span>
      <span className={cn("font-semibold", color)}>{value}</span>
    </div>
  );
}

function taskColor(t: string) {
  return t === "easy" ? "text-green-400" : t === "medium" ? "text-yellow-400" : "text-red-400";
}

function phaseColor(p: string) {
  return p === "cascade" ? "text-red-400" : p === "spike" ? "text-orange-400" : p === "normal" ? "text-green-400" : "text-slate-500";
}

