"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { RewardPoint } from "@/lib/types";
import { PhaseTimeline } from "./PhaseTimeline";
import { InfoBadge } from "@/components/ui/InfoBadge";
import { TrendingUp } from "lucide-react";

interface RewardChartProps {
  data: RewardPoint[];
  cumulativeReward: number;
  phases: string[];
}

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: RewardPoint }> }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-[#0f1117] border border-[#1e2535] rounded p-2 text-[11px] font-mono">
      <div className="text-slate-400">Step {d.step}</div>
      <div className="text-blue-400">Reward: {d.reward.toFixed(3)}</div>
      <div className="text-cyan-400">Cumulative: {d.cumulative.toFixed(3)}</div>
    </div>
  );
};

export function RewardChart({ data, cumulativeReward, phases }: RewardChartProps) {
  return (
    <div className="flex flex-col gap-3 bg-[#161b27] border border-[#1e2535] rounded-xl p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">Live Step Rewards</h3>
          <InfoBadge
            title="Live Step Rewards Chart"
            lines={[
              "Blue line = per-step reward returned by POST /step. Can be noisy.",
              "Dashed cyan line = cumulative reward. A steady upward slope = agent learning well.",
              "Zero reference line helps spot net-negative episodes.",
              "Phase timeline below the chart shows which sim phase was active at each step — correlates reward dips with env conditions.",
            ]}
          />
        </div>
        <span className="text-[11px] font-mono text-slate-500">
          Cumulative:{" "}
          <span className={cumulativeReward >= 0 ? "text-green-400" : "text-red-400"}>
            {cumulativeReward.toFixed(2)}
          </span>
        </span>
      </div>

      {data.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 gap-2">
          <TrendingUp className="w-8 h-8 text-slate-700" />
          <p className="text-slate-600 text-xs font-mono">No reward data yet</p>
          <p className="text-slate-700 text-[10px] font-mono">Step rewards will appear here in real-time as the agent acts</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={130}>
          <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2535" vertical={false} />
            <XAxis dataKey="step" tick={{ fill: "#475569", fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
            <YAxis tick={{ fill: "#475569", fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} width={40} />
            <ReferenceLine y={0} stroke="#334155" strokeDasharray="4 2" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="reward" stroke="#3b82f6" strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: "#3b82f6" }} />
            <Line type="monotone" dataKey="cumulative" stroke="#06b6d4" strokeWidth={1} dot={false} strokeDasharray="4 2" activeDot={{ r: 3, fill: "#06b6d4" }} />
          </LineChart>
        </ResponsiveContainer>
      )}

      <PhaseTimeline phases={phases} />

      <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
        <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 bg-blue-400" /><span>Step Reward</span></div>
        <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 bg-cyan-400" /><span>Cumulative</span></div>
      </div>
    </div>
  );
}
