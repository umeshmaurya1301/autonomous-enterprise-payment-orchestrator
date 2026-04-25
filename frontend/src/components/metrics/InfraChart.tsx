"use client";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";
import { InfoBadge } from "@/components/ui/InfoBadge";
import { Activity } from "lucide-react";

export interface InfraPoint {
  step: number;
  kafkaLag: number;
  p99: number;
  dbPool: number;
}

interface InfraChartProps {
  data: InfraPoint[];
}

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: InfraPoint }> }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-[#0f1117] border border-[#1e2535] rounded p-2 text-[11px] font-mono">
      <div className="text-slate-400">Step {d.step}</div>
      <div className="text-orange-400">P99: {Math.round(d.p99)}ms</div>
      <div className="text-red-400">Kafka: {Math.round(d.kafkaLag * 100)}</div>
      <div className="text-purple-400">DB Pool: {Math.round(d.dbPool)}%</div>
    </div>
  );
};

export function InfraChart({ data }: InfraChartProps) {
  return (
    <div className="flex flex-col gap-3 bg-[#161b27] border border-[#1e2535] rounded-xl p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">Infra Trend</h3>
          <InfoBadge
            title="Infrastructure Trend Chart"
            lines={[
              "Orange area = Rolling P99 latency over time. Breaching 800ms activates SLA penalty.",
              "Red area = Kafka lag ÷ 100 (scaled to share axis with P99). Watch for sudden spikes — they trigger cascade phase.",
              "Populated from GET /state polling every 500ms. Updates continuously while episode is running.",
            ]}
          />
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
          <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-orange-400" />P99 (ms)</span>
          <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-red-400" />Lag ÷100</span>
        </div>
      </div>

      {data.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-24 gap-1.5">
          <Activity className="w-7 h-7 text-slate-700" />
          <p className="text-slate-600 text-[11px] font-mono">Infra signals appear here once the episode starts</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2535" vertical={false} />
            <XAxis dataKey="step" hide />
            <YAxis tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="p99" stroke="#f97316" strokeWidth={1.5} fill="#f97316" fillOpacity={0.08} dot={false} />
            <Area type="monotone" dataKey="kafkaLag" stroke="#ef4444" strokeWidth={1} fill="#ef4444" fillOpacity={0.05} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
