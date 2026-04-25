"use client";
import { Tooltip } from "@/components/ui/Tooltip";

interface PhaseTimelineProps {
  phases: string[];
}

const PHASE_COLOR: Record<string, string> = {
  normal:  "#22c55e",
  spike:   "#f97316",
  cascade: "#ef4444",
};

const PHASE_BG: Record<string, string> = {
  normal:  "bg-green-500",
  spike:   "bg-orange-500",
  cascade: "bg-red-500",
};

export function PhaseTimeline({ phases }: PhaseTimelineProps) {
  if (phases.length === 0) return null;

  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">Phase history</span>
      <div className="flex items-end gap-px h-5 overflow-hidden">
        {phases.map((p, i) => {
          const color = PHASE_COLOR[p] ?? "#475569";
          return (
            <Tooltip
              key={i}
              content={
                <div className="text-[11px] font-mono">
                  <span className="text-slate-400">Step {i + 1}:</span>{" "}
                  <span style={{ color }}>{p}</span>
                </div>
              }
              side="top"
            >
              <div
                className="flex-1 min-w-[2px] rounded-sm cursor-pointer transition-opacity hover:opacity-100 opacity-80"
                style={{ height: p === "cascade" ? "100%" : p === "spike" ? "70%" : "40%", backgroundColor: color }}
              />
            </Tooltip>
          );
        })}
      </div>
      <div className="flex items-center gap-3 text-[10px] font-mono text-slate-600">
        <span className="flex items-center gap-1"><span className="w-2 h-1 rounded-sm bg-green-500 inline-block" />normal</span>
        <span className="flex items-center gap-1"><span className="w-2 h-1 rounded-sm bg-orange-500 inline-block" />spike</span>
        <span className="flex items-center gap-1"><span className="w-2 h-1 rounded-sm bg-red-500 inline-block" />cascade</span>
      </div>
    </div>
  );
}
