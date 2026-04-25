"use client";
import { cn } from "@/lib/utils";
import { Tooltip, SignalTooltip } from "@/components/ui/Tooltip";

interface CurriculumProgressProps {
  level: number;
  task: string;
}

const LEVELS = [
  { label: "L0", name: "Easy Conditions", color: "bg-green-500", textColor: "text-green-400", description: "Low adversary, stable signals, relaxed thresholds." },
  { label: "L1", name: "Medium Conditions", color: "bg-yellow-500", textColor: "text-yellow-400", description: "Moderate adversary escalation, occasional spikes, tighter SLA window." },
  { label: "L2", name: "Hard Conditions", color: "bg-red-500", textColor: "text-red-400", description: "Full adversary, cascade events, maximum entropy variance." },
];

export function CurriculumProgress({ level, task }: CurriculumProgressProps) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider shrink-0">Curriculum</span>
      <div className="flex items-center gap-1.5 flex-1">
        {LEVELS.map((l, i) => (
          <Tooltip
            key={i}
            content={
              <SignalTooltip
                title={`${l.label} — ${l.name}`}
                range=""
                description={l.description}
                tip={i <= level ? "Current or completed level" : "Not yet reached"}
              />
            }
            side="top"
          >
            <div className="flex items-center gap-1 cursor-help">
              <div className={cn(
                "h-1.5 w-10 rounded-full transition-all duration-500",
                i <= level ? l.color : "bg-[#1e2535]"
              )} />
              <span className={cn(
                "text-[9px] font-mono font-semibold",
                i === level ? l.textColor : i < level ? "text-slate-500" : "text-slate-700"
              )}>
                {l.label}
              </span>
            </div>
          </Tooltip>
        ))}
      </div>
      <span className="text-[10px] font-mono text-slate-600 shrink-0">{task}</span>
    </div>
  );
}
