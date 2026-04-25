"use client";
import { cn } from "@/lib/utils";
import type { EpisodeHistoryEntry } from "@/lib/types";
import { CheckCircle, XCircle, ClipboardList } from "lucide-react";
import { InfoBadge } from "@/components/ui/InfoBadge";

const THRESHOLDS = { easy: 0.75, medium: 0.45, hard: 0.30 };

function taskColor(t: string) {
  return t === "easy" ? "text-green-400" : t === "medium" ? "text-yellow-400" : "text-red-400";
}
function phaseColor(p: string) {
  return p === "cascade" ? "text-red-400" : p === "spike" ? "text-orange-400" : "text-slate-500";
}

export function EpisodeHistory({ entries }: { entries: EpisodeHistoryEntry[] }) {
  return (
    <div className="flex flex-col gap-3 bg-[#161b27] border border-[#1e2535] rounded-xl p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">Episode History</h3>
          <InfoBadge
            title="Episode History"
            lines={[
              "A row is added here each time an episode ends (done=true from POST /step).",
              "Avg Reward = total reward ÷ steps. This is the number judged against the threshold.",
              "PASS = avg reward met the task threshold (Easy ≥0.75, Medium ≥0.45, Hard ≥0.30).",
              "Final Phase shows what the env was doing at episode end — cascade = hardest conditions.",
            ]}
          />
        </div>
        {entries.length > 0 && (
          <span className="text-[10px] text-slate-600 font-mono">{entries.length} episode{entries.length > 1 ? "s" : ""}</span>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 py-6">
          <ClipboardList className="w-8 h-8 text-slate-700" />
          <p className="text-slate-600 text-xs font-mono">No episodes completed yet</p>
          <p className="text-slate-700 text-[10px] font-mono text-center max-w-[240px]">
            Complete an episode (100 steps or done=true) and a summary row will appear here
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-slate-600 text-[10px] uppercase">
                <th className="text-left pb-2 pr-3 font-normal">#</th>
                <th className="text-left pb-2 pr-3 font-normal">Task</th>
                <th className="text-right pb-2 pr-3 font-normal">Steps</th>
                <th className="text-right pb-2 pr-3 font-normal">Avg Rwd</th>
                <th className="text-right pb-2 pr-3 font-normal">Threshold</th>
                <th className="text-left pb-2 pr-3 font-normal">Final Phase</th>
                <th className="text-center pb-2 font-normal">Result</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2535]">
              {entries.map((ep) => (
                <tr key={ep.id} className="hover:bg-[#1a2234]/50 transition-colors">
                  <td className="py-1.5 pr-3 text-slate-600">{ep.id}</td>
                  <td className={cn("py-1.5 pr-3 font-semibold", taskColor(ep.task))}>{ep.task.toUpperCase()}</td>
                  <td className="py-1.5 pr-3 text-right text-slate-400">{ep.steps}</td>
                  <td className={cn("py-1.5 pr-3 text-right font-semibold", ep.passed ? "text-green-400" : "text-red-400")}>
                    {ep.avgReward.toFixed(3)}
                  </td>
                  <td className="py-1.5 pr-3 text-right text-slate-600">{THRESHOLDS[ep.task]}</td>
                  <td className={cn("py-1.5 pr-3", phaseColor(ep.finalPhase))}>{ep.finalPhase}</td>
                  <td className="py-1.5 text-center">
                    {ep.passed
                      ? <CheckCircle className="w-3.5 h-3.5 text-green-400 inline" />
                      : <XCircle className="w-3.5 h-3.5 text-red-400 inline" />
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
