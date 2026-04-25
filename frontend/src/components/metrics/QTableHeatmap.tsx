"use client";
import { useMemo } from "react";
import { InfoBadge } from "@/components/ui/InfoBadge";

const ACTION_LABELS = ["App·Full", "App·Skip", "Rej·Full", "Rej·Skip", "Chl·Full", "Chl·Skip"];
const STATE_LABELS  = ["LowRisk·LowLag", "LowRisk·HiLag", "MidRisk·LowLag", "MidRisk·HiLag", "HiRisk·LowLag", "HiRisk·HiLag"];

function lerp(a: number[], b: number[], t: number): string {
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
}
const LOW=[30,37,53], MID=[30,53,37], HIGH=[239,68,68];
function valueToColor(v: number, min: number, max: number): string {
  const n = (v - min) / (max - min + 1e-9);
  return n < 0.5 ? lerp(LOW, MID, n * 2) : lerp(MID, HIGH, (n - 0.5) * 2);
}

interface QTableHeatmapProps {
  qValues?: number[][];
}

export function QTableHeatmap({ qValues }: QTableHeatmapProps) {
  const isReal = Boolean(qValues);

  const data = useMemo<number[][]>(() => {
    if (qValues) return qValues;
    return STATE_LABELS.map((_, si) =>
      ACTION_LABELS.map((_, ai) => parseFloat((Math.sin(si * 1.3 + ai * 0.7) * 0.8).toFixed(3)))
    );
  }, [qValues]);

  const allValues = data.flat();
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  return (
    <div className="flex flex-col gap-3 bg-[#161b27] border border-[#1e2535] rounded-xl p-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs uppercase tracking-widest text-slate-400 font-mono">Q-Table Heatmap</h3>
          <InfoBadge
            title="Q-Table Heatmap"
            lines={[
              "Shows the learned Q-values for each state × action pair from the trained RL agent.",
              "Rows = env states bucketed by (risk level × kafka lag). Columns = primary risk/verify action combos.",
              "High Q (red) = agent strongly prefers this action in this state. Low Q (blue) = action is rarely optimal.",
              "Currently showing representative dummy values. Wire real values by passing qValues prop after running python train.py.",
            ]}
            side="bottom"
          />
        </div>
        <div className="flex items-center gap-2">
          {!isReal && (
            <span className="text-[10px] font-mono text-slate-600 border border-[#1e2535] px-2 py-0.5 rounded-full">
              demo data · run train.py for real
            </span>
          )}
          <span className="text-[10px] text-slate-600 font-mono">State × Action</span>
        </div>
      </div>

      {/* Legend row explaining axes */}
      <div className="flex items-center justify-between text-[10px] font-mono text-slate-600 bg-[#0f1117] rounded-lg px-3 py-2 border border-[#1e2535]">
        <span><span className="text-slate-500">Rows →</span> env state buckets (risk + lag level)</span>
        <span><span className="text-slate-500">Cols →</span> agent action combos (decision · verify)</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[10px] font-mono border-separate border-spacing-0.5">
          <thead>
            <tr>
              <th className="text-slate-600 text-left pr-2 pb-1 font-normal w-28">State \ Action</th>
              {ACTION_LABELS.map((a) => (
                <th key={a} className="text-slate-500 font-normal pb-1 text-center">{a}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {STATE_LABELS.map((state, si) => (
              <tr key={state}>
                <td className="text-slate-500 pr-2 text-right whitespace-nowrap">{state}</td>
                {ACTION_LABELS.map((_, ai) => {
                  const v = data[si]?.[ai] ?? 0;
                  const bg = valueToColor(v, min, max);
                  const textColor = v > (min + max) / 2 ? "#f1f5f9" : "#94a3b8";
                  return (
                    <td key={ai} className="text-center rounded py-1.5 px-1 transition-all duration-500 cursor-default"
                      style={{ backgroundColor: bg, color: textColor, minWidth: 54 }}
                      title={`State: ${state}\nAction: ${ACTION_LABELS[ai]}\nQ-value: ${v.toFixed(3)}`}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-3 text-[10px] text-slate-500 font-mono">
        <span>Low Q</span>
        <div className="flex-1 h-1.5 rounded-full bg-gradient-to-r from-[rgb(30,37,53)] via-[rgb(30,53,37)] to-[rgb(239,68,68)]" />
        <span>High Q</span>
      </div>
    </div>
  );
}
