"use client";
import { AlertTriangle, Zap } from "lucide-react";

interface KafkaCrisisAlertProps {
  kafkaLag: number;
  onFix?: () => void;
}

export function KafkaCrisisAlert({ kafkaLag, onFix }: KafkaCrisisAlertProps) {
  const isCrash = kafkaLag > 4000;
  const isWarning = !isCrash && kafkaLag > 3000;
  if (!isCrash && !isWarning) return null;

  return (
    <div className={`
      flex items-center justify-between gap-4 px-6 py-2.5 border-b font-mono text-xs
      ${isCrash
        ? "bg-red-950/60 border-red-500/50 animate-pulse"
        : "bg-orange-950/40 border-orange-500/40"
      }
    `}>
      <div className="flex items-center gap-3">
        <div className={`flex items-center gap-1.5 font-bold ${isCrash ? "text-red-400" : "text-orange-400"}`}>
          <AlertTriangle className="w-3.5 h-3.5" />
          {isCrash ? "🔴 KAFKA CRASH THRESHOLD" : "⚠️ KAFKA WARNING"}
        </div>
        <span className="text-slate-400">
          Lag: <span className={isCrash ? "text-red-300 font-semibold" : "text-orange-300 font-semibold"}>
            {Math.round(kafkaLag).toLocaleString()} msgs
          </span>
          {isCrash && " · System stability critical · Cascade phase active"}
          {isWarning && " · Latency compounding · Use CircuitBreaker routing"}
        </span>
      </div>
      {onFix && (
        <button
          onClick={onFix}
          className={`
            flex items-center gap-1.5 px-3 py-1 rounded border text-[11px] font-semibold transition-colors shrink-0
            ${isCrash
              ? "border-red-500/50 text-red-400 hover:bg-red-500/20 bg-red-500/10"
              : "border-orange-500/40 text-orange-400 hover:bg-orange-500/20 bg-orange-500/10"
            }
          `}
        >
          <Zap className="w-3 h-3" />
          Apply CircuitBreaker
        </button>
      )}
    </div>
  );
}
