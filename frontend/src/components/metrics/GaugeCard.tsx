"use client";
import { cn } from "@/lib/utils";
import { InfoBadge } from "@/components/ui/InfoBadge";

interface GaugeCardProps {
  title: string;
  icon: React.ReactNode;
  value: number;
  displayValue: string;
  subtitle: string;
  color: string;
  alerting?: boolean;
  children?: React.ReactNode;
  info?: { title: string; lines: string[] };
  statusLabel?: string;
  statusColor?: string;
}

const RADIUS = 54;
const CIRCUMFERENCE = Math.PI * RADIUS;

export function GaugeCard({ title, icon, value, displayValue, subtitle, color, alerting = false, children, info, statusLabel, statusColor }: GaugeCardProps) {
  const arcLength = Math.min(1, Math.max(0, value)) * CIRCUMFERENCE;

  return (
    <div className={cn(
      "relative flex flex-col bg-[#161b27] border rounded-xl p-5 gap-3 transition-all duration-300",
      alerting ? "border-red-500/70 glow-red animate-pulse" : "border-[#1e2535]"
    )}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-400 text-xs uppercase tracking-widest font-mono">
          {icon}
          <span>{title}</span>
          {info && <InfoBadge title={info.title} lines={info.lines} side="bottom" />}
        </div>
        {alerting && (
          <span className="text-[10px] bg-red-500/20 text-red-400 border border-red-500/40 px-2 py-0.5 rounded-full font-mono">ALERT</span>
        )}
      </div>

      <div className="flex flex-col items-center gap-1">
        <svg viewBox="0 0 120 70" className="w-32 h-20">
          <path d="M 10 65 A 50 50 0 0 1 110 65" fill="none" stroke="#1e2535" strokeWidth="10" strokeLinecap="round" />
          <path d="M 10 65 A 50 50 0 0 1 110 65" fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
            strokeDasharray={`${arcLength} ${CIRCUMFERENCE}`}
            style={{ transition: "stroke-dasharray 0.5s ease" }}
          />
          <text x="60" y="62" textAnchor="middle" fill={color} fontSize="14" fontFamily="monospace" fontWeight="700">
            {displayValue}
          </text>
        </svg>
        {statusLabel && (
          <span
            className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border"
            style={{
              color: statusColor ?? color,
              borderColor: `${statusColor ?? color}40`,
              backgroundColor: `${statusColor ?? color}15`,
            }}
          >
            {statusLabel}
          </span>
        )}
        <span className="text-slate-500 text-[11px] font-mono">{subtitle}</span>
      </div>

      {children && <div className="border-t border-[#1e2535] pt-3">{children}</div>}
    </div>
  );
}
