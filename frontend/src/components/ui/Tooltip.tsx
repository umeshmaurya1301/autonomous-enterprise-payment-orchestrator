"use client";
import { useState, useRef } from "react";
import { cn } from "@/lib/utils";

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export function Tooltip({ content, children, side = "top", className }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const show = () => {
    timerRef.current = setTimeout(() => setVisible(true), 200);
  };
  const hide = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setVisible(false);
  };

  const positionClass = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
  }[side];

  const arrowClass = {
    top: "top-full left-1/2 -translate-x-1/2 border-t-[#2a3448] border-x-transparent border-b-transparent border-4",
    bottom: "bottom-full left-1/2 -translate-x-1/2 border-b-[#2a3448] border-x-transparent border-t-transparent border-4",
    left: "left-full top-1/2 -translate-y-1/2 border-l-[#2a3448] border-y-transparent border-r-transparent border-4",
    right: "right-full top-1/2 -translate-y-1/2 border-r-[#2a3448] border-y-transparent border-l-transparent border-4",
  }[side];

  return (
    <div className="relative inline-flex" onMouseEnter={show} onMouseLeave={hide}>
      {children}
      {visible && (
        <div className={cn("absolute z-50 pointer-events-none", positionClass)}>
          <div className={cn(
            "bg-[#1a2234] border border-[#2a3448] rounded-lg px-3 py-2 text-[11px] font-mono text-slate-300 shadow-xl max-w-[260px] min-w-[180px]",
            className
          )}>
            {content}
          </div>
          <div className={cn("absolute w-0 h-0", arrowClass)} />
        </div>
      )}
    </div>
  );
}

interface TooltipContentProps {
  title: string;
  range?: string;
  description: string;
  warnAt?: string;
  critAt?: string;
  tip?: string;
}

export function SignalTooltip({ title, range, description, warnAt, critAt, tip }: TooltipContentProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-slate-200 font-semibold text-[11px]">{title}</div>
      {range && <div className="text-slate-500 text-[10px]">Range: {range}</div>}
      <div className="text-slate-400 text-[10px] leading-relaxed">{description}</div>
      {(warnAt || critAt) && (
        <div className="flex flex-col gap-0.5 pt-1 border-t border-[#2a3448]">
          {warnAt && <div className="text-[10px] text-yellow-400">⚠ Warn: {warnAt}</div>}
          {critAt && <div className="text-[10px] text-red-400">🔴 Crit: {critAt}</div>}
        </div>
      )}
      {tip && (
        <div className="text-[10px] text-cyan-400 pt-0.5 border-t border-[#2a3448]">
          💡 {tip}
        </div>
      )}
    </div>
  );
}
