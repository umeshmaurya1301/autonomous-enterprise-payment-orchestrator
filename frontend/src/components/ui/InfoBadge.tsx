"use client";
import { Info } from "lucide-react";
import { Tooltip } from "./Tooltip";
import { cn } from "@/lib/utils";

interface InfoBadgeProps {
  title: string;
  lines: string[];
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export function InfoBadge({ title, lines, side = "top", className }: InfoBadgeProps) {
  return (
    <Tooltip
      side={side}
      className="max-w-[300px]"
      content={
        <div className="flex flex-col gap-1.5">
          <div className="text-slate-200 font-semibold text-[11px] border-b border-[#2a3448] pb-1">{title}</div>
          {lines.map((l, i) => (
            <p key={i} className="text-slate-400 text-[10px] leading-relaxed">{l}</p>
          ))}
        </div>
      }
    >
      <Info className={cn("w-3.5 h-3.5 text-slate-600 hover:text-slate-400 cursor-help transition-colors", className)} />
    </Tooltip>
  );
}
