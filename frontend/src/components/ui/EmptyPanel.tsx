"use client";
import { cn } from "@/lib/utils";

interface EmptyPanelProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  hint?: string;
  height?: string;
  className?: string;
}

export function EmptyPanel({ icon, title, description, hint, height = "h-36", className }: EmptyPanelProps) {
  return (
    <div className={cn(
      "flex flex-col items-center justify-center gap-2 bg-[#161b27] border border-[#1e2535] border-dashed rounded-xl p-5 text-center",
      height,
      className
    )}>
      {icon && <div className="text-slate-700 mb-1">{icon}</div>}
      <p className="text-slate-500 text-xs font-mono font-semibold">{title}</p>
      <p className="text-slate-700 text-[11px] font-mono leading-relaxed max-w-[260px]">{description}</p>
      {hint && (
        <p className="text-blue-500/60 text-[10px] font-mono mt-1 border border-blue-500/20 bg-blue-500/5 px-3 py-1 rounded-full">
          {hint}
        </p>
      )}
    </div>
  );
}
