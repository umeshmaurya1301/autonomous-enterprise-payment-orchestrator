"use client";
import { useState, useRef, useEffect } from "react";
import { X, Search, BookOpen, ChevronDown, ChevronRight, ArrowRight } from "lucide-react";
import { GLOSSARY, CATEGORIES, searchGlossary, type GlossaryEntry, type GlossaryCategory } from "@/lib/glossary";
import { cn } from "@/lib/utils";

interface GlossaryPanelProps {
  open: boolean;
  onClose: () => void;
}

export function GlossaryPanel({ open, onClose }: GlossaryPanelProps) {
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState<GlossaryCategory | "all">("all");
  const [expandedTerms, setExpandedTerms] = useState<Set<string>>(new Set());
  const [highlightedTerm, setHighlightedTerm] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const termRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const filtered = searchGlossary(query).filter(
    (e) => activeCategory === "all" || e.category === activeCategory
  );

  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 150);
    } else {
      setQuery("");
      setActiveCategory("all");
      setHighlightedTerm(null);
    }
  }, [open]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const toggleExpand = (term: string) => {
    setExpandedTerms((prev) => {
      const next = new Set(prev);
      if (next.has(term)) next.delete(term);
      else next.add(term);
      return next;
    });
  };

  const navigateToTerm = (term: string) => {
    setQuery("");
    setActiveCategory("all");
    setHighlightedTerm(term);
    setExpandedTerms((prev) => new Set(prev).add(term));
    setTimeout(() => {
      termRefs.current[term]?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 50);
    setTimeout(() => setHighlightedTerm(null), 2000);
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 bg-black/60 z-50 transition-opacity duration-300",
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full w-[480px] max-w-[95vw] bg-[#0d1019] border-l border-[#1e2535] z-50",
          "flex flex-col shadow-2xl transition-transform duration-300 ease-in-out",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#1e2535] bg-[#0f1117]">
          <div className="flex items-center gap-2.5">
            <BookOpen className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-mono font-bold text-slate-200">Metrics Guide</span>
            <span className="text-[10px] font-mono text-slate-600 bg-[#1e2535] px-1.5 py-0.5 rounded">
              {GLOSSARY.length} terms
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-300 transition-colors p-1 rounded-md hover:bg-[#1e2535]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Search */}
        <div className="px-4 pt-3 pb-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
            <input
              ref={searchRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search terms, e.g. kafka, entropy, P99…"
              className="w-full bg-[#161b27] border border-[#1e2535] rounded-lg pl-8 pr-3 py-2 text-[12px] font-mono text-slate-300 placeholder-slate-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 transition-all"
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-400"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex items-center gap-1.5 px-4 pb-3 overflow-x-auto">
          <CategoryChip
            label="All"
            active={activeCategory === "all"}
            onClick={() => setActiveCategory("all")}
            color="text-slate-300 bg-slate-700/30 border-slate-600/30"
            activeColor="text-slate-100 bg-slate-600/40 border-slate-500/50"
          />
          {CATEGORIES.map((cat) => (
            <CategoryChip
              key={cat.id}
              label={cat.label}
              active={activeCategory === cat.id}
              onClick={() => setActiveCategory(cat.id)}
              color={cat.color}
              activeColor={cat.color}
            />
          ))}
        </div>

        {/* Results count */}
        {query && (
          <div className="px-4 pb-1.5">
            <span className="text-[10px] font-mono text-slate-600">
              {filtered.length} result{filtered.length !== 1 ? "s" : ""} for &ldquo;{query}&rdquo;
            </span>
          </div>
        )}

        {/* Entries */}
        <div className="flex-1 overflow-y-auto px-4 pb-6 space-y-2">
          {filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-slate-600 text-[12px] font-mono gap-2">
              <Search className="w-6 h-6 opacity-40" />
              <span>No terms match &ldquo;{query}&rdquo;</span>
            </div>
          ) : (
            filtered.map((entry) => (
              <GlossaryEntryCard
                key={entry.term}
                entry={entry}
                expanded={expandedTerms.has(entry.term)}
                highlighted={highlightedTerm === entry.term}
                onToggle={() => toggleExpand(entry.term)}
                onNavigate={navigateToTerm}
                ref={(el) => { termRefs.current[entry.term] = el; }}
              />
            ))
          )}
        </div>

        {/* Footer hint */}
        <div className="px-4 py-2.5 border-t border-[#1e2535] bg-[#0f1117]">
          <p className="text-[10px] font-mono text-slate-700 text-center">
            Press <kbd className="bg-[#1e2535] border border-[#2a3448] px-1 py-0.5 rounded text-slate-500">Esc</kbd> to close · Hover any metric for inline context
          </p>
        </div>
      </div>
    </>
  );
}

function CategoryChip({ label, active, onClick, color, activeColor }: {
  label: string; active: boolean; onClick: () => void; color: string; activeColor: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2.5 py-1 rounded-full border text-[10px] font-mono font-semibold whitespace-nowrap shrink-0 transition-all",
        active ? activeColor : cn(color, "opacity-60 hover:opacity-100")
      )}
    >
      {label}
    </button>
  );
}

const GlossaryEntryCard = ({
  entry, expanded, highlighted, onToggle, onNavigate, ref
}: {
  entry: GlossaryEntry;
  expanded: boolean;
  highlighted: boolean;
  onToggle: () => void;
  onNavigate: (term: string) => void;
  ref: (el: HTMLDivElement | null) => void;
}) => {
  const cat = CATEGORIES.find((c) => c.id === entry.category);

  return (
    <div
      ref={ref}
      className={cn(
        "rounded-xl border transition-all duration-500",
        highlighted
          ? "border-blue-500/60 bg-blue-500/5 ring-1 ring-blue-500/20"
          : "border-[#1e2535] bg-[#161b27] hover:border-[#2a3448]"
      )}
    >
      {/* Entry header — always visible */}
      <button
        onClick={onToggle}
        className="w-full flex items-start gap-3 p-4 text-left"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1.5">
            <span className="text-[13px] font-mono font-bold text-slate-200">{entry.term}</span>
            {cat && (
              <span className={cn("text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded-full border uppercase tracking-wider", cat.color)}>
                {cat.label}
              </span>
            )}
          </div>
          <p className="text-[11px] font-mono text-slate-400 leading-relaxed">{entry.plain}</p>
          {entry.range && (
            <p className="text-[10px] font-mono text-slate-600 mt-1">Range: {entry.range}</p>
          )}
        </div>
        <div className="text-slate-600 mt-0.5 shrink-0">
          {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 flex flex-col gap-3 border-t border-[#1e2535]">

          {/* Levels / thresholds */}
          {entry.levels && entry.levels.length > 0 && (
            <div className="pt-3">
              <p className="text-[9px] font-mono text-slate-600 uppercase tracking-wider mb-2">Levels &amp; Thresholds</p>
              <div className="flex flex-col gap-1.5">
                {entry.levels.map((level) => (
                  <div key={level.range} className="flex items-start gap-2.5">
                    <div
                      className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
                      style={{ backgroundColor: level.color }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2 flex-wrap">
                        <span
                          className="text-[10px] font-mono font-bold"
                          style={{ color: level.color }}
                        >
                          {level.label}
                        </span>
                        <span className="text-[10px] font-mono text-slate-600">{level.range}</span>
                      </div>
                      <p className="text-[10px] font-mono text-slate-400 leading-relaxed">{level.meaning}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Detail explanation */}
          <div>
            <p className="text-[9px] font-mono text-slate-600 uppercase tracking-wider mb-1.5">How it works</p>
            <p className="text-[11px] font-mono text-slate-400 leading-relaxed">{entry.detail}</p>
          </div>

          {/* See Also */}
          {entry.seeAlso && entry.seeAlso.length > 0 && (
            <div>
              <p className="text-[9px] font-mono text-slate-600 uppercase tracking-wider mb-1.5">See Also</p>
              <div className="flex flex-wrap gap-1.5">
                {entry.seeAlso.map((related) => (
                  <button
                    key={related}
                    onClick={() => onNavigate(related)}
                    className="flex items-center gap-1 text-[10px] font-mono text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded-full hover:bg-blue-500/20 hover:border-blue-500/40 transition-all"
                  >
                    <ArrowRight className="w-2.5 h-2.5" />
                    {related}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
