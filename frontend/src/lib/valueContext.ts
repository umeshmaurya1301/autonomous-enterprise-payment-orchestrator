// Returns a plain-English interpretation of a raw observation value

export interface ValueContext {
  label: string;
  meaning: string;
  advice: string;
  color: string;
}

export function getValueContext(key: string, value: number): ValueContext {
  switch (key) {
    case "kafka_lag":
      if (value > 4000) return { label: "CRASH ZONE", color: "#ef4444", meaning: `${Math.round(value).toLocaleString()} messages backlogged — past the crash threshold.`, advice: "Apply CircuitBreaker routing immediately. Cascade phase active." };
      if (value > 3000) return { label: "WARNING",    color: "#f97316", meaning: `${Math.round(value).toLocaleString()} messages — latency is compounding.`,              advice: "Switch to Throttle or CircuitBreaker routing now." };
      if (value > 1500) return { label: "BUILDING",   color: "#f59e0b", meaning: `${Math.round(value).toLocaleString()} messages — backlog is growing.`,                  advice: "Monitor. Throttle routing will slow growth." };
      return                    { label: "HEALTHY",    color: "#22c55e", meaning: `${Math.round(value).toLocaleString()} messages — system is keeping up.`,                advice: "Normal routing is fine." };

    case "risk_score":
      if (value > 80) return { label: "CRITICAL FRAUD", color: "#ef4444", meaning: `${Math.round(value)}/100 — very high fraud probability.`,     advice: "Reject + FullVerify. Approving here costs -0.3 reward." };
      if (value > 60) return { label: "HIGH RISK",      color: "#f97316", meaning: `${Math.round(value)}/100 — elevated fraud signal.`,            advice: "Challenge or Reject. FullVerify strongly recommended." };
      if (value > 30) return { label: "MODERATE",       color: "#f59e0b", meaning: `${Math.round(value)}/100 — some risk, review carefully.`,      advice: "Challenge is safe. SkipVerify not recommended." };
      return                  { label: "LOW RISK",       color: "#22c55e", meaning: `${Math.round(value)}/100 — transaction looks clean.`,          advice: "Approve + SkipVerify is acceptable here." };

    case "adversary_threat_level":
      if (value > 8) return { label: "MAX ESCALATION", color: "#ef4444", meaning: `${value.toFixed(1)}/10 — adversary at peak aggression.`,      advice: "Curriculum level may promote soon. Expect volatile signals." };
      if (value > 5) return { label: "ESCALATING",     color: "#f97316", meaning: `${value.toFixed(1)}/10 — adversary actively challenging.`,    advice: "Increased risk score volatility. Stay alert to sudden spikes." };
      if (value > 3) return { label: "PROBING",        color: "#f59e0b", meaning: `${value.toFixed(1)}/10 — adversary testing the agent.`,       advice: "Normal play but watch for quick risk_score jumps." };
      return                 { label: "DORMANT",        color: "#22c55e", meaning: `${value.toFixed(1)}/10 — adversary not yet engaged.`,         advice: "Easy conditions. Focus on matching risk decisions." };

    case "system_entropy":
      if (value > 70) return { label: "SPIKE INCOMING", color: "#ef4444", meaning: `${Math.round(value)}% — latency spike fires next step.`,   advice: "Switch to Throttle routing to bleed entropy quickly." };
      if (value > 50) return { label: "UNSTABLE",       color: "#f97316", meaning: `${Math.round(value)}% — system stress building.`,          advice: "Throttle routing helps. Avoid adding more load." };
      if (value > 30) return { label: "MILD STRESS",    color: "#f59e0b", meaning: `${Math.round(value)}% — some background instability.`,     advice: "No immediate action needed. Monitor trend." };
      return                  { label: "STABLE",         color: "#22c55e", meaning: `${Math.round(value)}% — system is calm.`,                  advice: "No action needed on entropy." };

    case "rolling_p99":
      if (value > 800) return { label: "SLA BREACHED",  color: "#ef4444", meaning: `${Math.round(value)}ms — -1.0 reward penalty every step until this drops below 800ms.`, advice: "Reduce API latency: lower entropy, reduce kafka lag, avoid SkipVerify." };
      if (value > 700) return { label: "CRITICAL",      color: "#f97316", meaning: `${Math.round(value)}ms — 100ms from SLA breach.`,           advice: "Act now. One more latency spike will breach the SLA." };
      if (value > 500) return { label: "CAUTION",       color: "#f59e0b", meaning: `${Math.round(value)}ms — warming toward SLA limit.`,        advice: "Throttle routing. Avoid SkipVerify." };
      return                   { label: "HEALTHY",       color: "#22c55e", meaning: `${Math.round(value)}ms — well within SLA.`,                 advice: "No action needed on latency." };

    case "api_latency":
      if (value > 800) return { label: "VERY SLOW",  color: "#ef4444", meaning: `${Math.round(value)}ms — this spike is feeding into P99.`,    advice: "Reduce entropy to stop latency spikes." };
      if (value > 500) return { label: "SLOW",       color: "#f59e0b", meaning: `${Math.round(value)}ms — elevated latency this step.`,        advice: "Monitor P99. Multiple spikes will breach SLA." };
      return                   { label: "NORMAL",    color: "#22c55e", meaning: `${Math.round(value)}ms — latency is fine.`,                   advice: "No action needed." };

    case "db_connection_pool":
      if (value > 80) return { label: "SATURATED",  color: "#ef4444", meaning: `${Math.round(value)}% — retry overhead is now active.`,       advice: "Use FailFast policy to avoid the -0.10 ExpBackoff penalty." };
      if (value > 65) return { label: "BUSY",       color: "#f97316", meaning: `${Math.round(value)}% — approaching saturation.`,             advice: "Prefer FailFast. Watch for it crossing 80%." };
      if (value > 50) return { label: "MODERATE",   color: "#f59e0b", meaning: `${Math.round(value)}% — pool in use but manageable.`,         advice: "No action needed yet." };
      return                  { label: "HEALTHY",   color: "#22c55e", meaning: `${Math.round(value)}% — plenty of connections available.`,    advice: "DB is fine. No retry overhead." };

    case "bank_api_status": {
      const s = Math.round(value);
      if (s === 1) return { label: "DEGRADED (sim)", color: "#f59e0b", meaning: "Bank sim path is degraded.",  advice: "Switch to DeferredAsync settlement for +0.05 bonus." };
      if (s === 2) return { label: "UNKNOWN (sim)",  color: "#94a3b8", meaning: "Bank sim state is unknown.",  advice: "DeferredAsync is safer in unknown state." };
      return               { label: "HEALTHY (sim)", color: "#22c55e", meaning: "Bank sim path is healthy.",   advice: "StandardSync settlement is optimal." };
    }

    case "merchant_tier": {
      const isEnterprise = value >= 0.5;
      return isEnterprise
        ? { label: "ENTERPRISE", color: "#3b82f6", meaning: "Large enterprise merchant.", advice: "Set App Priority to UPI for +0.02/step bonus." }
        : { label: "SMALL",      color: "#94a3b8", meaning: "Small merchant.",            advice: "Set App Priority to Balanced for +0.02/step bonus." };
    }

    case "channel": {
      const labels = ["P2P", "P2M", "AutoPay"];
      const meanings = ["Person-to-person transfer.", "Payment to a merchant.", "Automated recurring payment."];
      const idx = Math.round(value);
      return { label: labels[idx] ?? "?", color: "#94a3b8", meaning: meanings[idx] ?? "", advice: "Channel is informational — affects which routing earns best reward." };
    }

    default:
      return { label: String(value), color: "#94a3b8", meaning: "", advice: "" };
  }
}
