"use client";
import { ShieldAlert, Server, Timer } from "lucide-react";
import { GaugeCard } from "./GaugeCard";
import type { AEPOObservation } from "@/lib/types";
import { getRiskColor, getLatencyColor, getKafkaColor, formatMs } from "@/lib/utils";

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] text-slate-500 font-mono uppercase">{label}</span>
      <span className="text-sm font-mono font-semibold" style={{ color }}>{value}</span>
    </div>
  );
}

function riskStatus(score: number): { label: string; color: string } {
  if (score >= 80) return { label: "CRITICAL", color: "#ef4444" };
  if (score >= 60) return { label: "HIGH RISK", color: "#f97316" };
  if (score >= 30) return { label: "MODERATE", color: "#f59e0b" };
  return { label: "LOW RISK", color: "#22c55e" };
}

function kafkaStatus(lag: number): { label: string; color: string } {
  if (lag >= 4000) return { label: "CRASH", color: "#ef4444" };
  if (lag >= 3000) return { label: "WARNING", color: "#f97316" };
  if (lag >= 1000) return { label: "BUILDING", color: "#f59e0b" };
  return { label: "HEALTHY", color: "#22c55e" };
}

function p99Status(p99: number): { label: string; color: string } {
  if (p99 >= 800) return { label: "SLA BREACH", color: "#ef4444" };
  if (p99 >= 700) return { label: "CRITICAL", color: "#f97316" };
  if (p99 >= 400) return { label: "CAUTION", color: "#f59e0b" };
  return { label: "HEALTHY", color: "#22c55e" };
}

export function RiskTriad({ obs }: { obs: AEPOObservation }) {
  const riskNorm = obs.risk_score / 100;
  const riskColor = getRiskColor(riskNorm);

  const kafkaNorm = obs.kafka_lag / 10000;
  const infraColor = getKafkaColor(obs.kafka_lag);

  const p99Norm = Math.min(1, obs.rolling_p99 / 1200);
  const slaColor = getLatencyColor(obs.rolling_p99);
  const entropyColor = obs.system_entropy > 70 ? "#ef4444" : obs.system_entropy > 50 ? "#f59e0b" : "#22c55e";

  const riskSt = riskStatus(obs.risk_score);
  const kafkaSt = kafkaStatus(obs.kafka_lag);
  const p99St = p99Status(obs.rolling_p99);

  return (
    <div className="grid grid-cols-3 gap-4">
      <GaugeCard
        title="Fraud Risk Signal"
        icon={<ShieldAlert className="w-3.5 h-3.5" />}
        value={riskNorm}
        displayValue={`${Math.round(obs.risk_score)}`}
        subtitle="sim risk score · /100"
        color={riskColor}
        statusLabel={riskSt.label}
        statusColor={riskSt.color}
        alerting={riskNorm >= 0.8 || obs.adversary_threat_level >= 8}
        info={{
          title: "Fraud Risk Gauge",
          lines: [
            "Arc = risk_score normalized 0–100. Turns red above 80.",
            "Adversary = multi-agent env adversary escalation level (0–10). Rises as agent performs well, testing robustness.",
            "Alert fires when risk > 80 OR adversary > 8. Agent should switch to Reject + FullVerify.",
          ],
        }}
      >
        <div className="grid grid-cols-2 gap-2">
          <Metric label="Risk Score" value={`${Math.round(obs.risk_score)}`} color={riskColor} />
          <Metric label="Adversary" value={`${obs.adversary_threat_level.toFixed(1)} / 10`}
            color={obs.adversary_threat_level >= 7 ? "#ef4444" : "#f59e0b"} />
        </div>
      </GaugeCard>

      <GaugeCard
        title="Infra Health Signal"
        icon={<Server className="w-3.5 h-3.5" />}
        value={kafkaNorm}
        displayValue={`${(obs.kafka_lag / 1000).toFixed(1)}k`}
        subtitle="sim kafka lag · 0–10k msgs"
        color={infraColor}
        statusLabel={kafkaSt.label}
        statusColor={kafkaSt.color}
        alerting={obs.kafka_lag > 4000 || obs.db_connection_pool > 85}
        info={{
          title: "Infrastructure Health Gauge",
          lines: [
            "Arc = kafka_lag normalized to 0–10,000 messages. Pulses red above 4,000 (crash threshold).",
            "Entropy = system chaos index. Above 70 causes a latency spike in the next step.",
            "Alert fires when lag > 4,000 (cascade phase) or DB pool > 85%. Use CircuitBreaker routing to recover.",
          ],
        }}
      >
        <div className="grid grid-cols-2 gap-2">
          <Metric label="Kafka Lag" value={`${Math.round(obs.kafka_lag)}`} color={infraColor} />
          <Metric label="Entropy" value={`${Math.round(obs.system_entropy)}%`} color={entropyColor} />
        </div>
      </GaugeCard>

      <GaugeCard
        title="SLA Compliance"
        icon={<Timer className="w-3.5 h-3.5" />}
        value={p99Norm}
        displayValue={formatMs(obs.rolling_p99)}
        subtitle="ema p99 latency · sla 800ms"
        color={slaColor}
        statusLabel={p99St.label}
        statusColor={p99St.color}
        alerting={obs.rolling_p99 > 800}
        info={{
          title: "SLA Compliance Gauge",
          lines: [
            "Arc = rolling_p99 (EMA-smoothed latency). Formula: 0.8×prev + 0.2×api_latency.",
            "SLA threshold is 800ms. Breaching it applies a -1.0 reward penalty every step until recovered.",
            "API Latency = raw instantaneous latency. P99 responds slower due to EMA smoothing — a single spike won't immediately breach SLA.",
          ],
        }}
      >
        <div className="grid grid-cols-2 gap-2">
          <Metric label="P99 (ema)" value={formatMs(obs.rolling_p99)} color={slaColor} />
          <Metric label="API Latency" value={formatMs(obs.api_latency)} color={getLatencyColor(obs.api_latency)} />
        </div>
      </GaugeCard>
    </div>
  );
}
