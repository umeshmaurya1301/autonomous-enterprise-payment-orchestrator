import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatMs(ms: number): string {
  return `${Math.round(ms)}ms`;
}

export function formatPercent(value: number, max: number): string {
  return `${Math.round((value / max) * 100)}%`;
}

export function getRiskColor(normalized: number): string {
  if (normalized >= 0.8) return "#ef4444";
  if (normalized >= 0.6) return "#f97316";
  if (normalized >= 0.4) return "#f59e0b";
  return "#22c55e";
}

export function getLatencyColor(ms: number): string {
  if (ms > 800) return "#ef4444";
  if (ms > 500) return "#f59e0b";
  return "#22c55e";
}

export function getKafkaColor(lag: number): string {
  if (lag > 4000) return "#ef4444";
  if (lag > 3000) return "#f97316";
  if (lag > 2000) return "#f59e0b";
  return "#22c55e";
}

export function getBankStatusLabel(status: number): string {
  return ["Healthy", "Degraded", "Unknown"][Math.round(status)] ?? "Unknown";
}

export function getBankStatusColor(status: number): string {
  const s = Math.round(status);
  if (s === 0) return "#22c55e";
  if (s === 1) return "#f59e0b";
  return "#ef4444";
}

export function getRiskDecisionLabel(d: number): string {
  return ["Approve", "Reject", "Challenge"][d] ?? "?";
}

export function getCryptoVerifyLabel(v: number): string {
  return ["FullVerify", "SkipVerify"][v] ?? "?";
}

export function getInfraRoutingLabel(r: number): string {
  return ["Normal", "Throttle", "CircuitBreaker"][r] ?? "?";
}

export function getDbRetryLabel(d: number): string {
  return ["FailFast", "ExpBackoff"][d] ?? "?";
}

export function getSettlementLabel(s: number): string {
  return ["StandardSync", "DeferredAsync"][s] ?? "?";
}

export function getAppPriorityLabel(p: number): string {
  return ["UPI", "Credit", "Balanced"][p] ?? "?";
}

export function getChannelLabel(c: number): string {
  return ["P2P", "P2M", "AutoPay"][Math.round(c)] ?? "?";
}

export function getMerchantTierLabel(t: number): string {
  return t >= 0.5 ? "Enterprise" : "Small";
}

export function generateActionLabel(action: {
  risk_decision: number;
  crypto_verify: number;
  infra_routing: number;
}): string {
  return `${getRiskDecisionLabel(action.risk_decision)} · ${getCryptoVerifyLabel(action.crypto_verify)} · ${getInfraRoutingLabel(action.infra_routing)}`;
}
