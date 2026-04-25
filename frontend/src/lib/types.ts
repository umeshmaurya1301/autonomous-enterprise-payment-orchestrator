export interface AEPOObservation {
  channel: number;           // 0-2  (P2P/P2M/AutoPay)
  risk_score: number;        // 0-100
  adversary_threat_level: number; // 0-10
  system_entropy: number;    // 0-100
  kafka_lag: number;         // 0-10000
  api_latency: number;       // 0-5000 ms
  rolling_p99: number;       // 0-5000 ms
  db_connection_pool: number; // 0-100
  bank_api_status: number;   // 0=Healthy, 1=Degraded, 2=Unknown
  merchant_tier: number;     // 0=Small, 1=Enterprise
}

export interface AEPOAction {
  risk_decision: number;     // 0=Approve, 1=Reject, 2=Challenge
  crypto_verify: number;     // 0=FullVerify, 1=SkipVerify
  infra_routing: number;     // 0=Normal, 1=Throttle, 2=CircuitBreaker
  db_retry_policy: number;   // 0=FailFast, 1=ExponentialBackoff
  settlement_policy: number; // 0=StandardSync, 1=DeferredAsyncFallback
  app_priority: number;      // 0=UPI, 1=Credit, 2=Balanced
}

export interface StepResult {
  observation: AEPOObservation;
  reward: number;
  reward_breakdown: Record<string, number>;
  done: boolean;
  info: Record<string, unknown>;
}

export interface StateResult {
  observation: AEPOObservation;
}

export interface ResetResult {
  observation: AEPOObservation;
  info: Record<string, unknown>;
}

export interface ActionLogEntry {
  id: number;
  timestamp: string;
  action: AEPOAction;
  reward: number;
  rewardBreakdown: Record<string, number>;
  step: number;
  label: string;
  phase: string;
}

export interface RewardPoint {
  step: number;
  reward: number;
  cumulative: number;
}

export interface CausalNotification {
  id: string;
  message: string;
  severity: "warning" | "critical" | "info";
  timestamp: number;
}

export type TaskDifficulty = "easy" | "medium" | "hard";

export interface EpisodeStats {
  task: TaskDifficulty;
  steps: number;
  totalReward: number;
  finalPhase: string;
  curriculumLevel: number;
}

export interface EpisodeHistoryEntry {
  id: number;
  task: TaskDifficulty;
  steps: number;
  avgReward: number;
  totalReward: number;
  finalPhase: string;
  passed: boolean;
}

export type ObsHistory = Record<keyof AEPOObservation, number[]>;
