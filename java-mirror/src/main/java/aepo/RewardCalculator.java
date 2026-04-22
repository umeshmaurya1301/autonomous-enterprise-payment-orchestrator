package aepo;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * PYTHON EQUIVALENT: UnifiedFintechEnv.step() — reward calculation block (③)
 *
 * Stateless helper: given a snapshot of the current observation fields,
 * the chosen action, and session-level counters, returns a typed reward
 * breakdown identical to the Python reward_breakdown dict.
 *
 * All threshold constants mirror unified_gateway.py exactly.
 */
public final class RewardCalculator {

    // ── Observation thresholds — mirror unified_gateway.py constants ──────
    private static final double CRASH_THRESHOLD      = 4000.0;
    private static final double SLA_BREACH_THRESHOLD = 800.0;
    private static final double SLA_PROXIMITY_LOWER  = 500.0;
    private static final double LAG_PROXIMITY_LOWER  = 3000.0;
    private static final double HIGH_RISK_THRESHOLD  = 80.0;

    // ── Action value constants ─────────────────────────────────────────────
    private static final int RISK_APPROVE    = 0;
    private static final int RISK_REJECT     = 1;
    private static final int RISK_CHALLENGE  = 2;
    private static final int CRYPTO_FULL     = 0;
    private static final int CRYPTO_SKIP     = 1;
    private static final int INFRA_NORMAL    = 0;
    private static final int INFRA_THROTTLE  = 1;
    private static final int INFRA_CB        = 2;
    private static final int DB_FAILFAST     = 0;
    private static final int DB_BACKOFF      = 1;
    private static final int SETTLE_SYNC     = 0;
    private static final int SETTLE_DEFERRED = 1;
    private static final int APP_UPI         = 0;
    private static final int APP_CREDIT      = 1;

    // PYTHON EQUIVALENT: dataclass / TypedDict returned from reward block
    public record RewardBreakdown(
        double base,
        double fraudPenalty,
        double slaPenalty,
        double infraPenalty,
        double dbPenalty,
        double settlementPenalty,
        double bonus,
        double finalReward,
        boolean isFraudCatastrophe,
        boolean crashed,
        boolean blindSpotTriggered
    ) {
        /** Sum of all components, before clamp (mirrors Python raw_reward). */
        public double rawReward() {
            return base + fraudPenalty + slaPenalty + infraPenalty
                    + dbPenalty + settlementPenalty + bonus;
        }

        /** Serialise to a Map<String,Object> for Spring Boot @ResponseBody / info dict. */
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("base",               base);
            m.put("fraud_penalty",      fraudPenalty);
            m.put("sla_penalty",        slaPenalty);
            m.put("infra_penalty",      infraPenalty);
            m.put("db_penalty",         dbPenalty);
            m.put("settlement_penalty", settlementPenalty);
            m.put("bonus",              bonus);
            m.put("final",              finalReward);
            return m;
        }
    }

    private RewardCalculator() {}   // utility class — no instantiation

    /**
     * PYTHON EQUIVALENT: step() reward block (③) in UnifiedFintechEnv.
     *
     * @param riskScore           obs.risk_score
     * @param kafkaLag            obs.kafka_lag
     * @param rollingP99          obs.rolling_p99
     * @param dbPool              obs.db_connection_pool
     * @param bankStatus          obs.bank_api_status  (0=Healthy, 1=Degraded)
     * @param merchantTier        obs.merchant_tier    (0=Small, 1=Enterprise)
     * @param currentPhase        "normal" | "spike" | "attack" | "recovery"
     * @param action              validated AEPOAction
     * @param consecutiveDeferred session counter for consecutive DeferredAsync steps
     * @param circuitBreakerTripped whether CB fired this step (accumulator already reset)
     */
    public static RewardBreakdown calculate(
            double riskScore,
            double kafkaLag,
            double rollingP99,
            double dbPool,
            double bankStatus,
            double merchantTier,
            String currentPhase,
            AEPOAction action,
            int consecutiveDeferred,
            boolean circuitBreakerTripped
    ) {
        double base               = 0.8;
        double fraudPenalty       = 0.0;
        double slaPenalty         = 0.0;
        double infraPenalty       = 0.0;
        double dbPenalty          = 0.0;
        double settlementPenalty  = 0.0;
        double bonus              = 0.0;
        boolean blindSpotTriggered = false;

        // ── Catastrophic fraud (highest priority) ─────────────────────────
        boolean isFraudCatastrophe = action.riskDecision() == RISK_APPROVE
                && action.cryptoVerify() == CRYPTO_SKIP
                && riskScore > HIGH_RISK_THRESHOLD;
        if (isFraudCatastrophe) {
            fraudPenalty = -base;   // cancels base so sum = 0.0
        }

        // ── System crash ───────────────────────────────────────────────────
        boolean crashed = kafkaLag > CRASH_THRESHOLD;

        // ── SLA penalty ────────────────────────────────────────────────────
        if (rollingP99 > SLA_BREACH_THRESHOLD) {
            slaPenalty = -0.30;
        } else if (rollingP99 > SLA_PROXIMITY_LOWER) {
            double prox = (rollingP99 - SLA_PROXIMITY_LOWER)
                    / (SLA_BREACH_THRESHOLD - SLA_PROXIMITY_LOWER);
            slaPenalty = round4(-0.10 * prox);
        }

        // ── Lag proximity ──────────────────────────────────────────────────
        if (kafkaLag > LAG_PROXIMITY_LOWER && kafkaLag <= CRASH_THRESHOLD) {
            double prox = (kafkaLag - LAG_PROXIMITY_LOWER)
                    / (CRASH_THRESHOLD - LAG_PROXIMITY_LOWER);
            infraPenalty += round4(-0.10 * prox);
        }

        // ── Infra routing ──────────────────────────────────────────────────
        if (action.infraRouting() == INFRA_THROTTLE) {
            infraPenalty += "spike".equals(currentPhase) ? -0.10 : -0.20;
        } else if (action.infraRouting() == INFRA_CB) {
            infraPenalty += -0.50;
        }

        // ── DB retry policy ────────────────────────────────────────────────
        if (action.dbRetryPolicy() == DB_BACKOFF) {
            if (dbPool > 80) {
                dbPenalty = 0.03;   // PYTHON EQUIVALENT: db_penalty += 0.03 bonus
            } else if (dbPool < 20) {
                dbPenalty = -0.10;
            }
        }

        // ── Settlement policy ──────────────────────────────────────────────
        if (action.settlementPolicy() == SETTLE_DEFERRED) {
            // consecutiveDeferred has already been incremented by the caller
            if (bankStatus == 1.0) {
                settlementPenalty += 0.04;          // correct use when Degraded
            } else if ("normal".equals(currentPhase)) {
                settlementPenalty += -0.15;         // wasteful in normal phase
            }
            if (consecutiveDeferred >= 5) {
                settlementPenalty += -0.20;         // over-reliance penalty
            }
        }

        // ── Risk / crypto bonuses ──────────────────────────────────────────
        if (riskScore > HIGH_RISK_THRESHOLD) {
            if (action.riskDecision() == RISK_CHALLENGE) {
                bonus += 0.05;
            }
            if (action.cryptoVerify() == CRYPTO_FULL) {
                bonus += 0.03;
            }
            if (action.riskDecision() == RISK_REJECT && action.cryptoVerify() == CRYPTO_SKIP) {
                // PYTHON EQUIVALENT: blind spot #1 — Reject+SkipVerify is safe AND saves lag
                bonus += 0.04;
                blindSpotTriggered = true;
            }
        }

        // ── App priority / merchant-tier alignment ─────────────────────────
        if (action.appPriority() == APP_UPI && merchantTier == 0.0) {
            bonus += 0.02;
        } else if (action.appPriority() == APP_CREDIT && merchantTier == 1.0) {
            bonus += 0.02;
        }

        // ── Compute final reward ───────────────────────────────────────────
        double rawReward = base + fraudPenalty + slaPenalty + infraPenalty
                + dbPenalty + settlementPenalty + bonus;
        double finalReward;
        if (crashed || isFraudCatastrophe) {
            finalReward = 0.0;
        } else {
            // PYTHON EQUIVALENT: max(0.0, min(1.0, raw_reward))
            finalReward = Math.min(1.0, Math.max(0.0, rawReward));
        }

        return new RewardBreakdown(
                base, fraudPenalty, slaPenalty,
                round4(infraPenalty), dbPenalty, round4(settlementPenalty),
                round4(bonus), finalReward,
                isFraudCatastrophe, crashed, blindSpotTriggered
        );
    }

    private static double round4(double v) {
        return Math.round(v * 10_000.0) / 10_000.0;
    }
}
