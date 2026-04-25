package aepo;

import java.util.Map;

/**
 * HeuristicAgent.java — Phase 8 Java Mirror of heuristic_policy() in graders.py
 * ================================================================================
 * Intentionally-incomplete baseline policy for the Autonomous Enterprise
 * Payment Orchestrator (AEPO).
 *
 * PYTHON EQUIVALENT:
 *   graders.py — heuristic_policy(obs_normalized: dict[str, float]) -> AEPOAction
 *
 * This agent has THREE DELIBERATE BLIND SPOTS that the trained RL agent must
 * discover. These blind spots are the core learning story of the AEPO pitch.
 *
 * WHAT THE HEURISTIC COVERS (correctly):
 *   - risk_score > 0.8  → Reject + FullVerify      (suboptimal: FullVerify wastes lag)
 *   - kafka_lag > 0.3   → Throttle                  (prevents lag crash)
 *   - rolling_p99 > 0.6 → DeferredAsyncFallback, else StandardSync
 *   - db_retry_policy   = ExponentialBackoff always  (ignores pool level — BLIND SPOT #3)
 *   - app_priority      = Balanced always            (ignores merchant_tier — BLIND SPOT #2)
 *
 * BLIND SPOTS (what the trained agent must learn):
 *   #1: Reject + SkipVerify on high-risk → +0.04 bonus, saves 250 lag/step
 *       Heuristic uses FullVerify (+0.03 bonus, +150 lag/step instead)
 *   #2: app_priority should match merchant_tier → +0.02/step bonus
 *       Heuristic always uses Balanced (2) regardless of tier
 *   #3: ExponentialBackoff when db_pool < 0.2 → -0.10 penalty (pool has spare capacity)
 *       Heuristic always uses ExponentialBackoff, never checks pool level
 *
 * Blind spot #1 is the PRIMARY LEARNING STORY for the pitch:
 *   "Here's what the agent learned that our heuristic didn't."
 *
 * NOTE: Delete this file (along with all /java-mirror/) before final submission.
 */
public class HeuristicAgent {

    // ---------------------------------------------------------------------------
    // Named constants matching graders.py heuristic thresholds
    // ---------------------------------------------------------------------------

    /** Normalized risk_score above which the heuristic Rejects. */
    private static final double HIGH_RISK_THRESHOLD = 0.8;

    /**
     * Normalized kafka_lag above which the heuristic Throttles.
     * Set to 0.3 (not the spec's 0.6) to prevent crash before the normalized
     * crash cliff at 0.4 (raw 4000 / max 10000).
     */
    private static final double LAG_THROTTLE_THRESHOLD = 0.3;

    /** Normalized rolling_p99 above which the heuristic uses DeferredAsyncFallback. */
    private static final double P99_DEFERRED_THRESHOLD = 0.6;

    // ---------------------------------------------------------------------------
    // Core policy method
    // PYTHON EQUIVALENT: def heuristic_policy(obs_normalized: dict[str, float]) -> AEPOAction
    // ---------------------------------------------------------------------------

    /**
     * Decide the next action for the given normalized observation.
     *
     * All observation fields are expected in [0.0, 1.0] (output of
     * AEPOObservation.normalized()).
     *
     * // PYTHON EQUIVALENT:
     * // def heuristic_policy(obs_normalized: dict[str, float]) -> AEPOAction:
     * //     risk_score  = obs_normalized.get("risk_score", 0.0)
     * //     kafka_lag   = obs_normalized.get("kafka_lag", 0.0)
     * //     rolling_p99 = obs_normalized.get("rolling_p99", 0.0)
     * //
     * //     if risk_score > 0.8:
     * //         risk_decision = 1    # Reject
     * //         crypto_verify = 0    # FullVerify — BLIND SPOT #1 (SkipVerify is better)
     * //     else:
     * //         risk_decision = 0    # Approve
     * //         crypto_verify = 1    # SkipVerify
     * //
     * //     if kafka_lag > 0.3:
     * //         infra_routing = 1    # Throttle
     * //     else:
     * //         infra_routing = 0    # Normal
     * //
     * //     db_retry_policy   = 1  # ExponentialBackoff always — BLIND SPOT #3
     * //     settlement_policy = 1 if rolling_p99 > 0.6 else 0
     * //     app_priority      = 2  # Balanced always — BLIND SPOT #2
     *
     * @param obsNormalized Map of 10 normalized observation fields (all in [0.0, 1.0])
     * @return AEPOAction with all 6 fields set
     */
    public AEPOAction act(Map<String, Double> obsNormalized) {
        double riskScore  = obsNormalized.getOrDefault("risk_score",  0.0);
        double kafkaLag   = obsNormalized.getOrDefault("kafka_lag",   0.0);
        double rollingP99 = obsNormalized.getOrDefault("rolling_p99", 0.0);

        // ── Risk decision + crypto verify ────────────────────────────────────
        // BLIND SPOT #1: On high-risk, the optimal action is Reject + SkipVerify
        // (+0.04 bonus, saves 250 lag/step). The heuristic uses FullVerify instead
        // (+0.03 bonus, adds +150 kafka_lag per step).
        final int riskDecision;
        final int cryptoVerify;

        if (riskScore > HIGH_RISK_THRESHOLD) {
            riskDecision = 1;  // Reject
            cryptoVerify = 0;  // FullVerify — BLIND SPOT #1
        } else {
            riskDecision = 0;  // Approve
            cryptoVerify = 1;  // SkipVerify
        }

        // ── Infrastructure routing — lag-driven ──────────────────────────────
        // Throttle before the crash cliff (crash at normalized 0.4 = raw 4000)
        final int infraRouting = (kafkaLag > LAG_THROTTLE_THRESHOLD) ? 1 : 0;

        // ── DB retry policy — BLIND SPOT #3 ─────────────────────────────────
        // ExponentialBackoff always. Never checks db_connection_pool level.
        // When pool < 0.2 (spare capacity), this costs -0.10/step.
        // The trained agent should learn to use FailFast (0) when pool < 0.2.
        final int dbRetryPolicy = 1;  // ExponentialBackoff always — BLIND SPOT #3

        // ── Settlement policy — P99-driven ───────────────────────────────────
        final int settlementPolicy = (rollingP99 > P99_DEFERRED_THRESHOLD) ? 1 : 0;

        // ── App priority — BLIND SPOT #2 ─────────────────────────────────────
        // Balanced (2) always. Never checks merchant_tier.
        // merchant_tier=Small (0.0)      → optimal is UPI (0)     → +0.02/step
        // merchant_tier=Enterprise (1.0) → optimal is Credit (1)  → +0.02/step
        // The trained agent should learn to match priority to tier.
        final int appPriority = 2;  // Balanced always — BLIND SPOT #2

        return new AEPOAction(
            riskDecision,
            cryptoVerify,
            infraRouting,
            dbRetryPolicy,
            settlementPolicy,
            appPriority
        );
    }

    // ---------------------------------------------------------------------------
    // Blind spot documentation — readable by a Java/Spring Boot engineer
    // ---------------------------------------------------------------------------

    /**
     * Returns a human-readable summary of what the heuristic gets wrong.
     * Used for pitch deck narrative and README documentation.
     *
     * // PYTHON EQUIVALENT: (inline docstring in graders.py heuristic_policy)
     */
    public static String blindSpotSummary() {
        return """
            HEURISTIC AGENT — 3 DELIBERATE BLIND SPOTS
            ============================================

            BLIND SPOT #1 (Primary learning story):
              Heuristic: Reject + FullVerify on high-risk
                → +0.03 bonus, but adds +150 kafka_lag per step
              Optimal:   Reject + SkipVerify on high-risk
                → +0.04 bonus AND saves 250 lag units per step (net +400 lag saving)
              Impact: ~+0.01 reward + avoids lag cascade. Agent must discover this.

            BLIND SPOT #2:
              Heuristic: app_priority = Balanced (2) always
              Optimal:   UPI (0) for Small merchants, Credit (1) for Enterprise
                → +0.02/step bonus when matched
              Impact: Heuristic leaves +0.02/step on the table every step.

            BLIND SPOT #3:
              Heuristic: ExponentialBackoff (db_retry_policy=1) always
              Optimal:   FailFast (0) when db_pool < 0.2 (spare capacity)
                → ExponentialBackoff when pool < 0.2 costs -0.10/step
              Impact: Every low-pool step costs -0.10 the heuristic pays unnecessarily.
            """;
    }
}
