package aepo.agents;

import aepo.types.AEPOAction;

import java.util.Map;

/**
 * Hand-crafted heuristic baseline policy — Java mirror of the heuristic in CLAUDE.md
 * and {@code train.py}.
 *
 * <p><b>Intentionally incomplete:</b> three deliberate <i>blind spots</i> the
 * trained agent must discover. This is the central learning story of the pitch:
 * <ol>
 *   <li><b>Reject + SkipVerify</b> on high-risk → +0.04 bonus, saves 250 lag/step
 *       (heuristic uses FullVerify here — sensible but suboptimal).</li>
 *   <li><b>app_priority should match merchant_tier</b> → +0.02/step (heuristic
 *       always uses Balanced, leaving the bonus on the table).</li>
 *   <li><b>ExponentialBackoff when pool &lt; 20</b> → -0.10 (heuristic never checks
 *       pool level, so it is sometimes punished here).</li>
 * </ol>
 *
 * <p>Pure stateless policy → safe to share across threads.
 */
public final class HeuristicAgent {

    /**
     * Map a normalized observation dict (the same payload the agent receives) →
     * AEPOAction. Pure function — no internal state.
     */
    public AEPOAction act(Map<String, Double> obs) {
        double risk = obs.getOrDefault("risk_score", 0.0);
        double lag  = obs.getOrDefault("kafka_lag", 0.0);
        double p99  = obs.getOrDefault("rolling_p99", 0.0);

        int riskDecision;
        int cryptoVerify;
        int infraRouting;
        int settlementPolicy;

        // Risk gate: high risk → reject + FullVerify (suboptimal; blind spot #1).
        if (risk > 0.8) {
            riskDecision = AEPOAction.RISK_REJECT;
            cryptoVerify = AEPOAction.CRYPTO_FULL;
        } else {
            riskDecision = AEPOAction.RISK_APPROVE;
            cryptoVerify = AEPOAction.CRYPTO_FULL;
        }

        // Lag gate: lag > 60% of LAG_MAX → throttle.
        infraRouting = (lag > 0.6) ? AEPOAction.INFRA_THROTTLE : AEPOAction.INFRA_NORMAL;

        // P99 gate: p99 > 60% of P99_MAX → defer settlement.
        settlementPolicy = (p99 > 0.6)
                ? AEPOAction.SETTLE_DEFERRED_ASYNC
                : AEPOAction.SETTLE_STANDARD_SYNC;

        // Blind spots #2 and #3: always Balanced, always Backoff.
        return new AEPOAction(
                riskDecision,
                cryptoVerify,
                infraRouting,
                AEPOAction.DB_EXPONENTIAL_BACKOFF,    // blind spot #3 — ignores pool
                settlementPolicy,
                AEPOAction.APP_BALANCED               // blind spot #2 — ignores tier
        );
    }
}
