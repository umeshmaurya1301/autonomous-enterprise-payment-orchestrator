package aepo;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * PYTHON EQUIVALENT: unified_gateway.py → AEPOObservation (Pydantic BaseModel)
 *
 * Ten-field typed observation for the Autonomous Enterprise Payment Orchestrator.
 * Java record mirrors the Pydantic model: constructor validation corresponds to
 * Pydantic Field(ge=..., le=...) constraints; normalize() mirrors .normalized().
 *
 * Fields [2]–[9] are observed-but-inert in Phase 2.
 * Full causal wiring in Phase 5.
 */
public record AEPOObservation(
    // ── Risk layer ─────────────────────────────────────────────────────────────
    double channel,               // payment channel {0,1,2}    raw [0.0, 2.0]
    double riskScore,             // fraud risk signal           raw [0.0, 100.0]
    double adversaryThreatLevel,  // adversary escalation        raw [0.0, 10.0]   ★ Phase 2
    double systemEntropy,         // system entropy index        raw [0.0, 100.0]  ★ Phase 2
    // ── Infrastructure layer ───────────────────────────────────────────────────
    double kafkaLag,              // Kafka consumer lag (msgs)   raw [0.0, 10000.0]
    double apiLatency,            // bank API latency (ms)       raw [0.0, 5000.0]
    double rollingP99,            // EMA P99 SLA (ms)            raw [0.0, 5000.0]
    double dbConnectionPool,      // DB pool utilization         raw [0.0, 100.0]  ★ Phase 2
    // ── Business layer ─────────────────────────────────────────────────────────
    double bankApiStatus,         // bank status {0=Healthy,1=Degraded,2=Unknown} ★ Phase 2
    // POMDP Tweak #3: merchantTier may be 0.5 (MERCHANT_TIER_UNKNOWN) when the
    // environment hides the true tier. Range is [0.0, 1.0] — 0.5 is a valid sentinel.
    // PYTHON EQUIVALENT: merchant_tier: float = Field(ge=0.0, le=MERCHANT_TIER_MAX)
    //   where MERCHANT_TIER_UNKNOWN = 0.5
    double merchantTier           // 0=Small, 1=Enterprise, 0.5=UNKNOWN (POMDP-hidden 30% of steps)
) {

    // PYTHON EQUIVALENT: Pydantic Field(ge=..., le=...) — raises ValidationError on violation.
    // Java compact record constructor — throws IllegalArgumentException on violation.
    public AEPOObservation {
        validate("channel",               channel,              0.0,  2.0);
        validate("riskScore",             riskScore,            0.0,  100.0);
        validate("adversaryThreatLevel",  adversaryThreatLevel, 0.0,  10.0);
        validate("systemEntropy",         systemEntropy,        0.0,  100.0);
        validate("kafkaLag",              kafkaLag,             0.0,  10000.0);
        validate("apiLatency",            apiLatency,           0.0,  5000.0);
        validate("rollingP99",            rollingP99,           0.0,  5000.0);
        validate("dbConnectionPool",      dbConnectionPool,     0.0,  100.0);
        validate("bankApiStatus",         bankApiStatus,        0.0,  2.0);
        validate("merchantTier",          merchantTier,         0.0,  1.0);  // 0.5 is valid (unknown sentinel)
    }

    /**
     * PYTHON EQUIVALENT: AEPOObservation.normalized()
     *
     * Returns all 10 fields normalized to [0.0, 1.0].
     * Raw values are clipped before division (defence-in-depth).
     *
     * Key mapping:
     *   "channel" field → "transaction_type" key (AEPO spec naming)
     *   bank_api_status: 0 → 0.0, 1 → 0.5, 2 → 1.0  (÷ BANK_STATUS_MAX=2)
     *
     * PYTHON EQUIVALENT: numpy.clip(val, lo, hi) → Math.min(hi, Math.max(lo, val))
     */
    public Map<String, Double> normalized() {
        Map<String, Double> result = new LinkedHashMap<>();
        // 'channel' stored for Phase 2 backward compat; key uses AEPO spec name
        result.put("transaction_type",        clip(channel,              0.0, 2.0)     / 2.0);
        result.put("risk_score",              clip(riskScore,            0.0, 100.0)   / 100.0);
        result.put("adversary_threat_level",  clip(adversaryThreatLevel, 0.0, 10.0)    / 10.0);
        result.put("system_entropy",          clip(systemEntropy,        0.0, 100.0)   / 100.0);
        result.put("kafka_lag",               clip(kafkaLag,             0.0, 10000.0) / 10000.0);
        result.put("api_latency",             clip(apiLatency,           0.0, 5000.0)  / 5000.0);
        result.put("rolling_p99",             clip(rollingP99,           0.0, 5000.0)  / 5000.0);
        result.put("db_connection_pool",      clip(dbConnectionPool,     0.0, 100.0)   / 100.0);
        result.put("bank_api_status",         clip(bankApiStatus,        0.0, 2.0)     / 2.0);
        result.put("merchant_tier",           clip(merchantTier,         0.0, 1.0)     / 1.0);
        return result;
    }

    // PYTHON EQUIVALENT: float(numpy.clip(val, lo, hi))
    private static double clip(double val, double lo, double hi) {
        return Math.min(hi, Math.max(lo, val));
    }

    // PYTHON EQUIVALENT: pydantic ValidationError with ge/le constraints
    private static void validate(String field, double val, double lo, double hi) {
        if (val < lo || val > hi) {
            throw new IllegalArgumentException(
                String.format("AEPOObservation.%s = %.4f is outside [%.1f, %.1f]", field, val, lo, hi)
            );
        }
    }
}
