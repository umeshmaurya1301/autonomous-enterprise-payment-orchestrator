package aepo.types;

import java.util.LinkedHashMap;
import java.util.Map;

import static aepo.types.ObsBounds.*;

/**
 * Ten-field typed observation — Java mirror of {@code AEPOObservation} (Pydantic BaseModel).
 *
 * <p>Stored as raw values; agent always consumes {@link #normalized()} (each
 * field divided by its max so all values are in [0.0, 1.0]). The compact
 * constructor enforces the same range checks Pydantic's {@code Field(ge=, le=)}
 * does — invalid construction throws {@link IllegalArgumentException} immediately
 * (mapped to HTTP 422 by Spring).
 *
 * <p>Rationale for {@code record}: matches Pydantic's "data + validation, no
 * behaviour" contract. Adding methods to a record is fine; mutating fields is not.
 */
public record AEPOObservation(
        double channel,
        double riskScore,
        double adversaryThreatLevel,
        double systemEntropy,
        double kafkaLag,
        double apiLatency,
        double rollingP99,
        double dbConnectionPool,
        double bankApiStatus,
        double merchantTier
) {

    /**
     * Compact constructor — runs every time the record is built. Mirrors Pydantic's
     * {@code Field(ge=0.0, le=X)} bounds. Out-of-range values raise immediately
     * (Spring's exception handler converts to 422 Unprocessable Entity).
     */
    public AEPOObservation {
        require(channel,              0.0, CHANNEL_MAX,        "channel");
        require(riskScore,            0.0, RISK_MAX,           "riskScore");
        require(adversaryThreatLevel, 0.0, ADV_THREAT_MAX,     "adversaryThreatLevel");
        require(systemEntropy,        0.0, ENTROPY_MAX,        "systemEntropy");
        require(kafkaLag,             0.0, LAG_MAX,            "kafkaLag");
        require(apiLatency,           0.0, LATENCY_MAX,        "apiLatency");
        require(rollingP99,           0.0, P99_MAX,            "rollingP99");
        require(dbConnectionPool,     0.0, DB_POOL_MAX,        "dbConnectionPool");
        require(bankApiStatus,        0.0, BANK_STATUS_MAX,    "bankApiStatus");
        require(merchantTier,         0.0, MERCHANT_TIER_MAX,  "merchantTier");
    }

    /**
     * Return all 10 fields normalized to [0.0, 1.0] for agent consumption.
     *
     * <p>Keys are snake_case to match the Python contract — the agent code (or a
     * trained policy server) sees the same JSON whether it talks to Java or Python.
     * LinkedHashMap preserves insertion order so the obs vector ordering is stable.
     */
    public Map<String, Double> normalized() {
        Map<String, Double> m = new LinkedHashMap<>(10);
        m.put("transaction_type",       clamp(channel,              0.0, CHANNEL_MAX)        / CHANNEL_MAX);
        m.put("risk_score",             clamp(riskScore,            0.0, RISK_MAX)           / RISK_MAX);
        m.put("adversary_threat_level", clamp(adversaryThreatLevel, 0.0, ADV_THREAT_MAX)     / ADV_THREAT_MAX);
        m.put("system_entropy",         clamp(systemEntropy,        0.0, ENTROPY_MAX)        / ENTROPY_MAX);
        m.put("kafka_lag",              clamp(kafkaLag,             0.0, LAG_MAX)            / LAG_MAX);
        m.put("api_latency",            clamp(apiLatency,           0.0, LATENCY_MAX)        / LATENCY_MAX);
        m.put("rolling_p99",            clamp(rollingP99,           0.0, P99_MAX)            / P99_MAX);
        m.put("db_connection_pool",     clamp(dbConnectionPool,     0.0, DB_POOL_MAX)        / DB_POOL_MAX);
        m.put("bank_api_status",        clamp(bankApiStatus,        0.0, BANK_STATUS_MAX)    / BANK_STATUS_MAX);
        m.put("merchant_tier",          clamp(merchantTier,         0.0, MERCHANT_TIER_MAX)  / MERCHANT_TIER_MAX);
        return m;
    }

    /**
     * Serialize to a 10-element double[] in the canonical order. Mirrors
     * {@code AEPOObservation.to_array()} — used by anything that needs a raw vector
     * (e.g., a learned world model that consumes a flat array).
     */
    public double[] toArray() {
        return new double[]{
                channel, riskScore, adversaryThreatLevel, systemEntropy, kafkaLag,
                apiLatency, rollingP99, dbConnectionPool, bankApiStatus, merchantTier
        };
    }

    private static void require(double v, double lo, double hi, String field) {
        if (v < lo || v > hi || Double.isNaN(v)) {
            throw new IllegalArgumentException(
                    "AEPOObservation." + field + " out of range: " + v +
                            " (expected [" + lo + ", " + hi + "])"
            );
        }
    }
}
