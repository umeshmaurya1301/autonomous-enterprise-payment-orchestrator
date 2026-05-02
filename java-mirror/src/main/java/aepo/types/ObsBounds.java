package aepo.types;

/**
 * Observation-space bounds — Java mirror of {@code aepo_types.py} module-level constants.
 *
 * <p>Lives in a separate utility class so that both the data records (validation
 * in compact constructors) and the env (clipping inside step()) reference the
 * same numbers. Changing a bound in one place changes it everywhere.
 *
 * <p>PYTHON EQUIVALENT:
 * <pre>
 *   CHANNEL_MAX: float = 2.0
 *   RISK_MAX: float = 100.0
 *   ...etc
 * </pre>
 */
public final class ObsBounds {

    private ObsBounds() { /* constants only */ }

    /** Payment channel ID — {0=P2P, 1=P2M, 2=AutoPay}. */
    public static final double CHANNEL_MAX = 2.0;

    /** Fraud risk score raw range. */
    public static final double RISK_MAX = 100.0;

    /** Adversary escalation level — capped at 10 by Transition #7. */
    public static final double ADV_THREAT_MAX = 10.0;

    /** System entropy index — driven by lag (Transition #9). */
    public static final double ENTROPY_MAX = 100.0;

    /** Kafka consumer lag in messages. CRASH_THRESHOLD lives in EnvConstants. */
    public static final double LAG_MAX = 10000.0;

    /** Bank API per-step latency in ms. */
    public static final double LATENCY_MAX = 5000.0;

    /** EMA-smoothed P99 latency in ms. SLA_BREACH_THRESHOLD lives in EnvConstants. */
    public static final double P99_MAX = 5000.0;

    /** DB connection pool utilisation %. */
    public static final double DB_POOL_MAX = 100.0;

    /** Bank API status — {0=Healthy, 1=Degraded, 2=Down}. */
    public static final double BANK_STATUS_MAX = 2.0;

    /** Merchant tier — {0=Small, 1=Enterprise}. */
    public static final double MERCHANT_TIER_MAX = 1.0;

    /**
     * Clamp helper used by both records (validation) and the env (post-mutation
     * clipping). Mirrors {@code numpy.clip(x, lo, hi)}.
     */
    public static double clamp(double v, double lo, double hi) {
        return Math.min(hi, Math.max(lo, v));
    }
}
