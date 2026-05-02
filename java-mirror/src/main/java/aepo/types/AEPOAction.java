package aepo.types;

/**
 * Six-field typed action — Java mirror of {@code AEPOAction} (Pydantic BaseModel).
 *
 * <p>Action layout (matches CLAUDE.md spec exactly):
 * <pre>
 *   risk_decision     {0=Approve, 1=Reject, 2=Challenge}
 *   crypto_verify     {0=FullVerify, 1=SkipVerify}
 *   infra_routing     {0=Normal, 1=Throttle, 2=CircuitBreaker}
 *   db_retry_policy   {0=FailFast, 1=ExponentialBackoff}
 *   settlement_policy {0=StandardSync, 1=DeferredAsyncFallback}
 *   app_priority      {0=UPI, 1=Credit, 2=Balanced}
 * </pre>
 *
 * <p>Compact constructor enforces ranges. Out-of-range integers (e.g., risk_decision=9)
 * throw immediately — Spring maps this to HTTP 422, identical to Pydantic ValidationError.
 */
public record AEPOAction(
        int riskDecision,
        int cryptoVerify,
        int infraRouting,
        int dbRetryPolicy,
        int settlementPolicy,
        int appPriority
) {

    // ── Action constants ────────────────────────────────────────────────
    // Public so reward / heuristic code can refer to RISK_REJECT instead of
    // a magic 1. Keeps causal-transition reading natural to a Java engineer.
    public static final int RISK_APPROVE   = 0;
    public static final int RISK_REJECT    = 1;
    public static final int RISK_CHALLENGE = 2;

    public static final int CRYPTO_FULL = 0;
    public static final int CRYPTO_SKIP = 1;

    public static final int INFRA_NORMAL = 0;
    public static final int INFRA_THROTTLE = 1;
    public static final int INFRA_CIRCUIT_BREAKER = 2;

    public static final int DB_FAIL_FAST = 0;
    public static final int DB_EXPONENTIAL_BACKOFF = 1;

    public static final int SETTLE_STANDARD_SYNC = 0;
    public static final int SETTLE_DEFERRED_ASYNC = 1;

    public static final int APP_UPI = 0;
    public static final int APP_CREDIT = 1;
    public static final int APP_BALANCED = 2;

    public AEPOAction {
        requireRange(riskDecision,    0, 2, "riskDecision");
        requireRange(cryptoVerify,    0, 1, "cryptoVerify");
        requireRange(infraRouting,    0, 2, "infraRouting");
        requireRange(dbRetryPolicy,   0, 1, "dbRetryPolicy");
        requireRange(settlementPolicy,0, 1, "settlementPolicy");
        requireRange(appPriority,     0, 2, "appPriority");
    }

    /**
     * Convenience factory mirroring Python's keyword-only construction —
     * applies the same defaults as the Pydantic model
     * ({@code db_retry_policy=0, settlement_policy=0, app_priority=2}).
     */
    public static AEPOAction of(int riskDecision, int cryptoVerify, int infraRouting) {
        return new AEPOAction(riskDecision, cryptoVerify, infraRouting,
                DB_FAIL_FAST, SETTLE_STANDARD_SYNC, APP_BALANCED);
    }

    /** Serialize to a 6-element int[] — mirrors {@code AEPOAction.to_array()}. */
    public int[] toArray() {
        return new int[]{ riskDecision, cryptoVerify, infraRouting,
                dbRetryPolicy, settlementPolicy, appPriority };
    }

    private static void requireRange(int v, int lo, int hi, String field) {
        if (v < lo || v > hi) {
            throw new IllegalArgumentException(
                    "AEPOAction." + field + " out of range: " + v +
                            " (expected [" + lo + ", " + hi + "])"
            );
        }
    }
}
