package aepo;

/**
 * PYTHON EQUIVALENT: unified_gateway.py → AEPOAction (Pydantic BaseModel)
 *
 * Six-field typed action for the Autonomous Enterprise Payment Orchestrator.
 * Java record mirrors the Pydantic model: compact constructor validates ranges,
 * mirroring Pydantic Field(ge=..., le=...) constraints.
 *
 * Fields db_retryPolicy, settlementPolicy, appPriority have safe defaults in
 * Python (via Field(default=...)); Java callers must supply all 6 values or
 * use the convenience factory methods below.
 */
public record AEPOAction(
    // ── Risk layer ─────────────────────────────────────────────────────────────
    int riskDecision,     // 0=Approve, 1=Reject, 2=Challenge           {0,1,2}
    int cryptoVerify,     // 0=FullVerify, 1=SkipVerify                  {0,1}
    // ── Infrastructure layer ───────────────────────────────────────────────────
    int infraRouting,     // 0=Normal, 1=Throttle, 2=CircuitBreaker      {0,1,2}
    int dbRetryPolicy,    // 0=FailFast, 1=ExponentialBackoff             {0,1}
    // ── Business layer ─────────────────────────────────────────────────────────
    int settlementPolicy, // 0=StandardSync, 1=DeferredAsyncFallback     {0,1}
    int appPriority       // 0=UPI, 1=Credit, 2=Balanced                 {0,1,2}
) {

    // PYTHON EQUIVALENT: Pydantic Field(ge=..., le=...) — raises ValidationError on violation.
    // Java compact record constructor — throws IllegalArgumentException on violation.
    public AEPOAction {
        validateRange("riskDecision",    riskDecision,    0, 2);
        validateRange("cryptoVerify",    cryptoVerify,    0, 1);
        validateRange("infraRouting",    infraRouting,    0, 2);
        validateRange("dbRetryPolicy",   dbRetryPolicy,   0, 1);
        validateRange("settlementPolicy",settlementPolicy,0, 1);
        validateRange("appPriority",     appPriority,     0, 2);
    }

    /**
     * PYTHON EQUIVALENT: AEPOAction(risk_decision=r, crypto_verify=c, infra_routing=i)
     * Factory for legacy 3-field construction; fills new fields with safe defaults.
     * Safe defaults mirror the Python Field(default=...) values.
     */
    public static AEPOAction withDefaults(int riskDecision, int cryptoVerify, int infraRouting) {
        return new AEPOAction(
            riskDecision,
            cryptoVerify,
            infraRouting,
            0,  // dbRetryPolicy  = FailFast  (PYTHON: default=0)
            0,  // settlementPolicy = StandardSync (PYTHON: default=0)
            2   // appPriority    = Balanced  (PYTHON: default=2)
        );
    }

    // PYTHON EQUIVALENT: pydantic ValidationError with ge/le constraints
    private static void validateRange(String field, int val, int lo, int hi) {
        if (val < lo || val > hi) {
            throw new IllegalArgumentException(
                String.format("AEPOAction.%s = %d is outside [%d, %d]", field, val, lo, hi)
            );
        }
    }
}
