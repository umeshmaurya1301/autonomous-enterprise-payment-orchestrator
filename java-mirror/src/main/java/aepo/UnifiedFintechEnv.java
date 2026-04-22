package aepo;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;

/**
 * UnifiedFintechEnv — Java mirror of unified_gateway.py (Phase 5 + Phase 6)
 * ==========================================================================
 * Gymnasium-compatible environment modelling a unified fintech risk gateway.
 *
 * This Java mirror preserves class names, method names, and variable names
 * from the Python implementation. It does NOT compile or run — it exists to
 * be readable to a Java developer who thinks in Spring Boot.
 *
 * PYTHON EQUIVALENT: gymnasium.Env subclass with Box(10,) obs and MultiDiscrete([3,2,3,2,2,3]) action.
 */
public class UnifiedFintechEnv {

    private static final Logger log = Logger.getLogger(UnifiedFintechEnv.class.getName());

    // ── Named constants — observation bounds ────────────────────────────
    public static final double CHANNEL_MAX = 2.0;
    public static final double RISK_MAX = 100.0;
    public static final double ADV_THREAT_MAX = 10.0;
    public static final double ENTROPY_MAX = 100.0;
    public static final double LAG_MAX = 10000.0;
    public static final double LATENCY_MAX = 5000.0;
    public static final double P99_MAX = 5000.0;
    public static final double DB_POOL_MAX = 100.0;
    public static final double BANK_STATUS_MAX = 2.0;
    public static final double MERCHANT_TIER_MAX = 1.0;

    // ── Named constants — reward / done thresholds ─────────────────────
    public static final double CRASH_THRESHOLD = 4000.0;
    public static final double SLA_BREACH_THRESHOLD = 800.0;
    public static final double SLA_PROXIMITY_LOWER = 500.0;
    public static final double LAG_PROXIMITY_LOWER = 3000.0;
    public static final double HIGH_RISK_THRESHOLD = 80.0;

    // ── Phase 5 constants ──────────────────────────────────────────────
    public static final double THROTTLE_RELIEF_PER_STEP = -150.0;
    public static final int THROTTLE_RELIEF_QUEUE_MAXLEN = 4;
    public static final double P99_EMA_ALPHA = 0.2;           // α for rolling_p99 EMA
    public static final double LATENCY_MEAN_REVERT_ALPHA = 0.2;
    public static final double LATENCY_BASELINE = 50.0;

    // ── Phase 6 curriculum constants ───────────────────────────────────
    // PYTHON EQUIVALENT: class-level tuple/int/float constants in UnifiedFintechEnv
    private static final double[] CURRICULUM_THRESHOLDS = {0.75, 0.45}; // easy→med, med→hard
    private static final int CURRICULUM_WINDOW = 5;    // consecutive episodes to advance
    private static final int ADVERSARY_WINDOW = 5;     // episodes before adversary reacts
    private static final double ADVERSARY_HIGH_THRESHOLD = 0.6; // avg > this → threat +0.5
    private static final double ADVERSARY_LOW_THRESHOLD = 0.3;  // avg < this → threat -0.5
    private static final double ADVERSARY_STEP = 0.5;

    // ── Episode configuration ──────────────────────────────────────────
    private final int maxSteps = 100;
    private int currentStep = 0;
    private String currentTask = "easy";

    // ── Phase machine state ────────────────────────────────────────────
    // PYTHON EQUIVALENT: list[str] — 100-element list of phase names
    private List<String> phaseSchedule = new ArrayList<>();

    // ── Direct accumulators ────────────────────────────────────────────
    private double kafkaLag = 0.0;
    private double apiLatency = LATENCY_BASELINE;
    private double rollingP99 = LATENCY_BASELINE;
    private double dbPool = 50.0;
    private double bankStatus = 0.0;
    private double systemEntropy = 0.0;
    private double merchantTier = 0.0;
    private double adversaryThreatLevel = 0.0;

    // ── Causal transition state ────────────────────────────────────────
    // Transition #2: Throttle relief queue (pops one item per step)
    // PYTHON EQUIVALENT: deque[float] with maxlen=4
    private final Deque<Double> throttleReliefQueue = new ArrayDeque<>();

    // Transition #1: Lag→Latency carry-over for next step
    private double lagLatencyCarry = 0.0;

    // ── Episode counters (cleared each reset) ─────────────────────────
    private int cumulativeSettlementBacklog = 0;
    private boolean isBurstStep = false;
    private String lastEventType = "normal";

    // ── Cross-episode curriculum state (NEVER reset between episodes) ──
    // PYTHON EQUIVALENT: self._curriculum_level (int) — 0=easy, 1=medium, 2=hard
    private int curriculumLevel = 0;

    // Per-step rewards for the current episode; drained in closeEpisode()
    // PYTHON EQUIVALENT: self._episode_step_rewards (list[float])
    private final List<Double> episodeStepRewards = new ArrayList<>();

    // 5-episode rolling window for curriculum advancement
    // PYTHON EQUIVALENT: self._rolling_5ep_avgs (deque[float], maxlen=5)
    // Note: Java ArrayDeque doesn't have maxlen; we enforce it manually.
    private final Deque<Double> rolling5epWindow = new ArrayDeque<>();
    private int consecutiveAboveThreshold = 0;

    // Separate 5-ep window for adversary escalation (Transition #7)
    // PYTHON EQUIVALENT: self._adversary_ep_window (deque[float], maxlen=5)
    private final Deque<Double> adversaryEpWindow = new ArrayDeque<>();

    // ── Current observation ────────────────────────────────────────────
    // PYTHON EQUIVALENT: AEPOObservation (Pydantic BaseModel)
    private AEPOObservation currentObs;

    // PYTHON EQUIVALENT: numpy RandomState (seeded via gymnasium)
    private Random rng = ThreadLocalRandom.current();

    // =====================================================================
    // Phase 6 — Adaptive Curriculum + Adversary Escalation
    // =====================================================================

    /**
     * Tally the just-finished episode and update curriculum / adversary state.
     *
     * Called at the START of reset() before any episode state is cleared.
     * Safe on first reset (empty episodeStepRewards → early return).
     *
     * PYTHON EQUIVALENT: env._close_episode()
     *
     * Curriculum Logic:
     *   easy→medium : 5 consecutive eps with mean > 0.75 → curriculumLevel = 1
     *   medium→hard  : 5 consecutive eps with mean > 0.45 → curriculumLevel = 2
     *   NEVER regresses.
     *
     * Adversary Logic (Transition #7, 5-episode lag):
     *   window mean > 0.6 → adversaryThreatLevel += 0.5 (max ADV_THREAT_MAX)
     *   window mean < 0.3 → adversaryThreatLevel -= 0.5 (min 0.0)
     */
    private void closeEpisode() {
        if (episodeStepRewards.isEmpty()) return;

        // Pad crashed episodes (missing steps count as 0.0)
        int padCount = Math.max(0, maxSteps - episodeStepRewards.size());
        double sum = episodeStepRewards.stream().mapToDouble(Double::doubleValue).sum();
        double epMean = sum / maxSteps;   // divide by full episode length (with 0.0 padding)

        // Curriculum advancement
        if (curriculumLevel < 2) {
            double threshold = CURRICULUM_THRESHOLDS[curriculumLevel];
            if (epMean >= threshold) {
                consecutiveAboveThreshold++;
            } else {
                consecutiveAboveThreshold = 0;
            }
            if (consecutiveAboveThreshold >= CURRICULUM_WINDOW) {
                curriculumLevel++;
                consecutiveAboveThreshold = 0;
                log.info(String.format("[CURRICULUM] Advanced to level %d", curriculumLevel));
            }
        }

        // Adversary escalation — 5-episode lag
        // PYTHON EQUIVALENT: deque with maxlen=5; enforce manually here
        adversaryEpWindow.addLast(epMean);
        if (adversaryEpWindow.size() > ADVERSARY_WINDOW) adversaryEpWindow.pollFirst();

        if (adversaryEpWindow.size() >= ADVERSARY_WINDOW) {
            double windowMean = adversaryEpWindow.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            if (windowMean > ADVERSARY_HIGH_THRESHOLD) {
                adversaryThreatLevel = Math.min(ADV_THREAT_MAX, adversaryThreatLevel + ADVERSARY_STEP);
            } else if (windowMean < ADVERSARY_LOW_THRESHOLD) {
                adversaryThreatLevel = Math.max(0.0, adversaryThreatLevel - ADVERSARY_STEP);
            }
        }
    }

    // =====================================================================
    // Phase Machine — build schedule fixed at reset
    // =====================================================================

    /**
     * Build the 100-step phase schedule for a given task.
     * 
     * Phase sequences are FIXED AT INIT, NEVER MIXED BY CURRICULUM:
     *   easy:   Normal × 100
     *   medium: Normal × 40  → Spike × 60
     *   hard:   Normal × 20  → Spike × 20 → Attack × 40 → Recovery × 20
     *
     * PYTHON EQUIVALENT: UnifiedFintechEnv._build_phase_schedule(task_name)
     */
    private static List<String> buildPhaseSchedule(String taskName) {
        List<String> schedule = new ArrayList<>(100);
        switch (taskName) {
            case "easy":
                for (int i = 0; i < 100; i++) schedule.add("normal");
                break;
            case "medium":
                for (int i = 0; i < 40; i++) schedule.add("normal");
                for (int i = 0; i < 60; i++) schedule.add("spike");
                break;
            case "hard":
                for (int i = 0; i < 20; i++) schedule.add("normal");
                for (int i = 0; i < 20; i++) schedule.add("spike");
                for (int i = 0; i < 40; i++) schedule.add("attack");
                for (int i = 0; i < 20; i++) schedule.add("recovery");
                break;
            default:
                throw new IllegalArgumentException("Unknown task: " + taskName);
        }
        return schedule;
    }

    // =====================================================================
    // reset()
    // =====================================================================

    /**
     * Reset the environment for a new episode under the given task.
     *
     * PYTHON EQUIVALENT: env.reset(seed, options={"task": task_name})
     *
     * @param seed     Optional PRNG seed (nullable)
     * @param taskName One of "easy", "medium", "hard"
     * @return Tuple of (initial observation, info dict)
     */
    public Map.Entry<AEPOObservation, Map<String, Object>> reset(Long seed, String taskName) {
        if (seed != null) {
            rng = new Random(seed);
        }

        if (!Set.of("easy", "medium", "hard").contains(taskName)) {
            throw new IllegalArgumentException(
                "Unknown task '" + taskName + "'; expected 'easy', 'medium', or 'hard'."
            );
        }

        // Phase 6: tally the just-finished episode BEFORE clearing state
        closeEpisode();
        episodeStepRewards.clear();

        this.currentTask = taskName;
        this.currentStep = 0;

        // Build phase schedule
        this.phaseSchedule = buildPhaseSchedule(taskName);

        // Reset direct accumulators to safe baselines
        this.kafkaLag = 0.0;
        this.apiLatency = LATENCY_BASELINE;
        this.rollingP99 = LATENCY_BASELINE;
        this.dbPool = 50.0;
        this.bankStatus = 0.0;
        this.systemEntropy = 0.0;
        this.isBurstStep = false;

        // Merchant tier: Enterprise on hard task; Small elsewhere
        this.merchantTier = "hard".equals(taskName) ? 1.0 : 0.0;

        // BOUNDARY RULE: throttleReliefQueue.clear() MUST be called in reset()
        this.throttleReliefQueue.clear();
        this.lagLatencyCarry = 0.0;
        this.cumulativeSettlementBacklog = 0;

        // Generate initial observation
        this.currentObs = generatePhaseObservation();

        Map<String, Object> info = Map.of("task", taskName);
        return Map.entry(currentObs, info);
    }

    /**
     * Return the current observation without advancing the clock.
     *
     * PYTHON EQUIVALENT: env.state()
     */
    public AEPOObservation state() {
        return currentObs;
    }

    // =====================================================================
    // Phase-driven observation generation
    // =====================================================================

    /**
     * Generate a phase-driven observation using internal accumulators.
     *
     * Replaces the old memoryless _generate_transaction() with a causally-
     * structured generator where:
     * - Phase determines risk_score range, kafka_lag delta, bank_api_status
     * - Throttle relief queue pops one item per step (Transition #2)
     * - Lag→Latency carry-over applied (Transition #1)
     * - api_latency mean-reverts toward baseline
     *
     * PYTHON EQUIVALENT: env._generate_phase_observation()
     */
    private AEPOObservation generatePhaseObservation() {
        int stepIdx = Math.min(currentStep, phaseSchedule.size() - 1);
        String phase = phaseSchedule.isEmpty() ? "normal" : phaseSchedule.get(stepIdx);

        // Channel (common to all phases)
        double channel = rng.nextInt(3);

        // Phase-driven risk_score and kafka_lag delta
        double riskScore;
        double lagDelta;
        isBurstStep = false;

        switch (phase) {
            case "normal":
                riskScore = uniform(5.0, 30.0);
                lagDelta = uniform(50.0, 150.0);
                bankStatus = 0.0;
                break;

            case "spike":
                if (rng.nextDouble() < 0.80) {
                    riskScore = uniform(5.0, 30.0);
                    lagDelta = uniform(50.0, 150.0);
                } else {
                    riskScore = uniform(0.0, 10.0);
                    lagDelta = uniform(500.0, 1000.0);
                    isBurstStep = true;
                }
                bankStatus = rng.nextDouble() < 0.3 ? 1.0 : 0.0;
                break;

            case "attack":
                riskScore = uniform(85.0, 100.0);
                lagDelta = uniform(100.0, 400.0);
                bankStatus = 1.0;
                break;

            case "recovery":
                riskScore = uniform(40.0, 70.0);
                lagDelta = uniform(-200.0, -100.0);
                // Gradual Degraded→Healthy transition
                int firstRecovery = phaseSchedule.indexOf("recovery");
                long totalRecovery = phaseSchedule.stream().filter("recovery"::equals).count();
                int stepsIntoRecovery = Math.max(0, currentStep - firstRecovery);
                double healProb = Math.min(1.0, (double) stepsIntoRecovery / Math.max(1, totalRecovery));
                bankStatus = rng.nextDouble() < healProb ? 0.0 : 1.0;
                break;

            default:
                riskScore = uniform(5.0, 30.0);
                lagDelta = uniform(50.0, 150.0);
                bankStatus = 0.0;
        }

        // Apply kafka_lag delta
        kafkaLag += lagDelta;

        // Transition #2: Throttle relief queue (pop one per step)
        if (!throttleReliefQueue.isEmpty()) {
            kafkaLag += throttleReliefQueue.pollFirst();
        }
        kafkaLag = Math.max(0.0, kafkaLag);

        // Transition #1: Apply lag→latency carry-over from previous step
        apiLatency += lagLatencyCarry;
        lagLatencyCarry = 0.0;

        // api_latency: natural mean-reversion + small random variation
        apiLatency = LATENCY_MEAN_REVERT_ALPHA * LATENCY_BASELINE
                     + (1.0 - LATENCY_MEAN_REVERT_ALPHA) * apiLatency
                     + uniform(-10.0, 10.0);
        apiLatency = Math.max(10.0, apiLatency);

        // System entropy (random each step)
        systemEntropy = uniform(0.0, ENTROPY_MAX);

        // DB pool (varies by phase)
        if ("normal".equals(phase)) {
            dbPool = uniform(30.0, 70.0);
        } else if ("spike".equals(phase) && isBurstStep) {
            dbPool = uniform(60.0, 95.0);
        } else {
            dbPool = uniform(50.0, 90.0);
        }
        dbPool = Math.min(1.0, Math.max(0.0, dbPool / DB_POOL_MAX)) * DB_POOL_MAX;

        // Event type for backward compat
        switch (phase) {
            case "spike":
                lastEventType = isBurstStep ? "flash_sale" : "normal";
                break;
            case "attack":
                lastEventType = "botnet_attack";
                break;
            case "recovery":
                lastEventType = "recovery";
                break;
            default:
                lastEventType = "normal";
        }

        // POMDP: Apply bounded Gaussian noise to infra metrics
        double noisyKafkaLag = clamp(
            rng.nextGaussian() * (0.05 * Math.max(1.0, kafkaLag)) + kafkaLag,
            0.0, LAG_MAX
        );
        double noisyApiLatency = clamp(
            rng.nextGaussian() * (0.02 * Math.max(1.0, apiLatency)) + apiLatency,
            0.0, LATENCY_MAX
        );

        return new AEPOObservation(
            clamp(channel, 0.0, CHANNEL_MAX),
            clamp(riskScore, 0.0, RISK_MAX),
            clamp(adversaryThreatLevel, 0.0, ADV_THREAT_MAX),
            clamp(systemEntropy, 0.0, ENTROPY_MAX),
            noisyKafkaLag,
            noisyApiLatency,
            clamp(rollingP99, 0.0, P99_MAX),
            clamp(dbPool, 0.0, DB_POOL_MAX),
            clamp(bankStatus, 0.0, BANK_STATUS_MAX),
            clamp(merchantTier, 0.0, MERCHANT_TIER_MAX)
        );
    }

    // =====================================================================
    // step() — with all 8 causal transitions
    // =====================================================================

    /**
     * Run one time-step of the environment's dynamics.
     *
     * Phase 5: Applies all 8 causal state transitions before reward calculation.
     *
     * PYTHON EQUIVALENT: env.step(action) → (obs, UFRGReward, done, info)
     *
     * @param action Validated AEPOAction
     * @return StepResult containing observation, reward, done flag, and info dict
     */
    public StepResult step(AEPOAction action) {
        // ① Determine phase and snapshot current observation
        int stepIdx = Math.min(currentStep, phaseSchedule.size() - 1);
        String currentPhase = phaseSchedule.isEmpty() ? "normal" : phaseSchedule.get(stepIdx);

        double riskScore = currentObs.riskScore();
        double obsKafkaLag = currentObs.kafkaLag();
        double obsDbPool = currentObs.dbConnectionPool();
        double obsBankStatus = currentObs.bankApiStatus();
        double obsMerchantTier = currentObs.merchantTier();
        double obsEntropy = currentObs.systemEntropy();

        boolean circuitBreakerTripped = false;
        boolean done = false;
        String terminationReason = null;
        boolean blindSpotTriggered = false;

        // ② Causal transitions that affect THIS step's reward context

        double effectiveApiLatency = apiLatency;

        // Transition #4: DB pressure
        if (obsDbPool > 80 && action.dbRetryPolicy() == 1) {
            effectiveApiLatency += 100.0;
        }

        // Transition #6: Entropy spike
        if (obsEntropy > 70) {
            effectiveApiLatency += uniform(100.0, 300.0);
        }

        // Transition #8: P99 EMA
        // rolling_p99[t] = 0.8 × rolling_p99[t-1] + 0.2 × api_latency[t]
        double effectiveP99 = (1.0 - P99_EMA_ALPHA) * rollingP99 + P99_EMA_ALPHA * effectiveApiLatency;

        // Transition #3: Bank coupling
        if (obsBankStatus == 1.0 && action.settlementPolicy() == 0) {
            effectiveP99 += 200.0;
        }

        // Update internal accumulators
        apiLatency = effectiveApiLatency;
        rollingP99 = effectiveP99;

        // ③ Reward calculation — AEPO Reward v2 (CLAUDE.md spec)
        double base = 0.8;
        double fraudPenalty = 0.0;
        double slaPenalty = 0.0;
        double infraPenalty = 0.0;
        double dbPenalty = 0.0;
        double settlementPenalty = 0.0;
        double bonus = 0.0;

        // Catastrophic fraud gate
        boolean isFraudCatastrophe = action.riskDecision() == 0
                                     && action.cryptoVerify() == 1
                                     && riskScore > HIGH_RISK_THRESHOLD;
        if (isFraudCatastrophe) {
            fraudPenalty = -base;
            done = true;
            terminationReason = "fraud";
        }

        // System crash
        boolean crashed = obsKafkaLag > CRASH_THRESHOLD;
        if (crashed && !done) {
            done = true;
            terminationReason = "crash";
        }

        // SLA penalty (using causal-transition-modified P99)
        if (effectiveP99 > SLA_BREACH_THRESHOLD) {
            slaPenalty = -0.30;
        } else if (effectiveP99 > SLA_PROXIMITY_LOWER && effectiveP99 <= SLA_BREACH_THRESHOLD) {
            double prox = (effectiveP99 - SLA_PROXIMITY_LOWER) / (SLA_BREACH_THRESHOLD - SLA_PROXIMITY_LOWER);
            slaPenalty = Math.round(-0.10 * prox * 10000.0) / 10000.0;
        }

        // Lag proximity
        if (obsKafkaLag > LAG_PROXIMITY_LOWER && obsKafkaLag <= CRASH_THRESHOLD) {
            double prox = (obsKafkaLag - LAG_PROXIMITY_LOWER) / (CRASH_THRESHOLD - LAG_PROXIMITY_LOWER);
            infraPenalty += Math.round(-0.10 * prox * 10000.0) / 10000.0;
        }

        // Infra routing penalties
        if (action.infraRouting() == 1) {           // Throttle
            infraPenalty += "spike".equals(currentPhase) ? -0.10 : -0.20;
        } else if (action.infraRouting() == 2) {    // CircuitBreaker
            infraPenalty += -0.50;
        }

        // DB retry policy (Transition #5: DB waste)
        if (action.dbRetryPolicy() == 1) {
            if (obsDbPool > 80) {
                dbPenalty = 0.03;
            } else if (obsDbPool < 20) {
                dbPenalty = -0.10;
            }
        }

        // Settlement policy
        if (action.settlementPolicy() == 1) {       // DeferredAsyncFallback
            cumulativeSettlementBacklog++;
            if (obsBankStatus == 1.0) {
                settlementPenalty += 0.04;
            } else if ("normal".equals(currentPhase)) {
                settlementPenalty += -0.15;
            }
            if (cumulativeSettlementBacklog > 10) {
                settlementPenalty += -0.20;
            }
        } else {
            cumulativeSettlementBacklog = Math.max(0, cumulativeSettlementBacklog - 2);
        }

        // Risk/crypto bonuses
        if (riskScore > HIGH_RISK_THRESHOLD) {
            if (action.riskDecision() == 2) bonus += 0.05;      // Challenge
            if (action.cryptoVerify() == 0) bonus += 0.03;      // FullVerify
            if (action.riskDecision() == 1 && action.cryptoVerify() == 1) {
                bonus += 0.04;  // Blind spot: Reject+SkipVerify
                blindSpotTriggered = true;
            }
        }

        // App priority / merchant-tier alignment
        if (action.appPriority() == 0 && obsMerchantTier == 0.0) bonus += 0.02;
        else if (action.appPriority() == 1 && obsMerchantTier == 1.0) bonus += 0.02;

        // Final reward
        double rawReward = base + fraudPenalty + slaPenalty + infraPenalty + dbPenalty + settlementPenalty + bonus;
        double finalReward;
        if (crashed || isFraudCatastrophe) {
            finalReward = 0.0;
        } else {
            finalReward = Math.min(1.0, Math.max(0.0, rawReward));
        }

        // ④ Action effects on kafka_lag accumulator
        if (action.cryptoVerify() == 0) {           // FullVerify
            kafkaLag += 150.0;
            apiLatency += 200.0;
        } else {                                    // SkipVerify
            kafkaLag -= 100.0;
        }

        if (action.infraRouting() == 0) {           // Normal
            kafkaLag += 100.0;
        } else if (action.infraRouting() == 1) {    // Throttle
            // Transition #2: schedule -150 for next 2 steps
            if (throttleReliefQueue.size() < THROTTLE_RELIEF_QUEUE_MAXLEN) {
                throttleReliefQueue.addLast(THROTTLE_RELIEF_PER_STEP);
            }
            if (throttleReliefQueue.size() < THROTTLE_RELIEF_QUEUE_MAXLEN) {
                throttleReliefQueue.addLast(THROTTLE_RELIEF_PER_STEP);
            }
        } else {                                    // CircuitBreaker
            kafkaLag = 0.0;
            apiLatency = LATENCY_BASELINE;
            circuitBreakerTripped = true;
        }

        kafkaLag = Math.max(0.0, kafkaLag);
        apiLatency = Math.max(0.0, apiLatency);

        // Transition #1: lag→latency carry for NEXT step
        lagLatencyCarry = 0.1 * Math.max(0.0, obsKafkaLag - 3000.0);

        // ⑤ Advance counter and generate next observation
        currentStep++;
        currentObs = generatePhaseObservation();

        if (currentStep >= maxSteps && !done) {
            done = true;
        }

        // ⑥ Build reward breakdown
        Map<String, Double> rewardBreakdown = new LinkedHashMap<>();
        rewardBreakdown.put("base", base);
        rewardBreakdown.put("fraud_penalty", fraudPenalty);
        rewardBreakdown.put("sla_penalty", slaPenalty);
        rewardBreakdown.put("infra_penalty", infraPenalty);
        rewardBreakdown.put("db_penalty", dbPenalty);
        rewardBreakdown.put("settlement_penalty", settlementPenalty);
        rewardBreakdown.put("bonus", bonus);
        rewardBreakdown.put("final", finalReward);

        // ⑦ Build info dict
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("phase", currentPhase);
        info.put("curriculum_level", curriculumLevel);
        info.put("step_in_episode", currentStep);
        info.put("reward_breakdown", rewardBreakdown);
        info.put("termination_reason", terminationReason);
        info.put("adversary_threat_level_raw", adversaryThreatLevel);
        info.put("blind_spot_triggered", blindSpotTriggered);
        info.put("cumulative_settlement_backlog", cumulativeSettlementBacklog);
        info.put("event_type", lastEventType);
        info.put("crashed", crashed);
        info.put("done", done);

        // Phase 6: collect per-step reward for end-of-episode averaging
        episodeStepRewards.add(finalReward);

        return new StepResult(currentObs, finalReward, rewardBreakdown, crashed,
                              circuitBreakerTripped, done, info);
    }

    // =====================================================================
    // Internal helpers
    // =====================================================================

    /** PYTHON EQUIVALENT: np.clip(val, lo, hi) → Math.min(hi, Math.max(lo, val)) */
    private static double clamp(double val, double lo, double hi) {
        return Math.min(hi, Math.max(lo, val));
    }

    /** PYTHON EQUIVALENT: rng.uniform(lo, hi) → ThreadLocalRandom.current().nextDouble(lo, hi) */
    private double uniform(double lo, double hi) {
        return lo + rng.nextDouble() * (hi - lo);
    }

    // =====================================================================
    // Inner records / DTOs
    // =====================================================================

    /**
     * Step result DTO — mirrors the Python 4-tuple (obs, reward, done, info).
     *
     * PYTHON EQUIVALENT: tuple[AEPOObservation, UFRGReward, bool, dict]
     */
    public record StepResult(
        AEPOObservation observation,
        double reward,
        Map<String, Double> rewardBreakdown,
        boolean crashed,
        boolean circuitBreakerTripped,
        boolean done,
        Map<String, Object> info
    ) {}
}
