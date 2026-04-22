package aepo;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Graders.java — Phase 7 Java Mirror of graders.py
 * ==================================================
 * Per-task programmatic graders for AEPO.
 *
 * PYTHON EQUIVALENT:
 *   graders.py — EasyGrader, MediumGrader, HardGrader, get_grader()
 *
 * Interface 1 — Spec-compliant evaluator (PRIMARY):
 *   grader.gradeAgent(policyFn, nEpisodes) → double in [0.0, 1.0]
 *   Runs nEpisodes full episodes; returns mean padded reward.
 *
 * Interface 2 — Trajectory scorer (legacy, used by inference):
 *   grader.grade(trajectory) → double in [0.0, 1.0]
 *   Scores a pre-collected info-dict trajectory.
 *   Phase 7: sentinel [0.01, 0.99] clamping REMOVED — now [0.0, 1.0].
 *
 * Task thresholds (CLAUDE.md):
 *   easy   ≥ 0.75   seed=42   Normal × 100
 *   medium ≥ 0.45   seed=43   Normal+Spike
 *   hard   ≥ 0.30   seed=44   All phases, adversary 7–10
 *
 * NOTE: Java does not run the gym environment. The gradeAgent() method
 * below is a stub — implement it by wrapping the Python server via HTTP
 * (same as inference.py) or calling the Python process directly.
 */
public class Graders {

    // ---------------------------------------------------------------------------
    // Type alias
    // PYTHON EQUIVALENT: PolicyFn = Callable[[dict[str, float]], AEPOAction]
    // ---------------------------------------------------------------------------
    @FunctionalInterface
    public interface PolicyFn extends Function<Map<String, Double>, AEPOAction> {}

    // ---------------------------------------------------------------------------
    // Internal: run N episodes and return mean padded reward
    // PYTHON EQUIVALENT: _run_episodes(task, policy_fn, seed, n_episodes) -> float
    // ---------------------------------------------------------------------------

    /**
     * Run {@code nEpisodes} full episodes using {@code policyFn} on the given task
     * and return the mean per-step reward over all episodes.
     *
     * Crashed episodes are padded to 100 steps with 0.0 reward.
     *
     * // PYTHON EQUIVALENT:
     * // def _run_episodes(task, policy_fn, seed, n_episodes=10) -> float:
     * //     env = UnifiedFintechEnv()
     * //     for ep in range(n_episodes):
     * //         ep_seed = seed + ep
     * //         obs, _ = env.reset(seed=ep_seed, options={"task": task})
     * //         step_rewards = []
     * //         while not done and len(step_rewards) < env.max_steps:
     * //             action = policy_fn(obs.normalized())
     * //             obs, typed_reward, done, _info = env.step(action)
     * //             step_rewards.append(typed_reward.value)
     * //         padded = step_rewards + [0.0] * max(0, 100 - len(step_rewards))
     * //         ep_means.append(sum(padded) / len(padded))
     * //     return round(sum(ep_means) / len(ep_means), 4)
     */
    public static double runEpisodes(
            String task,
            PolicyFn policyFn,
            int seed,
            int nEpisodes
    ) {
        // STUB: In production, instantiate UnifiedFintechEnv and run episodes.
        // Java mirror does not execute the Python gym environment directly.
        throw new UnsupportedOperationException(
            "runEpisodes() requires a live UnifiedFintechEnv — use the Python server or gRPC bridge."
        );
    }

    // ---------------------------------------------------------------------------
    // EasyGrader — seed=42, task=easy, threshold ≥ 0.75
    // PYTHON EQUIVALENT: class EasyGrader
    // ---------------------------------------------------------------------------
    public static class EasyGrader {

        public static final String TASK      = "easy";
        public static final int    SEED      = 42;
        public static final double THRESHOLD = 0.75;
        public static final double SLA_THRESHOLD_MS = 800.0;

        /**
         * Run nEpisodes of the easy task under policyFn and return mean reward.
         *
         * // PYTHON EQUIVALENT:
         * // def grade_agent(self, policy_fn, *, n_episodes=10) -> float:
         * //     return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)
         */
        public double gradeAgent(PolicyFn policyFn, int nEpisodes) {
            return runEpisodes(TASK, policyFn, SEED, nEpisodes);
        }

        public double gradeAgent(PolicyFn policyFn) {
            return gradeAgent(policyFn, 10);
        }

        /**
         * Score a pre-collected episode trajectory. Returns double in [0.0, 1.0].
         * Phase 7: sentinel [0.01, 0.99] REMOVED.
         *
         * // PYTHON EQUIVALENT:
         * // def grade(self, trajectory: list[dict]) -> float:
         * //     if not trajectory: return 0.0
         * //     for step in trajectory:
         * //         reward = step.get("reward_final", 0.0)
         * //         infra  = step.get("action_infra_routing", 0)
         * //         if reward >= 0.8:
         * //             total_credit += 1.0 if infra == 0 else 0.5
         * //     return round(max(0.0, min(1.0, total_credit / len(trajectory))), 2)
         */
        public double grade(List<Map<String, Object>> trajectory) {
            if (trajectory == null || trajectory.isEmpty()) return 0.0;

            double totalCredit = 0.0;
            for (Map<String, Object> step : trajectory) {
                double reward = toDouble(step.getOrDefault("reward_final", 0.0));
                int infra     = toInt(step.getOrDefault("action_infra_routing", 0));
                if (reward >= 0.8) {
                    totalCredit += (infra == 0) ? 1.0 : 0.5;
                }
            }
            double rawScore = totalCredit / trajectory.size();
            return Math.round(Math.max(0.0, Math.min(1.0, rawScore)) * 100.0) / 100.0;
        }
    }

    // ---------------------------------------------------------------------------
    // MediumGrader — seed=43, task=medium, threshold ≥ 0.45
    // PYTHON EQUIVALENT: class MediumGrader
    // ---------------------------------------------------------------------------
    public static class MediumGrader {

        public static final String TASK          = "medium";
        public static final int    SEED          = 43;
        public static final double THRESHOLD     = 0.45;
        public static final double SLA_THRESHOLD_MS = 800.0;
        public static final double THROTTLE_BONUS   = 0.1;

        /**
         * // PYTHON EQUIVALENT:
         * // def grade_agent(self, policy_fn, *, n_episodes=10) -> float:
         * //     return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)
         */
        public double gradeAgent(PolicyFn policyFn, int nEpisodes) {
            return runEpisodes(TASK, policyFn, SEED, nEpisodes);
        }

        public double gradeAgent(PolicyFn policyFn) {
            return gradeAgent(policyFn, 10);
        }

        /**
         * // PYTHON EQUIVALENT:
         * // def grade(self, trajectory: list[dict]) -> float:
         * //     if not trajectory: return 0.0
         * //     for step in trajectory:
         * //         if not crashed and p99 <= SLA_THRESHOLD_MS: clean_steps++
         * //         if event_type == "flash_sale" and infra == 1: throttle_bonus += 0.1
         * //     base_score = clean_steps / len(trajectory)
         * //     normalised_bonus = min(throttle_bonus / len(trajectory), 0.1)
         * //     return round(max(0.0, min(1.0, base_score + normalised_bonus)), 2)
         */
        public double grade(List<Map<String, Object>> trajectory) {
            if (trajectory == null || trajectory.isEmpty()) return 0.0;

            int    cleanSteps    = 0;
            double throttleBonus = 0.0;

            for (Map<String, Object> step : trajectory) {
                boolean crashed   = toBool(step.getOrDefault("crashed", false));
                double  p99       = toDouble(step.getOrDefault("obs_rolling_p99", 0.0));
                String  eventType = String.valueOf(step.getOrDefault("event_type", "normal"));
                int     infra     = toInt(step.getOrDefault("action_infra_routing", 0));

                if (!crashed && p99 <= SLA_THRESHOLD_MS) {
                    cleanSteps++;
                }
                if ("flash_sale".equals(eventType) && infra == 1) {
                    throttleBonus += THROTTLE_BONUS;
                }
            }

            double baseScore       = (double) cleanSteps / trajectory.size();
            double normalisedBonus = Math.min(throttleBonus / trajectory.size(), 0.1);
            double rawScore        = baseScore + normalisedBonus;
            return Math.round(Math.max(0.0, Math.min(1.0, rawScore)) * 100.0) / 100.0;
        }
    }

    // ---------------------------------------------------------------------------
    // HardGrader — seed=44, task=hard, threshold ≥ 0.30
    // PYTHON EQUIVALENT: class HardGrader
    // ---------------------------------------------------------------------------
    public static class HardGrader {

        public static final String TASK          = "hard";
        public static final int    SEED          = 44;
        public static final double THRESHOLD     = 0.30;
        public static final double RISK_THRESHOLD = 80.0;
        public static final double CRASH_PENALTY  = 0.15;

        /**
         * // PYTHON EQUIVALENT:
         * // def grade_agent(self, policy_fn, *, n_episodes=10) -> float:
         * //     return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)
         */
        public double gradeAgent(PolicyFn policyFn, int nEpisodes) {
            return runEpisodes(TASK, policyFn, SEED, nEpisodes);
        }

        public double gradeAgent(PolicyFn policyFn) {
            return gradeAgent(policyFn, 10);
        }

        /**
         * // PYTHON EQUIVALENT:
         * // def grade(self, trajectory: list[dict]) -> float:
         * //     if not trajectory: return 0.0
         * //     for step:
         * //         if crashed: crash_count++
         * //         if p99 <= 800: sla_ok_steps++
         * //         if risk_score > 80:
         * //             if decision == 1: catch_score = 1.0
         * //             elif decision == 2: catch_score = 0.8
         * //             else: catch_score = 0.0
         * //             if catch_score > 0 and crypto == 0: catch_score = min(1.0, catch_score + 0.1)
         * //             total_catch_score += catch_score
         * //     fcr = total_catch_score / high_risk_steps if high_risk_steps > 0 else 0.0
         * //     raw = fcr - crash_count * 0.15 + 0.1 * (sla_ok_steps / len(trajectory))
         * //     return round(max(0.0, min(1.0, raw)), 2)
         */
        public double grade(List<Map<String, Object>> trajectory) {
            if (trajectory == null || trajectory.isEmpty()) return 0.0;

            int    highRiskSteps   = 0;
            double totalCatchScore = 0.0;
            int    crashCount      = 0;
            int    slaOkSteps      = 0;

            for (Map<String, Object> step : trajectory) {
                double  riskScore = toDouble(step.getOrDefault("obs_risk_score", 0.0));
                int     decision  = toInt(step.getOrDefault("action_risk_decision", 0));
                int     crypto    = toInt(step.getOrDefault("action_crypto_verify", 0));
                boolean crashed   = toBool(step.getOrDefault("crashed", false));
                double  p99       = toDouble(step.getOrDefault("obs_rolling_p99", 0.0));

                if (crashed)    crashCount++;
                if (p99 <= 800.0) slaOkSteps++;

                if (riskScore > RISK_THRESHOLD) {
                    highRiskSteps++;
                    double catchScore;
                    if      (decision == 1) catchScore = 1.0;
                    else if (decision == 2) catchScore = 0.8;
                    else                   catchScore = 0.0;

                    if (catchScore > 0 && crypto == 0) {
                        catchScore = Math.min(1.0, catchScore + 0.1);
                    }
                    totalCatchScore += catchScore;
                }
            }

            double fcr          = highRiskSteps > 0 ? totalCatchScore / highRiskSteps : 0.0;
            double crashPenalty = crashCount * CRASH_PENALTY;
            double slaBonus     = 0.1 * ((double) slaOkSteps / trajectory.size());

            double rawScore = fcr - crashPenalty + slaBonus;
            return Math.round(Math.max(0.0, Math.min(1.0, rawScore)) * 100.0) / 100.0;
        }
    }

    // ---------------------------------------------------------------------------
    // Factory — get_grader(task_name)
    // PYTHON EQUIVALENT: get_grader(task_name: str) -> EasyGrader | MediumGrader | HardGrader
    // ---------------------------------------------------------------------------

    /**
     * Returns a new grader instance for the given task name.
     *
     * // PYTHON EQUIVALENT:
     * // def get_grader(task_name: str) -> EasyGrader | MediumGrader | HardGrader:
     * //     return _GRADER_MAP[task_name]()
     */
    public static Object getGrader(String taskName) {
        return switch (taskName) {
            case "easy"   -> new EasyGrader();
            case "medium" -> new MediumGrader();
            case "hard"   -> new HardGrader();
            default -> throw new IllegalArgumentException(
                "Unknown task '" + taskName + "'. Expected one of: [easy, medium, hard]"
            );
        };
    }

    // ---------------------------------------------------------------------------
    // Built-in policies
    // PYTHON EQUIVALENT: random_policy(), heuristic_policy() in graders.py
    // ---------------------------------------------------------------------------

    /**
     * Uniformly random policy.
     *
     * // PYTHON EQUIVALENT:
     * // def random_policy(obs_normalized: dict[str, float]) -> AEPOAction:
     * //     import random
     * //     return AEPOAction(risk_decision=random.randint(0,2), ...)
     */
    public static PolicyFn randomPolicy() {
        return obs -> {
            java.util.concurrent.ThreadLocalRandom rng = java.util.concurrent.ThreadLocalRandom.current();
            return new AEPOAction(
                rng.nextInt(0, 3),  // risk_decision
                rng.nextInt(0, 2),  // crypto_verify
                rng.nextInt(0, 3),  // infra_routing
                rng.nextInt(0, 2),  // db_retry_policy
                rng.nextInt(0, 2),  // settlement_policy
                rng.nextInt(0, 3)   // app_priority
            );
        };
    }

    /**
     * Intentionally-incomplete heuristic agent with 3 deliberate blind spots.
     *
     * BLIND SPOTS (agent must discover these):
     *   #1: Reject+SkipVerify on high-risk → +0.04 bonus (heuristic uses FullVerify)
     *   #2: app_priority should match merchant_tier → +0.02/step (heuristic always Balanced)
     *   #3: ExponentialBackoff when pool<0.2 → -0.10 penalty (heuristic always Backoff)
     *
     * // PYTHON EQUIVALENT:
     * // def heuristic_policy(obs_normalized: dict[str, float]) -> AEPOAction:
     * //     if risk_score > 0.8: risk_decision=1, crypto_verify=0  # BLIND SPOT #1
     * //     if kafka_lag > 0.6:  infra_routing=1
     * //     if rolling_p99>0.6:  settlement_policy=1
     * //     db_retry_policy = 1  # always Backoff — BLIND SPOT #3
     * //     app_priority = 2     # always Balanced — BLIND SPOT #2
     */
    public static PolicyFn heuristicPolicy() {
        return obs -> {
            double riskScore  = obs.getOrDefault("risk_score", 0.0);
            double kafkaLag   = obs.getOrDefault("kafka_lag", 0.0);
            double rollingP99 = obs.getOrDefault("rolling_p99", 0.0);

            int riskDecision, cryptoVerify;
            if (riskScore > 0.8) {
                riskDecision = 1;  // Reject
                cryptoVerify = 0;  // FullVerify — BLIND SPOT #1
            } else {
                riskDecision = 0;  // Approve
                cryptoVerify = 1;  // SkipVerify
            }

            // Throttle before crash cliff (crash at normalized 0.4)
            int infraRouting = (kafkaLag > 0.3) ? 1 : 0;  // Throttle or Normal

            int settlementPolicy = (rollingP99 > 0.6) ? 1 : 0;  // DeferredAsync or Sync

            int dbRetryPolicy = 1;  // ExponentialBackoff always — BLIND SPOT #3
            int appPriority   = 2;  // Balanced always — BLIND SPOT #2

            return new AEPOAction(
                riskDecision,
                cryptoVerify,
                infraRouting,
                dbRetryPolicy,
                settlementPolicy,
                appPriority
            );
        };
    }

    // ---------------------------------------------------------------------------
    // Utility helpers
    // ---------------------------------------------------------------------------

    private static double toDouble(Object val) {
        if (val instanceof Number n) return n.doubleValue();
        try { return Double.parseDouble(String.valueOf(val)); } catch (Exception e) { return 0.0; }
    }

    private static int toInt(Object val) {
        if (val instanceof Number n) return n.intValue();
        try { return Integer.parseInt(String.valueOf(val)); } catch (Exception e) { return 0; }
    }

    private static boolean toBool(Object val) {
        if (val instanceof Boolean b) return b;
        return Boolean.parseBoolean(String.valueOf(val));
    }
}
