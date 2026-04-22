package aepo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * TrainQTable.java — Java mirror of train.py (Phase 10)
 *
 * Tabular Q-learning agent trained on the AEPO hard task.
 * This file is a readable Java translation intended for Java/Spring Boot engineers.
 * It does NOT compile or run — it documents the Python logic in Java idioms.
 *
 * Algorithm: Tabular Q-Learning with ε-greedy exploration.
 * State   : int[10] — one bin index (0–7) per normalised observation dimension.
 * Actions : 216 combinations encoded as a single int via mixed-radix encoding.
 *
 * // PYTHON EQUIVALENT:
 * //   from collections import defaultdict
 * //   q_table = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
 * //   Python's defaultdict creates zero-arrays on first access.
 * //   Java uses HashMap<String, float[]> with explicit initialisation on miss.
 */
public class TrainQTable {

    private static final Logger logger = Logger.getLogger(TrainQTable.class.getName());

    // ── Named constants — training hyper-parameters ──────────────────────────

    /** Total training episodes (hard task only). */
    static final int N_EPISODES = 500;

    /** Bins per observation dimension for state discretisation. */
    static final int N_BINS = 8;

    /** Total actions = 3×2×3×2×2×3. */
    static final int N_ACTIONS = 216;

    /** Q-table learning rate (lr in Bellman update). */
    static final double LEARNING_RATE = 0.1;

    /** Future-reward discount factor γ. */
    static final double DISCOUNT = 0.95;

    /** Initial exploration rate ε. */
    static final double EPSILON_START = 1.0;

    /** Minimum exploration rate ε after decay. */
    static final double EPSILON_END = 0.05;

    /** Log a summary line every N episodes. */
    static final int LOG_EVERY = 10;

    /** Task the Q-table is trained on. */
    static final String TRAIN_TASK = "hard";

    // ── Mixed-radix strides for action encoding ──────────────────────────────
    // MultiDiscrete: [risk(3), crypto(2), infra(3), db_retry(2), settle(2), priority(3)]
    // Python: _STRIDES = (72, 36, 12, 6, 3, 1)
    static final int[] STRIDES = {72, 36, 12, 6, 3, 1};
    static final int[] MAXES   = { 3,  2,  3,  2,  2,  3};

    // ── Canonical obs key order ───────────────────────────────────────────────
    // Must match AEPOObservation.normalized() field declaration order.
    static final String[] OBS_KEYS = {
        "transaction_type",
        "risk_score",
        "adversary_threat_level",
        "system_entropy",
        "kafka_lag",
        "api_latency",
        "rolling_p99",
        "db_connection_pool",
        "bank_api_status",
        "merchant_tier",
    };

    // ── Sparse Q-table ────────────────────────────────────────────────────────
    // Python: defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    // Java:   HashMap<String, float[]> where key = Arrays.toString(state_tuple)
    private final Map<String, float[]> qTable = new HashMap<>();

    /**
     * Look up Q-values for a state, creating a zero-initialised entry if absent.
     *
     * // PYTHON EQUIVALENT:
     * //   q_table[state]  → defaultdict creates np.zeros(N_ACTIONS) on first access
     */
    private float[] getQValues(int[] state) {
        String key = Arrays.toString(state);
        return qTable.computeIfAbsent(key, k -> new float[N_ACTIONS]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Action encoding / decoding
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Encode AEPOAction to a single integer in [0, 215].
     *
     * // PYTHON EQUIVALENT:
     * //   def encode_action(action):
     * //       fields = (risk_decision, crypto_verify, infra_routing,
     * //                 db_retry_policy, settlement_policy, app_priority)
     * //       return sum(f * s for f, s in zip(fields, _STRIDES))
     */
    public static int encodeAction(AEPOAction action) {
        int[] fields = {
            action.riskDecision(),
            action.cryptoVerify(),
            action.infraRouting(),
            action.dbRetryPolicy(),
            action.settlementPolicy(),
            action.appPriority(),
        };
        int idx = 0;
        for (int i = 0; i < fields.length; i++) {
            idx += fields[i] * STRIDES[i];
        }
        return idx;
    }

    /**
     * Decode integer in [0, 215] back to an AEPOAction.
     *
     * // PYTHON EQUIVALENT:
     * //   def decode_action(idx):
     * //       remaining = idx
     * //       fields = []
     * //       for stride, maxi in zip(_STRIDES, _MAXES):
     * //           fields.append(remaining // stride)
     * //           remaining %= stride
     * //       return AEPOAction(...)
     */
    public static AEPOAction decodeAction(int idx) {
        int remaining = idx;
        int[] fields = new int[6];
        for (int i = 0; i < 6; i++) {
            fields[i] = remaining / STRIDES[i];
            remaining  = remaining % STRIDES[i];
        }
        return new AEPOAction(fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // State discretisation
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Discretise a normalised observation map into a state tuple (int[10]).
     *
     * // PYTHON EQUIVALENT:
     * //   def obs_to_state(obs_normalized):
     * //       bins = []
     * //       for key in _OBS_KEYS:
     * //           val = float(obs_normalized.get(key, 0.0))
     * //           bin_idx = int(val * N_BINS)
     * //           bins.append(min(bin_idx, N_BINS - 1))
     * //       return tuple(bins)
     */
    public static int[] obsToState(Map<String, Double> obsNormalized) {
        int[] state = new int[OBS_KEYS.length];
        for (int i = 0; i < OBS_KEYS.length; i++) {
            double val = obsNormalized.getOrDefault(OBS_KEYS[i], 0.0);
            int binIdx = (int)(val * N_BINS);
            state[i] = Math.min(binIdx, N_BINS - 1);
        }
        return state;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Q-table training loop
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Train the Q-table on the hard task for N_EPISODES episodes.
     *
     * Returns the per-episode mean rewards (length=N_EPISODES).
     *
     * // PYTHON EQUIVALENT:
     * //   def train_q_table(seed=44):
     * //       env = UnifiedFintechEnv()
     * //       lag_model = LagPredictor()
     * //       q_table = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
     * //       epsilon = EPSILON_START
     * //       epsilon_decay = (EPSILON_START - EPSILON_END) / N_EPISODES
     * //       episode_means = []
     * //       for ep in range(N_EPISODES):
     * //           obs, _ = env.reset(seed=seed+ep, options={"task": TRAIN_TASK})
     * //           ... [Q-learning loop] ...
     * //       return q_table, lag_model, episode_means
     *
     * In Java, UnifiedFintechEnv and DynamicsModel are stubs (see their respective
     * Java mirror files). The logic here mirrors the Python control flow exactly.
     */
    public List<Double> trainQTable(int baseSeed) {
        // PYTHON EQUIVALENT: env = UnifiedFintechEnv()
        UnifiedFintechEnv env = new UnifiedFintechEnv();

        // PYTHON EQUIVALENT: lag_model = LagPredictor()
        DynamicsModel lagModel = new DynamicsModel();

        double epsilon = EPSILON_START;
        double epsilonDecay = (EPSILON_START - EPSILON_END) / N_EPISODES;

        List<Double> episodeMeans = new ArrayList<>();
        boolean blindSpotLogged = false;

        long tStart = System.currentTimeMillis();

        for (int ep = 0; ep < N_EPISODES; ep++) {
            int epSeed = baseSeed + ep;

            // PYTHON EQUIVALENT: obs, _ = env.reset(seed=ep_seed, options={"task": TRAIN_TASK})
            // reset() returns Map.Entry<AEPOObservation, Map<String, Object>>
            // getKey() → raw AEPOObservation; getValue() → info dict
            Map.Entry<AEPOObservation, Map<String, Object>> resetResult =
                env.reset((long) epSeed, TRAIN_TASK);
            Map<String, Double> obsNorm = resetResult.getKey().normalized();
            int[] state = obsToState(obsNorm);

            List<Double> stepRewards = new ArrayList<>();
            boolean done = false;

            while (!done) {
                // ε-greedy action selection
                int actionIdx;
                if (ThreadLocalRandom.current().nextDouble() < epsilon) {
                    actionIdx = ThreadLocalRandom.current().nextInt(N_ACTIONS);
                } else {
                    float[] qVals = getQValues(state);
                    actionIdx = argmax(qVals);
                }
                AEPOAction action = decodeAction(actionIdx);

                // PYTHON EQUIVALENT: lag_input = build_input_vector(obs_norm, action)
                // Java stub — no PyTorch tensor; represented as double[]
                double[] lagInput = DynamicsModel.buildInputVector(obsNorm, action);

                // PYTHON EQUIVALENT: next_obs, typed_reward, done, info = env.step(action)
                // step() returns UnifiedFintechEnv.StepResult (typed record, not raw Map)
                UnifiedFintechEnv.StepResult stepResult = env.step(action);
                double reward           = stepResult.reward();
                done                    = stepResult.done();
                Map<String, Object> info = stepResult.info();
                // next observation — call .normalized() on the typed AEPOObservation
                Map<String, Double> nextObsNorm = stepResult.observation().normalized();

                // Log first blind spot #1 discovery
                if (!blindSpotLogged && Boolean.TRUE.equals(info.get("blind_spot_triggered"))) {
                    logger.info(String.format(
                        "[BLIND SPOT #1 DISCOVERED] ep=%d step=%s reward=%.4f | " +
                        "Reject+SkipVerify+high_risk -> +0.04 bonus, saves 250 lag/step",
                        ep + 1, info.get("step_in_episode"), reward
                    ));
                    blindSpotLogged = true;
                }

                // Bellman Q-learning update:
                //   Q[s][a] += lr × (r + γ × max(Q[s']) - Q[s][a])
                int[] nextState = obsToState(nextObsNorm);
                float[] nextQVals = getQValues(nextState);
                double target = reward + DISCOUNT * max(nextQVals);
                float[] currentQVals = getQValues(state);
                currentQVals[actionIdx] += (float)(LEARNING_RATE * (target - currentQVals[actionIdx]));

                // PYTHON EQUIVALENT: lag_model.store_transition(lag_input, next_lag_normalized)
                double nextLagNormalized = nextObsNorm.getOrDefault("kafka_lag", 0.0);
                lagModel.storeTransition(lagInput, nextLagNormalized);

                stepRewards.add(reward);
                obsNorm = nextObsNorm;
                state = nextState;
            }

            // PYTHON EQUIVALENT: lag_model.train_step()
            lagModel.trainStep();

            // Pad crashed episodes to 100 steps with 0.0
            while (stepRewards.size() < 100) {
                stepRewards.add(0.0);
            }
            double epMean = stepRewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            episodeMeans.add(epMean);

            // ε decay
            epsilon = Math.max(EPSILON_END, epsilon - epsilonDecay);

            // Periodic log every LOG_EVERY episodes
            if ((ep + 1) % LOG_EVERY == 0) {
                int fromIdx = Math.max(0, episodeMeans.size() - LOG_EVERY);
                double recentMean = episodeMeans.subList(fromIdx, episodeMeans.size())
                    .stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                long elapsed = System.currentTimeMillis() - tStart;
                logger.info(String.format(
                    "episode=%d/%d  recent_mean=%.4f  epsilon=%.3f  elapsed=%dms",
                    ep + 1, N_EPISODES, recentMean, epsilon, elapsed
                ));
            }
        }

        logger.info(String.format(
            "Training complete — %d episodes | Q-table states=%d",
            N_EPISODES, qTable.size()
        ));
        return episodeMeans;
    }

    // ── argmax / max helpers ──────────────────────────────────────────────────

    /** Return index of maximum value in float[]. */
    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    /** Return maximum value in float[]. */
    private static double max(float[] arr) {
        float best = arr[0];
        for (float v : arr) if (v > best) best = v;
        return best;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Entry point
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Main entry point — mirrors train.py main().
     *
     * // PYTHON EQUIVALENT: if __name__ == "__main__": main()
     */
    public static void main(String[] args) {
        logger.info("=== AEPO Phase 10 — Q-Table Training (Java Mirror) ===");
        TrainQTable trainer = new TrainQTable();
        List<Double> episodeMeans = trainer.trainQTable(44);  // seed=44 (hard task grader seed)
        logger.info("Episode means collected: " + episodeMeans.size());
        // NOTE: matplotlib plot → results/reward_curve.png is Python-only.
        // In Java, write episodeMeans to a CSV and use a charting library (JFreeChart).
    }
}
