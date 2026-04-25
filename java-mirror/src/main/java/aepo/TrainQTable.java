package aepo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * TrainQTable.java — Java mirror of train.py (Phase 10, Fix B)
 *
 * Tabular Q-learning agent trained across the AEPO curriculum
 * (easy → medium → hard) on a fixed per-level episode budget.
 * This file is a readable Java translation intended for Java/Spring Boot engineers.
 * It does NOT compile or run — it documents the Python logic in Java idioms.
 *
 * Algorithm: Tabular Q-Learning with ε-greedy exploration + Dyna-Q planning.
 * State   : int[7]  — one bin index (0–3) per key reward-driving obs feature
 *                     (risk_score, kafka_lag, rolling_p99, db_pool, bank_status,
 *                      merchant_tier, adversary_threat_level). 4^7 = 16384 states.
 * Actions : 216 combinations encoded as a single int via mixed-radix encoding.
 *
 * Curriculum (Fix B): EPISODES_PER_LEVEL = {200, 300, 1500} — deterministic
 * schedule, not env-driven adaptive curriculum.  ε is restarted at each
 * curriculum boundary so every task gets a fresh exploration budget.
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

    /** Total training episodes (curriculum: easy→medium→hard). */
    static final int N_EPISODES = 2000;

    /** Bins per state feature dimension (4 bins × 7 features = 16384 states). */
    static final int N_BINS = 4;

    /** Total actions = 3×2×3×2×2×3. */
    static final int N_ACTIONS = 216;

    /** Q-table learning rate (lr in Bellman update). */
    static final double LEARNING_RATE = 0.1;

    /** Future-reward discount factor γ. */
    static final double DISCOUNT = 0.95;

    /** Curriculum tasks ordered by difficulty. Indexed by level (0=easy, 1=medium, 2=hard). */
    static final String[] CURRICULUM_TASKS = {"easy", "medium", "hard"};

    /**
     * Fixed-schedule curriculum (Fix B). Deterministic episode budget per level.
     * Sum must equal N_EPISODES. Hard gets 85% of the budget — it's the primary
     * task per CLAUDE.md spec and needs the most Q-table coverage to beat the
     * heuristic threshold (≥ 0.30). Easy/medium just need enough episodes to
     * (a) populate per-task tables for evaluation and (b) make the staircase
     * visually clear in reward_staircase.png.
     * // PYTHON EQUIVALENT: EPISODES_PER_LEVEL: tuple[int, int, int] = (100, 200, 1700)
     */
    static final int[] EPISODES_PER_LEVEL = {100, 200, 1700};

    /**
     * Fraction of exploratory steps that follow the heuristic (vs uniform random).
     * 0.0 = pure random exploration (preserves full blind-spot #1 discovery rate).
     * Heuristic-mixed exploration was tested for adaptive-curriculum advancement
     * but is unnecessary now that the schedule is fixed — task progression is
     * driven by EPISODES_PER_LEVEL, not by clearing 0.75/0.45 episode-mean gates.
     * // PYTHON EQUIVALENT: HEURISTIC_EXPLORATION_RATIO: float = 0.0
     */
    static final double HEURISTIC_EXPLORATION_RATIO = 0.0;

    /** Initial exploration rate ε. */
    static final double EPSILON_START = 1.0;

    /** Minimum exploration rate ε after decay. */
    static final double EPSILON_END = 0.05;

    /** Log a summary line every N episodes. */
    static final int LOG_EVERY = 10;

    /** Fallback task name (only used if scheduledLevel is somehow out of range). */
    static final String TRAIN_TASK_FALLBACK = "easy";

    /**
     * Post-training fine-tune episodes per task (easy + medium only).
     * Even with the fixed-schedule curriculum allocating 200/300 episodes to
     * easy/medium, the per-task Q-tables benefit from extra targeted exposure:
     * the schedule's exploration is global (ε starts at 1.0) so early-task
     * Q-values are noisy. 600 fine-tune episodes per task densify the tables.
     * // PYTHON EQUIVALENT: FINETUNE_EPISODES: int = 600
     */
    static final int FINETUNE_EPISODES = 600;

    /** Imagined Q-updates performed per real env step (Dyna-Q). */
    static final int DYNA_PLANNING_STEPS = 5;

    /** Max real transitions stored in the Dyna replay buffer. */
    static final int DYNA_BUFFER_CAPACITY = 2000;

    // ────────────────────────────────────────────────────────────────────────
    // DynaPlanner — Dyna-Q world-model planner (inner class)
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Dyna-Q planner: augments real Q-learning with LagPredictor-imagined
     * transitions.  After every real env.step(), call store() then plan().
     *
     * // PYTHON EQUIVALENT:
     * //   class DynaPlanner:
     * //       _buffer: deque[tuple[dict, int, float, dict]] = deque(maxlen=capacity)
     * //
     * //       def store(self, obs_norm, action_idx, reward, next_obs_norm): ...
     * //       def plan(self, q_table, lag_model, n_steps) -> int: ...
     */
    static class DynaPlanner {

        // PYTHON EQUIVALENT: deque(maxlen=DYNA_BUFFER_CAPACITY)
        // Java: ArrayDeque with manual capacity enforcement (Deque has no maxlen).
        private final ArrayDeque<Object[]> buffer = new ArrayDeque<>();
        private final int capacity;

        DynaPlanner(int capacity) {
            this.capacity = capacity;
        }

        /**
         * Add a real (obs, action, reward, next_obs) transition to the buffer.
         *
         * // PYTHON EQUIVALENT:
         * //   self._buffer.append((obs_norm, action_idx, reward, next_obs_norm))
         */
        void store(
            Map<String, Double> obsNorm,
            int actionIdx,
            double reward,
            Map<String, Double> nextObsNorm
        ) {
            if (buffer.size() >= capacity) {
                buffer.pollFirst();  // evict oldest (FIFO, matching deque maxlen)
            }
            buffer.addLast(new Object[]{obsNorm, actionIdx, reward, nextObsNorm});
        }

        /**
         * Perform nSteps imagined Bellman updates using the LagPredictor world model.
         *
         * For each planning step:
         *   1. Sample a random real transition from the buffer.
         *   2. Predict next kafka_lag with DynamicsModel (stub for PyTorch forward pass).
         *   3. Substitute the prediction into next_obs to get the imagined next state.
         *   4. Compute Bellman target and update Q[s][a].
         *
         * Returns the number of planning updates actually performed.
         *
         * // PYTHON EQUIVALENT:
         * //   def plan(self, q_table, lag_model, n_steps=DYNA_PLANNING_STEPS) -> int:
         * //       with torch.no_grad():
         * //           for _ in range(min(n_steps, len(self._buffer))):
         * //               obs, action_idx, reward, next_obs = random.choice(buffer)
         * //               x = build_input_vector(obs, decode_action(action_idx))
         * //               predicted_lag = float(lag_model(x.unsqueeze(0)).squeeze())
         * //               imagined_next = dict(next_obs); imagined_next["kafka_lag"] = predicted_lag
         * //               state = obs_to_state(obs); next_state = obs_to_state(imagined_next)
         * //               target = reward + DISCOUNT * max(q_table[next_state])
         * //               q_table[state][action_idx] += lr * (target - q_table[state][action_idx])
         */
        int plan(
            Map<String, float[]> qTable,
            DynamicsModel lagModel,
            int nSteps
        ) {
            if (buffer.isEmpty()) return 0;

            int available = Math.min(nSteps, buffer.size());
            Object[][] entries = buffer.toArray(new Object[0][]);
            int updates = 0;

            for (int i = 0; i < available; i++) {
                // Random sample — ThreadLocalRandom is the Java equivalent of random.randrange
                int idx = ThreadLocalRandom.current().nextInt(entries.length);
                @SuppressWarnings("unchecked")
                Map<String, Double> obsNorm     = (Map<String, Double>) entries[idx][0];
                int actionIdx                   = (Integer)             entries[idx][1];
                double reward                   = (Double)              entries[idx][2];
                @SuppressWarnings("unchecked")
                Map<String, Double> nextObsNorm = (Map<String, Double>) entries[idx][3];

                // PYTHON EQUIVALENT: x = build_input_vector(obs, decode_action(action_idx))
                // Java: build double[] input via DynamicsModel.buildInputVector (stub)
                AEPOAction action = decodeAction(actionIdx);
                double[] x = DynamicsModel.buildInputVector(obsNorm, action);

                // PYTHON EQUIVALENT: predicted_lag = float(lag_model(x.unsqueeze(0)).squeeze())
                // Java: DynamicsModel.predictSingle is a stub returning a placeholder double.
                // In production Python this is a real PyTorch forward() call under no_grad().
                double predictedLag = lagModel.predictSingle(x);

                // Build imagined next_obs — substitute model-predicted kafka_lag
                Map<String, Double> imaginedNext = new HashMap<>(nextObsNorm);
                imaginedNext.put("kafka_lag", predictedLag);

                // Bellman update on imagined transition
                int[] state     = obsToState(obsNorm);
                int[] nextState = obsToState(imaginedNext);

                float[] qVals     = qTableGet(qTable, state);
                float[] nextQVals = qTableGet(qTable, nextState);
                double target = reward + DISCOUNT * max(nextQVals);
                qVals[actionIdx] += (float)(LEARNING_RATE * (target - qVals[actionIdx]));
                updates++;
            }
            return updates;
        }

        int bufferSize() { return buffer.size(); }

        // Helper: get Q-values for state, creating zeros on first access
        private float[] qTableGet(Map<String, float[]> qTable, int[] state) {
            String key = Arrays.toString(state);
            return qTable.computeIfAbsent(key, k -> new float[N_ACTIONS]);
        }
    }

    // ── Mixed-radix strides for action encoding ──────────────────────────────
    // MultiDiscrete: [risk(3), crypto(2), infra(3), db_retry(2), settle(2), priority(3)]
    // Python: _STRIDES = (72, 36, 12, 6, 3, 1)
    static final int[] STRIDES = {72, 36, 12, 6, 3, 1};
    static final int[] MAXES   = { 3,  2,  3,  2,  2,  3};

    // ── State feature keys (Python: STATE_FEATURE_KEYS) ───────────────────────
    // 7 causal-driver subset of normalized observation — see Python comment for
    // why we don't bin all 10 fields (sparse Q-table at 8^10 ≈ 1B states).
    // Order is significant: must match Python tuple to keep state keys identical.
    static final String[] OBS_KEYS = {
        "risk_score",
        "kafka_lag",
        "rolling_p99",
        "db_connection_pool",
        "bank_api_status",
        "merchant_tier",
        "adversary_threat_level",
    };

    // ── Sparse Q-table (shared, drives epsilon-greedy during training) ──────────
    // Python: defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    // Java:   HashMap<String, float[]> where key = Arrays.toString(state_tuple)
    private final Map<String, float[]> qTable = new HashMap<>();

    // ── Per-task Q-tables (accumulate all episodes of each task's exposure) ────
    // Python: q_tables_per_task = {task: defaultdict(...) for task in CURRICULUM_TASKS}
    // Each table is updated only when that task is active; used for task-specific eval.
    private final Map<String, Map<String, float[]>> qTablesPerTask = Map.of(
        "easy",   new HashMap<>(),
        "medium", new HashMap<>(),
        "hard",   new HashMap<>()
    );

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
     * Train the Q-table for N_EPISODES episodes using the fixed-schedule
     * curriculum (Fix B): 200 easy → 300 medium → 1500 hard. ε is restarted
     * to EPSILON_START at each curriculum advance and decays linearly across
     * the new level's episode budget.
     *
     * Returns the per-episode mean rewards (length=N_EPISODES).
     *
     * // PYTHON EQUIVALENT:
     * //   def train_q_table(seed=44, use_dyna=True):
     * //       env = UnifiedFintechEnv()
     * //       lag_model = LagPredictor()
     * //       dyna_planner = DynaPlanner()
     * //       q_table = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
     * //       epsilon = EPSILON_START
     * //       _level_decays = tuple((EPSILON_START - EPSILON_END) / max(1, n)
     * //                              for n in EPISODES_PER_LEVEL)
     * //       epsilon_decay = _level_decays[0]
     * //       for ep in range(N_EPISODES):
     * //           scheduled_level = lookup_in(EPISODES_PER_LEVEL, ep)
     * //           task_to_use = CURRICULUM_TASKS[scheduled_level]
     * //           obs, _ = env.reset(seed=seed+ep, options={"task": task_to_use})
     * //           if scheduled_level > prev_level:
     * //               epsilon = EPSILON_START
     * //               epsilon_decay = _level_decays[scheduled_level]
     * //           ... [Q-learning loop] ...
     * //           if use_dyna:
     * //               dyna_planner.store(...); dyna_planner.plan(...)
     * //       return q_table, lag_model, episode_means, curriculum_levels, ...
     *
     * @param useDyna  When true, run DYNA_PLANNING_STEPS imagined Bellman updates
     *                 per real step using DynaPlanner.  False = pure Q-table baseline
     *                 (used by trainQTableForComparison to generate the baseline curve).
     *
     * In Java, UnifiedFintechEnv and DynamicsModel are stubs (see their respective
     * Java mirror files). The logic here mirrors the Python control flow exactly.
     */
    public List<Double> trainQTable(int baseSeed, boolean useDyna) {
        // PYTHON EQUIVALENT: env = UnifiedFintechEnv()
        UnifiedFintechEnv env = new UnifiedFintechEnv();

        // PYTHON EQUIVALENT: lag_model = LagPredictor()
        DynamicsModel lagModel = new DynamicsModel();

        // PYTHON EQUIVALENT: dyna_planner = DynaPlanner()
        DynaPlanner dynaPlanner = new DynaPlanner(DYNA_BUFFER_CAPACITY);

        // Per-level epsilon schedule (Fix B). Each curriculum level restarts
        // at EPSILON_START and decays linearly to EPSILON_END across its own
        // episode budget. Without this, ε would carry over near zero from the
        // end of the previous level and the agent would never explore the
        // new task's state space.
        // PYTHON EQUIVALENT:
        //   _level_decays = tuple((EPSILON_START - EPSILON_END) / max(1, n)
        //                          for n in EPISODES_PER_LEVEL)
        double epsilon = EPSILON_START;
        double[] levelDecays = new double[EPISODES_PER_LEVEL.length];
        for (int i = 0; i < EPISODES_PER_LEVEL.length; i++) {
            levelDecays[i] = (EPSILON_START - EPSILON_END) / Math.max(1, EPISODES_PER_LEVEL[i]);
        }
        double epsilonDecay = levelDecays[0];

        List<Double> episodeMeans = new ArrayList<>();
        List<Integer> curriculumLevels = new ArrayList<>();
        boolean blindSpotLogged = false;
        int totalPlanningUpdates = 0;

        long tStart = System.currentTimeMillis();

        for (int ep = 0; ep < N_EPISODES; ep++) {
            int epSeed = baseSeed + ep;

            // Fixed-schedule curriculum (Fix B). The trainer drives task
            // progression on a deterministic episode budget, ignoring the
            // env's internal adaptive curriculum (which stalls under
            // adversary escalation: when the agent does well, threat level
            // rises and the next mean drops below 0.75, breaking the streak).
            // PYTHON EQUIVALENT:
            //   cum = 0
            //   scheduled_level = len(EPISODES_PER_LEVEL) - 1
            //   for lvl, n in enumerate(EPISODES_PER_LEVEL):
            //       cum += n
            //       if ep < cum: scheduled_level = lvl; break
            int cum = 0;
            int scheduledLevel = EPISODES_PER_LEVEL.length - 1;
            for (int lvl = 0; lvl < EPISODES_PER_LEVEL.length; lvl++) {
                cum += EPISODES_PER_LEVEL[lvl];
                if (ep < cum) { scheduledLevel = lvl; break; }
            }
            String taskToUse = CURRICULUM_TASKS[scheduledLevel];

            // Detect curriculum advance (scheduled boundary crossed) and
            // restart epsilon for the new task's exploration budget.
            // PYTHON EQUIVALENT:
            //   if ep > 0 and curriculum_levels and scheduled_level > curriculum_levels[-1]:
            //       epsilon = EPSILON_START
            //       epsilon_decay = _level_decays[scheduled_level]
            if (ep > 0 && !curriculumLevels.isEmpty()
                && scheduledLevel > curriculumLevels.get(curriculumLevels.size() - 1)) {
                int prevLevel = curriculumLevels.get(curriculumLevels.size() - 1);
                logger.info(String.format(
                    "[CURRICULUM ADVANCE] episode=%d  level %d (%s) -> %d (%s)",
                    ep + 1,
                    prevLevel, CURRICULUM_TASKS[prevLevel],
                    scheduledLevel, CURRICULUM_TASKS[scheduledLevel]
                ));
                epsilon = EPSILON_START;
                epsilonDecay = levelDecays[scheduledLevel];
            }

            // PYTHON EQUIVALENT: obs, _ = env.reset(seed=ep_seed, options={"task": task_to_use})
            // reset() returns Map.Entry<AEPOObservation, Map<String, Object>>
            // getKey() → raw AEPOObservation; getValue() → info dict
            Map.Entry<AEPOObservation, Map<String, Object>> resetResult =
                env.reset((long) epSeed, taskToUse);
            Map<String, Double> obsNorm = resetResult.getKey().normalized();
            int[] state = obsToState(obsNorm);

            List<Double> stepRewards = new ArrayList<>();
            boolean done = false;

            while (!done) {
                // ε-greedy action selection with optional heuristic-mixed exploration.
                // With HEURISTIC_EXPLORATION_RATIO=0.0 (current default) the
                // exploratory branch is pure uniform-random — strongest setting
                // for blind-spot #1 discovery (Reject+SkipVerify) because the
                // heuristic itself uses FullVerify and would never sample that
                // action. Dispatcher kept in place because adaptive-curriculum
                // variants (where the env decides advancement) need a heuristic
                // mix to clear 0.75/0.45 gates; fixed-schedule does not.
                // PYTHON EQUIVALENT:
                //   if rand < epsilon:
                //       if HEURISTIC_EXPLORATION_RATIO > 0 and rand < ratio:
                //           action = heuristic_policy(obs_norm)
                //       else:
                //           action = decode_action(random.randint(0, N_ACTIONS-1))
                //   else:
                //       action = decode_action(argmax(q_table[state]))
                int actionIdx;
                AEPOAction action;
                if (ThreadLocalRandom.current().nextDouble() < epsilon) {
                    if (HEURISTIC_EXPLORATION_RATIO > 0.0
                        && ThreadLocalRandom.current().nextDouble() < HEURISTIC_EXPLORATION_RATIO) {
                        action = HeuristicAgent.policy(obsNorm);
                        actionIdx = encodeAction(action);
                    } else {
                        actionIdx = ThreadLocalRandom.current().nextInt(N_ACTIONS);
                        action = decodeAction(actionIdx);
                    }
                } else {
                    float[] qVals = getQValues(state);
                    actionIdx = argmax(qVals);
                    action = decodeAction(actionIdx);
                }

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

                // Bellman Q-learning update on shared Q-table:
                //   Q[s][a] += lr × (r + γ × max(Q[s']) - Q[s][a])
                int[] nextState = obsToState(nextObsNorm);
                float[] nextQVals = getQValues(nextState);
                double target = reward + DISCOUNT * max(nextQVals);
                float[] currentQVals = getQValues(state);
                currentQVals[actionIdx] += (float)(LEARNING_RATE * (target - currentQVals[actionIdx]));

                // Per-task Q-table update — same Bellman target but task-specific table.
                // Uses the per-task table's own Q-values for the next-state max so its
                // values are self-consistent for greedy evaluation. The table picked
                // here matches the scheduled task for this episode (Fix B).
                // PYTHON EQUIVALENT:
                //   per_task_table = q_tables_per_task[task_to_use]
                //   per_task_target = reward + DISCOUNT * max(per_task_table[next_state])
                //   per_task_table[state][a] += lr * (per_task_target - per_task_table[state][a])
                Map<String, float[]> perTaskTable = qTablesPerTask.get(taskToUse);
                String stateKey     = Arrays.toString(state);
                String nextStateKey = Arrays.toString(nextState);
                float[] ptCurrent = perTaskTable.computeIfAbsent(stateKey,     k -> new float[N_ACTIONS]);
                float[] ptNext    = perTaskTable.computeIfAbsent(nextStateKey, k -> new float[N_ACTIONS]);
                double ptTarget = reward + DISCOUNT * max(ptNext);
                ptCurrent[actionIdx] += (float)(LEARNING_RATE * (ptTarget - ptCurrent[actionIdx]));

                // PYTHON EQUIVALENT: lag_model.store_transition(lag_input, next_lag_normalized)
                double nextLagNormalized = nextObsNorm.getOrDefault("kafka_lag", 0.0);
                lagModel.storeTransition(lagInput, nextLagNormalized);

                // PYTHON EQUIVALENT:
                //   if use_dyna:
                //       dyna_planner.store(obs_norm, action_idx, reward, next_obs_norm)
                //       total_planning_updates += dyna_planner.plan(q_table, lag_model)
                if (useDyna) {
                    dynaPlanner.store(obsNorm, actionIdx, reward, nextObsNorm);
                    totalPlanningUpdates += dynaPlanner.plan(qTable, lagModel, DYNA_PLANNING_STEPS);
                }

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
            curriculumLevels.add(scheduledLevel);

            // ε decay (uses the level-specific decay slope, refreshed at each
            // curriculum advance — see top of loop).
            // PYTHON EQUIVALENT: epsilon = max(EPSILON_END, epsilon - epsilon_decay)
            epsilon = Math.max(EPSILON_END, epsilon - epsilonDecay);

            // Periodic log every LOG_EVERY episodes
            if ((ep + 1) % LOG_EVERY == 0) {
                int fromIdx = Math.max(0, episodeMeans.size() - LOG_EVERY);
                double recentMean = episodeMeans.subList(fromIdx, episodeMeans.size())
                    .stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                long elapsed = System.currentTimeMillis() - tStart;
                // PYTHON EQUIVALENT: logger.info("episode=... planning_updates=%d dyna_buffer=%d ...")
                logger.info(String.format(
                    "episode=%d/%d  recent_mean=%.4f  epsilon=%.3f  " +
                    "planning_updates=%d  dyna_buffer=%d  elapsed=%dms",
                    ep + 1, N_EPISODES, recentMean, epsilon,
                    totalPlanningUpdates, dynaPlanner.bufferSize(), elapsed
                ));
            }
        }

        logger.info(String.format(
            "Training complete — %d episodes | Q-table states=%d | " +
            "Planning Updates Performed=%d",
            N_EPISODES, qTable.size(), totalPlanningUpdates
        ));
        return episodeMeans;
    }

    // ── Trained policy with heuristic fallback (Fix 9.5.A) ────────────────────

    /**
     * Per-task Q-confidence thresholds (Fix 9.5.A — empirically tuned).
     *
     * Empirical sweep on saved Q-tables (10 eval episodes each):
     *     T=0   easy=0.42 medium=0.10 hard=0.36
     *     T=8   easy=0.42 medium=0.51 hard=0.28
     *     T=∞   easy=0.76 medium=0.53 hard=0.25
     *
     * Hard wants T=0 (Q-table beats heuristic by +0.11). Easy/medium want
     * T=∞ (Q-table hurts: 600-episode fine-tune from empty produces Q-values
     * whose argmax picks systematically poor actions). Per-task thresholding
     * lets each task use its best policy.
     */
    static final Map<String, Double> Q_CONF_THRESHOLD_PER_TASK = Map.of(
        "easy",   Double.POSITIVE_INFINITY,
        "medium", Double.POSITIVE_INFINITY,
        "hard",   0.0
    );

    /**
     * Build a greedy policy over the per-task Q-table that falls back to the
     * heuristic policy for unknown / low-confidence states.
     *
     * Fix A (Bug 9.5.A): the previous fallback returned a fixed
     * Reject + SkipVerify action, which is catastrophic on easy/medium where
     * most transactions are low-risk and Approve is optimal. The heuristic
     * already scores 0.76/0.53/0.25 on easy/medium/hard — strictly better
     * than fixed-Reject as a fallback. The confidence threshold lets the
     * caller switch behaviour per task: T=0 always trusts the Q-table when
     * the state is present (use on hard); T=∞ always falls back to heuristic
     * (use on easy/medium where finetune Q-values are too noisy to trust).
     *
     * // PYTHON EQUIVALENT:
     * //   def policy_fn(obs):
     * //       state = obs_to_state(obs)
     * //       if state in q_table:
     * //           row = q_table[state]
     * //           if max(row) >= confidence_threshold:
     * //               return decode_action(int(np.argmax(row)))
     * //       return heuristic_policy(obs)
     */
    static AEPOAction trainedPolicy(
        Map<String, float[]> qTable,
        Map<String, Double> obsNormalized,
        double confidenceThreshold
    ) {
        int[] state = obsToState(obsNormalized);
        String key = Arrays.toString(state);
        float[] row = qTable.get(key);
        if (row != null && max(row) >= confidenceThreshold) {
            return decodeAction(argmax(row));
        }
        return HeuristicAgent.policy(obsNormalized);
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
    // Post-training fine-tuning (easy + medium per-task Q-tables)
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Fine-tune easy and medium per-task Q-tables with dedicated episodes.
     *
     * The fixed-schedule curriculum allocates 200 easy + 300 medium episodes
     * before advancing to hard for the remaining 1500 episodes.  Even with
     * that coverage, easy/medium per-task tables benefit from extra targeted
     * exposure: the schedule's exploration starts noisy (ε=1.0) so early-task
     * Q-values still need densifying.  This method runs FINETUNE_EPISODES
     * targeted episodes per task with a moderate epsilon.
     *
     * // PYTHON EQUIVALENT:
     * //   def finetune_per_task_qtables(q_tables_per_task, n_episodes=FINETUNE_EPISODES):
     * //       for task in ["easy", "medium"]:
     * //           q_table = q_tables_per_task[task]
     * //           env = UnifiedFintechEnv()
     * //           epsilon = 0.70  # explore-heavy fresh burst on top of curriculum exposure
     * //           for ep in range(n_episodes):
     * //               obs, _ = env.reset(seed=..., options={"task": task})
     * //               ... [Q-learning loop without dyna] ...
     * //           logger.info("[FINETUNE] task='%s' complete — %d episodes ...", task, n_episodes, ...)
     */
    public void finetunePerTaskQtables(int nEpisodes) {
        String[] finetuneTasks = {"easy", "medium"};
        // Explore-heavy starting ε: even with 200/300 curriculum episodes on
        // easy/medium, those episodes used ε≈1.0 → ε≈0.05 over the schedule,
        // so the per-task tables still benefit from a fresh exploration burst.
        double ftEpsilonStart = 0.70;
        double ftEpsilonDecay = (ftEpsilonStart - EPSILON_END) / Math.max(1, nEpisodes);

        for (int taskIdx = 0; taskIdx < finetuneTasks.length; taskIdx++) {
            String task = finetuneTasks[taskIdx];
            Map<String, float[]> qTable = qTablesPerTask.get(task);
            UnifiedFintechEnv env = new UnifiedFintechEnv();
            double epsilon = ftEpsilonStart;

            for (int ep = 0; ep < nEpisodes; ep++) {
                // Seeds offset beyond main training range to avoid overlap
                int epSeed = 44 + N_EPISODES + taskIdx * nEpisodes + ep;
                Map.Entry<AEPOObservation, Map<String, Object>> resetResult =
                    env.reset((long) epSeed, task);
                Map<String, Double> obsNorm = resetResult.getKey().normalized();
                int[] state = obsToState(obsNorm);
                boolean done = false;

                while (!done) {
                    int actionIdx;
                    if (ThreadLocalRandom.current().nextDouble() < epsilon) {
                        actionIdx = ThreadLocalRandom.current().nextInt(N_ACTIONS);
                    } else {
                        String key = Arrays.toString(state);
                        float[] qVals = qTable.computeIfAbsent(key, k -> new float[N_ACTIONS]);
                        actionIdx = argmax(qVals);
                    }

                    UnifiedFintechEnv.StepResult sr = env.step(decodeAction(actionIdx));
                    double reward = sr.reward();
                    done = sr.done();
                    Map<String, Double> nextObsNorm = sr.observation().normalized();
                    int[] nextState = obsToState(nextObsNorm);

                    // Bellman update on per-task Q-table only
                    String sk  = Arrays.toString(state);
                    String nsk = Arrays.toString(nextState);
                    float[] cur  = qTable.computeIfAbsent(sk,  k -> new float[N_ACTIONS]);
                    float[] next = qTable.computeIfAbsent(nsk, k -> new float[N_ACTIONS]);
                    double target = reward + DISCOUNT * max(next);
                    cur[actionIdx] += (float)(LEARNING_RATE * (target - cur[actionIdx]));

                    obsNorm = nextObsNorm;
                    state = nextState;
                }

                epsilon = Math.max(EPSILON_END, epsilon - ftEpsilonDecay);
            }

            logger.info(String.format(
                "[FINETUNE] task='%s' complete — %d episodes, states=%d",
                task, nEpisodes, qTable.size()
            ));
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Dyna-Q convergence comparison
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Run two training passes (with and without Dyna-Q) and log convergence
     * episode for each pass.  Mirrors Python plot_dyna_comparison().
     *
     * // PYTHON EQUIVALENT:
     * //   def plot_dyna_comparison(output_path, seed=44):
     * //       means_no_dyna   = train_q_table(seed=seed, use_dyna=False)[2]
     * //       means_with_dyna = train_q_table(seed=seed, use_dyna=True)[2]
     * //       # plot both curves on one chart titled "World Model Accelerates Learning"
     *
     * In Java we log the first-cross episodes instead of producing a matplotlib chart.
     */
    public void runDynaComparison(int seed) {
        logger.info("=== Dyna-Q Comparison: TWO training passes (seed=%d) ===", seed);

        logger.info("Pass 1/2 — Q-table WITHOUT Dyna-Q (baseline) ...");
        TrainQTable baselineTrainer = new TrainQTable();
        List<Double> meansNoDyna = baselineTrainer.trainQTable(seed, false);

        logger.info("Pass 2/2 — Q-table WITH Dyna-Q (world model) ...");
        TrainQTable dynaTrainer = new TrainQTable();
        List<Double> meansWithDyna = dynaTrainer.trainQTable(seed, true);

        double threshold = 0.30;  // hard task threshold
        int crossNoDyna   = firstCross(meansNoDyna,   threshold);
        int crossWithDyna = firstCross(meansWithDyna, threshold);

        if (crossNoDyna > 0 && crossWithDyna > 0) {
            double speedup = (double) crossNoDyna / crossWithDyna;
            logger.info(String.format(
                "[DYNA-Q PROOF] Baseline crosses threshold at ep %d. " +
                "Dyna-Q crosses at ep %d. Speedup: %.1f×",
                crossNoDyna, crossWithDyna, speedup
            ));
        } else if (crossWithDyna > 0) {
            logger.info(String.format(
                "[DYNA-Q PROOF] Dyna-Q crosses threshold at ep %d. " +
                "Baseline never crossed in %d episodes.",
                crossWithDyna, N_EPISODES
            ));
        } else {
            logger.warning(
                "[DYNA-Q PROOF] Neither run crossed the threshold. Check hyperparameters."
            );
        }
    }

    /** Return first 1-indexed episode where 10-ep rolling mean >= threshold, or -1. */
    private static int firstCross(List<Double> means, double threshold) {
        int w = 10;
        for (int i = w - 1; i < means.size(); i++) {
            double rolling = means.subList(i - w + 1, i + 1)
                .stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            if (rolling >= threshold) return i + 1;
        }
        return -1;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Entry point
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Main entry point — mirrors train.py main().
     *
     * Pass "--compare-dyna" as first arg to run the two-pass comparison.
     *
     * // PYTHON EQUIVALENT: if __name__ == "__main__": main()
     */
    public static void main(String[] args) {
        logger.info("=== AEPO Phase 10 — Q-Table Training (Java Mirror) ===");
        TrainQTable trainer = new TrainQTable();

        boolean compareDyna = args.length > 0 && "--compare-dyna".equals(args[0]);
        if (compareDyna) {
            // PYTHON EQUIVALENT: plot_dyna_comparison(results_dir / "dyna_comparison.png")
            trainer.runDynaComparison(44);
        }

        // PYTHON EQUIVALENT: train_q_table(seed=44, use_dyna=True)
        List<Double> episodeMeans = trainer.trainQTable(44, true);
        logger.info("Episode means collected: " + episodeMeans.size());

        // PYTHON EQUIVALENT: finetune_per_task_qtables(q_tables_per_task, n_episodes=FINETUNE_EPISODES)
        // Fine-tune easy + medium per-task Q-tables to overcome sparse curriculum exposure.
        trainer.finetunePerTaskQtables(FINETUNE_EPISODES);

        // NOTE: matplotlib plot → results/reward_curve.png is Python-only.
        // In Java, write episodeMeans to a CSV and use a charting library (JFreeChart).
    }
}
