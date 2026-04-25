package aepo;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * DynamicsModel.java — Phase 9 Java Mirror of dynamics_model.py
 * ==============================================================
 * Mirrors the LagPredictor 2-layer MLP from dynamics_model.py.
 *
 * PYTHON EQUIVALENT:
 *   dynamics_model.py — class LagPredictor(nn.Module)
 *
 * This file exists for readability by Java/Spring Boot engineers attending
 * the pitch. The Python version is the authoritative implementation.
 * This class does NOT compile or run MLP inference; it documents the
 * architecture, input encoding, and training loop as Java pseudocode.
 *
 * NOTE: Delete this file (along with all /java-mirror/) before final submission.
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * MODEL ARCHITECTURE
 * ──────────────────────────────────────────────────────────────────────────────
 *
 *   Input  : 16 float values
 *              [0..9]   AEPOObservation.normalized() — all in [0.0, 1.0]
 *              [10..15] 6 action scalars normalized to [0.0, 1.0]:
 *                         risk_decision    / 2.0   (max=2)
 *                         crypto_verify    / 1.0   (max=1)
 *                         infra_routing    / 2.0   (max=2)
 *                         db_retry_policy  / 1.0   (max=1)
 *                         settlement_policy/ 1.0   (max=1)
 *                         app_priority     / 2.0   (max=2)
 *
 *   Hidden : 64 units, ReLU activation
 *
 *   Output : 1 float — predicted next kafka_lag normalized [0.0, 1.0]
 *            Sigmoid applied to constrain output range
 *
 *   Python network definition:
 *     self.net = nn.Sequential(
 *         nn.Linear(16, 64),
 *         nn.ReLU(),
 *         nn.Linear(64, 1),
 *         nn.Sigmoid()
 *     )
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * WHY THIS JUSTIFIES "WORLD MODELING" (THEME 3.1)
 * ──────────────────────────────────────────────────────────────────────────────
 * The agent's main challenge is predicting Kafka lag cascades before they
 * cause system crashes. The LagPredictor models the causal transition:
 *
 *   Causal Transition #1: api_latency[t+1] += 0.1 × max(0, kafka_lag[t] - 3000)
 *   Causal Transition #2: Throttle action → -150 kafka_lag over next 2 steps
 *
 * By learning to predict next_lag from (obs, action), the agent can simulate
 * "what if I throttle now?" before committing to a decision — that is
 * world modeling in the operational RL sense.
 */
public class DynamicsModel {

    // ── Architecture constants ───────────────────────────────────────────────
    /** Input dimension: 10 obs fields + 6 action scalars. */
    private static final int INPUT_DIM = 16;

    /** Hidden layer width. */
    private static final int HIDDEN_DIM = 64;

    /** Output dimension: 1 (next kafka_lag normalized). */
    private static final int OUTPUT_DIM = 1;

    /** Adam learning rate. */
    private static final double LEARNING_RATE = 1e-3;

    /** Maximum replay buffer capacity (oldest entry evicted when full). */
    private static final int REPLAY_CAPACITY = 2000;

    /** Mini-batch size for each train step. */
    private static final int BATCH_SIZE = 32;

    /**
     * Max value for each of the 6 action fields.
     * Used to normalize discrete action scalars to [0.0, 1.0].
     *
     * // PYTHON EQUIVALENT:
     * // _ACTION_MAXES: tuple[float, ...] = (2.0, 1.0, 2.0, 1.0, 1.0, 2.0)
     */
    private static final double[] ACTION_MAXES = {2.0, 1.0, 2.0, 1.0, 1.0, 2.0};

    /**
     * Canonical obs field order (must match AEPOObservation.normalized()
     * field declaration order in unified_gateway.py).
     */
    private static final String[] OBS_KEYS = {
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

    // ── Replay buffer ────────────────────────────────────────────────────────

    /**
     * Transition record: (16-dim input, scalar target).
     *
     * // PYTHON EQUIVALENT:
     * // self._buffer: deque[tuple[torch.Tensor, float]] = deque(maxlen=REPLAY_CAPACITY)
     */
    private record Transition(double[] input, double target) {}

    private final Deque<Transition> buffer = new ArrayDeque<>(REPLAY_CAPACITY);

    // ── MLP weights (pseudocode — not initialized for real inference) ────────

    /**
     * Layer 1 weight matrix: shape (64, 16).
     *
     * // PYTHON EQUIVALENT: nn.Linear(16, 64) — weight shape [64, 16], bias [64]
     *
     * In Python, PyTorch initializes these with Kaiming uniform by default.
     * This Java stub documents the weight shapes for architectural clarity.
     */
    @SuppressWarnings("unused")
    private final double[][] w1 = new double[HIDDEN_DIM][INPUT_DIM];  // (64, 16)

    /**
     * Layer 1 bias vector: shape (64,).
     */
    @SuppressWarnings("unused")
    private final double[] b1 = new double[HIDDEN_DIM];

    /**
     * Layer 2 weight matrix: shape (1, 64).
     *
     * // PYTHON EQUIVALENT: nn.Linear(64, 1) — weight shape [1, 64], bias [1]
     */
    @SuppressWarnings("unused")
    private final double[][] w2 = new double[OUTPUT_DIM][HIDDEN_DIM];  // (1, 64)

    /**
     * Layer 2 bias vector: shape (1,).
     */
    @SuppressWarnings("unused")
    private final double[] b2 = new double[OUTPUT_DIM];

    // ── Input encoding ───────────────────────────────────────────────────────

    /**
     * Build the 16-dim input vector from a normalized obs dict and AEPOAction.
     *
     * // PYTHON EQUIVALENT:
     * // def build_input_vector(
     * //     obs_normalized: dict[str, float],
     * //     action: AEPOAction,
     * // ) -> torch.Tensor:
     * //     obs_vals = [float(obs_normalized[k]) for k in obs_keys]
     * //     action_vals = [float(v) / m for v, m in zip(action_vals_raw, _ACTION_MAXES)]
     * //     return torch.tensor(obs_vals + action_vals, dtype=torch.float32)
     *
     * @param obsNormalized Map of 10 normalized observation fields (all in [0.0, 1.0])
     * @param action        AEPOAction with 6 discrete fields
     * @return double[16] input vector with all values in [0.0, 1.0]
     */
    public static double[] buildInputVector(
            Map<String, Double> obsNormalized,
            AEPOAction action) {

        double[] x = new double[INPUT_DIM];

        // Obs fields [0..9] — canonical key order
        for (int i = 0; i < OBS_KEYS.length; i++) {
            x[i] = obsNormalized.getOrDefault(OBS_KEYS[i], 0.0);
        }

        // Action fields [10..15] — normalized to [0.0, 1.0] by max value
        int[] rawActions = {
            action.riskDecision(),
            action.cryptoVerify(),
            action.infraRouting(),
            action.dbRetryPolicy(),
            action.settlementPolicy(),
            action.appPriority(),
        };
        for (int i = 0; i < rawActions.length; i++) {
            x[10 + i] = rawActions[i] / ACTION_MAXES[i];
        }

        return x;
    }

    // ── Forward pass (pseudocode — not executable) ───────────────────────────

    /**
     * Forward pass through the 2-layer MLP.
     *
     * // PYTHON EQUIVALENT:
     * // def forward(self, x: torch.Tensor) -> torch.Tensor:
     * //     return self.net(x)   # Linear(16→64) → ReLU → Linear(64→1) → Sigmoid
     *
     * Java pseudocode showing the computation graph:
     *
     *   h = relu(w1 @ x + b1)   // (64,) hidden activations
     *   y = sigmoid(w2 @ h + b2) // (1,)  output in (0, 1)
     *
     * @param x input vector of shape (16,)
     * @return predicted next kafka_lag normalized — single float in (0.0, 1.0)
     */
    public double forward(double[] x) {
        // Layer 1: h = ReLU(W1 @ x + b1)
        double[] h = new double[HIDDEN_DIM];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            double z = b1[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                z += w1[j][i] * x[i];
            }
            h[j] = Math.max(0.0, z);  // ReLU
        }

        // Layer 2: y = Sigmoid(W2 @ h + b2)
        double z2 = b2[0];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            z2 += w2[0][j] * h[j];
        }
        return sigmoid(z2);
    }

    // ── Replay buffer management ─────────────────────────────────────────────

    /**
     * Add a transition to the replay buffer, evicting the oldest if at capacity.
     *
     * // PYTHON EQUIVALENT:
     * // def store_transition(self, x: torch.Tensor, next_kafka_lag_normalized: float) -> None:
     * //     self._buffer.append((x.detach(), float(next_kafka_lag_normalized)))
     *
     * @param input                    16-dim input vector from buildInputVector()
     * @param nextKafkaLagNormalized   actual next kafka_lag ÷ 10000, in [0.0, 1.0]
     */
    public void storeTransition(double[] input, double nextKafkaLagNormalized) {
        if (buffer.size() >= REPLAY_CAPACITY) {
            buffer.pollFirst();  // evict oldest
        }
        buffer.addLast(new Transition(input.clone(), nextKafkaLagNormalized));
    }

    /**
     * Return current replay buffer size.
     *
     * // PYTHON EQUIVALENT: def buffer_size(self) -> int: return len(self._buffer)
     */
    public int bufferSize() {
        return buffer.size();
    }

    /**
     * Predict next kafka_lag (normalized) for a single 16-dim input vector.
     *
     * Called by DynaPlanner.plan() during Dyna-Q imagined transitions.
     *
     * // PYTHON EQUIVALENT:
     * //   def predict_single(self, x: torch.Tensor) -> float:
     * //       self.eval()
     * //       with torch.no_grad():
     * //           out = self(x.unsqueeze(0))   # (1,16) → (1,1)
     * //       return float(out.squeeze().item())
     *
     * Java: wraps forward() in eval mode (no gradient tracking).
     * In Python this is a real PyTorch forward pass; here it is a stub
     * using the same linear + ReLU + linear + sigmoid math as forward().
     *
     * @param input 16-dim input vector from buildInputVector()
     * @return predicted next kafka_lag normalized to [0.0, 1.0]
     */
    public double predictSingle(double[] input) {
        // PYTHON EQUIVALENT: self.eval() — sets model to inference mode (no dropout/batchnorm update)
        // Java has no equivalent; forward() is always deterministic in this stub.
        return forward(input);
    }

    // ── Training step (pseudocode) ───────────────────────────────────────────

    /**
     * Draw one mini-batch from the replay buffer and perform one gradient step.
     *
     * // PYTHON EQUIVALENT:
     * // def train_step(self) -> float | None:
     * //     if len(self._buffer) < BATCH_SIZE:
     * //         return None
     * //     indices = torch.randint(len(self._buffer), (BATCH_SIZE,))
     * //     batch_x = torch.stack([self._buffer[i][0] for i in indices])
     * //     batch_y = torch.tensor([...]).unsqueeze(1)
     * //     loss = self._loss_fn(self(batch_x), batch_y)
     * //     self._optimizer.zero_grad(); loss.backward(); self._optimizer.step()
     * //     return float(loss.item())
     *
     * In Java, actual gradient computation requires a library (e.g., DL4J).
     * This stub documents the algorithm — the Python/PyTorch version is authoritative.
     *
     * @return MSE loss as double if batch available; null otherwise
     */
    public Double trainStep() {
        if (buffer.size() < BATCH_SIZE) {
            return null;
        }

        // Sample BATCH_SIZE random indices from the buffer (with replacement)
        List<Transition> bufferList = new ArrayList<>(buffer);
        Random rng = new Random();

        double mse = 0.0;
        for (int b = 0; b < BATCH_SIZE; b++) {
            Transition t = bufferList.get(rng.nextInt(bufferList.size()));
            double pred = forward(t.input());
            double err = pred - t.target();
            mse += err * err;
            // NOTE: In PyTorch, loss.backward() + optimizer.step() updates weights here.
            // Java pseudocode omits gradient computation — see dynamics_model.py.
        }
        mse /= BATCH_SIZE;

        return mse;
    }

    // ── Model-based infra planner (used in inference.py) ────────────────────

    /**
     * Normalized kafka_lag threshold above which the model-based planner fires.
     *
     * // PYTHON EQUIVALENT: LAG_OVERRIDE_THRESHOLD: float = 0.30  (in inference.py)
     */
    private static final double LAG_OVERRIDE_THRESHOLD = 0.30;

    /**
     * Model-based infra planner: queries the LagPredictor for all three
     * infra_routing choices and returns the index with lowest predicted next lag.
     *
     * Only fires when normalized kafka_lag exceeds LAG_OVERRIDE_THRESHOLD (0.30).
     * All other action fields are unchanged — only infra_routing may differ.
     *
     * This is what makes the "World Modeling" (Theme 3.1) claim defensible:
     * the agent consults its learned model before committing to an action.
     *
     * // PYTHON EQUIVALENT (in inference.py):
     * // def _model_based_infra_override(
     * //     lag_model: LagPredictor,
     * //     obs: AEPOObservation,
     * //     action: AEPOAction,
     * //     step: int,
     * // ) -> AEPOAction:
     * //     norm = obs.normalized()
     * //     if norm["kafka_lag"] <= LAG_OVERRIDE_THRESHOLD:
     * //         return action
     * //     best_infra, best_pred = action.infra_routing, float("inf")
     * //     for infra_choice in range(3):
     * //         candidate = AEPOAction(..., infra_routing=infra_choice, ...)
     * //         x = build_input_vector(norm, candidate)
     * //         pred = lag_model.predict_single(x)
     * //         if pred < best_pred: best_pred, best_infra = pred, infra_choice  // track minimum
     * //     if best_infra != action.infra_routing:
     * //         print(f"[MODEL-PLAN] step={step} override: ...")
     * //         return AEPOAction(..., infra_routing=best_infra, ...)
     * //     return action
     *
     * @param obsNormalized  Map of 10 normalized obs fields from AEPOObservation.normalized()
     * @param action         Current AEPOAction from LLM or heuristic
     * @return new AEPOAction with infra_routing possibly overridden; all other fields unchanged
     */
    public AEPOAction modelBasedInfraOverride(
            Map<String, Double> obsNormalized,
            AEPOAction action) {

        double currentLag = obsNormalized.getOrDefault("kafka_lag", 0.0);
        if (currentLag <= LAG_OVERRIDE_THRESHOLD) {
            return action;  // below threshold — model-based planning not needed
        }

        int bestInfra = action.infraRouting();
        double bestPred = Double.MAX_VALUE;

        // Probe all three infra_routing options (0=Normal, 1=Throttle, 2=CircuitBreaker)
        for (int infraChoice = 0; infraChoice <= 2; infraChoice++) {
            AEPOAction candidate = new AEPOAction(
                action.riskDecision(),
                action.cryptoVerify(),
                infraChoice,          // the only field varied
                action.dbRetryPolicy(),
                action.settlementPolicy(),
                action.appPriority()
            );
            double[] x = buildInputVector(obsNormalized, candidate);
            double pred = forward(x);  // pseudocode: forward() not trained in Java
            if (pred < bestPred) {
                bestPred = pred;
                bestInfra = infraChoice;
            }
        }

        if (bestInfra != action.infraRouting()) {
            // In Python, logs: [MODEL-PLAN] step=N kafka_lag=X.XXX override: Old→New pred=[...]
            return new AEPOAction(
                action.riskDecision(),
                action.cryptoVerify(),
                bestInfra,
                action.dbRetryPolicy(),
                action.settlementPolicy(),
                action.appPriority()
            );
        }

        return action;
    }

    // ── Utility ─────────────────────────────────────────────────────────────

    /**
     * Sigmoid activation: σ(z) = 1 / (1 + e^{-z}).
     *
     * // PYTHON EQUIVALENT: torch.sigmoid(x) or nn.Sigmoid()(x)
     */
    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
}
