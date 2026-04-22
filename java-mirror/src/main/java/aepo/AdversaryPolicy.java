package aepo;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;

/**
 * AdversaryPolicy.java — Phase 11 Java Mirror of AdversaryPolicy in unified_gateway.py
 * ======================================================================================
 * Tiny 9-state x 3-action Q-table adversary that maximises defender regret.
 *
 * PYTHON EQUIVALENT: class AdversaryPolicy (unified_gateway.py)
 *
 * This makes the Theme #4 (Self-Improvement) claim technically defensible:
 * there are now TWO learning policies in the environment — the defender and
 * the adversary — and they are genuinely antagonistic.
 *
 * NOTE: Delete this file (along with all /java-mirror/) before final submission.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * DESIGN
 * ────────────────────────────────────────────────────────────────────────────
 *
 * State (3 x 3 = 9 cells):
 *   perf_bin   — defender 5-episode rolling average bucketed: 0=low(<0.33) 1=mid 2=high(>0.67)
 *   threat_bin — current adversary_threat_level bucketed:     0=low(<3.33) 1=mid 2=high(>6.67)
 *
 * Actions:
 *   BURST   (0) — multiply lag_delta by 1.5x during spike/attack phases
 *   SUSTAIN (1) — no change to lag_delta (neutral pressure)
 *   FADE    (2) — multiply lag_delta by 0.6x, forces recovery trap
 *
 * Reward: -defender_ep_mean  (adversary wins when defender score is low)
 *
 * Q-update (episodic contextual bandit — one action per episode, terminal):
 *   Q(s,a) += lr * (-defender_ep_mean - Q(s,a))
 *
 * // PYTHON EQUIVALENT for select_action:
 * // def select_action(self, rng, defender_5ep_avg, threat_level) -> int:
 * //     state = self._state(defender_5ep_avg, threat_level)
 * //     self._last_state = state
 * //     if rng.uniform(0.0, 1.0) < self._epsilon():
 * //         action = int(rng.integers(0, 3))
 * //     else:
 * //         q_vals = [self._q[(*state, a)] for a in range(3)]
 * //         action = int(np.argmax(q_vals))
 * //     self._last_action = action
 * //     return action
 *
 * // PYTHON EQUIVALENT for update:
 * // def update(self, defender_ep_mean: float) -> None:
 * //     if self._last_state is None: return
 * //     key = (*self._last_state, self._last_action)
 * //     adv_reward = -defender_ep_mean
 * //     self._q[key] += ADV_POLICY_LR * (adv_reward - self._q[key])
 * //     self._ep_count += 1
 */
public class AdversaryPolicy {

    private static final Logger log = Logger.getLogger(AdversaryPolicy.class.getName());

    // ── Action constants ─────────────────────────────────────────────────────
    public static final int BURST   = 0;
    public static final int SUSTAIN = 1;
    public static final int FADE    = 2;

    // ── Lag multipliers per adversary action ─────────────────────────────────
    // PYTHON EQUIVALENT: LAG_MULTIPLIERS: dict[int, float] = {0: 1.5, 1: 1.0, 2: 0.6}
    public static final double[] LAG_MULTIPLIERS = {1.5, 1.0, 0.6};

    // ── Hyperparameters ───────────────────────────────────────────────────────
    private static final double LR         = 0.2;   // ADV_POLICY_LR
    private static final double EPS_START  = 0.8;   // ADV_POLICY_EPS_START
    private static final double EPS_END    = 0.1;   // ADV_POLICY_EPS_END
    private static final int    EPS_DECAY  = 200;   // ADV_POLICY_EPS_DECAY_EPS

    // ── Q-table: key = (perf_bin * 9 + threat_bin * 3 + action) for compact int key ──
    // PYTHON EQUIVALENT: self._q: dict[tuple[int, int, int], float] = defaultdict(float)
    private final Map<Integer, Double> q = new HashMap<>();

    // ── Episode tracking ──────────────────────────────────────────────────────
    private int   epCount    = 0;
    private int   lastPerfBin   = 1;
    private int   lastThreatBin = 0;
    private int   lastAction    = SUSTAIN;

    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * Bucket a value in [lo, hi] into {0, 1, 2} using two equal-width thresholds.
     *
     * // PYTHON EQUIVALENT:
     * // @staticmethod
     * // def _bin3(value: float, lo: float, hi: float) -> int:
     * //     mid = (hi - lo) / 3.0
     * //     if value < lo + mid: return 0
     * //     if value < lo + 2 * mid: return 1
     * //     return 2
     */
    private static int bin3(double value, double lo, double hi) {
        double mid = (hi - lo) / 3.0;
        if (value < lo + mid)      return 0;
        if (value < lo + 2 * mid)  return 1;
        return 2;
    }

    /** Compact int key for the (perf_bin, threat_bin, action) triple. */
    private static int qKey(int perfBin, int threatBin, int action) {
        return perfBin * 9 + threatBin * 3 + action;
    }

    private double qVal(int perfBin, int threatBin, int action) {
        return q.getOrDefault(qKey(perfBin, threatBin, action), 0.0);
    }

    /**
     * Current exploration probability, decaying linearly over EPS_DECAY episodes.
     *
     * // PYTHON EQUIVALENT:
     * // def _epsilon(self) -> float:
     * //     t = min(self._ep_count / max(1, ADV_POLICY_EPS_DECAY_EPS), 1.0)
     * //     return ADV_POLICY_EPS_START + t * (ADV_POLICY_EPS_END - ADV_POLICY_EPS_START)
     */
    private double epsilon() {
        double t = Math.min((double) epCount / Math.max(1, EPS_DECAY), 1.0);
        return EPS_START + t * (EPS_END - EPS_START);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Choose adversary action for the upcoming episode (epsilon-greedy).
     * Caches the selected state and action for use in the next {@code update()} call.
     *
     * @param rng              Java Random — used for epsilon-greedy exploration
     * @param defender5epAvg   defender 5-episode rolling average reward [0, 1]
     * @param threatLevel      current adversary_threat_level [0, 10]
     * @return selected adversary action: BURST(0), SUSTAIN(1), or FADE(2)
     */
    public int selectAction(Random rng, double defender5epAvg, double threatLevel) {
        int perfBin   = bin3(defender5epAvg, 0.0, 1.0);
        int threatBin = bin3(threatLevel, 0.0, 10.0);
        lastPerfBin   = perfBin;
        lastThreatBin = threatBin;

        int action;
        if (rng.nextDouble() < epsilon()) {
            action = rng.nextInt(3);  // random exploration
        } else {
            // Greedy: pick action with highest Q-value
            action = BURST;
            double bestQ = qVal(perfBin, threatBin, BURST);
            for (int a = 1; a <= FADE; a++) {
                double v = qVal(perfBin, threatBin, a);
                if (v > bestQ) { bestQ = v; action = a; }
            }
        }
        lastAction = action;
        return action;
    }

    /**
     * Episodic Q-update: reward = -defender_ep_mean (adversary maximises regret).
     *
     * @param defenderEpMean  mean reward the defender achieved this episode [0, 1]
     */
    public void update(double defenderEpMean) {
        double advReward = -defenderEpMean;
        int key = qKey(lastPerfBin, lastThreatBin, lastAction);
        double current = q.getOrDefault(key, 0.0);
        q.put(key, current + LR * (advReward - current));
        epCount++;
        log.fine(String.format(
            "[ADVERSARY-POLICY] ep=%d state=(%d,%d) action=%d adv_reward=%.3f eps=%.3f",
            epCount, lastPerfBin, lastThreatBin, lastAction, advReward, epsilon()
        ));
    }

    /**
     * Return the lag_delta multiplier for the last selected action.
     *
     * // PYTHON EQUIVALENT:
     * // def lag_multiplier(self) -> float:
     * //     return self.LAG_MULTIPLIERS[self._last_action]
     */
    public double lagMultiplier() {
        return LAG_MULTIPLIERS[lastAction];
    }

    /**
     * Human-readable label for the last selected action.
     *
     * // PYTHON EQUIVALENT:
     * // def action_name(self) -> str:
     * //     return {0: "Burst", 1: "Sustain", 2: "Fade"}[self._last_action]
     */
    public String actionName() {
        return switch (lastAction) {
            case BURST   -> "Burst";
            case SUSTAIN -> "Sustain";
            case FADE    -> "Fade";
            default      -> "Unknown";
        };
    }

    /** Return episode count (number of Q-updates performed). */
    public int episodeCount() {
        return epCount;
    }
}
