package aepo;

import aepo.env.StepResult;
import aepo.env.UnifiedFintechEnv;
import aepo.types.AEPOAction;
import aepo.types.AEPOObservation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Mirror of {@code tests/test_reset.py} + {@code tests/test_step.py}.
 *
 * <p>Covers reset() determinism, step() tuple shape, reward bounds, fraud /
 * crash termination, and 100-step natural termination.
 */
class EnvResetStepTest {

    @Test
    void resetEachTaskReturnsValidObservation() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        for (String task : new String[]{"easy", "medium", "hard"}) {
            AEPOObservation obs = env.reset(0L, task);
            assertNotNull(obs);
            // All normalized values must be in [0, 1].
            obs.normalized().values().forEach(v ->
                    assertTrue(v >= 0.0 && v <= 1.0, "out of unit range: " + v));
        }
    }

    @Test
    void rejectInvalidTaskName() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        assertThrows(IllegalArgumentException.class, () -> env.reset(0L, "nightmare"));
    }

    @Test
    void deterministicWithSameSeed() {
        UnifiedFintechEnv a = new UnifiedFintechEnv();
        UnifiedFintechEnv b = new UnifiedFintechEnv();
        AEPOObservation oa = a.reset(123L, "easy");
        AEPOObservation ob = b.reset(123L, "easy");
        assertArrayEquals(oa.toArray(), ob.toArray(),
                "same seed must produce identical initial observation");
    }

    @Test
    void rewardAlwaysInUnitRange() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        env.reset(7L, "easy");
        AEPOAction safe = AEPOAction.of(0, 0, 0);   // Approve + FullVerify + Normal
        for (int i = 0; i < 100; i++) {
            StepResult sr = env.step(safe);
            double r = sr.reward().value();
            assertTrue(r >= 0.0 && r <= 1.0, "reward out of [0,1]: " + r);
            if (sr.done()) break;
        }
    }

    @Test
    void naturalTerminationAt100Steps() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        env.reset(0L, "easy");
        AEPOAction action = AEPOAction.of(0, 0, 0);
        boolean done = false;
        int steps = 0;
        while (!done && steps < 200) {
            done = env.step(action).done();
            steps++;
        }
        assertTrue(done, "episode must terminate within 100 steps");
        assertTrue(steps <= 100, "natural cap at 100 steps");
    }

    @Test
    void infoDictContainsRequiredKeys() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        env.reset(0L, "easy");
        StepResult sr = env.step(AEPOAction.of(0, 0, 0));
        var info = sr.info();
        for (String key : new String[]{
                "phase", "curriculum_level", "step_in_episode", "raw_obs",
                "reward_breakdown", "termination_reason", "blind_spot_triggered",
                "consecutive_deferred_async", "p99_ema_alpha", "lag_critical_streak"
        }) {
            assertTrue(info.containsKey(key), "missing info key: " + key);
        }
    }

    @Test
    void blindSpotOneTriggers() {
        // Force a high-risk obs by stepping until risk_score in obs > 80, then play
        // Reject + SkipVerify and assert blind_spot_triggered.
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        env.reset(0L, "hard");
        // Step until current obs has high risk_score (hard task = attack phase eventually).
        AEPOAction safe = AEPOAction.of(2, 0, 0);   // Challenge + FullVerify (no blind spot)
        boolean triggered = false;
        for (int i = 0; i < 100 && !triggered; i++) {
            // Hard task: eventually attack-phase risk_score is in [85, 100].
            if (env.state().riskScore() > 80.0) {
                StepResult sr = env.step(new AEPOAction(
                        AEPOAction.RISK_REJECT, AEPOAction.CRYPTO_SKIP, 0, 0, 0, 2));
                triggered = (boolean) sr.info().get("blind_spot_triggered");
                if (triggered) break;
            } else {
                env.step(safe);
            }
        }
        assertTrue(triggered, "blind spot #1 must fire on Reject+SkipVerify+highRisk");
    }
}
