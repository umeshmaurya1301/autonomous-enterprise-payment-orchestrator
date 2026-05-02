package aepo;

import aepo.agents.HeuristicAgent;
import aepo.env.StepResult;
import aepo.env.UnifiedFintechEnv;
import aepo.types.AEPOAction;
import aepo.types.AEPOObservation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Mirror of {@code tests/test_heuristic.py}.
 *
 * <p>Critical assertion: the heuristic NEVER triggers blind_spot_triggered=True,
 * which is what gives the trained agent a learning differential to discover.
 */
class HeuristicAgentTest {

    @Test
    void heuristicNeverTriggersBlindSpotOne() {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        HeuristicAgent agent = new HeuristicAgent();
        env.reset(0L, "hard");

        boolean done = false;
        while (!done) {
            AEPOObservation obs = env.state();
            AEPOAction action = agent.act(obs.normalized());
            StepResult sr = env.step(action);
            assertFalse((Boolean) sr.info().get("blind_spot_triggered"),
                    "heuristic must never trigger blind spot #1");
            done = sr.done();
        }
    }

    @Test
    void heuristicAlwaysUsesBalancedAndBackoff() {
        // Blind spots #2 and #3: structural assertions on the action choice.
        HeuristicAgent agent = new HeuristicAgent();
        for (double risk : new double[]{0.1, 0.5, 0.9}) {
            for (double lag : new double[]{0.1, 0.7}) {
                AEPOAction a = agent.act(java.util.Map.of(
                        "risk_score", risk, "kafka_lag", lag, "rolling_p99", 0.3));
                assertEquals(AEPOAction.APP_BALANCED, a.appPriority(),
                        "blind spot #2: must always be Balanced");
                assertEquals(AEPOAction.DB_EXPONENTIAL_BACKOFF, a.dbRetryPolicy(),
                        "blind spot #3: must always be Backoff");
            }
        }
    }
}
