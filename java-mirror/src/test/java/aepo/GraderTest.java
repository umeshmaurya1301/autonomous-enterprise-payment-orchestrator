package aepo;

import aepo.agents.HeuristicAgent;
import aepo.graders.Graders;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Mirror of {@code tests/test_graders.py}. */
class GraderTest {

    @Test
    void registryReturnsExpectedTypes() {
        assertTrue(Graders.get("easy")   instanceof Graders.EasyGrader);
        assertTrue(Graders.get("medium") instanceof Graders.MediumGrader);
        assertTrue(Graders.get("hard")   instanceof Graders.HardGrader);
    }

    @Test
    void thresholdsMatchSpec() {
        assertEquals(0.75, Graders.get("easy").threshold());
        assertEquals(0.45, Graders.get("medium").threshold());
        assertEquals(0.30, Graders.get("hard").threshold());
    }

    @Test
    void seedsAreFixed() {
        assertEquals(42L, Graders.get("easy").seed());
        assertEquals(43L, Graders.get("medium").seed());
        assertEquals(44L, Graders.get("hard").seed());
    }

    @Test
    void deterministicGivenSamePolicyAndSeed() {
        // Two grader runs with the same stateless policy on the same seed must agree.
        HeuristicAgent agent = new HeuristicAgent();
        Graders.Grader g = Graders.get("easy");
        double a = g.gradeAgent(agent::act);
        double b = g.gradeAgent(agent::act);
        assertEquals(a, b, 1e-9);
    }

    @Test
    void scoresStayInUnitRange() {
        HeuristicAgent agent = new HeuristicAgent();
        for (String t : new String[]{"easy", "medium", "hard"}) {
            double s = Graders.get(t).gradeAgent(agent::act);
            assertTrue(s >= 0.0 && s <= 1.0, t + " score out of range: " + s);
        }
    }
}
