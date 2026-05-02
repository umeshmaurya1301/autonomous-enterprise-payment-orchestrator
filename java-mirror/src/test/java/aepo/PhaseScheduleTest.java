package aepo;

import aepo.env.Phase;
import aepo.env.UnifiedFintechEnv;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Mirror of {@code tests/test_phases.py} (schedule shape only — distributional checks are slow). */
class PhaseScheduleTest {

    @Test
    void easyIsAllNormal() {
        List<Phase> s = UnifiedFintechEnv.buildPhaseSchedule("easy");
        assertEquals(100, s.size());
        assertTrue(s.stream().allMatch(p -> p == Phase.NORMAL));
    }

    @Test
    void mediumIs40Normal60Spike() {
        List<Phase> s = UnifiedFintechEnv.buildPhaseSchedule("medium");
        assertEquals(100, s.size());
        assertEquals(40, s.stream().filter(p -> p == Phase.NORMAL).count());
        assertEquals(60, s.stream().filter(p -> p == Phase.SPIKE).count());
        // Order matters — Normal comes first.
        assertEquals(Phase.NORMAL, s.get(0));
        assertEquals(Phase.NORMAL, s.get(39));
        assertEquals(Phase.SPIKE,  s.get(40));
    }

    @Test
    void hardHasFourPhasesInExactOrder() {
        List<Phase> s = UnifiedFintechEnv.buildPhaseSchedule("hard");
        assertEquals(100, s.size());
        assertEquals(20, s.stream().filter(p -> p == Phase.NORMAL).count());
        assertEquals(20, s.stream().filter(p -> p == Phase.SPIKE).count());
        assertEquals(40, s.stream().filter(p -> p == Phase.ATTACK).count());
        assertEquals(20, s.stream().filter(p -> p == Phase.RECOVERY).count());
        assertEquals(Phase.NORMAL,   s.get(0));
        assertEquals(Phase.SPIKE,    s.get(20));
        assertEquals(Phase.ATTACK,   s.get(40));
        assertEquals(Phase.RECOVERY, s.get(80));
    }
}
