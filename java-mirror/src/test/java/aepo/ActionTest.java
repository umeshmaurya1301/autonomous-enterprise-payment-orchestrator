package aepo;

import aepo.types.AEPOAction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Mirror of {@code tests/test_action.py}. */
class ActionTest {

    @Test
    void acceptsValidAction() {
        assertDoesNotThrow(() -> new AEPOAction(0, 0, 0, 0, 0, 0));
        assertDoesNotThrow(() -> new AEPOAction(2, 1, 2, 1, 1, 2));
    }

    @Test
    void rejectsInvalidRiskDecision() {
        assertThrows(IllegalArgumentException.class,
                () -> new AEPOAction(3, 0, 0, 0, 0, 0));
    }

    @Test
    void rejectsNegativeInfraRouting() {
        assertThrows(IllegalArgumentException.class,
                () -> new AEPOAction(0, 0, -1, 0, 0, 0));
    }

    @Test
    void rejectsInvalidSettlementPolicy() {
        assertThrows(IllegalArgumentException.class,
                () -> new AEPOAction(0, 0, 0, 0, 5, 0));
    }

    @Test
    void factoryAppliesDefaults() {
        AEPOAction a = AEPOAction.of(0, 0, 0);
        assertEquals(0, a.dbRetryPolicy());
        assertEquals(0, a.settlementPolicy());
        assertEquals(2, a.appPriority());
    }

    @Test
    void toArraySerializesAllSixFields() {
        int[] arr = new AEPOAction(1, 0, 2, 1, 0, 1).toArray();
        assertArrayEquals(new int[]{1, 0, 2, 1, 0, 1}, arr);
    }
}
