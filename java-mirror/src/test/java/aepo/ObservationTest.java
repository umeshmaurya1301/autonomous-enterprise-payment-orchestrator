package aepo;

import aepo.types.AEPOObservation;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Mirror of {@code tests/test_observation.py}.
 *
 * <p>Validates: range enforcement in the compact constructor, normalisation
 * range, exact key set, and bank_api_status mapping (0→0.0, 1→0.5, 2→1.0).
 */
class ObservationTest {

    private AEPOObservation valid() {
        return new AEPOObservation(1, 50, 5, 50, 5000, 200, 200, 50, 1, 0);
    }

    @Test
    void acceptsAllValidFields() {
        assertDoesNotThrow(this::valid);
    }

    @Test
    void rejectsRiskScoreOverMax() {
        assertThrows(IllegalArgumentException.class,
                () -> new AEPOObservation(1, 101, 0, 0, 0, 0, 0, 50, 0, 0));
    }

    @Test
    void rejectsNegativeKafkaLag() {
        assertThrows(IllegalArgumentException.class,
                () -> new AEPOObservation(1, 50, 0, 0, -1, 0, 0, 50, 0, 0));
    }

    @Test
    void normalizedFieldsAreInUnitRange() {
        Map<String, Double> n = valid().normalized();
        n.values().forEach(v -> {
            assertTrue(v >= 0.0 && v <= 1.0,
                    "normalized value out of range: " + v);
        });
    }

    @Test
    void normalizedDictHasExactlyTenKeys() {
        assertEquals(10, valid().normalized().size());
    }

    @Test
    void bankApiStatusMapping() {
        assertEquals(0.0, healthy(0).normalized().get("bank_api_status"));
        assertEquals(0.5, healthy(1).normalized().get("bank_api_status"));
        assertEquals(1.0, healthy(2).normalized().get("bank_api_status"));
    }

    private AEPOObservation healthy(int bankStatus) {
        return new AEPOObservation(0, 10, 0, 0, 0, 50, 50, 50, bankStatus, 0);
    }
}
