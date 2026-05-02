package aepo.types;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed per-step reward signal — Java mirror of {@code UFRGReward}.
 *
 * <p>The reward {@code value} is always clipped to [0.0, 1.0]; the {@code breakdown}
 * map preserves the signed contribution of each penalty/bonus so judges and the
 * dashboard can audit the calculation. Construction is validating: passing a
 * value outside [0, 1] throws.
 *
 * <p>Why a record (vs. a POJO with setters)? Reward is computed once per step
 * and never mutated thereafter — immutability is the correct contract.
 */
public record UFRGReward(
        double value,
        Map<String, Double> breakdown,
        boolean crashed,
        boolean circuitBreakerTripped
) {
    public UFRGReward {
        if (value < 0.0 || value > 1.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException(
                    "UFRGReward.value must be in [0.0, 1.0], got " + value);
        }
        // Defensive copy — caller cannot mutate our internal breakdown map.
        breakdown = breakdown == null ? Map.of() : new LinkedHashMap<>(breakdown);
    }
}
