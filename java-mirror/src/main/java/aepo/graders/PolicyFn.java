package aepo.graders;

import aepo.types.AEPOAction;

import java.util.Map;
import java.util.function.Function;

/**
 * Policy-function alias: normalized observation dict → AEPOAction.
 *
 * <p>Java mirror of Python's {@code Callable[[dict[str, float]], AEPOAction]}
 * type alias from {@code graders.py}. We extend Function instead of using a
 * raw lambda so the type signature carries documentation.
 */
@FunctionalInterface
public interface PolicyFn extends Function<Map<String, Double>, AEPOAction> {
}
