package aepo.graders;

/**
 * Per-task graders — Java mirror of {@code EasyGrader, MediumGrader, HardGrader}.
 *
 * <p>Each grader hard-codes its task, fixed seed, and threshold (CLAUDE.md):
 * <pre>
 *   easy   ≥ 0.75   seed = 42
 *   medium ≥ 0.45   seed = 43
 *   hard   ≥ 0.30   seed = 44
 * </pre>
 *
 * <p>Implemented as a sealed interface + 3 records — gives the same "small set
 * of named grader instances" feel as Python's three classes, but with exhaustive
 * pattern-matching support for callers that switch on grader type.
 */
public final class Graders {

    private Graders() { /* static utility */ }

    public static final int N_EPISODES = 10;

    /** Common contract — mirrors the dual-interface design in the Python file. */
    public sealed interface Grader permits EasyGrader, MediumGrader, HardGrader {
        String task();
        long seed();
        double threshold();

        /** Spec-compliant evaluator — runs 10 fixed-seed episodes. */
        default double gradeAgent(PolicyFn policy) {
            return EpisodeRunner.runEpisodes(task(), policy, seed(), N_EPISODES);
        }
    }

    public record EasyGrader()   implements Grader {
        @Override public String task()      { return "easy"; }
        @Override public long seed()        { return 42L; }
        @Override public double threshold() { return 0.75; }
    }

    public record MediumGrader() implements Grader {
        @Override public String task()      { return "medium"; }
        @Override public long seed()        { return 43L; }
        @Override public double threshold() { return 0.45; }
    }

    public record HardGrader()   implements Grader {
        @Override public String task()      { return "hard"; }
        @Override public long seed()        { return 44L; }
        @Override public double threshold() { return 0.30; }
    }

    /** Factory mirroring {@code get_grader(name)} in graders.py. */
    public static Grader get(String name) {
        return switch (name) {
            case "easy"   -> new EasyGrader();
            case "medium" -> new MediumGrader();
            case "hard"   -> new HardGrader();
            default -> throw new IllegalArgumentException("Unknown grader: " + name);
        };
    }
}
