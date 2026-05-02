package aepo.graders;

import aepo.env.StepResult;
import aepo.env.UnifiedFintechEnv;
import aepo.types.AEPOAction;
import aepo.types.AEPOObservation;

import java.util.ArrayList;
import java.util.List;

/**
 * Internal episode runner — Java mirror of {@code _run_episodes()} in graders.py.
 *
 * <p>Runs N episodes deterministically (seed = base_seed + ep_index), pads
 * crashed/short episodes with 0.0 to {@code env.maxSteps()}, and returns the
 * mean per-step reward over all episodes. Return value is rounded to 4 decimals
 * to match the Python output exactly.
 */
public final class EpisodeRunner {

    private EpisodeRunner() { /* static utility */ }

    public static double runEpisodes(String task, PolicyFn policy, long baseSeed, int nEpisodes) {
        UnifiedFintechEnv env = new UnifiedFintechEnv();
        List<Double> episodeMeans = new ArrayList<>(nEpisodes);

        for (int ep = 0; ep < nEpisodes; ep++) {
            long epSeed = baseSeed + ep;
            AEPOObservation obs = env.reset(epSeed, task);
            List<Double> stepRewards = new ArrayList<>(env.maxSteps());

            boolean done = false;
            while (!done && stepRewards.size() < env.maxSteps()) {
                AEPOAction action = policy.apply(obs.normalized());
                StepResult sr = env.step(action);
                obs = sr.observation();
                stepRewards.add(sr.reward().value());
                done = sr.done();
            }

            // Pad to full episode length per CLAUDE.md "mean over 100 steps" rule.
            double sum = 0.0;
            for (double r : stepRewards) sum += r;
            episodeMeans.add(sum / env.maxSteps());
        }

        double agg = 0.0;
        for (double m : episodeMeans) agg += m;
        return Math.round((agg / episodeMeans.size()) * 10_000.0) / 10_000.0;
    }
}
