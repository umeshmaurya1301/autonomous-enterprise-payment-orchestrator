package aepo.server;

import aepo.env.UnifiedFintechEnv;
import org.springframework.stereotype.Component;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Singleton wrapper around {@link UnifiedFintechEnv} — Java mirror of the
 * module-level {@code env} + {@code _env_lock} + {@code _episode_active}
 * triple in {@code server/app.py}.
 *
 * <p>Why a Spring {@code @Component} (not a static field)? Lets the controller
 * test inject a fresh session per test class via {@code @MockBean}. The Python
 * file uses module-level state because Python lacks first-class DI — Spring's
 * idiomatic approach is the bean.
 *
 * <p>Threading: the embedded Tomcat thread pool fans /step calls out across
 * worker threads. Without a lock, two concurrent /step requests could
 * interleave on the env's mutable accumulators and produce inconsistent state.
 * {@link ReentrantLock} mirrors the {@code asyncio.Lock} used in Python.
 */
@Component
public class EnvSession {

    private final UnifiedFintechEnv env = new UnifiedFintechEnv();
    private final Lock lock = new ReentrantLock();

    /** True once the client has called POST /reset. /step + /state 400 until then. */
    private volatile boolean episodeActive = false;

    public EnvSession() {
        // Prime the env so /state has a valid observation immediately on startup
        // — same behaviour as the Python server's bootstrap reset(). episodeActive
        // stays false until the client explicitly resets.
        env.reset(null, "easy");
    }

    public UnifiedFintechEnv env() {
        return env;
    }

    public Lock lock() {
        return lock;
    }

    public boolean episodeActive() {
        return episodeActive;
    }

    public void markEpisodeActive() {
        episodeActive = true;
    }
}
