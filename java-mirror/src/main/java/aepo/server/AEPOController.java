package aepo.server;

import aepo.env.StepResult;
import aepo.env.UnifiedFintechEnv;
import aepo.server.dto.ResetRequest;
import aepo.server.dto.StepRequest;
import aepo.types.AEPOObservation;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.Lock;

/**
 * REST surface — Java mirror of {@code server/app.py}.
 *
 * <p>One-to-one with the FastAPI routes:
 * <pre>
 *   GET  /          → health
 *   GET  /reset     → grader pre-flight ping
 *   POST /reset     → start a new episode
 *   POST /step      → advance one tick
 *   GET  /state     → peek at current observation
 *   GET  /contract  → OpenEnv 4-tuple declaration
 * </pre>
 *
 * <p>All env-mutating routes are wrapped in {@code session.lock()} —
 * the Java mirror of the {@code asyncio.Lock} the Python server uses.
 */
@RestController
public class AEPOController {

    private static final Set<String> VALID_TASKS = Set.of("easy", "medium", "hard");

    private final EnvSession session;

    public AEPOController(EnvSession session) {
        this.session = session;
    }

    // ── Health checks ────────────────────────────────────────────────

    @GetMapping("/")
    public Map<String, String> rootHealth() {
        return Map.of(
                "status", "healthy",
                "message", "AEPO is live. Use POST /reset to initialise a task."
        );
    }

    @GetMapping("/reset")
    public Map<String, String> resetPreflight() {
        // GET /reset must return 200 — many graders ping it before issuing POST.
        return Map.of(
                "status", "healthy",
                "message", "Route /reset is live. Send POST /reset with {\"task\": \"easy|medium|hard\"}."
        );
    }

    @GetMapping("/contract")
    public Map<String, Object> contract() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("step_tuple", "4-tuple");
        body.put("step_format", UnifiedFintechEnv.STEP_TUPLE_FORMAT);
        body.put("openenv_compliant", UnifiedFintechEnv.IS_OPENENV_COMPLIANT);
        body.put("note", "AEPO never truncates — episodes end via crash, fraud, or 100-step limit.");
        return body;
    }

    // ── Episode lifecycle ───────────────────────────────────────────

    @PostMapping("/reset")
    public Map<String, Object> reset(@RequestBody(required = false) ResetRequest request) {
        String task = (request == null || request.task() == null) ? "easy" : request.task();
        if (!VALID_TASKS.contains(task)) {
            // Mirror Python's HTTP 422 + structured detail message.
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY,
                    "Invalid task '" + task + "'. Must be one of: easy, medium, hard.");
        }
        Long seed = (request == null) ? null : request.seed();

        Lock lock = session.lock();
        lock.lock();
        try {
            AEPOObservation obs = session.env().reset(seed, task);
            session.markEpisodeActive();

            Map<String, Object> body = new LinkedHashMap<>();
            body.put("observation", obs);
            body.put("info", Map.of("task", task));
            return body;
        } finally {
            lock.unlock();
        }
    }

    @PostMapping("/step")
    public Map<String, Object> step(@RequestBody StepRequest request) {
        if (!session.episodeActive()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "No active episode. Call POST /reset with a task before stepping.");
        }
        if (request == null || request.action() == null) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY,
                    "Request body must contain an 'action' object.");
        }

        Lock lock = session.lock();
        lock.lock();
        try {
            StepResult sr = session.env().step(request.action());
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("observation", sr.observation());
            body.put("reward", sr.reward().value());
            body.put("reward_breakdown", sr.reward().breakdown());
            body.put("done", sr.done());
            body.put("info", sr.info());
            return body;
        } finally {
            lock.unlock();
        }
    }

    @GetMapping("/state")
    public Map<String, Object> state() {
        if (!session.episodeActive()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "No active episode. Call POST /reset with a task first.");
        }
        Lock lock = session.lock();
        lock.lock();
        try {
            return Map.of("observation", session.env().state());
        } finally {
            lock.unlock();
        }
    }

    // ── Exception → 422 mapping for record-validation failures ──────
    // AEPOAction / AEPOObservation throw IllegalArgumentException from their
    // compact constructors. Without this handler they would surface as 500 —
    // matching the Python/Pydantic 422 contract is much more useful for clients.
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> onBadInput(IllegalArgumentException ex) {
        return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY)
                .body(Map.of("detail", ex.getMessage()));
    }
}
