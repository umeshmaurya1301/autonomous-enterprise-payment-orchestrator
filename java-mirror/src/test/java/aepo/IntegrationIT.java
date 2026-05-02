package aepo;

import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration test — boots the entire Spring Boot application on a
 * real random TCP port and exercises the REST surface over HTTP.
 *
 * <p>Distinct from {@code ControllerTest}: that test uses MockMvc (no socket,
 * no Tomcat). This one uses {@link TestRestTemplate} hitting an actual
 * embedded Tomcat instance, so it catches:
 * <ul>
 *   <li>Jackson snake_case wire format (would be missed by direct controller calls)</li>
 *   <li>Embedded servlet container error mapping (4xx/5xx from {@code ResponseStatusException})</li>
 *   <li>Concurrency under real worker-thread fan-out — exercises {@code EnvSession.lock}</li>
 *   <li>Full episode lifecycle: reset → 100 steps → done=true → state still queryable</li>
 * </ul>
 *
 * <p>Test ordering matters: episode lifecycle tests assume an active session
 * established by an earlier reset call. {@link Order} pins this contract so a
 * failing earlier test doesn't cascade-corrupt later ones.
 *
 * <p>Suffix {@code IT} (Integration Test) is conventional — separates these
 * from fast unit tests if a future Gradle config wires Failsafe-style splitting.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class IntegrationIT {

    @Autowired
    private TestRestTemplate http;

    // ──────────────────────────────────────────────────────────────────
    //  Health probes — must work without any prior reset.
    // ──────────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    void rootHealthIsReachableOverRealHttp() {
        ResponseEntity<Map> resp = http.getForEntity("/", Map.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals("healthy", resp.getBody().get("status"));
    }

    @Test
    @Order(2)
    void contractAdvertisesFourTuple() {
        ResponseEntity<Map> resp = http.getForEntity("/contract", Map.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        // Jackson snake_case in action: the property is "step_tuple", not "stepTuple".
        assertEquals("4-tuple", resp.getBody().get("step_tuple"));
        assertEquals(true, resp.getBody().get("openenv_compliant"));
    }

    @Test
    @Order(3)
    void getResetPreflightReturns200() {
        ResponseEntity<Map> resp = http.getForEntity("/reset", Map.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
    }

    // ──────────────────────────────────────────────────────────────────
    //  Episode lifecycle — full reset → step* → done over real HTTP.
    // ──────────────────────────────────────────────────────────────────

    @Test
    @Order(10)
    void fullEpisodeRunsToTerminationOverHttp() {
        // ① Start a fresh easy episode.
        ResponseEntity<Map> reset = http.postForEntity(
                "/reset", Map.of("task", "easy", "seed", 42), Map.class);
        assertEquals(HttpStatus.OK, reset.getStatusCode());
        assertNotNull(reset.getBody().get("observation"));
        assertEquals("easy", ((Map<?, ?>) reset.getBody().get("info")).get("task"));

        // ② Step until done — safe action mix (Approve + FullVerify + Normal).
        Map<String, Object> action = Map.of(
                "risk_decision", 0,
                "crypto_verify", 0,
                "infra_routing", 0,
                "db_retry_policy", 0,
                "settlement_policy", 0,
                "app_priority", 2
        );
        Map<String, Object> body = Map.of("action", action);

        boolean done = false;
        int steps = 0;
        double cumulative = 0.0;
        while (!done && steps < 200) {
            ResponseEntity<Map> resp = http.postForEntity("/step", body, Map.class);
            assertEquals(HttpStatus.OK, resp.getStatusCode(),
                    "step " + steps + " failed: " + resp.getBody());

            // Wire-format check on each tick — reward is a Number, breakdown a Map.
            Number reward = (Number) resp.getBody().get("reward");
            assertTrue(reward.doubleValue() >= 0.0 && reward.doubleValue() <= 1.0);
            assertNotNull(resp.getBody().get("reward_breakdown"));
            assertNotNull(resp.getBody().get("info"));

            cumulative += reward.doubleValue();
            done = (Boolean) resp.getBody().get("done");
            steps++;
        }

        // ③ Episode terminated within the 100-step contract.
        assertTrue(done, "episode must terminate by 100 steps");
        assertTrue(steps <= 100, "natural termination cap is 100, got " + steps);
        assertTrue(cumulative > 0.0, "easy task should accumulate positive reward");

        // ④ /state still works after done=true (no auto-clearing of episodeActive).
        ResponseEntity<Map> state = http.getForEntity("/state", Map.class);
        assertEquals(HttpStatus.OK, state.getStatusCode());
        assertNotNull(state.getBody().get("observation"));
    }

    @Test
    @Order(11)
    void invalidTaskReturns422OverHttp() {
        ResponseEntity<Map> resp = http.postForEntity(
                "/reset", Map.of("task", "nightmare"), Map.class);
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, resp.getStatusCode());
    }

    @Test
    @Order(12)
    void invalidActionReturns422OverHttp() {
        // Reset first to ensure episodeActive=true; then send risk_decision=9.
        http.postForEntity("/reset", Map.of("task", "easy"), Map.class);

        Map<String, Object> badAction = Map.of(
                "risk_decision", 9,         // out of [0, 2] → IllegalArgumentException → 422
                "crypto_verify", 0,
                "infra_routing", 0,
                "db_retry_policy", 0,
                "settlement_policy", 0,
                "app_priority", 0
        );
        ResponseEntity<Map> resp = http.postForEntity(
                "/step", Map.of("action", badAction), Map.class);
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, resp.getStatusCode());
    }

    // ──────────────────────────────────────────────────────────────────
    //  Concurrency — real Tomcat fans /step across threads.
    // ──────────────────────────────────────────────────────────────────

    @Test
    @Order(20)
    void concurrentStepsDoNotCorruptEnvState() throws Exception {
        // Reset, then fire N parallel /step requests. The ReentrantLock in
        // EnvSession must serialise them — proven by the absence of:
        //   * 5xx responses (env throwing on inconsistent state)
        //   * step_in_episode counter gaps or repeats
        // Each successful response carries an integer step_in_episode; the union
        // of those values must be a contiguous prefix of {1, 2, 3, ...}.
        http.postForEntity("/reset", Map.of("task", "easy", "seed", 7), Map.class);

        Map<String, Object> action = Map.of(
                "risk_decision", 0, "crypto_verify", 0, "infra_routing", 0,
                "db_retry_policy", 0, "settlement_policy", 0, "app_priority", 2);
        Map<String, Object> body = Map.of("action", action);

        int n = 20;
        ExecutorService pool = Executors.newFixedThreadPool(8);
        List<Future<Map>> futures = new ArrayList<>(n);
        try {
            for (int i = 0; i < n; i++) {
                futures.add(pool.submit(() -> {
                    ResponseEntity<Map> r = http.postForEntity("/step", body, Map.class);
                    assertEquals(HttpStatus.OK, r.getStatusCode());
                    return r.getBody();
                }));
            }

            // Collect step_in_episode values; lock should give us {1..n} once.
            // step_in_episode is nested inside the info sub-map, NOT at top level.
            boolean[] seen = new boolean[n + 1];
            for (Future<Map> f : futures) {
                Map result = f.get(5, TimeUnit.SECONDS);
                Map<?, ?> info = (Map<?, ?>) result.get("info");
                assertNotNull(info, "step response missing 'info'");
                int step = ((Number) info.get("step_in_episode")).intValue();
                assertTrue(step >= 1 && step <= n,
                        "step_in_episode " + step + " outside expected [1," + n + "]");
                assertFalse(seen[step], "duplicate step_in_episode " + step
                        + " — env state was corrupted by concurrent access");
                seen[step] = true;
            }
            for (int i = 1; i <= n; i++) {
                assertTrue(seen[i], "missing step_in_episode " + i);
            }
        } finally {
            pool.shutdownNow();
        }
    }
}
