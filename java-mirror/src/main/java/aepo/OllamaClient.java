package aepo;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * OllamaClient — Java-only LLM integration for AEPO debug mode.
 * ================================================================
 * [STUB]: This class has no Python equivalent in unified_gateway.py.
 * It is Java-only simulation code for driving the UnifiedFintechEnv
 * with a locally-running Ollama model instead of a trained Q-table.
 *
 * Flow:
 *   1. Call obs.normalized() to get agent-facing [0,1] values.
 *   2. Serialize to JSON and embed in an instructional prompt.
 *   3. POST to Ollama /v1/chat/completions (OpenAI-compatible API).
 *   4. Parse the 6-integer JSON action from the LLM response.
 *   5. Construct a validated AEPOAction and call env.step().
 *
 * Prerequisites:
 *   - Ollama running locally: https://ollama.com/download
 *   - A model pulled: `ollama pull llama3.2`
 *   - Start server: `ollama serve`  (usually auto-starts)
 *
 * Usage:
 *   OllamaClient.runDebugEpisode("hard");
 *   // or: java -cp target/aepo-mirror.jar aepo.OllamaClient hard
 */
public class OllamaClient {

    // [STUB]: Ollama runs locally; change MODEL to any pulled model name.
    private static final String OLLAMA_URL  = "http://localhost:11434/v1/chat/completions";
    private static final String MODEL       = "llama3.2";
    private static final int    TIMEOUT_SEC = 30;

    private final HttpClient httpClient;

    public OllamaClient() {
        // [STUB]: java.net.http.HttpClient (Java 11+) — no external dependency needed.
        // PYTHON EQUIVALENT: httpx.Client() or requests.Session()
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    // =====================================================================
    // Public API
    // =====================================================================

    /**
     * Ask Ollama to decide the next action given the current observation.
     *
     * [MIRROR]: Mirrors the agent decision loop in inference.py.
     * The Python inference.py calls a trained policy; this calls Ollama instead.
     *
     * @param obs Current environment observation (raw values — will be normalized internally)
     * @return Validated AEPOAction parsed from LLM response; falls back to heuristic on error
     */
    public AEPOAction decideAction(AEPOObservation obs) {
        // [MIRROR]: obs.normalized() mirrors AEPOObservation.normalized() in Python
        Map<String, Double> normalized = obs.normalized();

        String obsJson    = buildObsJson(normalized);
        String prompt     = buildActionPrompt(obsJson);
        String reqBody    = buildOllamaRequest(prompt);

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(OLLAMA_URL))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(reqBody))
                    .timeout(Duration.ofSeconds(TIMEOUT_SEC))
                    .build();

            HttpResponse<String> response =
                    httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                System.err.printf("[OllamaClient] HTTP %d — falling back to heuristic%n",
                        response.statusCode());
                return heuristicFallback(obs);
            }

            return parseActionFromResponse(response.body());

        } catch (Exception e) {
            System.err.printf("[OllamaClient] Request failed (%s) — falling back to heuristic%n",
                    e.getMessage());
            return heuristicFallback(obs);
        }
    }

    /**
     * Run a complete debug episode with Ollama as the decision policy.
     *
     * [STUB]: Java-only debug harness. The Python equivalent is inference.py
     * but that uses a trained Q-table loaded from disk, not an LLM.
     *
     * Prints [START] / [STEP] / [END] in the same format as inference.py
     * so output can be diffed for parity checks.
     *
     * @param taskName One of "easy", "medium", "hard"
     */
    public static void runDebugEpisode(String taskName) {
        UnifiedFintechEnv env    = new UnifiedFintechEnv();
        OllamaClient      client = new OllamaClient();

        // [MIRROR]: env.reset(seed, options={"task": taskName}) in Python
        var resetResult  = env.reset(42L, taskName);
        AEPOObservation obs = resetResult.getKey();

        System.out.printf("[START] task=%s seed=42%n", taskName);

        int    step        = 0;
        boolean done       = false;
        double totalReward = 0.0;

        while (!done && step < 100) {
            // Ask Ollama for the action
            AEPOAction action = client.decideAction(obs);

            // [MIRROR]: env.step(action) → (obs, reward, done, info) — Python 4-tuple
            UnifiedFintechEnv.StepResult sr = env.step(action);

            obs         = sr.observation();
            totalReward += sr.reward();
            done        = sr.done();
            step++;

            // Print per-step summary in inference.py format
            System.out.printf(
                "[STEP %3d] phase=%-8s reward=%.4f done=%-5b blind_spot=%s " +
                "kafka_lag=%.0f p99=%.0f%n",
                step,
                sr.info().get("phase"),
                sr.reward(),
                done,
                sr.info().get("blind_spot_triggered"),
                obs.kafkaLag(),
                obs.rollingP99()
            );
        }

        double meanReward = totalReward / Math.max(1, step);
        System.out.printf("[END] steps=%d total_reward=%.4f mean_reward=%.4f%n",
                step, totalReward, meanReward);
    }

    // =====================================================================
    // Request / response helpers
    // =====================================================================

    /**
     * Serialize normalized observation map to a compact JSON object string.
     *
     * [STUB]: Uses manual string building. In production, use Jackson:
     *   new ObjectMapper().writeValueAsString(normalized)
     */
    private String buildObsJson(Map<String, Double> obs) {
        StringBuilder sb = new StringBuilder("{");
        obs.forEach((k, v) ->
            sb.append('"').append(k).append("\": ").append(String.format("%.4f", v)).append(", ")
        );
        if (sb.length() > 1) sb.setLength(sb.length() - 2);   // strip trailing ", "
        sb.append('}');
        return sb.toString();
    }

    /**
     * Build a zero-shot prompt that instructs the LLM to output a 6-integer JSON action.
     *
     * [STUB]: Prompt engineering for Ollama. The Python policy uses a trained Q-table
     * loaded via numpy; this replaces it with an LLM prompt for debugging purposes.
     */
    private String buildActionPrompt(String obsJson) {
        return "You are an autonomous payment orchestrator agent managing risk, " +
               "infrastructure, and settlement for a UPI payment gateway.\n\n" +
               "Current normalized observation (all values in [0.0, 1.0]):\n" +
               obsJson + "\n\n" +
               "Key thresholds:\n" +
               "  risk_score > 0.8 → HIGH RISK transaction\n" +
               "  kafka_lag  > 0.4 → approaching crash threshold\n" +
               "  rolling_p99 > 0.16 → SLA breach imminent\n\n" +
               "Respond with ONLY a JSON object containing exactly these 6 integer fields:\n" +
               "{\"risk_decision\": X, \"crypto_verify\": X, \"infra_routing\": X, " +
               "\"db_retry_policy\": X, \"settlement_policy\": X, \"app_priority\": X}\n\n" +
               "Field values:\n" +
               "  risk_decision:    0=Approve  1=Reject  2=Challenge\n" +
               "  crypto_verify:    0=FullVerify  1=SkipVerify\n" +
               "  infra_routing:    0=Normal  1=Throttle  2=CircuitBreaker\n" +
               "  db_retry_policy:  0=FailFast  1=ExponentialBackoff\n" +
               "  settlement_policy:0=StandardSync  1=DeferredAsyncFallback\n" +
               "  app_priority:     0=UPI  1=Credit  2=Balanced\n\n" +
               "Return ONLY the JSON. No explanation. No markdown.";
    }

    /**
     * Build the Ollama /v1/chat/completions request body (OpenAI-compatible format).
     *
     * [STUB]: Manually constructed JSON. In production, use Jackson:
     *   Map<String,Object> body = Map.of("model", MODEL, "messages", List.of(...));
     *   new ObjectMapper().writeValueAsString(body)
     */
    private String buildOllamaRequest(String userPrompt) {
        return String.format(
            "{\"model\": \"%s\", \"messages\": [{\"role\": \"user\", \"content\": %s}], " +
            "\"stream\": false, \"temperature\": 0.0}",
            MODEL,
            escapeJsonString(userPrompt)
        );
    }

    /**
     * Extract the 6-integer AEPOAction from Ollama's raw response body.
     *
     * The response follows the OpenAI chat completions schema:
     *   {"choices": [{"message": {"content": "{\"risk_decision\": 1, ...}"}}]}
     *
     * [STUB]: Uses basic substring parsing. Replace with Jackson in production:
     *   JsonNode root = mapper.readTree(responseBody);
     *   String content = root.at("/choices/0/message/content").asText();
     *   JsonNode action = mapper.readTree(content);
     */
    private AEPOAction parseActionFromResponse(String responseBody) {
        try {
            // Find the content field in the Ollama response
            int contentIdx = responseBody.indexOf("\"content\":");
            if (contentIdx == -1) {
                System.err.println("[OllamaClient] No 'content' field in response");
                return heuristicFallback(null);
            }

            // Locate the action JSON object within the content string
            // Content may be wrapped as "content": "{...}" (escaped) or "content": {...}
            int braceOpen = responseBody.indexOf('{', contentIdx + 10);
            if (braceOpen == -1) {
                System.err.println("[OllamaClient] No JSON object in content");
                return heuristicFallback(null);
            }

            // Find matching close brace (simple single-level scan — LLM output is flat)
            int braceClose = responseBody.indexOf('}', braceOpen) + 1;
            if (braceClose == 0) {
                System.err.println("[OllamaClient] Unclosed JSON in content");
                return heuristicFallback(null);
            }

            String actionJson = responseBody.substring(braceOpen, braceClose);

            int riskDecision     = extractInt(actionJson, "risk_decision");
            int cryptoVerify     = extractInt(actionJson, "crypto_verify");
            int infraRouting     = extractInt(actionJson, "infra_routing");
            int dbRetryPolicy    = extractInt(actionJson, "db_retry_policy");
            int settlementPolicy = extractInt(actionJson, "settlement_policy");
            int appPriority      = extractInt(actionJson, "app_priority");

            // AEPOAction constructor validates ranges — will throw on bad LLM output
            return new AEPOAction(riskDecision, cryptoVerify, infraRouting,
                                  dbRetryPolicy, settlementPolicy, appPriority);

        } catch (Exception e) {
            System.err.printf("[OllamaClient] Action parse failed (%s) — fallback%n", e.getMessage());
            return heuristicFallback(null);
        }
    }

    /**
     * Extract a named integer value from a flat JSON string using substring search.
     * Handles both {@code "key": N} and {@code "key":N} formatting.
     */
    private int extractInt(String json, String key) {
        // Try "key": N  and  "key":N
        String[] patterns = {"\"" + key + "\": ", "\"" + key + "\":"};
        for (String pattern : patterns) {
            int idx = json.indexOf(pattern);
            if (idx == -1) continue;
            int valStart = idx + pattern.length();
            while (valStart < json.length() && json.charAt(valStart) == ' ') valStart++;
            int valEnd = valStart;
            while (valEnd < json.length() && Character.isDigit(json.charAt(valEnd))) valEnd++;
            if (valEnd > valStart) {
                return Integer.parseInt(json.substring(valStart, valEnd));
            }
        }
        throw new IllegalArgumentException("Key not found in action JSON: " + key);
    }

    /** Wrap a string in JSON double-quotes with internal quotes and newlines escaped. */
    private String escapeJsonString(String s) {
        return "\"" + s.replace("\\", "\\\\")
                       .replace("\"", "\\\"")
                       .replace("\n", "\\n")
                       .replace("\r", "\\r")
               + "\"";
    }

    // =====================================================================
    // Fallback policy
    // =====================================================================

    /**
     * Safe fallback action when Ollama is unavailable or returns unparseable output.
     *
     * [STUB]: Mirrors the intentionally-incomplete HeuristicAgent defaults.
     * Uses obs-aware risk check if obs is non-null; otherwise uses safe conservative defaults.
     *
     * PYTHON EQUIVALENT: heuristic_policy() in graders.py
     */
    private AEPOAction heuristicFallback(AEPOObservation obs) {
        if (obs == null) {
            // Ultra-safe conservative default: Reject + FullVerify + Normal + Backoff + Sync + Balanced
            return new AEPOAction(1, 0, 0, 1, 0, 2);
        }

        // [MIRROR]: Mirrors HeuristicAgent.decide() logic (without blind spot awareness)
        Map<String, Double> norm = obs.normalized();
        double riskNorm  = norm.get("risk_score");
        double lagNorm   = norm.get("kafka_lag");
        double p99Norm   = norm.get("rolling_p99");

        int riskDecision  = riskNorm > 0.8 ? 1 : 0;      // Reject if high-risk, else Approve
        int cryptoVerify  = riskNorm > 0.8 ? 0 : 1;      // FullVerify on high-risk (blind spot #1)
        int infraRouting  = lagNorm  > 0.6 ? 1 : 0;      // Throttle if lag high
        int dbRetryPolicy = 1;                             // Always ExponentialBackoff (blind spot #3)
        int settlementPolicy = p99Norm > 0.6 ? 1 : 0;    // DeferredAsync if P99 high
        int appPriority   = 2;                             // Always Balanced (blind spot #2)

        return new AEPOAction(riskDecision, cryptoVerify, infraRouting,
                              dbRetryPolicy, settlementPolicy, appPriority);
    }

    // =====================================================================
    // Entry point for standalone debug runs
    // =====================================================================

    /**
     * Main entry point — runs a debug episode with Ollama as the policy.
     *
     * Usage: java -cp target/aepo-mirror.jar aepo.OllamaClient [easy|medium|hard]
     *
     * [STUB]: Java-only debug runner. Python equivalent: python inference.py
     */
    public static void main(String[] args) {
        String task = (args.length > 0) ? args[0] : "easy";
        System.out.printf("[OllamaClient] Starting debug episode: task=%s model=%s url=%s%n",
                task, MODEL, OLLAMA_URL);
        runDebugEpisode(task);
    }
}
