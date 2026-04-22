/**
 * <h1>AEPO Java Mirror — {@code aepo} package</h1>
 *
 * <p>
 * This package is a <strong>readable Java translation</strong> of the Python
 * Autonomous Enterprise Payment Orchestrator (AEPO) codebase.
 * It does <em>not</em> compile or run independently — it exists so that a
 * Java / Spring Boot engineer can understand the Python system without reading
 * Python. Every class carries a {@code // PYTHON EQUIVALENT:} comment block
 * that points to the corresponding Python file and construct.
 * </p>
 *
 * <hr>
 *
 * <h2>Package Structure</h2>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Java mirror files and their Python counterparts</caption>
 *   <tr>
 *     <th>Java class</th>
 *     <th>Python equivalent</th>
 *     <th>Purpose</th>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.AEPOObservation}</td>
 *     <td>{@code unified_gateway.py → AEPOObservation}</td>
 *     <td>10-field typed observation; constructor validates ranges; {@code normalized()} returns agent-facing [0,1] dict</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.AEPOAction}</td>
 *     <td>{@code unified_gateway.py → AEPOAction}</td>
 *     <td>6-field typed action; constructor validates discrete ranges</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.UnifiedFintechEnv}</td>
 *     <td>{@code unified_gateway.py → UnifiedFintechEnv}</td>
 *     <td>Core RL environment — phase machine, all 8 causal transitions, curriculum, adversary escalation</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.RewardCalculator}</td>
 *     <td>{@code unified_gateway.py → step() reward block ③}</td>
 *     <td>Stateless reward calculator extracted for readability; not a separate class in Python</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.HeuristicAgent}</td>
 *     <td>{@code graders.py → heuristic_policy()}</td>
 *     <td>Intentionally-incomplete baseline with 3 deliberate blind spots the trained agent must discover</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.DynamicsModel}</td>
 *     <td>{@code dynamics_model.py → LagPredictor}</td>
 *     <td>2-layer MLP pseudocode (16→64→1); documents architecture and input encoding; weights uninitialized</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.Graders}</td>
 *     <td>{@code graders.py → EasyGrader, MediumGrader, HardGrader, get_grader()}</td>
 *     <td>Per-task programmatic graders with fixed seeds; {@code runEpisodes()} stubs out (cannot call Python env from Java)</td>
 *   </tr>
 *   <tr>
 *     <td>{@link aepo.TrainQTable}</td>
 *     <td>{@code train.py → train_q_table()}</td>
 *     <td>Tabular Q-learning loop — state discretisation, ε-greedy selection, Bellman update, blind-spot logging</td>
 *   </tr>
 * </table>
 *
 * <hr>
 *
 * <h2>Observation Space (10 fields)</h2>
 *
 * <p>All 10 fields are stored as raw values in {@link aepo.AEPOObservation}.
 * The agent always receives the normalized form via {@code .normalized()}.
 * Raw values are only exposed in {@code info["raw_obs"]} on the server side.</p>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Observation fields, ranges, and normalization</caption>
 *   <tr>
 *     <th>Layer</th><th>Field (Java)</th><th>Field (Python key)</th><th>Raw range</th><th>Normalization</th>
 *   </tr>
 *   <tr><td>Risk</td>        <td>channel</td>              <td>transaction_type</td>       <td>{0,1,2}</td>      <td>÷ 2</td></tr>
 *   <tr><td>Risk</td>        <td>riskScore</td>            <td>risk_score</td>             <td>[0, 100]</td>     <td>÷ 100</td></tr>
 *   <tr><td>Risk</td>        <td>adversaryThreatLevel</td> <td>adversary_threat_level</td> <td>[0, 10]</td>      <td>÷ 10</td></tr>
 *   <tr><td>Risk</td>        <td>systemEntropy</td>        <td>system_entropy</td>         <td>[0, 100]</td>     <td>÷ 100</td></tr>
 *   <tr><td>Infra</td>       <td>kafkaLag</td>             <td>kafka_lag</td>              <td>[0, 10000]</td>   <td>÷ 10000</td></tr>
 *   <tr><td>Infra</td>       <td>apiLatency</td>           <td>api_latency</td>            <td>[0, 5000]</td>    <td>÷ 5000</td></tr>
 *   <tr><td>Infra</td>       <td>rollingP99</td>           <td>rolling_p99</td>            <td>[0, 5000]</td>    <td>÷ 5000</td></tr>
 *   <tr><td>Infra</td>       <td>dbConnectionPool</td>     <td>db_connection_pool</td>     <td>[0, 100]</td>     <td>÷ 100</td></tr>
 *   <tr><td>Business</td>    <td>bankApiStatus</td>        <td>bank_api_status</td>        <td>{0,1,2}</td>      <td>÷ 2 → 0.0/0.5/1.0</td></tr>
 *   <tr><td>Business</td>    <td>merchantTier</td>         <td>merchant_tier</td>          <td>{0,1}</td>        <td>÷ 1 → 0.0/1.0</td></tr>
 * </table>
 *
 * <hr>
 *
 * <h2>Action Space (6 decisions)</h2>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Action fields, valid values, and failure conditions</caption>
 *   <tr>
 *     <th>Layer</th><th>Field (Java)</th><th>Values</th><th>Failure condition</th>
 *   </tr>
 *   <tr>
 *     <td>Risk</td>
 *     <td>riskDecision</td>
 *     <td>0=Approve, 1=Reject, 2=Challenge</td>
 *     <td>Approve + SkipVerify + risk_score &gt; 80 → fraud catastrophe, reward=0, done=true</td>
 *   </tr>
 *   <tr>
 *     <td>Risk</td>
 *     <td>cryptoVerify</td>
 *     <td>0=FullVerify, 1=SkipVerify</td>
 *     <td>See above; FullVerify adds +150 kafka_lag/step</td>
 *   </tr>
 *   <tr>
 *     <td>Infra</td>
 *     <td>infraRouting</td>
 *     <td>0=Normal, 1=Throttle, 2=CircuitBreaker</td>
 *     <td>CircuitBreaker → −0.50/step; Throttle in Normal phase → −0.20/step</td>
 *   </tr>
 *   <tr>
 *     <td>Infra</td>
 *     <td>dbRetryPolicy</td>
 *     <td>0=FailFast, 1=ExponentialBackoff</td>
 *     <td>ExponentialBackoff when db_pool &lt; 20 → −0.10/step</td>
 *   </tr>
 *   <tr>
 *     <td>Business</td>
 *     <td>settlementPolicy</td>
 *     <td>0=StandardSync, 1=DeferredAsyncFallback</td>
 *     <td>DeferredAsync in Normal phase → −0.15; 5+ consecutive → −0.20</td>
 *   </tr>
 *   <tr>
 *     <td>Business</td>
 *     <td>appPriority</td>
 *     <td>0=UPI, 1=Credit, 2=Balanced</td>
 *     <td>Mismatch with merchant_tier → misses +0.02/step bonus</td>
 *   </tr>
 * </table>
 *
 * <hr>
 *
 * <h2>The 8 Causal State Transitions</h2>
 *
 * <p>
 * These transitions are what separate AEPO from a memoryless simulator.
 * Each is implemented in {@link aepo.UnifiedFintechEnv#step(AEPOAction)}.
 * </p>
 *
 * <ol>
 *   <li><strong>Lag → Latency:</strong> {@code api_latency[t+1] += 0.1 × max(0, kafka_lag[t] − 3000)}</li>
 *   <li><strong>Throttle relief:</strong> Throttle action queues −150 to kafka_lag for each of the next 2 steps.
 *       The queue is stored in {@code throttleReliefQueue} (a {@link java.util.Deque}).
 *       <em>BOUNDARY RULE:</em> {@code throttleReliefQueue.clear()} is called inside {@code reset()}
 *       to prevent lag relief bleeding into the next episode.</li>
 *   <li><strong>Bank coupling:</strong> {@code bank_api_status=Degraded} AND {@code StandardSync} → {@code rolling_p99 += 200}</li>
 *   <li><strong>DB pressure:</strong> {@code db_pool > 80} AND {@code ExponentialBackoff} → {@code api_latency += 100}</li>
 *   <li><strong>DB waste:</strong> {@code db_pool < 20} AND {@code ExponentialBackoff} → −0.10 reward penalty</li>
 *   <li><strong>Entropy spike:</strong> {@code system_entropy > 70} → {@code api_latency += uniform(100, 300)}</li>
 *   <li><strong>Adversary escalation (5-episode lag):</strong>
 *       rolling 5-ep avg &gt; 0.6 → {@code adversary_threat_level += 0.5} (max 10);
 *       rolling 5-ep avg &lt; 0.3 → {@code adversary_threat_level −= 0.5} (min 0)</li>
 *   <li><strong>P99 EMA:</strong> {@code rolling_p99[t] = 0.8 × rolling_p99[t−1] + 0.2 × api_latency[t]}</li>
 * </ol>
 *
 * <hr>
 *
 * <h2>Episode and Phase Structure</h2>
 *
 * <ul>
 *   <li>Every episode is exactly <strong>100 steps</strong>.</li>
 *   <li>Early termination when {@code kafka_lag > 4000} (crash) or fraud catastrophe; remaining steps score 0.0.</li>
 *   <li>Episode score = mean of all 100 step rewards (crashed episodes padded with 0.0).</li>
 * </ul>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Phase sequences per task — fixed at reset, never mixed by curriculum</caption>
 *   <tr><th>Task</th><th>Phase sequence</th></tr>
 *   <tr><td>easy</td>   <td>Normal × 100</td></tr>
 *   <tr><td>medium</td> <td>Normal × 40 → Spike × 60</td></tr>
 *   <tr><td>hard</td>   <td>Normal × 20 → Spike × 20 → Attack × 40 → Recovery × 20</td></tr>
 * </table>
 *
 * <hr>
 *
 * <h2>Reward Function Summary</h2>
 *
 * <p>
 * {@code final = clamp(0.8 + bonuses − penalties, 0.0, 1.0)}
 * </p>
 *
 * <p>See {@link aepo.RewardCalculator} for the full implementation.
 * Key rules:</p>
 *
 * <ul>
 *   <li>Approve + SkipVerify + risk_score &gt; 80 → reward = 0.0, episode ends (fraud)</li>
 *   <li>kafka_lag &gt; 4000 → reward = 0.0, episode ends (crash)</li>
 *   <li>rolling_p99 &gt; 800 → −0.30 (SLA breach)</li>
 *   <li>CircuitBreaker → −0.50</li>
 *   <li>Challenge on high-risk → +0.05</li>
 *   <li><strong>Blind spot #1:</strong> Reject + SkipVerify on high-risk → +0.04 (heuristic never finds this)</li>
 *   <li>app_priority matches merchant_tier → +0.02</li>
 * </ul>
 *
 * <hr>
 *
 * <h2>Adaptive Curriculum</h2>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Curriculum advancement conditions</caption>
 *   <tr><th>Transition</th><th>Condition</th><th>Threshold</th></tr>
 *   <tr><td>easy → medium</td>  <td>5 consecutive episodes with mean reward</td> <td>&ge; 0.75</td></tr>
 *   <tr><td>medium → hard</td>  <td>5 consecutive episodes with mean reward</td> <td>&ge; 0.45</td></tr>
 * </table>
 *
 * <p>Curriculum level <strong>never regresses</strong>.
 * Adversary threat level responds to agent performance with a 5-episode lag
 * (Causal Transition #7 above), producing the staircase reward curve that is
 * the central pitch narrative.</p>
 *
 * <hr>
 *
 * <h2>Heuristic Agent Blind Spots</h2>
 *
 * <p>See {@link aepo.HeuristicAgent} for the full policy and
 * {@link aepo.HeuristicAgent#blindSpotSummary()} for a human-readable explanation.
 * Three deliberate blind spots the trained agent must discover:</p>
 *
 * <ol>
 *   <li><strong>#1 (primary learning story):</strong>
 *       Reject + SkipVerify on high-risk → +0.04 bonus AND saves 250 lag/step.
 *       Heuristic uses FullVerify (+0.03, +150 lag/step instead).</li>
 *   <li><strong>#2:</strong> app_priority should match merchant_tier → +0.02/step.
 *       Heuristic always uses Balanced.</li>
 *   <li><strong>#3:</strong> ExponentialBackoff when db_pool &lt; 20 → −0.10/step.
 *       Heuristic always uses ExponentialBackoff regardless of pool level.</li>
 * </ol>
 *
 * <hr>
 *
 * <h2>Task Grader Thresholds</h2>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Passing thresholds and seeds</caption>
 *   <tr><th>Task</th><th>Success threshold</th><th>Fixed seed</th><th>Episodes</th></tr>
 *   <tr><td>easy</td>   <td>&ge; 0.75 mean reward</td> <td>42</td> <td>10</td></tr>
 *   <tr><td>medium</td> <td>&ge; 0.45 mean reward</td> <td>43</td> <td>10</td></tr>
 *   <tr><td>hard</td>   <td>&ge; 0.30 mean reward</td> <td>44</td> <td>10</td></tr>
 * </table>
 *
 * <hr>
 *
 * <h2>Type System Mapping: Python → Java</h2>
 *
 * <table border="1" cellpadding="6" cellspacing="0">
 *   <caption>Python construct to Java equivalent</caption>
 *   <tr><th>Python</th><th>Java equivalent used in this package</th></tr>
 *   <tr><td>Pydantic {@code BaseModel}</td>                <td>{@code record} with compact constructor validation</td></tr>
 *   <tr><td>{@code gymnasium.Env}</td>                     <td>Plain class with {@code reset()}, {@code step()}, {@code state()}</td></tr>
 *   <tr><td>{@code numpy.ndarray}</td>                     <td>{@code double[]} or {@code List<Double>}</td></tr>
 *   <tr><td>{@code rng.uniform(a, b)}</td>                 <td>{@code lo + rng.nextDouble() * (hi - lo)}</td></tr>
 *   <tr><td>{@code np.clip(val, lo, hi)}</td>              <td>{@code Math.min(hi, Math.max(lo, val))}</td></tr>
 *   <tr><td>{@code deque(maxlen=N)}</td>                   <td>{@code ArrayDeque} with manual size enforcement</td></tr>
 *   <tr><td>{@code defaultdict(lambda: np.zeros(N))}</td>  <td>{@code HashMap.computeIfAbsent(key, k -> new float[N])}</td></tr>
 *   <tr><td>{@code dict} return</td>                       <td>Typed {@code record} (e.g., {@code StepResult}) or {@code Map<String,Object>}</td></tr>
 *   <tr><td>{@code Optional[X]}</td>                       <td>{@code Optional<X>} or nullable parameter</td></tr>
 *   <tr><td>{@code nn.Module} (PyTorch)</td>               <td>Pseudocode class — weights declared, not trained</td></tr>
 *   <tr><td>{@code @app.post("/step")} FastAPI</td>        <td>{@code @PostMapping("/step")} Spring Boot</td></tr>
 * </table>
 *
 * <hr>
 *
 * <p><strong>NOTE:</strong> Delete the entire {@code /java-mirror/} directory before final submission.
 * It is a development aid, not part of the OpenEnv submission.</p>
 *
 * @see aepo.AEPOObservation
 * @see aepo.AEPOAction
 * @see aepo.UnifiedFintechEnv
 * @see aepo.RewardCalculator
 * @see aepo.HeuristicAgent
 * @see aepo.DynamicsModel
 * @see aepo.Graders
 * @see aepo.TrainQTable
 */
package aepo;
