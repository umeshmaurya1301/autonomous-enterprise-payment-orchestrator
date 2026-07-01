# 13 — Glossary

> **Goal:** A fast lookup for every term in this project. Each entry: a one-line plain-English definition + a Java analogy or AEPO anchor where useful. Grouped by area; skim or Ctrl-F.

## Reinforcement Learning terms

| Term | Definition | Java / AEPO anchor |
|------|-----------|--------------------|
| **Agent** | The decision-maker that observes and acts | the client calling `step()` in a loop; in AEPO: random/heuristic/Q-table/LLM |
| **Environment** | The world the agent acts in; holds state, responds to actions | a stateful `@Service` with a `step()` method; in AEPO: `UnifiedFintechEnv` |
| **State** | The full internal situation of the environment | all private fields of the service |
| **Observation** | The (possibly partial/noisy) slice of state the agent sees | the response DTO; in AEPO: 10 normalized floats |
| **Observation space** | The schema/bounds of observations | the DTO schema; `Box(10,)` |
| **Action** | The decision the agent sends back | the request DTO; 6 ints |
| **Action space** | The set of legal actions | `MultiDiscrete([3,2,3,2,2,3])` = 216 combos |
| **Step** | One agent-environment interaction (action → obs+reward) | one `step()` call |
| **Episode** | A full run from `reset()` to `done` | one client session; AEPO: 100 steps |
| **Reward** | A number scoring how good the last action was | a per-step score; AEPO: [0,1] |
| **Return** | Total (discounted) future reward from a step | sum of future scores |
| **Policy** | The strategy mapping observation → action | a `Function<Obs,Action>`; the thing RL learns |
| **Done** | Boolean: is the episode over? | a "session complete" flag |
| **Info** | Side-channel diagnostics dict (not the agent's input) | response metadata/headers |
| **MDP** | Markov Decision Process — next state depends only on current state + action | a memoryless-given-state machine |
| **POMDP** | Partially Observable MDP — agent sees only a slice of state | AEPO (noise + masking + hidden clock) |
| **Markov property** | The future depends only on the present state, not history | all history compressed into accumulators |
| **Value V(s)** | Expected long-term reward from state `s` | "how good is this state?" |
| **Q-value Q(s,a)** | Expected long-term reward of action `a` in state `s` | a cell in the Q-table |
| **Q-table** | Lookup table of Q-values per (state, action) | `Map<State, double[216]>` |
| **Bellman update** | The rule that refines a Q-value toward `reward + γ·max Q(s')` | an EMA-style nudge of a Q-cell |
| **Discount factor (γ)** | How much future rewards are worth (0–1) | an NPV discount rate; AEPO: 0.95 |
| **Learning rate (α / lr)** | How big each Q-update nudge is | a smoothing step size; AEPO: 0.1 |
| **Exploration** | Trying random actions to discover better ones | a chaos-monkey; finds blind spots |
| **Exploitation** | Taking the current best-known action | production-mode greedy choice |
| **ε-greedy** | With prob ε explore, else exploit | AEPO: ε 1.0→0.05 |
| **Tabular Q-learning** | Q-learning with an explicit Q-table (no neural net) | AEPO's primary agent (`train.py`) |
| **Dyna-Q** | Q-learning + imagined updates from a world model | `DynaPlanner`: 5 imagined updates/step |
| **World model** | A learned predictor of the next state | `LagPredictor`, `MultiObsPredictor` |
| **Curriculum learning** | Train on easier tasks first, then harder | easy→medium→hard schedule |
| **Catastrophic forgetting** | Learning a new task overwrites an old task's knowledge | fixed via per-task Q-tables |
| **Reward shaping** | Designing reward terms to guide learning | AEPO's bonuses/penalties/proximity gradients |
| **Reward hacking** | Agent exploits a reward loophole for cheap score | defeated by "no free actions" |
| **Training** | Learning the policy/weights | `mvn package` (build the artifact) |
| **Inference** | Using the learned policy to act | `java -jar` (run the artifact) |

## Machine Learning / Deep Learning terms

| Term | Definition | Java / AEPO anchor |
|------|-----------|--------------------|
| **AI** | Any technique making a machine act "smart" (incl. rule engines) | the heuristic is "AI" too |
| **Machine Learning (ML)** | Systems that learn rules from data instead of being hand-coded | a self-configuring component |
| **Deep Learning (DL)** | ML using multi-layer neural networks | the world models |
| **Model** | The learned thing you call for predictions | a compiled-from-data ruleset |
| **Neural network** | A tunable math function: numbers in → numbers out | a function with thousands of `double` knobs |
| **MLP** | Multi-Layer Perceptron — fully-connected neural net | `LagPredictor` (16→64→1) |
| **Layer** | One matrix-multiply + bias in a net | `nn.Linear(16, 64)` |
| **Weights / parameters** | The tunable numbers inside a net | what training adjusts |
| **Activation function** | A non-linear squish between layers | ReLU, Sigmoid |
| **ReLU** | `max(0, x)` — kills negatives, adds non-linearity | `nn.ReLU()` |
| **Sigmoid** | Squashes any number into (0,1) | used to bound predictions |
| **LayerNorm** | Normalizes activations per-sample (stable on small batches) | in `MultiObsPredictor` |
| **Forward pass** | Push input through the net to get a prediction | `model(x)` |
| **Loss** | A measure of prediction error | MSE in AEPO |
| **MSE** | Mean Squared Error — avg of `(pred − truth)²` | `nn.MSELoss()` |
| **Backpropagation** | Computing each weight's gradient to reduce loss | `loss.backward()` |
| **Gradient** | The direction to nudge a weight to lower loss | computed automatically |
| **Optimizer** | The thing that nudges weights using gradients | `optim.Adam(lr=1e-3)` |
| **Adam** | A popular adaptive optimizer | AEPO's choice for the nets |
| **Replay buffer** | A fixed-capacity store of past examples to train on | `deque(maxlen=2000)` |
| **Mini-batch** | A small sample of examples per gradient step | 32 in AEPO |
| **Tensor** | A numpy-array-like that PyTorch can autodiff through | `torch.Tensor` |
| **Epoch** | One pass over the training dataset | GRPO: `N_EPOCHS` |
| **Fine-tuning** | Further-training a pre-trained model on your task | the GRPO LLM path |
| **LLM** | Large Language Model — a big text-in/text-out net | Qwen 2.5 |
| **LoRA** | Train tiny adapter weights instead of the whole model | a lightweight patch |
| **GRPO** | Group Relative Policy Optimization — an LLM RL method | `train_grpo_hf.py` |
| **Quantization (4-bit)** | Storing weights in fewer bits to save memory | bitsandbytes |

## AEPO-specific terms

| Term | Definition |
|------|-----------|
| **AEPO** | Autonomous Enterprise Payment Orchestrator — this project |
| **UFRG** | Unified Fintech Risk Gateway — the Round-1 predecessor (aliases still in code) |
| **UnifiedFintechEnv** | The environment class (the simulated gateway) |
| **AEPOObservation / AEPOAction** | The 10-field obs / 6-field action Pydantic DTOs |
| **UFRGReward** | The typed reward DTO (`value` + `breakdown` + flags) |
| **AdversaryPolicy** | The 9-state×3-action Q-table that escalates difficulty (the "2nd learner") |
| **LagPredictor** | The world model predicting next kafka_lag (16→64→1) |
| **MultiObsPredictor** | The full world model predicting all 10 next obs (16→64→64→10) |
| **DynaPlanner** | The Dyna-Q planner running imagined updates in training |
| **Phase** | A segment within an episode: normal / spike / attack / recovery |
| **Task** | A difficulty tier: easy / medium / hard (fixes the phase sequence) |
| **Curriculum level** | 0/1/2 (easy/medium/hard); never regresses |
| **adversary_threat_level** | 0–10 pressure that escalates with a 5-episode lag |
| **Blind spot #1** | Reject+SkipVerify on high risk → +0.04, saves 250 lag (the discovery story) |
| **Blind spot #2** | Match app_priority to merchant_tier → +0.02 |
| **Blind spot #3** | FailFast when pool<20 (avoid −0.10 Backoff waste) |
| **Heuristic** | The hand-coded 3-blind-spot SRE rulebook (the fair baseline) |
| **Conservative policy** | Never-throttle baseline that crashes early (~0.08 on hard) — proves lag-mgmt necessary |
| **Causal transitions** | The 11 rules that make decisions echo across time |
| **Throttle relief** | −150 lag scheduled over the next 2 steps (delayed causality) |
| **Diurnal clock** | Hidden sine load cycle (peak step 25, trough 75) the agent can't observe |
| **Staircase** | The improve→escalate→adapt reward curve (Theme #4 proof) |
| **Crash** | kafka_lag > 4000 for 2 consecutive steps → reward 0, done |
| **Fraud catastrophe** | Approve+SkipVerify on risk>80 → reward 0, done |
| **Dual-mode** | One env class used in-process *and* behind HTTP, unchanged |
| **4-tuple** | `step()` returns `(obs, reward, done, info)` (OpenEnv, not Gymnasium 5-tuple) |
| **info dict** | The ~30-key telemetry envelope returned each step |

## Python / tooling terms

| Term | Definition | Java analogy |
|------|-----------|--------------|
| **`self`** | The instance reference (written explicitly) | `this` |
| **`__init__`** | The constructor | a constructor |
| **`None`** | The absence value | `null` |
| **dunder** | A `__name__`-style special method | language hook (`__len__`→`len()`) |
| **list / tuple / set / dict** | The four core collections | ArrayList / immutable record / HashSet / HashMap |
| **`defaultdict`** | Dict that auto-creates missing values | `computeIfAbsent` |
| **`deque`** | Bounded double-ended queue (ring buffer) | `ArrayDeque` |
| **comprehension** | `[expr for x in xs]` one-line transform | `stream().map().collect()` |
| **decorator** | `@x` wrapper above a function | annotation / AOP advice |
| **f-string** | `f"x={v}"` interpolation | `String.format` |
| **type hint** | `: int`, `-> str` — optional type annotation | a (non-enforced) type declaration |
| **Pydantic `BaseModel`** | Validated, serializable data model | `record` + `@Valid` + Jackson |
| **`Field(ge=, le=)`** | Field bounds | `@Min/@Max` |
| **`.model_dump()`** | Serialize a model to a dict | Jackson `writeValueAsMap` |
| **`async`/`await`** | Non-blocking coroutine syntax | reactive `Mono`/`CompletableFuture` |
| **`with`** | Scoped resource block | try-with-resources |
| **`if __name__=="__main__"`** | "run only when executed directly" | `public static void main` |
| **pip / requirements.txt** | Installer / dependency list | Maven / `<dependencies>` |
| **uv / pyproject.toml / uv.lock** | Fast pkg manager / metadata / lockfile | Gradle / build.gradle / lockfile |
| **virtual env (.venv)** | Per-project isolated libraries | project-local classpath/`.m2` |
| **pickle** | Python object serialization (unsafe on untrusted input) | Java serialization |
| **NumPy** | Vectorized numeric arrays | `double[]` + math library |
| **`np.clip(x,lo,hi)`** | Clamp | `Math.max(lo, Math.min(hi, x))` |
| **`np.argmax`** | Index of the max element | index-of-max loop |
| **pytest** | Test framework | JUnit |
| **coverage** | % of code exercised by tests | JaCoCo |

## Infra / deployment terms

| Term | Definition | Java analogy |
|------|-----------|--------------|
| **Gymnasium** | The standard RL environment interface | a framework SPI you implement |
| **`gym.Env`** | The base class with `reset()`/`step()` | an interface/abstract base |
| **`Box` / `MultiDiscrete`** | Continuous / discrete-vector space types | `double[]` schema / `int[]` enum-array schema |
| **`check_env`** | Gymnasium's conformance validator | a TCK test |
| **FastAPI** | Async Python REST framework | Spring Boot `@RestController` |
| **Uvicorn** | The ASGI server running FastAPI | embedded Tomcat |
| **`@app.post`** | Route decorator | `@PostMapping` |
| **httpx** | Async HTTP client | `WebClient` |
| **OpenAI SDK** | Client for OpenAI-compatible LLM APIs | a vendor SDK with pluggable base URL |
| **Docker image / container** | Packaged filesystem / running instance | (same) |
| **multi-stage build** | Build in one stage, run in a slim one | the same pattern you use for Spring Boot |
| **Hugging Face Spaces** | Free Docker-based ML hosting | Heroku/Cloud Run (ML-flavored) |
| **OpenEnv** | The hackathon's env contract + `validate` CLI | an interface + its TCK |
| **`openenv.yaml`** | The env manifest (tasks, spaces, entry point) | a deployment/service descriptor |
| **HTTP 422** | Unprocessable Entity (validation failure) | `@Valid` failure → 422 |
| **HTTP 400** | Bad Request (here: no active episode) | `ResponseStatusException(BAD_REQUEST)` |
| **`asyncio.Lock`** | Async mutex serializing coroutine access | `ReentrantLock` |
| **EMA** | Exponential Moving Average (`α·new + (1−α)·old`) | the smoothing you know from P99 work |

### Summary

This glossary is your decoder ring. The terms cluster into five families — RL (the loop and learning), ML/DL (the neural-net machinery), AEPO-specific (the project's nouns), Python/tooling (syntax and libraries), and infra/deployment (serving and contracts). When any doc uses a word you've half-forgotten, it's here with a Java handle to grab onto.

➡️ Next: [14_Cheat_Sheet.md](14_Cheat_Sheet.md)
