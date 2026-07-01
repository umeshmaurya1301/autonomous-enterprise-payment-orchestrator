# 02 — Python for Java Developers

> **Goal:** Give you enough Python to read every line of AEPO confidently. Each concept is shown as **Java first → Python second**, and most examples are pulled straight from this codebase so they're immediately useful.

## Table of Contents

1. [The mental shift from Java to Python](#1-the-mental-shift-from-java-to-python)
2. [Running Python & the project toolchain (pip, venv, requirements)](#2-running-python--the-project-toolchain)
3. [Variables & types](#3-variables--types)
4. [Functions & type hints](#4-functions--type-hints)
5. [Collections: list, tuple, set, dict](#5-collections-list-tuple-set-dict)
6. [Classes, objects, inheritance](#6-classes-objects-inheritance)
7. [Modules, packages & imports](#7-modules-packages--imports)
8. [Exceptions](#8-exceptions)
9. [Decorators (≈ Java annotations, but more powerful)](#9-decorators)
10. [Dataclasses & Pydantic models (≈ records/POJOs)](#10-dataclasses--pydantic-models)
11. [Context managers (`with`) (≈ try-with-resources)](#11-context-managers-with)
12. [Generators & comprehensions](#12-generators--comprehensions)
13. [Async/await (≈ reactive / CompletableFuture)](#13-asyncawait)
14. [`__name__ == "__main__"` (≈ public static void main)](#14-name--main)
15. [The 20 Python idioms you'll see in AEPO](#15-the-20-python-idioms-youll-see-in-aepo)
16. [Key takeaways](#16-key-takeaways)

---

## 1. The mental shift from Java to Python

| Java | Python | What it means for you |
|------|--------|-----------------------|
| Compiled (`javac` → bytecode → JVM) | Interpreted (run source directly) | No build step to run; just `python file.py` |
| Static typing enforced by compiler | **Dynamic** typing; type hints are *optional documentation* | `x = 5` then `x = "hi"` is legal. Type hints (`x: int`) are checked by tools, *not* at runtime |
| Braces `{}` define blocks | **Indentation** defines blocks | 4 spaces of indentation is the block. Misaligned code is a syntax error |
| `;` ends statements | newline ends statements | semicolons are legal but unused |
| Everything in a class | Functions & variables can live at **module top level** | a `.py` file can have loose functions, like a utility class but flatter |
| `private`/`public`/`protected` | Convention only: `_name` = "internal", `__name` = name-mangled | no real access control; `_throttle_relief_queue` means "don't touch from outside" |
| `null` | `None` | the absence value |
| `this` | `self` (and you must write it explicitly as the first method param) | every instance method's first parameter is `self` |
| `boolean` `true`/`false` | `bool` `True`/`False` | capitalized |

⚠️ **The #1 gotcha for Java devs:** **indentation is syntax**, not style. In Java you could write everything on one line; in Python the structure *is* the whitespace. If you paste code and the indentation breaks, the program breaks.

☕ The closest thing to "a Python file" is **a Java class file that also allows free-floating static methods and a `main` at the bottom**. `unified_gateway.py` defines classes *and* module-level constants *and* helper functions, all in one file.

---

## 2. Running Python & the project toolchain

### `python file.py`
Runs a file top to bottom. There's no separate compile step.
```bash
python train.py            # runs the training script
python inference.py        # runs the inference client
pytest tests/ -v           # runs all tests (pytest is like `mvn test`)
```

### `pip` ≈ Maven/Gradle dependency resolver
`pip` downloads and installs libraries from PyPI (the "Maven Central" of Python).
```bash
pip install -r requirements.txt   # install everything the project needs
```

### `requirements.txt` ≈ the `<dependencies>` block of `pom.xml`
A flat list of libraries and exact versions. From this project:
```text
gymnasium==0.29.1      # pinned version (== means exactly this)
numpy==1.26.4
pydantic==2.6.4
fastapi==0.110.0
torch==2.2.0+cpu       # the CPU-only build of PyTorch
pytest>=9.0.0          # >= means "this or newer"
```
☕ `==` is like `<version>0.29.1</version>`; `>=9.0.0` is like a version range.

### `pyproject.toml` ≈ `build.gradle` (project metadata + build config)
Modern Python projects also carry a `pyproject.toml`. AEPO's declares the package name, the Python version it needs (`>=3.10, <3.11`), its dependencies, and even an entry-point script (`server = "server.app:main"`, like a Gradle `application` mainClass). It also configures pytest.

### Virtual environment (`.venv/`) ≈ a per-project, isolated dependency set
Python installs libraries globally by default, which causes "project A needs lib v1, project B needs v2" conflicts. A **virtual environment** is a project-local folder (`.venv/`) holding *this project's* libraries, so projects don't clash.
```bash
python -m venv .venv         # create it (like initializing a project sandbox)
.venv\Scripts\activate       # "enter" it on Windows (PowerShell)
```
☕ Think of it as giving each project its **own classpath / its own `.m2`**, so versions never collide. The huge `.venv/Lib/site-packages/` folder you saw in the file listing is exactly that — this project's downloaded libraries (the equivalent of all the JARs Maven pulls into `~/.m2`). You never read or edit it; it's generated.

`uv` (and `uv.lock`) is a newer, *much faster* drop-in replacement for `pip`+`venv` (think "Gradle that's 10× faster than Maven"). `uv.lock` is a fully-resolved, reproducible dependency lockfile — like `gradle.lockfile` or a committed `package-lock.json`.

---

## 3. Variables & types

```java
// Java — type is declared, fixed
double riskScore = 92.0;
int riskDecision = 1;
String task = "hard";
final double CRASH = 4000.0;   // constant
```
```python
# Python — type is inferred, not declared; "constant" is just a naming convention
risk_score = 92.0          # a float
risk_decision = 1          # an int
task = "hard"              # a str
CRASH_THRESHOLD = 4000.0   # UPPER_CASE by convention means "don't reassign"
```

Python *has* types (`int`, `float`, `str`, `bool`, `None`), it just doesn't make you write them. Naming convention does the work Java keywords do:
- `snake_case` for variables/functions (Java uses `camelCase`).
- `UPPER_SNAKE_CASE` for constants (same as Java's `static final`).
- `PascalCase` for classes (same as Java).
- A leading underscore `_like_this` means "internal, treat as private."

From `unified_gateway.py` — module-level constants, exactly like a bag of `public static final`:
```python
CRASH_THRESHOLD: float = 4000.0       # Kafka lag above this = system crash
SLA_BREACH_THRESHOLD: float = 800.0   # P99 above this = SLA breach penalty
HIGH_RISK_THRESHOLD: float = 80.0     # risk_score above this = high risk
```
The `: float` part is a **type hint** (next section) — optional, for readers and tools.

---

## 4. Functions & type hints

```java
// Java
public static int encodeAction(AEPOAction action) {
    return action.riskDecision() * 72 + action.cryptoVerify() * 36 + ...;
}
```
```python
# Python — 'def' defines a function; '-> int' is the return type hint
def encode_action(action: AEPOAction) -> int:
    """Encode AEPOAction to a single integer in [0, 215]."""   # docstring
    fields = (action.risk_decision, action.crypto_verify, ...)
    return sum(f * s for f, s in zip(fields, _STRIDES))
```

Key points:
- `def name(params) -> ReturnType:` — the `-> ReturnType` is the **return type hint**.
- Parameter hints look like `action: AEPOAction` (name `:` type).
- The triple-quoted string right under `def` is a **docstring** — like Javadoc, but it's a real string object accessible at runtime (`encode_action.__doc__`). AEPO docstrings are unusually thorough; treat them as inline documentation.
- **Type hints are not enforced at runtime.** If you pass a `str` where `: int` is hinted, Python won't stop you — only a separate type-checker (like `mypy`) or your IDE flags it. This is the biggest difference from Java's compiler.

### Default parameter values ≈ overloading, but cleaner
```python
def _run_episodes(task: str, policy_fn, seed: int, n_episodes: int = 10) -> float:
    ...
```
`n_episodes: int = 10` means "if the caller omits it, use 10." In Java you'd write multiple overloaded methods; Python does it with defaults. Callers can also pass by name: `_run_episodes("hard", pol, 44, n_episodes=5)`.

### Functions are first-class objects (≈ passing a `Function<T,R>`)
You can store a function in a variable and pass it around, exactly like a Java lambda or method reference:
```python
# graders.py — a "policy function" is passed in and called later
PolicyFn = Callable[[dict[str, float]], AEPOAction]   # a type alias, like Function<Map,AEPOAction>

def _run_episodes(task, policy_fn: PolicyFn, ...):
    action = policy_fn(obs.normalized())   # call the passed-in function
```
☕ `Callable[[dict], AEPOAction]` is Python's way of writing `Function<Map<String,Double>, AEPOAction>`.

---

## 5. Collections: list, tuple, set, dict

This is where Python feels different. Four core collections, all with literal syntax:

| Python | Literal | Mutable? | Java analogy | Used in AEPO for |
|--------|---------|----------|--------------|------------------|
| `list` | `[1, 2, 3]` | yes | `ArrayList<>` | episode reward sequences, schedules |
| `tuple` | `(1, 2, 3)` | **no** (immutable) | an immutable `record`, or `List.of()` | the return of `step()`, Q-table keys |
| `set` | `{1, 2, 3}` | yes | `HashSet<>` | the set of valid tasks, required info keys |
| `dict` | `{"a": 1}` | yes | `HashMap<>` | the `info` dict, normalized observations |

### list ≈ ArrayList
```python
step_rewards = []                 # new empty list (like new ArrayList<>())
step_rewards.append(0.84)         # add (like .add())
step_rewards[0]                   # index access (like .get(0))
len(step_rewards)                 # size (like .size())
for r in step_rewards: ...        # iterate (like for-each)
mean = sum(step_rewards) / len(step_rewards)
```

### tuple ≈ an immutable record / a fixed-arity return
A tuple is a fixed, immutable sequence. Python uses it constantly to **return multiple values at once**, which Java can't do without a wrapper class:
```python
# unified_gateway.py — step() returns FOUR things as one tuple
return self._current_obs, typed_reward, done, info
#       ^obs              ^reward       ^bool ^dict

# The caller "unpacks" them in one line (destructuring):
obs, reward, done, info = env.step(action)
```
☕ In Java you'd define a `record StepResult(Obs obs, Reward reward, boolean done, Map info)` and return that. Python returns a 4-tuple and the caller destructures it. **This 4-tuple is one of the most important contracts in the whole project** (see Doc 03 & 07).

Tuples are also used as **composite map keys** — because they're immutable and hashable. The Q-table is a dict keyed by a tuple of bin indices:
```python
# train.py — state is a 7-tuple of small ints; used as a dict key
state = (2, 0, 1, 3, 0, 1, 2)
q_table[state]           # look up Q-values for that discretized state
```
☕ Java can't use an array as a `HashMap` key safely (array equality is by reference). A Python tuple is value-equal and hashable, so `(2,0,1,3,0,1,2)` is a perfectly good key — like a Java `record` used as a `Map` key.

### set ≈ HashSet
```python
# server/app.py — membership check, O(1)
if task_name not in {"easy", "medium", "hard"}:
    raise HTTPException(...)
```
`x in some_set` is `set.contains(x)`. The `_REQUIRED_INFO_KEYS` in `inference.py` is a `frozenset` (an **immutable** set, like `Set.of(...)`), used to validate that the server returned every required field.

### dict ≈ HashMap (the workhorse)
```python
# A dict literal — keys are strings, values are floats
breakdown = {
    "base": 0.8,
    "sla_penalty": -0.30,
    "bonus": 0.04,
}
breakdown["base"]                # get -> 0.8  (like .get("base"))
breakdown.get("missing", 0.0)    # get-or-default -> 0.0 (like getOrDefault)
breakdown["new_key"] = 1.0       # put
for key, value in breakdown.items():   # iterate entries (like entrySet())
    ...
```
The entire `info` object returned from `step()` is a `dict[str, Any]` — a `Map<String, Object>`. The agent's observation, after `.normalized()`, is a `dict[str, float]` — a `Map<String, Double>`.

⚠️ **`dict` vs Pydantic model:** AEPO uses *both*. `AEPOObservation` is a *typed, validated* Pydantic model (like a `record` with `@Valid`). Its `.normalized()` method returns a *plain dict* (an untyped `Map`) for fast math. Knowing which one you're holding matters when reading the code.

### `defaultdict` ≈ `computeIfAbsent`
```python
from collections import defaultdict
# train.py — a Q-table that auto-creates a zero-array on first access to a new key
q_table = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
q_table[new_state]   # if new_state absent, it's auto-initialized to zeros — no KeyError
```
☕ Exactly `map.computeIfAbsent(key, k -> new double[N])`. This is why the Q-table can be "sparse" — entries spring into existence only when first visited.

### `deque` ≈ ArrayDeque (a fixed-size sliding window)
```python
from collections import deque
# unified_gateway.py — a bounded queue; when full, the oldest item is dropped
self._latency_window: deque[float] = deque(maxlen=20)
self._latency_window.append(x)   # pushes; auto-evicts oldest if > 20 items
```
☕ Like an `ArrayDeque` with a cap, used as a ring buffer. AEPO uses deques for the throttle-relief queue and rolling windows.

---

## 6. Classes, objects, inheritance

```java
// Java
public class AdversaryPolicy {
    private Map<Key, Double> q = new HashMap<>();
    private int epCount = 0;

    public AdversaryPolicy() { }              // constructor

    public int selectAction(...) { ... }      // method
    private static int bin3(double v) { ... } // static helper
}
```
```python
# Python — unified_gateway.py (simplified)
class AdversaryPolicy:
    """Tiny 9-state x 3-action Q-table adversary."""   # class docstring

    BURST: int = 0          # class-level constant (like static final)
    SUSTAIN: int = 1

    def __init__(self) -> None:                # the constructor is __init__
        self._q = defaultdict(float)           # instance field (self. = this.)
        self._ep_count = 0

    def select_action(self, rng, perf, threat) -> int:   # method: note 'self'
        ...

    @staticmethod
    def _bin3(value, lo, hi) -> int:           # static method (no self)
        ...
```

The rules that surprise Java devs:
- **The constructor is named `__init__`** (double-underscore "dunder" method). Python calls it when you write `AdversaryPolicy()`.
- **`self` is explicit.** Every instance method takes `self` (≈ `this`) as the first parameter, *and you write it*. `self._q` is `this.q`.
- **No `new` keyword.** You construct with just `AdversaryPolicy()`.
- **Fields are created by assignment in `__init__`** — there's no separate field declaration block. `self._ep_count = 0` both declares and initializes.
- **`@staticmethod`** marks a method that doesn't use `self` — exactly Java's `static`.
- **`_q` (one underscore)** = "internal/private by convention." Python won't enforce it.

### Inheritance ≈ `extends`
```python
# unified_gateway.py
class UnifiedFintechEnv(gym.Env):        # extends gym.Env
    def __init__(self) -> None:
        super().__init__()               # call the parent constructor
        ...
    def reset(self, seed=None, options=None):   # override a parent method
        super().reset(seed=seed)                # call parent's version first
        ...
```
`class Child(Parent):` is `class Child extends Parent`. `super().__init__()` is `super()`. Overriding is implicit — you just define a method with the same name (no `@Override`). Here `UnifiedFintechEnv` *implements* the Gymnasium environment interface by extending `gym.Env` and overriding `reset()` and `step()`.

`UFRGReward(BaseModel)` and `AEPOObservation(BaseModel)` extend Pydantic's `BaseModel` — that's how they get free validation and serialization (next section).

### "Dunder" methods ≈ operator/interface hooks
Methods named `__like_this__` hook into language features:
- `__init__` — constructor.
- `__len__` — makes `len(obj)` work.
- `forward` (in PyTorch) — not a dunder, but similarly "the method the framework calls."

---

## 7. Modules, packages & imports

- A **module** = a single `.py` file. `unified_gateway.py` is the `unified_gateway` module.
- A **package** = a folder of modules (historically with an `__init__.py` marker). `server/` is a package; `tests/` is a package (it has `tests/__init__.py`).

```java
// Java
import aepo.types.AEPOObservation;
import static aepo.types.Bounds.RISK_MAX;
```
```python
# Python — import a name from a module
from unified_gateway import UnifiedFintechEnv, AEPOAction
from aepo_types import AEPOObservation, RISK_MAX
import numpy as np                 # import whole module under an alias
import gymnasium as gym            # 'gym' is now the alias for gymnasium
```

Patterns you'll see everywhere in AEPO:
- `import numpy as np` — import the whole library under a short alias. `np.clip(...)`, `np.zeros(...)`. (Like `import com.foo.Bar` then using `Bar`, but aliased.)
- `from x import y` — pull specific names into scope. `from collections import deque`.
- `from __future__ import annotations` — a forward-compat pragma at the top of most files; lets you use newer type-hint syntax (e.g. `int | None`) on Python 3.10. Harmless; think of it as a compiler flag.

☕ `numpy` ≈ a Maven artifact you `import`. The alias `np` is just a short local name. There's no package-name-must-match-folder rule as strict as Java's, but in practice module name = file name.

### The dependency rule AEPO deliberately enforces
`aepo_types.py` says it is *"the ONLY place AEPOObservation and AEPOAction are defined. Both the server and the client import from here — neither imports from the other."* This is a clean **shared-contract module** (like a `common-dto` Maven module that both `service` and `client` depend on, but which depends on neither). It satisfies OpenEnv's client/server separation rule.

---

## 8. Exceptions

Nearly identical to Java, different keywords:

| Java | Python |
|------|--------|
| `try { } catch (E e) { } finally { }` | `try: ... except E as e: ... finally: ...` |
| `throw new IllegalArgumentException("x")` | `raise ValueError("x")` |
| `extends Exception` | `class MyError(Exception):` |

```python
# unified_gateway.py — raising, exactly like throw new IllegalArgumentException
if task_name not in {"easy", "medium", "hard"}:
    raise ValueError(f"Unknown task {task_name!r}; expected 'easy', 'medium', or 'hard'.")
```
```python
# inference.py — try/except/fallback, with a bare 'except' catching everything
try:
    response = llm_client.chat.completions.create(...)
    return parse_llm_action(reply)
except Exception:                 # catch-all (like catch (Exception e))
    return get_action(..., agent_mode="heuristic")   # graceful fallback
```

Notes:
- `except Exception:` catches almost everything (like `catch (Exception e)`). A bare `except:` (no type) is even broader and discouraged.
- `raise X from Y` chains exceptions (preserves the cause), like Java's `new X(..., cause)`.
- The `f"...{var}..."` is an **f-string** (next idiom) — string interpolation, like a formatted template. `{task_name!r}` inserts the `repr()` (quoted) form.

---

## 9. Decorators

A **decorator** is a function that wraps another function/class to add behavior — visually like a Java annotation (`@Something` above a method), but it's actually executable code, more like an **AOP advice** or a wrapping `BeanPostProcessor`.

```java
// Java — annotations are metadata read by a framework
@PostMapping("/step")
public ResponseEntity<?> step(@RequestBody StepRequest req) { ... }

@Override
public void reset() { ... }
```
```python
# Python — decorators wrap the function below them
@app.post("/step", tags=["env"])         # FastAPI route — like @PostMapping
async def step_env(request: Request):
    ...

@staticmethod                            # like Java 'static'
def _build_phase_schedule(task_name): ...

@classmethod                             # method that gets the class, not instance
def from_array(cls, obs): ...

@property                                # makes a method callable like a field
def something(self): ...
```

What you need to recognize in AEPO:
- `@app.post("/reset")`, `@app.get("/state")` — **FastAPI routing**, the direct analog of `@PostMapping`/`@GetMapping`. The decorator registers the function as an HTTP handler.
- `@staticmethod` — Java `static`.
- `@classmethod` — a method whose first arg is the class (`cls`) not the instance; used as an alternative constructor. `AEPOObservation.from_array(...)` builds an instance from a numpy array.
- `@dataclass` (not used heavily here, but common) — auto-generates `__init__`, `equals`, `toString` for a class. Pydantic's `BaseModel` does the same job with validation on top.

🔧 **Technical reality:** `@app.post("/step")` literally means `step_env = app.post("/step")(step_env)` — the decorator is a function that takes your function and returns a registered version. You don't need to internalize that; just read `@app.post` as "this is the POST /step handler."

---

## 10. Dataclasses & Pydantic models

This is the most important section for reading AEPO's data layer.

In Java you'd model the observation as a validated record:
```java
public record AEPOAction(
    @Min(0) @Max(2) int riskDecision,
    @Min(0) @Max(1) int cryptoVerify,
    @Min(0) @Max(2) int infraRouting,
    @Min(0) @Max(1) int dbRetryPolicy,
    @Min(0) @Max(1) int settlementPolicy,
    @Min(0) @Max(2) int appPriority
) { }
```
AEPO does exactly this with **Pydantic**, the de-facto "Bean Validation for Python":
```python
# aepo_types.py
from pydantic import BaseModel, Field

class AEPOAction(BaseModel):
    risk_decision: int = Field(ge=0, le=2)               # ge=>=, le=<=
    crypto_verify: int = Field(ge=0, le=1)
    infra_routing: int = Field(ge=0, le=2)
    db_retry_policy: int = Field(default=0, ge=0, le=1)  # default + bounds
    settlement_policy: int = Field(default=0, ge=0, le=1)
    app_priority: int = Field(default=2, ge=0, le=2)
```
- `BaseModel` is the parent that grants validation + serialization (like extending a framework base class).
- `Field(ge=0, le=2)` is the constraint: `ge` = "greater-or-equal", `le` = "less-or-equal". This is `@Min(0) @Max(2)`.
- **Validation happens at construction.** `AEPOAction(risk_decision=9)` *throws* (`ValidationError`) — the env never sees invalid input. This is why the FastAPI server can return HTTP 422 automatically for bad payloads.
- `.model_dump()` serializes to a dict (like Jackson `writeValueAsMap`); `.model_dump_json()` to a JSON string (`writeValueAsString`).
- `AEPOAction(**action_dict)` constructs from a dict by spreading its keys as named args (`**` = "spread this map as keyword arguments"), like building a record from a `Map`.

☕ **Mental model:** `class X(BaseModel)` ≈ `record X` + `@Valid` + Jackson, all in one. When you see `BaseModel`, think "validated, serializable DTO."

`UFRGReward(BaseModel)` in `unified_gateway.py` is the typed reward DTO: it carries `value: float` (bounded 0–1), a `breakdown: dict[str, float]`, and two booleans (`crashed`, `circuit_breaker_tripped`).

---

## 11. Context managers (`with`)

Python's `with` block is **try-with-resources**: it guarantees setup and cleanup around a block.

```java
// Java
try (var client = WebClient.create()) {
    client.call();
}   // auto-closed
```
```python
# inference.py — the async HTTP client is opened and guaranteed-closed
async with httpx.AsyncClient(base_url=SPACE_URL, timeout=30.0) as http:
    obs = await http_reset(http, task)
    ...
# 'http' is automatically closed when the block exits, even on exception
```
```python
# A file is opened and guaranteed-closed
with open(qtable_path, "rb") as f:     # "rb" = read binary
    snapshots = pickle.load(f)
# f.close() happens automatically here
```

You'll also see `with torch.no_grad():` in the neural-net code — that's a context manager that temporarily turns off gradient tracking (a performance optimization during prediction). Doc 03 explains gradients; for now, read `with torch.no_grad():` as "we're just predicting, not learning, so skip the bookkeeping."

---

## 12. Generators & comprehensions

### List comprehension ≈ a Stream `.map().collect()` on one line
```java
// Java
List<Double> vals = keys.stream().map(k -> norm.get(k)).collect(toList());
```
```python
# Python — dynamics_model.py
obs_vals = [float(obs_normalized[k]) for k in obs_keys]
#           └─ map each k → value, collect into a list ─┘
```
Read `[EXPR for ITEM in ITERABLE]` as "for each item, compute expr, collect into a list." There are also dict comprehensions:
```python
# Build a dict from two parallel sequences (like zipping keys & values into a Map)
return {k: float(v.item()) for k, v in zip(_OBS_KEYS, out)}
```
…and conditional ones:
```python
bins = [min(int(val * N_BINS), N_BINS - 1) for val in values if val > 0]
```

### Generators ≈ lazy streams
A generator produces values on demand instead of building a whole list (memory-efficient, like a lazy `Stream`). `sum(f * s for f, s in zip(fields, _STRIDES))` uses a **generator expression** (note: parentheses, not brackets) — it computes products lazily and sums them, never materializing a list. `range(100)` is a generator-like lazy sequence (`for i in range(100)` ≈ `for (int i=0;i<100;i++)`).

`zip(a, b)` pairs two sequences element-wise (like iterating two lists in lockstep): `zip([1,2],[3,4])` → `(1,3), (2,4)`.

---

## 13. async/await

FastAPI and the inference client are **asynchronous** — the same idea as Java's reactive stack (`Mono`/`Flux`/`CompletableFuture`) or Node.js. An `async def` function can `await` other async calls without blocking a thread.

```python
# server/app.py
@app.post("/step")
async def step_env(request: Request):       # async handler
    body = await request.json()             # await: yield while I/O happens
    async with _env_lock:                   # async lock (serialize mutations)
        obs, reward, done, info = env.step(action)
    return {...}
```
```python
# inference.py
async def main() -> None:
    async with httpx.AsyncClient(...) as http:
        obs = await http_reset(http, task)         # await an HTTP call
        obs, reward, done, info = await http_step(http, action)

asyncio.run(main())     # bootstrap the async event loop (like starting the reactor)
```

What you need:
- `async def` = an async function (returns a coroutine, not a value, until awaited).
- `await x` = "wait for this async thing, but let other tasks run meanwhile."
- `asyncio.run(main())` = start the event loop and run `main()` to completion (the entry point).
- `asyncio.Lock()` = an async mutex. AEPO uses `_env_lock` so two concurrent HTTP requests can't corrupt the single shared environment's mid-episode state — exactly the concurrency concern you'd guard with a lock around shared mutable state in a Spring singleton.

☕ Read `async`/`await` as "this is the reactive/non-blocking layer." The *logic* is sequential; the keywords just let the server handle many requests on a small thread pool.

---

## 14. `__name__ == "__main__"`

```python
# Every runnable AEPO script ends with this
def main() -> None:
    ...

if __name__ == "__main__":
    main()
```
This is Python's `public static void main`. A `.py` file can be **imported as a library** *or* **run as a script**. When imported, `__name__` is the module's name (`"train"`); when run directly (`python train.py`), `__name__` is the special string `"__main__"`. So `if __name__ == "__main__": main()` means **"only run `main()` when this file is executed directly, not when it's imported."**

☕ It's the guard that says "this is the executable entry point." Without it, importing `train.py` to reuse one function would accidentally kick off a 20-minute training run.

---

## 15. The 20 Python idioms you'll see in AEPO

A quick-reference table. Skim now; return when a symbol confuses you.

| Idiom | Means | Java-ish equivalent |
|-------|-------|---------------------|
| `x: int = 5` | typed variable | `int x = 5` |
| `def f(a: int) -> str:` | function with hints | method signature |
| `self` | the instance | `this` |
| `None` | absence | `null` |
| `True` / `False` | booleans | `true` / `false` |
| `f"val={x:.2f}"` | formatted string | `String.format("val=%.2f", x)` |
| `obs, r, done, info = step()` | tuple unpacking | destructure a record |
| `[e for e in xs]` | list comprehension | `stream().map().collect()` |
| `{k: v for ...}` | dict comprehension | build a Map in a stream |
| `x in collection` | membership | `.contains(x)` |
| `len(xs)` | size | `.size()` / `.length` |
| `range(n)` | 0..n-1 lazy sequence | `for(i=0;i<n;i++)` |
| `zip(a, b)` | pair two sequences | iterate two lists in lockstep |
| `enumerate(xs)` | index + value pairs | `for(i,e)` with counter |
| `**kwargs` / `*args` | spread/collect args | varargs / Map spread |
| `AEPOAction(**d)` | build from dict | construct from Map |
| `@decorator` | wrap function | annotation / AOP advice |
| `with open(...) as f:` | scoped resource | try-with-resources |
| `lambda x: x+1` | anonymous function | `x -> x+1` |
| `import x as y` | aliased import | import + local rename |

Two more worth their own note:
- `*args` / `**kwargs`: `*args` collects extra positional arguments into a tuple; `**kwargs` collects extra named arguments into a dict (like Java varargs, but for both positions and names). You'll see `def env_reward_func(completions, prompts, seed_val, task_name, **kwargs)` — the `**kwargs` swallows any extra fields the framework passes.
- `min`/`max`/`sum`/`round`: built-in functions, not methods. `round(infra_penalty, 4)` rounds to 4 decimals; `max(0.0, min(1.0, raw))` is the clamp-to-[0,1] idiom used all over the reward function (≈ `Math.max(0.0, Math.min(1.0, raw))`).

---

## 16. Key takeaways

- Python is **dynamically typed**; type hints (`: int`, `-> str`) are optional documentation, not compiler rules.
- **Indentation is structure** — the #1 thing to respect.
- **Constructor = `__init__`**, **`this` = `self`** (written explicitly), no `new`, `_name` = "private by convention."
- The four collections — `list` (ArrayList), `tuple` (immutable record / multi-return), `set` (HashSet), `dict` (HashMap) — are everywhere; `defaultdict`/`deque` cover `computeIfAbsent`/`ArrayDeque`.
- **`step()` returns a 4-tuple** `(obs, reward, done, info)` and callers **unpack** it — memorize this; it's the core loop contract.
- **Pydantic `BaseModel` = validated, serializable DTO** (`record` + `@Valid` + Jackson). `Field(ge=, le=)` = `@Min/@Max`.
- **Decorators (`@app.post`)** are FastAPI's `@PostMapping`. **`with`** is try-with-resources. **`async/await`** is the reactive/non-blocking layer.
- **`if __name__ == "__main__":`** is `public static void main`.

### Summary

You can now read the AEPO source without the syntax fighting you. The remaining unknowns aren't Python — they're *Reinforcement Learning concepts* (what's an "observation," a "reward," a "Q-table"?). That's Doc 03, which assumes zero ML background and builds it up with diagrams.

➡️ Next: [03_RL_From_Scratch.md](03_RL_From_Scratch.md)
