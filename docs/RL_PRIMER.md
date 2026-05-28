# RL & PyTorch Primer for AEPO

> **Audience:** Umesh — Java/Spring Boot backend engineer, ~3.5 yrs, never touched ML before.
> **Goal:** Walk you from zero to "I can defend every line of `train.py` in front of a Meta judge."
> **Style:** Java analogies first, math second. Nothing magic.

---

## Table of contents

1. [What is Reinforcement Learning (RL)?](#1-what-is-reinforcement-learning-rl)
2. [The five vocabulary words you must know](#2-the-five-vocabulary-words-you-must-know)
3. [How an episode actually runs (Java-style trace)](#3-how-an-episode-actually-runs-java-style-trace)
4. [The core RL algorithms — what they are, when to use them](#4-the-core-rl-algorithms)
5. [Q-Learning — the algorithm AEPO uses](#5-q-learning--the-algorithm-aepo-uses)
6. [What is a "world model"? (LagPredictor explained)](#6-what-is-a-world-model)
7. [PyTorch — what it is, why it exists, what it does in AEPO](#7-pytorch)
8. [What is Gymnasium / OpenEnv?](#8-gymnasium--openenv)
9. [How AEPO trains, end to end](#9-how-aepo-trains-end-to-end)
10. [Reading reward curves (and what "the staircase" means)](#10-reading-reward-curves)
11. [Glossary](#11-glossary)

---

## 1. What is Reinforcement Learning (RL)?

### The one-sentence definition
**RL is how you teach a program to make a sequence of decisions by letting it try things and rewarding good outcomes.**

### Java analogy
You've written a UPI switch. Imagine you didn't know the rules of fraud detection — instead, you wrote a service that:

1. Receives a transaction (the **observation**).
2. Picks one of several actions (Approve / Reject / Challenge).
3. Gets a number back saying "that was good (+0.8)" or "that was bad (-0.5)" — the **reward**.
4. Slowly updates an internal table (`Map<State, Map<Action, Double>>`) so that next time it sees a similar transaction, it picks the action with the highest expected reward.

That's it. That's RL. There's no magic. The "intelligence" comes from doing this loop **millions of times** until the table converges to good answers.

### The three families of ML — where RL fits

| Family | What it does | Java analogy |
|---|---|---|
| **Supervised learning** | "Here's 1M labeled images. Learn to classify new ones." | Like writing a `if/else` rule from training examples. |
| **Unsupervised learning** | "Here's data. Find structure." | Like clustering customers by behavior — no labels. |
| **Reinforcement learning** | "Here's an environment. Try things. Maximize reward." | Like A/B testing with memory — but the system learns by itself. |

RL is the only one where **the data doesn't exist yet** — the agent generates it by acting.

### Why RL for payments?
Payment systems have:
- **Sequential decisions** (each transaction affects the next via lag/state).
- **Delayed consequences** (skipping verification saves time *now* but causes a fraud loss 50 steps later).
- **No labeled "correct answer"** — only outcomes (revenue, fraud loss, SLA breach).

That's exactly the shape RL is built for.

---

## 2. The five vocabulary words you must know

These five words map onto AEPO directly. Memorize them.

| Word | What it means | AEPO equivalent |
|---|---|---|
| **Environment** | The "world" the agent acts in. Has state, transitions, rewards. | `UnifiedFintechEnv` in `unified_gateway.py` |
| **Agent** | The decision-maker. Sees observations, picks actions. | The Q-table in `train.py` (during training); `inference.py` HTTP client (at eval) |
| **Observation** | What the agent can see at one moment. A vector of numbers. | The 10 normalized signals (risk_score, kafka_lag, …) |
| **Action** | What the agent does. Discrete or continuous. | The 6-tuple decision (risk_decision, crypto_verify, …) |
| **Reward** | A scalar number telling the agent "good or bad" | The `[0.0, 1.0]` value from `reward_breakdown.final` |

### Loop diagram (this is literally all RL is)

```
            ┌──────────────────────────────┐
            │                              │
            ▼                              │
   ┌────────────────┐    action     ┌──────┴──────┐
   │     AGENT      │──────────────▶│ ENVIRONMENT │
   │  (Q-table or   │               │  (UFE.step) │
   │   neural net)  │◀──────────────│             │
   └────────────────┘  obs, reward  └─────────────┘
```

**Every RL algorithm is a different way to update the agent given (obs, action, reward, next_obs) tuples.** That's it.

---

## 3. How an episode actually runs (Java-style trace)

Here's a hand-written trace of one AEPO episode, the way you'd debug a Spring Boot endpoint.

```python
# Pseudo-code with Java mental model in comments

env = UnifiedFintechEnv()              // new UnifiedFintechEnv();
obs = env.reset(task="hard")           // env.reset("hard"); returns AEPOObservation
total_reward = 0.0

for step in range(100):                // for (int step=0; step<100; step++)
    action = agent.choose(obs)         // Map<Action,Double> qVals = qTable.get(obs);
                                       // pick argmax, or random with ε probability
    obs, reward, done, info = env.step(action)
    total_reward += reward
    agent.update(obs, action, reward)  // qTable.put(...) — Bellman update
    if done:
        break                          // crash, fraud, or step==100

# An "episode" = one full reset → 100 steps → done.
# "Training" = run thousands of episodes, updating the agent each step.
```

### Key terms in this loop

- **Step**: one (obs, action, reward) cycle. AEPO has 100 steps per episode.
- **Episode**: one full reset-to-done. Bounded at 100 steps, or earlier on crash.
- **Rollout**: collecting many episodes' worth of data.
- **Trajectory**: the sequence of `(s, a, r, s')` tuples in one episode.
- **Return**: `sum(rewards)` over an episode. AEPO uses **mean reward** (sum/100) as the score.
- **Convergence**: when the agent stops improving — Q-values stabilize.

---

## 4. The core RL algorithms

You don't need to master all of these. You need to **know what they are** so you can answer "why did you pick X?" in Q&A.

### 4.1 Tabular methods (no neural networks)

| Algorithm | Idea | Pros | Cons |
|---|---|---|---|
| **Q-Learning** | Store `Q[state][action] = expected_reward`. Update via Bellman equation. | Simple, provably converges, no GPU needed. | Only works if state space is small/discrete. |
| **SARSA** | Like Q-learning but updates with the action *actually taken* next, not the best one. | More conservative, safer in risky envs. | Slower convergence. |
| **Monte Carlo** | Wait until episode ends, then update. | Unbiased. | High variance, slow. |

**AEPO uses Q-Learning** because (a) we discretize the 10-d obs into 8 bins each = manageable, (b) it runs on 2 vCPU in 20 minutes, (c) judges can read every line and verify it works.

### 4.2 Deep RL (neural networks instead of tables)

| Algorithm | Idea | When to use |
|---|---|---|
| **DQN (Deep Q-Network)** | Replace Q-table with a neural net `f(state) → Q-values`. | Continuous/high-dim state. |
| **Policy Gradient (REINFORCE)** | Learn `π(action \| state)` directly as a probability distribution. | When actions are continuous. |
| **PPO (Proximal Policy Optimization)** | Policy gradient with a "trust region" — small safe updates. | Industry default for most modern RL. |
| **GRPO (Group Relative Policy Optimization)** | PPO variant used to fine-tune LLMs. | LLM-as-agent scenarios. |
| **Actor-Critic** | Two networks: actor picks actions, critic predicts value. | When you want stability + speed. |

**AEPO has an optional GRPO mode** (`train.py` Option B) using HuggingFace TRL + Qwen-0.5B as the agent. The LLM reads obs as JSON, outputs an action as JSON. This is the "Scaler bonus / LLM-as-agent" angle.

### 4.3 Model-based RL

The agent learns a **model of the environment** (`f(s, a) → s'`) and uses it to plan.

**AEPO's `LagPredictor` is the model.** More on this in §6.

---

## 5. Q-Learning — the algorithm AEPO uses

This is the algorithm you'll explain in Q&A. Memorize the equation.

### The Bellman update

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max_a' Q(s', a') − Q(s, a) ]
                          └────────────────────────┘
                          "TD target" — what we now think Q(s,a) should be
```

In plain English: **"Move my old estimate of Q(s,a) a little bit toward the new target."**

### What each symbol means

| Symbol | Name | AEPO value | Meaning |
|---|---|---|---|
| `Q(s, a)` | Q-value | `Map<State, Map<Action, Double>>` entry | "Expected total future reward if I do `a` from `s`" |
| `α` (alpha) | Learning rate | `0.1` | How fast we move toward the target. Too big = unstable. |
| `γ` (gamma) | Discount factor | `0.95` | How much future rewards matter vs immediate. 0=greedy, 1=far-sighted. |
| `r` | Reward | from `env.step()` | Immediate reward this step. |
| `s'` | Next state | next `obs` | What we observed after the action. |
| `max_a' Q(s', a')` | Bootstrap | lookup in table | Best Q-value at the next state. |

### ε-greedy exploration

If the agent always picks `argmax Q(s, a)`, it never tries new things. So we use:

```python
if random() < epsilon:
    action = random_action()        // explore
else:
    action = argmax(Q[state])       // exploit
```

AEPO decays `ε` from `1.0` (always random) at episode 0 to `0.05` (5% random) by episode 500. Standard "explore early, exploit late" schedule.

### Discretization (so we can use a table)

Our observation has 10 continuous floats in `[0,1]`. A Q-table needs discrete keys. So:

```python
bin_index = int(obs_value * 8)     # 0..7
state_key = tuple(bin_indices)     # (3, 5, 1, 0, 7, 2, 4, 6, 1, 0)
```

Total state space: `8^10 ≈ 1 billion` entries. We won't visit all of them — only the ones the agent actually encounters, which is a few thousand. The Q-table is a `defaultdict`, so unseen states default to zeros.

---

## 6. What is a "world model"?

A **world model** is a learned function that **predicts what the environment will do next**. The agent can use it to "imagine" outcomes without acting in the real env.

### AEPO's world model: LagPredictor

```python
class LagPredictor(nn.Module):
    # Input:  10 obs values + 6 one-hot action = 16 numbers
    # Output: predicted kafka_lag at the next step (1 number)
    # 2-layer MLP (multi-layer perceptron — the simplest neural net)
```

**Why this matters for the pitch:** Theme #3.1 is "World Modeling." The judges want to see that you have a *learned* component that approximates environment dynamics. The LagPredictor:
- Trains alongside Q-learning on collected `(obs, action, next_kafka_lag)` triples.
- After training, you can ask "what if I throttle now?" and get a predicted lag without stepping the real env.
- Justifies the world-model claim in 40 lines of PyTorch.

### Important distinction (don't confuse these)

| Term | What it is |
|---|---|
| **Causally-structured simulation environment** | The `UnifiedFintechEnv` itself — the *physics*, hand-coded by us. |
| **Learned world model** | The `LagPredictor` MLP — the agent's *approximation* of that physics, learned from data. |

When judges ask "is this a world model?", you say: *"The environment is the ground-truth physics; the LagPredictor is a learned neural model of one slice of those dynamics — kafka lag transitions — and demonstrates that the agent can predict environment behavior, not just react to it."*

---

## 7. PyTorch

### What is PyTorch?

PyTorch is **a Python library for building and training neural networks**. Think of it as the Spring Boot of deep learning — it gives you the framework so you don't reinvent matrix math.

### What does it actually do?

Three things:

1. **Tensors** — multidimensional arrays (like `double[][][]` but on GPU).
2. **Autograd** — automatic differentiation. You write `y = f(x)`, PyTorch computes `dy/dx` for you.
3. **`nn.Module`** — base class for neural networks. You subclass it like Spring's `@Service`.

### A complete PyTorch example (this IS the LagPredictor)

```python
import torch
import torch.nn as nn

class LagPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Two linear layers with ReLU activation between them
        self.fc1 = nn.Linear(16, 32)   # 16 inputs → 32 hidden neurons
        self.fc2 = nn.Linear(32, 1)    # 32 hidden  → 1 output (predicted lag)

    def forward(self, x):
        x = torch.relu(self.fc1(x))    # ReLU = max(0, x), the standard nonlinearity
        return self.fc2(x)             # final output (no activation — regression)

# === Training loop ===
model = LagPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # the "trainer"
loss_fn = nn.MSELoss()                                       # mean-squared-error loss

for batch in training_data:
    inputs, targets = batch
    predictions = model(inputs)        # forward pass
    loss = loss_fn(predictions, targets)  # how wrong were we?
    optimizer.zero_grad()              # clear old gradients
    loss.backward()                    # compute new gradients (autograd magic)
    optimizer.step()                   # update weights using gradients
```

### Java analogy

| PyTorch concept | Java analogy |
|---|---|
| `torch.Tensor` | `double[][]` but with built-in math + GPU support |
| `nn.Module` | Abstract class like `JpaRepository` — you subclass and override `forward()` |
| `nn.Linear(in, out)` | A matrix-multiply layer: `y = W·x + b`, where W and b are learned |
| `loss.backward()` | "Walk back through the computation graph and compute gradients" — there's no Java analog |
| `optimizer.step()` | "Apply the gradients to update weights" — the training step |
| `model.train()` / `model.eval()` | Mode switch (some layers behave differently during training, e.g., dropout) |

### Why neural networks at all?

A Q-table has `8^10 ≈ 1B` cells. Most are never visited. A neural net **generalizes** — if it saw `(risk=0.7, lag=0.4)`, it will give a sensible answer for `(risk=0.71, lag=0.41)` even without training on it. That's the entire reason DL exists.

For AEPO, we use the simpler Q-table because the discretization is good enough and judges can read the code. PyTorch shows up only for the LagPredictor (world-model claim) and optional GRPO training.

---

## 8. Gymnasium / OpenEnv

### Gymnasium
The standard Python interface for RL environments. Defines `reset()` and `step()` so any agent can talk to any env. Originally OpenAI Gym, now maintained by Farama as `gymnasium`.

```python
class MyEnv(gym.Env):
    def reset(self) -> obs: ...
    def step(self, action) -> (obs, reward, done, info): ...
```

That's the entire contract. Every AEPO file follows it.

### OpenEnv
Meta's **deployment spec** built on top of Gymnasium. It says:
- Wrap your env in a FastAPI server with `/reset`, `/step`, `/state` endpoints.
- Provide `openenv.yaml` declaring tasks and graders.
- Containerize with Docker, port 7860.
- Make the whole thing reproducible via fixed seeds.

So: Gymnasium = the **interface**. OpenEnv = the **packaging + deployment + grading spec**.

AEPO is OpenEnv-compliant. That's the hackathon submission requirement.

### The 4-tuple vs 5-tuple footgun
Modern Gymnasium returns `(obs, reward, terminated, truncated, info)` — 5 items.
**OpenEnv requires the legacy 4-tuple** `(obs, reward, done, info)`.
Don't switch. CLAUDE.md locks this down. Breaking it breaks the grader, server, and inference simultaneously.

---

## 9. How AEPO trains, end to end

Here is the whole training story in one place.

### Step-by-step

```
1. Initialize Q-table as defaultdict(lambda: zeros(N_ACTIONS))
2. Initialize LagPredictor neural net (random weights)
3. For episode in range(2000):
   a. obs = env.reset("hard")
   b. For step in range(100):
      i.   action = epsilon_greedy(Q, obs, epsilon)
      ii.  next_obs, reward, done, info = env.step(action)
      iii. Bellman update Q[obs][action] using reward + gamma * max(Q[next_obs])
      iv.  Append (obs, action, next_obs.kafka_lag) to lag_buffer
      v.   obs = next_obs
      vi.  if done: break
   c. Decay epsilon: epsilon *= 0.995  (until floor 0.05)
   d. Every 50 episodes: train LagPredictor on lag_buffer for 10 mini-batches
   e. Log mean reward, curriculum_level, blind_spot_triggered count
4. Save Q-table, LagPredictor weights, results/reward_curve.png
```

### What you'll see in the logs

```
[EP 0010] reward=0.32  ε=0.95  curriculum=0  blind_spot_hits=0
[EP 0050] reward=0.41  ε=0.78  curriculum=0  blind_spot_hits=0
[EP 0120] reward=0.51  ε=0.55  curriculum=0  blind_spot_hits=2  ← agent discovers Reject+SkipVerify
[EP 0300] reward=0.62  ε=0.22  curriculum=1  blind_spot_hits=147
[EP 0800] reward=0.69  ε=0.05  curriculum=2  blind_spot_hits=412 ← adversary level rises
[EP 1200] reward=0.66  ε=0.05  curriculum=2  blind_spot_hits=688 ← env got harder, dip
[EP 1800] reward=0.71  ε=0.05  curriculum=2  blind_spot_hits=1023 ← agent adapted, recovered
```

That dip-and-recovery pattern at curriculum transitions IS the staircase. It's the pitch.

### How long?
On a 2-vCPU / 8-GB machine: **~15 minutes for 2000 episodes**. Q-learning + a tiny MLP is cheap.

### Evaluation (separate from training)
After training, the **graders** run the trained agent on each task with a fixed seed (42/43/44), 10 episodes each, and report mean reward. This score is what the leaderboard ranks on.

---

## 10. Reading reward curves

A reward curve plots **mean episode reward over time**. Reading it correctly is half the pitch.

```
reward
1.0 │                                       ╭────────  ← converged
0.8 │                                ╱─╮ ╭─╯
0.6 │                       ╱──────╯  ╰─╯           ← curriculum bumped, env got harder
0.4 │             ╱────────╯                        ← agent learning
0.2 │   ╱────────╯
0.0 │__/_______________________________________
    0    300   600   900   1200  1500  1800  2000  episode
```

### What to point at in the pitch

1. **Initial flat zone (0–100):** Random actions, no learning yet.
2. **First steep rise (100–400):** Agent discovers the obvious wins.
3. **First plateau (400–600):** Easy gains exhausted.
4. **Bump down at curriculum transition:** Adversary leveled up — this is **planned by the env design**, not a bug.
5. **Recovery (600–1200):** Agent adapts to harder dynamics.
6. **Final plateau (1500+):** Convergence.

When you see this shape, you say: *"That staircase is the recursive self-improvement story. The agent improves, the environment escalates, the agent adapts. This is Theme #4 (Self-Improvement) implemented as environment design."*

### Common pathologies (so you know what's wrong)

| Curve shape | What it means | Fix |
|---|---|---|
| Flat at 0.3 forever | No reward signal — bug in reward function | Print `reward_breakdown` per step |
| Rises then crashes to 0 | Catastrophic action exploit (Approve+SkipVerify loop) | Check anti-reward-hacking penalties |
| Wildly oscillating | Learning rate too high | Drop α from 0.1 to 0.05 |
| Slow rise, never converges | ε decaying too fast — not enough exploration early | Stretch decay over more episodes |

---

## 11. Glossary

A flat list you can grep when you forget a word.

- **Action:** what the agent does. AEPO: 6-tuple of discrete choices.
- **Agent:** the decision-maker. AEPO: Q-table (or LLM in GRPO mode).
- **Bellman equation:** the math that defines how Q-values get updated.
- **DQN:** Deep Q-Network. Q-learning with a neural net instead of a table.
- **Discount factor (γ):** how much future rewards matter. AEPO: 0.95.
- **Done:** boolean — episode is over. AEPO: `done=True` on crash, fraud, or step 100.
- **Environment:** the world. AEPO: `UnifiedFintechEnv`.
- **Epoch:** one full pass through training data (used in supervised learning more than RL).
- **Episode:** one full reset-to-done cycle. AEPO: ≤100 steps.
- **ε-greedy:** exploration strategy — random action with prob ε, else argmax.
- **Exploration vs exploitation:** the core RL trade-off. Try new things, or do what worked.
- **Forward pass:** running input through a neural net to get an output.
- **Backward pass / backprop:** computing gradients for weight updates.
- **Gradient:** the partial derivative of loss w.r.t. each weight. Tells us which way to nudge weights.
- **Gradient descent:** "nudge weights opposite to the gradient" — the learning step.
- **GRPO:** Group Relative Policy Optimization. PPO variant for fine-tuning LLMs.
- **Gymnasium:** the standard RL env interface (Python).
- **Heuristic:** a hand-coded baseline policy. AEPO: 3 deliberate blind spots — the trained agent must beat it.
- **Hyperparameter:** a setting you choose, not learn. (α, γ, ε, batch_size, episodes…)
- **Inference:** running a trained model in production (no learning).
- **Learning rate (α):** how big a step we take per update. AEPO: 0.1.
- **Loss function:** how wrong the prediction is. MSE for regression, cross-entropy for classification.
- **MDP (Markov Decision Process):** the formal name for an RL environment.
- **MLP (multi-layer perceptron):** the simplest neural net — stacked linear layers + nonlinearities. The LagPredictor is one.
- **Model-free RL:** Q-learning, PPO. Doesn't learn env dynamics.
- **Model-based RL:** learns dynamics, plans with them. AEPO LagPredictor is a small step in this direction.
- **Observation:** what the agent sees. AEPO: 10-d normalized vector.
- **OpenEnv:** Meta's spec for deploying RL envs (FastAPI + Docker + yaml manifest).
- **Optimizer:** the algorithm that applies gradient updates. Adam, SGD, etc.
- **Policy (π):** the agent's strategy. `π(a|s)` = probability of action `a` in state `s`.
- **PPO:** Proximal Policy Optimization. Modern policy-gradient default.
- **PyTorch:** Python deep-learning library. Tensors + autograd + nn.Module.
- **Q-value:** "expected total future reward if I take action `a` from state `s` and play optimally after."
- **Q-learning:** off-policy tabular RL. The Bellman update we use in AEPO.
- **ReLU:** `max(0, x)`. The default neural-net nonlinearity.
- **Reward:** scalar feedback per step. AEPO: `[0.0, 1.0]`.
- **Reward hacking:** agent finds a degenerate strategy that maximizes reward without doing the real task. CLAUDE.md lists 5 we explicitly defeat.
- **Rollout:** running the policy in the env to collect data.
- **State:** the env's internal description of "where we are." Often used interchangeably with observation.
- **Step:** one (obs → action → reward → next_obs) cycle.
- **Tensor:** PyTorch's multi-d array. Your inputs/outputs/weights are all tensors.
- **TRL:** HuggingFace's "Transformer Reinforcement Learning" library. Used in Option B GRPO mode.
- **Value function (V):** "expected total future reward from state `s` under current policy." Cousin of Q.
- **World model:** a learned function predicting environment dynamics. AEPO LagPredictor.

---

## What to study next (in order of value to your pitch)

1. **Re-read this doc.** Twice. Slowly. Translate every Python snippet into Java in your head.
2. **Read `unified_gateway.py` end to end** with §3 of this doc open. Match each line to a concept.
3. **Run `train.py` once on your laptop.** Watch the logs. See the reward curve shape. Believe it.
4. **Read the Bellman equation in §5 until you can re-derive it from memory.** The judges WILL ask.
5. **Memorize the pitch narrative in CLAUDE.md.** Practice saying "blind spot #1" out loud.

Resources, if you want more depth (optional):
- *Sutton & Barto, Reinforcement Learning: An Introduction* — chapters 1–6 cover everything in this doc, in textbook form. Free PDF online.
- *Karpathy's "A Recipe for Training Neural Networks"* blog post — the best practical PyTorch debugging guide.
- *Hugging Face Deep RL Course* — free, hands-on, takes a weekend.

---

*If anything in this doc contradicts CLAUDE.md, CLAUDE.md wins. Update this doc when specs change — never let it go stale.*
