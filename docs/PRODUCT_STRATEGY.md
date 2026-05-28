# AEPO — Product & Architecture Strategy

> **From hackathon prototype to B2B enterprise SaaS**
> Author: Strategy synthesis for Umesh Maurya (SDE 2, Java/Spring/Kafka)
> Date: 2026-05-07
> Audience: Founder-engineer planning a 12-week PoC with line-of-sight to a fundable V1.

---

## Executive summary (TL;DR)

The hackathon AEPO is a *causally-structured simulation* in which an RL agent learns the Asymmetric Risk Triad (Fraud × Infra × SLA). The production thesis is the inverse: **take the policies the agent learns in simulation and run them inline on real UPI traffic, while the simulator becomes the offline training rig and a continuous "what-if" box for SRE/risk teams.**

Three things make this a real product, not a science fair:

1. **Inline RL routing on UPI** is greenfield. Hyperswitch ships rule-based and "ML routing" but no public RL deployment; Razorpay Optimizer is single-PA closed-loop; Stripe Adaptive Acceptance is *card-network* retry — none of them touch the **fraud × infra × HSM compute** triad on UPI rails. ([Hyperswitch](https://hyperswitch.io/), [Stripe Adaptive Acceptance](https://stripe.com/blog/ai-enhancements-to-adaptive-acceptance))
2. **The "Blind Spot" — decide-then-verify** — is patent-defensible as a *system*. HSMs cost $40k+ each and run PBKDF2-HMAC-SHA256 with 600,000 iterations per UPI PIN verification. Skipping crypto on transactions you've already decided to reject saves real money on real silicon. ([Akshansh Jaiswal — UPI architecture](https://blog.akshanshjaiswal.com/the-upi-architecture-a-security-look))
3. **AI SRE is a $3B+ market with a credibility gap.** Datadog Bits AI shipped Dec 2025 and explains incidents *after* they happen. AEPO *acts* before they happen. 43% of SRE orgs report **more** toil despite the tool flood — the diagnosis market is saturated; the *autonomous mitigation* market is empty. ([Datadog Bits AI SRE](https://www.datadoghq.com/product/ai/bits-ai-sre/), [incident.io 2025 SRE Report](https://incident.io/blog/sre-ai-tools-transform-devops-2025))

The 3-month PoC produces: a Java/Spring/Kafka microservice mesh that processes dummy UPI traffic at 1k TPS, embeds the LagPredictor as an ONNX model inside the JVM hot path, swaps routing policy live based on Kafka lag, and shows a measurable P99 delta vs a static rule baseline.

---

## 1. Market Gap & Product Vision (the PM angle)

### 1.1 Where current payment switches break

Static rule engines fail in three predictable ways:

**a) They can't price the cost of their own decisions.**
A rule that says *"if risk_score > 80, do FullVerify"* never asks "is the HSM saturated right now?" During a botnet flood, every FullVerify on a transaction you'll reject anyway burns 600k PBKDF2 iterations of HSM compute. That compute has a fixed budget. When it saturates, the legitimate Approve path queues behind the malicious Reject path. UPI processed 16.73B transactions in December 2024 alone — bursty, peak-of-peak traffic is the only mode that matters. ([Coingeek — UPI saturation](https://coingeek.com/will-upi-exponential-growth-lead-to-saturation-in-2025/))

**b) They cascade silently.**
NPCI targets <1% technical decline; Baroda UP Bank hit **7.26% TD**. The April 1, 2026 outage hit SBI/HDFC/ICICI simultaneously because their CBS/UPI middleware shares Kafka backbone — when one bank's lag explodes, downstream rolling-P99 widening triggers others' circuit breakers, and the failure becomes systemic. ([PingTV — April 2026 outage](https://www.pingtvindia.com/upi-server-down-failed-transactions-april-2026/), [Daily Jagran — multi-bank outage](https://www.thedailyjagran.com/technology/hdfc-bank-server-down-today-status-bank-of-baroda-users-cry-upi-transactions-error-gpay-phonepe-news-x-twitter-latest-live-updates-10131966))

**c) They can't talk to each other.**
Fraud teams optimize Reject rate. SRE teams optimize P99. The two teams sit in different orgs, on different on-call rotations, with different dashboards. A rejection is a "win" for fraud and a "Kafka slot consumed" for SRE — and no one owns the joint metric. The 2025 SRE Report calls this directly: "poor integration between tools creating data silos … operational toil increased for 43% of organizations." ([incident.io](https://incident.io/blog/sre-ai-tools-transform-devops-2025))

### 1.2 Target audience (ranked by pain × budget × buying speed)

| Segment | Pain level | Annual budget for routing/SRE | Sales cycle | Land deal target |
|---|---|---|---|---|
| **Mid-size Payment Aggregators** (Cashfree, Easebuzz tier) | Severe — every basis point of acceptance is margin | ₹5–15 cr | 3–6 months | ₹40 lakh ARR pilot |
| **NBFCs running their own PA-P** | High — RBI 2025 mandate forces them onto compliant infra by Sept 2026 | ₹10–25 cr | 6–9 months | ₹80 lakh ARR |
| **Tier-2/3 banks running UPI-PSP** (RBL, IDFC First, Federal) | High — their TD rates are visible to NPCI, valuation risk | ₹30–80 cr | 9–18 months | ₹2–4 cr ARR |
| **Cross-border players** (Wise/Remitly entering India) | Medium — they want one vendor, not a stack | 6–12 months | ₹1–2 cr ARR |
| **Tier-1 banks** (HDFC, ICICI, SBI) | Severe but politically slow | Unlimited | 18–36 months | Don't chase first |
| **Razorpay/Pine Labs/Pluxee** | They build their own — sell the *Compute Optimizer* module only | — | 6–12 months | Component licensing |

**Beachhead recommendation:** mid-size PAs and NBFCs with PA-P licenses. They have RBI compliance deadlines (escrow audits, CERT-In annual reviews, FIU registration — all by 15 Sept 2026), engineering teams of 30–60, no in-house ML, and they cannot afford a Hyperswitch-level fork. ([RBI Master Direction 2025](https://www.fidcindia.org.in/wp-content/uploads/2025/09/RBI-PAYMENT-AGGREGATORS-DIRECTIONS-15-09-25.pdf), [Lexplosion — PA-P deadline](https://lexplosion.in/rbi-master-direction-2025-compliance-mandate-for-payment-aggregators-and-pa-p-deadline/))

### 1.3 UVP — "from simulation to live dynamic router"

The hackathon's framing is *simulation*. The product framing is *closed-loop control*. The shift looks like this:

| Hackathon framing | Product framing |
|---|---|
| Agent learns routing in 100-step episodes | Agent learns from 24h windows of production telemetry |
| Reward shapes behavior | Acceptance rate × P99 × HSM-cost-per-approved-txn shapes behavior |
| Adversary escalates inside env | Real botnets and outages do the escalation |
| LagPredictor is a "world model" claim | LagPredictor is a 5-step-ahead routing input that prevents 60% of cascading lag failures |
| Output: reward curve, OpenEnv compliance | Output: live policy table, explainable decisions, regulatory audit trail |

**The one-line UVP:**
> *AEPO is the only payment orchestrator where your fraud, infra, and crypto-compute teams optimize a single shared objective — automatically, in production, with 5-step lookahead and a regulatory-grade audit trail of every decision.*

Stripe's Adaptive Acceptance shows the precedent: AI-driven payment optimization yields **+3.8% acceptance rate** and **−3.3% interchange cost** in production, on cards. UPI is bigger, more concentrated (NPCI is the single switch), and has *no incumbent doing this*. ([Stripe Authorization Boost](https://stripe.com/payments/authorization))

---

## 2. Core Product Features (from hackathon to enterprise SaaS)

Package the platform as **four sellable modules**, with the first as the wedge.

### 2.1 Module 1 — AEPO Live Router *(the wedge)*

The inline routing brain. Replaces or augments a customer's existing rule engine.

- **Inputs:** 10-signal observation vector (fraud, infra, business layers) computed in real time per transaction.
- **Outputs:** the 6-action decision vector (`risk_decision`, `crypto_verify`, `infra_routing`, `db_retry_policy`, `settlement_policy`, `app_priority`).
- **Mode 1 — Shadow:** customer keeps their current rules; AEPO predicts in parallel. Output is logged, not applied. (Day-1 onboarding mode — the "trust ladder.")
- **Mode 2 — Suggest:** AEPO returns a recommendation; customer's rule engine has final say. (Mid-onboarding — risk teams stay in the loop.)
- **Mode 3 — Inline:** AEPO is the rule engine. Customer's policy team uses Studio (Module 4) for governance.
- **Pricing:** 0.5–1.5 bps on routed TPV, with a floor (₹2L/month). Mirrors Hyperswitch commercial tier and FlyCode-style orchestration pricing. ([FlyCode — orchestration pricing trends](https://www.flycode.com/blog/smart-payment-orchestration-from-simple-rules-to-ai-unlocking-failed-payment-boost-with-multi-processor-strategy))

### 2.2 Module 2 — AEPO Compute Optimizer *(the crown jewel)*

This is where the "Blind Spot" becomes a product.

**The mechanism, productized:**
> *Decide-then-verify scheduling for HSM workloads. The router commits to a routing decision (Approve/Reject/Challenge) using cheap signal features, then schedules cryptographic verification only when the committed decision requires it. Reject + SkipVerify is the dominant strategy on high-risk traffic — it saves 250 lag units per step in our env, and ~80–150ms of HSM round-trip in production.*

**Why this is a real product and not a clever trick:**
- HSMs cost $40k+ each; PBKDF2-HMAC-SHA256 at 600k iterations is the dominant cycle cost on UPI PIN verify. ([Akshansh Jaiswal](https://blog.akshanshjaiswal.com/the-upi-architecture-a-security-look))
- The HSM market is $3.73B (2024) → $7.22B (2033), 7.6% CAGR — customers are budget-constrained. ([Straits Research — HSM market](https://straitsresearch.com/report/hardware-security-modules-market))
- A 30% reduction in HSM operations during peak load = direct OpEx + capex deferral.

**IP angle (read this part carefully):**
- **Don't try to patent the *idea*** of skipping crypto on rejected transactions. Software/business-method patents are weak in India and would be obvious-over-prior-art in the US.
- **Do patent the *system*:** "A method and system for risk-aware cryptographic resource scheduling in distributed payment networks, comprising (a) a learned policy that produces a tentative authorization decision from a feature vector excluding cryptographic verification output, (b) a deterministic gate that maps the tentative decision to a verification requirement, (c) an HSM scheduler that prioritizes verification operations by transaction-level expected utility under bounded compute capacity, and (d) an audit subsystem cryptographically committing to (a)-(c) for regulatory replay."
- File a provisional in India and a PCT. Talk to a fintech-IP firm; budget ₹4–8 lakh for filing. The defensibility comes from the **scheduler + audit commitment** combo, not from the high-level idea.
- **Trademark "Compute Optimizer"** as a product name; it's a clean B2B handle.

**Pricing:** Revenue share — 40-50% of the measured HSM cost reduction (CapEx deferral + cloud-HSM bill). Independent quarterly audit. Customers love rev-share because risk shifts to AEPO; AEPO loves it because it scales with customer success.

### 2.3 Module 3 — AEPO SRE Copilot *(the dashboard play)*

The "single pane of glass" that the 2025 SRE Report says nobody has. ([incident.io](https://incident.io/blog/sre-ai-tools-transform-devops-2025))

- Real-time visualization of: predicted vs actual Kafka lag, routing decisions per rail, HSM utilization, P99 by merchant tier, blind-spot triggers per hour.
- LLM-driven incident assistant — not a Bits AI clone. AEPO already *acts*; the assistant explains the actions: *"Why did we throttle UPI rail at 14:32?"* → *"Predicted lag was 3,847 (LagPredictor confidence 0.91), bank_api_status=Degraded for HDFC, current rolling_p99=712ms approaching 800ms breach. Diverted 30% of traffic to Credit rail for 4 minutes. Recovered to baseline at 14:36."*
- WebSocket live feed; Grafana embed for the metric layer; React shell for the chat/explain UI.
- **Pricing:** flat per-seat ($150/seat/month) bundled into Live Router contract.

### 2.4 Module 4 — AEPO Policy Studio *(the governance play)*

Risk and compliance teams will *not* let an RL agent decide unsupervised. This module solves that.

- No-code rule editor (mirrors customer's existing rule engine semantics, so migration is paste-in).
- Side-by-side: "your rule" vs "AEPO recommendation" with reward delta and cost impact.
- "Pin" overrides — risk team can pin an action under specific conditions (e.g., always FullVerify when transaction > ₹50,000), and the RL policy is constrained to respect pins during training and inference.
- Approval workflow: when the RL agent proposes a new policy version, it goes to a human for sign-off. Trail is immutable, regulator-exportable.
- **Pricing:** included in Live Router; no separate SKU. It's the trust mechanism that unblocks the upsell to Inline mode.

### 2.5 Why these four and not more

A solo engineer cannot build seven products. These four are picked because:
- Module 1 is the wedge — without it, modules 2–4 have no inline data.
- Module 2 has the highest defensibility (HSM cost lever, IP angle).
- Module 3 is the upsell that justifies the seat-based ARR layer.
- Module 4 is the *blocker remover* — risk/compliance teams will say no without it.

---

## 3. Production Microservices Architecture (the architect angle)

### 3.1 Design principles (non-negotiable)

1. **Hot path stays Java/synchronous, ≤20ms p99 added latency.** UPI's NPCI-side budget is ~30s end-to-end, but practical bank-side budgets are 200–500ms. AEPO must add no more than 20ms or it doesn't ship.
2. **AI never blocks the hot path.** The RL inference happens *out-of-band* — it updates a policy table that the hot path reads from local cache.
3. **Every decision is replayable.** Cryptographically chained audit log (Merkle-tree commits per minute), regulator-exportable, with the exact policy version, feature vector, and decision rationale.
4. **Multi-tenant isolation is structural, not configurable.** One customer's policy and feature data can never be in the same Redis namespace as another's.
5. **No Python on the hot path.** ONNX Runtime in JVM only. Python is allowed in offline training, batch jobs, and the World Model service if needed for prototyping.

### 3.2 Service map (target V1, 8 services)

```
                          ┌────────────────────────────────────────┐
                          │      Spring Cloud Gateway              │
                          │   (TLS, mTLS, rate limit, idem-keys)   │
                          └────────────────┬───────────────────────┘
                                           │ HTTP/gRPC
                          ┌────────────────▼───────────────────────┐
                          │      AEPO Orchestrator Service         │  ← HOT PATH
                          │  - Caffeine cache: policy, features    │     (Java 21,
                          │  - ONNX in-process: LagPredictor       │     Spring Boot 3,
                          │  - 6-action decision in <8ms           │     Virtual Threads)
                          └────┬───────┬───────┬───────────┬───────┘
                               │       │       │           │
                               │       │       │           │ async event (Kafka)
              ┌────────────────▼┐  ┌───▼────┐  │  ┌────────▼─────────┐
              │ Crypto Gate Svc │  │ Bank   │  │  │ Audit Service    │
              │ (HSM scheduler, │  │ Adapter│  │  │ (Postgres + S3,  │
              │ decide-then-    │  │ Svc    │  │  │  Merkle commits) │
              │ verify policy)  │  │(NPCI/  │  │  └──────────────────┘
              └─────────────────┘  │ HDFC/  │  │
                                   │ SBI…)  │  │
                                   └────────┘  │
                                               │
              ┌────────────────────────────────▼────────────────────┐
              │  Kafka — `aepo.tx.events`, `aepo.policy.updates`,   │  ← BACKBONE
              │          `aepo.feature.vectors`, `aepo.audit.log`   │
              └─────┬───────────────────────────┬──────────────┬────┘
                    │                           │              │
        ┌───────────▼─────────────┐   ┌─────────▼────────┐   ┌─▼──────────────┐
        │ Telemetry Aggregator    │   │ Policy Engine    │   │ Replay Service │
        │ (Spring Boot Streams)   │   │ Service (Java)   │   │ (Python/Ray,   │
        │ - 10-signal feature     │   │ - reads features │   │  offline RL    │
        │   vector per tx         │   │ - calls RL Infer │   │  training)     │
        │ - writes Feast/Redis    │   │ - writes Redis   │   └────────────────┘
        └─────────────────────────┘   │   policy table   │
                                      └────────┬─────────┘
                                               │
                                      ┌────────▼─────────┐
                                      │ RL Inference Svc │
                                      │ (Java + ONNX or  │
                                      │  Python + Triton │
                                      │  for heavy nets) │
                                      └──────────────────┘
```

### 3.3 Service responsibilities (the architect's PRD)

#### **Spring Cloud Gateway** *(Java)*
- TLS termination, mTLS to bank rails, idempotency-key dedup (Redis-backed, 24h window).
- Rate limit per merchant, per rail, with token-bucket. Lyft's Envoy ratelimit-style for fairness.
- Trace context injection (W3C tracecontext), per-request budget timer (kills downstream calls at deadline).

#### **AEPO Orchestrator Service** *(Java, hot path)*
- The *only* synchronous service. Receives the transaction, fetches the feature vector (Redis, sub-1ms), looks up the current policy version (Caffeine, 0ms), runs the LagPredictor ONNX model in-process (~3–5ms on warm CPU), produces the 6-action decision, dispatches to Crypto Gate and Bank Adapter.
- Virtual threads (Java 21 Loom) for fan-out to Crypto Gate + Bank Adapter; async I/O end-to-end.
- Stateless. Horizontally scalable. Pod restart cost: ONNX warm-up (~200ms on first request — pre-warm in readiness probe).
- Exposes `/route` (the main API), `/decisions/{id}` (replay), `/health`.

#### **Crypto Gate Service** *(Java)*
- Wraps the customer's HSM (or cloud-HSM — Futurex VirtuCrypt, AWS CloudHSM, Azure Dedicated HSM).
- Implements decide-then-verify: receives `(decision, transaction)` and skips PBKDF2 if the decision is Reject. ([Futurex VirtuCrypt](https://www.futurex.com/products/cloud-products/virtucrypt-cloud-payment-hsm))
- HSM scheduler: priority queue keyed by transaction expected utility. Under saturation, low-utility verifications drop to a shadow queue with degraded latency SLA.
- Emits `aepo.crypto.metrics` to Kafka — feeds the Compute Optimizer dashboard.

#### **Bank Adapter Service** *(Java, one pod per major bank or NPCI rail)*
- NPCI UPI client, HDFC client, SBI client, etc. Each rail has different connection pool, retry semantics, latency profile.
- Circuit breaker (Resilience4j) per rail. State exported to Kafka for the Orchestrator's policy engine to react to.
- Reuses Umesh's existing Kafka and Crypto libraries here. **This is the leverage point — a battle-tested NPCI/UPI client is a moat in itself.**

#### **Telemetry Aggregator** *(Spring Boot Kafka Streams)*
- Subscribes to `aepo.tx.events`, computes the 10-signal vector with EMA aliases (rolling p99, kafka_lag, etc.) in 1s tumbling windows.
- Writes to Feast online store (Redis backend) — sub-50ms freshness; reads also from offline store (S3 parquet) for training. ([Redis — feature store benchmarks](https://redis.io/blog/feature-stores-for-real-time-artificial-intelligence-and-machine-learning/), [Feast docs](https://docs.feast.dev))
- Backpressure: if Feast write lags > 500ms, switch to direct Redis write and publish lag alarm.

#### **Policy Engine Service** *(Java)*
- The *brain's bridge to the hot path*. Every 30s (configurable), fetches the latest feature distribution from Feast, calls the RL Inference Service, writes the resulting policy snapshot to Redis under `policy:{tenant}:{version}`.
- Versioned policies: every change is a new version, old versions retained for replay.
- Hot reload: Orchestrator subscribes to `aepo.policy.updates` Kafka topic, swaps Caffeine cache atomically.
- **Failure mode (critical):** if RL Inference returns an unsafe policy (defined by validation harness), Policy Engine falls back to the last known-good policy and pages on-call.

#### **RL Inference Service** *(Java + ONNX, or Python + Triton if model needs it)*
- For Q-table policies (your hackathon path): pure Java, no inference framework needed — it's a hashmap lookup. <1ms.
- For neural policies (Phase 2 productization): ONNX Runtime in JVM if model fits (most policies will). ([ONNX Runtime Java guide](https://onnxruntime.ai/docs/get-started/with-java.html), [InfoQ — ONNX in Java enterprise](https://www.infoq.com/articles/onnx-ai-inference-with-java/))
- For very large models: NVIDIA Triton or TorchServe behind gRPC. **Avoid this for V1** — keep in JVM.

#### **Audit Service** *(Java + Postgres + S3)*
- Append-only ledger. Every decision: input features (hashed), policy version, action chosen, downstream outcomes (approved/rejected/timeout), HSM ops triggered, latency.
- Per-minute Merkle-tree commit; root hash anchored to a customer-controlled write target (their own audit DB, or S3 with object lock).
- Regulatory export — RBI Cyber Resilience Directions 2024 require this; CERT-In annual audits will demand it. ([RBI Master Direction 2025](https://www.fidcindia.org.in/wp-content/uploads/2025/09/RBI-PAYMENT-AGGREGATORS-DIRECTIONS-15-09-25.pdf))

#### **Replay & Training Service** *(Python)*
- Reads audit logs + features into a training dataset; runs offline RL (D3RLPy or RLlib) overnight on customer-isolated GPU pods.
- Outputs ONNX-exported policy candidates → goes to Policy Studio for human approval before promotion.
- The hackathon environment lives here — *as a simulator for what-if analysis and counterfactual evaluation.*

### 3.4 The hot-path latency budget (the only number that matters)

| Stage | Budget | Notes |
|---|---|---|
| Gateway parse + auth | 3ms | Spring Cloud Gateway, no per-request DB |
| Feature fetch (Redis) | 1ms | Pre-warmed, pipelined Lettuce |
| Policy lookup (Caffeine) | 0.1ms | In-process |
| LagPredictor ONNX inference | 4ms | 16→64→1 MLP on CPU, JIT-warm |
| Decision computation | 1ms | Pure Java |
| Crypto Gate dispatch | 2ms | Decide-then-verify |
| HSM op (only if needed) | 80–150ms | Skipped for ~30% of high-risk Rejects |
| Bank dispatch | 100–300ms | NPCI is the bottleneck; AEPO can't fix this |
| **Total AEPO-added (excl. HSM, bank)** | **~12ms p50, 18ms p99** | |

**This budget is achievable only if** ONNX is in-process. A Python sidecar over gRPC adds 5–8ms minimum and breaks the p99 under contention.

### 3.5 Sidecar pattern — when and where

I'd *not* use a literal Envoy/Istio sidecar for the inference path — too much hop overhead. The pattern that fits AEPO is **"in-process model + sidecar telemetry"**:
- Inference: in-JVM ONNX (no sidecar).
- Telemetry: a thin sidecar (or a daemonset) that scrapes the Orchestrator's Prometheus endpoint and forwards to the central observability stack. Doesn't touch the hot path.
- Service mesh (Istio + Envoy) handles mTLS, retry, traffic shifting between policy versions (canary new policies to 5% of traffic before full promote).

### 3.6 How the AI/RL component never adds latency (the trick)

The trick is **separation of policy training and policy execution**:
- **Training** (offline, async, slow) produces a policy table or small ONNX model.
- **Execution** (online, sync, fast) is a lookup or a tiny inference call.

Concretely:
- Q-table policy: a `Map<DiscretizedState, Action>`. Hot path does discretization (5 lines of code) + lookup. <1ms.
- ONNX MLP policy: 4-input, 6-output network. CPU inference <5ms warm.
- **The RL agent's "thinking" happens in the Replay & Training Service**, runs nightly, takes 20 minutes on a single GPU box. The hot path never sees a training loop.

This is the same pattern Stripe uses for Adaptive Acceptance: the AI updates retry tables out-of-band; the hot path applies the table. ([Stripe AI enhancements blog](https://stripe.com/blog/ai-enhancements-to-adaptive-acceptance))

---

## 4. Technology Stack & Infrastructure

You said "infrastructure-agnostic" — so this is in two layers: (a) the application stack you commit to, and (b) the infra primitives you abstract over.

### 4.1 Application stack (commit to these)

| Layer | Choice | Why |
|---|---|---|
| **Language (hot path)** | Java 21 (Loom virtual threads) | Your strength, mature, virtual threads make per-tx fan-out cheap |
| **Framework** | Spring Boot 3.3 | Industry default for Indian fintech; ecosystem alignment |
| **API Gateway** | Spring Cloud Gateway | Stays in JVM, less operational surface than Kong; reconsider Kong only if multi-language services emerge |
| **Service mesh** | Istio + Envoy | Standard. mTLS for free, traffic-shifting for canaries |
| **Cache** | Redis Cluster (Lettuce client) | Sub-ms latency for features and policy table |
| **Message bus** | Apache Kafka | You already have a library — leverage it |
| **DB (transactional)** | PostgreSQL 16 (Patroni HA) for audit; **CockroachDB or TiDB** for transaction state when multi-region | Postgres for what fits in a single region; distributed SQL when geo-distribution is forced |
| **Time-series** | Prometheus + Thanos (long retention) | Standard; integrates with Grafana cleanly |
| **Tracing** | OpenTelemetry → Tempo or Jaeger | Required for replay and incident analysis |
| **Logs** | Loki | Cheap, Grafana-native; ELK if customer demands it |
| **ML serving (small)** | ONNX Runtime in JVM | Sub-5ms inference, no extra ops surface |
| **ML serving (large)** | NVIDIA Triton (gRPC) — *only if needed* | Don't reach for this in V1 |
| **ML training** | Python + Ray + PyTorch (offline only) | Industry standard; the existing AEPO sim becomes the training environment |
| **Feature store** | Feast + Redis online | ([Feast docs](https://docs.feast.dev)) |
| **Container** | Docker + Kubernetes | Default. EKS/GKE/AKS depending on customer; abstract via Helm + ArgoCD |
| **IaC** | Terraform (cloud) + Helm (K8s) | Cloud-portable; per-customer values.yaml |
| **CI/CD** | GitHub Actions → ArgoCD | GitOps; every policy promotion is a Git commit, signed |
| **Secrets** | HashiCorp Vault (or cloud-native KMS) | Crypto keys never leave HSM; Vault holds API tokens, DB passwords |
| **Feature flags** | Unleash (self-hosted) | For canary policy promotion + A/B mode toggles |

### 4.2 Where to lean on what you already have

- **Your Kafka library:** plug it into Telemetry Aggregator and Bank Adapter. Don't rewrite.
- **Your Crypto library:** wrap it inside Crypto Gate Service. The HSM-scheduling logic sits *on top* of it.
- **The hackathon `unified_gateway.py`:** becomes the *simulator* under Replay & Training. Keep it, don't port it. Java doesn't need to mirror it in production — it only needs to mirror the policy interface.

### 4.3 ML serving — concrete recommendation

For the LagPredictor (16→64→1 MLP, ~5KB model), the answer is **ONNX in-process Java**. Specifically:

```java
// In Orchestrator startup
private final OrtEnvironment env = OrtEnvironment.getEnvironment();
private final OrtSession session = env.createSession("models/lag_predictor_v1.onnx", opts);

// In hot path
public float predictLag(float[] features) {
    try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(features), new long[]{1, 16});
         OrtSession.Result out = session.run(Map.of("input", input))) {
        return ((float[][]) out.get(0).getValue())[0][0];
    }
}
```

This is what InfoQ recommends for low-latency enterprise inference: *"Embedding this logic directly into the Java service, rather than relying on a Python-based microservice, reduces latency and avoids fragile infrastructure dependencies."* ([InfoQ — ONNX in Java](https://www.infoq.com/articles/onnx-ai-inference-with-java/))

For the RL policy itself (Q-table or small NN), same approach. Don't reach for Triton/TorchServe until the model is genuinely large (>100MB) or needs GPU.

---

## 5. Execution Roadmap — 12 Weeks, Solo SDE 2

The constraint: you're one person, working ~20–25 hours per week on this, with a Java/Spring/Kafka background and no full-time ML ops support. The PoC must demonstrate **dynamic routing under simulated Kafka lag**, end-to-end, with a measurable delta vs static baseline.

### Month 1 — Skeleton (Weeks 1–4)

**Goal at end of month: a transaction goes from Gateway → Orchestrator → Bank Adapter, audit-logged, observable.**

| Week | Deliverable | Acceptance |
|---|---|---|
| 1 | Gradle multi-module monorepo. Spring Boot 3.3 + Java 21. Docker Compose with Postgres, Redis, Kafka, Prometheus, Grafana. CI on GitHub Actions. | `./gradlew build` green; compose up brings everything healthy |
| 2 | Spring Cloud Gateway service. `POST /v1/payment` accepts a fake UPI payload, returns 200 with a request-id. mTLS scaffolding (self-signed). | k6 load test: 1000 RPS, p99 < 30ms |
| 3 | Orchestrator service with **static** routing logic (just sends to Mock Bank Adapter). Audit log to Postgres (request, decision, response). Idempotency keys in Redis. | A 100-tx batch produces 100 audit rows; replay endpoint returns the original decision |
| 4 | Mock Bank Adapter (1 rail) with configurable latency. Telemetry Aggregator skeleton: consumes `aepo.tx.events`, computes rolling p99 + Kafka lag, writes to Redis under `features:{tenant}:latest`. Grafana dashboard with these two metrics. | Load test produces visible spike on dashboard when adapter latency is dialed up |

**Anti-scope creep:** no Crypto Gate yet. No real HSM. No RL. No Policy Studio. Just the spine.

### Month 2 — Intelligence (Weeks 5–8)

**Goal at end of month: routing decision changes dynamically based on Kafka lag, with the LagPredictor making the call.**

| Week | Deliverable | Acceptance |
|---|---|---|
| 5 | Policy Engine service. Reads features from Redis, applies a *static rule set* (just `if predicted_lag > 3000 → throttle`), writes to `policy:{tenant}:vN` in Redis. Orchestrator picks up via Kafka topic + Caffeine cache. | Policy version increments visible in dashboard; Orchestrator hot-reload latency < 100ms |
| 6 | Port LagPredictor PyTorch → ONNX. Embed in Orchestrator JVM. Replace the static `kafka_lag` reading with `LagPredictor.predict()`. | ONNX inference p99 < 6ms; predicted-vs-actual lag chart in Grafana |
| 7 | Mock botnet traffic generator. Reuses the AEPO simulation env to drive `aepo.tx.events` with realistic phase patterns (Normal/Spike/Attack/Recovery). | Visible Kafka lag burst followed by routing decision flip on dashboard |
| 8 | Crypto Gate service v0. Mock HSM client with configurable latency (80ms typical). Decide-then-verify: skip HSM if `decision == REJECT`. Emit `aepo.crypto.metrics`. | Dashboard shows HSM ops/sec drop ~30% during high-risk-traffic phases |

### Month 3 — Demo (Weeks 9–12)

**Goal at end of month: an investor-ready demo showing measurable P99 improvement vs static baseline, with the AI SRE Copilot dashboard live.**

| Week | Deliverable | Acceptance |
|---|---|---|
| 9 | Trained Q-table policy (from your hackathon `train.py`) exported to JSON, loaded by Policy Engine. RL-driven routing replaces static rules. | Shadow-mode comparison: RL vs static, on identical traffic, side-by-side reward/p99 |
| 10 | AI SRE Copilot dashboard v1 — React shell + Grafana embeds + WebSocket live decision feed. LLM explainer (OpenAI / Anthropic / local) for "why this decision?" | Click any decision → get a 3-sentence explanation citing the feature vector |
| 11 | Compute Optimizer module v1 — measure HSM ops avoided per hour, project monthly cost saving. Audit cryptographically chained (Merkle root every minute). | Saving figure in dashboard + a CSV regulatory export |
| 12 | Stress test + demo recording. Static baseline vs AEPO under 5x traffic burst, 50% high-risk skew. Demo video. README + architecture diagrams. Pitch deck v1. | Static p99 ~400ms; AEPO p99 ~90ms; HSM ops reduction 30%+; demo video <8 minutes |

### 5.1 Weekly Friday ritual (single-engineer discipline)

- 30 min: write a 5-line update — what shipped, what didn't, the single biggest unknown for next week.
- 30 min: cut next week's scope to *the smallest thing that proves something*. Resist building the "right" thing — build the next thing that disproves a critical assumption.
- The CLAUDE.md "no half-finished implementations" rule applies double here.

### 5.2 What you should *not* build in 12 weeks

- Multi-tenant pod-level isolation (one tenant per cluster is fine for PoC; do this in V1.5)
- Real HSM integration (mock with realistic latency; integrate at design partner stage)
- Policy Studio UI (a Postgres table + a Java admin endpoint is enough for PoC)
- Cross-region replication (single region for PoC)
- Customer onboarding portal (manual provisioning is acceptable for design partners)
- A second bank adapter rail (one mock rail is fine; expand at design partner stage)

### 5.3 Post-PoC: getting to first design partner

Once weeks 1–12 are done:
- **Week 13–16:** Pick 1 design partner (mid-size PA from §1.2). Run AEPO in *Shadow* mode against their staging traffic. Measure shadow accuracy vs their current rules.
- **Week 17–24:** Move to *Suggest* mode in production, low-volume merchants only. Measure: acceptance rate, P99, HSM cost. Publish results with the partner's permission.
- **Week 25–36:** Inline mode for 1 segment. First commercial contract.

---

## 6. Risks & open strategic questions (the Staff Engineer flags)

These are the things a Meta Staff Engineer reviewing this strategy would push back on. Don't ignore them.

1. **Cold-start of the RL policy.** Day-1 customers don't have months of audit data to train on. Mitigation: ship with a "transfer policy" trained on the AEPO simulator that produces sane defaults; let it personalize over the first 30 days in shadow mode.
2. **Decide-then-verify is regulatorily sensitive.** RBI's Master Direction 2025 mandates strong KYC and security controls; skipping crypto on rejects must be defensible to NPCI and CERT-In auditors. ([RBI MD 2025](https://www.fidcindia.org.in/wp-content/uploads/2025/09/RBI-PAYMENT-AGGREGATORS-DIRECTIONS-15-09-25.pdf)) **Engage compliance counsel before the first design partner.** The "we never approve without verify; we only skip *on rejects*" framing is the right one — but get an opinion in writing.
3. **Multi-tenant policy contamination.** If Tenant A's policy ever influences Tenant B's decisions (shared training data, shared feature normalization), it's a regulatory and competitive disaster. Solve this *structurally* — separate Redis namespaces, separate training jobs, separate ONNX models per tenant. No shortcuts.
4. **Hot-path failure mode.** What happens if Policy Engine writes a corrupt policy? Need a validation harness: every new policy is dry-run on a held-out trace and must beat the previous policy by ≥0 reward before promotion. If validation fails, freeze policy and page on-call.
5. **Vendor lock-in for the customer.** Customers will balk at routing 100% of traffic through AEPO unless there's a sane bypass. Ship a "kill switch" — a single Redis flag that reverts Orchestrator to pass-through mode. Document the SLA on this switch.
6. **Hyperswitch is now open-source and gaining traction.** They will commoditize the orchestration layer. AEPO's defense is the **RL + HSM compute layer** — be explicit that we are not competing with Hyperswitch for "PSP routing"; we are layering above their routing with risk + infra optimization. ([Juspay open-sources Hyperswitch](https://yourstory.com/2025/03/juspay-open-sources-payment-orchestrator-fintech-industry-shakeup-phonepe-razorpay))
7. **You are one person.** Burnout risk is the single largest project risk. The roadmap above is aggressive — if a week slips, cut scope, don't extend hours. Solo SDE projects die from sustained over-allocation, not from missed weeks.

---

## 7. The 90-day pitch deck — slide titles only

Use this as a writing prompt, not the final deck.

1. The Asymmetric Risk Triad — fraud, infra, and HSM compute optimize against each other today
2. UPI processed 16.73B txns in Dec 2024 — and TD rates are *rising* at peak banks
3. Existing orchestrators do routing. Existing AI SRE tools do explanation. Nobody does **closed-loop control**.
4. AEPO Live Router — the inline brain (demo: routing decision under load)
5. AEPO Compute Optimizer — decide-then-verify saves HSM dollars (demo: HSM ops/sec graph)
6. AEPO SRE Copilot — the dashboard your fraud + SRE teams have been asking for
7. Architecture — Java/Spring/Kafka hot path + Python offline RL + ONNX in JVM
8. Defensibility — IP filing on Compute Optimizer; data flywheel; integration moat
9. Customer #1: design partner pipeline (3 mid-size PAs in late-stage convo)
10. The team (you + 2 future hires); the ask; the milestones

---

## Sources

- [NPCI — UPI ecosystem statistics](https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics)
- [Hyperswitch — open-source payment orchestrator](https://hyperswitch.io/)
- [Juspay open-sources Hyperswitch — YourStory](https://yourstory.com/2025/03/juspay-open-sources-payment-orchestrator-fintech-industry-shakeup-phonepe-razorpay)
- [Stripe — AI enhancements to Adaptive Acceptance](https://stripe.com/blog/ai-enhancements-to-adaptive-acceptance)
- [Stripe Authorization Boost — increases acceptance 3.8%](https://stripe.com/payments/authorization)
- [FlyCode — smart payment orchestration trends 2025](https://www.flycode.com/blog/smart-payment-orchestration-from-simple-rules-to-ai-unlocking-failed-payment-boost-with-multi-processor-strategy)
- [Datadog Bits AI SRE — launched Dec 2025](https://www.datadoghq.com/product/ai/bits-ai-sre/)
- [incident.io — 5 AI-powered SRE tools 2025](https://incident.io/blog/sre-ai-tools-transform-devops-2025)
- [PingTV — UPI April 2026 outage report](https://www.pingtvindia.com/upi-server-down-failed-transactions-april-2026/)
- [Daily Jagran — multi-bank UPI outage Feb 2024](https://www.thedailyjagran.com/technology/hdfc-bank-server-down-today-status-bank-of-baroda-users-cry-upi-transactions-error-gpay-phonepe-news-x-twitter-latest-live-updates-10131966)
- [Akshansh Jaiswal — UPI architecture deep dive (HSM PBKDF2 numbers)](https://blog.akshanshjaiswal.com/the-upi-architecture-a-security-look)
- [Avekshaa — UPI 10K TPS engineering](https://avekshaa.com/application-performance-management/upi-transaction-performance-engineering-systems-for-10000-tps/)
- [Coingeek — UPI saturation 2025 outlook](https://coingeek.com/will-upi-exponential-growth-lead-to-saturation-in-2025/)
- [Straits Research — HSM market size 2024–2033](https://straitsresearch.com/report/hardware-security-modules-market)
- [Futurex VirtuCrypt cloud HSM](https://www.futurex.com/products/cloud-products/virtucrypt-cloud-payment-hsm)
- [RBI Master Direction on Payment Aggregators — 15 Sept 2025](https://www.fidcindia.org.in/wp-content/uploads/2025/09/RBI-PAYMENT-AGGREGATORS-DIRECTIONS-15-09-25.pdf)
- [Lexplosion — RBI MD 2025 PA-P deadline analysis](https://lexplosion.in/rbi-master-direction-2025-compliance-mandate-for-payment-aggregators-and-pa-p-deadline/)
- [Mondaq — RBI MD 2025 deep dive](https://www.mondaq.com/india/financial-services/1706010/rbi-master-direction-2025-compliance-mandate-for-payment-aggregators-and-pa-p-deadline)
- [Mordor Intelligence — payment orchestration market](https://www.mordorintelligence.com/industry-reports/payment-orchestration-platform-market)
- [Grand View Research — payment orchestration market 2030](https://www.grandviewresearch.com/industry-analysis/payment-orchestration-platform-market-report)
- [InfoQ — ONNX AI inference in Java for enterprise architects](https://www.infoq.com/articles/onnx-ai-inference-with-java/)
- [ONNX Runtime Java — getting started](https://onnxruntime.ai/docs/get-started/with-java.html)
- [Milvus — ONNX in Java](https://milvus.io/blog/no-python-no-problem-model-inference-with-onnx-in-java-or-any-other-language.md)
- [Redis — feature stores for real-time AI/ML](https://redis.io/blog/feature-stores-for-real-time-artificial-intelligence-and-machine-learning/)
- [Feast docs — feature store](https://docs.feast.dev)
- [Calmops — Feast vs Tecton vs Redis 2025](https://calmops.com/ai/feature-store-feast-tecton-redis/)
- [GoCodeo — top 5 feature stores 2025](https://www.gocodeo.com/post/top-5-feature-stores-in-2025-tecton-feast-and-beyond)
