# AEPO Core — Universal Intelligent Decision & Routing Platform

> **From a UPI orchestrator to a domain-agnostic decision substrate.**
> Author: CPO + Principal Distributed Systems Architect synthesis for Umesh Maurya
> Date: 2026-05-07
> Audience: Solo backend engineer pivoting an existing PoC into a multi-domain platform.

---

## 0. Naming and the pivot frame

Before architecture: name the thing properly. **AEPO** stays — the acronym expands from "Autonomous Enterprise *Payment* Orchestrator" to **"Autonomous Enterprise *Policy* Orchestrator."** One word. Hackathon brand equity preserved. Domain-agnostic from word two.

The pivot in one paragraph: AEPO's UPI-specific RL loop was always a special case of a far more general pattern — *learn a policy that maps real-time context to a discrete decision, where each decision has a measurable outcome, a resource cost, an experience cost, and a compliance footprint*. Payments are one instance. Identity is another. API abuse, content moderation, ad fraud, supply-chain decisioning — all the same shape. The product is not "a UPI router with AI," it's **"the substrate on which any team builds a decision pipeline they can train, audit, and operate."**

That puts AEPO in a category that does not yet have a clear leader: **the programmable decision platform**. Sift ships fraud-only at $200k+ ARR with custom quotes, no transparent per-decision pricing. ([Vendr — Sift pricing](https://www.vendr.com/buyer-guides/sift-science)) Castle ships auth-only. Cloudflare Bot Management is bot-only. AWS Fraud Detector is fraud-only and only on AWS. Hyperswitch is payments-only. **No incumbent owns the cross-domain decision plane.** That is the shaped hole in the market.

---

## 1. The Generic Event Abstraction

### 1.1 The decision pattern is universal

Every domain that lands on AEPO has the same five-part shape:

```
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌────────┐
   │ TRIGGER │ →  │ CONTEXT │ →  │ DECIDE  │ →  │ OUTCOME  │ →  │ AUDIT  │
   └─────────┘    └─────────┘    └─────────┘    └──────────┘    └────────┘
   an event       features       a discrete     observed         immutable
   arrives        + telemetry    action from    downstream       trace
                                 a typed        signal
                                 catalog
```

A UPI payment, a user registration, a login attempt, a KYC document upload, an outbound API call, a scraping signal — all five-tuples. The platform's job is to make every one of those tuples (a) cheap to express, (b) safe to compose, (c) auditable to a regulator, (d) trainable by an RL agent.

### 1.2 The Universal Event Envelope

Every event entering AEPO is wrapped in a single envelope. The envelope is rigid; the payload is polymorphic. This is the key abstraction.

```protobuf
// proto/aepo/event/v1/envelope.proto
syntax = "proto3";
package aepo.event.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/any.proto";

message Envelope {
  // Identity
  string envelope_id        = 1;   // ULID, globally unique, ingestion-assigned
  string tenant_id          = 2;   // multi-tenant root key
  string idempotency_key    = 3;   // client-supplied; dedup window 24h
  string correlation_id     = 4;   // links envelopes in a saga
  string causation_id       = 5;   // upstream envelope that produced this one

  // Domain routing
  string domain             = 10;  // "payments", "identity", "content", ...
  string event_type         = 11;  // "payments.upi.collect_request", "identity.user.registration", ...
  uint32 schema_version     = 12;  // monotonic, per event_type

  // Time
  google.protobuf.Timestamp ingest_ts   = 20;
  google.protobuf.Timestamp event_ts    = 21;  // client-asserted, MAY differ
  uint32 processing_deadline_ms         = 22;  // hard deadline for hot path

  // Provenance & trust
  string origin_service     = 30;  // ingestion adapter that produced this
  string origin_ip          = 31;
  string user_agent         = 32;
  bytes  origin_signature   = 33;  // optional mTLS-derived signature

  // Routing hints (Quality of Service)
  enum QoS { BEST_EFFORT = 0; STANDARD = 1; PREMIUM = 2; }
  QoS qos                   = 40;

  // Domain payload — opaque to the platform, typed by the domain plugin
  google.protobuf.Any payload = 100;
}
```

**Why this exact shape:**

- `envelope_id` (ULID, not UUID): time-sortable, indexable, K-sorted Kafka partitioning works.
- `tenant_id` is mandatory and at the top — every Kafka topic, Redis key, Postgres row, audit row is prefixed by it. Multi-tenancy is structural, not configurable.
- `idempotency_key` decouples the client from internal retries. 24h dedup window in Redis (Lettuce + Lua atomic SETNX-with-TTL).
- `correlation_id` + `causation_id` make sagas free — a payment authorization that triggers a downstream MFA challenge that triggers a settlement event is a trivially-traceable chain.
- `processing_deadline_ms` is propagated to every downstream call as a deadline budget. The pipeline aborts with a typed `DeadlineExceeded` rather than racing to completion.
- `qos` is set at ingest by the adapter (e.g., RTGS gets PREMIUM, public-API hits get BEST_EFFORT). The infrastructure layer reads this for shedding decisions.
- `payload` is `Any` so the platform can ingest any registered schema without recompilation. The domain plugin downcasts it.

### 1.3 The Domain Plugin Contract

A "domain" is a pluggable bundle of:

1. **Avro/Protobuf schemas** for the event payloads it understands (registered in Schema Registry under `aepo.tenant.<id>.domain.<name>.event.<type>.v<n>`).
2. **A `DomainPlugin` Spring bean** that exposes:
   - `Set<String> supportedEventTypes()`
   - `PipelineDefinition pipeline(String eventType)` — the DAG (see §3.3)
   - `TelemetryProjector projector()` — maps domain fields → universal Telemetry Primitives
   - `ActionExecutor executor()` — implements the strategy actions for this domain
3. **A reward function** (for RL training) that maps `(decision, observed_outcome) → scalar`.
4. **A constraint manifest** — hard rules that no policy is ever allowed to violate (e.g., `never_approve_from_country IN ('CU','IR','KP','SY')`).

Adding a new domain is: ship 4 files, register the plugin JAR, deploy. No core code changes. **This is the platform thesis.**

### 1.4 Concrete event taxonomy (the V1 catalog)

A non-exhaustive catalog showing the ground the platform must cover. Each row is one Avro schema.

| Domain | Event type | Decision space examples |
|---|---|---|
| **payments.upi** | `collect_request`, `pay_request`, `mandate_create`, `refund` | Approve / Reject / Challenge / Throttle / Defer-Async |
| **payments.cards** | `auth_request` (Visa/MC), `tokenize`, `3ds_challenge_response` | Approve / Reject / StepUp(3DS) / Issuer-fallback |
| **payments.tap_pay** | `nfc_present`, `wallet_token_provision` | Approve / Reject (silent) / Challenge(biometric) |
| **payments.imps** | `transfer_request`, `beneficiary_validate` | Approve / Reject / Defer / Manual-review |
| **payments.rtgs** | `transfer_request` | Approve / Reject / Compliance-hold |
| **payments.crossborder.swift** | `mt103_send`, `compliance_screen` | Approve / Reject / OFAC-hold / Enrich-and-retry |
| **identity.user.registration** | `signup_request`, `email_verify`, `phone_verify` | Approve / Reject (silent) / Quarantine / Challenge |
| **identity.user.login** | `password_login`, `oauth_login`, `magic_link` | Approve / StepUp(MFA) / Challenge(captcha) / Reject / Tarpit |
| **identity.kyc** | `document_upload`, `liveness_check`, `address_verify` | Approve / Manual-review / Reject / Re-request |
| **api.gateway** | `api_call`, `rate_limit_check` | Allow / Throttle / 429-Reject / Tarpit / Honeypot |
| **content.moderation** | `post_create`, `comment_create`, `media_upload` | Auto-approve / Hold-for-review / Auto-reject / Shadow-ban |
| **commerce.checkout** | `cart_finalize`, `coupon_apply`, `inventory_reserve` | Allow / Hold / Decline / Manual-review |

The platform is the *same code path* for every row. The plugin is the *only* thing that differs.

### 1.5 The Decision Tetrad — generalizing the Asymmetric Risk Triad

The hackathon used a triad: **Fraud × Infra × SLA**. That doesn't generalize. A user-registration event has no "Kafka lag" in the payment sense, and a content moderation event has no "fraud" in the same way. We need axes that mean something in *every* domain.

**The Decision Tetrad** (4 dimensions, every domain instantiates each):

| Axis | Meaning | UPI mapping | Login mapping | Content mod mapping |
|---|---|---|---|---|
| **Outcome Risk** | P(bad downstream consequence \| approve) | fraud probability | account-takeover probability | toxicity / abuse score |
| **Resource Cost** | compute / money / 3rd-party fee for the decision pipeline | HSM cycles + bank API call | MFA SMS cost + auth-service CPU | classifier GPU + human review hours |
| **Experience Cost** | friction / latency / abandonment for the legitimate user | latency added by FullVerify | step-up MFA friction | false-positive removal of legit content |
| **Compliance Exposure** | regulatory / reputational risk of the decision | RBI/NPCI/AML | GDPR/SOX/CCPA/auth-log retention | DSA/Section-230/local hate-speech laws |

The RL reward function for any domain becomes a tenant-configurable weighted sum:

```
R(decision, outcome) = w_o * outcome_signal
                     - w_r * resource_cost
                     - w_e * experience_cost
                     - w_c * compliance_exposure
                     - w_safety * constraint_violation_indicator
```

`w_safety` is set by the platform, not the tenant — it's how we guarantee that hard constraints (never-approve-from-sanctioned-geo) dominate every other consideration.

This is the abstraction that lets one trained agent, with a shared encoder and per-domain heads, optimize across domains.

---

## 2. The Intelligent Decision Core — domain-agnostic AI/RL

### 2.1 The two universal vocabularies

To make the RL core domain-agnostic, two things must be standardized: the **input vocabulary** (telemetry primitives) and the **output vocabulary** (strategy catalog). Domain plugins translate domain fields into and out of these vocabularies.

#### 2.1.1 Telemetry Primitives (input vocabulary)

A typed registry of generic context signals. Every domain plugin's `TelemetryProjector` produces one or more of these on every event. The RL agent only ever sees primitives.

```java
public sealed interface TelemetryPrimitive
    permits QueueDepth, ThreatScore, ResourceUtilization,
            LatencyEMA, RateLimitProximity, SuccessRateEMA,
            VelocityAnomaly, GeographyAnomaly, DeviceTrust,
            NetworkTrust, EntityFreshness, PolicyAdherence,
            CounterpartyTrust, EntropyEMA {

    String name();           // canonical name for feature store keying
    double normalized();     // always in [0.0, 1.0]
    Range raw();             // domain-original values for replay
    Confidence confidence(); // [0.0, 1.0] — lets RL discount stale features
}
```

**The full V1 primitive set** (14 primitives, mapping below):

| Primitive | Semantic | UPI mapping | Login mapping | Content moderation mapping |
|---|---|---|---|---|
| `QueueDepth` | backlog cardinality | `kafka_lag` | auth-service queue size | moderation backlog |
| `ThreatScore` | upstream risk score | UPI fraud model output | login anomaly score | toxicity classifier output |
| `ResourceUtilization` | % capacity of bottleneck | `db_pool` + HSM util | auth-service CPU | classifier GPU util |
| `LatencyEMA` | rolling p99 (ms) | `rolling_p99` | auth check latency | inference latency |
| `RateLimitProximity` | % of bucket consumed | per-merchant TPS | per-IP login rate | per-user post rate |
| `SuccessRateEMA` | % of recent approvals | merchant approval rate | login success rate | auto-approve rate |
| `VelocityAnomaly` | z-score of activity rate | txn velocity | login velocity | post velocity |
| `GeographyAnomaly` | binary or score | cross-border flag | login from new geo | post from spam geo |
| `DeviceTrust` | device fingerprint trust | tokenized device | device fingerprint | device fingerprint |
| `NetworkTrust` | IP/ASN reputation | IP reputation | IP reputation | IP reputation |
| `EntityFreshness` | hours since first seen | merchant age | account age | account age |
| `PolicyAdherence` | recent compliance score | recent VAR violations | recent ToS violations | recent ToS violations |
| `CounterpartyTrust` | downstream trust score | bank API trust | OAuth provider trust | publisher trust |
| `EntropyEMA` | system unpredictability | `system_entropy` | login pattern entropy | content type entropy |

Every domain implements `TelemetryProjector`:

```java
public interface TelemetryProjector {
    TelemetryVector project(Envelope envelope, EnvironmentSnapshot env);

    // The set of primitives this projector produces — allows the RL agent
    // to know which inputs are populated for a given event type
    Set<Class<? extends TelemetryPrimitive>> producedPrimitives();
}
```

The RL agent sees a sparse, schema-versioned vector of primitives. Missing primitives carry `Confidence(0.0)` — the agent learns to ignore them rather than crashing.

#### 2.1.2 Strategy Catalog (output vocabulary)

A typed catalog of action *classes*. Each class is a kind of decision the platform supports. Each class has 0+ parameters. Domain plugins declare which classes are valid for which events.

```java
public sealed interface Strategy permits
    HardApprove, SoftApprove, HardReject, SilentReject,
    QuarantineReject, Challenge, StepUp, Throttle,
    CircuitBreak, Tarpit, ShadowProcess, Defer,
    Escalate, Honeypot {

    StrategyClass type();
    Map<String, Object> parameters();  // e.g., {"mfa_factor": "TOTP", "challenge_difficulty": "medium"}
    String rationaleId();              // links to the policy decision rationale
}
```

| Strategy | Semantics | Domains where it applies |
|---|---|---|
| `HardApprove` | Approve with no caveat | All |
| `SoftApprove` | Approve provisionally; subject to async re-evaluation | Payments (provisional credit), Content (publish + monitor) |
| `HardReject` | Reject with explicit error to client | All |
| `SilentReject` | Return success-shape to attacker, drop internally | Identity (registration anti-enumeration), Bot defense |
| `QuarantineReject` | Reject + place entity on watchlist | Identity, Payments fraud |
| `Challenge` | Inline friction (captcha, cognitive challenge) | Identity, API |
| `StepUp` | Demand stronger auth (parameterized: `mfa_factor`, `biometric`) | Identity, Payments-3DS |
| `Throttle` | Rate-limit this entity for window | API, Identity, Payments-flood |
| `CircuitBreak` | Stop processing this domain temporarily | All |
| `Tarpit` | Slow-respond to attacker (defensive cost imposition) | API, Identity |
| `ShadowProcess` | Process but log differently; for canary policy mode | All |
| `Defer` | Queue for async / off-peak processing | Payments (DeferredAsync), KYC (manual review) |
| `Escalate` | Route to human reviewer | KYC, content moderation, fraud-ops |
| `Honeypot` | Route to honeypot environment for attacker profiling | Bot defense, fraud |

#### 2.1.3 Action masking (per-domain restrictions)

An action mask is computed per event from `(domain, event_type, tenant_compliance_profile)` and applied before policy sampling:

```java
public interface ActionMaskProvider {
    BitSet maskFor(String domain, String eventType, TenantPolicy tenantPolicy, Envelope env);
}

// Example: a registration event is per-event, not per-stream — Throttle/CircuitBreak don't apply.
// Example: a tenant on a "regulated" compliance tier may forbid SilentReject.
```

This is where domain knowledge constrains the universal RL agent without baking it into the agent.

### 2.2 The RL architecture (concretely)

The hackathon shipped a Q-table. The platform needs a more flexible model. The right architecture is **a shared backbone with per-domain heads, hierarchical action selection, and constrained policy gradient**.

#### 2.2.1 The model

```
                ┌──────────────────────────────────────────┐
                │       Telemetry Vector (14-dim, sparse,  │
                │       confidence-weighted)               │
                └──────────────────┬───────────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  Shared Transformer Encoder      │  ← learns universal
                  │  (4 layers, 64-dim, ~250k params)│    representation
                  └────────────────┬────────────────┘
                                   │ embedding (64-dim)
                  ┌────────────────┼────────────────┐
                  │                │                │
            ┌─────▼───────┐  ┌─────▼───────┐  ┌─────▼───────┐
            │ Meta-Policy │  │ Payment Head │  │ Identity Head│ ...
            │ (which      │  │ (full action │  │ (full action │
            │  strategy   │  │ params)      │  │ params)      │
            │  class?)    │  └─────────────┘  └──────────────┘
            └─────────────┘
                                   │
                            apply action mask
                                   │
                            sample / argmax
                                   │
                                Strategy
```

- **Shared encoder**: same parameters across domains. Trained on multi-task replay buffer. Cross-domain transfer learning is automatic — a fraud pattern learned in payments helps identity.
- **Domain heads**: small (1 hidden layer, 32-dim). Each outputs a distribution over the strategy classes valid for that domain, plus parameter heads for parameterized strategies.
- **Meta-policy**: a top-level head that picks *which strategy class* to use. Sub-policies pick parameters. This is **hierarchical RL** — it dramatically reduces the action space the meta-policy must search.
- **Action mask**: applied as `-inf` logits before softmax. Constraints are hard-baked.
- **Constrained policy gradient**: training uses Lagrangian relaxation for soft constraints (cost budgets) and action-mask masking for hard constraints (regulatory).

Total model size: ~500k parameters. Inferred in <5ms on CPU via ONNX Runtime in JVM. ([ONNX Runtime — Java](https://onnxruntime.ai/docs/get-started/with-java.html))

#### 2.2.2 The world model — TelemetryPredictor (generalized LagPredictor)

The hackathon `LagPredictor` predicted next-step `kafka_lag`. The generic version predicts the **next-step telemetry vector** given the current vector and a candidate action:

```
TelemetryPredictor:  (telemetry_t, action_t) → telemetry_{t+1}
```

This is the world model. It serves three purposes:

1. **1-step lookahead in the policy**: before committing to an action, the meta-policy can evaluate `predicted_outcome = world_model(state, candidate_action)` for each candidate and pick the best. This is the AEPO "veto" generalized.
2. **Dyna-Q imagined rollouts**: during training, augment real experience with imagined trajectories. Sample-efficient on rare events (which is most of fraud and abuse).
3. **Anomaly detection (free!)**: `divergence = |actual_telemetry_{t+1} - predicted_telemetry_{t+1}|`. Sustained large divergence = the world model is wrong, which usually means the world has changed. This is a signal to alert SREs *before* the policy starts misbehaving.

#### 2.2.3 Counterfactual evaluation — the safe-deployment lever

A new policy version is *never* deployed cold to production. The flow:

```
1. Train candidate policy π_new on replay buffer.
2. For each historical decision (s_t, a_t, r_t):
   - Compute π_new(a | s_t) — what would the new policy have done?
   - Use Inverse Propensity Scoring (IPS) or Doubly Robust (DR) estimators
     to project the expected reward under π_new.
3. Compare projected reward against current production policy π_prod.
4. If projected_reward(π_new) ≥ projected_reward(π_prod) + safety_margin:
   - Promote π_new to canary (5% traffic) for N hours.
   - Auto-rollback if observed reward drops below threshold.
5. If canary holds, promote to 100%.
```

This is mandatory infrastructure, not optional. The Policy Engine refuses to promote without a counterfactual report.

### 2.3 Strategy selection in context — examples

The same RL core selects different strategies for different events under different conditions. Three concrete walkthroughs:

**Example A — UPI payment, low risk, healthy infra:**
- Telemetry: `ThreatScore=0.05, QueueDepth=0.3, LatencyEMA=0.2, ResourceUtilization=0.4`
- Action mask: full payment action set
- Meta-policy: probably `HardApprove` (highest expected reward)
- Sub-policy: parameters minimal
- Outcome: approve, dispatch to bank

**Example B — Login attempt, anomalous geography, low device trust:**
- Telemetry: `GeographyAnomaly=1.0, DeviceTrust=0.15, ThreatScore=0.6, RateLimitProximity=0.85`
- Action mask: identity action set; `Tarpit, Throttle, Challenge, StepUp, SilentReject` valid
- Meta-policy: `StepUp` (don't reject — it's possibly a real user; don't approve — anomaly is high)
- Sub-policy: `mfa_factor=TOTP_or_PUSH` (whichever the user has registered)
- Outcome: send push notification, await response

**Example C — Botnet flood on registration endpoint, premium tenant:**
- Telemetry: `VelocityAnomaly=4.5, RateLimitProximity=0.99, NetworkTrust=0.05, EntityFreshness=0.0`
- Action mask: identity registration action set
- Meta-policy: `SilentReject` for the flood (anti-enumeration), `Tarpit` for the suspected attacker IPs
- Sub-policy: tarpit delay = exponential backoff up to 30s
- Outcome: legitimate signups continue uninterrupted (separate token bucket), attacker burns time

The **same neural network** produces all three decisions. The domain plugin and action mask shape the search space; the shared encoder reads the universal telemetry; the head produces a domain-valid action distribution.

---

## 3. Production-Grade Microservices Architecture (Java/Spring Boot/Kafka)

### 3.1 Six service planes

The platform is organized as six logical planes, each with explicit ingress/egress contracts. Don't mix concerns across planes.

```
            ╔══════════════════════════════════════════════════════════════╗
            ║                       CONTROL PLANE                          ║
            ║  Tenant mgmt · Plugin registry · Policy Studio · Audit query ║
            ╚════════════════════════════╤═════════════════════════════════╝
                                         │ (admin APIs, gRPC)
            ┌────────────────────────────▼────────────────────────────────┐
            │                       EDGE PLANE                            │
            │  API Gateway (Spring Cloud Gateway) · Adapter Services      │
            │  (REST, gRPC, Webhook, Kafka source) · TLS/mTLS · Rate-Limit │
            └────────────┬─────────────────────────────────────┬───────────┘
                         │                                     │
                         │ Envelope+Payload (Avro/Proto)        │
                         ▼                                     ▼
            ┌──────────────────────────┐           ┌──────────────────────┐
            │  DECISION PLANE          │           │ DATA PLANE           │
            │  - Orchestrator          │◄─────────►│ - Kafka cluster      │
            │  - Pipeline DAG engine   │           │ - Schema Registry    │
            │  - Policy Engine         │           │ - Feature Store      │
            │  - RL Inference (ONNX)   │           │   (Feast + Redis)    │
            │  - Telemetry Projector   │           │ - Postgres (audit)   │
            └────────────┬─────────────┘           └──────────────────────┘
                         │ Strategy
                         ▼
            ┌──────────────────────────┐
            │  ACTION PLANE            │
            │  - Strategy Executors    │
            │    (per side-effect      │
            │     type: bank dispatch, │
            │     MFA send, hold,      │
            │     escalate, …)         │
            └────────────┬─────────────┘
                         │ Outcome events
                         ▼
            ┌──────────────────────────┐
            │  AUDIT PLANE             │
            │  - Append-only ledger    │
            │  - Merkle commit         │
            │  - Replay service        │
            └──────────────────────────┘
```

### 3.2 The Strategy + Plugin pattern (Spring Boot, no `if/else`)

Hardcoded `if/else` is the failure mode. The right pattern is **registry-of-strategies**, lookup by qualifier, dynamic plugin discovery.

#### 3.2.1 Domain plugin interface

```java
public interface DomainPlugin {
    String domain();                                     // "payments.upi"
    Set<String> supportedEventTypes();
    PipelineDefinition pipeline(String eventType);
    TelemetryProjector projector();
    ActionExecutor executor();
    ActionMaskProvider actionMaskProvider();
    RewardFunction rewardFunction();
    ConstraintManifest constraints();

    // Schema declarations — registered with Schema Registry on startup
    List<SchemaDeclaration> schemas();
}
```

#### 3.2.2 The plugin registry

```java
@Component
public class DomainPluginRegistry {
    private final Map<String, DomainPlugin> byDomain;
    private final Map<String, DomainPlugin> byEventType;

    public DomainPluginRegistry(List<DomainPlugin> plugins) {
        // Spring auto-collects every @Component implementing DomainPlugin
        this.byDomain = plugins.stream()
            .collect(toUnmodifiableMap(DomainPlugin::domain, p -> p));
        this.byEventType = plugins.stream()
            .flatMap(p -> p.supportedEventTypes().stream().map(et -> Map.entry(et, p)))
            .collect(toUnmodifiableMap(Entry::getKey, Entry::getValue));
        validateNoConflicts();
        logRegisteredPlugins();
    }

    public DomainPlugin forEvent(String eventType) {
        DomainPlugin p = byEventType.get(eventType);
        if (p == null) throw new UnknownEventTypeException(eventType);
        return p;
    }
}
```

This is the Spring Plugin pattern — Spring Boot collects every `DomainPlugin` bean into a list at startup; the registry indexes them. ([Javarevisited — Strategy Pattern with Spring Plugin](https://medium.com/javarevisited/the-strategy-design-pattern-with-spring-plugin-e99021c8f6eb))

Adding a new domain = drop a JAR into the classpath with a `@Component` `DomainPlugin` and Spring picks it up. Or, for true runtime extensibility, package each plugin as a separate Spring Boot module that exposes a gRPC `DomainPlugin` server — the core service does service discovery via Consul/etcd and treats remote plugins identically to local ones.

#### 3.2.3 Hot-swap and feature gating

```java
@Component
@ConditionalOnProperty(value = "aepo.plugin.payments-upi.enabled", havingValue = "true")
public class UpiPaymentPlugin implements DomainPlugin { ... }
```

Combined with Unleash feature flags, this lets you canary a plugin to specific tenants before rolling globally.

### 3.3 The Pipeline DAG — composable stages, no orchestration code

Every event is processed by a DAG. The DAG is defined by the plugin, executed by a generic engine. This eliminates orchestration code.

#### 3.3.1 The Stage type

```java
public interface Stage<I, O> {
    String name();
    Mono<O> apply(StageContext ctx, I in);
    Duration budget();           // per-stage timeout
    Class<I> inputType();
    Class<O> outputType();
}
```

Stages are typed; the engine validates that the output of stage N matches the input of stage N+1 *at registration time*, not runtime.

#### 3.3.2 PipelineDefinition

```java
PipelineDefinition pipeline = PipelineDefinition.builder("payments.upi.collect_request")
    .stage(SchemaValidator.class)             // Envelope -> ValidatedEnvelope
    .stage(IdempotencyCheck.class)            // ValidatedEnvelope -> DedupedEnvelope
    .stage(TelemetryProjection.class)         // DedupedEnvelope -> ProjectedEvent
    .stage(FeatureEnrichment.class)           // ProjectedEvent -> EnrichedEvent (Feast lookup)
    .parallel(                                 // ← parallel fan-out
        ParallelGroup.of(
            FraudScoring.class,
            VelocityCheck.class,
            SanctionsScreen.class
        ))
    .stage(StrategyDecision.class)            // EnrichedEvent -> Decision
    .stage(ActionMaskApplication.class)
    .stage(CryptoGate.class)                  // Decision -> ScheduledDecision
    .stage(BankDispatch.class)                // ScheduledDecision -> Outcome
    .stage(AuditWriter.class)
    .deadline(Duration.ofMillis(200))
    .build();
```

#### 3.3.3 Engine execution

```java
@Service
public class PipelineEngine {
    public Mono<Outcome> execute(Envelope envelope) {
        DomainPlugin plugin = registry.forEvent(envelope.getEventType());
        PipelineDefinition pipeline = plugin.pipeline(envelope.getEventType());
        StageContext ctx = StageContext.from(envelope);

        return pipeline.stages().stream()
            .reduce(
                Mono.just((Object) envelope),
                (acc, stage) -> acc
                    .timeout(stage.budget())
                    .flatMap(in -> stage.apply(ctx, in)),
                (a, b) -> a  // not used in sequential reduce
            )
            .cast(Outcome.class)
            .timeout(pipeline.deadline());
    }
}
```

(In production: replace `Mono.reduce` with a topologically-aware executor that handles parallel groups properly. The above is the conceptual sketch.)

#### 3.3.4 Why DAGs over `if/else`

- **Composition**: a stage like `SanctionsScreen` is reused across payments + identity + content with no edits.
- **Observability**: each stage emits a span (OpenTelemetry); the trace is the DAG. Free distributed tracing.
- **Testing**: each stage is a unit. Pipelines are integration tests.
- **Hot-reload**: pipeline definitions are stored in Postgres + git-backed config; `RefreshScope` reloads on policy team request.
- **Safety**: type-checked at registration; can't deploy a pipeline that wires wrong stages together.

### 3.4 Schema Registry — multi-tenant, multi-domain, multi-version

Confluent Schema Registry (or Karapace, the Apache Kafka community alternative) is mandatory. Subject naming convention:

```
aepo.tenant.<tenant_id>.domain.<domain>.event.<event_type>.v<schema_version>-value
aepo.tenant.<tenant_id>.domain.<domain>.event.<event_type>.v<schema_version>-key
```

Compatibility settings, per ([Confluent — Schema Registry Best Practices](https://www.confluent.io/blog/best-practices-for-confluent-schema-registry/)):

- **Envelope schema**: `BACKWARD_TRANSITIVE`. We never want to break old consumers.
- **Domain payload schemas**: `FULL_TRANSITIVE` if the tenant requires it; otherwise `BACKWARD_TRANSITIVE`.
- **Reward / outcome schemas**: `FULL_TRANSITIVE`. Replay must always work.

Schema evolution rules (enforced by CI on every plugin update):

1. Never remove a required field. Mark it deprecated with a deprecation timestamp.
2. Never rename a field. Add an alias.
3. Always provide a default for new fields.
4. Bump `schema_version` on every breaking change.
5. Schemas have a `valid_from_ts`; consumers older than 90 days past `deprecated_at_ts` get hard-rejected at the gateway with a clear migration error.

#### 3.4.1 Tenant-isolated subject namespace

The `aepo.tenant.<tenant_id>.` prefix means a tenant's schemas live in their own namespace. A misbehaving tenant cannot break another tenant's compatibility checks. ACLs on the registry are subject-prefix scoped.

#### 3.4.2 Generated DTOs

Plugins ship Avro `.avsc` files in `src/main/resources/avro/`. The Avro Maven/Gradle plugin generates Java DTOs. The build system enforces a check: every `@Component DomainPlugin` must declare its schemas, and CI fails if a schema is referenced but not registered.

### 3.5 Hot-path execution model

#### 3.5.1 Java 21 virtual threads + structured concurrency

```java
public Mono<Outcome> route(Envelope envelope) {
    try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
        Subtask<TelemetryVector> telemetryTask =
            scope.fork(() -> telemetryProjector.project(envelope, env));
        Subtask<TenantPolicy> policyTask =
            scope.fork(() -> policyEngine.policyFor(envelope.getTenantId()));

        scope.join().throwIfFailed();

        TelemetryVector telemetry = telemetryTask.get();
        TenantPolicy policy = policyTask.get();
        BitSet mask = actionMaskProvider.maskFor(envelope, policy);

        Strategy strategy = rlInference.infer(telemetry, mask, policy);
        return executor.execute(strategy, envelope);
    }
}
```

Virtual threads make per-event fan-out free. Structured concurrency makes deadline propagation correct. No thread-pool sizing nightmares.

#### 3.5.2 Backpressure and deadlines

Every downstream call carries the remaining budget:

```java
Duration remaining = ctx.deadline().minus(Duration.between(ctx.startedAt(), Instant.now()));
if (remaining.isNegative()) throw new DeadlineExceededException();
return downstream.callWithTimeout(req, remaining);
```

Combined with Resilience4j circuit breakers (per downstream + per tenant), the system fails fast under stress rather than queueing.

#### 3.5.3 Outbox pattern for reliable Kafka emit

Audit and outcome events MUST never be lost. Use the Outbox pattern:

```sql
CREATE TABLE outbox (
  id UUID PRIMARY KEY,
  envelope_id TEXT NOT NULL,
  topic TEXT NOT NULL,
  partition_key TEXT,
  payload BYTEA NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  delivered_at TIMESTAMPTZ
);
```

The Orchestrator writes the audit row + outbox row in a single Postgres transaction. A separate `OutboxRelay` service (Debezium CDC or polling) ships outbox rows to Kafka with at-least-once semantics. This decouples the hot path from Kafka availability.

### 3.6 The Audit Plane — regulator-grade replay

Every decision must be replayable years later. The Audit Plane stores:

```sql
CREATE TABLE decision_audit (
  id ULID PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  envelope_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  ingest_ts TIMESTAMPTZ NOT NULL,
  decision_ts TIMESTAMPTZ NOT NULL,
  policy_version TEXT NOT NULL,
  feature_vector_hash TEXT NOT NULL,
  feature_vector_pointer TEXT NOT NULL,  -- S3 URI for full vector
  strategy_class TEXT NOT NULL,
  strategy_parameters JSONB NOT NULL,
  rationale_id TEXT NOT NULL,
  outcome_observed_ts TIMESTAMPTZ,
  outcome_classification TEXT,
  merkle_leaf BYTEA NOT NULL
);

CREATE TABLE audit_merkle_root (
  tenant_id TEXT NOT NULL,
  window_start TIMESTAMPTZ NOT NULL,
  window_end TIMESTAMPTZ NOT NULL,
  root_hash BYTEA NOT NULL,
  PRIMARY KEY (tenant_id, window_start)
);
```

A `MerkleCommitter` job runs every minute per tenant: build a Merkle tree over all decisions in the window, commit the root hash to (a) Postgres, (b) S3 with object lock, (c) optionally a tenant-controlled write target (their own DB, or a public blockchain anchor for highest-grade tenants).

The Replay Service can reconstruct any historical decision exactly: feature vector → policy version → action distribution → sampled strategy. Inclusion proofs allow regulators to verify single-decision integrity without reading the full ledger.

---

## 4. Infrastructure & Scalability — preventing cross-domain starvation

The design constraint: a flood of registration spam must not delay an RTGS payment. The platform must offer **structural** isolation, not best-effort.

### 4.1 Cell-based architecture

Adopt the AWS / Slack cell pattern. ([Slack Engineering — Migration to Cellular Architecture](https://slack.engineering/slacks-migration-to-a-cellular-architecture/), [AWS Well-Architected — Cell-Based Architecture](https://docs.aws.amazon.com/wellarchitected/latest/reducing-scope-of-impact-with-cell-based-architecture/cell-deployment.html))

```
                    ┌─────────────────────────────┐
                    │    Tenant Router (cell-     │
                    │    routing layer; sticky    │
                    │    by tenant_id hash)       │
                    └───┬──────────┬─────────┬────┘
                        │          │         │
              ┌─────────▼──┐  ┌────▼─────┐  ┌▼──────────┐
              │   CELL A   │  │  CELL B  │  │ CELL "P"  │
              │ (1000s of  │  │ (1000s   │  │ (1 mega-  │
              │  small     │  │  small   │  │  tenant,  │
              │  tenants)  │  │  tenants)│  │  dedicated│
              └────────────┘  └──────────┘  └───────────┘
                  │  full          │  full         │ full
                  │  stack         │  stack        │ stack
                  ▼                ▼               ▼
              [Gateway,        [Gateway,        [Gateway,
               Orchestrator,    Orchestrator,    Orchestrator,
               RL Inference,    RL Inference,    RL Inference,
               Kafka, Redis,    Kafka, Redis,    Kafka, Redis,
               Postgres,        Postgres,        Postgres,
               Audit]           Audit]           Audit]
```

- **Each cell is a complete, independent deployment of the platform.**
- A tenant lives in exactly one cell. Migrations are explicit and rare.
- A cell-level failure is bounded: blast radius = the tenants in that cell, not the platform.
- New cells are commissioned via Terraform; takes ~30 min.

This is how you get to 99.99% availability without 99.99%-level engineering on every line — most of the engineering goes into making cells trivially replicable.

### 4.2 The Quality of Service (QoS) Tetrad

Inside a cell, every event has a QoS class set by the ingest adapter:

| QoS | SLO | Resource guarantee | When shed |
|---|---|---|---|
| `PREMIUM` | p99 < 50ms | Dedicated thread pool, dedicated Kafka partitions, dedicated Redis slots | **Never sheddable.** If we can't serve PREMIUM, we trip a cell-level circuit breaker. |
| `STANDARD` | p99 < 200ms | Burstable; shared with other STANDARD | Shed when cell load > 80% |
| `BEST_EFFORT` | p99 < 1000ms | Whatever's left | Shed first (load > 60%) |
| `BATCH` | seconds-scale | Off-peak only | Always sheddable; runs at 10% of peak capacity |

Mapping to event types is per-tenant configurable but with platform defaults:

- RTGS, SWIFT, premium-tenant payments → PREMIUM
- UPI, IMPS, cards, login → STANDARD
- Public API, registration, KYC upload → BEST_EFFORT (with promotion paths if attacked)
- Backfill jobs, replay, training data hydration → BATCH

### 4.3 Bulkhead pattern at multiple layers

```
                        ┌─── Premium Pool ──── Kafka [QoS=premium] ─── Redis namespace [premium]
                        │
       Inbound ────────►├─── Standard Pool ─── Kafka [QoS=standard] ── Redis namespace [standard]
                        │
                        └─── Best-Effort Pool─ Kafka [QoS=besteffort] Redis namespace [besteffort]
```

**Concretely** in a Spring Boot service:

```java
@Configuration
public class ThreadPoolConfig {
    @Bean("premiumExecutor")
    public Executor premiumExecutor() {
        return Executors.newThreadPerTaskExecutor(
            Thread.ofVirtual().name("premium-", 0).factory());
    }
    @Bean("standardExecutor")
    public ThreadPoolBulkhead standardBulkhead() {
        return ThreadPoolBulkhead.of("standard", ThreadPoolBulkheadConfig.custom()
            .maxThreadPoolSize(200).coreThreadPoolSize(50).queueCapacity(1000).build());
    }
    @Bean("bestEffortExecutor")
    public ThreadPoolBulkhead bestEffortBulkhead() {
        return ThreadPoolBulkhead.of("best-effort", ThreadPoolBulkheadConfig.custom()
            .maxThreadPoolSize(50).coreThreadPoolSize(10).queueCapacity(100).build());
    }
}
```

Premium uses unbounded virtual threads (it must never queue). Standard and Best-Effort have bounded queues — they shed under load.

### 4.4 Multi-tier rate limiting

Rate limiting is layered and multiplicative. A request must pass *all* layers:

```
1. Tenant cell quota:           N events/sec across all domains, all event types
2. Tenant domain quota:         N events/sec for {tenant, domain}
3. Tenant event-type quota:     N events/sec for {tenant, event_type}
4. Tenant entity quota:         N events/sec for {tenant, entity_id}  (e.g., merchant, user)
5. Cell-wide global quota:      N events/sec across all tenants in this cell, this domain
                                 (protects shared infra: HSM, third-party APIs)
```

Implementation: Lyft's Envoy-style rate limit service backed by Redis Cluster with Lua scripts for atomic decrement-and-check. Latency: ~1ms per check; checks parallelized when possible.

When a quota is exceeded, the response is QoS-dependent:
- `PREMIUM` → never quota-rejected; quotas are sized to never trigger for premium
- `STANDARD` → 429 with `Retry-After` header
- `BEST_EFFORT` → silently absorbed into a delayed queue or rejected

### 4.5 Adaptive load shedding (CoDel-style)

Inspired by CoDel (Controlled Delay) for network queues. Each Orchestrator pod tracks its own queueing delay (time from event ingest to start of pipeline execution):

```java
public class CoDelShedder {
    private final Duration target = Duration.ofMillis(5);
    private final Duration interval = Duration.ofMillis(100);
    private Instant nextDropAt;

    public boolean shouldShed(Event event, Duration observedDelay) {
        if (event.qos() == PREMIUM) return false;
        if (observedDelay.compareTo(target) <= 0) {
            nextDropAt = null;
            return false;
        }
        if (nextDropAt == null) nextDropAt = now().plus(interval);
        if (now().isAfter(nextDropAt)) {
            nextDropAt = nextDropAt.plus(interval);
            return event.qos() == BEST_EFFORT;
        }
        return false;
    }
}
```

Under sustained load, BEST_EFFORT traffic gets shed first; STANDARD gets shed second; PREMIUM never gets shed (the cell trips its circuit breaker first if it can't keep up).

### 4.6 The system telemetry feedback loop

Critically: **the load shedding signal feeds back into the RL agent as a telemetry primitive** (`SystemLoadIndex`). The agent learns that under high load, *its own decisions should be cheaper*. So during a flood, the agent might switch from `Challenge` (which costs an MFA SMS) to `Tarpit` (which costs nothing), without explicit programming.

This is the closing of the loop: the platform's awareness of its own state becomes an input to the policy. Self-aware infrastructure.

### 4.7 Storage isolation

- **Kafka**: per-cell cluster (or shared cluster with strict per-tenant ACLs and per-tenant topic prefixes). Quotas enforced via Kafka 3.x client quotas (`producer_byte_rate`, `consumer_byte_rate`).
- **Redis**: per-cell Cluster; per-tenant namespace `tenant:{tid}:`. ACLs scoped to namespace.
- **Postgres**: schema-per-tenant in the small-tenant cell; database-per-tenant in dedicated cells. Connection pools sized per QoS class.
- **Feature Store**: per-cell Feast deployment; offline store partitioned by tenant.
- **Audit ledger**: dedicated Postgres in every cell. Audit data NEVER crosses cells.

---

## 5. The Zero-to-One Execution Plan

You are one engineer. The plan must respect that. The Minimum Viable Architecture proves the abstraction across **two completely different domains** — that is the only goal of the first 12 weeks.

### 5.1 The MVA contract

**Objective**: One Spring Boot application processes both:
- A UPI payment event
- A user registration event

…using the same Envelope, the same Orchestrator, the same Pipeline DAG engine, the same RL inference path, and the same audit ledger. The only domain-specific code is the two `DomainPlugin` implementations.

**Acceptance criteria** (every one is binary):
1. ✅ The same JVM process serves `POST /v1/payments/upi` and `POST /v1/identity/registration`.
2. ✅ Both event types are wrapped in a common `Envelope` with `tenant_id` and `domain`.
3. ✅ Both pipelines are defined declaratively by their `DomainPlugin`, not in `if/else`.
4. ✅ Both produce decisions via the same `RLInferenceService`, with action masks restricting to domain-valid strategies.
5. ✅ Both write to the same `decision_audit` table; replay works for both.
6. ✅ Adding a third domain (e.g., `content.moderation`) requires zero modifications to the core service — only a new plugin module + Avro schemas.
7. ✅ A flood of registration events does NOT degrade UPI p99 by more than 10ms (bulkhead test).

### 5.2 Phase-by-phase, week-by-week

#### Phase 0 — Decisions before code (Week 0, before W1 starts)

- Choose Avro vs Protobuf. Recommendation: **Avro** for the cleaner schema-evolution story with Confluent, **Protobuf** if you anticipate gRPC at the edge. Pick one and stick with it.
- Choose Java 21 (yes, virtual threads) and Spring Boot 3.3+.
- Decide cell topology for V1 — recommendation: **single cell, multi-tenant.** Cell-routing is a V2 concern.
- Set the deadline budget for the hot path: **150ms p99** for UPI, **300ms p99** for registration. Everything else is derived from this.

#### Phase 1 — Spine (Weeks 1–2)

**Goal**: An envelope flows from gateway to orchestrator to a no-op decision and back, with audit.

| Week | Deliverable |
|---|---|
| 1 | Gradle multi-module monorepo. Modules: `aepo-core` (envelope, plugin SPI, pipeline engine, audit), `aepo-gateway` (Spring Cloud Gateway), `aepo-orchestrator` (Spring Boot service), `aepo-plugin-payments-upi` (stub), `aepo-plugin-identity-registration` (stub). Docker Compose: Postgres, Redis, Kafka, Schema Registry, Prometheus, Grafana. |
| 2 | Envelope Avro schema. `Envelope` Java DTO generated. `DomainPlugin` SPI defined. `PipelineEngine` skeleton (sequential reduce, no parallel groups yet). Audit table created. End-to-end smoke test: send an envelope to the gateway, see a no-op decision and an audit row. |

#### Phase 2 — Domain 1: UPI (Weeks 3–4)

**Goal**: A real UPI plugin processes real (mock) events through a real pipeline.

| Week | Deliverable |
|---|---|
| 3 | `UpiPaymentPlugin` registered. Avro schema for `payments.upi.collect_request`. Pipeline: `[SchemaValidator, IdempotencyCheck, TelemetryProjection, RuleBasedDecision, MockBankDispatch, AuditWriter]`. `RuleBasedDecision` is a hardcoded rule for now (will become RL in Phase 5). |
| 4 | `UpiTelemetryProjector` produces 6 of the 14 primitives (the ones UPI cares about): `ThreatScore, QueueDepth, LatencyEMA, ResourceUtilization, RateLimitProximity, GeographyAnomaly`. Mock telemetry source (synthetic Kafka lag). Grafana dashboard shows projected primitives in real time. |

#### Phase 3 — Domain 2: Registration (Weeks 5–6)

**Goal**: A second domain on the same code path, proving generality.

| Week | Deliverable |
|---|---|
| 5 | `RegistrationPlugin` registered. Avro schema for `identity.user.registration`. Pipeline: `[SchemaValidator, IdempotencyCheck, TelemetryProjection, GeoIPEnrichment, SanctionsScreen, RuleBasedDecision, EmailVerifyDispatch, AuditWriter]`. Note: `SchemaValidator`, `IdempotencyCheck`, `TelemetryProjection`, `RuleBasedDecision`, `AuditWriter` are reused — proof of stage composition. |
| 6 | `RegistrationTelemetryProjector` produces a different subset of primitives: `GeographyAnomaly, NetworkTrust, DeviceTrust, VelocityAnomaly, RateLimitProximity, ThreatScore`. Pipeline emits a registration decision (Approve / SilentReject / Quarantine / Challenge). Demo: registration from a sanctioned-country IP gets `SilentReject`. |

**At end of Week 6, the abstraction is proven.** Two completely different events flow through the same infrastructure.

#### Phase 4 — Generic Policy Engine (Weeks 7–8)

**Goal**: Replace `RuleBasedDecision` with a real Policy Engine that fetches policies from Redis.

| Week | Deliverable |
|---|---|
| 7 | Policy Engine service. Postgres-backed policy store. Redis cache (`policy:{tenant}:{domain}:current`). Hot-reload via Kafka topic `aepo.policy.updates`. Caffeine cache in Orchestrator. |
| 8 | `StrategyDecision` stage replaces the hardcoded rule. It reads the cached policy, applies the action mask from `DomainPlugin.actionMaskProvider()`, and selects a strategy. Both domains now share the same decision stage. |

#### Phase 5 — RL Inference (Weeks 9–10)

**Goal**: A trained RL policy makes the decisions, ported from the hackathon Q-table to a generic format.

| Week | Deliverable |
|---|---|
| 9 | Train a Q-table on the AEPO simulator (existing). Export to a generic policy format: a JSON map from `(domain, telemetry_bucket) -> action_distribution`. Load into Redis at startup. `StrategyDecision` looks up. Domain-1 (UPI) uses learned policy; domain-2 still uses rule for Phase 5a. |
| 10 | Train a separate Q-table for registration on a synthetic environment (you'll write a 200-line Python sim of the registration flow with botnet/legit/sanctioned traffic mix). Export to same format. Load into Redis under different key. Both domains now use RL. |

#### Phase 6 — Resilience and Audit (Week 11)

| Week | Deliverable |
|---|---|
| 11 | Outbox pattern for audit events. Merkle-tree commit job per minute. Replay endpoint: `GET /v1/audit/replay/{envelope_id}` returns the exact feature vector + policy version + action distribution that produced the decision. Bulkhead test: hammer registration at 10x baseline; UPI p99 must stay flat. |

#### Phase 7 — Demo (Week 12)

| Week | Deliverable |
|---|---|
| 12 | "Cross-domain decision platform" demo. Two browser tabs: one shows UPI traffic decisions in real time; the other shows registration decisions. Trigger a botnet wave on registration → watch Orchestrator route it to `SilentReject`; UPI stays unaffected. Trigger a Kafka lag spike → watch UPI route to `Throttle`; registration stays unaffected. Pitch deck v1. Demo video <8 minutes. |

### 5.3 Anti-scope-creep rules (read these every Friday)

You will be tempted to build any of these in V1. Don't.

- ❌ Cell-based deployment (single cell is fine for MVA)
- ❌ Hierarchical RL with shared encoder (start with separate Q-tables per domain — prove the SPI first)
- ❌ Schema Registry integration (Avro schemas in classpath is fine for MVA; integrate Schema Registry in V1.5)
- ❌ Counterfactual policy evaluation (V1.5)
- ❌ TelemetryPredictor world model (V2 — demonstrate it on the existing UPI sim only)
- ❌ Policy Studio UI (Postgres table + admin endpoint for MVA)
- ❌ Multi-cell tenant routing (V2)
- ❌ Cross-region replication (V3)
- ❌ Real third-party domain plugins beyond payments + identity (V1.5+, with a design partner)

The discipline: **one engineer ships an abstraction by proving it across two domains, then sells the platform on extensibility.** If you build seven domains, you have seven verticals and zero platform. If you build two domains plus a clean SPI plus great docs, you have a platform.

### 5.4 Post-MVA roadmap (months 4–12) — for the founder pitch

- **Months 4–5**: Plugin SDK + onboarding docs. First design partner integration (e.g., a payments customer adding a domain plugin themselves).
- **Months 6–7**: Counterfactual evaluation + canary deployment. Shared encoder + per-domain RL heads.
- **Months 8–9**: Cell-based architecture rollout. First multi-cell production deployment.
- **Months 10–12**: Policy Studio v1, marketplace for community-contributed plugins, first commercial launch.

---

## 6. Strategic positioning — where this sits in the market

### 6.1 Competitive map

| Vendor | Domain coverage | Programmability | Pricing | AEPO advantage |
|---|---|---|---|---|
| **Sift** ([sift.com](https://sift.com/)) | Fraud only (digital + payments) | Black-box ML; rule editor | Custom, ~$200k+ ARR ([Vendr](https://www.vendr.com/buyer-guides/sift-science)) | Multi-domain, transparent per-decision pricing |
| **Castle** | Auth/account abuse only | API + rules | Tiered SaaS | Multi-domain |
| **Cloudflare Bot Management** | Bot/API only | Limited rules | Cloudflare bundle | Open infra (not Cloudflare-locked), payment-class workloads |
| **AWS Fraud Detector** | Fraud only, AWS-bound | AWS Console + SDK | Per-prediction | Cloud-portable, multi-domain, RL-native |
| **Hyperswitch / Juspay** | Payments only | OSS + commercial | Self-host or SaaS | Decision/risk layer above Hyperswitch — complementary |
| **Stripe Radar / Adaptive Acceptance** | Stripe payments only | Stripe-bound | Per-txn within Stripe | Cross-PSP, cross-domain |
| **Datadog Bits AI SRE** | SRE explanation | Limited action | Bundle | AEPO acts; Bits explains. Different products. |

The shaped hole: **a programmable, multi-domain, cross-cloud decision platform with transparent pricing and an open extension SDK.** Nobody owns it.

### 6.2 Pricing model

Three pricing layers:

1. **Per-decision API** — $0.0001 to $0.001 per decision depending on tier; transparent and self-service. (Sift's opacity is a big-customer comfort but a small-customer dealbreaker.)
2. **Subscription per active domain** — $1k–10k/month per domain in production, includes plugin support and advisory hours.
3. **Premium dedicated cell** — $50k–500k+ ARR; SLA 99.99%, dedicated infra, custom plugins.

The differentiator versus Sift's $200k floor: a developer with a credit card can go to production with AEPO in an afternoon. That's a Stripe-style PLG motion in a category that doesn't have one yet.

### 6.3 Defensibility

- **The plugin marketplace** — once 50+ community plugins exist, the platform's gravity is irreducible.
- **The shared encoder's transfer advantage** — every customer's traffic improves every other customer's policies (privacy-preserving via federated learning + DP, contract-permitted).
- **The audit chain** — once a customer ties their regulatory reporting to AEPO's Merkle commits, switching cost is measured in years, not quarters.

---

## Sources

- [Slack Engineering — Migration to a Cellular Architecture](https://slack.engineering/slacks-migration-to-a-cellular-architecture/)
- [AWS Well-Architected — Reducing Scope of Impact with Cell-Based Architecture](https://docs.aws.amazon.com/wellarchitected/latest/reducing-scope-of-impact-with-cell-based-architecture/cell-deployment.html)
- [AWS Solutions Library — Cell-Based Architecture Guidance](https://github.com/aws-solutions-library-samples/guidance-for-cell-based-architecture-on-aws)
- [Mayank Raj — Cell-Based Architecture: When Isolation Beats Integration](https://mayankraj.com/blog/cell-based-architecture-blast-radius-containment/)
- [Confluent — Schema Registry Best Practices](https://www.confluent.io/blog/best-practices-for-confluent-schema-registry/)
- [Confluent — Schema Registry Concepts](https://docs.confluent.io/platform/current/schema-registry/fundamentals/index.html)
- [Tom Kaszuba — Avro Schema Evolution Strategies on Kafka](https://tkaszuba.medium.com/avro-schema-evolution-strategies-on-kafka-3c072a9a5347)
- [Javarevisited — The Strategy Design Pattern with Spring Plugin](https://medium.com/javarevisited/the-strategy-design-pattern-with-spring-plugin-e99021c8f6eb)
- [DZone — Building a Dynamic Rules Engine in Spring Boot](https://medium.com/@venkatsai0398/building-a-dynamic-rules-engine-in-spring-boot-with-the-strategy-registry-pattern-c8bafacc1031)
- [InfoQ — ONNX AI Inference in Java](https://www.infoq.com/articles/onnx-ai-inference-with-java/)
- [ONNX Runtime — Java Getting Started](https://onnxruntime.ai/docs/get-started/with-java.html)
- [Sift — Platform Overview](https://sift.com/platform/)
- [Vendr — Sift Science Pricing 2025](https://www.vendr.com/buyer-guides/sift-science)
- [Datadog Bits AI SRE](https://www.datadoghq.com/product/ai/bits-ai-sre/)
- [Redis — Feature Stores for Real-Time AI/ML](https://redis.io/blog/feature-stores-for-real-time-artificial-intelligence-and-machine-learning/)
- [Hyperswitch — Open-source Payment Orchestrator](https://hyperswitch.io/)
