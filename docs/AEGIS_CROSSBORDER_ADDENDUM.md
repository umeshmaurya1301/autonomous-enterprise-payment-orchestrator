# Aegis Orchestrator — Cross-Border, Universal Schema & Value-Based Shedding

> **Addendum to**: `docs/PRODUCT_STRATEGY.md` and `docs/UNIVERSAL_PLATFORM_STRATEGY.md`
> **Brand consolidation**: Going forward, the platform is named **Aegis Orchestrator**. The hackathon code-name AEPO is preserved as the simulation/RL-environment subsystem (the "Aegis Sim Core"). All product-facing references use Aegis.
> **Hackathon**: DevNetwork AI Hackathon
> **Author**: CPO + Principal Distributed Systems Architect synthesis for Umesh Maurya
> **Date**: 2026-05-07

---

## Why this addendum exists

The previous strategy docs proved Aegis is **a domain-agnostic decision platform**. This addendum proves it survives the hardest possible mixed workload: **a single instance routing $1 UPI scans and $50,000 SWIFT transfers through the same brain**, choosing optimally even when infrastructure is degraded. This is the demo that wins a fintech-AI hackathon, and it's also the V2 thesis that opens enterprise-bank conversations.

Three things below:
1. **The Cross-Border Risk Triad** — how the original triad shifts when transactions stop being millisecond-scale and start being multi-day, multi-currency, regulator-watched.
2. **The Universal Data Contract** — a concrete Avro schema that ingests UPI and SWIFT events identically, lets the RL agent read both without knowing the rail.
3. **Value-Based Load Shedding** — the math behind why the agent learns to drop 1,000 UPI to save one SWIFT during an outage, with no rule explicitly written for it.

A short bonus section scripts the 8-minute demo that demonstrates all three live.

---

## 1. The Cross-Border Risk Triad

### 1.1 The original triad doesn't survive contact with SWIFT

The hackathon AEPO triad — **Fraud × Infra × SLA** — was calibrated for UPI-class workloads:
- *Fraud*: probability of an unauthorized payment, milliseconds-to-detect, ~₹100–₹50,000 per loss event.
- *Infra*: Kafka lag, DB pool, HSM saturation — failure visible in seconds.
- *SLA*: P99 latency budget, 200–500ms.

Drop a SWIFT MT103 corporate transfer onto that triad and every variable changes by 2–6 orders of magnitude:

| Axis | UPI character | Cross-border SWIFT character | Magnitude shift |
|---|---|---|---|
| Risk velocity | seconds | days (settlement window T+0 to T+2) | 10⁵× slower |
| Loss event size | ₹50K typical worst case | $50K–$50M typical | 10²–10⁴× larger |
| Compliance penalty | 0–₹10L (PMLA, RBI) | $1B–$10B (HSBC $1.9B 2012, BNP Paribas $8.9B 2014, Standard Chartered $1.1B 2019) | 10⁵× larger |
| Counterparties | 1 (NPCI) + 2 banks | 4–7 banks (originator, correspondent, intermediary, beneficiary) | 3–7× more hops |
| Regulators | RBI, NPCI | OFAC, FinCEN, EU AML, FATF, RBI, beneficiary-country regulator | 6× more |
| Throughput | 16B/month UPI | ~38M/day SWIFT globally | 10²× lower volume |
| FX exposure | None (INR↔INR) | 0.5–3% during the settlement window | New axis |

A naive transposition of the UPI triad onto SWIFT produces a policy that optimizes the wrong things. The triad must be redefined.

### 1.2 The Cross-Border Risk Triad — formal definition

For cross-border high-value rails, the three axes become:

#### **Axis 1 — AML/OFAC Sanctions Risk** *(replaces "Fraud")*

Not "is this a fraudulent payment by a stolen credential?" but "**is approving this payment a federal crime?**" The screen list is multi-jurisdictional:

- **OFAC SDN** (Specially Designated Nationals — US Treasury) — ~13,000 entities
- **EU Consolidated Sanctions** — ~5,000 entities
- **UN 1267** (terrorism), UN 1718 (DPRK), UN 1737 (Iran) — overlapping
- **HMT (UK), AUSTRAC (Australia), MAS (Singapore)** — country-specific
- **PEP lists** (Politically Exposed Persons) — risk-tier match, not auto-block
- **Adverse media** — fuzzy match against negative news mentions

Failure modes:
- **Type-1 error (false approve)**: process a sanctioned-entity payment → criminal liability, billion-dollar fines, executive personal liability under Bank Secrecy Act.
- **Type-2 error (false reject)**: block a legitimate corporate transfer → multi-million-dollar customer relationship damage; corporate clients change banks.

The economics are radically asymmetric: the *expected cost of a single Type-1 error* dominates the *expected cost of thousands of Type-2 errors*. Any policy that doesn't reflect this asymmetry in the reward function will fail in production.

#### **Axis 2 — Nostro Account Liquidity** *(replaces "Infra")*

The "infrastructure" of cross-border isn't Kafka. It's **pre-funded correspondent bank accounts**.

- A bank in India that wants to settle USD payments holds USD in a *Nostro account* at a US correspondent (e.g., JP Morgan Chase, Citi, BNY Mellon).
- That account has a balance — call it **L** (liquidity, in USD).
- Each outbound USD payment debits L.
- L is replenished asynchronously (typically T+1 from FX desk hedging).
- If L falls below threshold L_min, no further USD payments can settle until replenishment.

This makes liquidity a **finite, depletable resource exactly like a DB connection pool**, but with a refill horizon of *hours to days* instead of seconds. The RL agent needs to learn:

- *"I have $2.3M of USD Nostro liquidity. There are 47 pending USD payments queued. If I approve all of them, I'll deplete L below L_min by 14:00, blocking the next 6 hours of USD throughput. Optimal: approve the top-K by Value-at-Risk-weighted utility, defer the rest."*

This is **inventory management**, not infrastructure routing. The math is closer to airline revenue management (yield management on perishable seats) than to Kafka backpressure.

Telemetry primitives (mapping to §2.1 of the previous platform doc):
- `ResourceUtilization` → `nostro_liquidity_usd / nostro_threshold_usd`
- `QueueDepth` → number of pending USD payments awaiting settlement
- `EntityFreshness` → time-since-last-replenishment of the Nostro account

#### **Axis 3 — FX Volatility & Settlement Window** *(replaces "SLA")*

UPI SLA is "the user got their money in 10 seconds." SWIFT SLA is "the beneficiary received the right amount, in the right currency, by T+2 settlement, at the FX rate the originator was quoted."

Two new sub-axes:

**(a) FX exposure during the float.**
At t=0, originator is quoted USD/INR = 83.45. At t=2 days, actual settlement rate may be 83.20 or 83.70. If we lock the rate at quote time (forward contract) we eat hedging cost. If we don't, we eat slippage. Either way, **every SWIFT payment carries a residual FX-risk variable**.

The agent's decision is no longer just "approve / reject" — it's "approve at quoted rate (we hedge), approve at floating rate (customer hedges), or defer to next FX-window batch (cheapest hedging cost)." Three actions, each with different (Outcome × Resource × Experience) trade-offs.

**(b) Settlement window cliff.**
Cross-border has hard cut-off times — *FX window closes 14:00 IST for same-day settlement; after that, settlement defers to T+1 at uncertain rate*. A SWIFT payment arriving at 13:55 has a different optimal action than one arriving at 14:05.

Telemetry primitives:
- `LatencyEMA` → reframed as `time_to_settlement_window_close`
- New primitive: `FXVolatilityIndex` (rolling stddev of the relevant pair) — feeds the agent's decision on hedge/no-hedge
- `EntropyEMA` → cross-currency rate flicker

### 1.3 Why mixing UPI + SWIFT in one orchestrator is a billion-dollar engineering problem

Three reasons that compound:

**(a) The optimization function is non-convex over heterogeneous economics.**
A policy that's optimal for ₹100 UPI is *catastrophic* for $50K SWIFT. The solution is not "two separate orchestrators" — that's the trivial, bad answer that loses the platform thesis. The right solution is **a single agent whose state vector includes value/margin/compliance-class as features, and whose reward function is value-weighted**, so the same model learns both regimes and the trade-off between them.

**(b) Resource starvation across orders of magnitude.**
A botnet flood of 50,000 spam UPI events/sec consumes the same DB pool, the same Kafka backbone, the same HSM cycles, the same Sanctions Screen API quota that one $50K SWIFT needs. Without **value-aware admission control**, the spam wins by sheer volume — the SWIFT times out and your bank loses a corporate client. (See §3.)

**(c) Compliance asymmetry.**
A UPI false-approve costs ₹100. A SWIFT false-approve to a sanctioned entity costs $10B and personal criminal liability. **The reward function must encode this asymmetry as a multiplicative penalty term, not a flat scalar.** Otherwise the agent will trade compliance for revenue.

Solving (a)+(b)+(c) in **one inference engine** is the moat. Sift, Hyperswitch, Stripe Adaptive Acceptance — none of them attempt this. The closest analogue is what Wise built internally for their cross-border product, but that's vertically integrated and not a platform anyone else can buy.

---

## 2. The Universal Data Contract

### 2.1 Design principles (non-negotiable)

1. **One Avro schema covers every payment rail** — UPI, IMPS, RTGS, Cards, Tap-Pay, SWIFT, ACH, SEPA, FedNow.
2. **The RL agent reads one face**: `risk_context` + universal payment metadata. It must never need to know the rail to make a decision.
3. **Domain-specific data lives in a payload union**, opaque to the platform but typed for stages that need it (e.g., the SWIFT compliance stage needs MT103 fields).
4. **Value is normalized to a comparable unit at ingest**: USD minor units (cents). FX-converted at ingest using a frozen rate snapshot recorded inline.
5. **Schema evolves with `BACKWARD_TRANSITIVE` compatibility** so consumers older than the producer don't break.

### 2.2 The universal payment event — three-layer structure

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — Envelope (universal, identical across all events)          │
│   envelope_id, tenant_id, event_type, schema_version, ingest_ts,     │
│   processing_deadline_ms, qos, idempotency_key, correlation_id       │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 2 — Universal Payment Metadata + Risk Context                  │
│   payment: { rail, value_minor_units, currency,                      │
│              value_normalized_usd_minor_units, fx_rate_snapshot,     │
│              settlement_window_seconds, counterparty_chain }         │
│   risk_context: { outcome_risk_score, value_at_risk_usd,             │
│                   compliance_flags, regulatory_exposure_class,       │
│                   liquidity_class, fx_volatility_index }             │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 3 — Domain Payload (union — exactly one of)                    │
│   UpiPayload | CardPayload | SwiftPayload | AchPayload | SepaPayload │
└──────────────────────────────────────────────────────────────────────┘
```

The RL agent reads Layer 1 + Layer 2. The domain plugin reads Layer 3.

### 2.3 The Avro schema — full deliverable

```json
{
  "type": "record",
  "name": "UniversalPaymentEvent",
  "namespace": "com.aegis.event.payment.v1",
  "doc": "The universal payment event. Same schema for UPI ($1) and SWIFT ($50k).",
  "fields": [

    {
      "name": "envelope",
      "doc": "Universal envelope. Identical structure for every event class.",
      "type": {
        "type": "record",
        "name": "Envelope",
        "fields": [
          {"name": "envelope_id",          "type": "string", "doc": "ULID, ingestion-assigned, globally unique"},
          {"name": "tenant_id",            "type": "string"},
          {"name": "idempotency_key",      "type": "string"},
          {"name": "correlation_id",       "type": ["null", "string"], "default": null},
          {"name": "causation_id",         "type": ["null", "string"], "default": null},
          {"name": "domain",               "type": "string", "doc": "e.g. 'payments.upi', 'payments.swift'"},
          {"name": "event_type",           "type": "string", "doc": "e.g. 'payments.swift.mt103.send'"},
          {"name": "schema_version",       "type": "int"},
          {"name": "ingest_ts_ms",         "type": "long", "logicalType": "timestamp-millis"},
          {"name": "event_ts_ms",          "type": "long", "logicalType": "timestamp-millis"},
          {"name": "processing_deadline_ms","type": "long", "doc": "Hard deadline for hot-path processing"},
          {"name": "qos",                  "type": {"type": "enum", "name": "QoS",
                                                    "symbols": ["BEST_EFFORT", "STANDARD", "PREMIUM", "BATCH"]}},
          {"name": "origin_service",       "type": "string"},
          {"name": "origin_signature",     "type": ["null", "bytes"], "default": null,
                                            "doc": "mTLS-derived ingestion signature"}
        ]
      }
    },

    {
      "name": "payment",
      "doc": "Universal payment metadata — readable by the RL agent without knowing the rail.",
      "type": {
        "type": "record",
        "name": "PaymentMetadata",
        "fields": [
          {"name": "rail", "type": {"type": "enum", "name": "PaymentRail",
                                    "symbols": ["UPI", "IMPS", "RTGS", "NEFT",
                                                "CARD_VISA", "CARD_MASTERCARD", "CARD_RUPAY",
                                                "TAP_PAY_NFC", "WALLET_TOKEN",
                                                "SWIFT_MT103", "SWIFT_MX_PACS_008",
                                                "ACH_US", "SEPA_SCT", "SEPA_INST",
                                                "FEDNOW", "FPS_UK", "PIX_BR"]}},
          {"name": "rail_class", "type": {"type": "enum", "name": "RailClass",
                                          "symbols": ["DOMESTIC_INSTANT", "DOMESTIC_BATCH",
                                                      "CARD_AUTH", "CROSS_BORDER_HIGH_VALUE",
                                                      "CROSS_BORDER_RETAIL"]},
                                  "doc": "Coarse class — used by RL for value-class features"},

          {"name": "value_minor_units", "type": "long",
                                       "doc": "Native currency, minor units (e.g., paise, cents)"},
          {"name": "currency_iso4217", "type": "string", "doc": "INR, USD, EUR, GBP..."},
          {"name": "value_normalized_usd_minor_units", "type": "long",
                                                       "doc": "FX-converted at ingest. Single comparable scale across all rails."},

          {"name": "fx_snapshot", "type": ["null", {
              "type": "record", "name": "FxSnapshot",
              "fields": [
                {"name": "pair",            "type": "string", "doc": "e.g., 'USD/INR'"},
                {"name": "rate_bps",        "type": "long", "doc": "Rate × 10000 (e.g., USD/INR 83.45 → 834500)"},
                {"name": "rate_source",     "type": "string"},
                {"name": "rate_observed_ts_ms", "type": "long"},
                {"name": "volatility_30d_bps", "type": ["null", "int"], "default": null}
              ]
          }], "default": null, "doc": "Null for same-currency rails"},

          {"name": "settlement_window_seconds", "type": "long",
                                                "doc": "UPI≈10, RTGS≈600, SEPA-INST≈10, SWIFT≈172800 (T+2)"},
          {"name": "settlement_target_ts_ms", "type": "long",
                                              "doc": "Latest acceptable settlement"},

          {"name": "counterparty_chain", "type": {"type": "array", "items": {
              "type": "record", "name": "Counterparty",
              "fields": [
                {"name": "role",           "type": {"type": "enum", "name": "CpRole",
                                                    "symbols": ["ORIGINATOR", "ORIGINATOR_BANK",
                                                                "CORRESPONDENT", "INTERMEDIARY",
                                                                "BENEFICIARY_BANK", "BENEFICIARY"]}},
                {"name": "identifier",     "type": "string", "doc": "BIC/IFSC/VPA/UPI handle/IBAN"},
                {"name": "jurisdiction_iso3166","type": "string"},
                {"name": "trust_score",    "type": ["null", "double"], "default": null}
              ]}},
              "doc": "Length 2 for UPI, 4–7 for SWIFT"},

          {"name": "merchant_or_payee_tier", "type": ["null", "string"], "default": null}
        ]
      }
    },

    {
      "name": "risk_context",
      "doc": "Universal feature surface for the RL agent. Computed at ingest by the TelemetryProjector.",
      "type": {
        "type": "record",
        "name": "RiskContext",
        "fields": [
          {"name": "outcome_risk_score", "type": "double",
                                         "doc": "[0,1] — P(bad outcome | approve). Domain-projector populates."},
          {"name": "outcome_risk_confidence", "type": "double", "default": 1.0},

          {"name": "value_at_risk_usd_minor_units", "type": "long",
                                                    "doc": "value_normalized × outcome_risk_score, in USD cents"},

          {"name": "compliance_flags", "type": {"type": "array", "items": "string"},
                                       "doc": "['OFAC_SDN_HIT','PEP_MEDIUM','ADVERSE_MEDIA','SANCTIONED_JURISDICTION', ...]"},
          {"name": "regulatory_exposure_class", "type": {"type": "enum", "name": "RegExposure",
                                                          "symbols": ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                                                "doc": "CRITICAL = sanctions-risk; treated specially by reward function"},

          {"name": "liquidity_class", "type": {"type": "enum", "name": "LiquidityClass",
                                                "symbols": ["UNCONSTRAINED", "AMPLE", "MODERATE", "TIGHT", "CRITICAL"]},
                                       "doc": "Nostro-account headroom for the destination currency, bucketed"},
          {"name": "fx_volatility_index", "type": ["null", "double"], "default": null,
                                          "doc": "[0,1] — null for same-currency rails"},

          {"name": "telemetry_primitives", "type": {"type": "map", "values": {
              "type": "record", "name": "TelemetryReading",
              "fields": [
                {"name": "raw",         "type": "double"},
                {"name": "normalized",  "type": "double", "doc": "[0,1]"},
                {"name": "confidence",  "type": "double", "default": 1.0},
                {"name": "observed_ts_ms", "type": "long"}
              ]}},
              "doc": "Keys: 'QueueDepth', 'ThreatScore', 'ResourceUtilization', etc. — see TelemetryPrimitive registry"}
        ]
      }
    },

    {
      "name": "payload",
      "doc": "Domain-specific payload. Union — exactly one set. Opaque to RL agent; typed for domain stages.",
      "type": [
        {
          "type": "record", "name": "UpiPayload", "namespace": "com.aegis.event.payment.upi.v1",
          "fields": [
            {"name": "payer_vpa",            "type": "string"},
            {"name": "payee_vpa",            "type": "string"},
            {"name": "txn_type",             "type": {"type": "enum", "name": "UpiTxnType",
                                                      "symbols": ["P2P", "P2M", "AUTOPAY", "MANDATE", "REFUND"]}},
            {"name": "device_fingerprint",   "type": ["null", "string"], "default": null},
            {"name": "psp_handle",           "type": "string"},
            {"name": "merchant_category_code","type": ["null", "string"], "default": null}
          ]
        },
        {
          "type": "record", "name": "CardPayload", "namespace": "com.aegis.event.payment.card.v1",
          "fields": [
            {"name": "card_token",           "type": "string"},
            {"name": "scheme",               "type": "string"},
            {"name": "auth_type",            "type": {"type": "enum", "name": "CardAuthType",
                                                      "symbols": ["AUTH", "CAPTURE", "REFUND", "VOID", "VERIFY"]}},
            {"name": "three_ds_status",      "type": ["null", "string"], "default": null},
            {"name": "merchant_id",          "type": "string"},
            {"name": "mcc",                  "type": "string"}
          ]
        },
        {
          "type": "record", "name": "SwiftPayload", "namespace": "com.aegis.event.payment.swift.v1",
          "doc": "Models MT103 + ISO 20022 pacs.008 fields the platform/RL needs (subset).",
          "fields": [
            {"name": "message_type",         "type": {"type": "enum", "name": "SwiftMessageType",
                                                      "symbols": ["MT103", "MT202", "MX_PACS_008", "MX_PACS_009"]}},
            {"name": "uetr",                 "type": "string", "doc": "Unique End-to-end Transaction Reference"},
            {"name": "originator_bic",       "type": "string"},
            {"name": "beneficiary_bic",      "type": "string"},
            {"name": "originator_country",   "type": "string", "doc": "ISO 3166-1 alpha-2"},
            {"name": "beneficiary_country",  "type": "string"},
            {"name": "purpose_code",         "type": ["null", "string"], "default": null,
                                              "doc": "ISO 20022 ExternalPurpose1Code (e.g., 'GDDS','SUPP')"},
            {"name": "remittance_information","type": ["null", "string"], "default": null},
            {"name": "originator_name",      "type": "string"},
            {"name": "beneficiary_name",     "type": "string"},
            {"name": "originator_address_country", "type": "string"},
            {"name": "beneficiary_address_country","type": "string"},
            {"name": "intermediary_chain",   "type": {"type": "array", "items": "string"},
                                              "doc": "Ordered list of intermediary BICs"},
            {"name": "regulatory_reporting_codes","type": {"type": "array", "items": "string"}},
            {"name": "fx_instruction",       "type": {"type": "enum", "name": "FxInstruction",
                                                      "symbols": ["LOCK_AT_QUOTE", "FLOAT_TO_SETTLEMENT", "DEFER_TO_BATCH"]}}
          ]
        },
        {
          "type": "record", "name": "AchPayload", "namespace": "com.aegis.event.payment.ach.v1",
          "fields": [
            {"name": "sec_code",             "type": "string"},
            {"name": "originator_routing",   "type": "string"},
            {"name": "receiver_routing",     "type": "string"},
            {"name": "trace_number",         "type": "string"}
          ]
        },
        {
          "type": "record", "name": "SepaPayload", "namespace": "com.aegis.event.payment.sepa.v1",
          "fields": [
            {"name": "sepa_scheme",          "type": {"type": "enum", "name": "SepaScheme",
                                                      "symbols": ["SCT", "SCT_INST", "SDD_CORE", "SDD_B2B"]}},
            {"name": "originator_iban",      "type": "string"},
            {"name": "beneficiary_iban",     "type": "string"},
            {"name": "end_to_end_id",        "type": "string"}
          ]
        }
      ]
    }
  ]
}
```

### 2.4 The two-event walkthrough

#### **Event A — $1 UPI scan (₹83 at 83.00 INR/USD)**

```json
{
  "envelope": {
    "envelope_id": "01HXY7Z9P2K3M4N5Q6R7S8T9V0",
    "tenant_id": "acme_pa_001",
    "idempotency_key": "upi-collect-99887766",
    "domain": "payments.upi",
    "event_type": "payments.upi.collect_request",
    "schema_version": 1,
    "ingest_ts_ms": 1746604800123,
    "event_ts_ms":  1746604800100,
    "processing_deadline_ms": 200,
    "qos": "STANDARD",
    "origin_service": "upi-ingest-adapter"
  },
  "payment": {
    "rail": "UPI",
    "rail_class": "DOMESTIC_INSTANT",
    "value_minor_units": 8300,
    "currency_iso4217": "INR",
    "value_normalized_usd_minor_units": 100,
    "fx_snapshot": null,
    "settlement_window_seconds": 10,
    "settlement_target_ts_ms": 1746604810000,
    "counterparty_chain": [
      {"role": "ORIGINATOR",       "identifier": "rohan@oksbi", "jurisdiction_iso3166": "IN"},
      {"role": "BENEFICIARY",      "identifier": "kirana@okhdfc","jurisdiction_iso3166": "IN"}
    ],
    "merchant_or_payee_tier": "SMALL"
  },
  "risk_context": {
    "outcome_risk_score": 0.07,
    "outcome_risk_confidence": 0.96,
    "value_at_risk_usd_minor_units": 7,
    "compliance_flags": [],
    "regulatory_exposure_class": "NEGLIGIBLE",
    "liquidity_class": "UNCONSTRAINED",
    "fx_volatility_index": null,
    "telemetry_primitives": {
      "QueueDepth":           {"raw": 1850,  "normalized": 0.46, "confidence": 1.0, "observed_ts_ms": 1746604800100},
      "ThreatScore":          {"raw": 0.07,  "normalized": 0.07, "confidence": 0.96, "observed_ts_ms": 1746604800100},
      "ResourceUtilization":  {"raw": 0.42,  "normalized": 0.42, "confidence": 1.0,  "observed_ts_ms": 1746604800100},
      "LatencyEMA":           {"raw": 187,   "normalized": 0.23, "confidence": 1.0,  "observed_ts_ms": 1746604800100}
    }
  },
  "payload": {
    "com.aegis.event.payment.upi.v1.UpiPayload": {
      "payer_vpa": "rohan@oksbi",
      "payee_vpa": "kirana@okhdfc",
      "txn_type": "P2M",
      "psp_handle": "okhdfc",
      "merchant_category_code": "5411"
    }
  }
}
```

#### **Event B — $50,000 SWIFT MT103 (USD)**

```json
{
  "envelope": {
    "envelope_id": "01HXY7Z9P2K3M4N5Q6R7S8T9V1",
    "tenant_id": "acme_pa_001",
    "idempotency_key": "swift-mt103-44556677",
    "domain": "payments.swift",
    "event_type": "payments.swift.mt103.send",
    "schema_version": 1,
    "ingest_ts_ms": 1746604800200,
    "event_ts_ms":  1746604800180,
    "processing_deadline_ms": 5000,
    "qos": "PREMIUM",
    "origin_service": "swift-ingest-adapter"
  },
  "payment": {
    "rail": "SWIFT_MT103",
    "rail_class": "CROSS_BORDER_HIGH_VALUE",
    "value_minor_units": 5000000,
    "currency_iso4217": "USD",
    "value_normalized_usd_minor_units": 5000000,
    "fx_snapshot": {
      "pair": "USD/INR",
      "rate_bps": 834500,
      "rate_source": "REUTERS_RIC_USDINR=R",
      "rate_observed_ts_ms": 1746604800050,
      "volatility_30d_bps": 47
    },
    "settlement_window_seconds": 172800,
    "settlement_target_ts_ms": 1746777600000,
    "counterparty_chain": [
      {"role": "ORIGINATOR",       "identifier": "ACME-CORP-IND-7788",  "jurisdiction_iso3166": "IN"},
      {"role": "ORIGINATOR_BANK",  "identifier": "HDFCINBB",            "jurisdiction_iso3166": "IN"},
      {"role": "CORRESPONDENT",    "identifier": "CHASUS33",            "jurisdiction_iso3166": "US"},
      {"role": "BENEFICIARY_BANK", "identifier": "DEUTDEFF",            "jurisdiction_iso3166": "DE"},
      {"role": "BENEFICIARY",      "identifier": "BERLIN-VENDOR-GMBH",  "jurisdiction_iso3166": "DE"}
    ],
    "merchant_or_payee_tier": "ENTERPRISE"
  },
  "risk_context": {
    "outcome_risk_score": 0.012,
    "outcome_risk_confidence": 0.98,
    "value_at_risk_usd_minor_units": 60000,
    "compliance_flags": ["PEP_LOW", "ADVERSE_MEDIA_NONE"],
    "regulatory_exposure_class": "MEDIUM",
    "liquidity_class": "MODERATE",
    "fx_volatility_index": 0.31,
    "telemetry_primitives": {
      "QueueDepth":           {"raw": 1850,    "normalized": 0.46, "confidence": 1.0,  "observed_ts_ms": 1746604800180},
      "ThreatScore":          {"raw": 0.012,   "normalized": 0.012,"confidence": 0.98, "observed_ts_ms": 1746604800180},
      "ResourceUtilization":  {"raw": 0.42,    "normalized": 0.42, "confidence": 1.0,  "observed_ts_ms": 1746604800180},
      "LatencyEMA":           {"raw": 187,     "normalized": 0.23, "confidence": 1.0,  "observed_ts_ms": 1746604800180},
      "GeographyAnomaly":     {"raw": 0.0,     "normalized": 0.0,  "confidence": 1.0,  "observed_ts_ms": 1746604800180},
      "CounterpartyTrust":    {"raw": 0.91,    "normalized": 0.91, "confidence": 0.95, "observed_ts_ms": 1746604800180}
    }
  },
  "payload": {
    "com.aegis.event.payment.swift.v1.SwiftPayload": {
      "message_type": "MT103",
      "uetr": "f3a4b2c1-9876-5432-10ab-cdef01234567",
      "originator_bic": "HDFCINBB",
      "beneficiary_bic": "DEUTDEFF",
      "originator_country": "IN",
      "beneficiary_country": "DE",
      "purpose_code": "GDDS",
      "remittance_information": "INVOICE 2026-Q1-447 ACME-BERLIN VENDOR",
      "originator_name": "ACME CORP INDIA PVT LTD",
      "beneficiary_name": "BERLIN VENDOR GMBH",
      "originator_address_country": "IN",
      "beneficiary_address_country": "DE",
      "intermediary_chain": ["CHASUS33"],
      "regulatory_reporting_codes": ["P0103"],
      "fx_instruction": "LOCK_AT_QUOTE"
    }
  }
}
```

### 2.5 What the RL agent sees from each — and why this matters

The RL agent reads exactly the same fields from both: `payment.rail_class`, `payment.value_normalized_usd_minor_units`, `risk_context.*`, `risk_context.telemetry_primitives.*`. The agent never sees the `payload` — that's domain plugin territory.

The crucial signals the agent gets, side by side:

| Signal | UPI ($1) | SWIFT ($50k) |
|---|---|---|
| `value_normalized_usd_minor_units` | 100 | 5,000,000 |
| `value_at_risk_usd_minor_units` | 7 | 60,000 |
| `regulatory_exposure_class` | NEGLIGIBLE | MEDIUM |
| `liquidity_class` | UNCONSTRAINED | MODERATE |
| `outcome_risk_score` | 0.07 | 0.012 |

The 50,000:1 ratio in `value_normalized_usd_minor_units` is the entire reason the RL agent can learn value-based prioritization without anyone writing "if SWIFT prefer." The math in §3 below shows how this signal flows through the reward function to produce the right behavior under scarcity.

### 2.6 Implementation notes (Java/Kafka)

1. **Generate Java DTOs** via the Avro Maven plugin (`avro-maven-plugin` 1.11.x or 1.12.x). Each plugin module owns its schemas under `src/main/avro/`.
2. **Subject naming**: `aegis.tenant.<tenant_id>.domain.<domain>.event.<event_type>.v<n>-value`. (See §3.4 of the platform doc.)
3. **Compatibility**: `BACKWARD_TRANSITIVE` on envelope and risk_context; `BACKWARD` on payload unions (allows additive evolution, blocks removals without an alias).
4. **Serializer**: Confluent's `KafkaAvroSerializer` with `auto.register.schemas=false` in production (CI registers schemas explicitly so changes are git-tracked).
5. **FX rate snapshot freshness**: ingest adapter reads from a `fx-rates-cache` Redis key with 30s TTL; if stale, falls back to a primary FX feed and refreshes. Snapshot rate is *frozen into the event* — downstream stages never re-fetch; replay reproduces exactly.

---

## 3. Intelligent Value-Based Load Shedding — the math

The thesis: **when the system is degraded and capacity is finite, the RL agent learns — without explicit programming — that approving one $50k SWIFT is worth shedding 1,000 ₹100 UPIs.** This section shows how that emerges from the math.

### 3.1 The expected utility of a single decision

For any candidate decision *d* on event *i*, define the expected utility:

```
EU(d, i) = + value_i × P(success | d, state) × margin(rail_i)
           − value_i × P(failure | d, state) × loss_multiplier(d, i)
           − resource_cost(d) × shadow_price(state)
           − compliance_penalty(d, i)
```

Each term, unpacked:

#### `value_i × P(success | d, state) × margin(rail_i)` — **expected revenue**
- `value_i` from `payment.value_normalized_usd_minor_units` — single comparable scale.
- `P(success)` learned by the agent (or the world model) from historical traces.
- `margin(rail_i)` — typical effective basis points the platform earns on the rail. UPI ≈ 0.05 bps, IMPS ≈ 0.3 bps, Cards ≈ 50–200 bps, SWIFT ≈ 30–80 bps + FX spread of 50–200 bps. **The cross-border spread is where the asymmetry comes from.**

#### `value_i × P(failure) × loss_multiplier` — **expected loss**
- `loss_multiplier` is *per-failure-mode*. A revenue loss is 1×. A chargeback is 1.5–2×. A compliance failure (sanctions hit approved) is 100–10,000× depending on regulatory exposure class. From `risk_context.regulatory_exposure_class`:
  - NEGLIGIBLE → 1×
  - LOW → 2×
  - MEDIUM → 10×
  - HIGH → 100×
  - CRITICAL → 10,000× *(this is what makes the agent never approve under SDN match)*

#### `resource_cost(d) × shadow_price(state)` — **scarcity cost**
- `resource_cost(d)` — the cost of action *d* in the bottleneck resource (HSM ops, DB connections, Sanctions API quota, Nostro liquidity).
- `shadow_price(state)` — **the dual variable from the resource-constrained optimization. Under abundance, this is ≈0 (each op is free). Under scarcity, it rises rapidly — measured in `$ revenue lost per op`.**

This is the term that produces value-based shedding behavior. Under abundance, every approval has near-zero opportunity cost. Under scarcity, the opportunity cost equals *the next-best alternative use of that op*.

#### `compliance_penalty(d, i)` — **direct regulatory cost**
- For most decisions: 0.
- For approve-on-OFAC-hit: catastrophic, modeled as a hard constraint via action mask, not a penalty term. Defense in depth: even if the penalty term were bypassed, the action mask refuses.

### 3.2 The constrained optimization the agent learns to approximate

Under stable load, the platform has cell-level capacity *C* (ops/sec). Under degradation (Attack phase, partial outage), capacity drops to *C′ < C*. With *N* pending events, the agent's effective optimization is:

```
maximize    Σᵢ xᵢ · EU(approve, i) + Σᵢ (1 − xᵢ) · EU(shed, i)
subject to  Σᵢ xᵢ · cost(approve, i) ≤ C′
            xᵢ ∈ {0, 1}              ∀i
            xᵢ = 0                    if compliance_flags(i) ∩ HARD_BLOCK ≠ ∅
            Σᵢ_PREMIUM xᵢ = |PREMIUM events|   (PREMIUM never sheds)
```

This is **0/1 knapsack with side constraints**. NP-hard in the general case. The RL agent doesn't solve it exactly — it learns a *fractional* approximation: a value function *V(state)* such that the greedy policy on *V* converges to a near-optimal solution under the system's empirical workload distribution.

The Lagrangian relaxation gives intuition. Drop the integrality and write:

```
L(x, λ) = Σᵢ xᵢ · EU(approve, i) + λ · (C′ − Σᵢ xᵢ · cost(approve, i))
```

Differentiating w.r.t. *xᵢ*:

```
∂L/∂xᵢ = EU(approve, i) − λ · cost(approve, i)
```

So at optimum, the agent approves event *i* iff:

```
                                   EU(approve, i)
EU(approve, i) ≥ λ · cost(approve, i)  ⇔  ─────────────────  ≥ λ
                                       cost(approve, i)
```

**That ratio — expected utility per unit cost — is the *priority score*.** The agent ranks events by priority and approves down the list until the cost budget is exhausted. λ is the **shadow price** mentioned above; it's exactly the priority of the marginal event.

### 3.3 Worked example — UPI flood vs SWIFT

Setup: cell is degraded. Capacity *C′* = **100 HSM ops/sec**. Pending in this 1-second window:
- 1,000 UPI events @ $1 each, success-prob 0.99, margin 0.05 bps
- 1 SWIFT event @ $50,000, success-prob 0.99, margin 30 bps + FX spread 100 bps = 130 bps effective

Per-event expected utility (ignoring loss/compliance for clarity, both are similar small values):

```
EU(UPI)   = $1     × 0.99 × 0.0005   ≈ $0.000495    per approval
EU(SWIFT) = $50000 × 0.99 × 0.013    ≈ $643.50      per approval
```

Per-op cost: 1 HSM op for either rail.

Priority score (EU per op):

```
priority(UPI)   = $0.000495 / 1 = $0.000495 / op
priority(SWIFT) = $643.50   / 1 = $643.50   / op
```

**Ratio: 1,300,000 : 1.** The single SWIFT approval is worth ~1.3 million UPI approvals' worth of expected utility.

Under capacity *C′* = 100 ops/sec, the optimal policy is:

1. Approve the 1 SWIFT (1 op consumed; 99 ops remaining; $643.50 captured)
2. Approve 99 UPIs by priority (highest-utility UPIs first, e.g., higher-tier merchants)
3. Shed the remaining 901 UPIs

Total revenue captured: **$643.50 + 99 × $0.000495 ≈ $643.55**

Compare: if the agent treated all events equally and randomly approved 100, expected revenue:
```
E[revenue] = 100/1001 × ($643.50 + 1000 × $0.000495)
           ≈ 0.0999 × $643.99
           ≈ $64.32
```

The value-aware policy captures **10× more revenue** under the same capacity. That's the magnitude of the prize.

### 3.4 How the agent *learns* this without rules

No programmer wrote "prefer SWIFT." The agent infers it from three things baked into the training data:

1. **The state vector includes `payment.value_normalized_usd_minor_units` and `payment.rail_class`.** The agent sees value as a feature.
2. **The reward signal includes value-weighted revenue** (this is the only training-time choice that matters): when an approval succeeds, reward is `value × margin`; when it fails, penalty is `value × loss_multiplier`. So a $50k success is rewarded 50,000× more than a $1 success.
3. **The replay buffer contains episodes of capacity scarcity** (we generate them via the AEPO simulator's Attack phase). During scarcity, the agent observes that approving high-value events when ops are scarce produces high cumulative reward; approving low-value events when ops are scarce produces low cumulative reward.

Standard Q-learning / PPO converges to a value function where:

```
Q(state_with_high_value × scarce_resources, approve) >> Q(state_with_low_value × scarce_resources, approve)
```

…and the greedy policy on *Q* shows the value-aware shedding behavior. **The RL agent has rediscovered Lagrangian-priced knapsack solving from scratch.** Which is exactly the kind of result that makes a hackathon judge sit up.

### 3.5 Why this beats hand-coded "prefer SWIFT" rules

Three reasons a learned policy is strictly better than a hand-coded priority rule:

1. **Multi-dimensional priority.** A hand rule says "SWIFT > UPI." The learned policy says "SWIFT *with low Counterparty Trust during high Nostro liquidity* is worth less than UPI *from a high-tier merchant during a sanctions-screen-API outage that affects only USD payments*." It conditions on the full state vector, not the rail.
2. **Dynamic shadow prices.** The shadow price λ changes by the second as load changes. A hand rule has a static threshold. The learned policy has a state-dependent threshold that's correct at every moment.
3. **Extensibility for free.** Add a new rail (say, FedNow). The state vector picks up new feature values. The learned policy generalizes via the shared encoder. A hand rule needs a new line of `if/else` per rail, manually tuned, manually re-tuned at every bottleneck change.

### 3.6 Premium tier guarantee — non-negotiable carve-out

One important add: **PREMIUM-QoS events bypass the value-based shedding entirely.** They consume from a reserved capacity pool that's never sheddable. This is how we sell the platform — a regulated bank cannot ship a system where their RTGS payments might lose to a lucky high-EU non-RTGS event during congestion. The contract with PREMIUM-tier customers is: *we guarantee your event runs; if we can't, we trip the cell and roll over to the standby*.

Mathematically:

```
total capacity C' = C'_PREMIUM_RESERVED + C'_ELASTIC
```

`C'_PREMIUM_RESERVED` is sized to handle worst-case PREMIUM load + 30% headroom. Value-based shedding operates only within `C'_ELASTIC`. PREMIUM events are first-class; the math above applies to STANDARD and BEST_EFFORT.

---

## 4. The Hackathon Demo Script — 8 minutes that win

The demo must show, live and on stage:
- (a) one platform handling UPI + SWIFT identically
- (b) the agent acting differently on each based on universal features
- (c) the agent surviving a capacity-degradation event by value-based shedding

Time budget — strict.

### **0:00–1:00 — Setup the stakes**
*Slide:* "UPI processed 16.73B txns in Dec 2024. SWIFT moved $156T cross-border in 2023. No vendor routes both with one model. We do."
*Live screen:* Aegis dashboard showing two real-time event streams (`payments.upi.collect_request` and `payments.swift.mt103.send`), both flowing into one Orchestrator pod.

### **1:00–3:00 — The universal contract**
*Live:* open `Envelope`, `PaymentMetadata`, `RiskContext` fields side-by-side for a UPI event and a SWIFT event. Highlight `value_normalized_usd_minor_units` (100 vs 5,000,000) and `regulatory_exposure_class` (NEGLIGIBLE vs MEDIUM).
*Voiceover:* "Same envelope. Same risk context. Same code path. The agent reads the same fields and produces a domain-correct action."

### **3:00–5:00 — Routine routing**
*Live:* trigger 50 UPI/sec + 0.5 SWIFT/sec on the load gen.
- Show 95% UPI Approve, 0.1% Reject, the rest Throttle/Defer.
- Show 100% SWIFT Approve when no compliance flag, 100% Quarantine when one is injected (toggle a sanctioned-country test event).
*Talking point:* "Same agent. Different action mask per domain. Compliance constraint is enforced as a mask, not a soft penalty — that's why we're regulator-defensible."

### **5:00–7:00 — The capacity-degradation event (the hero moment)**
*Live:* trigger the Attack phase. HSM availability drops 60%. Capacity goes from 1,000 to 100 ops/sec.
*Show on dashboard:*
- Pending queue: 1,000 UPIs + 1 SWIFT.
- Agent's priority scores: UPI line items at ~$0.0005, the single SWIFT at ~$643.
- Decision distribution: SWIFT approved instantly; ~100 UPIs approved by tier; ~900 UPIs sheddedwith QoS=BEST_EFFORT 429.
- Nostro liquidity gauge: stays in `MODERATE` band — the agent didn't blow the budget.
- Revenue ticker: **$643.55 captured vs $64.32 random-baseline projection**, displayed live.
*Voiceover, hero line:* "Nobody told the model SWIFT outranks UPI. It learned the math from the universal envelope. Same model. Different rails. Right answer under scarcity."

### **7:00–8:00 — The platform pitch**
*Slide:* "Add a new rail in one PR. New plugin, new Avro schema. Zero core changes. Same cell, same agent, same audit ledger. We are not a payments product. We are the substrate you build payment, identity, and abuse decisions on."
*Last slide:* QR code to the GitHub + plugin SDK + the Aegis Decision API docs page.

### Stage-survival contingencies
- **Net dies.** The whole demo runs locally — Docker Compose with mock SWIFT/UPI generators. No cloud dependency on stage.
- **Live RL diverges.** Pre-trained Q-tables checked into the demo image. The "training loop" panel shows a recorded run alongside the live one.
- **Judges ask "is this just a heuristic?"** Toggle the dashboard view that shows the value function `V(state)` for the current state and the priority scores for each pending event. Heuristics don't have value functions. RL agents do.

---

## 5. What changes upstream of this addendum

To support what's described here, three artifacts in the prior docs need a small revision when you're back at code:

1. **`docs/UNIVERSAL_PLATFORM_STRATEGY.md` §1.2 (Universal Event Envelope)** — adopt the three-layer structure (Envelope + PaymentMetadata + RiskContext + Payload). The Avro schema in §2.3 above is the canonical representation; the Protobuf sketch in the previous doc was a conceptual placeholder.
2. **`docs/UNIVERSAL_PLATFORM_STRATEGY.md` §2.1 (Telemetry Primitives)** — add `LiquidityClass` and `FXVolatilityIndex` as new primitives. They are SWIFT-specific producers but defined in the universal registry so other domains can reuse them (e.g., `LiquidityClass` is meaningful for any pre-funded inventory: ad budgets, cloud quotas, even API rate-limit pools).
3. **`docs/PRODUCT_STRATEGY.md` §2.2 (Compute Optimizer)** — note that the IP claim language should also cover *Lagrangian-priced admission control under heterogeneous transaction economics* — that's the patentable mechanism this addendum demonstrates.

---

## Sources

- [SWIFT — MT103 message reference](https://www.swift.com/standards/data-standards/mt103-customer-credit-transfer)
- [SWIFT — ISO 20022 pacs.008 (Customer Credit Transfer)](https://www.swift.com/standards/iso-20022)
- [BIS — Cross-border payments improvement programme](https://www.bis.org/cpmi/cross_border.htm)
- [OFAC — Specially Designated Nationals (SDN) List](https://ofac.treasury.gov/specially-designated-nationals-and-blocked-persons-list-sdn-human-readable-lists)
- [Reuters — HSBC $1.9B AML settlement (2012)](https://www.reuters.com/article/idUSBRE8BA05M/)
- [DOJ — BNP Paribas $8.9B sanctions settlement (2014)](https://www.justice.gov/opa/pr/bnp-paribas-sentenced-conspiring-violate-international-emergency-economic-powers-act-and)
- [Confluent — Schema Registry concepts](https://docs.confluent.io/platform/current/schema-registry/fundamentals/index.html)
- [Avro — specification 1.11.x](https://avro.apache.org/docs/1.11.1/specification/)
- [NPCI — UPI ecosystem statistics](https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics)
- [Slack Engineering — Migration to a Cellular Architecture](https://slack.engineering/slacks-migration-to-a-cellular-architecture/)
