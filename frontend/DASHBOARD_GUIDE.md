# AEPO Dashboard — Module Guide

Quick reference for using and hacking the RL simulation dashboard.

---

## 1. Starting an Episode

The dashboard is **idle by default** — nothing runs until you explicitly start. Go to the **Control Panel** (bottom section):

1. Pick a difficulty from the dropdown: `Easy → Medium → Hard`
2. Click **Reset** — this calls `POST /reset` and initializes the RL environment
3. The header status pill changes to **RUNNING** and the Episode Summary Bar populates

| Difficulty | Avg Reward Threshold | What changes |
|---|---|---|
| Easy | ≥ 0.75 | Low adversary, stable infra signals |
| Medium | ≥ 0.45 | Occasional spikes, moderate adversary |
| Hard | ≥ 0.30 | Cascades, high adversary escalation |

---

## 2. Episode Summary Bar

The thin bar just below the header. Updates every step.

```
Task: EASY  Episode: 3  Step: 42  Phase: spike  Curriculum: Level 1  Cum. Reward: 33.840  Avg/Step: 0.806
```

### Fields

- **Task** — Current difficulty (EASY / MEDIUM / HARD)
- **Episode** — Total episode count since dashboard loaded
- **Step** — Current step within the running episode
- **Phase** — The env's current simulation phase:
  - `normal` (green) — stable conditions
  - `spike` (orange) — latency spike or entropy surge
  - `cascade` (red) — full adversary escalation
- **Curriculum** — Auto-promoted difficulty level (0 → 1 → 2) as agent performs well
- **Cum. Reward** — Total reward accumulated this episode
- **Avg/Step** — **This is the number judged at episode end** — must be ≥ threshold to PASS

---

## 3. Asymmetric Risk Triad (Top Row Gauges)

Three arc gauges, each covering a different risk dimension. All animate when alerting.

### Fraud Risk Signal

- **Arc** = `risk_score` normalized to 0–100. Turns red when > 80
- **Sub-metric 1** — Raw risk score (0–100)
- **Sub-metric 2** — `adversary_threat_level` (0–10). Rises as the agent performs well — the env "pushes back"

**Alert condition:** Score > 80 OR Adversary > 8

### Infra Health Signal

- **Arc** = `kafka_lag` normalized to 0–10,000 messages
- Pulses red + glows when lag > 4,000 (crash threshold in the sim)
- **Sub-metric 1** — Kafka lag (0–10k)
- **Sub-metric 2** — `system_entropy` (0–100). Entropy > 70 triggers latency spikes in next steps

**Alert condition:** Lag > 4,000 OR DB pool > 85%

### SLA Compliance

- **Arc** = `rolling_p99` (EMA-smoothed latency, 0–5,000 ms). SLA threshold is 800ms
- **Sub-metric 1** — Rolling P99 (what the SLA penalty is based on)
- **Sub-metric 2** — Raw `api_latency` (unsmoothed instantaneous latency)

**Alert condition:** P99 > 800ms (incurs SLA breach penalty)

---

## 4. Observation Space Grid (Middle-Left)

All 10 environment signals in one view with progress bars.

### Signal Reference

| Signal | Range | What it means | Warn | Crit |
|---|---|---|---|---|
| **Payment Channel** | 0–2 | P2P / P2M / AutoPay (categorical input) | — | — |
| **Risk Score [sim]** | 0–100 | Fraud probability simulation signal | 60 | 80 |
| **Adversary Threat [sim]** | 0–10 | Adversary escalation; rises when agent does well | 5 | 8 |
| **System Entropy [sim]** | 0–100 | Chaos index; above 70 causes latency spike | 50 | 70 |
| **Kafka Lag [sim]** | 0–10k | Message backlog simulation; > 4k = crash | 3k | 4k |
| **API Latency [sim]** | 0–5k ms | Downstream latency (unsmoothed) | 500 | 800 |
| **Rolling P99 [ema]** | 0–5k ms | EMA of latency — **what SLA is judged on** | 500 | 800 |
| **DB Pool Util [sim]** | 0–100 % | Connection pool usage; > 80 adds retry penalty | 65 | 80 |
| **Bank Sim Status** | 0–2 | 0=Healthy / 1=Degraded / 2=Unknown (purely simulation) | — | — |
| **Merchant Tier [sim]** | 0–1 | Small (0) or Enterprise (1) — affects `app_priority` bonus | — | — |

### Color Coding

- **Blue bar** = healthy (below warning threshold)
- **Yellow bar** = warning zone (50–75% of max)
- **Red bar + pulsing** = critical zone (above critical threshold)

### Phase Badge

The badge in the top-right corner shows the current sim phase with color:
- Green — `normal`
- Orange — `spike`
- Red — `cascade`

---

## 5. Stepping — Manual vs Auto

### Manual Step

Select all 6 action dimensions in the Control Panel, then click **Send Action**.

| Field | Options | Effect |
|---|---|---|
| **Risk Decision** | 0=Approve / 1=Reject / 2=Challenge | Core reward signal; approval on low-risk = +0.8, rejection on high-risk = +0.8 |
| **Crypto Verify** | 0=FullVerify / 1=SkipVerify | SkipVerify = faster but -0.3 if risk > 50 |
| **Infra Routing** | 0=Normal / 1=Throttle / 2=CircuitBreaker | Throttle/CB reduce `kafka_lag` at cost of throughput penalty |
| **DB Retry Policy** | 0=FailFast / 1=ExpBackoff | Backoff helps when DB pool > 80%, adds -0.10 to reward |
| **Settlement Policy** | 0=StandardSync / 1=DeferredAsync | Async fallback during bank degradation |
| **App Priority** | 0=UPI / 1=Credit / 2=Balanced | Match to merchant tier for +0.02 bonus |

### Auto Run

Fires the default balanced action every 600ms automatically:
```
Approve, FullVerify, Normal, FailFast, StandardSync, Balanced
```

Good for watching the env evolve without manual input. Useful for testing infrastructure changes.

---

## 6. Step Log (Right Sidebar)

Every action taken appears here, **newest first**.

### Entry Layout

**Row 1: Decision + Reward**
- Step number (e.g., `#42`)
- Decision badge (color-coded: green=Approve, red=Reject, yellow=Challenge)
- Verification mode (FullVerify / SkipVerify)
- Reward (green if positive, red if negative)

**Row 2: Infrastructure + Phase + Timestamp**
- Infra routing (Normal / Throttle / CircuitBreaker)
- DB retry policy (FailFast / ExpBackoff)
- Phase badge (only shown if phase ≠ normal)
- Wall-clock timestamp

**Expanded View (Click to expand)**
- Full reward breakdown showing each component:
  - `fraud_reward` — bonus for correct risk decision
  - `sla_penalty` — -1.0 if P99 > 800ms
  - `adversary_penalty` — rises each step
  - `entropy_spike_penalty` — triggered during entropy > 70
  - Other component scores

---

## 7. Reward Chart + Infra Trend

### Reward Chart (Top right)

Two lines tracking reward trajectory:

- **Solid blue line** = per-step reward (can be noisy)
- **Dashed cyan line** = cumulative reward (smooth upward trend = learning)
- **Zero reference line** = horizontal dashed line at y=0 (helps spot losses)

**How to read it:**
- Steep upward cyan = agent learning well
- Flat or downward cyan = agent losing ground
- Blue spikes = good individual decisions
- Blue dips = penalties triggered (SLA breach, bad risk decision, entropy spike)

### Infra Trend (Bottom right)

Area chart showing infrastructure signals over time:

- **Orange area** = rolling P99 latency (0–5000ms)
- **Red area** = kafka lag / 100 (scales to fit alongside P99)

**How to read it:**
- Watch for sudden spikes in both — they correspond to `spike` or `cascade` phases
- Sustained red = the agent is not controlling lag effectively

---

## 8. Q-Table Heatmap (Middle)

A 6×6 grid visualization of **state × action Q-values** in the trained RL model.

### Structure

**Rows (States):**
```
LowRisk·LowLag      (risk_score < 40, kafka_lag < 2000)
LowRisk·HiLag       (risk_score < 40, kafka_lag > 2000)
MidRisk·LowLag      (40 ≤ risk_score < 70, kafka_lag < 2000)
MidRisk·HiLag       (40 ≤ risk_score < 70, kafka_lag > 2000)
HiRisk·LowLag       (risk_score ≥ 70, kafka_lag < 2000)
HiRisk·HiLag        (risk_score ≥ 70, kafka_lag > 2000)
```

**Columns (Actions):**
```
Approve·FullVerify
Approve·SkipVerify
Reject·FullVerify
Reject·SkipVerify
Challenge·FullVerify
Challenge·SkipVerify
```

### Color Scale

- **Dark blue** = low Q-value (avoid this state-action pair)
- **Green** = medium Q-value
- **Red** = high Q-value (agent prefers this pair)

### Current Data

Uses **deterministic dummy values** so the structure is always visible. To wire real Q-values from a trained agent:

```tsx
// In page.tsx, pass qValues prop:
<QTableHeatmap qValues={yourTrainedQTable} />
```

---

## 9. Episode Done Overlay

Pops automatically when the env returns `done: true` (usually after 100 steps).

### What It Shows

- **PASS / FAIL badge** — color-coded against the per-task threshold
- **Stats grid:**
  - Task (EASY / MEDIUM / HARD)
  - Steps taken (usually 100)
  - Total reward (sum of all steps)
  - Avg reward/step (total ÷ steps) — **this is the judged metric**
  - Final phase (normal / spike / cascade)
  - Curriculum level reached

- **Threshold bar** — visual comparison showing how far above/below you landed

### Buttons

- **Restart Same Task** — immediately starts a new episode with the same difficulty
- **Dismiss** — closes the overlay and returns to idle state

### Pass Criteria

| Task | Threshold | Must have |
|---|---|---|
| Easy | ≥ 0.75 | Avg reward/step ≥ 0.75 |
| Medium | ≥ 0.45 | Avg reward/step ≥ 0.45 |
| Hard | ≥ 0.30 | Avg reward/step ≥ 0.30 |

---

## 10. Causal Notifications (Toast Stack)

Red/yellow toasts pop in the bottom-right when **causal transitions** fire.

### Examples

```
⚠️ Kafka Lag > 3000 — Latency Compounding        [yellow, warn]
🔴 Kafka Lag > 4000 — Sim Crash Threshold        [red, critical]
⚠️ Risk Score > 80 — Fraud Signal High           [yellow, warn]
🔴 P99 > 800ms — SLA Penalty Active              [red, critical]
⚠️ Adversary Threat > 7 — Escalation Phase       [yellow, warn]
⚠️ System Entropy > 70 — Latency Spike Imminent  [yellow, warn]
⚠️ DB Pool > 80% — Retry Overhead Sim Active     [yellow, warn]
```

Auto-dismiss after 5 seconds. Click the `×` to dismiss immediately.

---

## Frontend Quick Hacks

### Add a New Metric to the Triad

Edit `src/components/metrics/RiskTriad.tsx`:

```tsx
<GaugeCard
  title="Your Metric"
  icon={<Icon className="w-3.5 h-3.5" />}
  value={normalizedValue}  // must be 0–1
  displayValue={displayStr}
  subtitle="unit"
  color={colorHex}
  alerting={shouldPulse}
>
  {/* sub-metrics */}
</GaugeCard>
```

### Change Reward Color

Edit `src/lib/utils.ts`:

```tsx
export function rewardColor(r: number): string {
  if (r > 0.5) return "text-green-400";
  if (r > 0) return "text-emerald-400";
  // ... etc
}
```

### Disable Auto-Dismiss Toasts

In `src/hooks/useAEPO.ts`, remove or increase the timeout:

```tsx
// Was: setTimeout(() => dismissNotification(n.id), 5000);
// Change to: Infinity (never auto-dismiss)
setTimeout(() => dismissNotification(n.id), Infinity);
```

### Show/Hide the Q-Table

In `src/app/page.tsx`, comment out:

```tsx
{/* Row 3: Q-Table Heatmap */}
{/* <QTableHeatmap /> */}
```

### Customize Episode Summary Bar

Edit `src/app/page.tsx`, the `<EpisodeSummaryBar>` section — add/remove fields as needed.

### Change the Poll Interval

In `src/hooks/useAEPO.ts`:

```tsx
const POLL_INTERVAL = 500;  // milliseconds — change to 250 for faster updates
```

### Wire Real Q-Values

In `src/app/page.tsx`:

```tsx
<QTableHeatmap qValues={yourTrainedQTable} />
```

Where `yourTrainedQTable` is a `number[][]` loaded from your trained agent.

---

## File Structure

```
frontend/src/
├── app/
│   ├── page.tsx                    # Main dashboard (layout + state wiring)
│   ├── layout.tsx                  # Root layout
│   └── globals.css                 # Tailwind + custom styles
├── components/
│   ├── metrics/
│   │   ├── RiskTriad.tsx           # Top 3 gauges
│   │   ├── GaugeCard.tsx           # Reusable gauge component
│   │   ├── ObservationGrid.tsx     # 10-signal grid
│   │   ├── RewardChart.tsx         # Recharts line chart
│   │   ├── InfraChart.tsx          # Area chart (P99 + lag trend)
│   │   └── QTableHeatmap.tsx       # Heatmap grid
│   ├── controls/
│   │   └── ControlPanel.tsx        # Reset, auto-run, action form
│   ├── feed/
│   │   └── LiveActionFeed.tsx      # Step log (right sidebar)
│   └── ui/
│       ├── ToastNotification.tsx   # Causal alerts
│       └── EpisodeDoneOverlay.tsx  # Pass/fail modal
├── hooks/
│   └── useAEPO.ts                  # Main data hook (polling, step, reset)
└── lib/
    ├── types.ts                    # TypeScript interfaces
    ├── api.ts                       # HTTP client
    └── utils.ts                     # Color maps, formatters
```

---

## Quick Reference — Common Tasks

| Task | File | Line # |
|---|---|---|
| Change SLA threshold from 800ms to X | `src/components/metrics/RiskTriad.tsx` | line 22 |
| Add a new causal notification | `src/hooks/useAEPO.ts` | line 18–90 |
| Change poll interval (500ms) | `src/hooks/useAEPO.ts` | line 14 |
| Customize pass/fail thresholds | `src/components/ui/EpisodeDoneOverlay.tsx` | line 13–15 |
| Change default auto-run action | `src/hooks/useAEPO.ts` | line 178–186 |
| Hide a metric from the triad | `src/app/page.tsx` | line 98–102 |
| Change toast color scheme | `src/components/ui/ToastNotification.tsx` | line 19–29 |

---

## Testing Checklist

- [ ] Backend running: `curl http://localhost:7860/`
- [ ] Frontend running: `npm run dev` → http://localhost:3000
- [ ] Reset works (click Reset button in Control Panel)
- [ ] Episode Summary Bar populates
- [ ] Risk Triad gauges animate
- [ ] Observation Grid bars move
- [ ] Reward Chart shows line
- [ ] Step Log appends new entries
- [ ] Toasts pop on causal transitions
- [ ] Auto Run toggles and fires steps
- [ ] Episode Done overlay pops at done=true
- [ ] Type errors: `npx tsc --noEmit` (should be empty)

---

## Notes

- **All signals are simulation signals** — labeled `[sim]` to clarify this is a pure RL training environment
- **Thresholds are baked into the env**, not the dashboard — if you want to change penalty behavior, edit `unified_gateway.py`, not the UI
- **Q-Table is dummy data by default** — train via `python train.py` then wire the `.pkl` snapshot
- **Curriculum level auto-increments** — the env promotes difficulty when the agent's recent mean reward exceeds a threshold
- **Phase transitions** are computed by the env and returned in the `info` dict on each step

