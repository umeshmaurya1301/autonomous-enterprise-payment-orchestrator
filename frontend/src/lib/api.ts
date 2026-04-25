import type { AEPOAction, StepResult, StateResult, ResetResult, TaskDifficulty } from "./types";

const BASE = "/api";

export async function fetchState(): Promise<StateResult> {
  const res = await fetch(`${BASE}/state`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /state → ${res.status}`);
  return res.json();
}

export async function postReset(task: TaskDifficulty = "easy"): Promise<ResetResult> {
  const res = await fetch(`${BASE}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  });
  if (!res.ok) throw new Error(`POST /reset → ${res.status}`);
  return res.json();
}

export async function postStep(action: AEPOAction): Promise<StepResult> {
  const res = await fetch(`${BASE}/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) throw new Error(`POST /step → ${res.status}`);
  return res.json();
}
