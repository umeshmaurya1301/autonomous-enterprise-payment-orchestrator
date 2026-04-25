"""
train_grpo_hf.py — GRPO Training for AEPO on HF Space A10G
===========================================================
Production Python script equivalent of AEPO_Unsloth_GRPO.ipynb.
Run this file directly on the HF Space machine (or any CUDA machine).

Hardware targets
----------------
A10G · 24 GB VRAM → Qwen2.5-7B-Instruct, LoRA rank 32, 8 gens, 3 epochs
T4   · 16 GB VRAM → Qwen2.5-3B-Instruct, LoRA rank 16, 4 gens, 1 epoch

Usage
-----
On the HF Space machine (via SSH or Docker startup command):
    python train_grpo_hf.py

Override model / Hub target:
    HF_REPO=your-org/my-model HF_TOKEN=hf_xxx python train_grpo_hf.py

Push Hub + resume checkpoint:
    RESUME_FROM_CHECKPOINT=outputs/checkpoint-50 python train_grpo_hf.py

Environment variables
---------------------
HF_TOKEN              HF Hub token for pushing the trained adapter
HF_REPO               Target repo (default: umeshmaurya1301/aepo-qwen2.5-Xb-grpo)
RESUME_FROM_CHECKPOINT Path to a prior checkpoint directory (optional)
N_SAMPLES             Override dataset size (default: 2000 on A10G, 500 on T4)
N_EPOCHS              Override epoch count   (default: 3 on A10G, 1 on T4)

Outputs
-------
results/grpo_reward_curve.png     reward improvement chart
aepo-qwen2.5-Xb-grpo/            saved LoRA adapter (local)
HF Hub: HF_REPO                   pushed adapter (if HF_TOKEN is set)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Verify CUDA before importing heavy deps ───────────────────────────────────
if not torch.cuda.is_available():
    logger.error("No CUDA device found. train_grpo_hf.py requires a GPU.")
    sys.exit(1)

_VRAM_GB: float = torch.cuda.get_device_properties(0).total_memory / 1e9
_GPU_NAME: str  = torch.cuda.get_device_properties(0).name
logger.info("GPU: %s  |  VRAM: %.1f GB", _GPU_NAME, _VRAM_GB)

# ── Hardware-aware defaults ───────────────────────────────────────────────────
_IS_A10G: bool = _VRAM_GB >= 22

MODEL_NAME: str = "unsloth/Qwen2.5-7B-Instruct"  if _IS_A10G else "unsloth/Qwen2.5-3B-Instruct"
LORA_RANK:  int = 32                               if _IS_A10G else 16
GPU_MEM:  float = 0.85                             if _IS_A10G else 0.60
N_SAMPLES:  int = int(os.environ.get("N_SAMPLES",  2000 if _IS_A10G else 500))
N_EPOCHS:   int = int(os.environ.get("N_EPOCHS",   3    if _IS_A10G else 1))
BATCH:      int = 4                                if _IS_A10G else 2
GRAD_ACCUM: int = 4                                if _IS_A10G else 8
NUM_GENS:   int = 8                                if _IS_A10G else 4

_MODEL_SIZE_TAG: str  = "7b" if _IS_A10G else "3b"
LOCAL_SAVE_DIR:  str  = f"aepo-qwen2.5-{_MODEL_SIZE_TAG}-grpo"
HF_TOKEN:        str  = os.environ.get("HF_TOKEN", "")
HF_REPO:         str  = os.environ.get("HF_REPO",  f"umeshmaurya1301/aepo-qwen2.5-{_MODEL_SIZE_TAG}-grpo")
RESUME_FROM:     str  = os.environ.get("RESUME_FROM_CHECKPOINT", "")

logger.info(
    "Config: model=%s  lora_rank=%d  samples=%d  epochs=%d  "
    "batch=%d  grad_accum=%d  gens=%d",
    MODEL_NAME, LORA_RANK, N_SAMPLES, N_EPOCHS, BATCH, GRAD_ACCUM, NUM_GENS,
)

# ── Imports ───────────────────────────────────────────────────────────────────
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from unified_gateway import AEPOAction, UnifiedFintechEnv
from graders import EasyGrader, MediumGrader, HardGrader, heuristic_policy

# ── System prompt (identical to inference.py to ensure weight transfer) ───────
SYSTEM_PROMPT = """\
You are the autonomous control agent for the Autonomous Enterprise Payment Orchestrator (AEPO).

Every turn you receive ten real-time signals (all normalized to [0.0, 1.0]):
  transaction_type, risk_score, adversary_threat_level, system_entropy,
  kafka_lag, api_latency, rolling_p99, db_connection_pool,
  bank_api_status, merchant_tier

You must output EXACTLY six integers separated by spaces on a single line:
  risk_decision crypto_verify infra_routing db_retry_policy settlement_policy app_priority

Allowed values:
  risk_decision  : 0=Approve   1=Reject       2=Challenge
  crypto_verify  : 0=FullVerify  1=SkipVerify
  infra_routing  : 0=Normal    1=Throttle     2=CircuitBreaker
  db_retry_policy: 0=FailFast  1=ExponentialBackoff
  settlement_policy: 0=StandardSync  1=DeferredAsyncFallback
  app_priority   : 0=UPI       1=Credit       2=Balanced

Key rules:
  - risk_score > 0.8 → NEVER Approve+SkipVerify (instant fraud catastrophe, reward=0)
  - kafka_lag > 0.3 → use Throttle to prevent crash penalty
  - Reject + SkipVerify on high risk is optimal: saves lag AND is safe
  - Match app_priority to merchant_tier for +0.02 bonus

Output ONLY the six integers. No explanation. Example: 1 1 0 0 0 2\
"""

_ACTION_REGEX = re.compile(r'\b([012])\s+([01])\s+([012])\s+([01])\s+([01])\s+([012])\b')


def _parse_action(text: str) -> AEPOAction | None:
    """Parse 6 space-separated integers from LLM text. Returns None on parse failure."""
    match = _ACTION_REGEX.search(text)
    if not match:
        return None
    try:
        return AEPOAction(
            risk_decision    = int(match.group(1)),
            crypto_verify    = int(match.group(2)),
            infra_routing    = int(match.group(3)),
            db_retry_policy  = int(match.group(4)),
            settlement_policy= int(match.group(5)),
            app_priority     = int(match.group(6)),
        )
    except Exception:
        return None


def _format_obs_prompt(norm: dict) -> str:
    """Format a normalized observation dict as the user turn of the conversation."""
    lines = "\n".join(f"  {k}: {v:.3f}" for k, v in norm.items())
    return f"Current system state:\n{lines}\n\nOutput your action:"


# ── Reward function ───────────────────────────────────────────────────────────

_blind_spot_count: int = 0  # module-level counter for summary log


def env_reward_func(
    completions: list,
    prompts: list,
    seed_val: list[int],
    task_name: list[str],
    **kwargs,
) -> list[float]:
    """
    GRPO reward function: runs the AEPO environment and returns the step reward.

    GRPOTrainer calls this once per batch with all (completion, seed, task)
    tuples. We reconstruct the exact environment state (same seed + task) so
    the reward surface is consistent across GRPO iterations.

    Returns 0.0 for malformed completions (strong gradient pressure toward
    format compliance vs the 0.8 baseline any valid action earns).
    """
    global _blind_spot_count
    rewards: list[float] = []

    for completion, ep_seed, ep_task in zip(completions, seed_val, task_name):
        content = completion[0]["content"] if isinstance(completion, list) else completion

        action = _parse_action(content)
        if action is None:
            rewards.append(0.0)
            continue

        try:
            env = UnifiedFintechEnv()
            env.reset(seed=ep_seed, options={"task": ep_task})
            _, typed_reward, _, info = env.step(action)
            reward_value = float(typed_reward.value)

            if info.get("blind_spot_triggered", False):
                _blind_spot_count += 1
                if _blind_spot_count == 1:
                    logger.info(
                        "[BLIND SPOT #1 DISCOVERED by GRPO] Reject+SkipVerify+HighRisk "
                        "seed=%d  task=%s  reward=%.3f",
                        ep_seed, ep_task, reward_value,
                    )

            rewards.append(reward_value)
        except Exception:
            rewards.append(0.0)

    return rewards


# ── Dataset generation ────────────────────────────────────────────────────────

def build_dataset() -> Dataset:
    """
    Generate N_SAMPLES environment observations from all three tasks.

    Task weight: 50% hard / 33% medium / 17% easy — mirrors train.py's
    EPISODES_PER_LEVEL ratio (hard dominates because that's the submission task).
    Seeds start at 3000 to avoid collision with grader seeds (42/43/44) and
    training seeds (44..44+N_EPISODES in train.py).
    """
    logger.info("Building dataset: %d samples ...", N_SAMPLES)
    env_gen = UnifiedFintechEnv()
    task_weights = ["easy"] * 1 + ["medium"] * 2 + ["hard"] * 3
    train_data: list[dict] = []

    for i in range(N_SAMPLES):
        ep_seed = 3000 + i
        ep_task = task_weights[i % len(task_weights)]
        obs, _  = env_gen.reset(seed=ep_seed, options={"task": ep_task})
        norm    = obs.normalized()

        train_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _format_obs_prompt(norm)},
            ],
            "seed_val":  ep_seed,
            "task_name": ep_task,
        })

    dataset = Dataset.from_list(train_data)
    task_counts = {t: sum(1 for d in train_data if d["task_name"] == t) for t in ["easy", "medium", "hard"]}
    logger.info("Dataset ready: %d prompts  task_counts=%s", len(dataset), task_counts)
    return dataset


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load the base model with Unsloth LoRA and return (model, tokenizer)."""
    logger.info("Loading %s (LoRA rank=%d, gpu_mem=%.0f%%) ...", MODEL_NAME, LORA_RANK, GPU_MEM * 100)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEM,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
    )
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: total=%.0fM  trainable(LoRA)=%.1fM (%.2f%%)",
                total / 1e6, trainable / 1e6, 100 * trainable / total)
    return model, tokenizer


# ── GRPO training ─────────────────────────────────────────────────────────────

def run_training(model, tokenizer, dataset: Dataset):
    """Run GRPO training and return the trainer for post-training evaluation."""
    _eff_batch = BATCH * GRAD_ACCUM * NUM_GENS
    logger.info(
        "GRPO config: epochs=%d  batch=%d  grad_accum=%d  gens=%d  eff_batch=%d",
        N_EPOCHS, BATCH, GRAD_ACCUM, NUM_GENS, _eff_batch,
    )

    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        max_prompt_length=512,
        max_completion_length=32,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENS,
        report_to="none",
        save_steps=50,
        save_total_limit=2,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    resume = RESUME_FROM if RESUME_FROM and Path(RESUME_FROM).is_dir() else None
    if resume:
        logger.info("Resuming from checkpoint: %s", resume)

    trainer.train(resume_from_checkpoint=resume)
    return trainer


# ── Reward curve ──────────────────────────────────────────────────────────────

def plot_reward_curve(trainer) -> None:
    """Extract trainer log history and save the GRPO reward curve to results/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping reward curve plot")
        return

    log_history  = trainer.state.log_history
    reward_logs  = [x for x in log_history if "reward" in x]
    if not reward_logs:
        logger.warning("No reward logs in trainer.state — check logging_steps=1")
        return

    steps   = [x["step"]   for x in reward_logs]
    rewards = [x["reward"] for x in reward_logs]

    window  = 5
    rolling = [float(np.mean(rewards[max(0, i - window + 1): i + 1])) for i in range(len(rewards))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, rewards, alpha=0.25, color="steelblue", linewidth=0.8, label="Raw step reward")
    ax.plot(steps, rolling, color="steelblue", linewidth=2.2, label=f"{window}-step rolling mean")
    ax.axhline(y=0.75, color="green",  linestyle="--", linewidth=1.0, label="Easy threshold (0.75)")
    ax.axhline(y=0.45, color="orange", linestyle="--", linewidth=1.0, label="Medium threshold (0.45)")
    ax.axhline(y=0.30, color="red",    linestyle="--", linewidth=1.2, label="Hard threshold (0.30)")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("AEPO Environment Reward [0, 1]", fontsize=12)
    ax.set_title(
        f"AEPO: {MODEL_NAME.split('/')[-1]} GRPO Training — Reward Improvement Curve\n"
        f"LLM learns to navigate the AEPO environment via RL from environment feedback",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "grpo_reward_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Reward curve saved to %s", out_path)
    if rewards:
        logger.info(
            "Training summary: initial=%.3f  final=%.3f  improvement=%+.3f  "
            "blind_spot_triggers=%d",
            rewards[0], rewards[-1], rewards[-1] - rewards[0], _blind_spot_count,
        )


# ── Before / after evaluation ─────────────────────────────────────────────────

def evaluate(model, tokenizer) -> None:
    """Compare heuristic baseline vs GRPO-trained policy on all three tasks."""
    FastLanguageModel.for_inference(model)

    def grpo_policy(obs_normalized: dict) -> AEPOAction:
        state_str = "\n".join(f"  {k}: {v:.3f}" for k, v in obs_normalized.items())
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Current system state:\n{state_str}\n\nOutput your action:"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=16, temperature=0.0, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        action = _parse_action(response)
        return action if action is not None else AEPOAction(
            risk_decision=1, crypto_verify=1, infra_routing=1,
            db_retry_policy=0, settlement_policy=0, app_priority=2,
        )

    graders = [
        ("easy",   EasyGrader(),   0.75),
        ("medium", MediumGrader(), 0.45),
        ("hard",   HardGrader(),   0.30),
    ]
    N_EVAL = 5
    logger.info("Evaluating policies (%d episodes per task) ...", N_EVAL)
    print(f"\n{'Task':<8} {'Heuristic':>12} {'GRPO-Trained':>14} {'Threshold':>11} {'Delta':>8} {'Pass?':>7}")
    print("-" * 66)

    for task, grader, threshold in graders:
        h_score = grader.grade_agent(heuristic_policy, n_episodes=N_EVAL)
        g_score = grader.grade_agent(grpo_policy,      n_episodes=N_EVAL)
        delta   = g_score - h_score
        status  = "PASS" if g_score >= threshold else "FAIL"
        print(f"{task:<8} {h_score:>12.4f} {g_score:>14.4f} {threshold:>11.2f} {delta:>+8.4f}  {status}")


# ── Save + push to Hub ────────────────────────────────────────────────────────

def save_and_push(model, tokenizer, dataset: Dataset) -> None:
    """Save LoRA adapter locally and push to HF Hub if HF_TOKEN is set."""
    model.save_pretrained(LOCAL_SAVE_DIR)
    tokenizer.save_pretrained(LOCAL_SAVE_DIR)
    logger.info("LoRA adapter saved to %s/", LOCAL_SAVE_DIR)

    metadata = {
        "base_model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "training_samples": len(dataset),
        "epochs": N_EPOCHS,
        "num_generations": NUM_GENS,
        "effective_batch": BATCH * GRAD_ACCUM * NUM_GENS,
        "hardware": f"{_GPU_NAME} {_VRAM_GB:.0f}GB",
        "blind_spot_triggers_during_training": _blind_spot_count,
    }
    Path(f"{LOCAL_SAVE_DIR}/aepo_training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        logger.info("Pushing to Hub: %s ...", HF_REPO)
        model.push_to_hub(HF_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
        logger.info("Model live at: https://huggingface.co/%s", HF_REPO)
    else:
        logger.warning("HF_TOKEN not set — skipping Hub push. Set HF_TOKEN to enable.")
        logger.info("Manually push: huggingface-cli upload %s %s", LOCAL_SAVE_DIR, HF_REPO)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end: load → dataset → train → evaluate → save → push."""
    logger.info("=== AEPO GRPO Training — %s ===", _GPU_NAME)

    dataset          = build_dataset()
    model, tokenizer = load_model()
    trainer          = run_training(model, tokenizer, dataset)
    plot_reward_curve(trainer)
    evaluate(model, tokenizer)
    save_and_push(model, tokenizer, dataset)

    logger.info("=== Training complete. Outputs: %s/ | results/grpo_reward_curve.png ===", LOCAL_SAVE_DIR)


if __name__ == "__main__":
    main()
