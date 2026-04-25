from graders import heuristic_policy, _run_episodes, EasyGrader
from unified_gateway import UnifiedFintechEnv

env = UnifiedFintechEnv()
obs, _ = env.reset(seed=42, options={"task": "easy"})
rewards = []
done = False
while not done and len(rewards) < 100:
    action = heuristic_policy(obs.normalized())
    obs, r, done, info = env.step(action)
    rewards.append(r.value)
    if len(rewards) <= 3:
        bd = info["reward_breakdown"]
        print(f"step={len(rewards)} reward={r.value:.3f} breakdown={bd}")

print(f"Episode mean: {sum(rewards)/len(rewards):.4f}  steps={len(rewards)} done={done}")
print(f"Termination: {info.get('termination_reason')}")
