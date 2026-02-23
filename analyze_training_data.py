"""
Analyze training data action distribution
"""
import pickle
import numpy as np

# Load training data
with open('expert_demos_vehicle_v2/vehicle_expert_demos.pkl', 'rb') as f:
    episodes = pickle.load(f)

print(f"Loaded {len(episodes)} episodes")

# Analyze actions by vehicle type
main_actions = []
ramp_actions = []
diverge_actions = []

for ep in episodes:
    for trans in ep.get('transitions', []):
        vehicle_type = trans.get('vehicle_type', '')
        action_main = trans.get('action_main', None)
        action_ramp = trans.get('action_ramp', None)

        if action_main is not None:
            if vehicle_type == 'main':
                main_actions.append(action_main[0])
            elif vehicle_type == 'ramp':
                ramp_actions.append(action_main[0])
            elif vehicle_type == 'diverge':
                diverge_actions.append(action_main[0])

print(f"\nAction distributions:")
print(f"  Main vehicles: {len(main_actions)} samples")
print(f"    mean: {np.mean(main_actions):.4f}")
print(f"    std: {np.std(main_actions):.4f}")
print(f"    min: {np.min(main_actions):.4f}")
print(f"    max: {np.max(main_actions):.4f}")
print(f"    median: {np.median(main_actions):.4f}")

print(f"\n  Ramp vehicles: {len(ramp_actions)} samples")
print(f"    mean: {np.mean(ramp_actions):.4f}")
print(f"    std: {np.std(ramp_actions):.4f}")
print(f"    min: {np.min(ramp_actions):.4f}")
print(f"    max: {np.max(ramp_actions):.4f}")
print(f"    median: {np.median(ramp_actions):.4f}")

print(f"\n  Diverge vehicles: {len(diverge_actions)} samples")
print(f"    mean: {np.mean(diverge_actions):.4f}")
print(f"    std: {np.std(diverge_actions):.4f}")
print(f"    min: {np.min(diverge_actions):.4f}")
print(f"    max: {np.max(diverge_actions):.4f}")
print(f"    median: {np.median(diverge_actions):.4f}")

# Check histogram bins
print(f"\nMain action histogram:")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(main_actions, bins=bins)
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    print(f"  [{low:.1f}, {high:.1f}): {hist[i]}")
