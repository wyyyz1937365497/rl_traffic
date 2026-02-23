"""
分析v3训练数据的action分布
"""
import pickle
import numpy as np

with open('expert_demos_vehicle_v3/vehicle_expert_demos.pkl', 'rb') as f:
    episodes = pickle.load(f)

print(f"Episodes: {len(episodes)}")

# 统计所有action
all_actions = []
main_actions = []
ramp_actions = []
control_actions = []  # action < 0.95（有控制）

for ep in episodes:
    for trans in ep.get('transitions', []):
        vtype = trans.get('vehicle_type', '')
        action = trans.get('action_main', [0])[0]
        all_actions.append(action)

        if vtype == 'main':
            main_actions.append(action)
        elif vtype == 'ramp':
            ramp_actions.append(action)

        if action < 0.95:
            control_actions.append(action)

all_actions = np.array(all_actions)
main_actions = np.array(main_actions)
ramp_actions = np.array(ramp_actions)
control_actions = np.array(control_actions)

print(f"\n=== 总体统计 ===")
print(f"总样本数: {len(all_actions)}")
print(f"Action = 1.0: {np.sum(all_actions >= 0.999)} ({np.sum(all_actions >= 0.999)/len(all_actions)*100:.2f}%)")
print(f"Action < 0.95: {len(control_actions)} ({len(control_actions)/len(all_actions)*100:.2f}%)")

print(f"\n=== Main车辆 ===")
print(f"样本数: {len(main_actions)}")
print(f"均值: {np.mean(main_actions):.4f}")
print(f"中位数: {np.median(main_actions):.4f}")
print(f"最小值: {np.min(main_actions):.4f}")
print(f"最大值: {np.max(main_actions):.4f}")
print(f"Action = 1.0比例: {np.sum(main_actions >= 0.999)/len(main_actions)*100:.2f}%")

print(f"\n=== Ramp车辆 ===")
print(f"样本数: {len(ramp_actions)}")
print(f"均值: {np.mean(ramp_actions):.4f}")
print(f"中位数: {np.median(ramp_actions):.4f}")
print(f"最小值: {np.min(ramp_actions):.4f}")
print(f"最大值: {np.max(ramp_actions):.4f}")
print(f"Action = 1.0比例: {np.sum(ramp_actions >= 0.999)/len(ramp_actions)*100:.2f}%")

print(f"\n=== 有控制的样本 (action < 0.95) ===")
print(f"样本数: {len(control_actions)}")
print(f"均值: {np.mean(control_actions):.4f}")
print(f"中位数: {np.median(control_actions):.4f}")
print(f"最小值: {np.min(control_actions):.4f}")
print(f"最大值: {np.max(control_actions):.4f}")

# 直方图
print(f"\n=== Action直方图 ===")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
hist, _ = np.histogram(all_actions, bins=bins)
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    bar = '█' * int(hist[i] / len(all_actions) * 100)
    print(f"[{low:.2f}, {high:.2f}): {hist[i]:6d} {bar}")
