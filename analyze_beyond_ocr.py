"""
分析当OCR接近95%时，什么对得分影响更大
基于评分公式的敏感性分析
"""

import numpy as np

print("=" * 80)
print("评分公式敏感性分析 - 超越OCR优化")
print("=" * 80)

# 评分公式参数
W_EFFICIENCY = 0.5
W_STABILITY = 0.5

# 当前状态（高分提交）
current_ocr = 0.9545
baseline_ocr = 0.94  # 假设baseline

# 1. 效率得分敏感性分析
print("\n1. 效率得分 (S_efficiency) = 100 × max(0, (OCR - OCR_base) / OCR_base)")
print("-" * 80)

current_efficiency_score = 100 * (current_ocr - baseline_ocr) / baseline_ocr
print(f"当前OCR: {current_ocr:.4f}")
print(f"Baseline OCR: {baseline_ocr:.4f}")
print(f"当前效率得分: {current_efficiency_score:.2f}")

# OCR提升的边际收益
ocr_increases = [0.001, 0.005, 0.01, 0.02]  # 0.1%, 0.5%, 1%, 2%
print(f"\nOCR提升的边际收益分析:")
print(f"{'OCR提升':<15} {'新OCR':<15} {'效率得分变化':<20} {'提升率':<15}")
print("-" * 80)

for inc in ocr_increases:
    new_ocr = current_ocr + inc
    new_score = 100 * (new_ocr - baseline_ocr) / baseline_ocr
    delta = new_score - current_efficiency_score
    improvement_rate = delta / current_efficiency_score * 100
    print(f"{inc*100:.2f}%{'':<10} {new_ocr:.4f}{'':<10} {delta:+.2f}{'':<15} {improvement_rate:+.2f}%")

# 2. 稳定性得分敏感性分析
print("\n\n2. 稳定性得分 (S_stability) = 100 × [0.4 × I_σv + 0.6 × I_|a|]")
print("-" * 80)

print("稳定性得分取决于与baseline的相对改善:")
print("  I_σv = -(σv_AI - σv_base) / σv_base")
print("  I_|a| = -(|a|_AI - |a|_base) / |a|_base")

# 假设baseline值
baseline_speed_std = 8.0  # m/s
baseline_mean_accel = 1.2  # m/s²

# 当前AI值（从pkl分析中估计）
current_speed_std = 6.5  # 假设改善20%
current_mean_accel = 0.8  # 假设改善33%

I_speed_std = -(current_speed_std - baseline_speed_std) / baseline_speed_std
I_mean_accel = -(current_mean_accel - baseline_mean_accel) / baseline_mean_accel

current_stability_score = 100 * (0.4 * max(0, I_speed_std) + 0.6 * max(0, I_mean_accel))

print(f"\n当前稳定性指标:")
print(f"  速度标准差: {current_speed_std:.2f} m/s (baseline: {baseline_speed_std:.2f} m/s)")
print(f"  平均绝对加速度: {current_mean_accel:.2f} m/s² (baseline: {baseline_mean_accel:.2f} m/s²)")
print(f"  I_σv: {I_speed_std:.4f}")
print(f"  I_|a|: {I_mean_accel:.4f}")
print(f"  当前稳定性得分: {current_stability_score:.2f}")

# 稳定性改善的边际收益
speed_std_improvements = [0.5, 1.0, 1.5, 2.0]  # m/s
mean_accel_improvements = [0.1, 0.2, 0.3, 0.4]  # m/s²

print(f"\n稳定性改善的边际收益分析:")
print(f"{'速度标准差降低':<20} {'稳定性得分变化':<20} {'加速度降低':<20} {'稳定性得分变化':<20}")
print("-" * 80)

for i in range(4):
    # 速度标准差改善
    new_speed_std = current_speed_std - speed_std_improvements[i]
    new_I_speed_std = -(new_speed_std - baseline_speed_std) / baseline_speed_std
    new_score_speed = 100 * (0.4 * max(0, new_I_speed_std) + 0.6 * max(0, I_mean_accel))
    delta_speed = new_score_speed - current_stability_score

    # 加速度改善
    new_mean_accel = current_mean_accel - mean_accel_improvements[i]
    new_I_mean_accel = -(new_mean_accel - baseline_mean_accel) / baseline_mean_accel
    new_score_accel = 100 * (0.4 * max(0, I_speed_std) + 0.6 * max(0, new_I_mean_accel))
    delta_accel = new_score_accel - current_stability_score

    print(f"{speed_std_improvements[i]:.1f} m/s{'':<13} {delta_speed:+.2f}{'':<17} {mean_accel_improvements[i]:.1f} m/s²{'':<13} {delta_accel:+.2f}")

# 3. 干预成本惩罚分析
print("\n\n3. 干预成本惩罚 (P_int) = e^(-k × C_int)")
print("-" * 80)

k = 10.0  # 惩罚系数（假设）

# 干预成本
C_int_values = [0.01, 0.02, 0.05, 0.1, 0.2]

print(f"干预成本对惩罚因子的影响 (k={k}):")
print(f"{'干预成本 C_int':<20} {'惩罚因子 P_int':<20} {'得分保留率':<15}")
print("-" * 80)

for C_int in C_int_values:
    P_int = np.exp(-k * C_int)
    retention_rate = P_int * 100
    print(f"{C_int:<20.3f} {P_int:<20.4f} {retention_rate:<15.2f}%")

# 4. 综合分析
print("\n\n4. 综合得分敏感性分析")
print("=" * 80)

print("\n假设当前状态:")
print(f"  效率得分: {current_efficiency_score:.2f}")
print(f"  稳定性得分: {current_stability_score:.2f}")
print(f"  干预惩罚: 0.95 (假设C_int=0.005)")

current_total = (W_EFFICIENCY * current_efficiency_score +
                 W_STABILITY * current_stability_score) * 0.95

print(f"  当前总分: {current_total:.2f}")

# 不同优化策略的潜在收益
print("\n潜在优化策略分析:")
print("-" * 80)

strategies = [
    ("OCR提升1%", {
        'ocr_delta': 0.01,
        'speed_std_delta': 0,
        'accel_delta': 0
    }),
    ("速度标准差降低1 m/s", {
        'ocr_delta': 0,
        'speed_std_delta': 1.0,
        'accel_delta': 0
    }),
    ("平均加速度降低0.2 m/s²", {
        'ocr_delta': 0,
        'speed_std_delta': 0,
        'accel_delta': 0.2
    }),
    ("组合优化", {
        'ocr_delta': 0.005,
        'speed_std_delta': 0.5,
        'accel_delta': 0.1
    }),
]

print(f"{'策略':<30} {'效率得分':<15} {'稳定性得分':<15} {'总分变化':<15}")
print("-" * 80)

for strategy_name, deltas in strategies:
    # 计算新得分
    new_ocr = current_ocr + deltas['ocr_delta']
    new_efficiency = 100 * (new_ocr - baseline_ocr) / baseline_ocr

    new_speed_std = current_speed_std - deltas['speed_std_delta']
    new_mean_accel = current_mean_accel - deltas['accel_delta']
    new_I_speed_std = -(new_speed_std - baseline_speed_std) / baseline_speed_std
    new_I_mean_accel = -(new_mean_accel - baseline_mean_accel) / baseline_mean_accel
    new_stability = 100 * (0.4 * max(0, new_I_speed_std) + 0.6 * max(0, new_I_mean_accel))

    new_total = (W_EFFICIENCY * new_efficiency +
                 W_STABILITY * new_stability) * 0.95

    total_delta = new_total - current_total

    print(f"{strategy_name:<30} {new_efficiency:<15.2f} {new_stability:<15.2f} {total_delta:+.2f}")

# 5. 加速度参数影响分析
print("\n\n5. 加速度参数对稳定性的影响")
print("=" * 80)

print("\n26分脚本 vs BC专家策略的参数差异:")
print("-" * 80)
print(f"{'参数':<20} {'26分脚本':<15} {'BC策略':<15} {'影响':<30}")
print(f"{'accel':<20} {'2.1':<15} {'0.8':<15} {'加速度响应速度不同':<30}")
print(f"{'tau':<20} {'0.9':<15} {'0.9':<15} {'跟驰时间':<30}")
print(f"{'sigma':<20} {'0.0':<15} {'0.0':<15} {'驾驶不完美性':<30}")

print("\n关键问题:")
print("  1. BC克隆使用了错误的accel参数 (0.8 vs 2.1)")
print("  2. accel=0.8会导致:")
print("     - 车辆加速缓慢，速度变化更平缓")
print("     - 速度标准差可能更低，但吞吐量也会降低")
print("     - 与26分脚本的行为不一致")

print("\n建议的修复:")
print("  1. 更新collect_expert_demos.py中的accel参数为2.1")
print("  2. 重新收集专家演示数据")
print("  3. 重新训练BC模型")

# 6. 超越OCR的优化建议
print("\n\n6. 超越OCR的优化建议")
print("=" * 80)

print("""
当OCR接近95%时，优化策略应该从"效率优先"转向"稳定性优先":

【稳定性优化】(权重0.5)
1. 速度平滑性优化
   - 降低速度标准差 σv
   - 方法: 温和的速度引导，避免急加速/急减速
   - 潜在收益: 速度标准差每降低1 m/s，稳定性得分提升约2-3分

2. 加速度平滑性优化
   - 降低平均绝对加速度 |a|_avg
   - 方法: 使用更温和的accel参数，增加tau（跟驰时间）
   - 潜在收益: 加速度每降低0.2 m/s²，稳定性得分提升约3-5分

3. 速度一致性优化
   - 减少车辆间的速度差异
   - 方法: 让所有CV车辆保持相似的速度

【干预成本优化】(指数惩罚)
1. 减少控制频率
   - 增加控制间隔（如从每步控制改为每5步控制）
   - 仅在关键时刻干预（如接近边末端50米时）

2. 减少动作变化幅度
   - 使用更平滑的动作变化
   - 避免频繁的速度调整

3. 智能选择控制车辆
   - 不控制所有CV，只控制关键位置的CV

【参数调优建议】
基于26分脚本的成功经验:
- CV accel = 2.1 (较高加速度，但通过主动控制保持平稳)
- CV tau = 0.9 (安全跟驰距离)
- CV sigma = 0.0 (完美驾驶)
- 控制间隔 = 5步
- 接近距离阈值 = 50米
- 拥堵速度阈值 = 5.0 m/s
- 速度系数 = 1.5
- 速度下限 = 3.0 m/s

【关键洞察】
26分脚本的成功不仅仅在于OCR高，更在于:
1. 高accel (2.1) 保证了吞吐量
2. 温和的主动控制 (仅在50米内干预) 保持了速度平滑
3. 适当的控制间隔 (5步) 减少了干预成本
4. vType参数优化 (sigma=0, tau=0.9) 改善了跟驰行为
""")

print("\n" + "=" * 80)
