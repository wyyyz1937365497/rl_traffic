"""
分析两个pkl提交文件，对比性能指标并指导奖励函数设计
基于实际的数据结构
"""

import pickle
import numpy as np


def load_pkl(pkl_path):
    """加载pkl文件"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def analyze_pkl(data):
    """分析pkl数据的各项指标"""
    stats = {}

    # 基本统计
    stats['total_steps'] = len(data['step_data'])
    stats['total_vehicles'] = len(data['vehicle_od_data'])

    # 从statistics中提取关键指标
    if 'statistics' in data:
        stats.update(data['statistics'])

    # 分析step_data - 每个时间步的状态
    step_data = data['step_data']

    active_vehicles_list = []
    arrived_vehicles_list = []
    departed_vehicles_list = []

    for step_info in step_data:
        active_vehicles_list.append(step_info.get('active_vehicles', 0))
        arrived_vehicles_list.append(step_info.get('arrived_vehicles', 0))
        departed_vehicles_list.append(step_info.get('departed_vehicles', 0))

    stats['mean_active_vehicles'] = np.mean(active_vehicles_list)
    stats['max_active_vehicles'] = np.max(active_vehicles_list)
    stats['final_arrived'] = arrived_vehicles_list[-1] if arrived_vehicles_list else 0
    stats['total_departed'] = departed_vehicles_list[-1] if departed_vehicles_list else 0

    # 分析vehicle_od_data - OD完成情况
    od_data = data['vehicle_od_data']
    arrived_count = 0
    enroute_count = 0
    total_distance_traveled = 0
    total_distance_required = 0

    for veh_id, od_info in od_data.items():
        distance_traveled = od_info.get('distance_traveled', 0)
        distance_total = od_info.get('distance_total', 1)

        if od_info.get('arrived', False):
            arrived_count += 1
            total_distance_traveled += distance_traveled
        else:
            enroute_count += 1
            total_distance_traveled += distance_traveled
            total_distance_required += distance_total

    stats['arrived_count'] = arrived_count
    stats['enroute_count'] = enroute_count
    stats['completion_rate'] = arrived_count / stats['total_vehicles'] if stats['total_vehicles'] > 0 else 0

    # 计算OCR (OD完成率)
    # OCR = (N_arrived + Σ(d_traveled / d_total)) / N_total
    if stats['total_vehicles'] > 0:
        if total_distance_required > 0:
            ocr = (arrived_count + total_distance_traveled / total_distance_required) / stats['total_vehicles']
        else:
            ocr = arrived_count / stats['total_vehicles']
    else:
        ocr = 0

    stats['ocr'] = ocr

    return stats


def compare_submissions(high_score_pkl, low_score_pkl, high_score, low_score):
    """对比两个提交文件"""
    print("=" * 80)
    print("提交文件对比分析")
    print("=" * 80)

    print(f"\n高分提交: {high_score_pkl} (得分: {high_score:.4f})")
    print(f"低分提交: {low_score_pkl} (得分: {low_score:.4f})")
    print(f"得分差异: {high_score - low_score:.4f} ({(high_score/low_score - 1)*100:.2f}% 提升)")

    # 分析两个文件
    print("\n" + "=" * 80)
    print("加载和分析数据...")
    print("=" * 80)

    high_data = load_pkl(high_score_pkl)
    low_data = load_pkl(low_score_pkl)

    high_stats = analyze_pkl(high_data)
    low_stats = analyze_pkl(low_data)

    # 对比结果
    print("\n" + "=" * 80)
    print("指标对比")
    print("=" * 80)

    print(f"\n{'指标':<30} {'高分':<15} {'低分':<15} {'差异':<15}")
    print("-" * 80)

    # 基本统计
    print(f"{'总时间步数':<30} {high_stats['total_steps']:<15} {low_stats['total_steps']:<15} {high_stats['total_steps'] - low_stats['total_steps']:<15}")
    print(f"{'总车辆数':<30} {high_stats['total_vehicles']:<15} {low_stats['total_vehicles']:<15} {high_stats['total_vehicles'] - low_stats['total_vehicles']:<15}")

    # 车辆活动情况
    print(f"\n{'--- 车辆活动情况 ---':<80}")
    print(f"{'平均活动车辆数':<30} {high_stats['mean_active_vehicles']:<15.1f} {low_stats['mean_active_vehicles']:<15.1f} {high_stats['mean_active_vehicles'] - low_stats['mean_active_vehicles']:<15.1f}")
    print(f"{'最大活动车辆数':<30} {high_stats['max_active_vehicles']:<15} {low_stats['max_active_vehicles']:<15} {high_stats['max_active_vehicles'] - low_stats['max_active_vehicles']:<15}")
    print(f"{'总发出车辆数':<30} {high_stats['total_departed']:<15} {low_stats['total_departed']:<15} {high_stats['total_departed'] - low_stats['total_departed']:<15}")

    # OD完成情况
    print(f"\n{'--- OD完成情况 ---':<80}")
    print(f"{'已到达车辆数':<30} {high_stats['arrived_count']:<15} {low_stats['arrived_count']:<15} {high_stats['arrived_count'] - low_stats['arrived_count']:<15}")
    print(f"{'在途车辆数':<30} {high_stats['enroute_count']:<15} {low_stats['enroute_count']:<15} {high_stats['enroute_count'] - low_stats['enroute_count']:<15}")
    print(f"{'完成率':<30} {high_stats['completion_rate']:<15.4f} {low_stats['completion_rate']:<15.4f} {(high_stats['completion_rate'] - low_stats['completion_rate']):<15.4f}")
    print(f"{'OCR (OD完成率)':<30} {high_stats['ocr']:<15.6f} {low_stats['ocr']:<15.6f} {(high_stats['ocr'] - low_stats['ocr']):<15.6f}")

    # 检查是否有额外的统计数据
    if 'mean_speed' in high_stats:
        print(f"\n{'--- 速度统计 ---':<80}")
        print(f"{'平均速度':<30} {high_stats.get('mean_speed', 0):<15.2f} {low_stats.get('mean_speed', 0):<15.2f} {high_stats.get('mean_speed', 0) - low_stats.get('mean_speed', 0):<15.2f}")
        print(f"{'速度标准差':<30} {high_stats.get('std_speed', 0):<15.2f} {low_stats.get('std_speed', 0):<15.2f} {high_stats.get('std_speed', 0) - low_stats.get('std_speed', 0):<15.2f}")

    if 'mean_abs_acceleration' in high_stats:
        print(f"\n{'--- 加速度统计 ---':<80}")
        print(f"{'平均绝对加速度':<30} {high_stats.get('mean_abs_acceleration', 0):<15.6f} {low_stats.get('mean_abs_acceleration', 0):<15.6f} {high_stats.get('mean_abs_acceleration', 0) - low_stats.get('mean_abs_acceleration', 0):<15.6f}")

    print("\n" + "=" * 80)
    print("奖励函数设计建议")
    print("=" * 80)

    # 根据对比结果给出建议
    print("\n根据评分公式，最终得分由三部分组成：")
    print("  S_total = (W_efficiency × S_efficiency + W_stability × S_stability) × e^(-k × C_int)")
    print("\n建议的奖励函数设计原则：\n")

    # 1. 效率奖励
    print("1. 效率奖励 (基于OCR):")
    print(f"   - 高分OCR: {high_stats['ocr']:.4f}, 低分OCR: {low_stats['ocr']:.4f}")
    ocr_improvement = (high_stats['ocr'] - low_stats['ocr']) / max(low_stats['ocr'], 0.001) * 100
    print(f"   - OCR改善: {ocr_improvement:+.2f}%")

    if high_stats['ocr'] > low_stats['ocr']:
        print("   → ✓ 高分提交OCR更高，应该重点奖励到达车辆数")

    print("\n   建议奖励项:")
    print("   ```python")
    print("   # 基础到达奖励")
    print("   reward_arrival = +10.0  # 每辆车到达目的地")
    print()
    print("   # 行驶进度奖励（鼓励车辆前进）")
    print("   reward_progress = +0.01  # 每米行驶距离")
    print()
    print("   # OCR增量奖励")
    print("   reward_ocr_delta = (OCR_t - OCR_{t-1}) * 100")
    print()
    print("   # 吞吐量奖励")
    print("   reward_throughput = +1.0  # 每辆新发出的车")
    print("   ```")

    # 2. 稳定性奖励
    print("\n2. 稳定性奖励 (基于速度标准差和加速度):")

    if 'std_speed' in high_stats:
        print(f"   - 高分速度标准差: {high_stats.get('std_speed', 0):.4f}")
        print(f"   - 低分速度标准差: {low_stats.get('std_speed', 0):.4f}")
        if high_stats.get('std_speed', 0) < low_stats.get('std_speed', 0):
            improvement = (low_stats.get('std_speed', 0) - high_stats.get('std_speed', 0)) / max(low_stats.get('std_speed', 0.001), 0.001) * 100
            print(f"   → ✓ 高分提交速度更稳定 (改善 {improvement:.2f}%)")

    if 'mean_abs_acceleration' in high_stats:
        print(f"   - 高分平均绝对加速度: {high_stats.get('mean_abs_acceleration', 0):.4f}")
        print(f"   - 低分平均绝对加速度: {low_stats.get('mean_abs_acceleration', 0):.4f}")
        if high_stats.get('mean_abs_acceleration', 0) < low_stats.get('mean_abs_acceleration', 0):
            improvement = (low_stats.get('mean_abs_acceleration', 0) - high_stats.get('mean_abs_acceleration', 0)) / max(low_stats.get('mean_abs_acceleration', 0.001), 0.001) * 100
            print(f"   → ✓ 高分提交加速度更平缓 (改善 {improvement:.2f}%)")

    print("\n   建议奖励项:")
    print("   ```python")
    print("   # 惩罚速度波动（速度标准差越小越好）")
    print("   reward_speed_smoothness = -std_speed × 0.5")
    print()
    print("   # 惩罚急加速/减速（加速度绝对值越小越好）")
    print("   reward_accel_smoothness = -mean(|acceleration|) × 1.0")
    print()
    print("   # 相邻时间步速度变化惩罚")
    print("   reward_speed_consistency = -mean(|speed_t - speed_{t-1}|) × 0.2")
    print()
    print("   # 最大加速度限制")
    print("   penalty_hard_accel = -2.0 if max(|accel|) > 3.0")
    print("   ```")

    # 3. 干预成本惩罚
    print("\n3. 干预成本惩罚 (C_int):")
    print("   评分公式中的干预成本:")
    print("   C_int = (1/(T × N)) × Σ(α × acmd + β × δlc)")
    print("   其中 α=1 (加速度指令), β=5 (换道指令)")

    print("\n   建议惩罚项:")
    print("   ```python")
    print("   # 动作变化惩罚（鼓励平稳控制）")
    print("   penalty_action_change = -0.001 × Σ|action_t - action_{t-1}|")
    print()
    print("   # 控制频率惩罚（减少不必要的干预）")
    print("   control_frequency = num_controlled_vehicles / total_vehicles")
    print("   penalty_control_freq = -0.01 × control_frequency")
    print()
    print("   # 换道惩罚（换道成本高）")
    print("   penalty_lane_change = -0.05 per lane change")
    print("   ```")

    # 4. 综合奖励函数
    print("\n4. 综合奖励函数建议:")
    print("   ```python")
    print("   def calculate_reward(env, actions, next_state):")
    print("       reward = 0.0")
    print()
    print("       # ===== 效率奖励 (权重 0.5) =====")
    print("       # 到达奖励")
    print("       newly_arrived = count_newly_arrived_vehicles()")
    print("       reward += newly_arrived * 10.0")
    print()
    print("       # 进度奖励")
    print("       total_distance = get_total_distance_traveled()")
    print("       reward += total_distance * 0.01")
    print()
    print("       # OCR增量")
    print("       ocr_delta = current_ocr - previous_ocr")
    print("       reward += ocr_delta * 100.0")
    print()
    print("       # ===== 稳定性奖励 (权重 0.3) =====")
    print("       # 速度平滑性")
    print("       speeds = [v.speed for v in vehicles]")
    print("       speed_std = np.std(speeds)")
    print("       reward -= speed_std * 0.5")
    print()
    print("       # 加速度平滑性")
    print("       accels = [v.acceleration for v in vehicles]")
    print("       mean_abs_accel = np.mean(np.abs(accels))")
    print("       reward -= mean_abs_accel * 1.0")
    print()
    print("       # ===== 干预成本惩罚 (权重 0.2) =====")
    print("       # 动作变化")
    print("       action_changes = sum(abs(a - prev_a))")
    print("       reward -= action_changes * 0.001")
    print()
    print("       return reward")
    print("   ```")

    print("\n" + "=" * 80)
    print("关键洞察")
    print("=" * 80)

    # 找出最关键的差异
    differences = []

    # OCR差异
    if high_stats['ocr'] != low_stats['ocr']:
        ocr_diff_pct = (high_stats['ocr'] - low_stats['ocr']) / max(low_stats['ocr'], 0.001) * 100
        differences.append(('OCR完成率提升', ocr_diff_pct, 'efficiency'))

    # 到达车辆数差异
    arrival_diff = high_stats['arrived_count'] - low_stats['arrived_count']
    if arrival_diff != 0:
        arrival_diff_pct = (high_stats['arrived_count'] - low_stats['arrived_count']) / max(low_stats['arrived_count'], 1) * 100
        differences.append(('到达车辆数增加', arrival_diff_pct, 'efficiency'))

    # 速度稳定性
    if 'std_speed' in high_stats and high_stats.get('std_speed', 0) != low_stats.get('std_speed', 0):
        speed_std_diff_pct = (low_stats.get('std_speed', 0) - high_stats.get('std_speed', 0)) / max(low_stats.get('std_speed', 0.001), 0.001) * 100
        differences.append(('速度标准差降低(稳定性提升)', speed_std_diff_pct, 'stability'))

    # 加速度平缓性
    if 'mean_abs_acceleration' in high_stats and high_stats.get('mean_abs_acceleration', 0) != low_stats.get('mean_abs_acceleration', 0):
        accel_diff_pct = (low_stats.get('mean_abs_acceleration', 0) - high_stats.get('mean_abs_acceleration', 0)) / max(low_stats.get('mean_abs_acceleration', 0.001), 0.001) * 100
        differences.append(('平均绝对加速度降低(平缓性提升)', accel_diff_pct, 'stability'))

    # 排序并显示最关键的差异
    differences.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n最关键的性能差异（按影响程度排序）:")
    for i, (name, diff_pct, category) in enumerate(differences[:5], 1):
        icon = "✓" if diff_pct > 0 else "✗"
        print(f"{i}. {icon} {name}: {diff_pct:+.2f}% [{category}]")

    print("\n" + "=" * 80)
    print("奖励函数权重建议")
    print("=" * 80)

    # 基于实际差异给出权重建议
    efficiency_score = sum([d[1] for d in differences if d[2] == 'efficiency'])
    stability_score = sum([d[1] for d in differences if d[2] == 'stability'])

    total_score = efficiency_score + stability_score
    if total_score > 0:
        w_efficiency = efficiency_score / total_score
        w_stability = stability_score / total_score

        print(f"\n基于数据分析的建议权重:")
        print(f"  - W_efficiency (效率): {w_efficiency:.2f}")
        print(f"  - W_stability (稳定性): {w_stability:.2f}")
        print(f"  - 干预成本惩罚: 指数衰减 e^(-k × C_int)")
        print(f"\n注意: 实际训练时可能需要根据收敛情况微调这些权重")

    print("\n" + "=" * 80)

    return high_stats, low_stats


if __name__ == '__main__':
    high_score_pkl = 'relu_based/rl_traffic/sumo/competition_results/submit.pkl'
    low_score_pkl = 'relu_based/rl_traffic/submission.pkl'

    high_score = 25.7926
    low_score = 15.7650

    compare_submissions(high_score_pkl, low_score_pkl, high_score, low_score)
