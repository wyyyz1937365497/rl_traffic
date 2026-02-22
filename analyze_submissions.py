"""
分析两个pkl提交文件，对比性能指标并指导奖励函数设计
"""

import pickle
import numpy as np
from collections import defaultdict


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

    # 分析step_data - 每个时间步的控制情况
    step_data = data['step_data']

    # 统计干预情况
    total_accel_commands = 0
    total_lane_change_commands = 0
    speed_changes = []

    for step_info in step_data.values():
        for veh_id, veh_actions in step_info.items():
            if isinstance(veh_actions, dict):
                # 统计加速度指令
                if veh_actions.get('acceleration', 0) != 0:
                    total_accel_commands += 1

                # 统计换道指令
                if veh_actions.get('lane_change', None) is not None:
                    total_lane_change_commands += 1

                # 记录速度变化
                speed = veh_actions.get('speed', None)
                if speed is not None:
                    speed_changes.append(speed)

    stats['total_accel_commands'] = total_accel_commands
    stats['total_lane_change_commands'] = total_lane_change_commands

    # 计算干预成本
    # C_int = (1/(T * N)) * Σ(α * acmd + β * δlc)
    # α = 1, β = 5
    alpha = 1
    beta = 5
    T = stats['total_steps']
    N = stats['total_vehicles']

    if T > 0 and N > 0:
        C_int = (alpha * total_accel_commands + beta * total_lane_change_commands) / (T * N)
    else:
        C_int = 0

    stats['intervention_cost'] = C_int

    # 分析vehicle_data - 车辆状态
    vehicle_data = data['vehicle_data']

    all_speeds = []
    all_accelerations = []
    all_positions = []

    for veh_id, veh_steps in vehicle_data.items():
        for step_info in veh_steps:
            speed = step_info.get('speed', 0)
            acceleration = step_info.get('acceleration', 0)
            position = step_info.get('position', 0)

            all_speeds.append(speed)
            all_accelerations.append(acceleration)
            all_positions.append(position)

    # 计算统计指标
    if all_speeds:
        stats['mean_speed'] = np.mean(all_speeds)
        stats['std_speed'] = np.std(all_speeds)
        stats['min_speed'] = np.min(all_speeds)
        stats['max_speed'] = np.max(all_speeds)

    if all_accelerations:
        stats['mean_abs_acceleration'] = np.mean(np.abs(all_accelerations))
        stats['std_acceleration'] = np.std(all_accelerations)
        stats['max_acceleration'] = np.max(np.abs(all_accelerations))

    # 分析OD完成情况
    od_data = data['vehicle_od_data']
    arrived_count = 0
    enroute_count = 0
    total_distance_traveled = 0
    total_distance_required = 0

    for veh_id, od_info in od_data.items():
        if od_info.get('arrived', False):
            arrived_count += 1
            total_distance_traveled += od_info.get('distance_traveled', 0)
        else:
            enroute_count += 1
            total_distance_traveled += od_info.get('distance_traveled', 0)
            total_distance_required += od_info.get('distance_total', 0)

    stats['arrived_count'] = arrived_count
    stats['enroute_count'] = enroute_count

    # 计算OCR
    # OCR = (N_arrived + Σ(d_traveled / d_total)) / N_total
    if stats['total_vehicles'] > 0:
        ocr = (arrived_count + total_distance_traveled / max(total_distance_required, 1)) / stats['total_vehicles']
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

    # 干预成本
    print(f"\n{'--- 干预成本 ---':<80}")
    print(f"{'总加速度指令数':<30} {high_stats['total_accel_commands']:<15} {low_stats['total_accel_commands']:<15} {high_stats['total_accel_commands'] - low_stats['total_accel_commands']:<15}")
    print(f"{'总换道指令数':<30} {high_stats['total_lane_change_commands']:<15} {low_stats['total_lane_change_commands']:<15} {high_stats['total_lane_change_commands'] - low_stats['total_lane_change_commands']:<15}")
    print(f"{'干预成本 (C_int)':<30} {high_stats['intervention_cost']:<15.6f} {low_stats['intervention_cost']:<15.6f} {high_stats['intervention_cost'] - low_stats['intervention_cost']:<15.6f}")

    # OD完成情况
    print(f"\n{'--- OD完成情况 ---':<80}")
    print(f"{'已到达车辆数':<30} {high_stats['arrived_count']:<15} {low_stats['arrived_count']:<15} {high_stats['arrived_count'] - low_stats['arrived_count']:<15}")
    print(f"{'在途车辆数':<30} {high_stats['enroute_count']:<15} {low_stats['enroute_count']:<15} {high_stats['enroute_count'] - low_stats['enroute_count']:<15}")
    print(f"{'OCR (完成率)':<30} {high_stats['ocr']:<15.6f} {low_stats['ocr']:<15.6f} {(high_stats['ocr'] - low_stats['ocr']):<15.6f}")

    # 速度统计
    if 'mean_speed' in high_stats and 'mean_speed' in low_stats:
        print(f"\n{'--- 速度统计 ---':<80}")
        print(f"{'平均速度':<30} {high_stats['mean_speed']:<15.2f} {low_stats['mean_speed']:<15.2f} {high_stats['mean_speed'] - low_stats['mean_speed']:<15.2f}")
        print(f"{'速度标准差':<30} {high_stats['std_speed']:<15.2f} {low_stats['std_speed']:<15.2f} {high_stats['std_speed'] - low_stats['std_speed']:<15.2f}")
        print(f"{'最小速度':<30} {high_stats['min_speed']:<15.2f} {low_stats['min_speed']:<15.2f} {high_stats['min_speed'] - low_stats['min_speed']:<15.2f}")
        print(f"{'最大速度':<30} {high_stats['max_speed']:<15.2f} {low_stats['max_speed']:<15.2f} {high_stats['max_speed'] - low_stats['max_speed']:<15.2f}")

    # 加速度统计
    if 'mean_abs_acceleration' in high_stats and 'mean_abs_acceleration' in low_stats:
        print(f"\n{'--- 加速度统计 ---':<80}")
        print(f"{'平均绝对加速度':<30} {high_stats['mean_abs_acceleration']:<15.6f} {low_stats['mean_abs_acceleration']:<15.6f} {high_stats['mean_abs_acceleration'] - low_stats['mean_abs_acceleration']:<15.6f}")
        print(f"{'加速度标准差':<30} {high_stats['std_acceleration']:<15.6f} {low_stats['std_acceleration']:<15.6f} {high_stats['std_acceleration'] - low_stats['std_acceleration']:<15.6f}")
        print(f"{'最大加速度':<30} {high_stats['max_acceleration']:<15.6f} {low_stats['max_acceleration']:<15.6f} {high_stats['max_acceleration'] - low_stats['max_acceleration']:<15.6f}")

    print("\n" + "=" * 80)
    print("奖励函数设计建议")
    print("=" * 80)

    # 根据对比结果给出建议
    print("\n根据评分公式，最终得分由三部分组成：")
    print("  S_total = (W_efficiency × S_efficiency + W_stability × S_stability) × e^(-k × C_int)")
    print("\n建议的奖励函数设计原则：\n")

    # 1. 效率奖励
    print("1. 效率奖励 (基于OCR):")
    print("   - 高分OCR: {:.4f}, 低分OCR: {:.4f}".format(high_stats['ocr'], low_stats['ocr']))
    if high_stats['ocr'] > low_stats['ocr']:
        print("   → ✓ 高分提交OCR更高，应该重点奖励到达车辆数")
    print("   建议奖励项:")
    print("   - reward_ocr = OCR_t - OCR_{t-1}  (OCR增量)")
    print("   - reward_arrival = +10.0  (每辆车到达)")
    print("   - reward_distance = +0.01  (每米行驶距离)")

    # 2. 稳定性奖励
    print("\n2. 稳定性奖励 (基于速度标准差和加速度):")
    if 'std_speed' in high_stats:
        print("   - 高分速度标准差: {:.4f}, 低分: {:.4f}".format(high_stats['std_speed'], low_stats['std_speed']))
        if high_stats['std_speed'] < low_stats['std_speed']:
            print("   → ✓ 高分提交速度更稳定")
    if 'mean_abs_acceleration' in high_stats:
        print("   - 高分平均绝对加速度: {:.4f}, 低分: {:.4f}".format(high_stats['mean_abs_acceleration'], low_stats['mean_abs_acceleration']))
        if high_stats['mean_abs_acceleration'] < low_stats['mean_abs_acceleration']:
            print("   → ✓ 高分提交加速度更平缓")
    print("   建议奖励项:")
    print("   - reward_smooth_speed = -std_speed × 0.1  (惩罚速度波动)")
    print("   - reward_smooth_accel = -mean(|accel|) × 0.5  (惩罚急加速/减速)")
    print("   - reward_consistent = -|speed_t - speed_{t-1}| × 0.1  (相邻时间步速度变化)")

    # 3. 干预成本惩罚
    print("\n3. 干预成本惩罚:")
    print("   - 高分干预成本: {:.6f}, 低分: {:.6f}".format(high_stats['intervention_cost'], low_stats['intervention_cost']))
    if high_stats['intervention_cost'] < low_stats['intervention_cost']:
        print("   → ✓ 高分提交干预更少")
    print("   建议惩罚项:")
    print("   - penalty_accel = -0.01 × (accel_cmd != 0)  (每次加速度指令)")
    print("   - penalty_lane_change = -0.05 × (lane_change_cmd != 0)  (每次换道指令)")
    print("   - penalty_action = -0.001 × |action - last_action|  (动作变化)")

    # 4. 综合奖励函数
    print("\n4. 综合奖励函数建议:")
    print("""
   def calculate_reward(state, action, next_state):
       # 效率奖励
       r_efficiency = 0
       if new_vehicle_arrived:
           r_efficiency += 10.0
       r_efficiency += total_distance_traveled * 0.01

       # 稳定性奖励
       r_stability = 0
       r_stability -= speed_std * 0.1
       r_stability -= mean_abs_acceleration * 0.5
       r_stability -= np.mean(np.abs(acceleration)) * 0.5

       # 干预成本惩罚
       r_intervention = 0
       for veh_id in controlled_vehicles:
           if action_changed[veh_id]:
               r_intervention -= 0.01
           if lane_change[veh_id]:
               r_intervention -= 0.05

       # 总奖励
       reward = r_efficiency + r_stability + r_intervention
       return reward
    """)

    print("\n" + "=" * 80)
    print("关键洞察")
    print("=" * 80)

    # 找出最关键的差异
    differences = []
    if high_stats['ocr'] != low_stats['ocr']:
        ocr_diff_pct = (high_stats['ocr'] - low_stats['ocr']) / max(low_stats['ocr'], 0.001) * 100
        differences.append(('OCR差异', ocr_diff_pct))

    if 'std_speed' in high_stats and high_stats['std_speed'] != low_stats['std_speed']:
        speed_std_diff_pct = (low_stats['std_speed'] - high_stats['std_speed']) / max(low_stats['std_speed'], 0.001) * 100
        differences.append(('速度稳定性改善', speed_std_diff_pct))

    if 'mean_abs_acceleration' in high_stats and high_stats['mean_abs_acceleration'] != low_stats['mean_abs_acceleration']:
        accel_diff_pct = (low_stats['mean_abs_acceleration'] - high_stats['mean_abs_acceleration']) / max(low_stats['mean_abs_acceleration'], 0.001) * 100
        differences.append(('加速度平缓性改善', accel_diff_pct))

    if high_stats['intervention_cost'] != low_stats['intervention_cost']:
        intervention_diff_pct = (low_stats['intervention_cost'] - high_stats['intervention_cost']) / max(low_stats['intervention_cost'], 0.001) * 100
        differences.append(('干预成本降低', intervention_diff_pct))

    # 排序并显示最关键的差异
    differences.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n最关键的性能差异（按影响程度排序）:")
    for i, (name, diff_pct) in enumerate(differences[:5], 1):
        print(f"{i}. {name}: {diff_pct:+.2f}%")

    return high_stats, low_stats


if __name__ == '__main__':
    high_score_pkl = 'relu_based/rl_traffic/sumo/competition_results/submit.pkl'
    low_score_pkl = 'relu_based/rl_traffic/submission.pkl'

    high_score = 25.7926
    low_score = 15.7650

    compare_submissions(high_score_pkl, low_score_pkl, high_score, low_score)
