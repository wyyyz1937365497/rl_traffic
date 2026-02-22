"""
分析两个pkl提交文件，对比性能指标并指导奖励函数设计
最终版本 - 基于真实统计数据
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

    # 从statistics中提取关键指标
    if 'statistics' in data:
        stats['cumulative_departed'] = data['statistics'].get('cumulative_departed', 0)
        stats['cumulative_arrived'] = data['statistics'].get('cumulative_arrived', 0)
        stats['maxspeed_violations'] = len(data['statistics'].get('maxspeed_violations', []))

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
    stats['final_active'] = active_vehicles_list[-1]
    stats['final_arrived'] = arrived_vehicles_list[-1]
    stats['total_departed'] = departed_vehicles_list[-1]

    # 计算完成率（OCR）
    stats['completion_rate'] = stats['final_arrived'] / stats['total_departed'] if stats['total_departed'] > 0 else 0
    stats['ocr'] = stats['completion_rate']  # 在这个上下文中，OCR ≈ completion rate

    # 计算到达率
    stats['arrival_rate'] = stats['final_arrived'] / stats['total_steps'] if stats['total_steps'] > 0 else 0

    return stats


def compare_submissions(high_score_pkl, low_score_pkl, high_score, low_score):
    """对比两个提交文件"""
    print("=" * 80)
    print("提交文件对比分析 - 奖励函数设计指导")
    print("=" * 80)

    print(f"\n高分提交: {high_score_pkl}")
    print(f"  得分: {high_score:.4f}")
    print(f"\n低分提交: {low_score_pkl}")
    print(f"  得分: {low_score:.4f}")
    print(f"\n得分差异: {high_score - low_score:.4f} ({(high_score/low_score - 1)*100:.2f}% 提升)")

    # 加载和分析数据
    print("\n" + "=" * 80)
    print("数据加载与分析")
    print("=" * 80)

    high_data = load_pkl(high_score_pkl)
    low_data = load_pkl(low_score_pkl)

    high_stats = analyze_pkl(high_data)
    low_stats = analyze_pkl(low_data)

    # 对比结果
    print("\n" + "=" * 80)
    print("性能指标对比")
    print("=" * 80)

    print(f"\n{'指标':<30} {'高分提交':<20} {'低分提交':<20} {'差异':<20}")
    print("-" * 90)

    # 基本统计
    print(f"{'总时间步数':<30} {high_stats['total_steps']:<20} {low_stats['total_steps']:<20} {high_stats['total_steps'] - low_stats['total_steps']:<20}")

    # 车辆吞吐量
    print(f"\n{'--- 车辆吞吐量 ---':<90}")
    print(f"{'总发出车辆数':<30} {high_stats['total_departed']:<20} {low_stats['total_departed']:<20} {high_stats['total_departed'] - low_stats['total_departed']:<20}")
    print(f"{'总到达车辆数':<30} {high_stats['final_arrived']:<20} {low_stats['final_arrived']:<20} {high_stats['final_arrived'] - low_stats['final_arrived']:<20}")

    departure_diff_pct = (high_stats['total_departed'] - low_stats['total_departed']) / max(low_stats['total_departed'], 1) * 100
    arrival_diff_pct = (high_stats['final_arrived'] - low_stats['final_arrived']) / max(low_stats['final_arrived'], 1) * 100
    print(f"{'发出车辆数差异':<30} {'':<20} {'':<20} {departure_diff_pct:+.2f}%")
    print(f"{'到达车辆数差异':<30} {'':<20} {'':<20} {arrival_diff_pct:+.2f}%")

    # 完成率/OCR
    print(f"\n{'--- 完成效率 ---':<90}")
    print(f"{'OCR (完成率)':<30} {high_stats['ocr']:<20.4f} {low_stats['ocr']:<20.4f} {(high_stats['ocr'] - low_stats['ocr']):<20.4f}")
    ocr_improvement = (high_stats['ocr'] - low_stats['ocr']) / max(low_stats['ocr'], 0.001) * 100
    print(f"{'OCR相对改善':<30} {'':<20} {'':<20} {ocr_improvement:+.2f}%")

    print(f"{'到达率 (车辆/秒)':<30} {high_stats['arrival_rate']:<20.2f} {low_stats['arrival_rate']:<20.2f} {(high_stats['arrival_rate'] - low_stats['arrival_rate']):<20.2f}")

    # 活动车辆情况
    print(f"\n{'--- 路网负载 ---':<90}")
    print(f"{'平均活动车辆数':<30} {high_stats['mean_active_vehicles']:<20.1f} {low_stats['mean_active_vehicles']:<20.1f} {high_stats['mean_active_vehicles'] - low_stats['mean_active_vehicles']:<20.1f}")
    print(f"{'最大活动车辆数':<30} {high_stats['max_active_vehicles']:<20} {low_stats['max_active_vehicles']:<20} {high_stats['max_active_vehicles'] - low_stats['max_active_vehicles']:<20}")
    print(f"{'仿真结束时在途车辆':<30} {high_stats['final_active']:<20} {low_stats['final_active']:<20} {high_stats['final_active'] - low_stats['final_active']:<20}")

    # 超速违规
    print(f"\n{'--- 安全性 ---':<90}")
    print(f"{'超速违规次数':<30} {high_stats['maxspeed_violations']:<20} {low_stats['maxspeed_violations']:<20} {high_stats['maxspeed_violations'] - low_stats['maxspeed_violations']:<20}")

    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    findings = []

    # 吞吐量优势
    if high_stats['total_departed'] > low_stats['total_departed']:
        findings.append(f"✓ 高分提交多发出了 {high_stats['total_departed'] - low_stats['total_departed']} 辆车 ({departure_diff_pct:.1f}%)")

    # 到达率优势
    if high_stats['final_arrived'] > low_stats['final_arrived']:
        findings.append(f"✓ 高分提交多到达了 {high_stats['final_arrived'] - low_stats['final_arrived']} 辆车 ({arrival_diff_pct:.1f}%)")

    # OCR优势
    if high_stats['ocr'] > low_stats['ocr']:
        findings.append(f"✓ 高分提交OCR更高 ({high_stats['ocr']:.4f} vs {low_stats['ocr']:.4f}, 提升{ocr_improvement:.2f}%)")

    # 路网负载
    if high_stats['mean_active_vehicles'] < low_stats['mean_active_vehicles']:
        findings.append(f"✓ 高分提交平均活动车辆更少 (路网更通畅, {high_stats['mean_active_vehicles']:.0f} vs {low_stats['mean_active_vehicles']:.0f})")

    if high_stats['final_active'] < low_stats['final_active']:
        findings.append(f"✓ 高分提交仿真结束时在途车辆更少 (拥堵更少, {high_stats['final_active']} vs {low_stats['final_active']})")

    # 安全性
    if high_stats['maxspeed_violations'] < low_stats['maxspeed_violations']:
        findings.append(f"✓ 高分提交超速违规更少 ({high_stats['maxspeed_violations']} vs {low_stats['maxspeed_violations']})")

    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")

    print("\n" + "=" * 80)
    print("奖励函数设计建议")
    print("=" * 80)

    print("\n根据评分公式，最终得分由三部分组成：")
    print("  S_total = (W_efficiency × S_efficiency + W_stability × S_stability) × e^(-k × C_int)")
    print()
    print("其中:")
    print("  - S_efficiency = 100 × max(0, (OCR_AI - OCR_Base) / OCR_Base)")
    print("  - S_stability = 100 × [0.4 × I_σv + 0.6 × I_|a|_avg]")
    print("  - P_int = e^(-k × C_int)  [干预成本惩罚]")

    # 基于分析结果给出具体的奖励函数建议
    print("\n" + "-" * 80)
    print("1. 效率奖励 (Efficiency Rewards)")
    print("-" * 80)

    print(f"\n当前数据分析:")
    print(f"  - 高分提交OCR: {high_stats['ocr']:.4f}, 到达车辆: {high_stats['final_arrived']}")
    print(f"  - 低分提交OCR: {low_stats['ocr']:.4f}, 到达车辆: {low_stats['final_arrived']}")
    print(f"  - 差异: 多到达{high_stats['final_arrived'] - low_stats['final_arrived']}辆车, OCR提升{ocr_improvement:.2f}%")

    print("\n建议奖励项:")
    print("""
```python
# ===== 效率奖励 =====
def calculate_efficiency_reward(env, prev_state, curr_state):
    reward = 0.0

    # 1.1 车辆到达奖励 (最重要!)
    newly_arrived = curr_state.arrived_count - prev_state.arrived_count
    reward += newly_arrived * 10.0  # 每到达一辆车 +10分

    # 1.2 发车奖励 (鼓励更多车辆进入路网)
    newly_departed = curr_state.departed_count - prev_state.departed_count
    reward += newly_departed * 1.0  # 每发车一辆 +1分

    # 1.3 行驶进度奖励 (鼓励车辆前进，不要停滞)
    total_distance_traveled = get_total_distance_traveled(env)
    reward += total_distance_traveled * 0.01  # 每米 +0.01分

    # 1.4 OCR增量奖励 (直接优化评分指标)
    current_ocr = curr_state.arrived_count / max(curr_state.departed_count, 1)
    previous_ocr = prev_state.arrived_count / max(prev_state.departed_count, 1)
    ocr_delta = current_ocr - previous_ocr
    reward += ocr_delta * 100.0  # OCR提升1% = +1分

    # 1.5 吞吐量奖励
    throughput = curr_state.arrived_count / curr_state.step
    reward += throughput * 0.1

    return reward
```
    """)

    print("-" * 80)
    print("2. 稳定性奖励 (Stability Rewards)")
    print("-" * 80)

    print("\n稳定性指标 (根据评分公式):")
    print("  - 速度标准差 σv (越小越好, 权重0.4)")
    print("  - 平均绝对加速度 |a|_avg (越小越好, 权重0.6)")

    print("\n建议奖励项:")
    print("""
```python
# ===== 稳定性奖励 =====
def calculate_stability_reward(env, prev_state, curr_state):
    reward = 0.0

    # 2.1 速度平滑性 (降低速度标准差)
    speeds = [v.speed for v in env.vehicles]
    speed_std = np.std(speeds)
    reward -= speed_std * 0.5  # 速度标准差每增加1，扣0.5分

    # 2.2 加速度平滑性 (降低急加速/急减速)
    accels = [v.acceleration for v in env.vehicles]
    mean_abs_accel = np.mean(np.abs(accels))
    reward -= mean_abs_accel * 1.0  # 平均绝对加速度每增加1，扣1分

    # 2.3 相邻时间步速度一致性
    for v in env.vehicles:
        speed_change = abs(v.speed - v.prev_speed)
        reward -= speed_change * 0.1

    # 2.4 最大加速度限制 (惩罚过激驾驶)
    max_accel = max([abs(v.acceleration) for v in env.vehicles])
    if max_accel > 3.0:  # 超过3m/s²
        reward -= (max_accel - 3.0) * 2.0

    # 2.5 速度一致性奖励 (鼓励车辆速度相近，减少速度差)
    if len(speeds) > 1:
        speed_range = max(speeds) - min(speeds)
        reward -= speed_range * 0.05

    return reward
```
    """)

    print("-" * 80)
    print("3. 干预成本惩罚 (Intervention Cost Penalty)")
    print("-" * 80)

    print("\n干预成本公式:")
    print("  C_int = (1/(T × N)) × Σ(α × acmd + β × δlc)")
    print("  其中 α=1 (加速度指令), β=5 (换道指令)")

    print("\n建议惩罚项:")
    print("""
```python
# ===== 干预成本惩罚 =====
def calculate_intervention_penalty(actions, prev_actions, controlled_vehicles):
    penalty = 0.0

    # 3.1 动作变化惩罚 (鼓励平稳控制，减少频繁调整)
    for veh_id in controlled_vehicles:
        if veh_id in actions and veh_id in prev_actions:
            action_change = abs(actions[veh_id] - prev_actions[veh_id])
            penalty -= action_change * 0.001

    # 3.2 控制频率惩罚 (控制车辆越少越好)
    control_ratio = len(controlled_vehicles) / max(total_vehicles, 1)
    penalty -= control_ratio * 0.01

    # 3.3 换道惩罚 (换道成本高，β=5)
    for veh_id in controlled_vehicles:
        if lane_change_executed(veh_id):
            penalty -= 0.05  # 每次换道扣0.05分

    # 3.4 加速度指令惩罚 (频繁下达加速度指令会增加成本)
    for veh_id in controlled_vehicles:
        if accel_command_issued(veh_id):
            penalty -= 0.001

    return penalty
```
    """)

    print("-" * 80)
    print("4. 综合奖励函数")
    print("-" * 80)

    print("\n完整实现:")
    print()
    print("```python")
    print("class TrafficRewardCalculator:")
    def __init__(self,
                 w_efficiency=0.5,
                 w_stability=0.3,
                 w_intervention=0.2):
        self.w_efficiency = w_efficiency
        self.w_stability = w_stability
        self.w_intervention = w_intervention

        self.prev_state = None
        self.prev_actions = {}

    def calculate_reward(self, env, actions, controlled_vehicles):
        # 获取当前状态
        curr_state = self._get_state(env)

        # 初始化prev_state
        if self.prev_state is None:
            self.prev_state = curr_state
            return 0.0

        # 计算各项奖励
        r_efficiency = self._calculate_efficiency_reward(env, curr_state)
        r_stability = self._calculate_stability_reward(env, curr_state)
        r_intervention = self._calculate_intervention_penalty(
            actions, self.prev_actions, controlled_vehicles
        )

        # 总奖励
        reward = (self.w_efficiency * r_efficiency +
                  self.w_stability * r_stability +
                  self.w_intervention * r_intervention)

        # 更新状态
        self.prev_state = curr_state
        self.prev_actions = actions.copy()

        return reward

    def _calculate_efficiency_reward(self, env, curr_state):
        """效率奖励 - 基于OCR"""
        reward = 0.0

        # 到达奖励
        newly_arrived = curr_state.arrived_count - self.prev_state.arrived_count
        reward += newly_arrived * 10.0

        # 发车奖励
        newly_departed = curr_state.departed_count - self.prev_state.departed_count
        reward += newly_departed * 1.0

        # OCR增量奖励
        current_ocr = curr_state.arrived_count / max(curr_state.departed_count, 1)
        previous_ocr = self.prev_state.arrived_count / max(self.prev_state.departed_count, 1)
        ocr_delta = current_ocr - previous_ocr
        reward += ocr_delta * 100.0

        return reward

    def _calculate_stability_reward(self, env, curr_state):
        """稳定性奖励 - 基于速度标准差和加速度"""
        reward = 0.0

        vehicles = env.get_vehicles()
        if not vehicles:
            return reward

        # 速度平滑性
        speeds = [v.speed for v in vehicles]
        speed_std = np.std(speeds)
        reward -= speed_std * 0.5

        # 加速度平滑性
        accels = [v.acceleration for v in vehicles]
        mean_abs_accel = np.mean(np.abs(accels))
        reward -= mean_abs_accel * 1.0

        return reward

    def _calculate_intervention_penalty(self, actions, prev_actions, controlled_vehicles):
        """干预成本惩罚"""
        penalty = 0.0

        for veh_id in controlled_vehicles:
            if veh_id in actions and veh_id in prev_actions:
                # 动作变化惩罚
                action_change = abs(actions[veh_id] - prev_actions[veh_id])
                penalty -= action_change * 0.001

        return penalty
```

    print("\n" + "=" * 80)
    print("奖励函数权重建议")
    print("=" * 80)

    # 基于实际得分差异给出权重建议
    print(f"\n得分分析:")
    print(f"  - 高分提交: {high_score:.4f} 分")
    print(f"  - 低分提交: {low_score:.4f} 分")
    print(f"  - 差异: {high_score - low_score:.4f} 分 ({(high_score/low_score - 1)*100:.1f}% 提升)")

    print("\n基于数据分析的建议:")
    print("  由于高分提交主要优势在于:")
    print(f"    1. 多发出 {high_stats['total_departed'] - low_stats['total_departed']} 辆车")
    print(f"    2. 多到达 {high_stats['final_arrived'] - low_stats['final_arrived']} 辆车")
    print(f"    3. OCR提升 {ocr_improvement:.2f}%")

    print("\n  建议权重配置:")
    print("    - W_efficiency = 0.6  (效率是主要因素)")
    print("    - W_stability = 0.2   (稳定性次要)")
    print("    - W_intervention = 0.2 (干预成本控制)")

    print("\n  训练策略:")
    print("    1. 前期训练: 使用较高的效率权重 (0.7) 快速提升OCR")
    print("    2. 中期训练: 逐步增加稳定性权重 (0.3) 优化驾驶平顺性")
    print("    3. 后期训练: 引入干预成本惩罚 (0.2) 减少不必要的控制")

    print("\n" + "=" * 80)
    print("实施建议")
    print("=" * 80)

    print("""
1. **优先实现效率奖励**:
   - 车辆到达奖励: +10分 (最重要)
   - OCR增量奖励: ×100倍放大
   - 这两项直接对应评分公式，应该优先实现

2. **逐步引入稳定性奖励**:
   - 先实现速度标准差惩罚 (权重0.5)
   - 等模型收敛后再加入加速度惩罚 (权重1.0)

3. **谨慎使用干预成本惩罚**:
   - 训练初期不要惩罚干预，让模型学会控制
   - 训练后期再引入干预成本惩罚

4. **监控指标**:
   - 每个epoch记录: OCR, 到达车辆数, 平均速度, 速度标准差
   - 对比baseline: 确保AI方案确实优于baseline

5. **超参数调优**:
   - 根据实际训练效果调整各项权重
   - 如果OCR提升不明显，增加效率奖励权重
   - 如果速度波动过大，增加稳定性惩罚权重
    """)

    print("\n" + "=" * 80)

    return high_stats, low_stats


if __name__ == '__main__':
    high_score_pkl = 'relu_based/rl_traffic/sumo/competition_results/submit.pkl'
    low_score_pkl = 'relu_based/rl_traffic/submission.pkl'

    high_score = 25.7926
    low_score = 15.7650

    compare_submissions(high_score_pkl, low_score_pkl, high_score, low_score)
