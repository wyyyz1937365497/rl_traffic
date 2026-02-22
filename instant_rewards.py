"""
基于瞬时性能指标的奖励函数

核心原则：
1. 不使用OCR绝对值（会自然上升）
2. 关注瞬时性能：速度、排队、等待时间、冲突
3. 事件驱动奖励：车辆离开控制区、成功汇入
4. 大幅减小/移除生存奖励
"""
import numpy as np
from typing import Dict
from collections import defaultdict


class InstantRewardCalculator:
    """
    基于瞬时性能指标的奖励计算器

    奖励组成：
    1. 速度奖励 - 主路和匝道车辆速度
    2. 流量奖励 - 车辆离开控制区
    3. 汇入奖励 - 匝道车辆成功汇入主路
    4. 惩罚 - 排队、等待时间、冲突风险、急加减速
    5. 无生存奖励（或极小）
    """

    def __init__(self):
        # 奖励权重
        self.weights = {
            # 正向奖励
            'speed_main': 0.1,           # 主路速度奖励
            'speed_ramp': 0.05,          # 匝道速度奖励
            'throughput': 1.0,           # 车辆离开控制区奖励
            'merge_success': 0.5,        # 匝道车成功汇入奖励

            # 负向惩罚
            'queue_main': 0.01,          # 主路排队惩罚
            'queue_ramp': 0.02,          # 匝道排队惩罚（更重要）
            'waiting': 0.005,            # 等待时间惩罚
            'conflict': 0.05,            # 冲突风险惩罚
            'speed_variance': 0.01,      # 速度不平稳惩罚

            # 生存奖励（极小或为0）
            'survival': 0.001,           # 每步微小生存奖励（可设为0）
        }

        # 历史追踪
        self.previous_in_zone = defaultdict(int)
        self.previous_ramp_waiting = defaultdict(int)

    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """
        计算奖励

        Args:
            agents: 路口智能体字典
            env_stats: 环境统计信息

        Returns:
            每个路口的奖励字典
        """
        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # ===== 正向奖励 =====

            # 1. 速度奖励（归一化到 0-1）
            speed_main = min(state.main_speed / 15.0, 1.0)  # 15m/s为理想速度
            speed_ramp = min(state.ramp_speed / 10.0, 1.0)  # 10m/s为匝道理想速度
            speed_reward = (
                speed_main * self.weights['speed_main'] +
                speed_ramp * self.weights['speed_ramp']
            )

            # 2. 流量奖励（车辆离开控制区）
            current_in_zone = len(state.main_vehicles) + len(state.ramp_vehicles)
            departed_delta = max(0, self.previous_in_zone[junc_id] - current_in_zone)
            self.previous_in_zone[junc_id] = current_in_zone
            throughput_reward = departed_delta * self.weights['throughput']

            # 3. 汇入奖励（匝道车成功进入主路）
            merge_reward = 0.0
            if state.gap_acceptance > 0.5 and len(state.ramp_vehicles) > 0:
                # 有可接受间隙且匝道有车 → 给奖励
                merge_reward = state.gap_acceptance * self.weights['merge_success']

            # ===== 负向惩罚 =====

            # 1. 排队惩罚（线性增长）
            queue_penalty = -(
                state.main_queue_length * self.weights['queue_main'] +
                state.ramp_queue_length * self.weights['queue_ramp']
            )

            # 2. 等待时间惩罚（超过30s才惩罚）
            waiting_penalty = 0.0
            if state.ramp_waiting_time > 30:
                waiting_penalty = -(state.ramp_waiting_time - 30) * self.weights['waiting']

            # 3. 冲突风险惩罚
            conflict_penalty = -state.conflict_risk * self.weights['conflict']

            # 4. 速度不平稳惩罚（主路和匝道速度差异）
            speed_variance_penalty = -abs(state.main_speed - state.ramp_speed) * self.weights['speed_variance']

            # ===== 总奖励 =====
            total_reward = (
                speed_reward +
                throughput_reward +
                merge_reward +
                queue_penalty +
                waiting_penalty +
                conflict_penalty +
                speed_variance_penalty +
                self.weights['survival']  # 微小生存奖励
            )

            rewards[junc_id] = total_reward

            # 记录奖励详情
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'speed_reward': speed_reward,
                    'throughput_reward': throughput_reward,
                    'merge_reward': merge_reward,
                    'queue_penalty': queue_penalty,
                    'waiting_penalty': waiting_penalty,
                    'conflict_penalty': conflict_penalty,
                    'speed_variance_penalty': speed_variance_penalty,
                    'survival': self.weights['survival'],
                    'total': total_reward
                }

        return rewards

    def reset(self):
        """重置追踪状态（新episode开始时调用）"""
        self.previous_in_zone.clear()
        self.previous_ramp_waiting.clear()


class NormalizedInstantRewardCalculator:
    """
    归一化的瞬时奖励计算器

    改进：
    - 所有奖励分量归一化到相似尺度
    - 使用非线性变换避免极端值
    """

    def __init__(self):
        # 奖励权重（调整到相似尺度）
        self.weights = {
            'speed': 0.2,              # 速度奖励权重
            'throughput': 2.0,         # 离开奖励权重
            'queue': 0.02,             # 排队惩罚权重
            'waiting': 0.01,           # 等待惩罚权重
            'conflict': 0.1,           # 冲突惩罚权重
            'survival': 0.0001,        # 极小生存奖励
        }

        self.previous_in_zone = defaultdict(int)
        self.episode_throughput = defaultdict(int)

    def sigmoid(self, x, scale=1.0):
        """Sigmoid归一化"""
        return 1.0 / (1.0 + np.exp(-scale * x))

    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """计算归一化奖励"""
        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # ===== 1. 速度奖励（sigmoid归一化）=====
            # 主路速度：目标13.89m/s (50km/h)
            speed_score = self.sigmoid((state.main_speed - 8.0) / 5.0)  # 8m/s为中点
            speed_reward = speed_score * self.weights['speed']

            # ===== 2. 流量奖励 =====
            current_in_zone = len(state.main_vehicles) + len(state.ramp_vehicles)
            departed_delta = max(0, self.previous_in_zone[junc_id] - current_in_zone)
            self.previous_in_zone[junc_id] = current_in_zone
            self.episode_throughput[junc_id] += departed_delta
            throughput_reward = departed_delta * self.weights['throughput']

            # ===== 3. 排队惩罚（平方惩罚，更敏感）=====
            queue_penalty = -(
                state.main_queue_length ** 1.5 * self.weights['queue'] * 0.5 +
                state.ramp_queue_length ** 1.5 * self.weights['queue']
            )

            # ===== 4. 等待惩罚（分段）=====
            waiting_penalty = 0.0
            if state.ramp_waiting_time > 20:
                waiting_penalty = -((state.ramp_waiting_time - 20) ** 0.8) * self.weights['waiting']

            # ===== 5. 冲突惩罚（非线性）=====
            conflict_penalty = -(state.conflict_risk ** 2) * self.weights['conflict']

            # ===== 总奖励 =====
            total_reward = (
                speed_reward +
                throughput_reward +
                queue_penalty +
                waiting_penalty +
                conflict_penalty +
                self.weights['survival']
            )

            # 裁剪到合理范围
            total_reward = np.clip(total_reward, -5.0, 5.0)

            rewards[junc_id] = total_reward

            # 记录奖励详情
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'speed_reward': speed_reward,
                    'speed_score': speed_score,
                    'throughput_reward': throughput_reward,
                    'queue_penalty': queue_penalty,
                    'waiting_penalty': waiting_penalty,
                    'conflict_penalty': conflict_penalty,
                    'survival': self.weights['survival'],
                    'total': total_reward
                }

        return rewards

    def reset(self):
        """重置追踪状态"""
        self.previous_in_zone.clear()
        self.episode_throughput.clear()


class ShapedInstantRewardCalculator:
    """
    Reward Shaping版本的瞬时奖励

    使用势函数进行reward shaping，不改变最优策略
    """

    def __init__(self):
        self.weights = {
            'speed': 0.1,
            'throughput': 1.0,
            'queue': 0.01,
            'waiting': 0.005,
            'conflict': 0.05,
            'survival': 0.0,  # 完全移除生存奖励
        }

        self.previous_in_zone = defaultdict(int)
        self.previous_queue = defaultdict(lambda: {'main': 0, 'ramp': 0})

    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """计算shaped奖励"""
        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # 基础奖励
            base_reward = 0.0

            # 1. 速度奖励
            base_reward += (state.main_speed / 15.0) * self.weights['speed']

            # 2. 流量奖励
            current_in_zone = len(state.main_vehicles) + len(state.ramp_vehicles)
            departed_delta = max(0, self.previous_in_zone[junc_id] - current_in_zone)
            self.previous_in_zone[junc_id] = current_in_zone
            base_reward += departed_delta * self.weights['throughput']

            # 3. 惩罚项
            base_reward -= (
                state.main_queue_length * self.weights['queue'] * 0.5 +
                state.ramp_queue_length * self.weights['queue']
            )

            if state.ramp_waiting_time > 30:
                base_reward -= (state.ramp_waiting_time - 30) * self.weights['waiting']

            base_reward -= state.conflict_risk * self.weights['conflict']

            # Reward Shaping: 排队改善的额外奖励
            # (当前排队 - 上次排队) 的负值 = 排队减少的奖励
            queue_improvement = (
                (self.previous_queue[junc_id]['main'] - state.main_queue_length) * 0.5 +
                (self.previous_queue[junc_id]['ramp'] - state.ramp_queue_length)
            ) * self.weights['queue']

            self.previous_queue[junc_id] = {
                'main': state.main_queue_length,
                'ramp': state.ramp_queue_length
            }

            # 总奖励 = 基础奖励 + shaping奖励
            total_reward = base_reward + queue_improvement

            # 裁剪
            total_reward = np.clip(total_reward, -3.0, 3.0)

            rewards[junc_id] = total_reward

            # 记录奖励详情
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'base_reward': base_reward,
                    'queue_improvement': queue_improvement,
                    'total': total_reward
                }

        return rewards

    def reset(self):
        """重置追踪状态"""
        self.previous_in_zone.clear()
        self.previous_queue.clear()


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("瞬时奖励函数测试")
    print("=" * 70)

    # 模拟测试
    class MockState:
        def __init__(self, main_speed=10.0, ramp_speed=5.0,
                     main_queue=2, ramp_queue=1, ramp_waiting=10.0,
                     conflict_risk=0.3, gap_acceptance=0.6):
            self.main_speed = main_speed
            self.ramp_speed = ramp_speed
            self.main_queue_length = main_queue
            self.ramp_queue_length = ramp_queue
            self.ramp_waiting_time = ramp_waiting
            self.conflict_risk = conflict_risk
            self.gap_acceptance = gap_acceptance
            self.main_vehicles = [{'speed': main_speed}] * 10
            self.ramp_vehicles = [{'speed': ramp_speed}] * 3

    class MockAgent:
        def __init__(self):
            self.current_state = MockState()
            self.reward_breakdown = {}

    # 测试1：正常状态
    print("\n【测试1】正常状态")
    agents = {'J5': MockAgent()}
    calc = NormalizedInstantRewardCalculator()
    rewards = calc.compute_rewards(agents, {})
    print(f"奖励: {rewards['J5']:.4f}")
    print(f"详情: {agents['J5'].reward_breakdown}")

    # 测试2：拥堵状态
    print("\n【测试2】拥堵状态（速度低、排队多）")
    calc.reset()
    agents['J5'].current_state = MockState(main_speed=3.0, ramp_speed=1.0,
                                           main_queue=10, ramp_queue=8,
                                           ramp_waiting=60.0, conflict_risk=0.8)
    rewards = calc.compute_rewards(agents, {})
    print(f"奖励: {rewards['J5']:.4f}")
    print(f"详情: {agents['J5'].reward_breakdown}")

    # 测试3：理想状态
    print("\n【测试3】理想状态（速度快、无排队）")
    calc.reset()
    agents['J5'].current_state = MockState(main_speed=13.0, ramp_speed=8.0,
                                           main_queue=0, ramp_queue=0,
                                           ramp_waiting=5.0, conflict_risk=0.1,
                                           gap_acceptance=0.9)
    rewards = calc.compute_rewards(agents, {})
    print(f"奖励: {rewards['J5']:.4f}")
    print(f"详情: {agents['J5'].reward_breakdown}")

    print("\n✅ 测试完成")
