"""
基于OCR增量的奖励函数

核心思想：
1. 定期计算OCR增量（每N步一次）
2. 将增量奖励分配到这N步中
3. 使奖励信号与目标（OCR）直接相关
"""
import numpy as np
from typing import Dict
from collections import defaultdict


class OCRIncrementRewardCalculator:
    """
    基于OCR增量的奖励计算器

    原理：
    - 每N步计算一次OCR（例如N=100）
    - 如果OCR增加，给正奖励
    - 如果OCR减少，给负奖励
    - 将增量奖励分配到这N步中的每一步
    """

    def __init__(self, reward_interval: int = 100):
        """
        Args:
            reward_interval: 计算OCR增量的间隔步数
        """
        self.reward_interval = reward_interval

        # 历史OCR追踪
        self.previous_ocr = defaultdict(float)
        self.step_counter = defaultdict(int)

        # 奖励权重
        self.ocr_improve_bonus = 10.0  # OCR改善奖励系数
        self.ocr_decline_penalty = 10.0  # OCR退化惩罚系数

        # 辅助奖励权重（保持较小）
        self.speed_weight = 0.01  # 速度奖励权重
        self.queue_penalty_weight = 0.001  # 排队惩罚权重

    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """
        计算奖励

        Args:
            agents: 路口智能体字典
            env_stats: 环境统计信息，必须包含 'ocr' 和 'step'

        Returns:
            每个路口的奖励字典
        """
        current_ocr = env_stats.get('ocr', 0.0)
        current_step = env_stats.get('step', 0)

        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # 初始化
            if junc_id not in self.previous_ocr:
                self.previous_ocr[junc_id] = current_ocr
                self.step_counter[junc_id] = current_step

            # 计算OCR增量奖励
            ocr_reward = 0.0

            # 每隔reward_interval步计算一次增量
            if current_step - self.step_counter[junc_id] >= self.reward_interval:
                ocr_delta = current_ocr - self.previous_ocr[junc_id]

                # OCR改善 → 正奖励
                if ocr_delta > 0:
                    ocr_reward = ocr_delta * self.ocr_improve_bonus
                # OCR退化 → 负奖励
                elif ocr_delta < 0:
                    ocr_reward = ocr_delta * self.ocr_decline_penalty

                # 更新历史
                self.previous_ocr[junc_id] = current_ocr
                self.step_counter[junc_id] = current_step

            # 计算辅助奖励（较小权重）
            speed_reward = (state.main_speed + state.ramp_speed) / 20.0 * self.speed_weight
            queue_penalty = -(state.main_queue_length + state.ramp_queue_length) * self.queue_penalty_weight

            # 总奖励
            total_reward = ocr_reward + speed_reward + queue_penalty

            rewards[junc_id] = total_reward

            # 记录奖励详情（用于调试）
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'ocr_reward': ocr_reward,
                    'speed_reward': speed_reward,
                    'queue_penalty': queue_penalty,
                    'total': total_reward
                }

        return rewards

    def reset(self):
        """重置追踪状态（新episode开始时调用）"""
        self.previous_ocr.clear()
        self.step_counter.clear()


class DenseOCRIncrementRewardCalculator:
    """
    密集OCR增量奖励计算器

    改进：
    - 每步都计算OCR，但使用滑动平均平滑
    - 基于OCR趋势给奖励
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: OCR滑动平均窗口大小
        """
        self.window_size = window_size

        # OCR历史（用于计算趋势）
        self.cr_history = defaultdict(list)

        # 奖励权重
        self.cr_trend_bonus = 50.0  # OCR趋势奖励系数
        self.speed_weight = 0.01
        self.queue_penalty_weight = 0.001
        self.departure_weight = 0.1  # 车辆离开奖励

    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """
        计算奖励（基于OCR趋势）
        """
        current_ocr = env_stats.get('ocr', 0.0)

        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # 更新OCR历史
            self.ocr_history[junc_id].append(current_ocr)
            if len(self.ocr_history[junc_id]) > self.window_size:
                self.ocr_history[junc_id].pop(0)

            # 计算OCR趋势
            ocr_trend_reward = 0.0
            if len(self.ocr_history[junc_id]) >= self.window_size:
                # 计算最近window_size步的OCR变化率
                recent_avg = np.mean(self.ocr_history[junc_id][-self.window_size//2:])
                old_avg = np.mean(self.ocr_history[junc_id][:self.window_size//2])
                ocr_trend = recent_avg - old_avg

                ocr_trend_reward = ocr_trend * self.ocr_trend_bonus

            # 辅助奖励
            speed_reward = (state.main_speed + state.ramp_speed) / 20.0 * self.speed_weight
            queue_penalty = -(state.main_queue_length + state.ramp_queue_length) * self.queue_penalty_weight

            # 车辆离开奖励
            departure_reward = 0.0
            if hasattr(state, 'cv_vehicles_main') and hasattr(state, 'cv_vehicles_ramp'):
                # 如果控制区有CV车辆，说明流量在维持
                if len(state.cv_vehicles_main) > 0 or len(state.cv_vehicles_ramp) > 0:
                    departure_reward = self.departure_weight

            # 总奖励
            total_reward = ocr_trend_reward + speed_reward + queue_penalty + departure_reward

            rewards[junc_id] = total_reward

            # 记录奖励详情
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'ocr_trend_reward': ocr_trend_reward,
                    'speed_reward': speed_reward,
                    'queue_penalty': queue_penalty,
                    'departure_reward': departure_reward,
                    'total': total_reward
                }

        return rewards

    def reset(self):
        """重置追踪状态"""
        self.ocr_history.clear()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("OCR增量奖励函数测试")
    print("=" * 70)

    # 模拟测试
    calculator = OCRIncrementRewardCalculator(reward_interval=100)

    # 模拟环境统计
    class MockState:
        def __init__(self, main_speed=10.0, ramp_speed=5.0, main_queue=2, ramp_queue=1):
            self.main_speed = main_speed
            self.ramp_speed = ramp_speed
            self.main_queue_length = main_queue
            self.ramp_queue_length = ramp_queue

    class MockAgent:
        def __init__(self):
            self.current_state = MockState()
            self.reward_breakdown = {}

    agents = {'J5': MockAgent()}

    # 模拟OCR从0.90提升到0.95
    print("\n测试OCR改善:")
    for step in range(0, 300, 50):
        ocr = 0.90 + step * 0.0002  # OCR逐渐改善
        env_stats = {'ocr': ocr, 'step': step}
        rewards = calculator.compute_rewards(agents, env_stats)
        print(f"Step {step}: OCR={ocr:.4f}, Reward={rewards['J5']:.4f}")

    calculator.reset()
    agents = {'J5': MockAgent()}

    # 模拟OCR从0.95下降到0.90
    print("\n测试OCR退化:")
    for step in range(0, 300, 50):
        ocr = 0.95 - step * 0.0002  # OCR逐渐退化
        env_stats = {'ocr': ocr, 'step': step}
        rewards = calculator.compute_rewards(agents, env_stats)
        print(f"Step {step}: OCR={ocr:.4f}, Reward={rewards['J5']:.4f}")

    print("\n✅ 测试完成")
