"""
改进版奖励函数 - 包含正向奖励
解决"只有惩罚没有奖励"的问题
"""

import numpy as np
from typing import Dict
from collections import defaultdict


class ImprovedRewardCalculator:
    """
    改进的奖励计算器
    包含正向奖励和负向惩罚
    """
    
    def __init__(self):
        # 历史状态追踪
        self.previous_ocr = defaultdict(float)
        self.previous_throughput = defaultdict(int)
        self.merge_success_count = defaultdict(int)
        self.previous_in_zone = defaultdict(int)  # 追踪控制区内车辆数

        # ===== 重新设计：专注于实时可观测指标的奖励权重 =====
        self.weights = {
            # ===== 正向奖励（实时有效）=====
            'vehicle_departure': 0.3,        # 车辆离开奖励（每辆离开控制区）
            'flow_maintenance': 1.0,         # 流量保持奖励（主路和匝道都有流动）
            'speed_maintenance': 0.5,        # 速度维持奖励（保持较高速度）
            'gap_utilization': 1.5,          # 间隙利用奖励（匝道车辆成功利用间隙）
            'no_stops': 0.3,                 # 无停车奖励（车辆不停车）
            'capacity_bonus': 1.0,           # 通行能力奖励（多车辆同时通过）

            # ===== 期末奖励（episode结束时）=====
            'ocr_final': 100.0,              # 最终OCR奖励（episode结束时一次性）
            'throughput_bonus': 5.0,         # 总通过量奖励

            # ===== 负向惩罚（保持较小权重）=====
            'queue_penalty': 0.005,          # 排队惩罚
            'waiting_penalty': 0.002,        # 等待惩罚
            'conflict_penalty': 0.02,        # 冲突惩罚
            'speed_variance': 0.001,         # 速度方差惩罚
        }

        # 奖励裁剪范围
        self.reward_clip_range = (-10.0, 10.0)  # 扩大裁剪范围，提供更多学习空间

        # Episode追踪
        self.episode_throughput = defaultdict(int)  # 累计通过量
        self.is_final_step = False  # 标记是否是最后一步
    
    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """
        计算奖励（专注于实时可观测指标）

        Args:
            agents: 路口智能体字典
            env_stats: 环境统计信息

        Returns:
            每个路口的奖励字典
        """
        # 检查是否是episode的最后一步
        current_step = env_stats.get('step', 0)
        self.is_final_step = (current_step >= 3600)  # 假设3600步为一个episode

        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # ===== 正向奖励（实时有效）=====

            # 1. 车辆离开奖励：基于控制区内车辆数的变化
            current_in_zone = len(state.main_vehicles) + len(state.ramp_vehicles)
            # 如果车辆数减少，说明有车辆离开控制区
            departed_delta = max(0, self.previous_in_zone[junc_id] - current_in_zone)
            self.previous_in_zone[junc_id] = current_in_zone

            # 车辆离开奖励（离开的车辆数 × 奖励权重）
            departure_reward = departed_delta * self.weights['vehicle_departure']

            # 累计通过量
            self.episode_throughput[junc_id] += departed_delta

            # 2. 流量保持奖励：主路和匝道都有车辆流动
            main_flow = len(state.main_vehicles) > 0 and state.main_speed > 1.0
            ramp_flow = len(state.ramp_vehicles) > 0 and state.ramp_speed > 1.0
            flow_reward = 0.0
            if main_flow and ramp_flow:
                flow_reward = self.weights['flow_maintenance']
            elif main_flow or ramp_flow:
                flow_reward = self.weights['flow_maintenance'] * 0.5

            # 3. 速度维持奖励
            speed_reward = 0.0
            if state.main_speed > 5.0:  # 主路速度大于5m/s
                speed_reward += self.weights['speed_maintenance']
            if state.ramp_speed > 3.0:  # 匝道速度大于3m/s
                speed_reward += self.weights['speed_maintenance'] * 0.5

            # 4. 间隙利用奖励
            gap_reward = self._compute_gap_reward(state)

            # 5. 无停车奖励
            no_stops_reward = 0.0
            total_vehicles = len(state.main_vehicles) + len(state.ramp_vehicles)
            if total_vehicles > 0:
                # 检查是否有停车
                moving_vehicles = 0
                for veh in state.main_vehicles + state.ramp_vehicles:
                    if veh.get('speed', 0) > 0.1:
                        moving_vehicles += 1
                moving_ratio = moving_vehicles / total_vehicles
                no_stops_reward = moving_ratio * self.weights['no_stops']

            # 6. 通行能力奖励：多车辆同时通过
            capacity_reward = 0.0
            if total_vehicles >= 5:  # 控制区有5辆以上车
                capacity_reward = min(total_vehicles / 10.0, 1.0) * self.weights['capacity_bonus']

            # ===== 负向惩罚 =====

            # 1. 排队惩罚
            queue_penalty = -(
                state.main_queue_length * self.weights['queue_penalty'] +
                state.ramp_queue_length * self.weights['queue_penalty'] * 2
            )

            # 2. 等待惩罚
            waiting_penalty = -state.ramp_waiting_time * self.weights['waiting_penalty']

            # 3. 冲突风险惩罚
            conflict_penalty = -state.conflict_risk * self.weights['conflict_penalty']

            # 4. 速度方差惩罚
            speed_variance_penalty = -abs(state.main_speed - state.ramp_speed) * self.weights['speed_variance']

            # ===== 总奖励 =====
            total_reward = (
                # 正向奖励
                departure_reward +
                flow_reward +
                speed_reward +
                gap_reward +
                no_stops_reward +
                capacity_reward +

                # 负向惩罚
                queue_penalty +
                waiting_penalty +
                conflict_penalty +
                speed_variance_penalty
            )

            # ===== 期末奖励（只在最后一步）=====
            if self.is_final_step:
                # 最终OCR奖励
                final_ocr = env_stats.get('ocr', 0.0)
                ocr_bonus = final_ocr * self.weights['ocr_final']

                # 总通过量奖励
                throughput_bonus = (self.episode_throughput[junc_id] / 100.0) * self.weights['throughput_bonus']
                throughput_bonus = min(throughput_bonus, 10.0)  # 上限

                total_reward += ocr_bonus + throughput_bonus

            # ===== 生存奖励 =====
            survival_bonus = 0.05  # 小的生存奖励
            total_reward += survival_bonus

            # ===== 奖励裁剪 =====
            total_reward = max(self.reward_clip_range[0], min(self.reward_clip_range[1], total_reward))

            rewards[junc_id] = total_reward

            # 记录详细奖励（用于调试）
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'departure_reward': departure_reward,
                    'flow_reward': flow_reward,
                    'speed_reward': speed_reward,
                    'gap_reward': gap_reward,
                    'no_stops_reward': no_stops_reward,
                    'capacity_reward': capacity_reward,
                    'queue_penalty': queue_penalty,
                    'waiting_penalty': waiting_penalty,
                    'conflict_penalty': conflict_penalty,
                    'survival_bonus': survival_bonus,
                    'total': total_reward
                }

        return rewards

    def _compute_gap_reward(self, state) -> float:
        """计算间隙利用奖励"""
        if not state.ramp_vehicles:
            return 0.0

        # 有匝道车辆且有可接受间隙
        if state.gap_acceptance > 0.5:
            return state.gap_acceptance * self.weights['gap_utilization']

        return 0.0

    def reset(self):
        """重置追踪状态（新episode开始时调用）"""
        self.previous_ocr.clear()
        self.previous_throughput.clear()
        self.merge_success_count.clear()
        self.previous_in_zone.clear()
        self.episode_throughput.clear()
        self.is_final_step = False
    
