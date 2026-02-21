"""
改进版奖励函数 - 基于评分公式
硬编码基准OCR并使用初赛评分公式作为奖励
"""

import numpy as np
from typing import Dict
from collections import defaultdict


class ImprovedRewardCalculator:
    """
    改进的奖励计算器
    基于初赛评分公式计算奖励
    """

    # 硬编码基准OCR (从评测结果反推)
    BASELINE_OCR = 0.8812

    def __init__(self):
        # 历史状态追踪
        self.previous_ocr = defaultdict(float)
        self.previous_throughput = defaultdict(int)
        self.merge_success_count = defaultdict(int)
        self.previous_in_zone = defaultdict(int)  # 追踪控制区内车辆数

        # ===== 重新设计：专注于实时可观测指标的奖励权重 =====
        self.weights = {
            # ===== 正向奖励（大幅增加，强化积极行为）=====
            'vehicle_departure': 1.0,        # 车辆离开奖励（从0.3增加到1.0）
            'flow_maintenance': 3.0,         # 流量保持奖励（从1.0增加到3.0）
            'speed_maintenance': 1.5,        # 速度维持奖励（从0.5增加到1.5）
            'gap_utilization': 3.0,          # 间隙利用奖励（从1.5增加到3.0）
            'no_stops': 1.0,                 # 无停车奖励（从0.3增加到1.0）
            'capacity_bonus': 2.0,           # 通行能力奖励（从1.0增加到2.0）

            # ===== 期末奖励（episode结束时）=====
            'ocr_final': 100.0,              # 最终OCR奖励（episode结束时一次性）
            'throughput_bonus': 5.0,         # 总通过量奖励

            # ===== 负向惩罚（进一步减小，避免累积效应）=====
            'queue_penalty': 0.002,          # 排队惩罚（从0.005减少到0.002）
            'waiting_penalty': 0.001,        # 等待惩罚（从0.002减少到0.001）
            'conflict_penalty': 0.01,        # 冲突惩罚（从0.02减少到0.01）
            'speed_variance': 0.0005,        # 速度方差惩罚（从0.001减少到0.0005）
        }

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

            # 3. 速度维持奖励（更细致的分层）
            speed_reward = 0.0

            # 主路速度奖励：4层精细分级
            if state.main_speed > 12.0:  # 43.2km/h - 优秀
                speed_reward += self.weights['speed_maintenance'] * 1.2
            elif state.main_speed > 10.0:  # 36km/h - 良好
                speed_reward += self.weights['speed_maintenance']
            elif state.main_speed > 8.0:  # 28.8km/h - 中等
                speed_reward += self.weights['speed_maintenance'] * 0.7
            elif state.main_speed > 5.0:  # 18km/h - 基础
                speed_reward += self.weights['speed_maintenance'] * 0.3

            # 匝道速度奖励：4层精细分级
            if state.ramp_speed > 10.0:  # 36km/h - 优秀
                speed_reward += self.weights['speed_maintenance'] * 0.7
            elif state.ramp_speed > 8.0:  # 28.8km/h - 良好
                speed_reward += self.weights['speed_maintenance'] * 0.5
            elif state.ramp_speed > 5.0:  # 18km/h - 中等
                speed_reward += self.weights['speed_maintenance'] * 0.3
            elif state.ramp_speed > 3.0:  # 10.8km/h - 基础
                speed_reward += self.weights['speed_maintenance'] * 0.15

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
                # 使用官方评分公式计算奖励
                # 公式: S_efficiency = 100 × max(0, ΔOCR)
                # 其中: ΔOCR = (OCR_AI - OCR_Base) / OCR_Base
                current_ocr = env_stats.get('ocr', 0.0)
                delta_ocr = (current_ocr - self.BASELINE_OCR) / self.BASELINE_OCR if self.BASELINE_OCR > 0 else 0
                score = 100 * max(0, delta_ocr)

                # 使用官方得分作为最终奖励
                total_reward += score

            # ===== 生存奖励 =====
            survival_bonus = 0.1  # 增加生存奖励（从0.05提高到0.1）
            total_reward += survival_bonus

            # ===== 奖励裁剪（已移除）=====
            # total_reward = max(self.reward_clip_range[0], min(self.reward_clip_range[1], total_reward))

            rewards[junc_id] = total_reward

            # 记录详细奖励（用于调试）
            if hasattr(agent, 'reward_breakdown'):
                breakdown = {
                    'departure_reward': departure_reward,
                    'flow_reward': flow_reward,
                    'speed_reward': speed_reward,
                    'gap_reward': gap_reward,
                    'no_stops_reward': no_stops_reward,
                    'capacity_reward': capacity_reward,
                    'queue_penalty': queue_penalty,
                    'waiting_penalty': waiting_penalty,
                    'conflict_penalty': conflict_penalty,
                    'speed_variance_penalty': speed_variance_penalty,
                    'survival_bonus': survival_bonus,
                    'total': total_reward
                }

                # 如果是最后一步，添加评分信息
                if self.is_final_step:
                    current_ocr = env_stats.get('ocr', 0.0)
                    delta_ocr = (current_ocr - self.BASELINE_OCR) / self.BASELINE_OCR if self.BASELINE_OCR > 0 else 0
                    score = 100 * max(0, delta_ocr)
                    breakdown.update({
                        'final_ocr': current_ocr,
                        'baseline_ocr': self.BASELINE_OCR,
                        'delta_ocr': delta_ocr,
                        'final_score': score
                    })

                agent.reward_breakdown = breakdown

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
    
