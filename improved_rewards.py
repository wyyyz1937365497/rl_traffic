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
        
        # 奖励权重
        self.weights = {
            # 正向奖励权重
            'ocr_improvement': 10.0,      # OCR提升奖励（最重要）
            'merge_success': 0.5,          # 成功汇入奖励
            'gap_utilization': 0.3,        # 间隙利用奖励
            'speed_coordination': 0.2,     # 速度协调奖励
            'no_conflict': 0.1,            # 无冲突奖励
            'throughput': 0.1,             # 通过量奖励
            
            # 负向惩罚权重
            'queue_penalty': 0.1,          # 排队惩罚
            'waiting_penalty': 0.05,       # 等待惩罚
            'conflict_penalty': 0.5,       # 冲突惩罚
            'speed_variance': 0.02,        # 速度方差惩罚
        }
    
    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """
        计算奖励（包含正向奖励）
        
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
            
            # 1. OCR提升奖励（核心指标）
            current_ocr = env_stats.get('ocr', 0.0)
            ocr_improvement = current_ocr - self.previous_ocr[junc_id]
            ocr_reward = max(0, ocr_improvement) * self.weights['ocr_improvement']
            self.previous_ocr[junc_id] = current_ocr
            
            # 2. 成功汇入奖励
            merge_reward = self._compute_merge_reward(state, junc_id)
            
            # 3. 间隙利用奖励
            gap_reward = self._compute_gap_reward(state)
            
            # 4. 速度协调奖励
            speed_reward = self._compute_speed_reward(state)
            
            # 5. 无冲突奖励
            no_conflict_reward = self._compute_no_conflict_reward(state)
            
            # 6. 通过量奖励
            throughput_reward = self._compute_throughput_reward(state, junc_id)
            
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
            
            # ===== 信号灯协调奖励 =====
            signal_reward = self._compute_signal_reward(state)
            
            # ===== 总奖励 =====
            total_reward = (
                # 正向奖励
                ocr_reward +
                merge_reward +
                gap_reward +
                speed_reward +
                no_conflict_reward +
                throughput_reward +
                
                # 负向惩罚
                queue_penalty +
                waiting_penalty +
                conflict_penalty +
                speed_variance_penalty +
                
                # 信号灯奖励
                signal_reward
            )
            
            rewards[junc_id] = total_reward
            
            # 记录详细奖励（用于调试）
            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'ocr_reward': ocr_reward,
                    'merge_reward': merge_reward,
                    'gap_reward': gap_reward,
                    'speed_reward': speed_reward,
                    'no_conflict_reward': no_conflict_reward,
                    'throughput_reward': throughput_reward,
                    'queue_penalty': queue_penalty,
                    'waiting_penalty': waiting_penalty,
                    'conflict_penalty': conflict_penalty,
                    'signal_reward': signal_reward,
                    'total': total_reward
                }
        
        return rewards
    
    def _compute_merge_reward(self, state, junc_id: str) -> float:
        """计算成功汇入奖励"""
        # 检查是否有车辆成功汇入
        if state.ramp_signal == 'G':
            # 匝道绿灯时，检查是否有车辆汇入
            if hasattr(state, 'merged_count'):
                merge_count = state.merged_count
                self.merge_success_count[junc_id] += merge_count
                return merge_count * self.weights['merge_success']
        
        return 0.0
    
    def _compute_gap_reward(self, state) -> float:
        """计算间隙利用奖励"""
        if not state.ramp_vehicles:
            return 0.0
        
        # 有匝道车辆且有可接受间隙
        if state.gap_acceptance > 0.5:
            return state.gap_acceptance * self.weights['gap_utilization']
        
        return 0.0
    
    def _compute_speed_reward(self, state) -> float:
        """计算速度协调奖励"""
        if not state.main_vehicles or not state.ramp_vehicles:
            return 0.0
        
        # 主路和匝道速度接近时给予奖励
        speed_diff = abs(state.main_speed - state.ramp_speed)
        if speed_diff < 2.0:  # 速度差小于2m/s
            return self.weights['speed_coordination']
        
        return 0.0
    
    def _compute_no_conflict_reward(self, state) -> float:
        """计算无冲突奖励"""
        if state.conflict_risk < 0.1:
            return self.weights['no_conflict']
        return 0.0
    
    def _compute_throughput_reward(self, state, junc_id: str) -> float:
        """计算通过量奖励"""
        current_throughput = (
            len(state.main_vehicles) + 
            len(state.ramp_vehicles)
        )
        
        throughput_diff = current_throughput - self.previous_throughput[junc_id]
        self.previous_throughput[junc_id] = current_throughput
        
        # 通过量增加给予奖励
        if throughput_diff > 0:
            return throughput_diff * self.weights['throughput']
        
        return 0.0
    
    def _compute_signal_reward(self, state) -> float:
        """计算信号灯协调奖励"""
        if not hasattr(state, 'ramp_signal'):
            return 0.0
        
        if state.ramp_signal == 'G' and state.ramp_vehicles:
            # 匝道绿灯且有车辆，鼓励汇入
            return 0.1
        elif state.ramp_signal == 'r' and state.ramp_vehicles:
            # 匝道红灯但有车辆，轻微惩罚
            return -0.05 * len(state.ramp_vehicles)
        
        return 0.0
    
    def reset(self):
        """重置追踪状态"""
        self.previous_ocr.clear()
        self.previous_throughput.clear()
        self.merge_success_count.clear()


# ===== 集成到环境中的示例 =====

def compute_rewards_improved(self) -> Dict[str, float]:
    """
    改进的奖励计算函数
    替换原来的 _compute_rewards 方法
    """
    if not hasattr(self, 'reward_calculator'):
        self.reward_calculator = ImprovedRewardCalculator()
    
    # 获取环境统计信息
    env_stats = {
        'ocr': self._compute_current_ocr(),
        'step': self.current_step
    }
    
    return self.reward_calculator.compute_rewards(self.agents, env_stats)


def _compute_current_ocr(self) -> float:
    """计算当前OCR"""
    try:
        import traci
        
        # 到达车辆数
        arrived = traci.simulation.getArrivedNumber()
        
        # 总车辆数
        total = traci.vehicle.getCount()
        
        # 在途车辆完成度
        inroute_completion = 0.0
        for veh_id in traci.vehicle.getIDList():
            try:
                route_idx = traci.vehicle.getRouteIndex(veh_id)
                route_len = len(traci.vehicle.getRoute(veh_id))
                if route_len > 0:
                    inroute_completion += route_idx / route_len
            except:
                continue
        
        # OCR = (到达 + 在途完成度) / 总数
        if total == 0:
            return 0.0
        
        ocr = (arrived + inroute_completion) / total
        return min(ocr, 1.0)
        
    except:
        return 0.0
