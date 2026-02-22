"""
混合控制器 - 结合规则和神经网络

方案：规则保证稳定性 + 神经网络优化OCR
- 规则控制：稳定性优化（低|a|avg）
- 神经网络：OCR优化（高通行效率）
- 融合策略：优先稳定性，兼顾效率
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

try:
    import traci
except ImportError:
    pass


class RuleBasedController:
    """
    基于规则的控制器（稳定性优化）

    核心思想：
    1. 只在必要时干预（接近边末尾50m）
    2. 温和减速（使用slowDown而非直接setSpeed）
    3. 不强制加速（避免急加速）
    """

    # 接近边末尾的距离阈值
    APPROACH_DIST = 50.0  # 米

    def __init__(self):
        self.controlled_lanes = {}  # 记录受控车道

    def get_action(self, vehicle_ids: List[str], lane_position: float, lane_length: float,
                   current_speed: float) -> Optional[float]:
        """
        获取规则控制动作

        Args:
            vehicle_ids: 受控车辆ID列表
            lane_position: 当前车道位置
            lane_length: 车道长度
            current_speed: 当前速度

        Returns:
            目标速度（如果需要干预），否则返回None
        """
        if not vehicle_ids:
            return None

        # 只控制第一辆车
        veh_id = vehicle_ids[0]

        # 1. 检查是否接近边末尾
        dist_to_end = lane_length - lane_position

        if dist_to_end > self.APPROACH_DIST:
            # 还没接近末尾，不干预
            return None

        # 2. 获取下游速度
        min_ds_speed = self._get_downstream_speed(veh_id)

        if min_ds_speed is None:
            return None

        # 3. 计算目标速度（温和减速）
        # 只在当前速度过快时干预
        if current_speed > min_ds_speed * 1.5:
            target_speed = max(min_ds_speed * 1.5, 3.0)
            return target_speed

        return None

    def _get_downstream_speed(self, veh_id: str) -> Optional[float]:
        """获取下游最小速度"""
        try:
            # 获取车辆路线
            route = traci.vehicle.getRoute(veh_id)
            if len(route) < 2:
                return None

            next_edge = route[1]

            # 获取下游边上的车辆
            next_vehicles = traci.edge.getLastStepVehicleIDs(next_edge)

            if not next_vehicles:
                return None

            # 计算下游平均速度
            speeds = [traci.vehicle.getSpeed(v) for v in next_vehicles]
            min_speed = min(speeds) if speeds else None

            return min_speed

        except Exception as e:
            return None

    def apply_action(self, veh_id: str, target_speed: float, duration: float = 3.0):
        """
        应用规则动作（使用slowDown温和减速）

        Args:
            veh_id: 车辆ID
            target_speed: 目标速度
            duration: 减速持续时间（秒）
        """
        try:
            traci.vehicle.slowDown(veh_id, target_speed, duration)
        except Exception as e:
            pass


class HybridController:
    """
    混合控制器 - 规则 + 神经网络

    融合策略：
    1. 神经网络给出初步动作
    2. 规则控制器给出稳定性建议
    3. 融合：优先使用更保守（更慢）的速度

    目标：在保证稳定性的前提下优化OCR
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.rule_controller = RuleBasedController()

        # 融合权重
        self.rule_priority = 0.6  # 规则优先级（稳定性优先）
        self.neural_priority = 0.4  # 神经网络权重

        # 统计
        self.rule_interventions = 0
        self.neural_controls = 0
        self.hybrid_controls = 0

    def get_control_actions(self, agents: Dict, obs_tensors: Dict,
                           vehicle_obs: Dict) -> Dict[str, Dict[str, float]]:
        """
        获取混合控制动作

        Args:
            agents: 路口智能体字典
            obs_tensors: 观察张量
            vehicle_obs: 车辆观察

        Returns:
            动作字典 {junction_id: {vehicle_id: speed}}
        """
        # 1. 神经网络控制
        with torch.no_grad():
            neural_actions, _, _ = self.model(obs_tensors, vehicle_obs, deterministic=False)

        # 2. 融合规则和神经网络
        action_dict = {}

        for junc_id, agent in agents.items():
            action_dict[junc_id] = {}
            controlled = agent.get_controlled_vehicles()

            # 主路控制
            if controlled['main']:
                for veh_id in controlled['main'][:1]:
                    neural_action = neural_actions.get(junc_id, {}).get('main', 0.5)
                    rule_action = self._get_rule_action(agent, 'main', veh_id)

                    final_action = self._merge_actions(neural_action, rule_action)
                    action_dict[junc_id][veh_id] = final_action

            # 匝道控制
            if controlled['ramp']:
                for veh_id in controlled['ramp'][:1]:
                    neural_action = neural_actions.get(junc_id, {}).get('ramp', 0.5)
                    rule_action = self._get_rule_action(agent, 'ramp', veh_id)

                    final_action = self._merge_actions(neural_action, rule_action)
                    action_dict[junc_id][veh_id] = final_action

        return action_dict

    def _get_rule_action(self, agent, control_type: str, veh_id: str) -> Optional[float]:
        """获取规则控制动作"""
        try:
            # 获取车辆信息
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_position = traci.vehicle.getLanePosition(veh_id)
            lane_length = traci.lane.getLength(lane_id)
            current_speed = traci.vehicle.getSpeed(veh_id)

            # 获取车辆列表
            if control_type == 'main':
                vehicle_ids = [veh_id]  # 已经是单辆控制
            else:
                vehicle_ids = [veh_id]

            # 获取规则建议
            rule_speed = self.rule_controller.get_action(
                vehicle_ids, lane_position, lane_length, current_speed
            )

            if rule_speed is not None:
                self.rule_interventions += 1
                # 转换为[0,1]范围的动作
                speed_limit = 13.89
                normalized_action = (rule_speed / speed_limit - 0.3) / 0.9
                return np.clip(normalized_action, 0.0, 1.0)

            return None

        except Exception as e:
            return None

    def _merge_actions(self, neural_action: float, rule_action: Optional[float]) -> float:
        """
        融合规则和神经网络动作

        策略：优先稳定性
        - 如果规则建议干预（rule_action不为None），使用更保守的速度
        - 否则使用神经网络动作
        """
        if rule_action is not None:
            # 规则干预：取最小值（最保守）
            self.hybrid_controls += 1
            return min(neural_action, rule_action)
        else:
            # 无规则干预：使用神经网络
            self.neural_controls += 1
            return neural_action

    def get_statistics(self) -> Dict:
        """获取控制统计"""
        total = self.rule_interventions + self.neural_controls
        return {
            'rule_interventions': self.rule_interventions,
            'neural_controls': self.neural_controls,
            'hybrid_controls': self.hybrid_controls,
            'rule_ratio': self.rule_interventions / max(total, 1),
            'neural_ratio': self.neural_controls / max(total, 1)
        }

    def reset_statistics(self):
        """重置统计"""
        self.rule_interventions = 0
        self.neural_controls = 0
        self.hybrid_controls = 0


def create_hybrid_controller(model_path: str, device='cuda') -> HybridController:
    """
    创建混合控制器

    Args:
        model_path: 模型路径
        device: 设备

    Returns:
        混合控制器实例
    """
    from junction_network import create_junction_model, JUNCTION_CONFIGS

    # 加载模型
    model = create_junction_model(JUNCTION_CONFIGS)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 创建混合控制器
    controller = HybridController(model, device)

    return controller
