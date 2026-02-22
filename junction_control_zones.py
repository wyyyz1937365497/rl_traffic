"""
路口级多智能体系统 - 控制区域划分版本
确保每个路口只控制其上游的CV车辆，避免控制权冲突
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict
import numpy as np

try:
    import traci
except ImportError:
    print("请安装traci: pip install traci")
    sys.exit(1)


class JunctionType(Enum):
    """路口类型"""
    TYPE_A = "type_a"  # 单纯匝道汇入
    TYPE_B = "type_b"  # 匝道汇入 + 主路转出


@dataclass
class ControlZone:
    """
    控制区域定义
    每个路口的控制区域不重叠
    """
    junction_id: str
    
    # 主路上游控制区域（只控制上游车辆）
    main_upstream_edges: List[str] = field(default_factory=list)
    main_upstream_range: float = 200.0  # 控制范围（米）
    
    # 匝道上游控制区域
    ramp_upstream_edges: List[str] = field(default_factory=list)
    ramp_upstream_range: float = 150.0
    
    # 转出引导区域（仅类型B）
    diverge_edges: List[str] = field(default_factory=list)
    diverge_range: float = 100.0
    
    # 排除区域（下游路口的控制区域，本路口不控制）
    excluded_edges: List[str] = field(default_factory=list)
    
    # 当前控制的车辆
    controlled_vehicles: Set[str] = field(default_factory=set)
    
    def get_controlled_vehicles_in_zone(self, all_vehicles: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        获取控制区域内的车辆
        
        Args:
            all_vehicles: 所有车辆信息 {veh_id: {edge, position, ...}}
        
        Returns:
            {'main': [...], 'ramp': [...], 'diverge': [...]}
        """
        main_vehicles = []
        ramp_vehicles = []
        diverge_vehicles = []
        
        for veh_id, veh_info in all_vehicles.items():
            edge = veh_info.get('edge', '')
            position = veh_info.get('lane_position', 0)
            is_cv = veh_info.get('is_cv', False)
            
            if not is_cv:
                continue
            
            # 检查是否在排除区域
            if edge in self.excluded_edges:
                continue
            
            # 检查主路上游区域
            if edge in self.main_upstream_edges:
                # 只控制距离路口一定范围内的车辆
                edge_length = veh_info.get('edge_length', 500)
                distance_to_junction = edge_length - position
                
                if distance_to_junction <= self.main_upstream_range:
                    main_vehicles.append(veh_id)
            
            # 检查匝道上游区域
            elif edge in self.ramp_upstream_edges:
                edge_length = veh_info.get('edge_length', 300)
                distance_to_junction = edge_length - position
                
                if distance_to_junction <= self.ramp_upstream_range:
                    ramp_vehicles.append(veh_id)
            
            # 检查转出区域（仅类型B）
            elif edge in self.diverge_edges:
                if position <= self.diverge_range:
                    diverge_vehicles.append(veh_id)
        
        return {
            'main': main_vehicles,
            'ramp': ramp_vehicles,
            'diverge': diverge_vehicles
        }


# 定义每个路口的控制区域（不重叠）
CONTROL_ZONES = {
    'J5': ControlZone(
        junction_id='J5',
        main_upstream_edges=['E2'],      # J5的主路上游
        main_upstream_range=200.0,
        ramp_upstream_edges=['E23'],     # J5的匝道上游
        ramp_upstream_range=150.0,
        excluded_edges=['E3', 'E9', 'E10', 'E11', 'E12', 'E13']  # 下游路口的控制区域
    ),
    
    'J14': ControlZone(
        junction_id='J14',
        main_upstream_edges=['E9'],      # J14的主路上游（注意：E2已被J5控制）
        main_upstream_range=200.0,
        ramp_upstream_edges=['E15'],     # J14的匝道上游
        ramp_upstream_range=150.0,
        excluded_edges=['E10', 'E11', 'E12', 'E13']  # 下游路口的控制区域
    ),
    
    'J15': ControlZone(
        junction_id='J15',
        main_upstream_edges=['E10'],     # J15的主路上游
        main_upstream_range=200.0,
        ramp_upstream_edges=['E17'],     # J15的匝道上游
        ramp_upstream_range=150.0,
        diverge_edges=['E16'],           # J15的转出匝道
        diverge_range=100.0,
        excluded_edges=['E11', 'E12', 'E13']  # 下游路口的控制区域
    ),
    
    'J17': ControlZone(
        junction_id='J17',
        main_upstream_edges=['E12'],     # J17的主路上游
        main_upstream_range=200.0,
        ramp_upstream_edges=['E19'],     # J17的匝道上游
        ramp_upstream_range=150.0,
        diverge_edges=['E18', 'E20'],    # J17的转出匝道
        diverge_range=100.0,
        excluded_edges=[]  # J17是最下游路口
    )
}


@dataclass
class JunctionConfig:
    """路口配置"""
    junction_id: str
    junction_type: JunctionType
    control_zone: ControlZone

    # 道路配置
    main_incoming: List[str] = field(default_factory=list)
    main_outgoing: List[str] = field(default_factory=list)
    ramp_incoming: List[str] = field(default_factory=list)
    ramp_outgoing: List[str] = field(default_factory=list)
    reverse_incoming: List[str] = field(default_factory=list)  # 反向主路上游（与匝道冲突的方向）
    reverse_outgoing: List[str] = field(default_factory=list)  # 反向主路下游（匝道汇入边）

    # 所有相关边和车道
    all_edges: List[str] = field(default_factory=list)
    all_lanes: List[str] = field(default_factory=list)

    # 车道级冲突信息（新增）
    num_main_lanes: int = 0        # 主路车道数
    num_ramp_lanes: int = 0        # 匝道车道数
    conflict_lanes: List[str] = field(default_factory=list)  # 冲突车道列表

    # 信号灯配置
    has_traffic_light: bool = False
    tl_id: str = ""
    num_phases: int = 2
    phases: List = field(default_factory=list)  # 信号灯相位列表

    def __post_init__(self):
        """初始化所有边和车道"""
        self.all_edges = (
            self.main_incoming + self.main_outgoing +
            self.ramp_incoming + self.ramp_outgoing +
            self.reverse_incoming + self.reverse_outgoing
        )


# 更新路口配置，包含控制区域和车道级冲突信息
# 基于 EDGE_TOPOLOGY 和 LANE_CONFLICTS 更新
JUNCTION_CONFIGS = {
    'J5': JunctionConfig(
        junction_id='J5',
        junction_type=JunctionType.TYPE_A,
        control_zone=CONTROL_ZONES['J5'],
        main_incoming=['E2'],
        main_outgoing=['E3'],
        ramp_incoming=['E23'],
        reverse_incoming=['-E3'],  # 反向主路上游（与匝道冲突的方向）
        reverse_outgoing=['-E2'],  # 反向主路下游（匝道汇入边）
        num_main_lanes=2,
        num_ramp_lanes=1,
        conflict_lanes=['-E3_0'],  # E23_0 与 -E3_0 冲突
        has_traffic_light=True,
        tl_id='J5',
        num_phases=2
    ),

    'J14': JunctionConfig(
        junction_id='J14',
        junction_type=JunctionType.TYPE_A,
        control_zone=CONTROL_ZONES['J14'],
        main_incoming=['E9'],
        main_outgoing=['E10'],
        ramp_incoming=['E15'],
        reverse_incoming=['-E10'],
        reverse_outgoing=['-E9'],
        num_main_lanes=2,
        num_ramp_lanes=1,
        conflict_lanes=['E9_0'],  # E15_0 与 E9_0 冲突（注意是E9不是-E9）
        has_traffic_light=True,
        tl_id='J14',
        num_phases=2
    ),

    'J15': JunctionConfig(
        junction_id='J15',
        junction_type=JunctionType.TYPE_B,
        control_zone=CONTROL_ZONES['J15'],
        main_incoming=['E10'],
        main_outgoing=['E11'],
        ramp_incoming=['E17'],
        ramp_outgoing=['E16'],
        reverse_incoming=['-E11'],  # 反向主路上游（与匝道冲突的方向）
        reverse_outgoing=['-E10'],  # 反向主路下游（匝道汇入边）
        num_main_lanes=3,
        num_ramp_lanes=1,
        conflict_lanes=['-E11_0', '-E11_1'],  # E17_0 只与前2条车道冲突
        has_traffic_light=True,
        tl_id='J15',
        num_phases=2
    ),

    'J17': JunctionConfig(
        junction_id='J17',
        junction_type=JunctionType.TYPE_B,
        control_zone=CONTROL_ZONES['J17'],
        main_incoming=['E12'],
        main_outgoing=['E13'],
        ramp_incoming=['E19'],
        ramp_outgoing=['E18', 'E20'],
        reverse_incoming=['-E13'],  # 反向主路上游（与匝道冲突的方向）
        reverse_outgoing=['-E12'],  # 反向主路下游（匝道汇入边）
        num_main_lanes=3,
        num_ramp_lanes=2,
        conflict_lanes=['-E13_0', '-E13_1'],  # E19_0, E19_1 只与前2条车道冲突
        has_traffic_light=True,
        tl_id='J17',
        num_phases=2
    )
}


class VehicleRegistry:
    """
    车辆注册表
    跟踪每辆车的控制权归属，确保一辆车只被一个路口控制
    """
    
    def __init__(self):
        # 车辆 -> 控制路口的映射
        self.vehicle_to_junction: Dict[str, str] = {}
        
        # 路口 -> 控制车辆的映射
        self.junction_to_vehicles: Dict[str, Set[str]] = defaultdict(set)
        
        # 车辆位置缓存
        self.vehicle_positions: Dict[str, Dict] = {}
    
    def update(self, all_vehicles: Dict[str, Dict]):
        """
        更新车辆注册表
        
        Args:
            all_vehicles: 所有车辆信息
        """
        # 清理已离开的车辆
        current_vehicles = set(all_vehicles.keys())
        left_vehicles = set(self.vehicle_to_junction.keys()) - current_vehicles
        
        for veh_id in left_vehicles:
            old_junction = self.vehicle_to_junction.get(veh_id)
            if old_junction:
                self.junction_to_vehicles[old_junction].discard(veh_id)
            del self.vehicle_to_junction[veh_id]
            if veh_id in self.vehicle_positions:
                del self.vehicle_positions[veh_id]
        
        # 更新车辆位置
        self.vehicle_positions = all_vehicles.copy()
        
        # 为每辆车分配控制路口
        for veh_id, veh_info in all_vehicles.items():
            if not veh_info.get('is_cv', False):
                continue
            
            # 如果车辆已有控制路口，检查是否需要转移
            current_junction = self.vehicle_to_junction.get(veh_id)
            
            # 根据车辆位置确定应该由哪个路口控制
            new_junction = self._assign_junction(veh_id, veh_info)
            
            if new_junction != current_junction:
                # 转移控制权
                if current_junction:
                    self.junction_to_vehicles[current_junction].discard(veh_id)
                
                if new_junction:
                    self.vehicle_to_junction[veh_id] = new_junction
                    self.junction_to_vehicles[new_junction].add(veh_id)
    
    def _assign_junction(self, veh_id: str, veh_info: Dict) -> Optional[str]:
        """
        为车辆分配控制路口
        
        原则：
        1. 车辆在上游路口的控制区域内 -> 由上游路口控制
        2. 车辆离开上游区域，进入下游区域 -> 转移到下游路口
        3. 车辆不在任何控制区域 -> 不控制
        """
        edge = veh_info.get('edge', '')
        position = veh_info.get('lane_position', 0)
        
        # 检查每个路口的控制区域
        for junc_id, zone in CONTROL_ZONES.items():
            # 检查主路上游区域
            if edge in zone.main_upstream_edges:
                edge_length = veh_info.get('edge_length', 500)
                distance_to_junction = edge_length - position
                
                if distance_to_junction <= zone.main_upstream_range:
                    return junc_id
            
            # 检查匝道上游区域
            if edge in zone.ramp_upstream_edges:
                edge_length = veh_info.get('edge_length', 300)
                distance_to_junction = edge_length - position
                
                if distance_to_junction <= zone.ramp_upstream_range:
                    return junc_id
            
            # 检查转出区域
            if edge in zone.diverge_edges:
                if position <= zone.diverge_range:
                    return junc_id
        
        return None
    
    def get_controlled_vehicles(self, junction_id: str) -> Set[str]:
        """获取指定路口控制的车辆"""
        return self.junction_to_vehicles.get(junction_id, set())
    
    def get_controlling_junction(self, veh_id: str) -> Optional[str]:
        """获取控制指定车辆的路口"""
        return self.vehicle_to_junction.get(veh_id)


class JunctionAgentWithZone:
    """
    路口智能体 - 带控制区域
    只控制分配给自己的车辆
    """
    
    def __init__(self, config: JunctionConfig, vehicle_registry: VehicleRegistry):
        self.config = config
        self.junction_id = config.junction_id
        self.junction_type = config.junction_type
        self.control_zone = config.control_zone
        self.vehicle_registry = vehicle_registry
        
        # 当前控制的车辆
        self.current_controlled_vehicles: Dict[str, List[str]] = {
            'main': [],
            'ramp': [],
            'diverge': []
        }
    
    def update_controlled_vehicles(self, all_vehicles: Dict[str, Dict]):
        """
        更新控制的车辆列表
        
        Args:
            all_vehicles: 所有车辆信息
        """
        # 从注册表获取本路口控制的车辆
        my_vehicles = self.vehicle_registry.get_controlled_vehicles(self.junction_id)
        
        # 按位置分类
        main_vehicles = []
        ramp_vehicles = []
        diverge_vehicles = []
        
        for veh_id in my_vehicles:
            veh_info = all_vehicles.get(veh_id, {})
            edge = veh_info.get('edge', '')
            
            if edge in self.control_zone.main_upstream_edges:
                main_vehicles.append(veh_id)
            elif edge in self.control_zone.ramp_upstream_edges:
                ramp_vehicles.append(veh_id)
            elif edge in self.control_zone.diverge_edges:
                diverge_vehicles.append(veh_id)
        
        self.current_controlled_vehicles = {
            'main': main_vehicles[:5],      # 最多控制5辆主路车
            'ramp': ramp_vehicles[:3],      # 最多控制3辆匝道车
            'diverge': diverge_vehicles[:2]  # 最多控制2辆转出车
        }
    
    def get_controlled_vehicles(self) -> Dict[str, List[str]]:
        """获取当前控制的车辆"""
        return self.current_controlled_vehicles
    
    def get_state_dim(self) -> int:
        """状态维度"""
        return 22
    
    def get_action_dim(self) -> int:
        """动作维度"""
        if self.junction_type == JunctionType.TYPE_A:
            return 3
        else:
            return 4


class MultiAgentEnvironmentWithZones:
    """
    多智能体环境 - 带控制区域划分
    确保每个路口只控制自己的车辆
    """
    
    def __init__(self, junction_ids: List[str] = None, sumo_cfg: str = None,
                 use_gui: bool = False, seed: int = None):
        self.junction_ids = junction_ids or list(JUNCTION_CONFIGS.keys())
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.seed = seed
        
        # 创建车辆注册表
        self.vehicle_registry = VehicleRegistry()
        
        # 创建路口智能体
        self.agents: Dict[str, JunctionAgentWithZone] = {}
        for junc_id in self.junction_ids:
            if junc_id in JUNCTION_CONFIGS:
                self.agents[junc_id] = JunctionAgentWithZone(
                    JUNCTION_CONFIGS[junc_id],
                    self.vehicle_registry
                )
        
        # 仿真状态
        self.current_step = 0
        self.is_running = False
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self._start_sumo()
        self.current_step = 0
        
        # 执行几步让车辆进入
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
        
        # 更新车辆注册表
        all_vehicles = self._get_all_vehicles()
        self.vehicle_registry.update(all_vehicles)
        
        # 更新每个智能体的控制车辆
        for agent in self.agents.values():
            agent.update_controlled_vehicles(all_vehicles)
        
        # 返回初始观察
        observations = {}
        for junc_id, agent in self.agents.items():
            observations[junc_id] = np.zeros(agent.get_state_dim(), dtype=np.float32)
        
        return observations
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        执行一步
        
        Args:
            actions: {路口ID: {车辆ID: 动作值}}
        
        Returns:
            observations, rewards, done, info
        """
        # 验证并应用动作（确保只控制自己的车辆）
        self._apply_actions_with_validation(actions)
        
        # 执行仿真步
        traci.simulationStep()
        self.current_step += 1
        
        # 更新车辆注册表
        all_vehicles = self._get_all_vehicles()
        self.vehicle_registry.update(all_vehicles)
        
        # 更新每个智能体的控制车辆
        for agent in self.agents.values():
            agent.update_controlled_vehicles(all_vehicles)
        
        # 计算奖励
        rewards = self._compute_rewards()
        
        # 检查是否结束
        done = self._is_done()
        
        # 返回观察
        observations = {}
        for junc_id, agent in self.agents.items():
            observations[junc_id] = np.zeros(agent.get_state_dim(), dtype=np.float32)
        
        info = {
            'step': self.current_step,
            'controlled_vehicles': {
                junc_id: agent.get_controlled_vehicles() 
                for junc_id, agent in self.agents.items()
            }
        }
        
        return observations, rewards, done, info
    
    def _apply_actions_with_validation(self, actions: Dict[str, Dict]):
        """
        应用动作并验证控制权
        
        确保每个路口只控制自己的车辆
        """
        for junc_id, action_dict in actions.items():
            agent = self.agents.get(junc_id)
            if agent is None:
                continue
            
            # 获取该路口控制的车辆
            controlled = agent.get_controlled_vehicles()
            all_controlled = set(controlled['main'] + controlled['ramp'] + controlled['diverge'])
            
            # 只对控制的车辆应用动作
            for veh_id, action in action_dict.items():
                # 验证控制权
                if veh_id not in all_controlled:
                    print(f"警告: {junc_id} 试图控制未授权的车辆 {veh_id}")
                    continue
                
                # 验证注册表
                controlling_junction = self.vehicle_registry.get_controlling_junction(veh_id)
                if controlling_junction != junc_id:
                    print(f"警告: 车辆 {veh_id} 当前由 {controlling_junction} 控制，而非 {junc_id}")
                    continue
                
                # 应用动作
                try:
                    speed_limit = 13.89
                    target_speed = speed_limit * (0.3 + 0.9 * action)
                    traci.vehicle.setSpeed(veh_id, target_speed)
                except Exception as e:
                    print(f"应用动作失败: {e}")
    
    def _get_all_vehicles(self) -> Dict[str, Dict]:
        """获取所有车辆信息"""
        all_vehicles = {}
        
        try:
            veh_ids = traci.vehicle.getIDList()
            
            for veh_id in veh_ids:
                try:
                    edge = traci.vehicle.getRoadID(veh_id)
                    edge_length = traci.edge.getLength(edge) if edge else 500
                    
                    all_vehicles[veh_id] = {
                        'id': veh_id,
                        'edge': edge,
                        'lane_position': traci.vehicle.getLanePosition(veh_id),
                        'speed': traci.vehicle.getSpeed(veh_id),
                        'edge_length': edge_length,
                        'is_cv': traci.vehicle.getTypeID(veh_id) == 'CV'
                    }
                except Exception as e:
                    print(f"获取车辆 {veh_id} 数据失败: {e}")
                    continue
        except Exception as e:
            print(f"获取所有车辆数据失败: {e}")

        return all_vehicles
    
    def _compute_rewards(self) -> Dict[str, float]:
        """计算奖励（修复版：包含正向奖励）"""
        if not hasattr(self, 'reward_calculator'):
            from improved_rewards import ImprovedRewardCalculator
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
            
            arrived = traci.simulation.getArrivedNumber()
            total = traci.vehicle.getCount()
            
            inroute_completion = 0.0
            for veh_id in traci.vehicle.getIDList():
                try:
                    route_idx = traci.vehicle.getRouteIndex(veh_id)
                    route_len = len(traci.vehicle.getRoute(veh_id))
                    if route_len > 0:
                        inroute_completion += route_idx / route_len
                except:
                    continue
            
            if total == 0:
                return 0.0
            
            ocr = (arrived + inroute_completion) / total
            return min(ocr, 1.0)
            
        except:
            return 0.0
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        if self.current_step >= 3600:
            return True

        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                return True
        except Exception as e:
            print(f"检查仿真是否完成失败: {e}")

        return False
    
    def _start_sumo(self):
        """启动SUMO"""
        if self.is_running:
            try:
                traci.close()
            except Exception as e:
                print(f"关闭已有TraCI连接失败: {e}")

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-warnings", "true",
            "--seed", str(self.seed if self.seed else 42)
        ]

        traci.start(sumo_cmd)
        self.is_running = True

    def close(self):
        """关闭环境"""
        if self.is_running:
            try:
                traci.close()
            except Exception as e:
                print(f"关闭TraCI连接失败: {e}")
            self.is_running = False
    
    def get_control_summary(self) -> Dict:
        """
        获取控制权分配摘要
        
        Returns:
            每个路口控制的车辆列表
        """
        summary = {}
        
        for junc_id, agent in self.agents.items():
            controlled = agent.get_controlled_vehicles()
            summary[junc_id] = {
                'main': controlled['main'],
                'ramp': controlled['ramp'],
                'diverge': controlled['diverge'],
                'total': len(controlled['main']) + len(controlled['ramp']) + len(controlled['diverge'])
            }
        
        return summary


def print_control_zones():
    """打印控制区域划分"""
    print("=" * 70)
    print("控制区域划分（不重叠）")
    print("=" * 70)
    
    for junc_id, zone in CONTROL_ZONES.items():
        print(f"\n【{junc_id}】")
        print(f"  主路上游控制: {zone.main_upstream_edges} (范围: {zone.main_upstream_range}m)")
        print(f"  匝道上游控制: {zone.ramp_upstream_edges} (范围: {zone.ramp_upstream_range}m)")
        if zone.diverge_edges:
            print(f"  转出引导区域: {zone.diverge_edges} (范围: {zone.diverge_range}m)")
        if zone.excluded_edges:
            print(f"  排除区域: {zone.excluded_edges}")


def main():
    """测试控制区域划分"""
    print_control_zones()
    
    print("\n" + "=" * 70)
    print("控制区域验证")
    print("=" * 70)
    
    # 检查是否有重叠
    all_controlled_edges = set()
    for junc_id, zone in CONTROL_ZONES.items():
        zone_edges = set(zone.main_upstream_edges + zone.ramp_upstream_edges + zone.diverge_edges)
        
        overlap = all_controlled_edges & zone_edges
        if overlap:
            print(f"\n警告: {junc_id} 与其他路口有重叠控制区域: {overlap}")
        
        all_controlled_edges.update(zone_edges)
        print(f"\n{junc_id} 控制区域: {zone_edges}")
    
    print(f"\n所有控制区域: {all_controlled_edges}")
    print(f"控制区域总数: {len(all_controlled_edges)}")
    
    # 验证控制链
    print("\n" + "=" * 70)
    print("控制链验证")
    print("=" * 70)
    
    print("\n主路控制链:")
    print("  J5 (E2) → J14 (E9) → J15 (E10) → J17 (E12)")
    
    print("\n匝道控制:")
    print("  J5: E23")
    print("  J14: E15")
    print("  J15: E17")
    print("  J17: E19")
    
    print("\n转出控制:")
    print("  J15: E16")
    print("  J17: E18, E20")


if __name__ == '__main__':
    main()
