"""
路口级多智能体系统 - 使用SUMO订阅模式
提高数据收集效率，将信号灯相位作为重要特征
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict
import numpy as np

try:
    import traci
    import sumolib
except ImportError:
    print("请安装traci和sumolib: pip install traci sumolib")
    sys.exit(1)


class JunctionType(Enum):
    """路口类型枚举"""
    TYPE_A = "type_a"  # 单纯匝道汇入
    TYPE_B = "type_b"  # 匝道汇入 + 主路转出
    TYPE_C = "type_c"  # 单纯主路转出
    UNKNOWN = "unknown"


@dataclass
class TrafficLightPhase:
    """信号灯相位信息"""
    phase_id: int
    state: str           # 信号灯状态字符串，如 "GGrrGG"
    duration: float      # 相位持续时间
    min_duration: float  # 最小持续时间
    max_duration: float  # 最大持续时间
    
    # 解析状态
    main_state: str = ""      # 主路信号状态
    ramp_state: str = ""      # 匝道信号状态
    diverge_state: str = ""   # 转出信号状态
    
    def __post_init__(self):
        """解析信号状态"""
        # 根据状态字符串解析各方向信号
        if len(self.state) >= 2:
            # 假设前两个字符是主路，中间是匝道等
            self.main_state = self.state[:2] if len(self.state) >= 2 else ""
            self.ramp_state = self.state[2:4] if len(self.state) >= 4 else ""
            self.diverge_state = self.state[4:6] if len(self.state) >= 6 else ""


@dataclass
class JunctionConfig:
    """路口配置"""
    junction_id: str
    junction_type: JunctionType
    
    # 道路配置
    main_incoming: List[str] = field(default_factory=list)
    main_outgoing: List[str] = field(default_factory=list)
    ramp_incoming: List[str] = field(default_factory=list)
    ramp_outgoing: List[str] = field(default_factory=list)
    reverse_incoming: List[str] = field(default_factory=list)
    reverse_outgoing: List[str] = field(default_factory=list)
    
    # 所有相关边
    all_edges: List[str] = field(default_factory=list)
    all_lanes: List[str] = field(default_factory=list)
    
    # 信号灯配置
    has_traffic_light: bool = False
    tl_id: str = ""
    num_phases: int = 2
    phases: List[TrafficLightPhase] = field(default_factory=list)
    
    # 控制范围
    detection_range: float = 200.0
    control_range: float = 100.0
    
    def __post_init__(self):
        """初始化所有边和车道"""
        self.all_edges = (
            self.main_incoming + self.main_outgoing + 
            self.ramp_incoming + self.ramp_outgoing +
            self.reverse_incoming + self.reverse_outgoing
        )


@dataclass
class JunctionState:
    """路口状态 - 使用订阅数据"""
    junction_id: str
    timestamp: float
    
    # 主路状态
    main_vehicles: List[Dict] = field(default_factory=list)
    main_speed: float = 0.0
    main_density: float = 0.0
    main_queue_length: float = 0.0
    main_flow: float = 0.0
    
    # 匝道状态
    ramp_vehicles: List[Dict] = field(default_factory=list)
    ramp_speed: float = 0.0
    ramp_queue_length: float = 0.0
    ramp_waiting_time: float = 0.0
    ramp_flow: float = 0.0
    
    # 转出状态（仅类型B）
    diverge_vehicles: List[Dict] = field(default_factory=list)
    diverge_queue_length: float = 0.0
    
    # 信号灯状态（重要特征）
    current_phase: int = 0
    phase_state: str = ""
    time_in_phase: float = 0.0
    time_to_switch: float = 0.0
    next_phase: int = 0
    
    # 主路信号状态
    main_signal: str = ""       # "G"=绿灯, "r"=红灯, "y"=黄灯
    ramp_signal: str = ""
    diverge_signal: str = ""
    
    # 冲突状态
    conflict_risk: float = 0.0
    gap_acceptance: float = 0.0
    
    # CV车辆
    cv_vehicles_main: List[str] = field(default_factory=list)
    cv_vehicles_ramp: List[str] = field(default_factory=list)
    cv_vehicles_diverge: List[str] = field(default_factory=list)


# 预定义的路口配置（基于路网分析）
JUNCTION_CONFIGS = {
    'J5': JunctionConfig(
        junction_id='J5',
        junction_type=JunctionType.TYPE_A,
        main_incoming=['E2'],
        main_outgoing=['E3'],
        ramp_incoming=['E23'],
        reverse_incoming=['-E3'],
        reverse_outgoing=['-E2'],
        has_traffic_light=True,
        tl_id='J5',
        num_phases=2
    ),
    'J14': JunctionConfig(
        junction_id='J14',
        junction_type=JunctionType.TYPE_A,
        main_incoming=['E9'],
        main_outgoing=['E10'],
        ramp_incoming=['E15'],
        reverse_incoming=['-E10'],
        reverse_outgoing=['-E9'],
        has_traffic_light=True,
        tl_id='J14',
        num_phases=2
    ),
    'J15': JunctionConfig(
        junction_id='J15',
        junction_type=JunctionType.TYPE_B,
        main_incoming=['E10'],
        main_outgoing=['E11'],
        ramp_incoming=['E17'],
        ramp_outgoing=['E16'],
        reverse_incoming=['-E11'],
        reverse_outgoing=['-E10'],
        has_traffic_light=True,
        tl_id='J15',
        num_phases=2
    ),
    'J17': JunctionConfig(
        junction_id='J17',
        junction_type=JunctionType.TYPE_B,
        main_incoming=['E12'],
        main_outgoing=['E13'],
        ramp_incoming=['E19'],
        ramp_outgoing=['E18', 'E20'],
        reverse_incoming=['-E13'],
        reverse_outgoing=['-E12'],
        has_traffic_light=True,
        tl_id='J17',
        num_phases=2
    )
}


class SubscriptionManager:
    """
    SUMO订阅管理器
    统一管理所有订阅，提高数据收集效率
    """
    
    def __init__(self):
        self.vehicle_subscriptions: Dict[str, List[str]] = {}
        self.edge_subscriptions: Dict[str, List[str]] = {}
        self.lane_subscriptions: Dict[str, List[str]] = {}
        self.tl_subscriptions: Dict[str, List[str]] = {}
        
        # 订阅结果缓存
        self.vehicle_results: Dict[str, Dict] = {}
        self.edge_results: Dict[str, Dict] = {}
        self.lane_results: Dict[str, Dict] = {}
        self.tl_results: Dict[str, Dict] = {}
        
        # 已订阅的车辆
        self.subscribed_vehicles: set = set()
        
        # 订阅上下文
        self.subscription_context = {}
    
    def setup_vehicle_subscription(self, veh_ids: List[str],
                                   variables: List[int] = None):
        """
        设置车辆订阅

        Args:
            veh_ids: 车辆ID列表
            variables: 要订阅的变量列表（使用常量值）
        """
        if variables is None:
            # 使用常量值（兼容所有SUMO版本）
            variables = [
                0x40,  # VAR_SPEED
                0x42,  # VAR_POSITION
                0x41,  # VAR_ANGLE
                0x53,  # VAR_LANE_INDEX
                0x43,  # VAR_LANE_POSITION
                0x49,  # VAR_ROAD_ID
                0x56,  # VAR_ROUTE_INDEX
                0x4B,  # VAR_WAITING_TIME
                0x4A,  # VAR_ACCELERATION
                0x4D,  # VAR_VEHICLECLASS
                0x4E   # VAR_TYPE
            ]

        for veh_id in veh_ids:
            if veh_id not in self.subscribed_vehicles:
                try:
                    traci.vehicle.subscribe(veh_id, variables)
                    self.subscribed_vehicles.add(veh_id)
                except:
                    pass
    
    def setup_edge_subscription(self, edge_ids: List[str],
                                variables: List[int] = None):
        """
        设置道路边订阅

        Args:
            edge_ids: 边ID列表
            variables: 要订阅的变量列表（使用常量值而不是名称）
        """
        if variables is None:
            # 使用常量值（兼容所有SUMO版本）
            variables = [
                0x11,  # VAR_LAST_STEP_VEHICLE_NUMBER
                0x12,  # VAR_LAST_STEP_MEAN_SPEED
                0x13,  # VAR_LAST_STEP_VEHICLE_DATA (替代LAST_STEP_VEHICLE_IDS)
                0x14   # VAR_LAST_STEP_OCCUPANCY
            ]

        for edge_id in edge_ids:
            try:
                traci.edge.subscribe(edge_id, variables)
                self.edge_subscriptions[edge_id] = variables
            except:
                pass
    
    def setup_lane_subscription(self, lane_ids: List[str],
                                variables: List[int] = None):
        """
        设置车道订阅

        Args:
            lane_ids: 车道ID列表
            variables: 要订阅的变量列表（使用常量值）
        """
        if variables is None:
            # 使用常量值（兼容所有SUMO版本）
            variables = [
                0x11,  # VAR_LAST_STEP_VEHICLE_NUMBER
                0x13,  # VAR_LAST_STEP_VEHICLE_DATA
                0x10,  # VAR_LAST_STEP_HALTING_NUMBER
                0x12   # VAR_LAST_STEP_MEAN_SPEED
            ]

        for lane_id in lane_ids:
            try:
                traci.lane.subscribe(lane_id, variables)
                self.lane_subscriptions[lane_id] = variables
            except:
                pass
    
    def setup_traffic_light_subscription(self, tl_ids: List[str],
                                         variables: List[int] = None):
        """
        设置信号灯订阅

        Args:
            tl_ids: 信号灯ID列表
            variables: 要订阅的变量列表（使用常量值）
        """
        if variables is None:
            # 使用常量值（兼容所有SUMO版本）
            variables = [
                0x50,  # VAR_TL_CURRENT_PHASE
                0x51,  # VAR_TL_CURRENT_PROGRAM
                0x54,  # VAR_TL_PHASE_DURATION
                0x5A,  # VAR_TL_NEXT_SWITCH
                0x59,  # VAR_TL_RED_YELLOW_GREEN_STATE
                0x5B,  # VAR_TL_CONTROLLED_LANES
                0x5C   # VAR_TL_CONTROLLED_LINKS
            ]

        for tl_id in tl_ids:
            try:
                traci.trafficlight.subscribe(tl_id, variables)
                self.tl_subscriptions[tl_id] = variables
            except:
                pass
    
    def setup_context_subscription(self, edge_ids: List[str],
                                   radius: float = 200.0):
        """
        设置上下文订阅（检测范围内的车辆）

        Args:
            edge_ids: 边ID列表
            radius: 检测半径
        """
        vehicle_vars = [
            0x40,  # VAR_SPEED
            0x42,  # VAR_POSITION
            0x43,  # VAR_LANE_POSITION
            0x49,  # VAR_ROAD_ID
            0x4B,  # VAR_WAITING_TIME
            0x4D   # VAR_VEHICLECLASS
        ]

        for edge_id in edge_ids:
            try:
                # 使用边订阅获取车辆ID
                traci.edge.subscribe(edge_id, [0x13])  # VAR_LAST_STEP_VEHICLE_DATA
            except:
                pass
    
    def update_results(self):
        """更新所有订阅结果"""
        # 获取车辆订阅结果
        self.vehicle_results = {}
        for veh_id in list(self.subscribed_vehicles):
            try:
                results = traci.vehicle.getSubscriptionResults(veh_id)
                if results:
                    self.vehicle_results[veh_id] = results
            except:
                # 车辆已离开，从订阅列表移除
                self.subscribed_vehicles.discard(veh_id)
        
        # 获取边订阅结果
        self.edge_results = {}
        for edge_id in self.edge_subscriptions.keys():
            try:
                results = traci.edge.getSubscriptionResults(edge_id)
                if results:
                    self.edge_results[edge_id] = results
            except:
                pass
        
        # 获取车道订阅结果
        self.lane_results = {}
        for lane_id in self.lane_subscriptions.keys():
            try:
                results = traci.lane.getSubscriptionResults(lane_id)
                if results:
                    self.lane_results[lane_id] = results
            except:
                pass
        
        # 获取信号灯订阅结果
        self.tl_results = {}
        for tl_id in self.tl_subscriptions.keys():
            try:
                results = traci.trafficlight.getSubscriptionResults(tl_id)
                if results:
                    self.tl_results[tl_id] = results
            except:
                pass
    
    def get_vehicle_data(self, veh_id: str) -> Optional[Dict]:
        """获取单个车辆数据"""
        return self.vehicle_results.get(veh_id)
    
    def get_edge_data(self, edge_id: str) -> Optional[Dict]:
        """获取单个边数据"""
        return self.edge_results.get(edge_id)
    
    def get_tl_data(self, tl_id: str) -> Optional[Dict]:
        """获取信号灯数据"""
        return self.tl_results.get(tl_id)
    
    def cleanup_left_vehicles(self, current_vehicles: set):
        """清理已离开的车辆订阅"""
        left_vehicles = self.subscribed_vehicles - current_vehicles
        for veh_id in left_vehicles:
            self.subscribed_vehicles.discard(veh_id)
            if veh_id in self.vehicle_results:
                del self.vehicle_results[veh_id]


class JunctionAgent:
    """
    路口智能体 - 使用订阅模式
    """
    
    def __init__(self, config: JunctionConfig, subscription_manager: SubscriptionManager = None):
        self.config = config
        self.junction_id = config.junction_id
        self.junction_type = config.junction_type
        
        # 订阅管理器
        self.sub_manager = subscription_manager or SubscriptionManager()
        
        # 状态缓存
        self.current_state: Optional[JunctionState] = None
        self.state_history: List[JunctionState] = []
        
        # 动作历史
        self.action_history: List[Dict] = []
        
        # 初始化信号灯相位信息
        self._init_traffic_light_phases()
    
    def _init_traffic_light_phases(self):
        """初始化信号灯相位信息"""
        if not self.config.has_traffic_light:
            return
        
        try:
            # 获取信号灯程序
            program = traci.trafficlight.getAllProgramLogics(self.config.tl_id)
            if program:
                logic = program[0]  # 获取第一个程序
                self.config.phases = []
                
                for i, phase in enumerate(logic.phases):
                    tl_phase = TrafficLightPhase(
                        phase_id=i,
                        state=phase.state,
                        duration=phase.duration,
                        min_duration=phase.minDur,
                        max_duration=phase.maxDur
                    )
                    self.config.phases.append(tl_phase)
                
                self.config.num_phases = len(self.config.phases)
        except:
            # 如果无法获取，使用默认相位
            self.config.phases = [
                TrafficLightPhase(0, "GGrrGG", 90, 0, 0),
                TrafficLightPhase(1, "GGGGGG", 60, 0, 0)
            ]
    
    def setup_subscriptions(self):
        """设置该路口的所有订阅"""
        # 设置边订阅
        self.sub_manager.setup_edge_subscription(self.config.all_edges)
        
        # 设置车道订阅（为每条边的每个车道）
        lane_ids = []
        for edge_id in self.config.all_edges:
            try:
                lane_count = traci.edge.getLaneNumber(edge_id)
                for i in range(lane_count):
                    lane_ids.append(f"{edge_id}_{i}")
            except:
                pass
        
        self.sub_manager.setup_lane_subscription(lane_ids)
        
        # 设置信号灯订阅
        if self.config.has_traffic_light:
            self.sub_manager.setup_traffic_light_subscription([self.config.tl_id])
    
    def observe(self) -> JunctionState:
        """
        观察路口状态 - 使用订阅数据
        """
        timestamp = traci.simulation.getTime()
        
        state = JunctionState(
            junction_id=self.junction_id,
            timestamp=timestamp
        )
        
        # 1. 获取主路状态（使用订阅数据）
        state.main_vehicles = self._get_vehicles_from_edges(self.config.main_incoming)
        state.main_speed = self._get_mean_speed(self.config.main_incoming)
        state.main_density = self._compute_density(state.main_vehicles, self.config.main_incoming)
        state.main_queue_length = self._get_queue_length(self.config.main_incoming)
        state.main_flow = self._compute_flow(self.config.main_incoming, state.main_speed)
        
        # 2. 获取匝道状态
        state.ramp_vehicles = self._get_vehicles_from_edges(self.config.ramp_incoming)
        state.ramp_speed = self._get_mean_speed(self.config.ramp_incoming)
        state.ramp_queue_length = self._get_queue_length(self.config.ramp_incoming)
        state.ramp_waiting_time = self._get_avg_waiting_time(state.ramp_vehicles)
        state.ramp_flow = self._compute_flow(self.config.ramp_incoming, state.ramp_speed)
        
        # 3. 获取转出状态（仅类型B）
        if self.junction_type == JunctionType.TYPE_B:
            state.diverge_vehicles = self._get_vehicles_from_edges(self.config.ramp_outgoing)
            state.diverge_queue_length = self._get_queue_length(self.config.ramp_outgoing)
        
        # 4. 获取信号灯状态（重要特征）
        if self.config.has_traffic_light:
            state = self._observe_traffic_light(state)
        
        # 5. 计算冲突风险
        state.conflict_risk = self._compute_conflict_risk(state)
        state.gap_acceptance = self._compute_gap_acceptance(state)
        
        # 6. 识别CV车辆
        state.cv_vehicles_main = [v['id'] for v in state.main_vehicles if v.get('is_cv', False)]
        state.cv_vehicles_ramp = [v['id'] for v in state.ramp_vehicles if v.get('is_cv', False)]
        if self.junction_type == JunctionType.TYPE_B:
            state.cv_vehicles_diverge = [v['id'] for v in state.diverge_vehicles if v.get('is_cv', False)]
        
        # 缓存状态
        self.current_state = state
        self.state_history.append(state)
        
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        return state
    
    def _observe_traffic_light(self, state: JunctionState) -> JunctionState:
        """
        观察信号灯状态（重要特征）
        """
        tl_data = self.sub_manager.get_tl_data(self.config.tl_id)
        
        if tl_data:
            # 当前相位
            state.current_phase = tl_data.get(traci.constants.TL_CURRENT_PHASE, 0)
            
            # 信号状态字符串
            state.phase_state = tl_data.get(traci.constants.TL_RED_YELLOW_GREEN_STATE, "")
            
            # 下次切换时间
            next_switch = tl_data.get(traci.constants.TL_NEXT_SWITCH, 0)
            state.time_to_switch = next_switch - state.timestamp
            
            # 当前相位持续时间
            state.time_in_phase = state.time_to_switch
            
            # 下一个相位
            state.next_phase = (state.current_phase + 1) % self.config.num_phases
            
            # 解析各方向信号状态
            if state.phase_state:
                # 根据相位状态解析主路、匝道、转出信号
                phase_str = state.phase_state
                
                # 主路信号（假设前几个字符）
                if len(phase_str) >= 2:
                    state.main_signal = phase_str[0]  # 第一个字符
                
                # 匝道信号
                if len(phase_str) >= 4:
                    state.ramp_signal = phase_str[2]  # 第三个字符
                
                # 转出信号
                if len(phase_str) >= 6:
                    state.diverge_signal = phase_str[4]  # 第五个字符
        
        return state
    
    def _get_vehicles_from_edges(self, edge_ids: List[str]) -> List[Dict]:
        """从边获取车辆信息（使用订阅数据）"""
        vehicles = []
        
        for edge_id in edge_ids:
            edge_data = self.sub_manager.get_edge_data(edge_id)
            
            if edge_data:
                veh_ids = edge_data.get(traci.constants.LAST_STEP_VEHICLE_IDS, [])
                
                for veh_id in veh_ids:
                    veh_data = self.sub_manager.get_vehicle_data(veh_id)
                    
                    if veh_data:
                        veh_info = {
                            'id': veh_id,
                            'speed': veh_data.get(traci.constants.VAR_SPEED, 0),
                            'position': veh_data.get(traci.constants.VAR_POSITION, (0, 0)),
                            'lane': veh_data.get(traci.constants.VAR_LANE_INDEX, 0),
                            'lane_position': veh_data.get(traci.constants.VAR_LANE_POSITION, 0),
                            'edge': edge_id,
                            'waiting_time': veh_data.get(traci.constants.VAR_WAITING_TIME, 0),
                            'accel': veh_data.get(traci.constants.VAR_ACCELERATION, 0),
                            'is_cv': veh_data.get(traci.constants.VAR_VEHICLECLASS, '') == 'CV',
                            'route_index': veh_data.get(traci.constants.VAR_ROUTE_INDEX, 0)
                        }
                        vehicles.append(veh_info)
        
        # 按位置排序
        vehicles.sort(key=lambda v: -v['lane_position'])
        
        return vehicles
    
    def _get_mean_speed(self, edge_ids: List[str]) -> float:
        """获取平均速度（使用订阅数据）"""
        speeds = []
        
        for edge_id in edge_ids:
            edge_data = self.sub_manager.get_edge_data(edge_id)
            if edge_data:
                speed = edge_data.get(traci.constants.LAST_STEP_MEAN_SPEED, -1)
                if speed >= 0:
                    speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
    
    def _compute_density(self, vehicles: List[Dict], edge_ids: List[str]) -> float:
        """计算密度"""
        if not edge_ids:
            return 0.0
        
        total_length = 0
        for edge_id in edge_ids:
            try:
                total_length += traci.edge.getLength(edge_id)
            except:
                total_length += 100
        
        if total_length <= 0:
            return 0.0
        
        return len(vehicles) / (total_length / 1000)
    
    def _get_queue_length(self, edge_ids: List[str]) -> int:
        """获取排队长度（使用车道订阅数据）"""
        queue_length = 0
        
        for edge_id in edge_ids:
            # 检查每条车道
            try:
                lane_count = traci.edge.getLaneNumber(edge_id)
                for i in range(lane_count):
                    lane_id = f"{edge_id}_{i}"
                    lane_data = self.sub_manager.get_lane_data(lane_id)
                    
                    if lane_data:
                        halting = lane_data.get(traci.constants.LAST_STEP_HALTING_NUMBER, 0)
                        queue_length += halting
            except:
                pass
        
        return queue_length
    
    def _get_avg_waiting_time(self, vehicles: List[Dict]) -> float:
        """获取平均等待时间"""
        if not vehicles:
            return 0.0
        return np.mean([v['waiting_time'] for v in vehicles])
    
    def _compute_flow(self, edge_ids: List[str], mean_speed: float) -> float:
        """计算流量"""
        if not edge_ids or mean_speed <= 0:
            return 0.0
        
        # 获取车辆数
        total_vehicles = 0
        for edge_id in edge_ids:
            edge_data = self.sub_manager.get_edge_data(edge_id)
            if edge_data:
                total_vehicles += edge_data.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, 0)
        
        # 流量 = 速度 × 密度
        return mean_speed * total_vehicles
    
    def _compute_conflict_risk(self, state: JunctionState) -> float:
        """计算冲突风险"""
        if not state.main_vehicles or not state.ramp_vehicles:
            return 0.0
        
        # 基于信号灯状态调整风险
        signal_factor = 1.0
        if state.ramp_signal == 'G':
            signal_factor = 0.3  # 匝道绿灯，风险降低
        elif state.ramp_signal == 'r':
            signal_factor = 0.1  # 匝道红灯，风险最低
        
        # 密度因素
        main_density = len(state.main_vehicles) / max(len(self.config.main_incoming), 1)
        ramp_density = len(state.ramp_vehicles) / max(len(self.config.ramp_incoming), 1)
        
        # 速度差
        speed_diff = abs(state.main_speed - state.ramp_speed)
        
        risk = (main_density * ramp_density) * (speed_diff / 20.0) * signal_factor
        
        return min(risk, 1.0)
    
    def _compute_gap_acceptance(self, state: JunctionState) -> float:
        """计算可接受间隙"""
        if len(state.main_vehicles) < 2:
            return 1.0
        
        # 计算相邻车辆间距
        gaps = []
        for i in range(len(state.main_vehicles) - 1):
            gap = state.main_vehicles[i]['lane_position'] - state.main_vehicles[i+1]['lane_position']
            gaps.append(gap)
        
        if not gaps:
            return 0.5
        
        avg_gap = np.mean(gaps)
        
        # 根据信号灯状态调整
        if state.ramp_signal == 'G':
            return min(avg_gap / 30.0, 1.0)  # 绿灯时更容易汇入
        else:
            return min(avg_gap / 50.0, 1.0)
    
    def get_state_vector(self, state: JunctionState = None) -> np.ndarray:
        """
        将状态转换为向量（包含信号灯特征）
        """
        if state is None:
            state = self.current_state
        
        if state is None:
            return np.zeros(self.get_state_dim())
        
        # 基础特征
        features = [
            # 主路特征
            len(state.main_vehicles) / 20.0,
            state.main_speed / 20.0,
            state.main_density / 50.0,
            state.main_queue_length / 20.0,
            state.main_flow / 1000.0,
            
            # 匝道特征
            len(state.ramp_vehicles) / 10.0,
            state.ramp_speed / 20.0,
            state.ramp_queue_length / 10.0,
            state.ramp_waiting_time / 60.0,
            state.ramp_flow / 500.0,
            
            # 信号灯特征（重要）
            state.current_phase / max(self.config.num_phases, 1),
            state.time_to_switch / 100.0,
            float(state.main_signal == 'G'),
            float(state.ramp_signal == 'G'),
            float(state.diverge_signal == 'G') if self.junction_type == JunctionType.TYPE_B else 0.0,
            
            # 冲突特征
            state.conflict_risk,
            state.gap_acceptance,
            
            # CV车辆
            len(state.cv_vehicles_main) / max(len(state.main_vehicles), 1),
            len(state.cv_vehicles_ramp) / max(len(state.ramp_vehicles), 1),
        ]
        
        # 类型B特有特征
        if self.junction_type == JunctionType.TYPE_B:
            features.extend([
                len(state.diverge_vehicles) / 10.0,
                state.diverge_queue_length / 10.0,
                len(state.cv_vehicles_diverge) / max(len(state.diverge_vehicles), 1),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 时间特征
        features.append(state.timestamp / 3600.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 23  # 基础19 + 类型B特有3 + 时间1 = 23
    
    def get_action_dim(self) -> int:
        """获取动作维度"""
        if self.junction_type == JunctionType.TYPE_A:
            return 3
        else:
            return 4
    
    def get_controlled_vehicles(self) -> Dict[str, List[str]]:
        """获取当前可控制的车辆"""
        if self.current_state is None:
            return {'main': [], 'ramp': [], 'diverge': []}
        
        return {
            'main': self.current_state.cv_vehicles_main[:5],
            'ramp': self.current_state.cv_vehicles_ramp[:3],
            'diverge': self.current_state.cv_vehicles_diverge[:2] if self.junction_type == JunctionType.TYPE_B else []
        }


class MultiAgentEnvironment:
    """
    多智能体环境 - 使用订阅模式
    """
    
    def __init__(self, junction_ids: List[str] = None, sumo_cfg: str = None,
                 use_gui: bool = False, seed: int = None):
        self.junction_ids = junction_ids or list(JUNCTION_CONFIGS.keys())
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.seed = seed
        
        # 创建订阅管理器
        self.sub_manager = SubscriptionManager()
        
        # 创建路口智能体
        self.agents: Dict[str, JunctionAgent] = {}
        for junc_id in self.junction_ids:
            if junc_id in JUNCTION_CONFIGS:
                self.agents[junc_id] = JunctionAgent(
                    JUNCTION_CONFIGS[junc_id], 
                    self.sub_manager
                )
        
        # 仿真状态
        self.current_step = 0
        self.is_running = False
        
        # 全局统计
        self.global_stats = {
            'total_ocr': 0.0,
            'total_throughput': 0,
            'total_waiting': 0
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self._start_sumo()
        
        self.current_step = 0
        
        # 设置订阅
        self._setup_all_subscriptions()
        
        # 执行几步让车辆进入
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
            self._update_subscriptions()
        
        # 观察初始状态
        observations = {}
        for junc_id, agent in self.agents.items():
            state = agent.observe()
            observations[junc_id] = agent.get_state_vector(state)
        
        return observations
    
    def _setup_all_subscriptions(self):
        """设置所有订阅"""
        # 设置全局订阅
        self.sub_manager.setup_edge_subscription(
            list(set([edge for agent in self.agents.values() 
                     for edge in agent.config.all_edges]))
        )
        
        # 设置信号灯订阅
        tl_ids = [agent.config.tl_id for agent in self.agents.values() 
                 if agent.config.has_traffic_light]
        self.sub_manager.setup_traffic_light_subscription(tl_ids)
        
        # 为每个智能体设置订阅
        for agent in self.agents.values():
            agent.setup_subscriptions()
    
    def _update_subscriptions(self):
        """更新订阅"""
        # 更新订阅结果
        self.sub_manager.update_results()
        
        # 更新车辆订阅（为新进入的车辆）
        current_vehicles = set(traci.vehicle.getIDList())
        
        # 订阅新车辆
        new_vehicles = current_vehicles - self.sub_manager.subscribed_vehicles
        if new_vehicles:
            self.sub_manager.setup_vehicle_subscription(list(new_vehicles))
        
        # 清理已离开的车辆
        self.sub_manager.cleanup_left_vehicles(current_vehicles)
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """执行一步"""
        # 应用动作
        self._apply_actions(actions)
        
        # 执行仿真步
        traci.simulationStep()
        self.current_step += 1
        
        # 更新订阅
        self._update_subscriptions()
        
        # 观察新状态
        observations = {}
        for junc_id, agent in self.agents.items():
            state = agent.observe()
            observations[junc_id] = agent.get_state_vector(state)
        
        # 计算奖励
        rewards = self._compute_rewards()
        
        # 检查是否结束
        done = self._is_done()
        
        # 额外信息
        info = {
            'step': self.current_step,
            'global_stats': self.global_stats.copy()
        }
        
        return observations, rewards, done, info
    
    def _start_sumo(self):
        """启动SUMO"""
        if self.is_running:
            try:
                traci.close()
            except:
                pass
        
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-warnings", "true",
            "--seed", str(self.seed if self.seed else 42)
        ]
        
        traci.start(sumo_cmd)
        self.is_running = True
    
    def _apply_actions(self, actions: Dict[str, Dict]):
        """应用动作"""
        for junc_id, action_dict in actions.items():
            agent = self.agents.get(junc_id)
            if agent is None:
                continue
            
            for veh_id, action in action_dict.items():
                try:
                    speed_limit = 13.89
                    target_speed = speed_limit * (0.3 + 0.9 * action)
                    traci.vehicle.setSpeed(veh_id, target_speed)
                except:
                    continue
    
    def _compute_rewards(self) -> Dict[str, float]:
        """计算奖励"""
        rewards = {}
        
        for junc_id, agent in self.agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue
            
            # 奖励组成
            throughput_reward = -state.main_queue_length * 0.1 - state.ramp_queue_length * 0.2
            waiting_penalty = -state.ramp_waiting_time * 0.05
            conflict_penalty = -state.conflict_risk * 0.5
            gap_reward = state.gap_acceptance * 0.2 if state.ramp_vehicles else 0
            speed_stability = -abs(state.main_speed - state.ramp_speed) * 0.02
            
            # 信号灯协调奖励
            signal_reward = 0.0
            if state.ramp_signal == 'G' and state.ramp_vehicles:
                # 匝道绿灯且有车辆，鼓励汇入
                signal_reward = 0.1
            elif state.ramp_signal == 'r' and state.ramp_vehicles:
                # 匝道红灯但有车辆，惩罚排队
                signal_reward = -0.1 * len(state.ramp_vehicles)
            
            total_reward = (throughput_reward + waiting_penalty + conflict_penalty + 
                          gap_reward + speed_stability + signal_reward)
            
            rewards[junc_id] = total_reward
        
        return rewards
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        if self.current_step >= 3600:
            return True
        
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                return True
        except:
            pass
        
        return False
    
    def close(self):
        """关闭环境"""
        if self.is_running:
            try:
                traci.close()
            except:
                pass
            self.is_running = False
    
    def get_agent(self, junction_id: str) -> Optional[JunctionAgent]:
        """获取指定路口的智能体"""
        return self.agents.get(junction_id)
    
    def get_all_agents(self) -> Dict[str, JunctionAgent]:
        """获取所有智能体"""
        return self.agents
