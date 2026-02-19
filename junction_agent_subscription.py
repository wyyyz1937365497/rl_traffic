"""
路口级多智能体系统 - 使用SUMO订阅模式 (最终修复版)
修复内容：
1. 强制修正 LAST_STEP_VEHICLE_ID_LIST 为 0x13 (解决 float 迭代错误)
2. 强制修正 VAR_LANEPOSITION 命名 (解决属性不存在错误)
3. 完善所有必要的常量定义，确保跨版本兼容
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict
import numpy as np

# --- 1. 导入逻辑，支持 libsumo ---
if os.environ.get("USE_LIBSUMO", "0") == "1":
    try:
        import libsumo as traci
        print("成功加载 libsumo 作为 traci 后端")
    except ImportError:
        print("未找到 libsumo，回退到 traci")
        import traci
else:
    try:
        import traci
    except ImportError:
        pass

try:
    import sumolib
    import traci.constants as tc
except ImportError:
    print("请安装traci和sumolib: pip install traci sumolib")
    sys.exit(1)

# # --- 2. 核心修复：强制常量定义 (解决所有 AttributeError 和类型错误) ---
# # 不管原有定义如何，强制覆盖为正确的 TraCI 协议值

# # A. 边/车道订阅相关
# tc.LAST_STEP_VEHICLE_ID_LIST = 0x13          # 【关键修复】强制设为 0x13 (车辆ID列表)，防止取到速度(0x12)
# tc.LAST_STEP_VEHICLE_NUMBER = 0x11       # 车辆数量
# tc.LAST_STEP_MEAN_SPEED = 0x12           # 平均速度
# tc.LAST_STEP_VEHICLE_NUMBER = 0x10       # 停止车辆数
# tc.LAST_STEP_OCCUPANCY = 0x14            # 占有率

# # B. 车辆订阅相关
# tc.VAR_SPEED = 0x40
# tc.VAR_POSITION = 0x42
# tc.VAR_LANEPOSITION = 0x56               # 【关键修复】无下划线版本
# tc.VAR_LANE_POSITION = 0x56              # 兼容有下划线版本
# tc.VAR_LANE_INDEX = 0x53
# tc.VAR_ROAD_ID = 0x50
# tc.VAR_ROUTE_INDEX = 0x69
# tc.VAR_WAITING_TIME = 0x7a
# tc.VAR_ACCELERATION = 0x46
# tc.VAR_VEHICLECLASS = 0x49
# tc.VAR_TYPE = 0x03

# # C. 信号灯订阅相关
# tc.TL_CURRENT_PHASE = 0x50
# tc.VAR_TL_RED_YELLOW_GREEN_STATE = 0x59
# tc.VAR_TL_NEXT_SWITCH = 0x5a
# # 兼容旧版命名
# tc.LAST_STEP_TLS_CURRENT_PHASE = 0x50
# tc.LAST_STEP_TLS_CURRENT_PHASE = 0x51
# tc.LAST_STEP_TLS_PHASE_DURATION = 0x54
# tc.LAST_STEP_TLS_NEXT_SWITCH = 0x5a
# tc.LAST_STEP_TLS_RED_YELLOW_GREEN_STATE = 0x59
# tc.LAST_STEP_TLS_CONTROLLED_LANES = 0x5b
# tc.LAST_STEP_TLS_CONTROLLED_LINKS = 0x5c

# ------------------------------------------------


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
        if len(self.state) >= 2:
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

    @staticmethod
    def get_subscription_value(data_dict, key, default=0):
        """
        从订阅数据中提取值，处理可能的元组格式
        """
        value = data_dict.get(key, default)
        # 处理可能的元组格式 (某些旧版traci可能返回)
        if isinstance(value, tuple):
            value = value[0] if len(value) > 0 else default
        return value

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
    
    def setup_vehicle_subscription(self, veh_ids: List[str],
                                   variables: List[int] = None):
        """
        设置车辆订阅
        注意：libsumo不支持关键字参数，必须使用位置参数
        """
        if variables is None:
            # 使用强制定义的常量
            variables = [
                tc.VAR_SPEED,
                tc.VAR_POSITION,
                # tc.VAR_ANGLE, # 可选
                tc.VAR_LANE_INDEX,
                tc.VAR_LANEPOSITION, # 使用修正后的名称
                tc.VAR_ROAD_ID,
                tc.VAR_ROUTE_INDEX,
                tc.VAR_WAITING_TIME,
                tc.VAR_ACCELERATION,
                tc.VAR_VEHICLECLASS,
                tc.VAR_TYPE
            ]
        print(f"  - 订阅车辆: {veh_ids}")
        for veh_id in veh_ids:
            if veh_id not in self.subscribed_vehicles:
                try:
                    # 关键修复：使用位置参数，不使用 varIDs=...
                    traci.vehicle.subscribe(veh_id, variables)
                    self.subscribed_vehicles.add(veh_id)
                except Exception as e:
                    # libsumo可能会在车辆消失时抛出异常，忽略即可
                    pass
    
    def setup_edge_subscription(self, edge_ids: List[str],
                                variables: List[int] = None):
        """
        设置道路边订阅
        """
        if variables is None:
            variables = [
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_VEHICLE_ID_LIST,  # 强制使用 0x13
                tc.LAST_STEP_OCCUPANCY
            ]

        for edge_id in edge_ids:
            try:
                traci.edge.subscribe(edge_id, variables)
                self.edge_subscriptions[edge_id] = variables
            except:
                print(f"无法订阅边 {edge_id}，可能已消失")
                pass
    
    def setup_lane_subscription(self, lane_ids: List[str],
                                variables: List[int] = None):
        """
        设置车道订阅
        """
        if variables is None:
            variables = [
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_VEHICLE_ID_LIST,
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_MEAN_SPEED
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
        """
        if variables is None:
            variables = [
                tc.TL_CURRENT_PHASE,
                tc.TL_CURRENT_PROGRAM,
                tc.TL_PHASE_DURATION,
                tc.TL_NEXT_SWITCH,
                tc.TL_RED_YELLOW_GREEN_STATE,
                tc.TL_CONTROLLED_LANES,
                tc.TL_CONTROLLED_LINKS
            ]

        for tl_id in tl_ids:
            try:
                traci.trafficlight.subscribe(tl_id, variables)
                self.tl_subscriptions[tl_id] = variables
            except:
                pass
    
    def update_results(self):
        """
        更新所有订阅结果
        核心修复：使用 getAllSubscriptionResults 批量获取，效率更高
        """
        # 1. 批量获取车辆订阅结果
        try:
            # 该函数返回 {veh_id: {var_id: value}} 的字典
            self.vehicle_results = traci.vehicle.getAllSubscriptionResults()
            # 同步更新已订阅车辆集合（移除已消失的车辆）
            self.subscribed_vehicles = set(self.vehicle_results.keys())
        except:
            self.vehicle_results = {}
        
        # 2. 批量获取边订阅结果
        try:
            self.edge_results = traci.edge.getAllSubscriptionResults()
        except:
            self.edge_results = {}
        
        # 3. 批量获取车道订阅结果
        try:
            self.lane_results = traci.lane.getAllSubscriptionResults()
        except:
            self.lane_results = {}
        
        # 4. 批量获取信号灯订阅结果
        try:
            self.tl_results = traci.trafficlight.getAllSubscriptionResults()
        except:
            self.tl_results = {}
    
    def get_vehicle_data(self, veh_id: str) -> Optional[Dict]:
        """获取单个车辆数据"""
        return self.vehicle_results.get(veh_id)
    
    def get_edge_data(self, edge_id: str) -> Optional[Dict]:
        """获取单个边数据"""
        # print(self.edge_results)
        return self.edge_results.get(edge_id)
    
    def get_tl_data(self, tl_id: str) -> Optional[Dict]:
        """获取信号灯数据"""
        return self.tl_results.get(tl_id)
    
    def cleanup_left_vehicles(self, current_vehicles: set):
        """清理已离开的车辆订阅 (已由 update_results 中的逻辑自动处理)"""
        pass

# ... 后续的 JunctionAgent 和 MultiAgentEnvironment 类保持不变 ...
# 请确保 JunctionAgent 类中的 _get_vehicles_from_edges 方法使用 tc.VAR_LANEPOSITION (无下划线)
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
            except Exception as e:
                print(f"❌ [订阅失败] Edge ID: '{edge_id}' 不存在或无法订阅。错误: {e}")
                raise e
        
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
            state.current_phase = self.sub_manager.get_subscription_value(tl_data, tc.TL_CURRENT_PHASE, 0)

            # 信号状态字符串
            state.phase_state = tl_data.get(tc.TL_RED_YELLOW_GREEN_STATE, "")

            # 下次切换时间
            next_switch = self.sub_manager.get_subscription_value(tl_data, tc.TL_NEXT_SWITCH, 0)
            state.time_to_switch = next_switch - state.timestamp
            
            # 当前相位持续时间
            state.time_in_phase = state.time_to_switch
            
            # 下一个相位
            state.next_phase = (state.current_phase + 1) % self.config.num_phases
            
            # 解析各方向信号状态
            if state.phase_state:
                phase_str = state.phase_state
                if len(phase_str) >= 2:
                    state.main_signal = phase_str[0]
                if len(phase_str) >= 4:
                    state.ramp_signal = phase_str[2]
                if len(phase_str) >= 6:
                    state.diverge_signal = phase_str[4]
        
        return state
    
    def _get_vehicles_from_edges(self, edge_ids: List[str]) -> List[Dict]:
        """
        核心修复：从边订阅数据中获取车辆ID列表，而不是即时查询
        """
        vehicles = []

        for edge_id in edge_ids:
            # 1. 从订阅缓存中获取该边的数据
            edge_data = self.sub_manager.get_edge_data(edge_id)
            
            # 2. 获取车辆ID列表 (使用标准常量)
            if edge_data and tc.LAST_STEP_VEHICLE_ID_LIST in edge_data:
                veh_ids = edge_data[tc.LAST_STEP_VEHICLE_ID_LIST]
            else:
                # 如果订阅数据中暂时没有，回退到即时查询（通常在第一步发生）
                try:
                    veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                except:
                    continue

            # 3. 获取车辆详细订阅数据
            for veh_id in veh_ids:
                veh_data = self.sub_manager.get_vehicle_data(veh_id)

                if veh_data:
                    veh_info = {
                        'id': veh_id,
                        'speed': veh_data.get(tc.VAR_SPEED, 0),
                        'position': veh_data.get(tc.VAR_POSITION, (0, 0)),
                        'lane': veh_data.get(tc.VAR_LANE_INDEX, 0),
                        'lane_position': veh_data.get(tc.VAR_LANEPOSITION, 0),
                        'edge': edge_id,
                        'waiting_time': veh_data.get(tc.VAR_WAITING_TIME, 0),
                        'accel': veh_data.get(tc.VAR_ACCELERATION, 0),
                        'is_cv': veh_data.get(tc.VAR_VEHICLECLASS, '') == 'CV',
                        'route_index': veh_data.get(tc.VAR_ROUTE_INDEX, 0)
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
                speed = self.sub_manager.get_subscription_value(edge_data, tc.LAST_STEP_MEAN_SPEED, -1)
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
            try:
                lane_count = traci.edge.getLaneNumber(edge_id)
                for i in range(lane_count):
                    lane_id = f"{edge_id}_{i}"
                    lane_data = self.sub_manager.get_lane_data(lane_id)

                    if lane_data:
                        halting = self.sub_manager.get_subscription_value(lane_data, tc.LAST_STEP_VEHICLE_NUMBER, 0)
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
        
        total_vehicles = 0
        for edge_id in edge_ids:
            edge_data = self.sub_manager.get_edge_data(edge_id)
            if edge_data:
                veh_num = self.sub_manager.get_subscription_value(edge_data, tc.LAST_STEP_VEHICLE_NUMBER, 0)
                total_vehicles += veh_num

        return mean_speed * total_vehicles
    
    def _compute_conflict_risk(self, state: JunctionState) -> float:
        """计算冲突风险"""
        if not state.main_vehicles or not state.ramp_vehicles:
            return 0.0
        
        signal_factor = 1.0
        if state.ramp_signal == 'G':
            signal_factor = 0.3
        elif state.ramp_signal == 'r':
            signal_factor = 0.1
        
        main_density = len(state.main_vehicles) / max(len(self.config.main_incoming), 1)
        ramp_density = len(state.ramp_vehicles) / max(len(self.config.ramp_incoming), 1)
        
        speed_diff = abs(state.main_speed - state.ramp_speed)
        
        risk = (main_density * ramp_density) * (speed_diff / 20.0) * signal_factor
        
        return min(risk, 1.0)
    
    def _compute_gap_acceptance(self, state: JunctionState) -> float:
        """计算可接受间隙"""
        if len(state.main_vehicles) < 2:
            return 1.0
        
        gaps = []
        for i in range(len(state.main_vehicles) - 1):
            gap = state.main_vehicles[i]['lane_position'] - state.main_vehicles[i+1]['lane_position']
            gaps.append(gap)
        
        if not gaps:
            return 0.5
        
        avg_gap = np.mean(gaps)
        
        if state.ramp_signal == 'G':
            return min(avg_gap / 30.0, 1.0)
        else:
            return min(avg_gap / 50.0, 1.0)
    
    def get_state_vector(self, state: JunctionState = None) -> np.ndarray:
        """将状态转换为向量"""
        if state is None:
            state = self.current_state
        
        if state is None:
            return np.zeros(self.get_state_dim())
        
        features = [
            len(state.main_vehicles) / 20.0,
            state.main_speed / 20.0,
            state.main_density / 50.0,
            state.main_queue_length / 20.0,
            state.main_flow / 1000.0,
            
            len(state.ramp_vehicles) / 10.0,
            state.ramp_speed / 20.0,
            state.ramp_queue_length / 10.0,
            state.ramp_waiting_time / 60.0,
            state.ramp_flow / 500.0,
            
            state.current_phase / max(self.config.num_phases, 1),
            state.time_to_switch / 100.0,
            float(state.main_signal == 'G'),
            float(state.ramp_signal == 'G'),
            float(state.diverge_signal == 'G') if self.junction_type == JunctionType.TYPE_B else 0.0,
            
            state.conflict_risk,
            state.gap_acceptance,
            
            len(state.cv_vehicles_main) / max(len(state.main_vehicles), 1),
            len(state.cv_vehicles_ramp) / max(len(state.ramp_vehicles), 1),
        ]
        
        if self.junction_type == JunctionType.TYPE_B:
            features.extend([
                len(state.diverge_vehicles) / 10.0,
                state.diverge_queue_length / 10.0,
                len(state.cv_vehicles_diverge) / max(len(state.diverge_vehicles), 1),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        features.append(state.timestamp / 3600.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        return 23
    
    def get_action_dim(self) -> int:
        if self.junction_type == JunctionType.TYPE_A:
            return 3
        else:
            return 4
    
    def get_controlled_vehicles(self) -> Dict[str, List[str]]:
        if self.current_state is None:
            return {'main': [], 'ramp': [], 'diverge': []}
        
        return {
            'main': self.current_state.cv_vehicles_main[:5],
            'ramp': self.current_state.cv_vehicles_ramp[:3],
            'diverge': self.current_state.cv_vehicles_diverge[:2] if self.junction_type == JunctionType.TYPE_B else []
        }


class MultiAgentEnvironment:
    """多智能体环境"""
    
    def __init__(self, junction_ids: List[str] = None, sumo_cfg: str = None,
                 use_gui: bool = False, seed: int = None):
        self.junction_ids = junction_ids or list(JUNCTION_CONFIGS.keys())
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.seed = seed
        
        self.sub_manager = SubscriptionManager()
        
        self.agents: Dict[str, JunctionAgent] = {}
        for junc_id in self.junction_ids:
            if junc_id in JUNCTION_CONFIGS:
                self.agents[junc_id] = JunctionAgent(
                    JUNCTION_CONFIGS[junc_id], 
                    self.sub_manager
                )
        
        self.current_step = 0
        self.is_running = False
        
        self.global_stats = {
            'total_ocr': 0.0,
            'total_throughput': 0,
            'total_waiting': 0
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self._start_sumo()
        
        self.current_step = 0
        
        self._setup_all_subscriptions()
        
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
            self._update_subscriptions()
        
        observations = {}
        for junc_id, agent in self.agents.items():
            state = agent.observe()
            observations[junc_id] = agent.get_state_vector(state)
        
        return observations
    
    def _setup_all_subscriptions(self):
        """设置所有订阅"""
        self.sub_manager.setup_edge_subscription(
            list(set([edge for agent in self.agents.values() 
                     for edge in agent.config.all_edges]))
        )
        
        tl_ids = [agent.config.tl_id for agent in self.agents.values() 
                 if agent.config.has_traffic_light]
        self.sub_manager.setup_traffic_light_subscription(tl_ids)
        
        for agent in self.agents.values():
            agent.setup_subscriptions()
    
    def _update_subscriptions(self):
        """更新订阅"""
        self.sub_manager.update_results()
        
        current_vehicles = set(traci.vehicle.getIDList())
        
        new_vehicles = current_vehicles - self.sub_manager.subscribed_vehicles
        if new_vehicles:
            self.sub_manager.setup_vehicle_subscription(list(new_vehicles))
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """执行一步"""
        self._apply_actions(actions)
        
        traci.simulationStep()
        self.current_step += 1
        
        self._update_subscriptions()
        
        observations = {}
        for junc_id, agent in self.agents.items():
            state = agent.observe()
            observations[junc_id] = agent.get_state_vector(state)
        
        rewards = self._compute_rewards()
        
        done = self._is_done()
        
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
            
            throughput_reward = -state.main_queue_length * 0.1 - state.ramp_queue_length * 0.2
            waiting_penalty = -state.ramp_waiting_time * 0.05
            conflict_penalty = -state.conflict_risk * 0.5
            gap_reward = state.gap_acceptance * 0.2 if state.ramp_vehicles else 0
            speed_stability = -abs(state.main_speed - state.ramp_speed) * 0.02
            
            signal_reward = 0.0
            if state.ramp_signal == 'G' and state.ramp_vehicles:
                signal_reward = 0.1
            elif state.ramp_signal == 'r' and state.ramp_vehicles:
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
        return self.agents.get(junction_id)
    
    def get_all_agents(self) -> Dict[str, JunctionAgent]:
        return self.agents
