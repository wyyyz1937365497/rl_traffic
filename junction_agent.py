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

    # ===== 新增：车道级冲突信息 =====
    num_main_lanes: int = 0        # 主路车道数
    num_ramp_lanes: int = 0        # 匝道车道数
    conflict_lanes: List[str] = field(default_factory=list)  # 冲突车道列表

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


# ============================================================================
# 交叉口配置（从 road_topology_hardcoded.py 动态生成）
# ============================================================================
# 为了向后兼容，保留 JUNCTION_CONFIGS 变量
# 现在从 road_topology_hardcoded.py 的简化配置动态生成完整的 JunctionConfig
# ============================================================================

def _build_junction_configs_from_topology():
    """从 road_topology_hardcoded.py 动态构建 JUNCTION_CONFIGS"""
    try:
        from road_topology_hardcoded import (
            JUNCTION_CONFIG,
            create_junction_config_from_dict
        )
        # 使用转换函数从简化配置生成完整的 JunctionConfig 对象
        return {
            junc_id: create_junction_config_from_dict(junc_id, config_dict)
            for junc_id, config_dict in JUNCTION_CONFIG.items()
        }
    except ImportError:
        # 如果导入失败，返回空字典（向后兼容）
        print("警告：无法从 road_topology_hardcoded.py 导入配置，JUNCTION_CONFIGS 将为空")
        return {}

# 动态生成 JUNCTION_CONFIGS
JUNCTION_CONFIGS = _build_junction_configs_from_topology()


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
        for veh_id in veh_ids:
            if veh_id not in self.subscribed_vehicles:
                try:
                    # 关键修复：使用位置参数，不使用 varIDs=...
                    traci.vehicle.subscribe(veh_id, variables)
                    self.subscribed_vehicles.add(veh_id)
                except Exception as e:
                    # libsumo可能会在车辆消失时抛出异常，忽略即可
                    print(f"订阅车辆 {veh_id} 失败: {e}")
    
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
            except Exception as e:
                print(f"订阅边 {edge_id} 失败: {e}")
    
    def setup_lane_subscription(self, lane_ids: List[str],
                                variables: List[int] = None):
        """
        设置车道订阅
        """
        if variables is None:
            variables = [
                0x10,  # LAST_STEP_VEHICLE_NUMBER (车辆总数)
                0x13,  # LAST_STEP_VEHICLE_ID_LIST (车辆ID列表)
                0x15,  # LAST_STEP_VEHICLE_HALTING_NUMBER (停止车辆数) - 关键修复
                0x12   # LAST_STEP_MEAN_SPEED (平均速度)
            ]

        for lane_id in lane_ids:
            try:
                traci.lane.subscribe(lane_id, variables)
                self.lane_subscriptions[lane_id] = variables
            except Exception as e:
                print(f"订阅车道 {lane_id} 失败: {e}")
    
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
            except Exception as e:
                print(f"订阅信号灯 {tl_id} 失败: {e}")
    
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
        except Exception as e:
            print(f"获取车辆订阅结果失败: {e}")
            self.vehicle_results = {}

        # 2. 批量获取边订阅结果
        try:
            self.edge_results = traci.edge.getAllSubscriptionResults()
        except Exception as e:
            print(f"获取边订阅结果失败: {e}")
            self.edge_results = {}

        # 3. 批量获取车道订阅结果
        try:
            self.lane_results = traci.lane.getAllSubscriptionResults()
        except Exception as e:
            print(f"获取车道订阅结果失败: {e}")
            self.lane_results = {}

        # 4. 批量获取信号灯订阅结果
        try:
            self.tl_results = traci.trafficlight.getAllSubscriptionResults()
        except Exception as e:
            print(f"获取信号灯订阅结果失败: {e}")
            self.tl_results = {}
    
    def get_vehicle_data(self, veh_id: str) -> Optional[Dict]:
        """获取单个车辆数据"""
        return self.vehicle_results.get(veh_id)

    def get_edge_data(self, edge_id: str) -> Optional[Dict]:
        """获取单个边数据"""
        # print(self.edge_results)
        return self.edge_results.get(edge_id)

    def get_lane_data(self, lane_id: str) -> Optional[Dict]:
        """获取单个车道数据"""
        return self.lane_results.get(lane_id)

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

        # 奖励分解（用于调试）
        self.reward_breakdown = {}

        # 不在 __init__ 中初始化信号灯相位，等 SUMO 启动后在 setup_subscriptions 中初始化
    
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
        except Exception as e:
            print(f"获取信号灯 {self.config.tl_id} 相位信息失败，使用默认相位: {e}")
            # 如果无法获取，使用默认相位
            self.config.phases = [
                TrafficLightPhase(0, "GGrrGG", 90, 0, 0),
                TrafficLightPhase(1, "GGGGGG", 60, 0, 0)
            ]
    
    def setup_subscriptions(self):
        """设置该路口的所有订阅"""
        # 初始化信号灯相位信息（此时 SUMO 已启动，只初始化一次）
        if self.config.has_traffic_light and not getattr(self.config, 'phases', None):
            self._init_traffic_light_phases()

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
        step = int(traci.simulation.getTime())

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
                except Exception as e:
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
                        'is_cv': veh_data.get(tc.VAR_TYPE, '') == 'CV',
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
                # 通过车道获取边长度（SUMO中边由车道组成）
                lane_count = traci.edge.getLaneNumber(edge_id)
                if lane_count > 0:
                    # 获取第一条车道的长度作为边的长度
                    first_lane_length = traci.lane.getLength(f"{edge_id}_0")
                    total_length += first_lane_length
                else:
                    total_length += 100  # 默认值
            except Exception as e:
                print(f"获取边 {edge_id} 长度失败，使用默认值100: {e}")
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
                        # 使用 0x15 (LAST_STEP_VEHICLE_HALTING_NUMBER) 获取停止车辆数
                        halting = self.sub_manager.get_subscription_value(lane_data, 0x15, 0)
                        queue_length += halting
            except Exception as e:
                print(f"获取车道 {lane_id} 排队长度时出错: {e}")
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
        """计算冲突风险（使用车道级冲突矩阵）"""
        if not state.main_vehicles or not state.ramp_vehicles:
            return 0.0

        # 导入车道冲突矩阵
        try:
            from road_topology_hardcoded import LANE_CONFLICTS
        except ImportError:
            # 如果导入失败，使用默认冲突矩阵（关键路口）
            LANE_CONFLICTS = {
                # J5: E23匝道汇入-E2
                'E23_0': ['-E2_0', '-E2_1'],
                # J14: E15匝道汇入-E9
                'E15_0': ['-E10_0', '-E10_1'],
                # J15: E17匝道汇入-E10
                'E17_0': ['-E11_0', '-E11_1'],
                # J17: E19匝道汇入-E12
                'E19_0': ['-E13_0', '-E13_1'],
                'E19_1': ['-E13_0'],
            }
            print("[警告] 使用默认车道冲突矩阵（可能不完全准确）")

        # 基础密度风险
        main_density = len(state.main_vehicles) / max(len(self.config.main_incoming), 1)
        ramp_density = len(state.ramp_vehicles) / max(len(self.config.ramp_incoming), 1)

        # 速度差风险
        speed_diff = abs(state.main_speed - state.ramp_speed)

        # 信号灯影响
        signal_factor = 1.0
        if state.ramp_signal == 'G':
            signal_factor = 0.3
        elif state.ramp_signal == 'r':
            signal_factor = 0.1

        # 车道冲突风险计算
        lane_conflict_risk = 0.0
        if LANE_CONFLICTS:
            # 获取该路口配置的车道冲突信息
            junction_config = LANE_CONFLICTS

            # 检查匝道车辆与主路车辆的车道冲突
            for ramp_veh in state.ramp_vehicles:
                # 构建车道ID（格式：edge_lane）
                edge_id = ramp_veh.get('edge', '')
                lane_idx = ramp_veh.get('lane', 0)
                ramp_lane_id = f"{edge_id}_{lane_idx}"

                # 获取该匝道车道的冲突车道列表
                conflict_lanes = junction_config.get(ramp_lane_id, [])

                # 检查主路车辆是否在冲突车道上
                for main_veh in state.main_vehicles:
                    main_edge = main_veh.get('edge', '')
                    main_lane_idx = main_veh.get('lane', 0)
                    main_lane_id = f"{main_edge}_{main_lane_idx}"

                    if main_lane_id in conflict_lanes:
                        # 存在车道冲突，增加风险
                        lane_conflict_risk += 0.15

                        # 如果速度差异大，增加额外风险
                        veh_speed_diff = abs(ramp_veh.get('speed', 0) - main_veh.get('speed', 0))
                        if veh_speed_diff > 5.0:  # 速度差超过5m/s
                            lane_conflict_risk += 0.10

            # 归一化车道冲突风险
            max_possible_conflicts = len(state.ramp_vehicles) * len(state.main_vehicles)
            if max_possible_conflicts > 0:
                lane_conflict_risk = min(lane_conflict_risk / max_possible_conflicts * 5.0, 1.0)

        # 综合风险计算（加权组合）
        risk = (
            main_density * ramp_density * 0.25 +           # 密度风险
            (speed_diff / 20.0) * 0.25 +                    # 速度差风险
            min(lane_conflict_risk, 1.0) * 0.35 +           # 车道冲突风险
            (1.0 - signal_factor) * 0.15                    # 信号灯风险
        )

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
            # 主路状态（基于3600步实际数据，向上取整到整十数）
            len(state.main_vehicles) / 10.0,     # 99分位数7.0 → 取整10
            state.main_speed / 20.0,              # 99分位数15.5 m/s → 取整20
            state.main_density / 10.0,            # 99分位数7.0 → 取整10
            state.main_queue_length / 10.0,       # 最大值2 → 取整10
            state.main_flow / 100.0,              # 99分位数85.1 → 取整100

            # 匝道状态（基于3600步实际数据，向上取整到整十数）
            len(state.ramp_vehicles) / 40.0,      # 99分位数32 → 取整40
            state.ramp_speed / 10.0,              # 99分位数5.2 m/s → 取整10
            state.ramp_queue_length / 40.0,       # 99分位数32 → 取整40
            state.ramp_waiting_time / 80.0,       # 99分位数72.4s → 取整80
            state.ramp_flow / 80.0,               # 99分位数67.9 → 取整80

            # 信号灯状态
            state.current_phase / max(self.config.num_phases, 1),
            state.time_to_switch / 10.0,          # 最大值0 → 取整10（保留余地）
            float(state.main_signal == 'G'),
            float(state.ramp_signal == 'G'),
            float(state.diverge_signal == 'G') if self.junction_type == JunctionType.TYPE_B else 0.0,

            # 风险和间隙（已归一化）
            state.conflict_risk,
            state.gap_acceptance,

            # CV比例（已归一化）
            len(state.cv_vehicles_main) / max(len(state.main_vehicles), 1),
            len(state.cv_vehicles_ramp) / max(len(state.ramp_vehicles), 1),
        ]
        
        if self.junction_type == JunctionType.TYPE_B:
            features.extend([
                len(state.diverge_vehicles) / 10.0,     # 最大值0 → 取整10（保留余地）
                state.diverge_queue_length / 10.0,      # 最大值0 → 取整10（保留余地）
                len(state.cv_vehicles_diverge) / max(len(state.diverge_vehicles), 1),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        features.append(state.timestamp / 3600.0)      # 最大值3600 → 保持3600
        
        return np.array(features, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        return 23
    
    def get_action_dim(self) -> int:
        if self.junction_type == JunctionType.TYPE_A:
            return 3
        else:
            return 4
    
    def get_controlled_vehicles(self) -> Dict[str, List[str]]:
        """
        获取该路口控制的所有CV车辆

        修复：控制所有车道上的CV车辆，而不是只控制路口附近的车辆
        """
        if self.current_state is None:
            return {'main': [], 'ramp': [], 'diverge': []}

        # ✅ 返回所有监听到的CV车辆（移除数量限制）
        return {
            'main': self.current_state.cv_vehicles_main,  # 移除[:5]限制
            'ramp': self.current_state.cv_vehicles_ramp,  # 移除[:3]限制
            'diverge': self.current_state.cv_vehicles_diverge if self.junction_type == JunctionType.TYPE_B else []
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

        # ✅ 全局CV车辆分配缓存（确保每个CV只被一个路口控制）
        self._global_cv_assignment: Dict[str, str] = {}  # {veh_id: junction_id}
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境，返回状态向量"""
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
            observations[junc_id] = agent.get_state_vector(state)  # ✅ 返回向量，不是 JunctionState

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

        # ✅ 更新全局CV车辆分配
        self._assign_all_cv_vehicles()

    def _assign_all_cv_vehicles(self):
        """
        全局分配所有CV车辆给各个路口

        核心原则：
        对路口交通流有影响的是其上游到上一个路口前的车道

        分配策略（基于road_topology_hardcoded.py）：
        - J5: 控制 E2 + E23 + -E3（会到达或影响J5的边）
        - J14: 控制 E9 + E15 + -E10
        - J15: 控制 E10 + E17 + -E11 + E16
        - J17: 控制 E12 + E19 + -E13 + E18 + E20
        """
        try:
            from road_topology_hardcoded import JUNCTION_CONFIG, EDGE_TOPOLOGY

            # 获取所有CV车辆
            all_cv_vehicles = []
            for veh_id in traci.vehicle.getIDList():
                try:
                    if traci.vehicle.getTypeID(veh_id) == 'CV':
                        all_cv_vehicles.append(veh_id)
                except:
                    continue

            # 清空之前的分配
            self._global_cv_assignment.clear()

            # 为每个CV车辆分配路口
            for veh_id in all_cv_vehicles:
                try:
                    current_edge = traci.vehicle.getRoadID(veh_id)

                    # 根据车辆所在边，分配给对应的路口
                    # 原则：如果车辆在某个路口的"影响范围"内，就分配给该路口

                    assigned_junction = None

                    # J5的影响范围：E2(主路上游) + E23(匝道) + -E3(反向冲突)
                    if current_edge in ['E2', 'E23', '-E3']:
                        assigned_junction = 'J5'

                    # J14的影响范围：E9(主路上游) + E15(匝道) + -E10(反向冲突)
                    elif current_edge in ['E9', 'E15', '-E10']:
                        assigned_junction = 'J14'

                    # J15的影响范围：E10(主路上游) + E17(匝道) + -E11(反向冲突) + E16(转出)
                    elif current_edge in ['E10', 'E17', '-E11', 'E16']:
                        assigned_junction = 'J15'

                    # J17的影响范围：E12(主路上游) + E19(匝道) + -E13(反向冲突) + E18/E20(转出)
                    elif current_edge in ['E12', 'E19', '-E13', 'E18', 'E20']:
                        assigned_junction = 'J17'

                    # 如果车辆不在任何路口的直接影响范围内，根据拓扑关系分配
                    if assigned_junction is None:
                        # 查找这条边的下游路口
                        if current_edge in EDGE_TOPOLOGY:
                            edge_info = EDGE_TOPOLOGY[current_edge]
                            # 检查下游边
                            for downstream_edge in edge_info.downstream:
                                # 递归检查下游边属于哪个路口
                                for junc_id, junc_config in JUNCTION_CONFIG.items():
                                    affected_edges = (
                                        junc_config['main_incoming'] +
                                        junc_config.get('ramp_incoming', []) +
                                        junc_config.get('main_reverse', []) +
                                        junc_config.get('ramp_outgoing', [])
                                    )
                                    if downstream_edge in affected_edges:
                                        assigned_junction = junc_id
                                        break
                                if assigned_junction:
                                    break

                    # 如果还是找不到，使用默认分配策略
                    if assigned_junction is None:
                        # 正向边E1-E24：根据下游路口分配
                        if current_edge.startswith('E') and not current_edge.startswith('-E'):
                            if 'E1' <= current_edge <= 'E9':
                                assigned_junction = 'J14'  # 下游是J14
                            elif 'E10' <= current_edge <= 'E13':
                                assigned_junction = 'J15'  # 下游是J15
                            else:
                                assigned_junction = 'J14'
                        # 反向边-E1--E24
                        elif current_edge.startswith('-E'):
                            if '-E1' <= current_edge <= '-E3':
                                assigned_junction = 'J5'
                            elif '-E4' <= current_edge <= '-E11':
                                assigned_junction = 'J15'
                            elif '-E12' <= current_edge <= '-E13':
                                assigned_junction = 'J17'
                            else:
                                assigned_junction = 'J5'
                        else:
                            # 匝道边，根据其下游主路分配
                            assigned_junction = self.junction_ids[0] if self.junction_ids else 'J14'

                    self._global_cv_assignment[veh_id] = assigned_junction

                except Exception as e:
                    # 如果分配失败，默认分配给第一个路口
                    default_junction = self.junction_ids[0] if self.junction_ids else 'J14'
                    self._global_cv_assignment[veh_id] = default_junction

            # 调试：第一次分配时打印详细统计
            if not hasattr(self, '_cv_assignment_printed'):
                self._cv_assignment_printed = True

                # 统计每个路口的车辆分布
                junction_stats = {}
                edge_distribution = {}  # {edge_id: count}

                for veh_id, junc_id in self._global_cv_assignment.items():
                    if junc_id not in junction_stats:
                        junction_stats[junc_id] = 0
                    junction_stats[junc_id] += 1

                    # 统计边分布
                    try:
                        edge = traci.vehicle.getRoadID(veh_id)
                        edge_key = f"{edge}→{junc_id}"
                        edge_distribution[edge_key] = edge_distribution.get(edge_key, 0) + 1
                    except:
                        pass

                print(f"\n{'='*70}")
                print(f"CV车辆分配统计（基于road_topology_hardcoded.py）")
                print(f"{'='*70}")
                print(f"总CV车辆数: {len(all_cv_vehicles)}")
                print(f"\n路口分配:")
                for junc_id in sorted(junction_stats.keys()):
                    print(f"  {junc_id}: {junction_stats[junc_id]}辆")

                print(f"\n边→路口映射（前20条）:")
                for i, (edge_key, count) in enumerate(sorted(edge_distribution.items())[:20]):
                    print(f"  {edge_key}: {count}辆")
                print(f"{'='*70}\n")

        except Exception as e:
            print(f"[警告] 全局CV车辆分配失败: {e}")

    def get_controlled_vehicles_for_junction(self, junc_id: str) -> Dict[str, List[str]]:
        """
        获取指定路口控制的所有CV车辆

        基于road_topology_hardcoded.py的拓扑关系分类车辆

        Args:
            junc_id: 路口ID

        Returns:
            {'main': [...], 'ramp': [...], 'diverge': [...]}
        """
        if junc_id not in self.agents:
            return {'main': [], 'ramp': [], 'diverge': []}

        agent = self.agents[junc_id]
        config = agent.config

        # 获取分配给这个路口的所有CV车辆
        assigned_cvs = [
            veh_id for veh_id, assigned_junc in self._global_cv_assignment.items()
            if assigned_junc == junc_id
        ]

        # 从road_topology_hardcoded导入拓扑信息
        try:
            from road_topology_hardcoded import JUNCTION_CONFIG, EDGE_TOPOLOGY
            junc_topology = JUNCTION_CONFIG.get(junc_id, {})
        except:
            junc_topology = {}

        # 根据车辆所在的边，分类到main/ramp/diverge
        main_cvs = []
        ramp_cvs = []
        diverge_cvs = []

        for veh_id in assigned_cvs:
            try:
                current_edge = traci.vehicle.getRoadID(veh_id)

                # 使用拓扑信息判断车辆类型
                if junc_topology:
                    # 主路上游车辆
                    if current_edge in junc_topology.get('main_incoming', []):
                        main_cvs.append(veh_id)
                    # 匝道上游车辆
                    elif current_edge in junc_topology.get('ramp_incoming', []):
                        ramp_cvs.append(veh_id)
                    # 转出匝道车辆
                    elif current_edge in junc_topology.get('ramp_outgoing', []):
                        diverge_cvs.append(veh_id)
                    # 反向主路车辆（会与匝道汇入冲突）
                    elif current_edge in junc_topology.get('main_reverse', []):
                        main_cvs.append(veh_id)
                    else:
                        # 使用EDGE_TOPOLOGY判断
                        if current_edge in EDGE_TOPOLOGY:
                            edge_info = EDGE_TOPOLOGY[current_edge]
                            if edge_info.is_ramp:
                                # 判断是汇入匝道还是转出匝道
                                if agent.junction_type == JunctionType.TYPE_B:
                                    # 复杂路口，需要区分
                                    if current_edge in ['E16', 'E18', 'E20']:
                                        diverge_cvs.append(veh_id)
                                    else:
                                        ramp_cvs.append(veh_id)
                                else:
                                    ramp_cvs.append(veh_id)
                            else:
                                main_cvs.append(veh_id)
                        else:
                            main_cvs.append(veh_id)
                else:
                    # 回退到旧逻辑
                    if current_edge in config.ramp_incoming or current_edge in config.ramp_outgoing:
                        ramp_cvs.append(veh_id)
                    elif agent.junction_type == JunctionType.TYPE_B and \
                         (current_edge in config.ramp_outgoing or \
                          any(edge in current_edge for edge in ['E16', 'E18', 'E20'])):
                        diverge_cvs.append(veh_id)
                    else:
                        main_cvs.append(veh_id)
            except Exception:
                main_cvs.append(veh_id)

        return {
            'main': main_cvs,
            'ramp': ramp_cvs,
            'diverge': diverge_cvs if agent.junction_type == JunctionType.TYPE_B else []
        }
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """执行一步"""
        self._apply_actions(actions)

        traci.simulationStep()
        self.current_step += 1

        self._update_subscriptions()

        # ❌ 移除主动控制，让RL模型完全接管
        # self._active_cv_control()

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

        # 配置vType参数（关键优化！）
        self._configure_vtypes()

    def _configure_vtypes(self):
        """
        配置vType参数（基于规则方法的核心优化）

        根据分析报告：
        - sigma=0 消除随机减速 → |a|avg降低24%
        - tau=0.9 平滑跟车 → 减少急刹急加速
        - 温和加速参数 → 提升稳定性
        """
        try:
            # CV参数：消除随机减速，平滑跟车
            traci.vehicletype.setImperfection('CV', 0.0)  # sigma=0
            traci.vehicletype.setTau('CV', 0.9)           # tau=0.9
            traci.vehicletype.setSpeedFactor('CV', 1.0)   # 正常速度因子
            traci.vehicletype.setSpeedDeviation('CV', 0.0)  # 消除速度偏差
            traci.vehicletype.setAccel('CV', 0.8)         # 温和加速
            traci.vehicletype.setDecel('CV', 1.5)         # 温和减速

            # HV参数：同样优化
            traci.vehicletype.setImperfection('HV', 0.0)
            traci.vehicletype.setTau('HV', 0.9)
            traci.vehicletype.setSpeedFactor('HV', 1.0)
            traci.vehicletype.setSpeedDeviation('HV', 0.0)
            traci.vehicletype.setAccel('HV', 0.8)
            traci.vehicletype.setDecel('HV', 1.5)

            print("✓ vType参数已配置（sigma=0, tau=0.9）→ 预期|a|avg降低24%")
        except Exception as e:
            print(f"配置vType参数失败: {e}")

    def _active_cv_control(self):
        """
        CV主动速度引导（每一步都执行）

        策略:
        1. 仅在CV靠近边末尾50m时轻微减速
        2. 检测下游拥堵（前探2条边）
        3. 温和减速（使用slowDown，3秒持续时间）
        4. 不强制加速，避免创造额外拥堵
        """
        # 道路拓扑：边的连接关系
        NEXT_EDGE = {
            '-E13': '-E12', '-E12': '-E11', '-E11': '-E10', '-E10': '-E9',
            '-E9': '-E8', '-E8': '-E7', '-E7': '-E6', '-E6': '-E5',
            '-E5': '-E3', '-E3': '-E2', '-E2': '-E1',
            'E1': 'E2', 'E2': 'E3', 'E3': 'E5', 'E5': 'E6',
            'E6': 'E7', 'E7': 'E8', 'E8': 'E9', 'E9': 'E10',
            'E10': 'E11', 'E11': 'E12', 'E12': 'E13',
            'E23': '-E2', 'E15': 'E10', 'E17': '-E10', 'E19': '-E12',
        }

        SPEED_LIMIT = 13.89
        CONGEST_SPEED = 5.0  # 拥堵判定阈值
        LOOKAHEAD = 2        # 前探边数
        APPROACH_DIST = 50.0 # 干预距离阈值
        SPEED_FACTOR = 1.5   # 减速系数
        SPEED_FLOOR = 3.0    # 最小速度

        # 初始化受控车辆集合
        if not hasattr(self, '_controlled_cvs'):
            self._controlled_cvs = set()

        new_controlled = set()

        for veh_id in traci.vehicle.getIDList():
            try:
                # 只控制CV车辆
                if traci.vehicle.getTypeID(veh_id) != 'CV':
                    continue

                # 获取车辆当前位置
                edge = traci.vehicle.getRoadID(veh_id)
                if edge.startswith(':'):  # 跳过内部边
                    continue
                if edge not in NEXT_EDGE:
                    continue

                # 检查是否接近边末端
                pos = traci.vehicle.getLanePosition(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                try:
                    lane_len = traci.lane.getLength(lane_id)
                except:
                    continue

                dist_to_end = lane_len - pos

                # 太远 → 不干预
                if dist_to_end > APPROACH_DIST:
                    if veh_id in self._controlled_cvs:
                        traci.vehicle.setSpeed(veh_id, -1)  # 释放控制
                    continue

                # 查看下游边拥堵情况
                congested = False
                min_ds_speed = SPEED_LIMIT
                nxt = edge

                for _ in range(LOOKAHEAD):
                    nxt = NEXT_EDGE.get(nxt)
                    if nxt is None:
                        break
                    try:
                        ds_speed = traci.edge.getLastStepMeanSpeed(nxt)
                    except:
                        ds_speed = SPEED_LIMIT

                    if ds_speed < CONGEST_SPEED:
                        congested = True
                        min_ds_speed = min(min_ds_speed, ds_speed)

                # 如果下游拥堵，且当前速度过快，则温和减速
                if congested:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    target = max(min_ds_speed * SPEED_FACTOR, SPEED_FLOOR)
                    target = min(target, SPEED_LIMIT)

                    # 只在需要减速时干预（不会强制加速）
                    if target < current_speed:
                        traci.vehicle.slowDown(veh_id, target, 3.0)
                        new_controlled.add(veh_id)
                elif veh_id in self._controlled_cvs:
                    # 释放之前的控制
                    traci.vehicle.setSpeed(veh_id, -1)

            except Exception:
                continue

        # 释放不再需要控制的车辆
        for veh_id in self._controlled_cvs - new_controlled:
            try:
                traci.vehicle.setSpeed(veh_id, -1)
            except:
                pass

        self._controlled_cvs = new_controlled

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
                except Exception as e:
                    print(f"设置车辆 {veh_id} 速度失败: {e}")
                    continue
    
    def _compute_rewards(self) -> Dict[str, float]:
        """计算奖励（改进版，包含正向奖励）"""
        # 导入改进的奖励计算器
        from improved_rewards import ImprovedRewardCalculator

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
            total = traci.vehicle.getIDCount()

            if total == 0:
                return 0.0

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
            ocr = (arrived + inroute_completion) / total
            return min(ocr, 1.0)

        except Exception as e:
            print(f"计算OCR失败: {e}")
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
    
    def close(self):
        """关闭环境"""
        if self.is_running:
            try:
                traci.close()
            except Exception as e:
                print(f"关闭TraCI连接失败: {e}")
            self.is_running = False
    
    def get_agent(self, junction_id: str) -> Optional[JunctionAgent]:
        return self.agents.get(junction_id)
    
    def get_all_agents(self) -> Dict[str, JunctionAgent]:
        return self.agents
