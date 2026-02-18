"""
路口级多智能体强化学习系统
每个路口作为独立智能体，根据拓扑类型设计专门的状态空间和策略
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

try:
    import traci
except ImportError:
    print("请安装traci: pip install traci")
    sys.exit(1)


class JunctionType(Enum):
    """路口类型枚举"""
    TYPE_A = "type_a"  # 单纯匝道汇入
    TYPE_B = "type_b"  # 匝道汇入 + 主路转出
    TYPE_C = "type_c"  # 单纯主路转出（较少见）
    UNKNOWN = "unknown"


@dataclass
class JunctionConfig:
    """路口配置"""
    junction_id: str
    junction_type: JunctionType
    
    # 道路配置
    main_incoming: List[str] = field(default_factory=list)      # 主路入边
    main_outgoing: List[str] = field(default_factory=list)      # 主路出边
    ramp_incoming: List[str] = field(default_factory=list)      # 匝道入边
    ramp_outgoing: List[str] = field(default_factory=list)      # 匝道出边
    reverse_incoming: List[str] = field(default_factory=list)   # 反向入边
    reverse_outgoing: List[str] = field(default_factory=list)   # 反向出边
    
    # 信号灯配置
    has_traffic_light: bool = False
    num_phases: int = 2
    current_phase: int = 0
    phase_time: float = 0.0
    
    # 控制范围
    detection_range: float = 200.0  # 检测范围（米）
    control_range: float = 100.0    # 控制范围（米）


@dataclass
class JunctionState:
    """路口状态"""
    junction_id: str
    timestamp: float
    
    # 主路状态
    main_vehicles: List[Dict] = field(default_factory=list)      # 主路车辆列表
    main_speed: float = 0.0                                       # 主路平均速度
    main_density: float = 0.0                                     # 主路密度
    main_queue_length: float = 0.0                               # 主路排队长度
    
    # 匝道状态
    ramp_vehicles: List[Dict] = field(default_factory=list)      # 匝道车辆列表
    ramp_speed: float = 0.0                                       # 匝道平均速度
    ramp_queue_length: float = 0.0                               # 匝道排队长度
    ramp_waiting_time: float = 0.0                               # 匝道等待时间
    
    # 转出状态（仅类型B）
    diverge_vehicles: List[Dict] = field(default_factory=list)   # 转出车辆列表
    diverge_queue_length: float = 0.0                            # 转出排队长度
    
    # 信号灯状态
    phase: int = 0
    phase_time: float = 0.0
    time_to_switch: float = 0.0
    
    # 冲突状态
    conflict_risk: float = 0.0  # 冲突风险指数
    gap_acceptance: float = 0.0  # 可接受间隙
    
    # CV车辆
    cv_vehicles_main: List[str] = field(default_factory=list)    # 主路CV车辆
    cv_vehicles_ramp: List[str] = field(default_factory=list)    # 匝道CV车辆


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
        num_phases=2
    )
}


class JunctionAgent:
    """
    路口智能体
    每个路口作为独立智能体，负责局部观测和决策
    """
    
    def __init__(self, config: JunctionConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.junction_id = config.junction_id
        self.junction_type = config.junction_type
        
        # 状态缓存
        self.current_state: Optional[JunctionState] = None
        self.state_history: List[JunctionState] = []
        
        # 动作历史
        self.action_history: List[Dict] = []
        
        # 统计信息
        self.stats = {
            'total_throughput': 0,
            'avg_wait_time': 0,
            'conflict_count': 0
        }
    
    def observe(self, traci_conn=None) -> JunctionState:
        """
        观察路口状态
        
        Args:
            traci_conn: TraCI连接（默认使用全局traci）
        
        Returns:
            路口状态
        """
        if traci_conn is None:
            traci_conn = traci
        
        state = JunctionState(
            junction_id=self.junction_id,
            timestamp=traci_conn.simulation.getTime()
        )
        
        # 1. 观察主路状态
        state.main_vehicles = self._observe_vehicles(
            self.config.main_incoming, 
            traci_conn
        )
        state.main_speed = self._compute_avg_speed(state.main_vehicles)
        state.main_density = self._compute_density(state.main_vehicles, self.config.main_incoming)
        state.main_queue_length = self._compute_queue_length(
            self.config.main_incoming, 
            traci_conn
        )
        
        # 2. 观察匝道状态
        state.ramp_vehicles = self._observe_vehicles(
            self.config.ramp_incoming,
            traci_conn
        )
        state.ramp_speed = self._compute_avg_speed(state.ramp_vehicles)
        state.ramp_queue_length = self._compute_queue_length(
            self.config.ramp_incoming,
            traci_conn
        )
        state.ramp_waiting_time = self._compute_waiting_time(state.ramp_vehicles)
        
        # 3. 观察转出状态（仅类型B）
        if self.junction_type == JunctionType.TYPE_B:
            state.diverge_vehicles = self._observe_vehicles(
                self.config.ramp_outgoing,
                traci_conn,
                incoming=False
            )
            state.diverge_queue_length = self._compute_queue_length(
                self.config.ramp_outgoing,
                traci_conn
            )
        
        # 4. 观察信号灯状态
        if self.config.has_traffic_light:
            try:
                state.phase = traci_conn.trafficlight.getPhase(self.junction_id)
                state.time_to_switch = traci_conn.trafficlight.getNextSwitch(self.junction_id) - state.timestamp
            except:
                pass
        
        # 5. 计算冲突风险
        state.conflict_risk = self._compute_conflict_risk(state)
        state.gap_acceptance = self._compute_gap_acceptance(state)
        
        # 6. 识别CV车辆
        state.cv_vehicles_main = [
            v['id'] for v in state.main_vehicles 
            if v.get('is_cv', False)
        ]
        state.cv_vehicles_ramp = [
            v['id'] for v in state.ramp_vehicles 
            if v.get('is_cv', False)
        ]
        
        # 缓存状态
        self.current_state = state
        self.state_history.append(state)
        
        # 限制历史长度
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        return state
    
    def _observe_vehicles(self, edge_ids: List[str], traci_conn, incoming: bool = True) -> List[Dict]:
        """观察指定道路上的车辆"""
        vehicles = []
        
        for edge_id in edge_ids:
            try:
                veh_ids = traci_conn.edge.getLastStepVehicleIDs(edge_id)
                
                for veh_id in veh_ids:
                    try:
                        veh_info = {
                            'id': veh_id,
                            'speed': traci_conn.vehicle.getSpeed(veh_id),
                            'position': traci_conn.vehicle.getLanePosition(veh_id),
                            'lane': traci_conn.vehicle.getLaneIndex(veh_id),
                            'edge': edge_id,
                            'waiting_time': traci_conn.vehicle.getWaitingTime(veh_id),
                            'accel': traci_conn.vehicle.getAcceleration(veh_id),
                            'is_cv': traci_conn.vehicle.getTypeID(veh_id) == 'CV',
                            'route_index': traci_conn.vehicle.getRouteIndex(veh_id)
                        }
                        vehicles.append(veh_info)
                    except:
                        continue
                        
            except:
                continue
        
        # 按位置排序（距离路口近的在前）
        if incoming:
            vehicles.sort(key=lambda v: -v['position'])  # 入边：位置大的在前
        else:
            vehicles.sort(key=lambda v: v['position'])   # 出边：位置小的在前
        
        return vehicles
    
    def _compute_avg_speed(self, vehicles: List[Dict]) -> float:
        """计算平均速度"""
        if not vehicles:
            return 0.0
        return np.mean([v['speed'] for v in vehicles])
    
    def _compute_density(self, vehicles: List[Dict], edge_ids: List[str]) -> float:
        """计算密度"""
        if not edge_ids:
            return 0.0
        
        # 获取道路总长度
        total_length = 0
        for edge_id in edge_ids:
            try:
                total_length += traci.edge.getLength(edge_id)
            except:
                total_length += 100
        
        if total_length <= 0:
            return 0.0
        
        return len(vehicles) / (total_length / 1000)  # veh/km
    
    def _compute_queue_length(self, edge_ids: List[str], traci_conn) -> float:
        """计算排队长度"""
        queue_length = 0
        
        for edge_id in edge_ids:
            try:
                # 获取等待车辆数
                veh_ids = traci_conn.edge.getLastStepVehicleIDs(edge_id)
                for veh_id in veh_ids:
                    speed = traci_conn.vehicle.getSpeed(veh_id)
                    if speed < 0.1:  # 几乎停止
                        queue_length += 1
            except:
                continue
        
        return queue_length
    
    def _compute_waiting_time(self, vehicles: List[Dict]) -> float:
        """计算平均等待时间"""
        if not vehicles:
            return 0.0
        return np.mean([v['waiting_time'] for v in vehicles])
    
    def _compute_conflict_risk(self, state: JunctionState) -> float:
        """
        计算冲突风险指数
        基于主路和匝道的车辆密度、速度差等因素
        """
        if not state.main_vehicles or not state.ramp_vehicles:
            return 0.0
        
        # 主路密度
        main_density = len(state.main_vehicles) / max(len(self.config.main_incoming), 1)
        
        # 匝道密度
        ramp_density = len(state.ramp_vehicles) / max(len(self.config.ramp_incoming), 1)
        
        # 速度差
        speed_diff = abs(state.main_speed - state.ramp_speed)
        
        # 风险指数 = 密度乘积 * 速度差归一化
        risk = (main_density * ramp_density) * (speed_diff / 20.0)
        
        return min(risk, 1.0)
    
    def _compute_gap_acceptance(self, state: JunctionState) -> float:
        """
        计算可接受间隙
        基于主路车辆间距
        """
        if len(state.main_vehicles) < 2:
            return 1.0  # 主路车辆少，间隙大
        
        # 计算相邻车辆间距
        gaps = []
        for i in range(len(state.main_vehicles) - 1):
            gap = state.main_vehicles[i]['position'] - state.main_vehicles[i+1]['position']
            gaps.append(gap)
        
        if not gaps:
            return 0.5
        
        # 平均间隙归一化（假设安全间隙为50米）
        avg_gap = np.mean(gaps)
        return min(avg_gap / 50.0, 1.0)
    
    def get_state_vector(self, state: JunctionState = None) -> np.ndarray:
        """
        将状态转换为向量
        
        Args:
            state: 路口状态（默认使用当前状态）
        
        Returns:
            状态向量
        """
        if state is None:
            state = self.current_state
        
        if state is None:
            return np.zeros(self.get_state_dim())
        
        # 基础特征（所有类型共有）
        features = [
            # 主路特征
            len(state.main_vehicles) / 20.0,          # 主路车辆数归一化
            state.main_speed / 20.0,                   # 主路速度归一化
            state.main_density / 50.0,                 # 主路密度归一化
            state.main_queue_length / 20.0,            # 主路排队长度归一化
            
            # 匝道特征
            len(state.ramp_vehicles) / 10.0,           # 匝道车辆数归一化
            state.ramp_speed / 20.0,                   # 匝道速度归一化
            state.ramp_queue_length / 10.0,            # 匝道排队长度归一化
            state.ramp_waiting_time / 60.0,            # 匝道等待时间归一化
            
            # 信号灯特征
            state.phase / max(self.config.num_phases, 1),  # 当前相位
            state.time_to_switch / 100.0,              # 切换时间归一化
            
            # 冲突特征
            state.conflict_risk,                        # 冲突风险
            state.gap_acceptance,                       # 可接受间隙
            
            # CV车辆
            len(state.cv_vehicles_main) / max(len(state.main_vehicles), 1),  # 主路CV比例
            len(state.cv_vehicles_ramp) / max(len(state.ramp_vehicles), 1),  # 匝道CV比例
        ]
        
        # 类型B特有特征
        if self.junction_type == JunctionType.TYPE_B:
            features.extend([
                len(state.diverge_vehicles) / 10.0,    # 转出车辆数
                state.diverge_queue_length / 10.0,     # 转出排队长度
            ])
        else:
            features.extend([0.0, 0.0])  # 填充
        
        # 时间特征
        features.append(state.timestamp / 3600.0)  # 时间归一化
        
        return np.array(features, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 17  # 基础14 + 类型B特有2 + 时间1
    
    def get_action_dim(self) -> int:
        """获取动作维度"""
        if self.junction_type == JunctionType.TYPE_A:
            return 3  # 主路CV控制 + 匝道CV控制
        else:  # TYPE_B
            return 4  # 主路CV控制 + 匝道CV控制 + 转出引导
    
    def get_controlled_vehicles(self) -> Dict[str, List[str]]:
        """
        获取当前可控制的车辆
        
        Returns:
            {'main': [...], 'ramp': [...], 'diverge': [...]}
        """
        if self.current_state is None:
            return {'main': [], 'ramp': [], 'diverge': []}
        
        return {
            'main': self.current_state.cv_vehicles_main[:5],   # 最多控制5辆主路CV
            'ramp': self.current_state.cv_vehicles_ramp[:3],   # 最多控制3辆匝道CV
            'diverge': [] if self.junction_type != JunctionType.TYPE_B 
                      else [v['id'] for v in self.current_state.diverge_vehicles 
                            if v.get('is_cv', False)][:2]
        }


class MultiAgentEnvironment:
    """
    多智能体环境
    管理多个路口智能体
    """
    
    def __init__(self, junction_ids: List[str] = None, sumo_cfg: str = None, 
                 use_gui: bool = False, seed: int = None):
        """
        初始化多智能体环境
        
        Args:
            junction_ids: 要控制的路口ID列表
            sumo_cfg: SUMO配置文件路径
            use_gui: 是否使用GUI
            seed: 随机种子
        """
        self.junction_ids = junction_ids or list(JUNCTION_CONFIGS.keys())
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.seed = seed
        
        # 创建路口智能体
        self.agents: Dict[str, JunctionAgent] = {}
        for junc_id in self.junction_ids:
            if junc_id in JUNCTION_CONFIGS:
                self.agents[junc_id] = JunctionAgent(JUNCTION_CONFIGS[junc_id])
        
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
        
        # 重置智能体
        for agent in self.agents.values():
            agent.state_history.clear()
            agent.action_history.clear()
        
        # 执行几步让车辆进入
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
        
        # 观察初始状态
        observations = {}
        for junc_id, agent in self.agents.items():
            state = agent.observe()
            observations[junc_id] = agent.get_state_vector(state)
        
        return observations
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        执行一步
        
        Args:
            actions: {路口ID: {车辆ID: 动作值}}
        
        Returns:
            observations: 新观察
            rewards: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 应用动作
        self._apply_actions(actions)
        
        # 执行仿真步
        traci.simulationStep()
        self.current_step += 1
        
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
                    # 动作值转换为速度
                    # action范围: 0-1, 映射到速度范围
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    speed_limit = 13.89  # 默认限速
                    
                    target_speed = speed_limit * (0.3 + 0.9 * action)
                    traci.vehicle.setSpeed(veh_id, target_speed)
                    
                except:
                    continue
    
    def _compute_rewards(self) -> Dict[str, float]:
        """计算每个智能体的奖励"""
        rewards = {}
        
        for junc_id, agent in self.agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue
            
            # 奖励组成
            # 1. 通过量奖励
            throughput_reward = -state.main_queue_length * 0.1 - state.ramp_queue_length * 0.2
            
            # 2. 等待时间惩罚
            waiting_penalty = -state.ramp_waiting_time * 0.05
            
            # 3. 冲突风险惩罚
            conflict_penalty = -state.conflict_risk * 0.5
            
            # 4. 间隙利用奖励（匝道车辆能汇入）
            gap_reward = state.gap_acceptance * 0.2 if state.ramp_vehicles else 0
            
            # 5. 速度稳定性
            speed_stability = -abs(state.main_speed - state.ramp_speed) * 0.02
            
            total_reward = throughput_reward + waiting_penalty + conflict_penalty + gap_reward + speed_stability
            
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
