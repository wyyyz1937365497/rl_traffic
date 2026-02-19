"""
SUMO交通仿真环境封装
实现Gym接口，用于强化学习训练
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
from collections import deque

try:
    import traci
    import sumolib
except ImportError:
    print("请安装traci和sumolib: pip install traci sumolib")
    sys.exit(1)

from config import EnvironmentConfig


@dataclass
class VehicleState:
    """车辆状态"""
    veh_id: str
    position: np.ndarray  # [x, y]
    speed: float
    angle: float
    lane_id: str
    lane_index: int
    lane_position: float
    edge_id: str
    route_index: int
    route_length: int
    distance_traveled: float
    distance_total: float
    completion_rate: float
    is_cv: bool  # 是否是智能网联车
    waiting_time: float
    acceleration: float


@dataclass
class EdgeState:
    """道路边状态"""
    edge_id: str
    length: float
    speed_limit: float
    lane_count: int
    vehicle_count: int
    mean_speed: float
    density: float  # 车辆密度
    flow: float  # 流量
    queue_length: float  # 排队长度


@dataclass
class JunctionState:
    """交叉口状态"""
    junction_id: str
    phase: int
    phase_time: float
    waiting_vehicles: int
    controlled_lanes: List[str]


class TrafficEnvironment:
    """
    SUMO交通仿真环境
    实现Gym风格的接口
    """
    
    def __init__(self, config: EnvironmentConfig, use_gui: bool = False, seed: int = None):
        """
        初始化环境
        
        Args:
            config: 环境配置
            use_gui: 是否使用GUI
            seed: 随机种子
        """
        self.config = config
        self.use_gui = use_gui
        self.seed_value = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 状态变量
        self.current_step = 0
        self.is_running = False
        
        # 车辆追踪
        self.vehicle_states: Dict[str, VehicleState] = {}
        self.cv_vehicles: List[str] = []  # 智能网联车列表
        self.controlled_vehicles: List[str] = []  # 当前被控制的车辆
        
        # 道路和交叉口状态
        self.edge_states: Dict[str, EdgeState] = {}
        self.junction_states: Dict[str, JunctionState] = {}
        
        # 历史状态
        self.state_history: deque = deque(maxlen=config.history_length)
        
        # 统计信息
        self.total_departed = 0
        self.total_arrived = 0
        self.total_distance_traveled = 0.0
        self.total_distance_total = 0.0
        
        # 动作记录
        self.last_actions: Dict[str, float] = {}  # 车辆ID -> 上一次速度动作
        
        # 路网信息
        self.edge_lengths: Dict[str, float] = {}
        self.edge_speed_limits: Dict[str, float] = {}
        self.edge_lane_counts: Dict[str, int] = {}
        
        # 初始化路网信息
        self._init_network_info()
    
    def _init_network_info(self):
        """初始化路网信息"""
        try:
            net = sumolib.net.readNet(self.config.net_file)
            for edge in net.getEdges():
                edge_id = edge.getID()
                self.edge_lengths[edge_id] = edge.getLength()
                self.edge_speed_limits[edge_id] = edge.getSpeed()
                self.edge_lane_counts[edge_id] = edge.getLaneNumber()
        except Exception as e:
            print(f"警告: 无法读取路网文件: {e}")
    
    def _start_sumo(self):
        """启动SUMO仿真"""
        if self.is_running:
            self._close_sumo()
        
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.config.sumo_cfg,
            "--no-warnings", "true",
            "--duration-log.statistics", "true",
            "--time-to-teleport", "600",
            "--ignore-route-errors", "true",
            "--collision.action", "warn",
            "--collision.stoptime", "5",
            "--collision.check-junctions", "true",
            "--collision.mingap-factor", "0",
            "--seed", str(self.seed_value if self.seed_value else random.randint(0, 1000000))
        ]
        
        try:
            traci.start(sumo_cmd)
            self.is_running = True
        except Exception as e:
            print(f"SUMO启动失败: {e}")
            raise
    
    def _close_sumo(self):
        """关闭SUMO仿真"""
        if self.is_running:
            try:
                traci.close()
            except Exception as e:
                print(f"关闭TraCI连接失败: {e}")
            self.is_running = False
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境
        
        Returns:
            初始状态
        """
        self._start_sumo()
        
        # 重置状态变量
        self.current_step = 0
        self.vehicle_states.clear()
        self.cv_vehicles.clear()
        self.controlled_vehicles.clear()
        self.edge_states.clear()
        self.junction_states.clear()
        self.state_history.clear()
        
        # 重置统计
        self.total_departed = 0
        self.total_arrived = 0
        self.total_distance_traveled = 0.0
        self.total_distance_total = 0.0
        self.last_actions.clear()
        
        # 执行几步让车辆进入
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
        
        # 初始化状态
        self._update_states()
        
        return self._get_observation()
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        执行一步
        
        Args:
            actions: 动作字典 {车辆ID: 速度比例}
        
        Returns:
            observation: 观察状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 应用动作
        self._apply_actions(actions)
        
        # 执行仿真步
        for _ in range(self.config.control_interval):
            traci.simulationStep()
            self.current_step += 1
            
            if self.current_step >= self.config.max_steps:
                break
        
        # 更新状态
        self._update_states()
        
        # 计算奖励
        reward, reward_info = self._compute_reward(actions)
        
        # 检查是否结束
        done = self._is_done()
        
        # 获取观察
        observation = self._get_observation()
        
        # 额外信息
        info = {
            'step': self.current_step,
            'total_departed': self.total_departed,
            'total_arrived': self.total_arrived,
            'ocr': self._compute_ocr(),
            **reward_info
        }
        
        return observation, reward, done, info
    
    def _apply_actions(self, actions: Dict[str, float]):
        """
        应用动作到车辆
        
        Args:
            actions: 动作字典 {车辆ID: 速度比例}
        """
        for veh_id, speed_ratio in actions.items():
            if veh_id not in self.cv_vehicles:
                continue
            
            try:
                # 获取当前车辆状态
                veh_state = self.vehicle_states.get(veh_id)
                if veh_state is None:
                    continue
                
                # 计算目标速度
                speed_limit = self.edge_speed_limits.get(veh_state.edge_id, 13.89)
                min_speed = speed_limit * self.config.min_speed_ratio
                max_speed = speed_limit * self.config.max_speed_ratio
                
                # 速度比例转换为实际速度
                target_speed = min_speed + speed_ratio * (max_speed - min_speed)
                target_speed = np.clip(target_speed, 0, max_speed)
                
                # 应用速度控制
                traci.vehicle.setSpeed(veh_id, target_speed)
                
                # 记录动作
                self.last_actions[veh_id] = speed_ratio
                
            except Exception as e:
                # 车辆可能已经离开
                pass
    
    def _update_states(self):
        """更新所有状态"""
        # 更新车辆状态
        self._update_vehicle_states()
        
        # 更新道路状态
        self._update_edge_states()
        
        # 更新交叉口状态
        self._update_junction_states()
        
        # 更新统计信息
        self._update_statistics()
    
    def _update_vehicle_states(self):
        """更新车辆状态"""
        # 获取当前车辆列表
        current_vehicles = set(traci.vehicle.getIDList())
        
        # 获取新到达的车辆
        arrived = set(traci.simulation.getArrivedIDList())
        departed = set(traci.simulation.getDepartedIDList())
        
        # 更新统计
        self.total_arrived += len(arrived)
        self.total_departed += len(departed)
        
        # 移除已离开的车辆
        vehicles_to_remove = set(self.vehicle_states.keys()) - current_vehicles
        for veh_id in vehicles_to_remove:
            del self.vehicle_states[veh_id]
            if veh_id in self.cv_vehicles:
                self.cv_vehicles.remove(veh_id)
            if veh_id in self.controlled_vehicles:
                self.controlled_vehicles.remove(veh_id)
        
        # 更新或添加车辆状态
        for veh_id in current_vehicles:
            try:
                # 获取车辆信息
                position = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                lane_position = traci.vehicle.getLanePosition(veh_id)
                edge_id = traci.vehicle.getRoadID(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)
                route = traci.vehicle.getRoute(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                accel = traci.vehicle.getAcceleration(veh_id)
                
                # 获取车辆类型
                veh_type = traci.vehicle.getTypeID(veh_id)
                is_cv = veh_type == 'CV'
                
                # 计算路径信息
                route_length = len(route)
                distance_traveled = self._calculate_distance_traveled(veh_id, route, route_index, lane_position)
                distance_total = self._calculate_route_length(route)
                completion_rate = distance_traveled / max(distance_total, 1.0)
                
                # 创建车辆状态
                veh_state = VehicleState(
                    veh_id=veh_id,
                    position=np.array(position, dtype=np.float32),
                    speed=speed,
                    angle=angle,
                    lane_id=lane_id,
                    lane_index=lane_index,
                    lane_position=lane_position,
                    edge_id=edge_id,
                    route_index=route_index,
                    route_length=route_length,
                    distance_traveled=distance_traveled,
                    distance_total=distance_total,
                    completion_rate=min(completion_rate, 1.0),
                    is_cv=is_cv,
                    waiting_time=waiting_time,
                    acceleration=accel
                )
                
                self.vehicle_states[veh_id] = veh_state
                
                # 更新CV车辆列表
                if is_cv and veh_id not in self.cv_vehicles:
                    self.cv_vehicles.append(veh_id)
                
            except Exception as e:
                continue
        
        # 更新被控制的车辆（选择部分CV车辆进行控制）
        self._update_controlled_vehicles()
    
    def _update_controlled_vehicles(self):
        """更新被控制的车辆列表"""
        # 选择一定比例的CV车辆进行控制
        max_control = max(1, int(len(self.cv_vehicles) * self.config.cv_ratio))
        
        # 优先选择在关键路段的车辆
        priority_vehicles = []
        other_vehicles = []
        
        for veh_id in self.cv_vehicles:
            veh_state = self.vehicle_states.get(veh_id)
            if veh_state:
                if veh_state.edge_id in self.config.critical_edges:
                    priority_vehicles.append(veh_id)
                else:
                    other_vehicles.append(veh_id)
        
        # 选择车辆
        selected = priority_vehicles[:max_control]
        remaining = max_control - len(selected)
        if remaining > 0:
            selected.extend(other_vehicles[:remaining])
        
        self.controlled_vehicles = selected
    
    def _update_edge_states(self):
        """更新道路边状态"""
        for edge_id in self.edge_lengths.keys():
            try:
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                
                length = self.edge_lengths[edge_id]
                speed_limit = self.edge_speed_limits[edge_id]
                lane_count = self.edge_lane_counts[edge_id]
                
                # 计算密度和流量
                density = vehicle_count / max(length * lane_count / 1000, 0.1)  # veh/km
                flow = mean_speed * density  # veh/h
                
                # 计算排队长度
                queue_length = 0
                for lane_idx in range(lane_count):
                    lane_id = f"{edge_id}_{lane_idx}"
                    try:
                        queue_length += traci.lane.getWaitingTime(lane_id) > 0
                    except Exception as e:
                        print(f"获取车道 {lane_id} 等待时间失败: {e}")
                
                self.edge_states[edge_id] = EdgeState(
                    edge_id=edge_id,
                    length=length,
                    speed_limit=speed_limit,
                    lane_count=lane_count,
                    vehicle_count=vehicle_count,
                    mean_speed=mean_speed,
                    density=density,
                    flow=flow,
                    queue_length=queue_length
                )
                
            except Exception as e:
                continue
    
    def _update_junction_states(self):
        """更新交叉口状态"""
        for junction_id in self.config.critical_junctions:
            try:
                phase = traci.trafficlight.getPhase(junction_id)
                phase_time = traci.trafficlight.getNextSwitch(junction_id) - traci.simulation.getTime()
                controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                
                # 计算等待车辆数
                waiting_vehicles = 0
                for lane_id in controlled_lanes:
                    try:
                        waiting_vehicles += traci.lane.getLastStepHaltingNumber(lane_id)
                    except Exception as e:
                        print(f"获取车道 {lane_id} 等待车辆数失败: {e}")
                
                self.junction_states[junction_id] = JunctionState(
                    junction_id=junction_id,
                    phase=phase,
                    phase_time=phase_time,
                    waiting_vehicles=waiting_vehicles,
                    controlled_lanes=controlled_lanes
                )
                
            except Exception as e:
                continue
    
    def _update_statistics(self):
        """更新统计信息"""
        self.total_distance_traveled = 0.0
        self.total_distance_total = 0.0
        
        for veh_state in self.vehicle_states.values():
            self.total_distance_traveled += veh_state.distance_traveled
            self.total_distance_total += veh_state.distance_total
    
    def _calculate_distance_traveled(self, veh_id: str, route: List[str], route_index: int, lane_position: float) -> float:
        """计算车辆已行驶距离"""
        distance = 0.0
        
        # 累加已完成路段的距离
        for i in range(route_index):
            edge_id = route[i]
            distance += self.edge_lengths.get(edge_id, 100)
        
        # 加上当前路段的位置
        distance += lane_position
        
        return distance
    
    def _calculate_route_length(self, route: List[str]) -> float:
        """计算路径总长度"""
        total = 0.0
        for edge_id in route:
            total += self.edge_lengths.get(edge_id, 100)
        return total
    
    def _compute_ocr(self) -> float:
        """计算OD完成率"""
        if self.total_distance_total <= 0:
            return 0.0
        
        # OCR = (到达车辆数 + 在途车辆完成度之和) / 总车辆数
        arrived_contribution = self.total_arrived
        inroute_contribution = 0.0
        
        for veh_state in self.vehicle_states.values():
            inroute_contribution += veh_state.distance_traveled / max(veh_state.distance_total, 1.0)
        
        total_vehicles = self.total_departed + len(self.vehicle_states)
        if total_vehicles <= 0:
            return 0.0
        
        ocr = (arrived_contribution + inroute_contribution) / total_vehicles
        return ocr
    
    def _compute_reward(self, actions: Dict[str, float]) -> Tuple[float, Dict]:
        """
        计算奖励
        
        Args:
            actions: 执行的动作
        
        Returns:
            reward: 总奖励
            info: 奖励分解信息
        """
        reward_info = {}
        total_reward = 0.0
        
        # 1. OCR奖励（主要目标）
        current_ocr = self._compute_ocr()
        ocr_reward = current_ocr * self.config.ocr_weight
        reward_info['ocr_reward'] = ocr_reward
        total_reward += ocr_reward
        
        # 2. 速度稳定性奖励
        speeds = [v.speed for v in self.vehicle_states.values()]
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            speed_reward = -speed_std * self.config.speed_weight
            reward_info['speed_reward'] = speed_reward
            total_reward += speed_reward
        
        # 3. 等待时间惩罚
        waiting_times = [v.waiting_time for v in self.vehicle_states.values()]
        if waiting_times:
            avg_waiting = np.mean(waiting_times)
            waiting_penalty = -avg_waiting * abs(self.config.waiting_penalty)
            reward_info['waiting_penalty'] = waiting_penalty
            total_reward += waiting_penalty
        
        # 4. 动作平滑性奖励（避免剧烈变化）
        if actions and self.last_actions:
            action_diffs = []
            for veh_id, action in actions.items():
                if veh_id in self.last_actions:
                    action_diffs.append(abs(action - self.last_actions[veh_id]))
            
            if action_diffs:
                smoothness_penalty = -np.mean(action_diffs) * 0.1
                reward_info['smoothness_penalty'] = smoothness_penalty
                total_reward += smoothness_penalty
        
        # 5. 关键路段拥堵惩罚
        congestion_penalty = 0.0
        for edge_id in self.config.critical_edges:
            edge_state = self.edge_states.get(edge_id)
            if edge_state:
                if edge_state.mean_speed < 5.0:  # 严重拥堵
                    congestion_penalty -= 0.1
                elif edge_state.mean_speed < 10.0:  # 中度拥堵
                    congestion_penalty -= 0.05
        
        reward_info['congestion_penalty'] = congestion_penalty
        total_reward += congestion_penalty
        
        reward_info['total_reward'] = total_reward
        
        return total_reward, reward_info
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        if self.current_step >= self.config.max_steps:
            return True
        
        # 检查是否所有车辆都已离开
        if traci.simulation.getMinExpectedNumber() <= 0 and self.current_step > 100:
            return True
        
        return False
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        获取观察状态
        
        Returns:
            包含各种特征的字典
        """
        # 1. 车辆特征
        vehicle_features = self._get_vehicle_features()
        
        # 2. 道路特征
        edge_features = self._get_edge_features()
        
        # 3. 交叉口特征
        junction_features = self._get_junction_features()
        
        # 4. 全局特征
        global_features = self._get_global_features()
        
        # 5. 构建图结构
        graph_data = self._build_graph()
        
        observation = {
            'vehicle_features': vehicle_features,
            'edge_features': edge_features,
            'junction_features': junction_features,
            'global_features': global_features,
            'graph': graph_data,
            'controlled_vehicles': self.controlled_vehicles.copy(),
            'cv_vehicles': self.cv_vehicles.copy()
        }
        
        # 保存到历史
        self.state_history.append(observation)
        
        return observation
    
    def _get_vehicle_features(self) -> np.ndarray:
        """获取车辆特征矩阵"""
        max_vehicles = self.config.max_vehicles
        feature_dim = 15  # 特征维度
        
        features = np.zeros((max_vehicles, feature_dim), dtype=np.float32)
        
        for i, (veh_id, veh_state) in enumerate(self.vehicle_states.items()):
            if i >= max_vehicles:
                break
            
            # 归一化特征
            speed_limit = self.edge_speed_limits.get(veh_state.edge_id, 13.89)
            
            features[i] = [
                veh_state.position[0] / 2000.0,  # x坐标归一化
                veh_state.position[1] / 400.0,   # y坐标归一化
                veh_state.speed / speed_limit,    # 速度比例
                veh_state.angle / 360.0,          # 角度归一化
                veh_state.lane_index / 3.0,       # 车道索引归一化
                veh_state.lane_position / 500.0,  # 车道位置归一化
                veh_state.route_index / max(veh_state.route_length, 1),  # 路径进度
                veh_state.completion_rate,        # 完成率
                float(veh_state.is_cv),           # 是否是CV
                min(veh_state.waiting_time / 60.0, 1.0),  # 等待时间归一化
                veh_state.acceleration / 5.0,     # 加速度归一化
                float(veh_id in self.controlled_vehicles),  # 是否被控制
                self.last_actions.get(veh_id, 0.5),  # 上一次动作
                float(veh_state.edge_id in self.config.critical_edges),  # 是否在关键路段
                1.0  # 有效标志
            ]
        
        return features
    
    def _get_edge_features(self) -> np.ndarray:
        """获取道路特征矩阵"""
        edges = list(self.edge_lengths.keys())
        feature_dim = 10
        
        features = np.zeros((len(edges), feature_dim), dtype=np.float32)
        
        for i, edge_id in enumerate(edges):
            edge_state = self.edge_states.get(edge_id)
            if edge_state:
                features[i] = [
                    edge_state.length / 1000.0,  # 长度归一化
                    edge_state.speed_limit / 20.0,  # 限速归一化
                    edge_state.lane_count / 3.0,  # 车道数归一化
                    edge_state.vehicle_count / 50.0,  # 车辆数归一化
                    edge_state.mean_speed / edge_state.speed_limit if edge_state.speed_limit > 0 else 0,  # 平均速度比例
                    edge_state.density / 100.0,  # 密度归一化
                    edge_state.flow / 1000.0,  # 流量归一化
                    edge_state.queue_length / 10.0,  # 排队长度归一化
                    float(edge_id in self.config.critical_edges),  # 是否关键路段
                    1.0  # 有效标志
                ]
        
        return features
    
    def _get_junction_features(self) -> np.ndarray:
        """获取交叉口特征矩阵"""
        feature_dim = 8
        features = np.zeros((len(self.config.critical_junctions), feature_dim), dtype=np.float32)
        
        for i, junction_id in enumerate(self.config.critical_junctions):
            junction_state = self.junction_states.get(junction_id)
            if junction_state:
                features[i] = [
                    junction_state.phase / 10.0,  # 相位归一化
                    junction_state.phase_time / 100.0,  # 相位时间归一化
                    junction_state.waiting_vehicles / 50.0,  # 等待车辆数归一化
                    len(junction_state.controlled_lanes) / 10.0,  # 控制车道数归一化
                    float(junction_id in self.config.critical_junctions),  # 是否关键交叉口
                    0.0,  # 预留
                    0.0,  # 预留
                    1.0   # 有效标志
                ]
        
        return features
    
    def _get_global_features(self) -> np.ndarray:
        """获取全局特征"""
        # 计算全局统计
        speeds = [v.speed for v in self.vehicle_states.values()]
        waiting_times = [v.waiting_time for v in self.vehicle_states.values()]
        completion_rates = [v.completion_rate for v in self.vehicle_states.values()]
        
        features = np.array([
            self.current_step / self.config.max_steps,  # 时间进度
            len(self.vehicle_states) / self.config.max_vehicles,  # 车辆数比例
            self.total_arrived / max(self.total_departed, 1),  # 到达率
            self._compute_ocr(),  # OCR
            np.mean(speeds) / 13.89 if speeds else 0,  # 平均速度比例
            np.std(speeds) / 13.89 if len(speeds) > 1 else 0,  # 速度标准差
            np.mean(waiting_times) / 60.0 if waiting_times else 0,  # 平均等待时间
            np.mean(completion_rates) if completion_rates else 0,  # 平均完成率
            len(self.cv_vehicles) / max(len(self.vehicle_states), 1),  # CV比例
            len(self.controlled_vehicles) / max(len(self.cv_vehicles), 1),  # 控制比例
        ], dtype=np.float32)
        
        return features
    
    def _build_graph(self) -> Dict[str, np.ndarray]:
        """
        构建图结构用于GNN
        
        Returns:
            node_features: 节点特征
            edge_index: 边索引
            edge_attr: 边属性
        """
        # 节点：车辆 + 道路边
        vehicle_nodes = list(self.vehicle_states.keys())
        edge_nodes = list(self.edge_lengths.keys())
        
        # 节点特征
        vehicle_feat = self._get_vehicle_features()[:len(vehicle_nodes)]
        edge_feat = self._get_edge_features()
        
        # 合并节点特征
        node_features = np.vstack([
            vehicle_feat,
            np.pad(edge_feat, ((0, 0), (0, vehicle_feat.shape[1] - edge_feat.shape[1])), 'constant')
        ]) if len(edge_feat) > 0 else vehicle_feat
        
        # 构建边：车辆-道路关系
        edge_index = []
        edge_attr = []
        
        # 车辆到道路的边
        for i, veh_id in enumerate(vehicle_nodes):
            veh_state = self.vehicle_states.get(veh_id)
            if veh_state and veh_state.edge_id in edge_nodes:
                j = len(vehicle_nodes) + edge_nodes.index(veh_state.edge_id)
                edge_index.append([i, j])
                edge_attr.append([1.0, veh_state.lane_position / 500.0])
        
        # 道路之间的连接关系（基于路网拓扑）
        for i, edge_id in enumerate(edge_nodes):
            # 这里简化处理，实际可以从路网文件获取
            for j, other_edge_id in enumerate(edge_nodes):
                if edge_id != other_edge_id:
                    # 检查是否相邻
                    if edge_id.startswith('-') and other_edge_id == edge_id[1:]:
                        edge_index.append([len(vehicle_nodes) + i, len(vehicle_nodes) + j])
                        edge_attr.append([0.5, 0.0])
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 2), dtype=np.float32)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_vehicles': len(vehicle_nodes),
            'num_edges': len(edge_nodes)
        }
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return self.config.max_vehicles
    
    def close(self):
        """关闭环境"""
        self._close_sumo()
    
    def get_ocr(self) -> float:
        """获取当前OCR"""
        return self._compute_ocr()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'step': self.current_step,
            'total_departed': self.total_departed,
            'total_arrived': self.total_arrived,
            'active_vehicles': len(self.vehicle_states),
            'cv_vehicles': len(self.cv_vehicles),
            'controlled_vehicles': len(self.controlled_vehicles),
            'ocr': self._compute_ocr(),
            'total_distance_traveled': self.total_distance_traveled,
            'total_distance_total': self.total_distance_total
        }
