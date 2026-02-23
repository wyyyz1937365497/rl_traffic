"""
从训练好的车辆级模型checkpoint生成比赛提交用的PKL文件

使用训练好的车辆级BC模型进行仿真，并生成符合比赛格式的submit.pkl文件
"""
import os
import sys
import pickle
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# 使用traci（兼容性更好）
import traci

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NetworkConfig
from junction_agent import JUNCTION_CONFIGS
from junction_network import VehicleLevelMultiJunctionModel
from road_topology_hardcoded import (
    JUNCTION_CONFIG,
    EDGE_TOPOLOGY,
    is_ramp_edge,
    get_downstream_edges,
    get_upstream_edges,
    get_junction_main_edges,
    get_junction_ramp_edges,
    get_junction_diverge_edges,
    get_junction_edges
)


class VehicleLevelSubmissionGenerator:
    """从车辆级模型生成比赛提交PKL文件"""

    def __init__(self, checkpoint_path: str, sumo_cfg: str, device: str = 'cpu'):
        self.checkpoint_path = checkpoint_path
        self.sumo_cfg = sumo_cfg
        self.device = device

        # 数据存储（完全匹配比赛格式）
        self.vehicle_data = []
        self.step_data = []
        self.route_data = {}
        self.vehicle_od_data = {}
        self.vehicle_type_maxspeed = {}

        # 累计统计
        self.cumulative_departed = 0
        self.cumulative_arrived = 0
        self.all_departed_vehicles = set()
        self.all_arrived_vehicles = set()

        # 仿真参数
        self.flow_rate = 0
        self.simulation_time = 0
        self.step_length = 1.0
        self.total_demand = 0

        # 模型
        self.model = None

        # 全局CV车辆分配
        self.global_cv_assignment = {}  # {veh_id: junc_id}

    def load_model(self):
        """加载训练好的车辆级模型"""
        print(f"加载模型: {self.checkpoint_path}")

        # 创建车辆级模型（使用与训练时相同的配置：gnn_hidden_dim=64）
        from config import NetworkConfig
        config = NetworkConfig()
        config.gnn_hidden_dim = 64  # 训练时使用的gnn_hidden_dim

        self.model = VehicleLevelMultiJunctionModel(JUNCTION_CONFIGS, config).to(self.device)

        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint

        # 去除 'network.' 前缀（训练时使用了VehicleLevelBCModel包装器）
        new_model_state = {}
        for key, value in model_state.items():
            if key.startswith('network.'):
                new_key = key[8:]  # 去掉 'network.' 前缀
                new_model_state[new_key] = value
            else:
                new_model_state[key] = value

        self.model.load_state_dict(new_model_state)
        self.model.eval()

        print(f"✓ 模型加载成功 (设备: {self.device})")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  GNN hidden dim: {config.gnn_hidden_dim}")

    def initialize_simulation(self):
        """初始化SUMO仿真"""
        print(f"\n初始化SUMO仿真...")
        print(f"  配置文件: {self.sumo_cfg}")

        # 使用libsumo以获得更好的性能
        try:
            import libsumo as traci_local
            print(f"  使用 libsumo")
        except ImportError:
            import traci as traci_local
            print(f"  使用 traci")

        sumo_cmd = [
            'sumo',
            '-c', self.sumo_cfg,
            '--no-warnings', 'true',
            '--duration-log.statistics', 'true',
            '--seed', '42'
        ]

        traci_local.start(sumo_cmd)
        self.traci = traci_local

        print(f"✓ SUMO仿真已启动")

        # 解析routes文件获取车辆类型配置
        self.parse_routes_for_maxspeed()

        # 获取仿真参数
        end_time = self.traci.simulation.getEndTime()
        self.step_length = self.traci.simulation.getDeltaT()

        # 如果end_time是-1或无效值，设置默认值
        if end_time <= 0:
            self.simulation_time = 3600
            print(f"  警告: 无法获取仿真时长，使用默认值 3600s")
        else:
            self.simulation_time = end_time

        print(f"  仿真时长: {self.simulation_time}s")
        print(f"  步长: {self.step_length}s")

    def parse_routes_for_maxspeed(self):
        """解析routes文件获取车辆类型的maxSpeed配置"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.sumo_cfg)
            root = tree.getroot()

            # 获取routes文件路径
            config_dir = os.path.dirname(self.sumo_cfg)
            routes_file = None

            for input_elem in root.findall('.//input'):
                route_files = input_elem.find('route-files')
                if route_files is not None:
                    route_file_path = route_files.get('value')
                    if not os.path.isabs(route_file_path):
                        route_file_path = os.path.join(config_dir, route_file_path)
                    routes_file = route_file_path
                    break

            if routes_file and os.path.exists(routes_file):
                tree = ET.parse(routes_file)
                root = tree.getroot()

                # 记录所有车辆类型的maxSpeed
                for vtype in root.findall('vType'):
                    vtype_id = vtype.get('id')
                    max_speed = vtype.get('maxSpeed')
                    if max_speed is not None:
                        self.vehicle_type_maxspeed[vtype_id] = float(max_speed)
                        print(f"  车辆类型 {vtype_id}: maxSpeed = {max_speed} m/s")

                # 同时计算总需求
                total_demand = 0
                for flow in root.findall('flow'):
                    vehs_per_hour = float(flow.get('vehsPerHour', 0))
                    begin_time = float(flow.get('begin', 0))
                    end_time = float(flow.get('end', 0))
                    duration_hours = (end_time - begin_time) / 3600.0
                    flow_demand = vehs_per_hour * duration_hours
                    total_demand += flow_demand

                trip_count = len(root.findall('trip'))
                total_demand += trip_count

                self.total_demand = total_demand
                print(f"  理论总需求: {total_demand:.0f} 辆车")

        except Exception as e:
            print(f"  警告: 无法解析routes文件: {e}")

    def get_junction_for_vehicle(self, veh_id: str) -> str:
        """获取车辆所属的路口"""
        if veh_id in self.global_cv_assignment:
            return self.global_cv_assignment[veh_id]

        # 根据车辆位置分配路口
        try:
            road_id = self.traci.vehicle.getRoadID(veh_id)
            route_index = self.traci.vehicle.getRouteIndex(veh_id)

            # 查找该边所属的路口
            for junc_id, junc_config in JUNCTION_CONFIGS.items():
                all_edges = (junc_config['main_edges'] +
                            junc_config['ramp_edges'] +
                            junc_config['reverse_edges'])

                if junc_config['type'] == 'TYPE_B':
                    all_edges.extend(junc_config['diverge_edges'])

                if road_id in all_edges:
                    self.global_cv_assignment[veh_id] = junc_id
                    return junc_id

        except Exception as e:
            pass

        # 默认分配
        self.global_cv_assignment[veh_id] = 'J5'
        return 'J5'

    def get_vehicle_features(self, veh_id: str, junc_id: str, junc_config: Dict) -> np.ndarray:
        """获取车辆特征（8维）"""
        try:
            speed = self.traci.vehicle.getSpeed(veh_id)
            max_speed = self.traci.vehicle.getAllowedSpeed(veh_id)

            lane_id = self.traci.vehicle.getLaneID(veh_id)
            lane_pos = self.traci.vehicle.getLanePosition(veh_id)
            lane_idx = int(lane_id.split('_')[-1]) if '_' in lane_id else 0

            accel = self.traci.vehicle.getAcceleration(veh_id)
            waiting_time = 0.0

            road_id = self.traci.vehicle.getRoadID(veh_id)

            # 确定车辆类型
            if road_id in junc_config['main_edges'] or road_id in junc_config['reverse_edges']:
                veh_type = 0  # main
            elif road_id in junc_config['ramp_edges']:
                veh_type = 1  # ramp
            elif junc_config['type'] == 'TYPE_B' and road_id in junc_config['diverge_edges']:
                veh_type = 2  # diverge
            else:
                veh_type = 0

            features = np.array([
                speed / 20.0,
                lane_pos / 500.0,
                lane_idx / 3.0,
                waiting_time / 60.0,
                accel / 5.0,
                1.0,  # is_cv
                veh_type / 2.0,  # vehicle_type
                0.0   # padding
            ], dtype=np.float32)

            return features

        except Exception as e:
            return np.zeros(8, dtype=np.float32)

    def get_controlled_vehicles(self, junc_id: str, junc_config: Dict) -> Dict[str, List[str]]:
        """获取路口受控车辆"""
        current_vehicles = self.traci.vehicle.getIDList()
        controlled = {'main': [], 'ramp': [], 'diverge': []}

        for veh_id in current_vehicles:
            try:
                if self.traci.vehicle.getTypeID(veh_id) != 'CV':
                    continue

                road_id = self.traci.vehicle.getRoadID(veh_id)

                if road_id in junc_config['main_edges'] or road_id in junc_config['reverse_edges']:
                    controlled['main'].append(veh_id)
                elif road_id in junc_config['ramp_edges']:
                    controlled['ramp'].append(veh_id)
                elif junc_config['type'] == 'TYPE_B' and road_id in junc_config['diverge_edges']:
                    controlled['diverge'].append(veh_id)

            except:
                continue

        return controlled

    def apply_model_control(self, step: int):
        """应用模型控制（完全匹配26分脚本的控制逻辑）"""
        # === Phase 1: vType参数配置（仅执行一次）===
        if not hasattr(self, '_vtype_configured'):
            self._vtype_configured = False
            self._controlled_cvs = set()

        if not self._vtype_configured:
            self._configure_vtypes()
            self._vtype_configured = True

        # === Phase 2: 模型控制（每5步）===
        if step % 5 == 0:
            self._apply_bc_control(step)

    def _configure_vtypes(self):
        """配置CV和HV的vType参数（完全匹配26分脚本）"""
        import os
        try:
            # --- CV 参数（使用环境变量）---
            sigma   = float(os.environ.get('CTRL_SIGMA',       '0.0'))
            tau     = float(os.environ.get('CTRL_TAU',         '0.9'))
            accel_v = os.environ.get('CTRL_ACCEL',      '2.1')
            sfactor = os.environ.get('CTRL_SPEEDFACTOR','1.0')
            sdev    = os.environ.get('CTRL_SPEEDDEV',   '0.0')
            decel   = os.environ.get('CTRL_DECEL',      '')
            mingap  = os.environ.get('CTRL_MINGAP',     '')

            self.traci.vehicletype.setImperfection('CV', sigma)
            self.traci.vehicletype.setTau('CV', tau)
            if accel_v:
                self.traci.vehicletype.setAccel('CV', float(accel_v))
            if sfactor:
                self.traci.vehicletype.setSpeedFactor('CV', float(sfactor))
            if sdev:
                self.traci.vehicletype.setSpeedDeviation('CV', float(sdev))
            if decel:
                self.traci.vehicletype.setDecel('CV', float(decel))
            if mingap:
                self.traci.vehicletype.setMinGap('CV', float(mingap))

            # --- HV 参数 ---
            hv_sigma = os.environ.get('CTRL_HV_SIGMA', '0.0')
            if hv_sigma:
                self.traci.vehicletype.setImperfection('HV', float(hv_sigma))
            hv_tau = os.environ.get('CTRL_HV_TAU', '0.9')
            if hv_tau:
                self.traci.vehicletype.setTau('HV', float(hv_tau))
            hv_accel = os.environ.get('CTRL_HV_ACCEL', '2.1')
            if hv_accel:
                self.traci.vehicletype.setAccel('HV', float(hv_accel))
            hv_sdev = os.environ.get('CTRL_HV_SPEEDDEV', '0.0')
            if hv_sdev:
                self.traci.vehicletype.setSpeedDeviation('HV', float(hv_sdev))
            hv_decel = os.environ.get('CTRL_HV_DECEL', '')
            if hv_decel:
                self.traci.vehicletype.setDecel('HV', float(hv_decel))
            hv_mingap = os.environ.get('CTRL_HV_MINGAP', '')
            if hv_mingap:
                self.traci.vehicletype.setMinGap('HV', float(hv_mingap))
        except Exception as e:
            pass

    def _apply_bc_control(self, step: int):
        """应用BC模型控制"""
        junction_observations = {}  # {junc_id: state_tensor}
        junction_vehicle_obs = {}  # {junc_id: {'main': [N, 8], 'ramp': [M, 8], 'diverge': [K, 8]}}
        junction_controlled = {}  # {junc_id: {'main': [veh_ids], 'ramp': [veh_ids], 'diverge': [veh_ids]}}

        for junc_id, junc_config in JUNCTION_CONFIGS.items():
            # 获取受控车辆
            controlled = self.get_controlled_vehicles(junc_id, junc_config)

            if not any(controlled.values()):
                continue

            # 获取全局状态（23维，这里简化为全零）
            state_tensor = torch.zeros(1, 23, dtype=torch.float32).to(self.device)

            # 获取车辆特征
            veh_obs_dict = {'main': [], 'ramp': [], 'diverge': []}

            for veh_type in ['main', 'ramp', 'diverge']:
                if not controlled[veh_type]:
                    veh_obs_dict[veh_type] = None
                    continue

                veh_features = []
                for veh_id in controlled[veh_type]:
                    feat = self.get_vehicle_features(veh_id, junc_id, junc_config)
                    veh_features.append(feat)

                if veh_features:
                    veh_obs_dict[veh_type] = torch.tensor(
                        np.array(veh_features, dtype=np.float32),
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)  # [1, N, 8]
                else:
                    veh_obs_dict[veh_type] = None

            # 保存观测
            junction_observations[junc_id] = state_tensor
            junction_vehicle_obs[junc_id] = veh_obs_dict
            junction_controlled[junc_id] = controlled

        if not junction_observations:
            return

        # 模型推理（所有路口一起）
        with torch.no_grad():
            all_actions, _, _ = self.model.network(
                junction_observations,
                junction_vehicle_obs,
                deterministic=True
            )

        # 应用动作
        for junc_id, actions in all_actions.items():
            controlled = junction_controlled[junc_id]

            # 应用主路动作
            if 'main_actions' in actions and actions['main_actions'] is not None:
                main_actions = actions['main_actions'][0]  # [N]

                for i, veh_id in enumerate(controlled['main']):
                    if i < len(main_actions):
                        action_val = main_actions[i].item()
                        target_speed = 13.89 * max(action_val, 0.1)
                        self.traci.vehicle.setSpeed(veh_id, target_speed)

            # 应用匝道动作
            if 'ramp_actions' in actions and actions['ramp_actions'] is not None:
                ramp_actions = actions['ramp_actions'][0]

                for i, veh_id in enumerate(controlled['ramp']):
                    if i < len(ramp_actions):
                        action_val = ramp_actions[i].item()
                        target_speed = 13.89 * max(action_val, 0.1)
                        self.traci.vehicle.setSpeed(veh_id, target_speed)

            # 应用分流动作
            if 'diverge_actions' in actions and actions['diverge_actions'] is not None:
                diverge_actions = actions['diverge_actions'][0]

                for i, veh_id in enumerate(controlled['diverge']):
                    if i < len(diverge_actions):
                        action_val = diverge_actions[i].item()
                        target_speed = 13.89 * max(action_val, 0.1)
                        self.traci.vehicle.setSpeed(veh_id, target_speed)

    def get_traffic_light_states(self):
        """获取红绿灯状态"""
        tl_states = {}
        traffic_lights = ['J5', 'J14', 'J15', 'J17']

        for tl_id in traffic_lights:
            try:
                state = self.traci.trafficlight.getRedYellowGreenState(tl_id)
                phase = self.traci.trafficlight.getPhase(tl_id)
                remaining_time = self.traci.trafficlight.getNextSwitch(tl_id) - self.traci.simulation.getTime()

                tl_states[f'{tl_id}_state'] = state
                tl_states[f'{tl_id}_phase'] = phase
                tl_states[f'{tl_id}_remaining_time'] = remaining_time

            except Exception as e:
                tl_states[f'{tl_id}_state'] = 'unknown'
                tl_states[f'{tl_id}_phase'] = -1
                tl_states[f'{tl_id}_remaining_time'] = -1

        return tl_states

    def get_vehicle_od(self, veh_id):
        """获取车辆OD信息和车辆类型的原始maxSpeed配置"""
        if veh_id in self.vehicle_od_data:
            return self.vehicle_od_data[veh_id]

        try:
            route = self.traci.vehicle.getRoute(veh_id)
            if len(route) >= 2:
                origin = route[0]
                destination = route[-1]
            elif len(route) == 1:
                origin = route[0]
                destination = route[0]
            else:
                origin = "unknown"
                destination = "unknown"

            # 获取车辆类型
            vehicle_type = self.traci.vehicle.getTypeID(veh_id)

            # 获取该车辆类型的原始maxSpeed配置
            original_max_speed = self.vehicle_type_maxspeed.get(vehicle_type, None)

            od_info = {
                'origin': origin,
                'destination': destination,
                'route_length': len(route),
                'vehicle_type': vehicle_type,
                'original_max_speed': original_max_speed
            }

            self.vehicle_od_data[veh_id] = od_info
            return od_info

        except Exception as e:
            od_info = {
                'origin': "unknown",
                'destination': "unknown",
                'route_length': 0,
                'vehicle_type': "unknown",
                'original_max_speed': None
            }
            self.vehicle_od_data[veh_id] = od_info
            return od_info

    def get_route_length(self, edges):
        """计算路径总长度"""
        total_length = 0
        for edge_id in edges:
            try:
                edge_length = self.traci.edge.getLength(edge_id)
                total_length += edge_length
            except:
                try:
                    lane_id = f"{edge_id}_0"
                    edge_length = self.traci.lane.getLength(lane_id)
                    total_length += edge_length
                except:
                    total_length += 100
        return total_length

    def calculate_traveled_distance(self, veh_id, route_info):
        """计算车辆已行驶距离"""
        try:
            current_edge = self.traci.vehicle.getRoadID(veh_id)
            current_position = self.traci.vehicle.getLanePosition(veh_id)
            route_edges = route_info['route_edges']

            traveled = 0
            for edge in route_edges:
                if edge == current_edge:
                    traveled += current_position
                    break
                else:
                    try:
                        edge_length = self.traci.edge.getLength(edge)
                        traveled += edge_length
                    except:
                        traveled += 100

            return min(traveled, route_info['route_length'])
        except:
            return 0

    def record_step_data(self, step: int):
        """记录步数据（完全匹配26分脚本格式）"""
        current_time = step * self.step_length

        # 获取当前所有车辆
        current_vehicle_ids = set(self.traci.vehicle.getIDList())

        # 更新累计统计
        current_arrived_ids = set(self.traci.simulation.getArrivedIDList())
        current_departed_ids = set(self.traci.simulation.getDepartedIDList())

        new_arrivals = current_arrived_ids - self.all_arrived_vehicles
        self.all_arrived_vehicles.update(new_arrivals)
        self.cumulative_arrived = len(self.all_arrived_vehicles)

        new_departures = current_departed_ids - self.all_departed_vehicles
        self.all_departed_vehicles.update(new_departures)
        self.cumulative_departed = len(self.all_departed_vehicles)

        # 获取红绿灯状态
        traffic_light_states = self.get_traffic_light_states()

        # 记录时间步级数据
        step_record = {
            'step': step,
            'time': current_time,
            'active_vehicles': len(current_vehicle_ids),
            'arrived_vehicles': self.cumulative_arrived,
            'departed_vehicles': self.cumulative_departed,
            'current_arrivals': len(new_arrivals),
            'current_departures': len(new_departures)
        }
        step_record.update(traffic_light_states)
        self.step_data.append(step_record)

        # 收集车辆级数据（所有车辆，不只是CV）
        for veh_id in current_vehicle_ids:
            try:
                speed = self.traci.vehicle.getSpeed(veh_id)
                position = self.traci.vehicle.getLanePosition(veh_id)
                edge_id = self.traci.vehicle.getRoadID(veh_id)
                route_index = self.traci.vehicle.getRouteIndex(veh_id)

                od_info = self.get_vehicle_od(veh_id)

                if veh_id not in self.route_data:
                    route_edges = self.traci.vehicle.getRoute(veh_id)
                    route_length = self.get_route_length(route_edges)
                    self.route_data[veh_id] = {
                        'route_edges': route_edges,
                        'route_length': route_length
                    }

                route_info = self.route_data[veh_id]
                traveled_distance = self.calculate_traveled_distance(veh_id, route_info)
                completion_rate = min(traveled_distance / max(route_info['route_length'], 1), 1.0)

                vehicle_record = {
                    'step': step,
                    'time': current_time,
                    'vehicle_id': veh_id,
                    'speed': speed,
                    'position': position,
                    'edge_id': edge_id,
                    'route_index': route_index,
                    'traveled_distance': traveled_distance,
                    'route_length': route_info['route_length'],
                    'completion_rate': completion_rate,
                    'origin': od_info['origin'],
                    'destination': od_info['destination'],
                    'route_edges_count': od_info['route_length'],
                    'max_speed': od_info['original_max_speed'],
                    'vehicle_type': od_info['vehicle_type']
                }
                self.vehicle_data.append(vehicle_record)

            except Exception as e:
                continue

    def run_simulation(self):
        """运行完整仿真（完全匹配26分脚本的执行顺序）"""
        print(f"\n开始仿真...")
        print(f"  模型: 车辆级BC网络")
        print(f"  设备: {self.device}")

        for step in range(int(self.simulation_time)):
            # 1. 先执行仿真步（关键：必须在控制之前）
            self.traci.simulationStep()

            # 2. 然后应用控制
            self.apply_model_control(step)

            # 3. 然后收集数据
            self.record_step_data(step)

            # 打印进度
            if step % 300 == 0:
                print(f"  进度: {step}/{self.simulation_time} ({100*step/self.simulation_time:.1f}%) | "
                      f"离开: {self.cumulative_departed} | 到达: {self.cumulative_arrived}")

            # 检查是否结束
            if self.traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                break

        print(f"\n✓ 仿真完成!")
        print(f"  总离开: {self.cumulative_departed}")
        print(f"  总到达: {self.cumulative_arrived}")
        print(f"  OCR: {self.cumulative_arrived/max(self.cumulative_departed, 1):.4f}")

    def save_submission(self, output_path: str):
        """保存提交文件（完全匹配26分脚本格式）"""
        print(f"\n保存提交文件...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 准备完整数据包（完全匹配26分脚本的格式）
        data_package = {
            # 仿真参数
            'parameters': {
                'flow_rate': self.flow_rate,
                'simulation_time': self.simulation_time,
                'step_length': self.step_length,
                'total_steps': len(self.step_data),
                'total_demand': self.total_demand,
                'final_departed': self.cumulative_departed,
                'final_arrived': self.cumulative_arrived,
                'unique_vehicles': len(self.route_data),
                'monitored_traffic_lights': ['J5', 'J14', 'J15', 'J17'],
                'available_traffic_lights': ['J5', 'J14', 'J15', 'J17'],
                'collection_timestamp': timestamp,
                'vehicle_type_maxspeed': self.vehicle_type_maxspeed
            },
            # 原始数据
            'step_data': self.step_data,
            'vehicle_data': self.vehicle_data,
            'route_data': self.route_data,
            'vehicle_od_data': self.vehicle_od_data,
            # 累计统计
            'statistics': {
                'all_departed_vehicles': list(self.all_departed_vehicles),
                'all_arrived_vehicles': list(self.all_arrived_vehicles),
                'cumulative_departed': self.cumulative_departed,
                'cumulative_arrived': self.cumulative_arrived,
                'maxspeed_violations': []  # 暂时不检测
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"✓ 提交文件已保存: {output_path}")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  总步数: {len(self.step_data)}")
        print(f"  总车辆记录: {len(self.vehicle_data)}")
        print(f"  唯一车辆: {len(self.route_data)}")

    def generate(self, output_path: str = 'submit_vehicle_bc.pkl'):
        """生成完整的提交文件"""
        print("=" * 70)
        print("车辆级BC模型提交生成器")
        print("=" * 70)

        # 加载模型
        self.load_model()

        # 初始化仿真
        self.initialize_simulation()

        # 运行仿真
        self.run_simulation()

        # 关闭仿真
        self.traci.close()

        # 保存提交文件
        self.save_submission(output_path)

        print("\n" + "=" * 70)
        print("生成完成!")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='从车辆级BC模型生成提交文件')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg',
                        help='SUMO配置文件')
    parser.add_argument('--output', type=str, default='submit_vehicle_bc.pkl',
                        help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')

    args = parser.parse_args()

    # 创建生成器
    generator = VehicleLevelSubmissionGenerator(
        checkpoint_path=args.checkpoint,
        sumo_cfg=args.sumo_cfg,
        device=args.device
    )

    # 生成提交文件
    generator.generate(args.output)


if __name__ == '__main__':
    main()
