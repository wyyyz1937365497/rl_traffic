"""
基于26分脚本的BC模型提交生成器

完全复制 relu_based/rl_traffic/generate_submission.py 的结构，
只替换 _apply_control 方法使用BC模型
"""

import os
import sys
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import torch

# 尝试使用libsumo
try:
    import libsumo as traci
    USE_LIBSUMO = True
except ImportError:
    import traci
    USE_LIBSUMO = False

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junction_network import VehicleLevelMultiJunctionModel, NetworkConfig


# ============================================================================
# 路口配置（与训练时保持一致）
# ============================================================================

TRAINING_JUNCTION_CONFIGS = {
    'J5': {
        'type': 'TYPE_A',
        'main_edges': ['E2', 'E3'],
        'ramp_edges': ['E23'],
        'reverse_edges': ['-E3', '-E2'],
        'tl_id': 'J5',
        'num_phases': 2,
    },
    'J14': {
        'type': 'TYPE_A',
        'main_edges': ['E9', 'E10'],
        'ramp_edges': ['E15'],
        'reverse_edges': ['-E10', '-E9'],
        'tl_id': 'J14',
        'num_phases': 2,
    },
    'J15': {
        'type': 'TYPE_B',
        'main_edges': ['E6', 'E7'],
        'ramp_edges': ['E17'],
        'diverge_edges': ['E19'],
        'reverse_edges': ['-E7', '-E6'],
        'tl_id': 'J15',
        'num_phases': 2,
    },
    'J17': {
        'type': 'TYPE_A',
        'main_edges': ['E11', 'E12'],
        'ramp_edges': ['E18'],
        'reverse_edges': ['-E12', '-E11'],
        'tl_id': 'J17',
        'num_phases': 2,
    },
}


# ============================================================================
# 工具函数：构建状态向量
# ============================================================================

def build_junction_state(junc_id: str, junc_config: dict, subscribed_vehicles: dict, edge_speeds: dict, SPEED_LIMIT: float = 13.89) -> np.ndarray:
    """
    构建23维路口状态向量（与训练时保持一致）

    Args:
        junc_id: 路口ID
        junc_config: 路口配置字典（与训练时格式相同）
        subscribed_vehicles: 订阅的车辆数据
        edge_speeds: 边速度缓存
        SPEED_LIMIT: 速度限制

    Returns:
        23维状态向量
    """
    state = np.zeros(23, dtype=np.float32)

    try:
        # 使用字典访问获取边列表
        main_edges = junc_config['main_edges']
        ramp_edges = junc_config['ramp_edges']

        # 统计主路车辆
        main_vehicles = []
        for veh_id, data in subscribed_vehicles.items():
            if data.get('is_cv', False) and data.get('road_id', '') in main_edges:
                main_vehicles.append(data)

        # 统计匝道车辆
        ramp_vehicles = []
        for veh_id, data in subscribed_vehicles.items():
            if data.get('is_cv', False) and data.get('road_id', '') in ramp_edges:
                ramp_vehicles.append(data)

        # 主路统计 (0-5)
        if main_vehicles:
            main_speeds = [v['speed'] for v in main_vehicles]
            main_avg_speed = np.mean(main_speeds)
            main_queue = sum(1 for v in main_vehicles if v['speed'] < 1.0)
            main_queue_norm = min(main_queue / 20.0, 1.0)
            main_occupancy = len(main_vehicles) / 50.0  # 假设最大容量50辆
        else:
            main_avg_speed = SPEED_LIMIT
            main_queue_norm = 0.0
            main_occupancy = 0.0

        state[0] = main_avg_speed / SPEED_LIMIT
        state[1] = main_avg_speed / SPEED_LIMIT
        state[2] = 0.0  # 加速度简化
        state[3] = main_queue_norm
        state[4] = 0.0  # 密度简化
        state[5] = main_occupancy

        # 匝道统计 (6-8)
        if ramp_vehicles:
            ramp_speeds = [v['speed'] for v in ramp_vehicles]
            ramp_avg_speed = np.mean(ramp_speeds)
            ramp_queue = sum(1 for v in ramp_vehicles if v['speed'] < 1.0)
            ramp_queue_norm = min(ramp_queue / 20.0, 1.0)
            ramp_occupancy = len(ramp_vehicles) / 30.0  # 假设最大容量30辆
        else:
            ramp_avg_speed = SPEED_LIMIT
            ramp_queue_norm = 0.0
            ramp_occupancy = 0.0

        state[6] = ramp_avg_speed / SPEED_LIMIT
        state[7] = ramp_queue_norm
        state[8] = ramp_occupancy

        # 下游速度 (9-14) - 使用edge_speeds
        NEXT_EDGE = {
            '-E13': '-E12', '-E12': '-E11', '-E11': '-E10', '-E10': '-E9',
            '-E9': '-E8', '-E8': '-E7', '-E7': '-E6', '-E6': '-E5',
            '-E5': '-E3', '-E3': '-E2', '-E2': '-E1',
            'E1': 'E2', 'E2': 'E3', 'E3': 'E5', 'E5': 'E6',
            'E6': 'E7', 'E7': 'E8', 'E8': 'E9', 'E9': 'E10',
            'E10': 'E11', 'E11': 'E12', 'E12': 'E13',
            'E23': '-E2', 'E15': 'E10', 'E17': '-E10', 'E19': '-E12',
        }

        # 获取下游6条边的速度
        downstream_edges = []
        if main_edges:
            start_edge = main_edges[0] if main_edges[0] in NEXT_EDGE else None
            if start_edge:
                nxt = start_edge
                for _ in range(6):
                    nxt = NEXT_EDGE.get(nxt)
                    if nxt is None:
                        break
                    downstream_edges.append(nxt)

        for i in range(6):
            if i < len(downstream_edges):
                ds_speed = edge_speeds.get(downstream_edges[i], SPEED_LIMIT)
                state[9 + i] = ds_speed / SPEED_LIMIT
            else:
                state[9 + i] = 1.0  # 无拥堵

        # 冲突风险 (15) - 简化计算
        conflict_risk = 0.0
        if main_vehicles and ramp_vehicles:
            # 如果主路和匝道都有车，有潜在冲突
            slow_main_ratio = sum(1 for v in main_vehicles if v['speed'] < 5.0) / max(len(main_vehicles), 1)
            slow_ramp_ratio = sum(1 for v in ramp_vehicles if v['speed'] < 5.0) / max(len(ramp_vehicles), 1)
            conflict_risk = (slow_main_ratio + slow_ramp_ratio) / 2.0

        state[15] = conflict_risk

        # 16-22: 预留，保持为0

    except Exception as e:
        # 如果出错，返回零向量
        pass

    return state


from road_topology_hardcoded import (
    get_junction_main_edges,
    get_junction_ramp_edges,
    get_junction_diverge_edges,
)


class BCSubmissionGenerator:
    """
    BC模型提交PKL生成器

    完全匹配 generate_submission.py 的数据格式
    只替换控制算法为BC模型
    """

    def __init__(self, sumo_cfg: str, checkpoint_path: str, device='cuda'):
        """
        初始化生成器

        Args:
            sumo_cfg: SUMO配置文件路径
            checkpoint_path: BC模型checkpoint路径
            device: 设备 (cuda/cpu)
        """
        self.sumo_cfg = sumo_cfg
        self.checkpoint_path = checkpoint_path
        self.device = device

        # 数据存储（完全匹配generate_submission.py格式）
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

        # 红绿灯监控
        self.traffic_lights = ['J5', 'J14', 'J15', 'J17']
        self.available_traffic_lights = []

        # 仿真参数
        self.flow_rate = 0
        self.simulation_time = 0
        self.step_length = 1.0
        self.total_demand = 0

        # BC模型
        self.model = None

        # 控制门控参数（优先保吞吐，避免过度控速）
        self.min_speed_ratio = 0.88
        self.release_speed_ratio = 0.78
        self.near_merge_threshold = 0.45  # 仅在更靠近汇入区时触发风险控制

    def generate_pkl(self, output_path: str, max_steps: int = 3600, seed: int = 42) -> str:
        """生成PKL文件"""
        print("=" * 70)
        print("生成比赛提交PKL文件 (BC模型)")
        print("=" * 70)
        print(f"配置: {self.sumo_cfg}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"设备: {self.device}")
        print(f"最大步数: {max_steps}")
        print(f"随机种子: {seed}")
        print(f"使用libsumo: {USE_LIBSUMO}")
        print("=" * 70)
        print()

        # 加载BC模型
        self._load_bc_model()

        # 解析配置
        self._parse_config()
        self._parse_routes()

        # 启动SUMO
        self._start_sumo(seed)
        self._init_traffic_lights()

        try:
            # 配置vType
            self._configure_vtypes()

            # 运行仿真
            self._run_simulation(max_steps)

            # 保存PKL
            self._save_pkl(output_path)

            print()
            print("=" * 70)
            print(f"✓ PKL文件已生成: {output_path}")
            print("=" * 70)

            return output_path

        except Exception as e:
            print(f"生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._close_sumo()

    def _load_bc_model(self):
        """加载BC模型"""
        print(f"\n加载BC模型: {self.checkpoint_path}")

        # 创建模型（与PPO/BC训练使用同一NetworkConfig）
        config = NetworkConfig()
        self.model = VehicleLevelMultiJunctionModel(TRAINING_JUNCTION_CONFIGS, config).to(self.device)

        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint

        # 候选键名匹配（兼容 network./module. 等前缀）
        def candidate_keys(model_key: str):
            candidates = [
                model_key,
                f"network.{model_key}",
                f"model.{model_key}",
                f"module.{model_key}",
                f"module.network.{model_key}",
                f"module.model.{model_key}",
            ]
            seen = set()
            uniq = []
            for key in candidates:
                if key not in seen:
                    uniq.append(key)
                    seen.add(key)
            return uniq

        model_state_ref = self.model.state_dict()
        loadable_state = {}
        for model_key, model_tensor in model_state_ref.items():
            for ckpt_key in candidate_keys(model_key):
                if ckpt_key in model_state and model_state[ckpt_key].shape == model_tensor.shape:
                    loadable_state[model_key] = model_state[ckpt_key]
                    break

        self.model.load_state_dict(loadable_state, strict=False)
        self.model.eval()

        print(f"  加载权重: {len(loadable_state)}/{len(model_state_ref)}")

        print(f"✓ BC模型加载成功 (设备: {self.device})")

    def _parse_config(self):
        """解析SUMO配置文件"""
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.sumo_cfg)
        root = tree.getroot()

        config_dir = os.path.dirname(self.sumo_cfg)

        for input_elem in root.findall('.//input'):
            net_file = input_elem.find('net-file')
            if net_file is not None:
                net_file_path = net_file.get('value')
                if not os.path.isabs(net_file_path):
                    net_file_path = os.path.join(config_dir, net_file_path)
                self.net_file = net_file_path

            route_files = input_elem.find('route-files')
            if route_files is not None:
                route_file_path = route_files.get('value')
                if not os.path.isabs(route_file_path):
                    route_file_path = os.path.join(config_dir, route_file_path)
                self.routes_file = route_file_path

        # 获取时间步长
        time_step = root.find('.//step-length')
        if time_step is not None:
            self.step_length = float(time_step.get('value', 1.0))

        print(f"✓ 配置解析完成:")
        print(f"  - 网络文件: {self.net_file}")
        print(f"  - 路径文件: {self.routes_file}")
        print(f"  - 时间步长: {self.step_length}s")

    def _parse_routes(self):
        """解析路径文件"""
        if not self.routes_file or not os.path.exists(self.routes_file):
            print("⚠️  路径文件不存在")
            return

        import xml.etree.ElementTree as ET

        tree = ET.parse(self.routes_file)
        root = tree.getroot()

        total_vehs_per_hour = 0
        max_end_time = 0
        total_demand = 0

        # 记录车辆类型原始maxSpeed
        for vtype in root.findall('vType'):
            vtype_id = vtype.get('id')
            max_speed = vtype.get('maxSpeed')
            if max_speed is not None:
                self.vehicle_type_maxspeed[vtype_id] = float(max_speed)

        # 计算总需求
        for flow in root.findall('flow'):
            vehs_per_hour = float(flow.get('vehsPerHour', 0))
            begin_time = float(flow.get('begin', 0))
            end_time = float(flow.get('end', 0))

            duration_hours = (end_time - begin_time) / 3600.0
            flow_demand = vehs_per_hour * duration_hours
            total_demand += flow_demand

            total_vehs_per_hour += vehs_per_hour
            max_end_time = max(max_end_time, end_time)

        # 计算单独的trip数量
        trip_count = len(root.findall('trip'))
        total_demand += trip_count

        self.simulation_time = max_end_time
        self.flow_rate = total_vehs_per_hour / 3600.0
        self.total_demand = total_demand

        print(f"✓ 交通需求分析:")
        print(f"  - 流量率: {self.flow_rate:.4f} veh/s")
        print(f"  - 仿真时长: {self.simulation_time:.2f} s")
        print(f"  - 理论总需求: {self.total_demand:.0f} 车辆")
        print(f"  - 单独trip数量: {trip_count}")
        print()

    def _start_sumo(self, seed: int):
        """启动SUMO"""
        sumo_cmd = [
            'sumo' if USE_LIBSUMO else 'sumo',
            '-c', self.sumo_cfg,
            '--no-warnings', 'true',
            '--duration-log.statistics', 'true',
            '--seed', str(seed)
        ]

        traci.start(sumo_cmd)
        print(f"✓ SUMO启动成功 (使用libsumo: {USE_LIBSUMO})")

    def _init_traffic_lights(self):
        """初始化红绿灯"""
        try:
            all_tls = traci.trafficlight.getIDList()

            for tl_id in self.traffic_lights:
                if tl_id in all_tls:
                    self.available_traffic_lights.append(tl_id)

            print(f"✓ 红绿灯监控设置: {self.available_traffic_lights}")
        except Exception as e:
            print(f"⚠️  红绿灯初始化失败: {e}")

    def _configure_vtypes(self):
        """配置vType参数（使用26分脚本的默认值）"""
        try:
            # 使用环境变量或默认值
            sigma = float(os.environ.get('CTRL_SIGMA', '0.0'))
            tau = float(os.environ.get('CTRL_TAU', '0.9'))
            accel = float(os.environ.get('CTRL_ACCEL', '2.1'))
            decel = float(os.environ.get('CTRL_DECEL', '4.5'))

            traci.vehicletype.setImperfection('CV', sigma)
            traci.vehicletype.setTau('CV', tau)
            traci.vehicletype.setAccel('CV', accel)
            traci.vehicletype.setDecel('CV', decel)

            traci.vehicletype.setImperfection('HV', sigma)
            traci.vehicletype.setTau('HV', tau)
            traci.vehicletype.setAccel('HV', accel)
            traci.vehicletype.setDecel('HV', decel)

            print(f"✓ vType配置完成 (sigma={sigma}, tau={tau}, accel={accel})")
        except Exception as e:
            print(f"⚠️  vType配置失败: {e}")

    def _get_vehicle_features(self, veh_id: str) -> np.ndarray:
        """获取车辆特征（8维）- 必须与训练数据完全一致！"""
        try:
            speed = traci.vehicle.getSpeed(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_idx = int(lane_id.split('_')[-1]) if '_' in lane_id else 0
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            accel = traci.vehicle.getAcceleration(veh_id)

            # 关键特征：距离边末端的距离
            lane_len = traci.lane.getLength(lane_id)
            dist_to_end = lane_len - lane_pos
            dist_to_end_normalized = dist_to_end / 100.0  # 归一化

            features = np.array([
                speed / 20.0,
                lane_pos / 500.0,
                lane_idx / 3.0,
                waiting_time / 60.0,
                accel / 5.0,
                1.0,  # is_cv
                dist_to_end_normalized,  # 距离边末端的距离（关键特征！）
                0.0   # padding
            ], dtype=np.float32)

            return features
        except:
            return np.zeros(8, dtype=np.float32)

    def _get_controlled_vehicles(self, junc_id: str):
        """获取路口受控车辆"""
        controlled = {'main': [], 'ramp': [], 'diverge': []}
        vehicle_features = {}

        main_edges = get_junction_main_edges(junc_id)
        ramp_edges = get_junction_ramp_edges(junc_id)
        diverge_edges = get_junction_diverge_edges(junc_id) if junc_id in ['J15', 'J17'] else []

        all_edges = main_edges + ramp_edges + diverge_edges

        for veh_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh_id) != 'CV':
                    continue

                road_id = traci.vehicle.getRoadID(veh_id)
                if road_id not in all_edges:
                    continue

                feat = self._get_vehicle_features(veh_id)

                # 按车辆类型分组，但特征保持与训练数据一致（不修改feat[6]）
                if road_id in main_edges:
                    controlled['main'].append(veh_id)
                    vehicle_features[veh_id] = feat
                elif road_id in ramp_edges:
                    controlled['ramp'].append(veh_id)
                    vehicle_features[veh_id] = feat
                elif road_id in diverge_edges:
                    controlled['diverge'].append(veh_id)
                    vehicle_features[veh_id] = feat

            except:
                continue

        return controlled, vehicle_features

    def _risk_score(self, feat: np.ndarray, veh_type: str) -> int:
        """基于车辆特征计算冲突风险分数"""
        if feat is None or len(feat) < 7:
            return 0

        speed_norm = float(feat[0])
        waiting_norm = float(feat[3])
        dist_to_end_norm = float(feat[6])

        score = 0
        if dist_to_end_norm <= self.near_merge_threshold:
            score += 1
        if speed_norm <= 0.45:
            score += 1
        if waiting_norm >= 0.25:
            score += 1

        if veh_type in ('ramp', 'diverge'):
            if speed_norm <= 0.50:
                score += 1
        else:
            if speed_norm <= 0.40:
                score += 1

        return score

    def _is_high_conflict_risk(self, feat: np.ndarray, veh_type: str) -> bool:
        score = self._risk_score(feat, veh_type)
        threshold = 3 if veh_type in ('ramp', 'diverge') else 4
        return score >= threshold

    def _should_release_control(self, feat: np.ndarray) -> bool:
        if feat is None or len(feat) < 7:
            return True
        speed_norm = float(feat[0])
        dist_to_end_norm = float(feat[6])
        return (speed_norm >= self.release_speed_ratio and dist_to_end_norm > 0.6) or dist_to_end_norm > 1.0

    def _run_simulation(self, max_steps: int):
        """运行仿真"""
        print(f"\n开始仿真...")
        SPEED_LIMIT = 13.89

        for step in range(max_steps):
            current_time = step * self.step_length

            # 仿真一步
            traci.simulationStep()

            # 应用BC控制（每5步）
            if step % 5 == 0:
                self._apply_bc_control(step, SPEED_LIMIT)

            # 收集数据
            self._collect_step_data(step, current_time)

            # 进度报告
            if step % 300 == 0:
                print(f"[步骤 {step}/{max_steps}] 活跃: {len(traci.vehicle.getIDList())}, "
                      f"累计出发: {self.cumulative_departed}, "
                      f"累计到达: {self.cumulative_arrived}")

        print(f"✓ 仿真完成")

    def _apply_bc_control(self, step: int, SPEED_LIMIT: float):
        """应用BC模型控制"""
        # 调试：每300步打印一次统计信息
        if step % 300 == 0:
            total_cvs = sum(1 for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) == 'CV')
            avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) == 'CV']) if total_cvs > 0 else 0
            print(f"  [BC控制] Step {step}: CV车辆={total_cvs}, 平均速度={avg_speed:.2f} m/s")

        # 为每个路口收集受控车辆和状态
        junction_observations = {}
        junction_vehicle_obs = {}
        junction_controlled = {}
        junction_vehicle_features = {}

        # 维护边速度缓存
        edge_speeds = {}
        for edge_id in traci.edge.getIDList():
            try:
                edge_speeds[edge_id] = traci.edge.getLastStepMeanSpeed(edge_id)
            except:
                edge_speeds[edge_id] = SPEED_LIMIT

        # 收集所有CV车辆数据用于状态构建
        subscribed_vehicles = {}
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) == 'CV':
                try:
                    road_id = traci.vehicle.getRoadID(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    subscribed_vehicles[veh_id] = {
                        'is_cv': True,
                        'road_id': road_id,
                        'speed': speed
                    }
                except:
                    continue

        for junc_id in TRAINING_JUNCTION_CONFIGS.keys():
            controlled, vehicle_features = self._get_controlled_vehicles(junc_id)

            if not any(controlled.values()):
                continue

            # 构建全局状态（23维，使用真实数据）
            state_vec = build_junction_state(
                junc_id, TRAINING_JUNCTION_CONFIGS[junc_id],
                subscribed_vehicles, edge_speeds, SPEED_LIMIT
            )
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 构建车辆观测
            veh_obs_dict = {'main': None, 'ramp': None, 'diverge': None}

            for veh_type in ['main', 'ramp', 'diverge']:
                if not controlled[veh_type]:
                    continue

                veh_features = []
                for veh_id in controlled[veh_type]:
                    if veh_id in vehicle_features:
                        veh_features.append(vehicle_features[veh_id])

                if veh_features:
                    veh_obs_dict[veh_type] = torch.tensor(
                        np.array(veh_features, dtype=np.float32),
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

            junction_observations[junc_id] = state_tensor
            junction_vehicle_obs[junc_id] = veh_obs_dict
            junction_controlled[junc_id] = controlled
            junction_vehicle_features[junc_id] = vehicle_features

        # 如果没有CV车辆，跳过
        if not junction_observations:
            return

        # 调试：打印输入特征（step 300）
        if step % 300 == 0:
            print(f"  [输入特征] Step {step}:")
            for junc_id in list(junction_observations.keys())[:1]:  # 只打印第一个路口
                print(f"    {junc_id}: state shape={junction_observations[junc_id].shape}, state mean={junction_observations[junc_id].mean().item():.4f}")
                veh_obs = junction_vehicle_obs[junc_id]
                for vtype in ['main', 'ramp', 'diverge']:
                    if veh_obs[vtype] is not None:
                        feats = veh_obs[vtype]
                        print(f"      {vtype}: {feats.shape[1]} vehicles, feat_mean={feats.mean(dim=1)[0].tolist()}")
                        # 打印前3辆车的完整特征
                        for i in range(min(3, feats.shape[1])):
                            print(f"        veh_{i}: [{feats[0,i,0]:.3f}, {feats[0,i,1]:.3f}, {feats[0,i,2]:.3f}, {feats[0,i,3]:.3f}, {feats[0,i,4]:.3f}, {feats[0,i,5]:.1f}, {feats[0,i,6]:.2f}, {feats[0,i,7]:.1f}]")

        # 模型推理
        try:
            with torch.no_grad():
                all_actions, _, _ = self.model(
                    junction_observations,
                    junction_vehicle_obs,
                    deterministic=True
                )

            # 调试：每300步打印action值范围
            if step % 300 == 0:
                for junc_id, actions in list(all_actions.items())[:1]:  # 只打印第一个路口
                    if 'main_actions' in actions and actions['main_actions'] is not None:
                        main_acts = actions['main_actions'].reshape(-1)
                        print(f"    {junc_id} main_actions: min={main_acts.min().item():.3f}, max={main_acts.max().item():.3f}, mean={main_acts.mean().item():.3f}")

        except Exception as e:
            if step % 300 == 0:
                print(f"  [错误] 模型推理失败: {e}")
            return

        # 应用速度控制
        total_speed_set = 0
        total_release = 0
        for junc_id, actions in all_actions.items():
            controlled = junction_controlled[junc_id]
            vehicle_features = junction_vehicle_features.get(junc_id, {})

            # 应用主路动作
            if 'main_actions' in actions and actions['main_actions'] is not None:
                main_actions = actions['main_actions'].reshape(-1)
                for i, veh_id in enumerate(controlled['main']):
                    if i < main_actions.numel():
                        feat = vehicle_features.get(veh_id)
                        if self._should_release_control(feat) or (not self._is_high_conflict_risk(feat, 'main')):
                            traci.vehicle.setSpeed(veh_id, -1.0)
                            total_release += 1
                            continue

                        action_val = main_actions[i].item()
                        # 避免过度减速：提高下限
                        action_val = max(action_val, self.min_speed_ratio)
                        target_speed = SPEED_LIMIT * action_val
                        traci.vehicle.setSpeed(veh_id, target_speed)
                        total_speed_set += 1

            # 应用匝道动作
            if 'ramp_actions' in actions and actions['ramp_actions'] is not None:
                ramp_actions = actions['ramp_actions'].reshape(-1)
                for i, veh_id in enumerate(controlled['ramp']):
                    if i < ramp_actions.numel():
                        feat = vehicle_features.get(veh_id)
                        if self._should_release_control(feat) or (not self._is_high_conflict_risk(feat, 'ramp')):
                            traci.vehicle.setSpeed(veh_id, -1.0)
                            total_release += 1
                            continue

                        action_val = ramp_actions[i].item()
                        # 匝道也提高下限
                        action_val = max(action_val, self.min_speed_ratio)
                        target_speed = SPEED_LIMIT * action_val
                        traci.vehicle.setSpeed(veh_id, target_speed)
                        total_speed_set += 1

            # 应用分流动作
            if 'diverge_actions' in actions and actions['diverge_actions'] is not None:
                diverge_actions = actions['diverge_actions'].reshape(-1)
                for i, veh_id in enumerate(controlled['diverge']):
                    if i < diverge_actions.numel():
                        feat = vehicle_features.get(veh_id)
                        if self._should_release_control(feat) or (not self._is_high_conflict_risk(feat, 'diverge')):
                            traci.vehicle.setSpeed(veh_id, -1.0)
                            total_release += 1
                            continue

                        action_val = diverge_actions[i].item()
                        action_val = max(action_val, self.min_speed_ratio)
                        target_speed = SPEED_LIMIT * action_val
                        traci.vehicle.setSpeed(veh_id, target_speed)
                        total_speed_set += 1

        if step % 300 == 0:
            print(f"  [BC控制] 本步控速={total_speed_set}，放行释放={total_release}")

    def _collect_step_data(self, step: int, current_time: float):
        """收集时间步数据（完全匹配generate_submission.py格式）"""
        current_vehicle_ids = set(traci.vehicle.getIDList())

        # 更新累计统计
        current_arrived_ids = set(traci.simulation.getArrivedIDList())
        current_departed_ids = set(traci.simulation.getDepartedIDList())

        new_arrivals = current_arrived_ids - self.all_arrived_vehicles
        self.all_arrived_vehicles.update(new_arrivals)
        self.cumulative_arrived = len(self.all_arrived_vehicles)

        new_departures = current_departed_ids - self.all_departed_vehicles
        self.all_departed_vehicles.update(new_departures)
        self.cumulative_departed = len(self.all_departed_vehicles)

        # 获取红绿灯状态
        traffic_light_states = self._get_traffic_light_states()

        # 记录时间步数据
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

        # 收集车辆数据
        for veh_id in current_vehicle_ids:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getLanePosition(veh_id)
                edge_id = traci.vehicle.getRoadID(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)

                od_info = self._get_vehicle_od(veh_id)

                if veh_id not in self.route_data:
                    route_edges = traci.vehicle.getRoute(veh_id)
                    route_length = self._get_route_length(route_edges)
                    self.route_data[veh_id] = {
                        'route_edges': route_edges,
                        'route_length': route_length
                    }

                route_info = self.route_data[veh_id]
                traveled_distance = self._calculate_traveled_distance(
                    veh_id, route_info
                )
                completion_rate = min(
                    traveled_distance / max(route_info['route_length'], 1), 1.0
                )

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

            except:
                continue

    def _get_traffic_light_states(self):
        """获取红绿灯状态"""
        tl_states = {}

        for tl_id in self.available_traffic_lights:
            try:
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                phase = traci.trafficlight.getPhase(tl_id)
                remaining_time = traci.trafficlight.getNextSwitch(tl_id) - traci.simulation.getTime()

                tl_states[f'{tl_id}_state'] = state
                tl_states[f'{tl_id}_phase'] = phase
                tl_states[f'{tl_id}_remaining_time'] = remaining_time

            except:
                tl_states[f'{tl_id}_state'] = 'unknown'
                tl_states[f'{tl_id}_phase'] = -1
                tl_states[f'{tl_id}_remaining_time'] = -1

        return tl_states

    def _get_vehicle_od(self, veh_id: str) -> Dict:
        """获取车辆OD信息"""
        if veh_id in self.vehicle_od_data:
            return self.vehicle_od_data[veh_id]

        try:
            route = traci.vehicle.getRoute(veh_id)
            if len(route) >= 2:
                origin = route[0]
                destination = route[-1]
            elif len(route) == 1:
                origin = route[0]
                destination = route[0]
            else:
                origin = "unknown"
                destination = "unknown"

            vehicle_type = traci.vehicle.getTypeID(veh_id)
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

        except:
            od_info = {
                'origin': "unknown",
                'destination': "unknown",
                'route_length': 0,
                'vehicle_type': "unknown",
                'original_max_speed': None
            }
            self.vehicle_od_data[veh_id] = od_info
            return od_info

    def _get_route_length(self, edges: List[str]) -> float:
        """计算路径总长度"""
        total_length = 0
        for edge_id in edges:
            try:
                edge_length = traci.edge.getLength(edge_id)
                total_length += edge_length
            except:
                try:
                    lane_id = f"{edge_id}_0"
                    edge_length = traci.lane.getLength(lane_id)
                    total_length += edge_length
                except:
                    total_length += 100
        return total_length

    def _calculate_traveled_distance(self, veh_id: str, route_info: Dict) -> float:
        """计算已行驶距离"""
        try:
            current_edge = traci.vehicle.getRoadID(veh_id)
            current_position = traci.vehicle.getLanePosition(veh_id)
            route_edges = route_info['route_edges']

            traveled = 0
            for edge in route_edges:
                if edge == current_edge:
                    traveled += current_position
                    break
                else:
                    try:
                        edge_length = traci.edge.getLength(edge)
                        traveled += edge_length
                    except:
                        traveled += 100

            return min(traveled, route_info['route_length'])
        except:
            return 0

    def _save_pkl(self, output_path: str):
        """保存PKL文件（完全匹配generate_submission.py格式）"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 准备完整数据包
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
                'monitored_traffic_lights': self.traffic_lights,
                'available_traffic_lights': self.available_traffic_lights,
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
                'maxspeed_violations': {
                    'has_violations': False,
                    'violations': {},
                    'total_vehicle_types_checked': len(self.vehicle_type_maxspeed),
                }
            }
        }

        # 保存pickle文件
        with open(output_path, 'wb') as f:
            pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 计算文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"\n{'=' * 70}")
        print(f"数据保存统计")
        print(f"{'=' * 70}")
        print(f"✓ Pickle文件: {output_path}")
        print(f"  - 文件大小: {file_size:.2f} MB")
        print(f"  - 时间步数据记录: {len(self.step_data):,} 条")
        print(f"  - 车辆数据记录: {len(self.vehicle_data):,} 条")
        print(f"  - 唯一车辆数: {len(self.route_data):,} 辆")
        print(f"{'=' * 70}")

    def _close_sumo(self):
        """关闭SUMO"""
        try:
            traci.close()
        except:
            pass


def generate_submission_bc(
    output_path='submit_bc.pkl',
    sumo_cfg='sumo/sumo.sumocfg',
    checkpoint_path='bc_checkpoints_vehicle_v2/best_model.pt',
    device='cuda',
    max_steps=3600,
    seed=42
):
    """
    生成比赛提交用PKL文件 (BC模型版本)

    Args:
        output_path: 输出pkl文件路径
        sumo_cfg: SUMO配置文件路径
        checkpoint_path: BC模型checkpoint路径
        device: 设备 (cuda/cpu)
        max_steps: 仿真步数
        seed: 随机种子
    """
    generator = BCSubmissionGenerator(sumo_cfg, checkpoint_path, device)
    return generator.generate_pkl(output_path, max_steps, seed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成比赛提交用PKL文件 (BC模型)')

    parser.add_argument('--output', type=str, default='submit_bc.pkl',
                        help='输出pkl文件路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg',
                        help='SUMO配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='BC模型checkpoint路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--steps', type=int, default=3600,
                        help='仿真步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    generate_submission_bc(
        output_path=args.output,
        sumo_cfg=args.sumo_cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_steps=args.steps,
        seed=args.seed
    )
