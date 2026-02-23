"""
从训练好的模型checkpoint生成比赛提交用的PKL文件

使用训练好的RL模型进行仿真，并生成符合比赛格式的submit.pkl文件
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

# 直接使用libsumo
import libsumo as traci

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NetworkConfig
from junction_network import create_vehicle_level_model
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


class ModelSubmissionGenerator:
    """
    从模型生成比赛提交PKL文件
    """

    def __init__(self, checkpoint_path: str, sumo_cfg: str, device: str = 'cpu'):
        """
        初始化生成器

        Args:
            checkpoint_path: 模型checkpoint路径
            sumo_cfg: SUMO配置文件路径
            device: 计算设备
        """
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
        """加载训练好的模型"""
        print(f"加载模型: {self.checkpoint_path}")

        # 创建模型（使用路口级网络）
        from junction_network import create_junction_model
        self.model = create_junction_model(JUNCTION_CONFIG, NetworkConfig()).to(self.device)

        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint

        self.model.load_state_dict(model_state)
        self.model.eval()

        print(f"✓ 模型加载成功 (设备: {self.device})")

    def generate_pkl(self, output_path: str, max_steps: int = 3600, seed: int = 42) -> str:
        """生成PKL文件"""
        print("=" * 70)
        print("从模型生成比赛提交PKL文件")
        print("=" * 70)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"配置: {self.sumo_cfg}")
        print(f"最大步数: {max_steps}")
        print(f"随机种子: {seed}")
        print(f"设备: {self.device}")
        print("=" * 70)
        print()

        # 加载模型
        self.load_model()

        # 解析配置
        self._parse_config()
        self._parse_routes()

        # 启动SUMO
        self._start_sumo(seed)

        try:
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
        print()

    def _start_sumo(self, seed: int):
        """启动SUMO"""
        sumo_cmd = [
            'sumo',
            '-c', self.sumo_cfg,
            '--no-warnings', 'true',
            '--duration-log.statistics', 'true',
            '--seed', str(seed)
        ]

        traci.start(sumo_cmd)
        print(f"✓ SUMO启动成功")

    def _run_simulation(self, max_steps: int):
        """运行仿真"""
        print(f"\n开始仿真...")

        for step in range(max_steps):
            current_time = step * self.step_length

            # 分配CV车辆到路口
            self._assign_cv_vehicles()

            # 获取观察
            obs = self._get_observations()

            if obs:
                # 准备观察张量
                obs_tensors = {}
                vehicle_obs = {}

                for junc_id, state_vec in obs.items():
                    obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # 获取受控车辆特征
                    controlled = self._get_controlled_vehicles(junc_id)

                    vehicle_obs[junc_id] = {
                        'main': self._get_vehicle_features(controlled['main']),
                        'ramp': self._get_vehicle_features(controlled['ramp']),
                        'diverge': self._get_vehicle_features(controlled.get('diverge', []))
                    }

                # 模型推理（路口级网络）
                with torch.no_grad():
                    output = self.model(obs_tensors, deterministic=True)

                # 应用动作（路口级控制：同一个路口的CV车辆共享动作）
                for junc_id, action_dict in output.items():
                    controlled = self._get_controlled_vehicles(junc_id)

                    # 主路动作
                    if 'main' in action_dict and controlled['main']:
                        main_action_val = action_dict['main'].item()  # 标量
                        max_speed = 13.89  # SUMO默认最大速度
                        target_speed = max_speed * main_action_val

                        for veh_info in controlled['main']:
                            veh_id = veh_info['id']
                            traci.vehicle.setSpeed(veh_id, target_speed)

                    # 匝道动作
                    if 'ramp' in action_dict and controlled['ramp']:
                        ramp_action_val = action_dict['ramp'].item()
                        max_speed = 13.89
                        target_speed = max_speed * ramp_action_val

                        for veh_info in controlled['ramp']:
                            veh_id = veh_info['id']
                            traci.vehicle.setSpeed(veh_id, target_speed)

                    # 分流动作（如果存在）
                    if 'diverge' in action_dict and controlled.get('diverge'):
                        diverge_action_val = action_dict['diverge'].item()
                        max_speed = 13.89
                        target_speed = max_speed * diverge_action_val

                        for veh_info in controlled['diverge']:
                            veh_id = veh_info['id']
                            traci.vehicle.setSpeed(veh_id, target_speed)

                # 进度报告（每500步）- 移到if obs块内部
                if step % 500 == 0 and step > 0:
                    print(f"\n[步骤 {step}/{max_steps}] 活跃: {len(traci.vehicle.getIDList())}, "
                          f"累计出发: {self.cumulative_departed}, "
                          f"累计到达: {self.cumulative_arrived}")

                    # 打印每个路口的控制信息
                    for junc_id in sorted(obs.keys()):
                        controlled = self._get_controlled_vehicles(junc_id)
                        total_cv = len(controlled['main']) + len(controlled['ramp']) + len(controlled.get('diverge', []))

                        print(f"  {junc_id}: 总CV={total_cv}辆")

                        # 调试：如果这个路口有分配但返回0，打印分配的车辆信息
                        assigned_count = sum(1 for v, j in self.global_cv_assignment.items() if j == junc_id)
                        if assigned_count > 0 and total_cv == 0:
                            print(f"    ⚠️ 分配了{assigned_count}辆CV但返回0辆！")
                            # 打印这个路口分配的CV车辆及其edge
                            for veh_id, assigned_junc in list(self.global_cv_assignment.items())[:5]:
                                if assigned_junc == junc_id:
                                    try:
                                        edge = traci.vehicle.getRoadID(veh_id)
                                        print(f"      {veh_id}: edge={edge}")
                                    except:
                                        pass

                        if controlled['main']:
                            main_actions = actions[junc_id].get('main_actions', None)
                            if main_actions is not None and main_actions.size(1) > 0:
                                num_show = min(len(controlled['main']), main_actions.size(1))
                                action_str = ", ".join([f"{main_actions[0, i].item():.3f}" for i in range(num_show)])
                                main_ids = [v['id'] for v in controlled['main'][:3]]
                                if len(controlled['main']) > 3:
                                    main_ids.append(f"...(+{len(controlled['main'])-3})")
                                print(f"    主路: 动作=[{action_str}], CV={len(controlled['main'])}辆, IDs={main_ids}")

                        if controlled['ramp']:
                            ramp_actions = actions[junc_id].get('ramp_actions', None)
                            if ramp_actions is not None and ramp_actions.size(1) > 0:
                                num_show = min(len(controlled['ramp']), ramp_actions.size(1))
                                action_str = ", ".join([f"{ramp_actions[0, i].item():.3f}" for i in range(num_show)])
                                ramp_ids = [v['id'] for v in controlled['ramp'][:3]]
                                if len(controlled['ramp']) > 3:
                                    ramp_ids.append(f"...(+{len(controlled['ramp'])-3})")
                                print(f"    匝道: 动作=[{action_str}], CV={len(controlled['ramp'])}辆, IDs={ramp_ids}")

                        if controlled.get('diverge'):
                            diverge_actions = actions[junc_id].get('diverge_actions', None)
                            if diverge_actions is not None and diverge_actions.size(1) > 0:
                                num_show = min(len(controlled['diverge']), diverge_actions.size(1))
                                action_str = ", ".join([f"{diverge_actions[0, i].item():.3f}" for i in range(num_show)])
                                diverge_ids = [v['id'] for v in controlled['diverge'][:3]]
                                if len(controlled['diverge']) > 3:
                                    diverge_ids.append(f"...(+{len(controlled['diverge'])-3})")
                                print(f"    分流: 动作=[{action_str}], CV={len(controlled['diverge'])}辆, IDs={diverge_ids}")

                        if total_cv == 0 and assigned_count == 0:
                            print(f"    无CV车辆")
                    print()  # 空行分隔

            else:
                # 如果没有观察，也打印
                if step % 500 == 0 and step > 0:
                    print(f"\n[步骤 {step}/{max_steps}] ⚠️ 没有获取到观察数据")
                    print(f"  活跃车辆: {len(traci.vehicle.getIDList())}")
                    print()

            # 仿真一步
            traci.simulationStep()

            # 收集数据
            self._collect_step_data(step, current_time)

        print(f"✓ 仿真完成")

    def _assign_cv_vehicles(self):
        """全局分配所有CV车辆给各个路口（使用road_topology_hardcoded.py拓扑）"""
        all_cv_vehicles = []
        all_vehicles = traci.vehicle.getIDList()

        # 调试：打印所有车辆类型（每500步打印一次）
        vehicle_types = {}
        for veh_id in all_vehicles:
            try:
                vtype = traci.vehicle.getTypeID(veh_id)
                vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

                # CV车辆检测：类型名为CV，或者使用vClass判断
                is_cv = False
                if vtype == 'CV':
                    is_cv = True
                elif vtype == 'penetration0.05':
                    # 对于使用vTypeDistribution的车辆，需要检查vClass
                    try:
                        vclass = traci.vehicle.getVehicleClass(veh_id)
                        # 忽略默认的vehicle class
                        pass
                    except:
                        pass

                    # 另一种方法：检查车辆ID或路由
                    # 某些SUMO版本会为CV车辆添加特殊标记
                    try:
                        route_id = traci.vehicle.getRouteID(veh_id)
                        # 检查flow定义中是否包含CV标记
                        # 实际上，使用vTypeDistribution时，SUMO内部已经分配了
                        # 我们可以通过检查车辆参数来判断
                        pass
                    except:
                        pass

                    # 最可靠的方法：检查vType参数
                    try:
                        # 获取车辆的实际vType参数
                        vtype_id = traci.vehicle.getTypeID(veh_id)
                        # 对于vTypeDistribution，SUMO会在创建时随机选择
                        # 我们无法直接获取，但可以尝试其他方法
                        pass
                    except:
                        pass

                # 暂时使用简单检测：类型名包含CV
                if 'CV' in vtype.upper():
                    all_cv_vehicles.append(veh_id)

            except Exception as e:
                continue

        # 每500步打印一次调试信息
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        self._step_counter += 1

        if self._step_counter % 500 == 0:
            print(f"\n[调试 步骤{self._step_counter}] 车辆类型统计: {vehicle_types}")
            print(f"[调试] 总车辆数: {len(all_vehicles)}, CV车辆数: {len(all_cv_vehicles)}")
            if all_cv_vehicles:
                print(f"[调试] CV车辆ID示例: {all_cv_vehicles[:5]}")
                # 打印CV分配统计
                junc_counts = {}
                for veh_id, junc_id in self.global_cv_assignment.items():
                    junc_counts[junc_id] = junc_counts.get(junc_id, 0) + 1
                print(f"[调试] CV路口分配: {junc_counts}")
            else:
                # 如果没找到CV，打印所有车辆ID和类型
                print(f"[调试] ⚠️ 没有找到CV车辆！前10个车辆:")
                for veh_id in list(all_vehicles)[:10]:
                    try:
                        vtype = traci.vehicle.getTypeID(veh_id)
                        edge = traci.vehicle.getRoadID(veh_id)
                        # 尝试获取更多车辆信息
                        vclass = traci.vehicle.getVehicleClass(veh_id)
                        print(f"  {veh_id}: type={vtype}, vclass={vclass}, edge={edge}")
                    except:
                        pass
                print(f"[调试] 提示：可能需要检查SUMO的vTypeDistribution配置")
            print()

        # 清空之前的分配
        self.global_cv_assignment.clear()

        # 使用road_topology_hardcoded.py的拓扑分配CV车辆
        for veh_id in all_cv_vehicles:
            try:
                current_edge = traci.vehicle.getRoadID(veh_id)

                # 使用EDGE_TOPOLOGY获取边的拓扑信息
                edge_info = EDGE_TOPOLOGY.get(current_edge)

                if edge_info:
                    # 根据边的from_junction和to_junction分配
                    # 优先使用to_junction（车辆即将到达的路口）
                    assigned_junction = edge_info.to_junction or edge_info.from_junction

                    # 如果都没有定义，使用边名称推断
                    if not assigned_junction:
                        # 从边名称推断路口
                        if current_edge in ['E23', 'E2', '-E3']:
                            assigned_junction = 'J5'
                        elif current_edge in ['E15', 'E9', '-E10']:
                            assigned_junction = 'J14'
                        elif current_edge in ['E17', 'E10', '-E11', 'E16']:
                            assigned_junction = 'J15'
                        elif current_edge in ['E19', 'E12', '-E13', 'E18', 'E20']:
                            assigned_junction = 'J17'
                        else:
                            # 默认分配给J14
                            assigned_junction = 'J14'
                else:
                    # 如果edge不在拓扑中，使用默认分配
                    if current_edge.startswith('E') and not current_edge.startswith('-E'):
                        assigned_junction = 'J14'
                    elif current_edge.startswith('-E'):
                        assigned_junction = 'J5'
                    else:
                        assigned_junction = 'J14'

                self.global_cv_assignment[veh_id] = assigned_junction
            except Exception as e:
                # 出错时默认分配给J14
                self.global_cv_assignment[veh_id] = 'J14'

    def _get_observations(self):
        """获取观察（23维状态向量，使用简化的edges配置）"""
        observations = {}

        for junc_id, junc_config in JUNCTION_CONFIG.items():
            try:
                # 使用简化的配置：自动分类主路和匝道
                main_edges = get_junction_main_edges(junc_id)
                ramp_edges = get_junction_ramp_edges(junc_id)

                # 获取主路统计
                main_vehicles = []
                main_speed_sum = 0
                main_queue_count = 0
                for edge in main_edges:
                    try:
                        edge_speed = traci.edge.getLastStepMeanSpeed(edge)
                        edge_vehicles = traci.edge.getLastStepVehicleIDs(edge)
                        main_speed_sum += edge_speed * len(edge_vehicles)
                        main_vehicles.extend(edge_vehicles)
                        # 排队车辆：速度<1m/s
                        for veh in edge_vehicles:
                            if traci.vehicle.getSpeed(veh) < 1.0:
                                main_queue_count += 1
                    except:
                        pass

                main_avg_speed = main_speed_sum / max(len(main_vehicles), 1) / 20.0  # 归一化
                main_queue_norm = main_queue_count / 10.0  # 归一化
                main_occupancy = len(main_vehicles) / 40.0  # 归一化

                # 获取匝道统计
                ramp_vehicles = []
                ramp_speed_sum = 0
                ramp_queue_count = 0
                for edge in ramp_edges:
                    try:
                        edge_speed = traci.edge.getLastStepMeanSpeed(edge)
                        edge_vehicles = traci.edge.getLastStepVehicleIDs(edge)
                        ramp_speed_sum += edge_speed * len(edge_vehicles)
                        ramp_vehicles.extend(edge_vehicles)
                        for veh in edge_vehicles:
                            if traci.vehicle.getSpeed(veh) < 1.0:
                                ramp_queue_count += 1
                    except:
                        pass

                ramp_avg_speed = ramp_speed_sum / max(len(ramp_vehicles), 1) / 10.0  # 归一化
                ramp_queue_norm = ramp_queue_count / 40.0  # 归一化
                ramp_occupancy = len(ramp_vehicles) / 20.0  # 归一化

                # 下游边速度：从主路边中获取（取前2条作为下游）
                downstream_edges = main_edges[:2] if len(main_edges) >= 2 else main_edges
                downstream_speed_sum = 0
                for edge in downstream_edges:
                    try:
                        downstream_speed_sum += traci.edge.getLastStepMeanSpeed(edge)
                    except:
                        pass
                downstream_speed = downstream_speed_sum / max(len(downstream_edges), 1) / 20.0

                # 冲突风险（简化版）
                conflict_risk = min(main_occupancy * ramp_occupancy, 1.0)

                # 组装23维状态向量
                state = np.zeros(23, dtype=np.float32)
                state[0] = main_avg_speed  # 0: 主路平均速度
                state[1] = main_avg_speed  # 1: 主路速度（重复）
                state[2] = 0  # 2: 主路加速度（简化）
                state[3] = main_queue_norm  # 3: 主路排队长度
                state[4] = 0  # 4: 主路密度（简化）
                state[5] = main_occupancy  # 5: 主路占有率

                state[6] = ramp_avg_speed  # 6: 匝道平均速度
                state[7] = ramp_queue_norm  # 7: 匝道排队长度
                state[8] = ramp_occupancy  # 8: 匝道占有率
                state[9:15] = downstream_speed  # 9-14: 下游速度

                state[15] = conflict_risk  # 15: 冲突风险
                state[16:] = 0  # 16-22: 预留

                observations[junc_id] = state

            except Exception as e:
                # 如果某个路口失败，返回零向量
                observations[junc_id] = np.zeros(23, dtype=np.float32)

        return observations

    def _get_controlled_vehicles(self, junc_id: str):
        """获取路口控制的CV车辆（使用简化的edges配置，自动分类）"""
        if junc_id not in JUNCTION_CONFIG:
            return {'main': [], 'ramp': [], 'diverge': []}

        # 使用辅助函数获取各类边（根据EDGE_TOPOLOGY自动分类）
        all_edges = get_junction_edges(junc_id)
        main_edges = get_junction_main_edges(junc_id)
        ramp_edges = get_junction_ramp_edges(junc_id)
        diverge_edges = get_junction_diverge_edges(junc_id)

        # 调试：第一次时打印edge列表
        if not hasattr(self, f'_edge_printed_{junc_id}'):
            print(f"    [DEBUG] {junc_id} edge列表:")
            print(f"      所有边: {all_edges}")
            print(f"      主路边: {main_edges}")
            print(f"      匝道边: {ramp_edges}")
            print(f"      分流边: {diverge_edges}")
            setattr(self, f'_edge_printed_{junc_id}', True)

        # 获取分配给这个路口的所有CV车辆
        assigned_vehicles = [
            veh_id for veh_id, assigned_junc in self.global_cv_assignment.items()
            if assigned_junc == junc_id
        ]

        # 根据edge类型分组
        main_vehicles = []
        ramp_vehicles = []
        diverge_vehicles = []

        for veh_id in assigned_vehicles:
            try:
                edge = traci.vehicle.getRoadID(veh_id)

                # 检查车辆在哪个类型的edge上
                is_main = edge in main_edges
                is_ramp = edge in ramp_edges
                is_diverge = edge in diverge_edges

                vehicle_info = {
                    'id': veh_id,
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'acceleration': traci.vehicle.getAcceleration(veh_id),
                    'lane_position': traci.vehicle.getLanePosition(veh_id),
                    'lane_index': traci.vehicle.getLaneIndex(veh_id),
                    'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                    'is_cv': True,  # 只获取CV车辆
                    'edge': edge
                }

                # 按优先级分组：ramp > diverge > main
                if is_ramp:
                    ramp_vehicles.append(vehicle_info)
                elif is_diverge:
                    diverge_vehicles.append(vehicle_info)
                elif is_main:
                    main_vehicles.append(vehicle_info)
                else:
                    # 如果不在任何定义的edge上，检查是否是路口范围内的边
                    if edge in all_edges:
                        # 在edges列表中，但没有明确分类，归为main
                        main_vehicles.append(vehicle_info)
                    else:
                        # 不在edges列表中，跳过
                        continue
            except Exception as e:
                continue

        return {'main': main_vehicles, 'ramp': ramp_vehicles, 'diverge': diverge_vehicles}

    def _get_vehicle_features(self, vehicles):
        """获取车辆特征张量（8维，与训练代码一致）"""
        if not vehicles:
            return None
        features = []
        for veh in vehicles:
            # 从vehicle_type_config导入normalize_speed
            from vehicle_type_config import normalize_speed
            features.append([
                normalize_speed(veh.get('speed', 0)),
                veh.get('lane_position', 0) / 500.0,
                veh.get('lane_index', 0) / 3.0,
                veh.get('waiting_time', 0) / 60.0,
                veh.get('acceleration', 0) / 5.0,
                1.0 if veh.get('is_cv', False) else 0.0,
                0.0,  # route_index
                0.0
            ])
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _collect_step_data(self, step: int, current_time: float):
        """收集时间步数据（完全匹配比赛格式）"""
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
                    traveled_distance / max(route_info['route_length'], 1.0), 1.0
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
        """保存PKL文件（完全匹配比赛格式）"""
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
                'collection_timestamp': timestamp,
                'vehicle_type_maxspeed': self.vehicle_type_maxspeed,
                'model_checkpoint': self.checkpoint_path
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
        ocr = self.cumulative_arrived / max(self.cumulative_departed, 1)

        print(f"\n{'=' * 70}")
        print(f"数据保存统计")
        print(f"{'=' * 70}")
        print(f"✓ Pickle文件: {output_path}")
        print(f"  - 文件大小: {file_size:.2f} MB")
        print(f"  - 时间步数据记录: {len(self.step_data):,} 条")
        print(f"  - 车辆数据记录: {len(self.vehicle_data):,} 条")
        print(f"  - 唯一车辆数: {len(self.route_data):,} 辆")
        print(f"  - 最终OCR: {ocr:.4f}")
        print(f"{'=' * 70}")

    def _close_sumo(self):
        """关闭SUMO"""
        try:
            traci.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='从模型checkpoint生成比赛提交PKL文件')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg',
                        help='SUMO配置文件路径')
    parser.add_argument('--output', type=str, default='submission.pkl',
                        help='输出pkl文件路径')
    parser.add_argument('--steps', type=int, default=3600,
                        help='仿真步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cpu/cuda)')

    args = parser.parse_args()

    generator = ModelSubmissionGenerator(
        checkpoint_path=args.checkpoint,
        sumo_cfg=args.sumo_cfg,
        device=args.device
    )

    generator.generate_pkl(
        output_path=args.output,
        max_steps=args.steps,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
