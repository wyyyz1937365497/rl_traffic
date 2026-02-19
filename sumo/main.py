import os
from vehicle_type_config import normalize_speed, get_vehicle_max_speed
import sys
import traci
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import pickle
import torch
import numpy as np
import logging
import traceback as tb

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger('sumo_main')

# 添加父目录到路径以导入模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from junction_network import MultiJunctionModel
from junction_agent import JUNCTION_CONFIGS


class SUMOCompetitionFramework:
    """
    SUMO竞赛数据收集框架 - 集成强化学习模型
    """

    def __init__(self, sumo_cfg_path, model_path="../checkpoints/final_model.pt"):
        self.sumo_cfg_path = sumo_cfg_path
        self.model_path = model_path
        self.routes_file = None
        self.net_file = None

        # 数据存储
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

        # RL模型相关
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.agents = {}
        self.model_loaded = False

        print("=" * 70)
        print("SUMO竞赛框架 - 强化学习模型版本")
        print("=" * 70)

    # ========================================================================
    # 第一部分: 环境初始化
    # ========================================================================

    def parse_config(self):
        """解析SUMO配置文件"""
        print("\n[第一部分] 正在初始化环境...")

        tree = ET.parse(self.sumo_cfg_path)
        root = tree.getroot()

        config_dir = os.path.dirname(self.sumo_cfg_path)

        # 获取路网和路径文件
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

    def parse_routes(self):
        """解析路径文件"""
        if not self.routes_file or not os.path.exists(self.routes_file):
            print("⚠️  路径文件不存在")
            return

        tree = ET.parse(self.routes_file)
        root = tree.getroot()

        total_vehicles = 0
        for vehicle in root.findall('.//vehicle'):
            total_vehicles += 1

        self.total_demand = total_vehicles
        print(f"✓ 总需求: {total_vehicles} 辆车")

    def initialize_environment(self):
        """初始化SUMO环境"""
        print("\n正在启动SUMO仿真...")

        sumo_cmd = [
            "sumo",
            "-c", self.sumo_cfg_path,
            "--no-warnings", "true",
            "--seed", "42"
        ]

        traci.start(sumo_cmd)
        print("✓ SUMO已启动")

        # 检测可用的红绿灯
        for tl_id in self.traffic_lights:
            try:
                traci.trafficlight.getRedYellowGreenState(tl_id)
                self.available_traffic_lights.append(tl_id)
            except Exception as e:
                print(f"检测信号灯 {tl_id} 失败: {e}")

        print(f"✓ 检测到 {len(self.available_traffic_lights)} 个可控制红绿灯")

    def load_rl_model(self):
        """加载强化学习模型"""
        print(f"\n正在加载RL模型: {self.model_path}")

        if not os.path.exists(self.model_path):
            print(f"⚠️  模型文件不存在: {self.model_path}")
            print("   将不使用模型控制")
            return

        try:
            # 创建模型
            self.model = MultiJunctionModel(JUNCTION_CONFIGS)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True

            print(f"✓ 模型已加载到 {self.device}")

            # 创建智能体 - 将JunctionConfig转换为字典格式
            for junc_id, junc_config in JUNCTION_CONFIGS.items():
                # 转换为RLAgent期望的字典格式
                config_dict = {
                    'edges': {
                        'main': junc_config.main_incoming,
                        'ramp': junc_config.ramp_incoming
                    }
                }
                self.agents[junc_id] = RLAgent(junc_id, config_dict, self.model, self.device)

            print(f"✓ 已创建 {len(self.agents)} 个RL智能体")

        except Exception as e:
            print(f"⚠️  模型加载失败: {e}")
            print("   将不使用模型控制")
            self.model_loaded = False

    # ========================================================================
    # 第二部分: 控制算法实现
    # ========================================================================

    def apply_control_algorithm(self, step):
        """应用RL模型控制算法"""
        if not self.model_loaded or step < 10:
            return

        try:
            # 收集所有智能体的观察
            obs_tensors = {}
            vehicle_obs = {}

            for junc_id, agent in self.agents.items():
                state_vec = agent.observe(traci)
                if state_vec is not None:
                    obs_tensors[junc_id] = torch.tensor(
                        state_vec, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    controlled = agent.get_controlled_vehicles()
                    vehicle_obs[junc_id] = {
                        'main': agent.get_vehicle_features(controlled['main'], traci, self.device),
                        'ramp': agent.get_vehicle_features(controlled['ramp'], traci, self.device),
                        'diverge': agent.get_vehicle_features(controlled['diverge'], traci, self.device)
                    }

            if not obs_tensors:
                return

            # 模型推理
            with torch.no_grad():
                actions, values, info = self.model(obs_tensors, vehicle_obs, deterministic=True)

            # 应用控制动作（与训练时完全一致的逻辑）
            for junc_id, action in actions.items():
                agent = self.agents[junc_id]
                controlled = agent.get_controlled_vehicles()

                # 控制主路车辆
                if controlled['main'] and 'main' in action:
                    for veh_id in controlled['main'][:1]:
                        try:
                            action_value = action['main'].item()
                            # 与训练时完全一致的映射
                            speed_limit = 13.89
                            target_speed = speed_limit * (0.3 + 0.9 * action_value)
                            # 确保速度在合理范围内
                            target_speed = max(0.0, min(target_speed, speed_limit * 1.2))
                            traci.vehicle.setSpeed(veh_id, target_speed)
                        except Exception as e:
                            print(f"设置主路车辆 {veh_id} 速度失败: {e}")

                # 控制匝道车辆
                if controlled['ramp'] and 'ramp' in action:
                    for veh_id in controlled['ramp'][:1]:
                        try:
                            action_value = action['ramp'].item()
                            # 与训练时完全一致的映射
                            speed_limit = 13.89
                            target_speed = speed_limit * (0.3 + 0.9 * action_value)
                            # 确保速度在合理范围内
                            target_speed = max(0.0, min(target_speed, speed_limit * 1.2))
                            traci.vehicle.setSpeed(veh_id, target_speed)
                        except Exception as e:
                            print(f"设置匝道车辆 {veh_id} 速度失败: {e}")

        except Exception as e:
            # 静默失败，不影响仿真
            pass

    # ========================================================================
    # 第三部分: 数据收集与统计
    # ========================================================================

    def get_traffic_light_states(self):
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
            except Exception as e:
                print(f"获取信号灯 {tl_id} 状态失败: {e}")
                tl_states[f'{tl_id}_state'] = 'unknown'
                tl_states[f'{tl_id}_phase'] = -1
                tl_states[f'{tl_id}_remaining_time'] = -1
        return tl_states

    def get_vehicle_od(self, veh_id):
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
            original_max_speed = self.get_original_max_speed(vehicle_type)

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
            return {
                'origin': 'unknown',
                'destination': 'unknown',
                'route_length': 0,
                'vehicle_type': 'unknown',
                'original_max_speed': 0.0
            }

    def get_original_max_speed(self, vehicle_type):
        """获取车辆类型的原始maxSpeed配置"""
        if vehicle_type in self.vehicle_type_maxspeed:
            return self.vehicle_type_maxspeed[vehicle_type]

        try:
            max_speed = traci.vehicletype.getMaxSpeed(vehicle_type)
            self.vehicle_type_maxspeed[vehicle_type] = max_speed
            return max_speed
        except Exception as e:
            print(f"获取车辆类型 {vehicle_type} 最大速度失败: {e}")
            return 0.0

    def get_route_length(self, route_edges):
        """计算路径长度"""
        total_length = 0.0
        for edge_id in route_edges:
            try:
                total_length += traci.lane.getLength(edge_id + "_0")
            except Exception as e:
                print(f"获取边 {edge_id} 长度失败: {e}")
        return total_length

    def calculate_traveled_distance(self, veh_id, route_info):
        """计算已行驶距离"""
        traveled = 0.0
        route_index = traci.vehicle.getRouteIndex(veh_id)

        for i in range(route_index):
            try:
                edge_id = route_info['route_edges'][i]
                traveled += traci.lane.getLength(edge_id + "_0")
            except Exception as e:
                print(f"计算车辆 {veh_id} 行驶距离失败: {e}")

        traveled += traci.vehicle.getLanePosition(veh_id)
        return traveled

    def check_maxspeed_violations(self):
        """检查maxSpeed违规"""
        violations = []
        for veh_id in traci.vehicle.getIDList():
            try:
                current_speed = traci.vehicle.getSpeed(veh_id)
                max_speed = traci.vehicle.getMaxSpeed(veh_id)
                if current_speed > max_speed + 0.1:
                    violations.append({
                        'vehicle_id': veh_id,
                        'current_speed': current_speed,
                        'max_speed': max_speed,
                        'violation': current_speed - max_speed
                    })
            except Exception as e:
                print(f"检查车辆 {veh_id} maxSpeed违规失败: {e}")
        return violations

    def collect_step_data(self, step):
        """收集每步数据"""
        current_time = traci.simulation.getTime()
        current_vehicle_ids = traci.vehicle.getIDList()

        # 统计出发和到达（修复：使用累计统计）
        step_departed = traci.simulation.getDepartedNumber()  # 当前步出发
        step_arrived = traci.simulation.getArrivedNumber()    # 当前步到达

        # 累加到总数
        self.cumulative_departed += step_departed
        self.cumulative_arrived += step_arrived

        # 获取新出发和到达的车辆ID
        if step_departed > 0:
            new_departed = set(traci.simulation.getDepartedIDList())
            self.all_departed_vehicles.update(new_departed)

        if step_arrived > 0:
            new_arrived = set(traci.simulation.getArrivedIDList())
            self.all_arrived_vehicles.update(new_arrived)

        # 收集时间步级数据
        step_record = {
            'step': step,
            'time': current_time,
            'departed': self.cumulative_departed,  # 使用累计值
            'arrived': self.cumulative_arrived,    # 使用累计值
            'active_vehicles': len(current_vehicle_ids)
        }

        traffic_light_states = self.get_traffic_light_states()
        step_record.update(traffic_light_states)
        self.step_data.append(step_record)

        # 收集车辆级数据
        for veh_id in current_vehicle_ids:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getLanePosition(veh_id)
                edge_id = traci.vehicle.getRoadID(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)

                od_info = self.get_vehicle_od(veh_id)

                if veh_id not in self.route_data:
                    route_edges = traci.vehicle.getRoute(veh_id)
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

        # 进度报告
        if step % 100 == 0:
            print(f"[步骤 {step}] 活跃: {len(current_vehicle_ids)}, "
                  f"累计出发: {self.cumulative_departed}, "
                  f"累计到达: {self.cumulative_arrived}")

    def calculate_ocr_metrics(self):
        """
        计算OCR（Overall Completion Rate）指标

        Returns:
            dict: 包含各种OCR指标的字典
        """
        # 全局OCR
        global_ocr = self.cumulative_arrived / max(self.cumulative_departed, 1)

        # 根据OD信息计算分类OCR
        main_departed = 0
        main_arrived = 0
        ramp_departed = 0
        ramp_arrived = 0
        diverge_departed = 0
        diverge_arrived = 0

        for veh_id, od_info in self.vehicle_od_data.items():
            # 获取车辆出发边缘
            depart_edge = od_info.get('depart_edge', '')

            # 获取车辆到达边缘
            arrive_edge = od_info.get('arrive_edge', '')
            did_arrive = arrive_edge != ''

            # 判断车辆类型（基于出发边缘）
            if 'main' in depart_edge.lower() or depart_edge.startswith('-'):
                # 主路车辆
                main_departed += 1
                if did_arrive:
                    main_arrived += 1
            elif 'ramp' in depart_edge.lower():
                # 匝道车辆
                ramp_departed += 1
                if did_arrive:
                    ramp_arrived += 1

            # 判断转出车辆（基于到达边缘）
            if did_arrive and 'diverge' in arrive_edge.lower():
                diverge_arrived += 1
                # 如果出发是主路或匝道，也算作转出出发
                if 'main' in depart_edge.lower() or depart_edge.startswith('-'):
                    diverge_departed += 1
                elif 'ramp' in depart_edge.lower():
                    diverge_departed += 1

        # 计算各类OCR
        main_ocr = main_arrived / max(main_departed, 1)
        ramp_ocr = ramp_arrived / max(ramp_departed, 1)
        diverge_ocr = diverge_arrived / max(diverge_departed, 1)

        return {
            'global_ocr': global_ocr,
            'main_road_ocr': main_ocr,
            'ramp_road_ocr': ramp_ocr,
            'diverge_road_ocr': diverge_ocr,
            'statistics': {
                'total_departed': self.cumulative_departed,
                'total_arrived': self.cumulative_arrived,
                'main_departed': main_departed,
                'main_arrived': main_arrived,
                'ramp_departed': ramp_departed,
                'ramp_arrived': ramp_arrived,
                'diverge_departed': diverge_departed,
                'diverge_arrived': diverge_arrived
            }
        }

    def save_to_pickle(self, output_dir="competition_results"):
        """保存数据到pickle文件"""
        print(f"\n[第三部分] 正在保存仿真数据到Pickle格式...")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        maxspeed_violations = self.check_maxspeed_violations()

        data_package = {
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
                'vehicle_type_maxspeed': self.vehicle_type_maxspeed,
                'model_used': self.model_path if self.model_loaded else 'None'
            },
            'step_data': self.step_data,
            'vehicle_data': self.vehicle_data,
            'route_data': self.route_data,
            'vehicle_od_data': self.vehicle_od_data,
            'statistics': {
                'all_departed_vehicles': list(self.all_departed_vehicles),
                'all_arrived_vehicles': list(self.all_arrived_vehicles),
                'cumulative_departed': self.cumulative_departed,
                'cumulative_arrived': self.cumulative_arrived,
                'maxspeed_violations': maxspeed_violations
            }
        }

        pickle_file = os.path.join(output_dir, "submit.pkl")

        with open(pickle_file, 'wb') as f:
            pickle.dump(data_package, f)

        print(f"✓ 数据已保存到: {pickle_file}")
        print(f"\n最终统计:")
        print(f"  - 总需求: {self.total_demand}")
        print(f"  - 出发车辆: {self.cumulative_departed}")
        print(f"  - 到达车辆: {self.cumulative_arrived}")
        print(f"  - OCR: {self.cumulative_arrived / max(self.cumulative_departed, 1):.4f}")
        print(f"  - maxSpeed违规: {len(maxspeed_violations)}")

        return pickle_file

    def run_simulation(self):
        """运行完整仿真"""
        print("\n[第二部分] 开始仿真...")
        print(f"设备: {self.device}")
        print(f"模型状态: {'已加载' if self.model_loaded else '未加载'}")

        # 应用CACC参数优化（与训练环境完全一致）
        self._apply_cacc_parameters()

        step = 0
        while step < 3600:  # 默认3600步
            # 应用控制算法
            self.apply_control_algorithm(step)

            # 仿真一步
            traci.simulationStep()

            # 收集数据
            self.collect_step_data(step)

            step += 1

        print(f"\n✓ 仿真完成 ({step} 步)")

    def _apply_cacc_parameters(self):
        """
        应用CACC参数优化

        核心策略：
        - sigma=0: 消除随机减速（完美驾驶），提高交通流稳定性
        - tau=1.12: 微增跟车时距（抵消sigma=0带来的容量增加，保持安全性）

        这个设置与训练环境完全一致，确保训练和推理的动作空间一致。
        """
        logger.info("✓ 应用CACC参数优化 (sigma=0, tau=1.12)")

        cacc_applied = set()  # 跟踪已设置的车辆，避免重复设置
        failed_vehicles = []  # 记录失败的车辆

        try:
            all_vehicles = traci.vehicle.getIDList()
            logger.debug(f"开始应用CACC参数，车辆总数={len(all_vehicles)}")

            for veh_id in all_vehicles:
                if veh_id in cacc_applied:
                    continue

                try:
                    # 只对CV（Connected Vehicle）类型应用CACC参数
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    if veh_type == 'CV':
                        # 设置imperfection（sigma）为0，消除随机减速
                        traci.vehicle.setImperfection(veh_id, 0.0)

                        # 设置tau（跟车时距）为1.12秒，略微增大以保持安全距离
                        traci.vehicle.setTau(veh_id, 1.12)

                        cacc_applied.add(veh_id)
                except Exception as e:
                    # 车辆可能在设置过程中离开路网，记录但不中断
                    failed_vehicles.append((veh_id, str(e)))

            logger.info(f"  已为 {len(cacc_applied)} 辆CV车辆应用CACC参数，失败 {len(failed_vehicles)} 辆")
            if failed_vehicles and len(failed_vehicles) <= 5:
                for veh_id, err in failed_vehicles[:5]:
                    logger.debug(f"  车辆 {veh_id} 设置失败: {err}")

        except Exception as e:
            logger.error(f"  ⚠️  CACC参数设置过程中出现错误: {e}\n{tb.format_exc()}")

    def close(self):
        """关闭仿真"""
        try:
            traci.close()
        except Exception as e:
            print(f"关闭TraCI连接失败: {e}")


class RLAgent:
    """强化学习智能体"""

    def __init__(self, agent_id, config, model, device):
        self.agent_id = agent_id
        self.config = config
        self.model = model
        self.device = device

        self.edge_ids = config.get('edges', {})
        self.current_state = None

    def observe(self, traci_conn):
        """观察环境并返回状态向量"""
        try:
            main_edges = self.edge_ids.get('main', [])
            ramp_edges = self.edge_ids.get('ramp', [])

            # 获取主路状态
            main_vehicles = []
            main_queue_length = 0
            main_speed = 0
            main_density = 0

            for edge_id in main_edges:
                try:
                    veh_ids = traci_conn.edge.getLastStepVehicleIDs(edge_id)
                    main_vehicles.extend(veh_ids)
                    main_queue_length += traci_conn.edge.getWaitingTime(edge_id)

                    for veh_id in veh_ids:
                        main_speed += traci_conn.vehicle.getSpeed(veh_id)
                except Exception as e:
                    print(f"获取主路边 {edge_id} 状态失败: {e}")

            if main_vehicles:
                main_speed /= len(main_vehicles)
                main_density = len(main_vehicles) / (len(main_edges) * 500.0)

            # 获取匝道状态
            ramp_vehicles = []
            ramp_queue_length = 0
            ramp_speed = 0
            ramp_density = 0
            ramp_waiting_time = 0

            for edge_id in ramp_edges:
                try:
                    veh_ids = traci_conn.edge.getLastStepVehicleIDs(edge_id)
                    ramp_vehicles.extend(veh_ids)
                    ramp_queue_length += traci_conn.edge.getWaitingTime(edge_id)

                    for veh_id in veh_ids:
                        ramp_speed += traci_conn.vehicle.getSpeed(veh_id)
                        waiting_time = traci_conn.vehicle.getWaitingTime(veh_id)
                        ramp_waiting_time = max(ramp_waiting_time, waiting_time)
                except Exception as e:
                    print(f"获取匝道边 {edge_id} 状态失败: {e}")

            if ramp_vehicles:
                ramp_speed /= len(ramp_vehicles)
                ramp_density = len(ramp_vehicles) / (len(ramp_edges) * 500.0)

            # 计算间隙和冲突风险
            gap_size = 5.0 if (main_vehicles and ramp_vehicles) else 10.0
            gap_speed_diff = abs(main_speed - ramp_speed) if (main_vehicles and ramp_vehicles) else 0
            conflict_risk = min(len(main_vehicles), len(ramp_vehicles)) / 20.0

            # 检测CV
            has_cv = False
            all_vehicles = main_vehicles + ramp_vehicles
            for veh_id in all_vehicles:
                try:
                    if traci_conn.vehicle.getTypeID(veh_id) == 'CV':
                        has_cv = True
                        break
                except Exception as e:
                    print(f"检测车辆 {veh_id} 类型失败: {e}")

            self.current_state = {
                'main_queue_length': main_queue_length,
                'ramp_queue_length': ramp_queue_length,
                'main_speed': main_speed,
                'ramp_speed': ramp_speed,
                'main_density': main_density,
                'ramp_density': ramp_density,
                'ramp_waiting_time': ramp_waiting_time,
                'gap_size': gap_size,
                'gap_speed_diff': gap_speed_diff,
                'has_cv': has_cv,
                'conflict_risk': conflict_risk,
                'main_stop_count': 0,
                'ramp_stop_count': 0,
                'throughput': len(main_vehicles) + len(ramp_vehicles)
            }

            # 返回状态向量
            return np.array([
                main_queue_length,
                ramp_queue_length,
                main_speed / 20.0,
                ramp_speed / 20.0,
                main_density / 0.5,
                ramp_density / 0.5,
                ramp_waiting_time / 60.0,
                gap_size / 10.0,
                gap_speed_diff / 20.0,
                float(has_cv),
                conflict_risk,
                0.0,  # main_stop_count
                0.0,  # ramp_stop_count
                len(main_vehicles) + len(ramp_vehicles) / 100.0,
                0.0,  # phase_main
                0.0,  # phase_ramp
                0.0   # time_step
            ])

        except Exception as e:
            return None

    def get_controlled_vehicles(self):
        """获取受控车辆"""
        if not self.current_state:
            return {'main': [], 'ramp': [], 'diverge': []}

        try:
            main_edges = self.edge_ids.get('main', [])
            ramp_edges = self.edge_ids.get('ramp', [])

            main_vehicles = []
            for edge_id in main_edges:
                try:
                    main_vehicles.extend(traci.edge.getLastStepVehicleIDs(edge_id))
                except Exception as e:
                    print(f"获取主路边 {edge_id} 受控车辆失败: {e}")

            ramp_vehicles = []
            for edge_id in ramp_edges:
                try:
                    ramp_vehicles.extend(traci.edge.getLastStepVehicleIDs(edge_id))
                except Exception as e:
                    print(f"获取匝道边 {edge_id} 受控车辆失败: {e}")

            return {
                'main': main_vehicles[:1] if main_vehicles else [],
                'ramp': ramp_vehicles[:1] if ramp_vehicles else [],
                'diverge': []
            }

        except Exception as e:
            print(f"获取受控车辆失败: {e}")
            return {'main': [], 'ramp': [], 'diverge': []}

    def get_vehicle_features(self, vehicle_ids, traci_conn, device):
        """获取车辆特征"""
        if not vehicle_ids:
            return None

        features = []
        for veh_id in vehicle_ids[:10]:
            try:
                features.append([
                    traci_conn.vehicle.getSpeed(veh_id) / 20.0,
                    traci_conn.vehicle.getLanePosition(veh_id) / 500.0,
                    traci_conn.vehicle.getLaneIndex(veh_id) / 3.0,
                    traci_conn.vehicle.getWaitingTime(veh_id) / 60.0,
                    traci_conn.vehicle.getAcceleration(veh_id) / 5.0,
                    1.0 if traci_conn.vehicle.getTypeID(veh_id) == 'CV' else 0.0,
                    0.0,
                    0.0
                ])
            except Exception as e:
                print(f"获取车辆 {veh_id} 特征失败: {e}")
                features.append([0.0] * 8)

        while len(features) < 10:
            features.append([0.0] * 8)

        return torch.tensor(features, dtype=torch.float32, device=device)


# ========================================================================
# 主程序入口
# ========================================================================

if __name__ == "__main__":
    # 创建框架实例
    framework = SUMOCompetitionFramework(
        sumo_cfg_path="sumo.sumocfg",
        model_path="../checkpoints/final_model.pt"
    )

    # 第一部分: 初始化
    framework.parse_config()
    framework.parse_routes()
    framework.initialize_environment()
    framework.load_rl_model()

    # 第二部分: 运行仿真
    framework.run_simulation()

    # 第三部分: 保存结果
    framework.save_to_pickle()
    framework.close()

    print("\n" + "=" * 70)
    print("仿真完成！")
    print("=" * 70)
