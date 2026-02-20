import os
import sys
import traci
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import pickle

# 可选导入（仅在需要使用RL模型时）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SUMOCompetitionFramework:
    """
    SUMO竞赛数据收集框架

    框架结构:
    1. 环境初始化 (parse_config, parse_routes, initialize_environment)
    2. 控制算法实现 (apply_control_algorithm - 参赛者自定义)
    3. 数据收集与统计 (collect_step_data, save_to_pickle)
    """

    def __init__(self, sumo_cfg_path):
        self.sumo_cfg_path = sumo_cfg_path
        self.routes_file = None
        self.net_file = None

        # 数据存储
        self.vehicle_data = []  # 车辆级数据
        self.step_data = []  # 时间步级数据
        self.route_data = {}  # 车辆路径数据
        self.vehicle_od_data = {}  # 车辆OD信息存储
        self.vehicle_type_maxspeed = {}  # 记录每种车辆类型的原始maxSpeed

        # 累计统计
        self.cumulative_departed = 0  # 累计出发车辆数
        self.cumulative_arrived = 0  # 累计到达车辆数
        self.all_departed_vehicles = set()  # 所有出发过的车辆
        self.all_arrived_vehicles = set()  # 所有到达过的车辆

        # 红绿灯监控
        self.traffic_lights = ['J5', 'J14', 'J15', 'J17']  # 可修改
        self.available_traffic_lights = []  # 实际可用的红绿灯

        # 仿真参数
        self.flow_rate = 0
        self.simulation_time = 0
        self.step_length = 1.0
        self.total_demand = 0  # 理论总需求

        # RL模型相关（可选，不修改SUMO配置）
        self.model = None
        self.model_loaded = False
        self.device = 'cpu'
        self.agents = {}

        print("=" * 70)
        print("SUMO竞赛数据收集框架 ")
        print("=" * 70)
        print("框架结构:")
        print("  第一部分: 环境初始化 (Baseline环境)")
        print("  第二部分: 控制算法 (参赛者自定义)")
        print("  第三部分: 数据统计与保存 (Pickle格式)")
        print("=" * 70)

    # ========================================================================
    # 第一部分: 环境初始化 (Baseline环境)
    # ========================================================================

    def parse_config(self):
        """解析SUMO配置文件"""
        print("\n[第一部分] 正在初始化Baseline环境...")

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

        print(f"[OK] 配置解析完成:")
        print(f"  - 网络文件: {self.net_file}")
        print(f"  - 路径文件: {self.routes_file}")
        print(f"  - 时间步长: {self.step_length}s")

    def parse_routes(self):
        """解析路径文件,计算车流量和总需求,并记录原始maxSpeed配置"""
        if not self.routes_file or not os.path.exists(self.routes_file):
            print("[WARNING]  路径文件不存在,无法计算理论需求")
            return

        try:
            tree = ET.parse(self.routes_file)
            root = tree.getroot()

            total_vehs_per_hour = 0
            max_end_time = 0
            total_demand = 0

            # 记录所有车辆类型的原始maxSpeed配置
            for vtype in root.findall('vType'):
                vtype_id = vtype.get('id')
                max_speed = vtype.get('maxSpeed')
                if max_speed is not None:
                    self.vehicle_type_maxspeed[vtype_id] = float(max_speed)
                    print(f"  - 车辆类型 {vtype_id}: maxSpeed = {max_speed} m/s")

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

            print(f"[OK] 交通需求分析:")
            print(f"  - 流量率: {self.flow_rate:.4f} veh/s")
            print(f"  - 仿真时长: {self.simulation_time:.2f} s")
            print(f"  - 理论总需求: {self.total_demand:.0f} 车辆")
            print(f"  - 单独trip数量: {trip_count}")
            print(f"  - 记录车辆类型数: {len(self.vehicle_type_maxspeed)}")

        except Exception as e:
            print(f"[ERROR] 路径文件解析失败: {e}")

    def initialize_traffic_lights(self):
        """初始化红绿灯监控"""
        try:
            all_tls = traci.trafficlight.getIDList()

            for tl_id in self.traffic_lights:
                if tl_id in all_tls:
                    self.available_traffic_lights.append(tl_id)
                else:
                    print(f"[WARNING]  红绿灯 {tl_id} 不存在于当前网络中")

            print(f"[OK] 红绿灯监控设置:")
            print(f"  - 目标红绿灯: {self.traffic_lights}")
            print(f"  - 可用红绿灯: {self.available_traffic_lights}")
            print(f"  - 全部红绿灯: {list(all_tls)}")

        except Exception as e:
            print(f"[ERROR] 红绿灯初始化失败: {e}")
            self.available_traffic_lights = []

    def load_rl_model(self, model_path, device='cuda'):
        """
        加载RL模型（可选功能，不修改SUMO配置）

        注意: 此方法仅加载模型用于推理，不修改任何SUMO配置参数
              所有控制通过TraCI命令实现

        Args:
            model_path: 模型文件路径
            device: 设备 ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            print("[WARNING]  PyTorch未安装，无法加载RL模型")
            return False

        try:
            # 导入必要的模块
            from junction_network import create_junction_model, NetworkConfig
            from junction_agent import JUNCTION_CONFIGS, JunctionAgent, SubscriptionManager

            # 设置设备
            self.device = device if torch.cuda.is_available() else 'cpu'

            # 创建模型 - 使用JUNCTION_CONFIGS
            self.model = create_junction_model(JUNCTION_CONFIGS, NetworkConfig())
            checkpoint = torch.load(model_path, map_location=self.device)

            # 支持两种checkpoint格式
            if isinstance(checkpoint, dict):
                # 格式1: {'model_state_dict': ..., ...}
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                # 格式2: 直接是state_dict
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # 格式3: 直接是state_dict
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            # 创建智能体
            import traci
            for junc_id, junc_config in JUNCTION_CONFIGS.items():
                sub_manager = SubscriptionManager(junc_id)
                sub_manager.setup_subscriptions()
                self.agents[junc_id] = JunctionAgent(junc_config, sub_manager)

            self.model_loaded = True
            print(f"[OK] RL模型已加载: {model_path}")
            print(f"  设备: {self.device}")
            print(f"  智能体数量: {len(self.agents)}")
            return True

        except Exception as e:
            print(f"[WARNING]  模型加载失败: {e}")
            print("  将运行Baseline模式（无模型控制）")
            self.model_loaded = False
            return False

    def initialize_environment(self, use_gui=True, max_steps=1000):
        """初始化SUMO仿真环境"""
        print("\n[第一部分] 正在启动SUMO仿真...")

        # 解析配置
        self.parse_config()
        self.parse_routes()

        # 启动SUMO
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]

        try:
            traci.start(sumo_cmd)
            print(f"[OK] SUMO启动成功 (模式: {'GUI' if use_gui else 'CLI'})")
        except Exception as e:
            print(f"[ERROR] SUMO启动失败: {e}")
            return False

        # 初始化红绿灯
        self.initialize_traffic_lights()

        print("[OK] Baseline环境初始化完成!\n")
        return True

    # ========================================================================
    # 第二部分: 控制算法实现 (参赛者自定义)
    # ========================================================================

    def apply_control_algorithm(self, step):
        """
        应用控制优化算法 - 参赛者在此实现自己的算法

        参数:
            step: 当前仿真步数

        示例算法:
            - 自适应信号灯控制
            - 动态路径规划
            - 车辆速度控制
            - 交通流优化

        可用的TraCI函数示例:
            - traci.trafficlight.setPhase(tl_id, phase_index)
            - traci.trafficlight.setPhaseDuration(tl_id, duration)
            - traci.vehicle.setSpeed(veh_id, speed)
            - traci.vehicle.setRoute(veh_id, edge_list)
        """

        # ============================================================
        # 参赛者代码区域开始
        # ============================================================

        # 如果有加载RL模型，使用模型进行控制
        if hasattr(self, 'model_loaded') and self.model_loaded:
            try:
                self._apply_rl_control(step)
            except Exception as e:
                # 静默失败，不影响仿真
                pass

        # 示例1: 简单的固定相位时长控制
        # for tl_id in self.available_traffic_lights:
        #     current_phase = traci.trafficlight.getPhase(tl_id)
        #     # 设置固定相位时长为30秒
        #     traci.trafficlight.setPhaseDuration(tl_id, 30)

        # 示例2: 基于车辆数的自适应信号灯
        # for tl_id in self.available_traffic_lights:
        #     # 获取信号灯控制的车道
        #     controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        #     vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane)
        #                        for lane in controlled_lanes)
        #
        #     # 根据车辆数动态调整相位
        #     if vehicle_count > 10:
        #         traci.trafficlight.setPhaseDuration(tl_id, 45)
        #     else:
        #         traci.trafficlight.setPhaseDuration(tl_id, 20)

        # 示例3: 车辆速度控制
        # vehicle_ids = traci.vehicle.getIDList()
        # for veh_id in vehicle_ids:
        #     current_speed = traci.vehicle.getSpeed(veh_id)
        #     edge_id = traci.vehicle.getRoadID(veh_id)
        #     # 实现自定义的速度控制逻辑

        # ============================================================
        # 参赛者代码区域结束
        # ============================================================

        pass  # 默认不执行任何控制算法

    def _apply_rl_control(self, step):
        """使用RL模型进行控制（可选功能）"""
        if step < 10:
            return

        try:
            # 收集所有智能体的观察
            obs_tensors = {}
            vehicle_obs = {}

            for junc_id, agent in self.agents.items():
                # observe() 不接受参数
                state = agent.observe()
                if state is not None:
                    # 将JunctionState转换为tensor
                    import torch
                    state_vec = torch.tensor([
                        len(state.main_vehicles),  # 主路车辆数
                        state.main_speed,
                        state.main_density,
                        state.main_queue_length,
                        len(state.ramp_vehicles),  # 匝道车辆数
                        state.ramp_speed,
                        state.ramp_queue_length,
                        state.current_phase if state.current_phase is not None else 0
                    ], dtype=torch.float32).unsqueeze(0).to(self.device)

                    obs_tensors[junc_id] = state_vec

                    # 获取受控车辆特征
                    controlled = agent.get_controlled_vehicles()
                    vehicle_obs[junc_id] = {
                        'main': self._get_vehicle_tensor(controlled.get('main', [])),
                        'ramp': self._get_vehicle_tensor(controlled.get('ramp', []))
                    }

            if not obs_tensors:
                return

            # 模型推理
            with torch.no_grad():
                actions, _, _ = self.model(obs_tensors, vehicle_obs, deterministic=True)

            # 应用控制动作
            self._apply_actions(actions)

        except Exception as e:
            import traceback
            traceback.print_exc()
            pass

    def _get_vehicle_tensor(self, vehicle_ids):
        """将车辆ID列表转换为特征tensor"""
        import torch
        if not vehicle_ids:
            return None

        features = []
        for veh_id in vehicle_ids[:5]:  # 最多5辆车
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                accel = traci.vehicle.getAcceleration(veh_id)
                features.append([speed, accel])
            except:
                features.append([0.0, 0.0])

        # 填充到5辆车
        while len(features) < 5:
            features.append([0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _apply_actions(self, actions):
        """应用模型输出的动作"""
        for junc_id, action in actions.items():
            if junc_id not in self.agents:
                continue

            agent = self.agents[junc_id]
            controlled = agent.get_controlled_vehicles()

            if controlled['main'] and 'main' in action:
                for veh_id in controlled['main'][:1]:
                    try:
                        action_value = action['main'].item()
                        speed_limit = 13.89
                        target_speed = speed_limit * (0.3 + 0.9 * action_value)
                        target_speed = max(0.0, min(target_speed, speed_limit * 1.2))
                        traci.vehicle.setSpeed(veh_id, target_speed)
                    except Exception as e:
                        continue

            if controlled['ramp'] and 'ramp' in action:
                for veh_id in controlled['ramp'][:1]:
                    try:
                        action_value = action['ramp'].item()
                        speed_limit = 13.89
                        target_speed = speed_limit * (0.3 + 0.9 * action_value)
                        target_speed = max(0.0, min(target_speed, speed_limit * 1.2))
                        traci.vehicle.setSpeed(veh_id, target_speed)
                    except Exception as e:
                        continue

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
                tl_states[f'{tl_id}_state'] = 'unknown'
                tl_states[f'{tl_id}_phase'] = -1
                tl_states[f'{tl_id}_remaining_time'] = -1

        return tl_states

    def get_vehicle_od(self, veh_id):
        """获取车辆OD信息和车辆类型的原始maxSpeed配置"""
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

            # 获取车辆类型
            vehicle_type = traci.vehicle.getTypeID(veh_id)

            # 获取该车辆类型的原始maxSpeed配置(从route文件中读取的)
            original_max_speed = self.vehicle_type_maxspeed.get(vehicle_type, None)

            od_info = {
                'origin': origin,
                'destination': destination,
                'route_length': len(route),
                'vehicle_type': vehicle_type,
                'original_max_speed': original_max_speed  # 配置文件中的原始maxSpeed
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

    def calculate_traveled_distance(self, veh_id, route_info):
        """计算车辆已行驶距离"""
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

    def collect_step_data(self, step):
        """收集每个时间步的数据"""
        current_time = step * self.step_length

        # 获取当前活跃车辆
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

        # 收集车辆级数据
        for veh_id in current_vehicle_ids:
            try:
                speed = traci.vehicle.getSpeed(veh_id)  # 瞬时速度
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
                    'speed': speed,  # 瞬时速度
                    'position': position,
                    'edge_id': edge_id,
                    'route_index': route_index,
                    'traveled_distance': traveled_distance,
                    'route_length': route_info['route_length'],
                    'completion_rate': completion_rate,
                    'origin': od_info['origin'],
                    'destination': od_info['destination'],
                    'route_edges_count': od_info['route_length'],
                    'max_speed': od_info['original_max_speed'],  # 车辆类型的原始maxSpeed配置
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

    def save_to_pickle(self, output_dir="competition_results"):
        """保存数据到pickle文件"""
        print(f"\n[第三部分] 正在保存仿真数据到Pickle格式...")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 检查maxSpeed是否被修改
        maxspeed_violations = self.check_maxspeed_violations()

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
                'vehicle_type_maxspeed': self.vehicle_type_maxspeed  # 原始车辆类型配置
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
                'maxspeed_violations': maxspeed_violations  # maxSpeed违规检测结果
            }
        }

        # 保存pickle文件
        pickle_file = os.path.join(output_dir, f"submit.pkl")

        with open(pickle_file, 'wb') as f:
            pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 计算文件大小
        file_size = os.path.getsize(pickle_file) / (1024 * 1024)  # MB

        # 另外保存一个汇总JSON文件(可选,方便快速查看)
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
        summary_data = {
            'parameters': data_package['parameters'],
            'statistics': data_package['statistics']
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # 输出保存结果
        print(f"\n{'=' * 70}")
        print(f"数据保存统计")
        print(f"{'=' * 70}")
        print(f"[OK] Pickle文件已保存: {pickle_file}")
        print(f"  - 文件大小: {file_size:.2f} MB")
        print(f"  - 时间步数据记录: {len(self.step_data):,} 条")
        print(f"  - 车辆数据记录: {len(self.vehicle_data):,} 条")
        print(f"  - 唯一车辆数: {len(self.route_data):,} 辆")
        print(f"\n[OK] 汇总JSON已保存: {summary_file}")
        print(f"{'=' * 70}")

        # 数据统计报告
        print(f"\n{'=' * 70}")
        print(f"仿真结果统计")
        print(f"{'=' * 70}")
        print(f"理论总需求:     {self.total_demand:.0f} 车辆")
        print(f"实际累计出发:   {self.cumulative_departed} 车辆")
        print(f"实际累计到达:   {self.cumulative_arrived} 车辆")
        print(f"数据记录总数:   {len(self.step_data) + len(self.vehicle_data):,} 条")

        print(f"{'=' * 70}")

        return {
            'pickle_file': pickle_file,
            'summary_file': summary_file,
            'file_size_mb': file_size,
            'maxspeed_violations': maxspeed_violations
        }

    def check_maxspeed_violations(self):
        """检查是否存在maxSpeed违规（这里不再适用，因为现在记录的是瞬时速度）"""
        # 由于现在max_speed字段记录的是车辆类型的原始配置，不再需要检测违规
        # 但为了保持数据结构完整性，仍然返回基本信息
        return {
            'has_violations': False,
            'violations': {},
            'total_vehicle_types_checked': len(self.vehicle_type_maxspeed),
        }

    def run(self, max_steps=3600, use_gui=True):
        """运行完整的仿真流程"""
        print("\n开始运行SUMO竞赛仿真框架...")
        print(f"最大步数: {max_steps}\n")

        # 第一部分: 初始化Baseline环境
        if not self.initialize_environment(use_gui=use_gui, max_steps=max_steps):
            print("[ERROR] 环境初始化失败")
            return False

        # 仿真主循环
        print(f"\n{'=' * 70}")
        print("[第二部分] 开始运行控制算法...")
        print(f"{'=' * 70}\n")

        step = 0
        try:
            while step < max_steps:
                # 执行仿真步
                traci.simulationStep()

                # 第二部分: 应用控制算法
                self.apply_control_algorithm(step)

                # 第三部分: 收集数据
                self.collect_step_data(step)

                step += 1

                # 检查仿真是否结束
                if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                    print(f"\n仿真自然结束于步骤 {step}")
                    break

        except Exception as e:
            print(f"\n[ERROR] 仿真过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            traci.close()

        # 第三部分: 保存数据
        print(f"\n{'=' * 70}")
        result = self.save_to_pickle()

        print(f"\n✅ 仿真完成!")
        print(f"\n可使用此Pickle文件进行评测提交: {result['pickle_file']}")

        return True


# ============================================================================
# 数据读取辅助函数
# ============================================================================

def load_pickle_data(pickle_file):
    """
    从pickle文件加载数据

    参数:
        pickle_file: pickle文件路径

    返回:
        data_package: 包含所有数据的字典
            - parameters: 仿真参数
            - step_data: 时间步数据列表
            - vehicle_data: 车辆数据列表
            - route_data: 路径数据字典
            - vehicle_od_data: OD数据字典
            - statistics: 统计数据
    """
    print(f"正在加载数据: {pickle_file}")

    with open(pickle_file, 'rb') as f:
        data_package = pickle.load(f)

    print(f"[OK] 数据加载成功!")
    print(f"  - 参数: {len(data_package['parameters'])} 项")
    print(f"  - 时间步数据: {len(data_package['step_data']):,} 条")
    print(f"  - 车辆数据: {len(data_package['vehicle_data']):,} 条")
    print(f"  - 路径数据: {len(data_package['route_data']):,} 辆车")

    return data_package


def analyze_pickle_data(pickle_file):
    """
    分析pickle文件中的数据

    参数:
        pickle_file: pickle文件路径
    """
    data = load_pickle_data(pickle_file)

    print(f"\n{'=' * 70}")
    print("数据分析报告")
    print(f"{'=' * 70}")

    # 仿真参数
    params = data['parameters']
    print(f"\n仿真参数:")
    print(f"  - 仿真时长: {params['simulation_time']:.2f} 秒")
    print(f"  - 总步数: {params['total_steps']}")
    print(f"  - 时间步长: {params['step_length']:.2f} 秒")
    print(f"  - 理论总需求: {params['total_demand']:.0f} 车辆")
    print(f"  - 实际出发: {params['final_departed']} 车辆")
    print(f"  - 实际到达: {params['final_arrived']} 车辆")

    # 车辆类型配置
    if 'vehicle_type_maxspeed' in params:
        print(f"\n车辆类型原始maxSpeed配置:")
        for vtype, max_speed in params['vehicle_type_maxspeed'].items():
            print(f"  - {vtype}: {max_speed:.2f} m/s")

    # 数据量统计
    print(f"\n数据量统计:")
    print(f"  - 时间步记录: {len(data['step_data']):,} 条")
    print(f"  - 车辆记录: {len(data['vehicle_data']):,} 条")
    print(f"  - 唯一车辆: {len(data['route_data']):,} 辆")

    # 转换为DataFrame进行分析
    if data['vehicle_data']:
        df = pd.DataFrame(data['vehicle_data'])
        print(f"\n车辆数据分析:")
        print(f"  - 平均瞬时速度: {df['speed'].mean():.2f} m/s")
        print(f"  - 最大瞬时速度: {df['speed'].max():.2f} m/s")
        print(f"  - 最小瞬时速度: {df['speed'].min():.2f} m/s")
        print(f"  - 平均完成率: {df['completion_rate'].mean():.2%}")
        print(f"  - 唯一OD对数: {df.groupby('vehicle_id')[['origin', 'destination']].first().drop_duplicates().shape[0]}")

        # maxSpeed统计（现在是车辆类型配置）
        if 'max_speed' in df.columns:
            print(f"\n车辆类型maxSpeed配置统计:")
            unique_maxspeeds = df.groupby('vehicle_type')['max_speed'].first()
            for vtype, maxspeed in unique_maxspeeds.items():
                print(f"  - {vtype}: {maxspeed:.2f} m/s")

    if data['step_data']:
        step_df = pd.DataFrame(data['step_data'])
        print(f"\n时间步数据分析:")
        print(f"  - 最大活跃车辆数: {step_df['active_vehicles'].max()}")
        print(f"  - 平均活跃车辆数: {step_df['active_vehicles'].mean():.2f}")

    print(f"{'=' * 70}")

    return data


def export_to_csv(pickle_file, output_dir=None):
    """
    将pickle数据导出为CSV文件(用于进一步分析)

    参数:
        pickle_file: pickle文件路径
        output_dir: 输出目录,默认与pickle文件同目录
    """
    data = load_pickle_data(pickle_file)

    if output_dir is None:
        output_dir = os.path.dirname(pickle_file)

    timestamp = data['parameters']['collection_timestamp']

    # 导出时间步数据
    if data['step_data']:
        step_csv = os.path.join(output_dir, f"step_data_{timestamp}.csv")
        step_df = pd.DataFrame(data['step_data'])
        step_df.to_csv(step_csv, index=False, encoding='utf-8-sig')
        print(f"[OK] 时间步数据已导出: {step_csv}")

    # 导出车辆数据
    if data['vehicle_data']:
        vehicle_csv = os.path.join(output_dir, f"vehicle_data_{timestamp}.csv")
        vehicle_df = pd.DataFrame(data['vehicle_data'])
        vehicle_df.to_csv(vehicle_csv, index=False, encoding='utf-8-sig')
        print(f"[OK] 车辆数据已导出: {vehicle_csv}")

    # 导出参数
    params_csv = os.path.join(output_dir, f"parameters_{timestamp}.csv")
    params_df = pd.DataFrame([
        {'参数名': k, '参数值': str(v)}
        for k, v in data['parameters'].items()
    ])
    params_df.to_csv(params_csv, index=False, encoding='utf-8-sig')
    print(f"[OK] 参数已导出: {params_csv}")


def main():
    """主函数 - 参赛者使用入口"""

    # ========================================================================
    # 配置区域 - 参赛者修改此处
    # ========================================================================

    # 方式1: 从命令行参数获取配置文件路径
    if len(sys.argv) > 1:
        sumo_cfg = sys.argv[1]
    else:
        # 方式2: 直接指定配置文件路径
        sumo_cfg = ".\sumo.sumocfg"

    # 仿真参数设置
    MAX_STEPS = 3600  # 最大仿真步数
    USE_GUI = True  # 是否使用GUI界面

    # ========================================================================

    # 检查配置文件是否存在
    if not os.path.exists(sumo_cfg):
        print(f"[ERROR] 配置文件不存在: {sumo_cfg}")
        print("\n请修改main()函数中的sumo_cfg路径,或使用命令行参数:")
        print(f"python {sys.argv[0]} <your_config_file.sumocfg>")
        return

    try:
        # 创建框架实例
        framework = SUMOCompetitionFramework(sumo_cfg)

        # 运行仿真
        framework.run(max_steps=MAX_STEPS, use_gui=USE_GUI)

    except Exception as e:
        print(f"\n[ERROR] 程序运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()