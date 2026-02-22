"""
基于Baseline OCR比较的奖励函数

核心思想：
1. 预先记录baseline（专家/前次训练）在每个步数的OCR值
2. 训练时比较：当前OCR - baselineOCR（相同步数）
3. 奖励 = OCR增量 × 权重
"""
import numpy as np
import pickle
import os
import sys
import argparse
from typing import Dict, List
from collections import defaultdict

# 直接使用libsumo
import libsumo as traci

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BaselineOCRRewardCalculator:
    """
    基于Baseline OCR比较的奖励计算器
    """

    def __init__(self, baseline_file: str = None, reward_weight: float = 100.0):
        """
        Args:
            baseline_file: baseline OCR数据文件路径
            reward_weight: OCR增量奖励权重
        """
        self.reward_weight = reward_weight
        self.baseline_file = baseline_file

        # 加载baseline OCR数据
        self.baseline_ocr_history = {}  # {step: ocr}

        if baseline_file and os.path.exists(baseline_file):
            self._load_baseline(baseline_file)
        else:
            print(f"[警告] 未找到baseline文件，将使用固定baseline OCR = 0.95")

        # 瞬时奖励权重（辅助）
        self.instant_weights = {
            'speed': 0.05,
            'throughput': 1.0,
            'queue': 0.02,
            'waiting': 0.005,
            'conflict': 0.05,
            'survival': 0.000,
        }

        # 历史追踪
        self.previous_in_zone = defaultdict(int)
        self.current_ocr_cache = None

    def _load_baseline(self, baseline_file: str):
        """加载baseline OCR数据"""
        try:
            with open(baseline_file, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                if 'ocr_history' in data:
                    self.baseline_ocr_history = data['ocr_history']
                else:
                    self.baseline_ocr_history = data
            elif isinstance(data, list):
                self.baseline_ocr_history = {item['step']: item['ocr'] for item in data}

            print(f"✓ 加载baseline OCR数据: {len(self.baseline_ocr_history)} 个数据点")

        except Exception as e:
            print(f"✗ 加载baseline失败: {e}")

    def get_baseline_ocr(self, step: int) -> float:
        """获取指定步数的baseline OCR（线性插值）"""
        if not self.baseline_ocr_history:
            return 0.95

        if step in self.baseline_ocr_history:
            return self.baseline_ocr_history[step]

        steps = sorted(self.baseline_ocr_history.keys())

        if step < steps[0]:
            return self.baseline_ocr_history[steps[0]]
        if step > steps[-1]:
            return self.baseline_ocr_history[steps[-1]]

        # 线性插值
        for i in range(len(steps) - 1):
            if steps[i] <= step <= steps[i+1]:
                s1, s2 = steps[i], steps[i+1]
                o1, o2 = self.baseline_ocr_history[s1], self.baseline_ocr_history[s2]
                ratio = (step - s1) / (s2 - s1)
                return o1 + ratio * (o2 - o1)

        return 0.95

    def get_final_baseline_ocr(self) -> float:
        """获取最终baseline OCR值（用于评分显示）"""
        if not self.baseline_ocr_history:
            return 0.95
        steps = sorted(self.baseline_ocr_history.keys())
        return self.baseline_ocr_history[steps[-1]]

    def compute_rewards(self, agents: Dict, env_stats: Dict) -> Dict[str, float]:
        """计算奖励"""
        current_ocr = env_stats.get('ocr', 0.0)
        current_step = env_stats.get('step', 0)

        self.current_ocr_cache = current_ocr
        baseline_ocr = self.get_baseline_ocr(current_step)
        ocr_delta = current_ocr - baseline_ocr
        ocr_reward = ocr_delta * self.reward_weight

        rewards = {}

        for junc_id, agent in agents.items():
            state = agent.current_state
            if state is None:
                rewards[junc_id] = 0.0
                continue

            # 瞬时辅助奖励
            speed_score = min(state.main_speed / 15.0, 1.0)
            speed_reward = speed_score * self.instant_weights['speed']

            current_in_zone = len(state.main_vehicles) + len(state.ramp_vehicles)
            departed_delta = max(0, self.previous_in_zone[junc_id] - current_in_zone)
            self.previous_in_zone[junc_id] = current_in_zone
            throughput_reward = departed_delta * self.instant_weights['throughput']

            queue_penalty = -(
                state.main_queue_length * self.instant_weights['queue'] * 0.5 +
                state.ramp_queue_length * self.instant_weights['queue']
            )

            waiting_penalty = 0.0
            if state.ramp_waiting_time > 30:
                waiting_penalty = -(state.ramp_waiting_time - 30) * self.instant_weights['waiting']

            conflict_penalty = -state.conflict_risk * self.instant_weights['conflict']

            total_reward = (
                ocr_reward +
                speed_reward +
                throughput_reward +
                queue_penalty +
                waiting_penalty +
                conflict_penalty +
                self.instant_weights['survival']
            )

            total_reward = np.clip(total_reward, -10.0, 10.0)
            rewards[junc_id] = total_reward

            if hasattr(agent, 'reward_breakdown'):
                agent.reward_breakdown = {
                    'ocr_delta': ocr_delta,
                    'ocr_reward': ocr_reward,
                    'baseline_ocr': baseline_ocr,
                    'current_ocr': current_ocr,
                    'speed_reward': speed_reward,
                    'throughput_reward': throughput_reward,
                    'queue_penalty': queue_penalty,
                    'waiting_penalty': waiting_penalty,
                    'conflict_penalty': conflict_penalty,
                    'total': total_reward
                }

        return rewards

    def reset(self):
        """重置追踪状态"""
        self.previous_in_zone.clear()
        self.current_ocr_cache = None


class BaselineOCRCollector:
    """收集Baseline OCR数据（使用libsumo + 订阅模式）"""

    def __init__(self, record_interval: int = 100):
        self.record_interval = record_interval
        self.ocr_history = {}

    def record_ocr(self, step: int, ocr: float):
        """记录OCR值"""
        if step % self.record_interval == 0:
            self.ocr_history[step] = ocr

    def save(self, output_file: str):
        """保存baseline数据"""
        data = {
            'ocr_history': self.ocr_history,
            'num_records': len(self.ocr_history),
            'interval': self.record_interval
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"[OK] Baseline OCR saved: {output_file}")
        print(f"  Records: {len(self.ocr_history)}")


def generate_baseline_from_expert(sumo_cfg: str, output_file: str, max_steps: int = 3600):
    """
    从专家策略生成baseline OCR数据（使用libsumo + 订阅模式）

    参考：relu_based\rl_traffic\fast_pkl_generator.py
    """
    print("=" * 70)
    print("生成Baseline OCR数据（专家策略 - libsumo订阅模式）")
    print("=" * 70)

    # 启动SUMO
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "true", "--seed", "42"]
    traci.start(sumo_cmd)
    print("[OK] SUMO started")

    # 配置vType参数（专家策略）
    traci.vehicletype.setImperfection('CV', 0.0)
    traci.vehicletype.setTau('CV', 0.9)
    traci.vehicletype.setAccel('CV', 0.8)
    traci.vehicletype.setDecel('CV', 1.5)

    traci.vehicletype.setImperfection('HV', 0.0)
    traci.vehicletype.setTau('HV', 0.9)
    traci.vehicletype.setAccel('HV', 0.8)
    traci.vehicletype.setDecel('HV', 1.5)
    print("[OK] vType configured")

    # 订阅管理
    subscribed_vehicles = set()

    # 道路拓扑
    NEXT_EDGE = {
        '-E13': '-E12', '-E12': '-E11', '-E11': '-E10', '-E10': '-E9',
        '-E9': '-E8', '-E8': '-E7', '-E7': '-E6', '-E6': '-E5',
        '-E5': '-E3', '-E3': '-E2', '-E2': '-E1',
        'E1': 'E2', 'E2': 'E3', 'E3': 'E5', 'E5': 'E6',
        'E6': 'E7', 'E7': 'E8', 'E8': 'E9', 'E9': 'E10',
        'E10': 'E11', 'E11': 'E12', 'E12': 'E13',
        'E23': '-E2', 'E15': 'E10', 'E17': '-E10', 'E19': '-E12',
    }

    # 主动控制参数
    APPROACH_DIST = 50.0
    CONGEST_SPEED = 5.0
    LOOKAHEAD = 2
    SPEED_FACTOR = 1.5
    SPEED_FLOOR = 3.0

    controlled_cvs = set()
    edge_speed_cache_step = -1
    edge_speeds = {}

    # 累计统计
    cumulative_arrived = 0
    cumulative_departed = 0
    route_data = {}

    # OCR收集器（每10步记录一次，兼顾密度和稳定性）
    collector = BaselineOCRCollector(record_interval=10)

    try:
        # 运行仿真
        print("\n开始收集baseline OCR数据...")
        for step in range(max_steps):
            current_time = traci.simulation.getTime()

            # 订阅新车辆
            current_vehicles = traci.vehicle.getIDList()
            new_vehicles = set(current_vehicles) - subscribed_vehicles

            if new_vehicles:
                for veh_id in new_vehicles:
                    try:
                        traci.vehicle.subscribe(veh_id, [
                            traci.VAR_ROAD_ID,
                            traci.VAR_LANEPOSITION,
                            traci.VAR_SPEED,
                            traci.VAR_ROUTE_INDEX,
                        ])
                        subscribed_vehicles.add(veh_id)
                    except:
                        pass

            # 仿真一步
            traci.simulationStep()

            # 更新订阅结果
            for veh_id in subscribed_vehicles:
                try:
                    result = traci.vehicle.getSubscriptionResults(veh_id)
                    if result:
                        road_id = result.get(traci.VAR_ROAD_ID, "")
                        lane_position = result.get(traci.VAR_LANEPOSITION, 0.0)
                        speed = result.get(traci.VAR_SPEED, 0.0)
                        route_index = result.get(traci.VAR_ROUTE_INDEX, 0)

                        # 获取路线信息（只获取一次）
                        if veh_id not in route_data:
                            try:
                                route_edges = traci.vehicle.getRoute(veh_id)
                                route_length = sum(traci.edge.getLength(e) for e in route_edges if e in road_id)
                                route_data[veh_id] = {
                                    'route_edges': route_edges,
                                    'route_length': route_length
                                }
                            except:
                                route_data[veh_id] = {'route_edges': [], 'route_length': 100.0}

                        # 计算完成度
                        if veh_id in route_data:
                            route_info = route_data[veh_id]
                            traveled = 0.0
                            for i, edge in enumerate(route_info['route_edges']):
                                if i < route_index:
                                    try:
                                        traveled += traci.edge.getLength(edge)
                                    except:
                                        traveled += 100.0
                                elif i == route_index:
                                    traveled += lane_position
                                    break

                            completion = min(traveled / max(route_info['route_length'], 1), 1.0)
                except:
                    pass

            # 累计统计
            cumulative_arrived += traci.simulation.getArrivedNumber()
            cumulative_departed += traci.simulation.getDepartedNumber()

            # 计算OCR（每100步）
            if step % 100 == 0:
                enroute_completion = 0.0
                for veh_id in current_vehicles:
                    if veh_id in route_data:
                        route_info = route_data[veh_id]
                        traveled = 0.0
                        try:
                            result = traci.vehicle.getSubscriptionResults(veh_id)
                            if result:
                                lane_position = result.get(traci.VAR_LANEPOSITION, 0.0)
                                route_index = result.get(traci.VAR_ROUTE_INDEX, 0)

                                for i, edge in enumerate(route_info['route_edges']):
                                    if i < route_index:
                                        traveled += traci.edge.getLength(edge)
                                    elif i == route_index:
                                        traveled += lane_position
                                        break

                                completion = min(traveled / max(route_info['route_length'], 1), 1.0)
                                enroute_completion += completion
                        except:
                            pass

                n_total = cumulative_arrived + len(current_vehicles)
                ocr = (cumulative_arrived + enroute_completion) / max(n_total, 1)

                collector.record_ocr(step, ocr)

                if step % 100 == 0:
                    print(f"  步骤 {step}: OCR = {ocr:.4f}")

            # 应用主动控制
            # 采集各边平均速度（缓存）
            if edge_speed_cache_step != step:
                all_edges = set(NEXT_EDGE.keys()) | set(NEXT_EDGE.values())
                edge_speeds = {}
                for eid in all_edges:
                    try:
                        edge_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)
                    except:
                        edge_speeds[eid] = 13.89
                edge_speed_cache_step = step

            new_controlled = set()

            for veh_id in subscribed_vehicles:
                try:
                    if traci.vehicle.getTypeID(veh_id) != 'CV':
                        continue

                    result = traci.vehicle.getSubscriptionResults(veh_id)
                    if not result:
                        continue

                    road_id = result.get(traci.VAR_ROAD_ID, "")
                    lane_position = result.get(traci.VAR_LANEPOSITION, 0.0)
                    speed = result.get(traci.VAR_SPEED, 0.0)

                    if road_id.startswith(':') or road_id not in NEXT_EDGE:
                        continue

                    # 获取车道长度
                    try:
                        lane_len = traci.lane.getLength(f"{road_id}_0")
                    except:
                        try:
                            lane_len = traci.edge.getLength(road_id)
                        except:
                            continue

                    dist_to_end = lane_len - lane_position

                    if dist_to_end > APPROACH_DIST:
                        if veh_id in controlled_cvs:
                            traci.vehicle.setSpeed(veh_id, -1)
                        continue

                    # 检测下游拥堵
                    congested = False
                    min_ds_speed = 13.89
                    nxt = road_id

                    for _ in range(LOOKAHEAD):
                        nxt = NEXT_EDGE.get(nxt)
                        if nxt is None:
                            break
                        ds_speed = edge_speeds.get(nxt, 13.89)
                        if ds_speed < CONGEST_SPEED:
                            congested = True
                            min_ds_speed = min(min_ds_speed, ds_speed)

                    # 下游拥堵时减速
                    if congested:
                        target = max(min_ds_speed * SPEED_FACTOR, SPEED_FLOOR)
                        target = min(target, 13.89)
                        if speed > target:
                            traci.vehicle.slowDown(veh_id, target, 3.0)
                            new_controlled.add(veh_id)
                    elif veh_id in controlled_cvs:
                        traci.vehicle.setSpeed(veh_id, -1)

                except:
                    continue

            # 释放不再控制的车辆
            for veh_id in controlled_cvs - new_controlled:
                try:
                    traci.vehicle.setSpeed(veh_id, -1)
                except:
                    pass

            controlled_cvs = new_controlled

            # 清理已离开的车辆
            left_vehicles = subscribed_vehicles - set(current_vehicles)
            subscribed_vehicles -= left_vehicles
            for veh_id in left_vehicles:
                route_data.pop(veh_id, None)

            # 检查是否结束
            if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                print(f"\n仿真自然结束于步骤 {step}")
                break

    finally:
        traci.close()

    # 保存baseline
    collector.save(output_file)

    print("\n[OK] Baseline generation completed!")
    print(f"  Final OCR: {ocr:.4f}")
    print(f"  Output: {output_file}")


def generate_baseline_from_model(sumo_cfg: str, model_path: str, output_file: str,
                                  max_steps: int = 3600, device: str = 'cpu'):
    """
    从预训练模型生成baseline OCR数据

    Args:
        sumo_cfg: SUMO配置文件路径
        model_path: 预训练模型路径
        output_file: 输出文件路径
        max_steps: 最大仿真步数
        device: 设备（'cpu' 或 'cuda'）
    """
    print("=" * 70)
    print("生成Baseline OCR数据（预训练模型策略）")
    print("=" * 70)
    print(f"模型: {model_path}")
    print(f"设备: {device}")

    # 导入必要的模块
    import torch
    from junction_agent import JUNCTION_CONFIGS, JunctionAgent
    from junction_network import create_junction_model, NetworkConfig

    # 加载模型
    print("\n加载预训练模型...")
    model = create_junction_model(JUNCTION_CONFIGS, NetworkConfig())
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("[OK] 模型加载成功")

    # 创建路口智能体
    agents = {}
    for junc_id, config in JUNCTION_CONFIGS.items():
        agents[junc_id] = JunctionAgent(config)

    # 启动SUMO
    print("\n启动SUMO...")
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "true", "--seed", "42"]
    traci.start(sumo_cmd)
    print("[OK] SUMO已启动")

    # 配置vType参数
    traci.vehicletype.setImperfection('CV', 0.0)
    traci.vehicletype.setTau('CV', 0.9)
    traci.vehicletype.setAccel('CV', 0.8)
    traci.vehicletype.setDecel('CV', 1.5)
    print("[OK] vType已配置")

    # 设置 junction_agent 的 traci 连接
    import junction_agent
    junction_agent.traci = traci

    # OCR收集器（每10步记录一次）
    collector = BaselineOCRCollector(record_interval=10)

    # 累计统计
    cumulative_arrived = 0
    cumulative_departed = 0

    try:
        print("\n开始运行仿真并收集baseline OCR数据...")
        for step in range(max_steps):
            # 1. 获取观测
            observations = {}
            for junc_id, agent in agents.items():
                try:
                    obs = agent.get_observation()
                    if obs is not None:
                        observations[junc_id] = obs
                except:
                    pass

            # 2. 模型推理（确定性模式）
            with torch.no_grad():
                obs_tensors = {jid: obs['state'].unsqueeze(0).to(device)
                              for jid, obs in observations.items()}

                model_output = model(obs_tensors, {}, deterministic=True)

            # 3. 应用动作
            for junc_id, agent in agents.items():
                if junc_id not in model_output[0]:
                    continue

                actions = model_output[0][junc_id]
                main_action = actions['main'].cpu().item() / 10.0  # 归一化到 [0, 1]
                ramp_action = actions['ramp'].cpu().item() / 10.0

                try:
                    agent.apply_control(main_action, ramp_action)
                except:
                    pass

            # 4. 仿真一步
            traci.simulationStep()

            # 5. 累计统计
            cumulative_arrived += traci.simulation.getArrivedNumber()
            cumulative_departed += traci.simulation.getDepartedNumber()

            # 6. 计算OCR（每10步）
            if step % 10 == 0:
                current_vehicles = traci.vehicle.getIDList()

                # 计算在途车辆的完成度
                enroute_completion = 0.0
                for veh_id in current_vehicles:
                    try:
                        route_edges = traci.vehicle.getRoute(veh_id)
                        route_index = traci.vehicle.getRouteIndex(veh_id)
                        lane_position = traci.vehicle.getLanePosition(veh_id)

                        traveled = 0.0
                        for i, edge in enumerate(route_edges):
                            if i < route_index:
                                try:
                                    traveled += traci.edge.getLength(edge)
                                except:
                                    traveled += 100.0
                            elif i == route_index:
                                traveled += lane_position
                                break

                        route_length = sum(traci.edge.getLength(e) for e in route_edges)
                        completion = min(traveled / max(route_length, 1), 1.0)
                        enroute_completion += completion
                    except:
                        pass

                # 计算OCR
                n_total = cumulative_arrived + len(current_vehicles)
                ocr = (cumulative_arrived + enroute_completion) / max(n_total, 1)

                collector.record_ocr(step, ocr)

                if step % 100 == 0:
                    print(f"  步骤 {step}: OCR = {ocr:.4f}, 车辆数 = {len(current_vehicles)}")

            # 检查是否结束
            if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                print(f"\n仿真自然结束于步骤 {step}")
                break

    finally:
        traci.close()

    # 保存baseline
    collector.save(output_file)

    print("\n[OK] Baseline生成完成！")
    print(f"  最终OCR: {ocr:.4f}")
    print(f"  输出: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成Baseline OCR数据（每10步记录）')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg',
                        help='SUMO配置文件')
    parser.add_argument('--output', type=str, default='baseline_ocr/expert_baseline_10step.pkl',
                        help='输出文件路径')
    parser.add_argument('--max-steps', type=int, default=3600,
                        help='最大步数')
    parser.add_argument('--model', type=str, default=None,
                        help='预训练模型路径（如果指定，使用模型策略；否则使用专家策略）')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备（cpu 或 cuda）')

    args = parser.parse_args()

    # 生成baseline
    if args.model:
        # 使用预训练模型
        print(f"\n使用预训练模型生成baseline: {args.model}")
        generate_baseline_from_model(args.sumo_cfg, args.model, args.output, args.max_steps, args.device)
    else:
        # 使用专家策略
        print("\n使用专家策略生成baseline")
        generate_baseline_from_expert(args.sumo_cfg, args.output, args.max_steps)

    print("\n[OK] Done!")
    print("\nNext step: Start training (will auto-load baseline file)")
    print("  python rl_train.py --pretrained bc_checkpoints/best_model.pt")
