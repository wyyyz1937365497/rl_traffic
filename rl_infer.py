"""
多智能体路口交通控制 - 推理脚本
使用训练好的模型进行SUMO仿真推理
"""

import os
import sys
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# 导入必要的模块
try:
    import libsumo as traci
    USE_LIBSUMO = True
except ImportError:
    try:
        import traci
        USE_LIBSUMO = False
    except ImportError:
        print("请安装traci: pip install traci")
        sys.exit(1)

from junction_network import MultiJunctionModel, NetworkConfig, JUNCTION_CONFIGS


def create_junction_model(junction_configs):
    """创建多路口模型"""
    model = MultiJunctionModel(junction_configs)
    return model


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")

    # 创建模型
    model = create_junction_model(JUNCTION_CONFIGS)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"✓ 模型已加载到 {device}")
    return model


class InferenceAgent:
    """推理智能体"""

    def __init__(self, agent_id, agent_type, config, model, device):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.model = model
        self.device = device

        # 观察数据
        self.current_state = None
        self.controlled_vehicles = {'main': [], 'ramp': [], 'diverge': []}

        # 边缘ID
        self.edge_ids = config.get('edges', {})

    def reset(self):
        """重置智能体"""
        self.current_state = None
        self.controlled_vehicles = {'main': [], 'ramp': [], 'diverge': []}

    def get_state_vector(self):
        """获取状态向量"""
        if self.current_state is None:
            return np.zeros(17)

        return np.array([
            self.current_state.get('main_queue_length', 0),
            self.current_state.get('ramp_queue_length', 0),
            self.current_state.get('main_speed', 0) / 20.0,
            self.current_state.get('ramp_speed', 0) / 20.0,
            self.current_state.get('main_density', 0) / 0.5,
            self.current_state.get('ramp_density', 0) / 0.5,
            self.current_state.get('ramp_waiting_time', 0) / 60.0,
            self.current_state.get('gap_size', 0) / 10.0,
            self.current_state.get('gap_speed_diff', 0) / 20.0,
            float(self.current_state.get('has_cv', False)),
            float(self.current_state.get('conflict_risk', 0)),
            self.current_state.get('main_stop_count', 0) / 10.0,
            self.current_state.get('ramp_stop_count', 0) / 10.0,
            self.current_state.get('throughput', 0) / 100.0,
            0.0,  # phase_main
            0.0,  # phase_ramp
            0.0   # time_step
        ])

    def get_controlled_vehicles(self):
        """获取受控车辆"""
        return self.controlled_vehicles

    def observe(self, traci_conn):
        """观察环境"""
        try:
            # 获取边缘信息
            main_edges = self.edge_ids.get('main', [])
            ramp_edges = self.edge_ids.get('ramp', [])
            merge_edges = self.edge_ids.get('merge', [])

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

            # 计算间隙
            gap_size = 0
            gap_speed_diff = 0

            if main_vehicles and ramp_vehicles:
                # 简化计算
                gap_size = 5.0
                gap_speed_diff = abs(main_speed - ramp_speed)

            # 计算冲突风险
            conflict_risk = min(len(main_vehicles), len(ramp_vehicles)) / 20.0

            # 检测CV
            has_cv = any(
                traci_conn.vehicle.getTypeID(v) == 'CV'
                for v in main_vehicles + ramp_vehicles
                if v in traci_conn.vehicle.getIDList()
            )

            # 更新状态
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

            # 选择受控车辆
            self.controlled_vehicles = {
                'main': main_vehicles[:1] if main_vehicles else [],
                'ramp': ramp_vehicles[:1] if ramp_vehicles else [],
                'diverge': []
            }

            return self.current_state

        except Exception as e:
            print(f"观察错误: {e}")
            return None


def get_vehicle_features(vehicle_ids, traci_conn, device):
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
                0.0,  # route_index
                0.0
            ])
        except Exception as e:
            print(f"获取车辆 {veh_id} 特征失败: {e}")
            features.append([0.0] * 8)

    # 补齐到10个
    while len(features) < 10:
        features.append([0.0] * 8)

    return torch.tensor(features, dtype=torch.float32, device=device)


def run_inference(args):
    """运行推理"""
    print("=" * 70)
    print("多智能体路口控制 - 推理")
    print("=" * 70)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载模型
    model = load_model(args.model, device)

    # 创建智能体
    agents = {}
    for junc_id, config in JUNCTION_CONFIGS.items():
        agents[junc_id] = InferenceAgent(
            junc_id, config['type'], config, model, device
        )

    # 启动SUMO
    print(f"\n启动SUMO: {args.sumo_cfg}")
    sumo_binary = "sumo"

    sumo_cmd = [sumo_binary, "-c", args.sumo_cfg, "--no-warnings", "true"]

    if USE_LIBSUMO:
        sumo_cmd.extend(["--seed", str(args.seed)])
        traci.start(sumo_cmd)
    else:
        sumo_cmd.extend(["--remote-port", "0", "--seed", str(args.seed)])
        traci.start(sumo_cmd)

    print("✓ SUMO已启动")

    # 预热
    print("\n预热仿真 (10步)...")
    for _ in range(10):
        traci.simulationStep()

    for agent in agents.values():
        agent.observe(traci)

    # 推理循环
    print("\n开始推理...")
    print(f"仿真步数: {args.steps}")

    total_reward = 0.0
    start_time = time.time()

    for step in range(args.steps):
        # 准备输入
        obs_tensors = {}
        vehicle_obs = {}

        for junc_id, agent in agents.items():
            state_vec = agent.get_state_vector()
            obs_tensors[junc_id] = torch.tensor(
                state_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            controlled = agent.get_controlled_vehicles()
            vehicle_obs[junc_id] = {
                'main': get_vehicle_features(controlled['main'], traci, device) if controlled['main'] else None,
                'ramp': get_vehicle_features(controlled['ramp'], traci, device) if controlled['ramp'] else None,
                'diverge': get_vehicle_features(controlled['diverge'], traci, device) if controlled['diverge'] else None
            }

        # 模型推理
        with torch.no_grad():
            actions, values, info = model(obs_tensors, vehicle_obs, deterministic=True)

        # 应用动作
        action_dict = {}
        for junc_id, action in actions.items():
            action_dict[junc_id] = {}
            controlled = agents[junc_id].get_controlled_vehicles()

            if controlled['main'] and 'main' in action:
                for veh_id in controlled['main'][:1]:
                    action_value = action['main'].item()
                    # 将连续动作转换为速度调整
                    target_speed = 15.0 + action_value * 10.0  # [15, 25] m/s
                    try:
                        traci.vehicle.setSpeed(veh_id, target_speed)
                        action_dict[junc_id][veh_id] = target_speed
                    except Exception as e:
                        print(f"设置主路车辆 {veh_id} 速度失败: {e}")

            if controlled['ramp'] and 'ramp' in action:
                for veh_id in controlled['ramp'][:1]:
                    action_value = action['ramp'].item()
                    # 调整匝道车辆速度
                    target_speed = 10.0 + action_value * 10.0  # [10, 20] m/s
                    try:
                        traci.vehicle.setSpeed(veh_id, target_speed)
                        action_dict[junc_id][veh_id] = target_speed
                    except Exception as e:
                        print(f"设置匝道车辆 {veh_id} 速度失败: {e}")

        # 仿真一步
        traci.simulationStep()

        # 观察新状态
        for junc_id, agent in agents.items():
            agent.observe(traci)

        # 统计
        if step % 100 == 0:
            elapsed = time.time() - start_time
            fps = (step + 1) / elapsed
            print(f"  步数: {step}/{args.steps} | FPS: {fps:.1f}", end='\r')

    # 完成
    elapsed = time.time() - start_time
    print(f"\n\n推理完成!")
    print(f"  总时间: {elapsed:.2f}s")
    print(f"  平均FPS: {args.steps / elapsed:.1f}")

    # 计算最终统计
    arrived = traci.simulation.getArrivedNumber()
    departed = traci.simulation.getDepartedNumber()
    ocr = arrived / max(departed, 1)

    print(f"\n最终统计:")
    print(f"  出发车辆: {departed}")
    print(f"  到达车辆: {arrived}")
    print(f"  OCR: {ocr:.4f}")

    # 关闭SUMO
    traci.close()

    # 保存结果
    if args.output:
        result = {
            'model': args.model,
            'sumo_cfg': args.sumo_cfg,
            'steps': args.steps,
            'arrived': arrived,
            'departed': departed,
            'ocr': ocr,
            'elapsed_time': elapsed
        }

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n结果已保存: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='多智能体路口控制 - 推理')

    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--sumo-cfg', type=str, required=True, help='SUMO配置文件')
    parser.add_argument('--steps', type=int, default=3600, help='仿真步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, help='输出结果文件路径')

    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()
