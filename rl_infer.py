"""
多智能体路口交通控制 - 推理脚本
使用训练好的模型进行SUMO仿真推理
"""

import os
import sys
import argparse
import time
import json

# ===== OCR计算辅助函数 =====

def compute_gap_size(main_vehicles: list, ramp_vehicles: list) -> float:
    """计算间隙大小（考虑速度和加速度）"""
    if not main_vehicles or not ramp_vehicles:
        return 0.0

    # 获取主路车辆信息（位置、速度、加速度）
    vehicle_info = []
    for veh in main_vehicles:
        if isinstance(veh, dict):
            pos = veh.get('lane_position', veh.get('position', 0))
            speed = veh.get('speed', 0)
            accel = veh.get('accel', veh.get('acceleration', 0))
        else:
            pos, speed, accel = 0, 0, 0
        vehicle_info.append({'pos': pos, 'speed': speed, 'accel': accel})

    if len(vehicle_info) < 2:
        return 10.0

    # 按位置排序（从前到后）
    vehicle_info.sort(key=lambda x: x['pos'], reverse=True)

    # 计算有效间隙（考虑速度和加速度）
    gaps = []
    for i in range(len(vehicle_info) - 1):
        leader = vehicle_info[i]
        follower = vehicle_info[i + 1]

        # 基础位置间隙
        pos_gap = leader['pos'] - follower['pos']

        # 速度差（前车速度 - 后车速度）
        speed_diff = leader['speed'] - follower['speed']

        # 计算时间间隙（秒）
        if follower['speed'] > 0.1:
            time_gap = pos_gap / follower['speed']
        else:
            time_gap = 999.0  # 后车停止

        # 有效间隙计算
        # 1. 位置间隙基础分
        effective_gap = pos_gap

        # 2. 速度调整：如果前车更快，间隙会增大；反之减小
        # 使用2秒后的预期位置变化
        speed_adjustment = speed_diff * 2.0
        effective_gap += speed_adjustment

        # 3. 加速度调整（如果加速度差异大，额外调整）
        accel_diff = leader.get('accel', 0) - follower.get('accel', 0)
        accel_adjustment = accel_diff * 1.0  # 1秒的加速度影响
        effective_gap += accel_adjustment

        # 4. 时间间隙阈值检查（至少2秒安全间隙）
        min_safe_gap = follower['speed'] * 2.0  # 2秒跟车距离
        if effective_gap < min_safe_gap and time_gap < 2.0:
            # 间隙不足，降低有效间隙值
            effective_gap *= 0.5

        # 只考虑大于5米的间隙
        if effective_gap > 5.0:
            gaps.append(effective_gap)

    return max(gaps) if gaps else 0.0

def compute_gap_speed_diff(main_vehicles: list, ramp_vehicles: list) -> float:
    """计算速度差"""
    if not main_vehicles or not ramp_vehicles:
        return 0.0
    
    # 计算平均速度
    main_speeds = []
    for veh in main_vehicles:
        if isinstance(veh, dict):
            speed = veh.get('speed', 0)
        else:
            speed = 0
        main_speeds.append(speed)
    
    ramp_speeds = []
    for veh in ramp_vehicles:
        if isinstance(veh, dict):
            speed = veh.get('speed', 0)
        else:
            speed = 0
        ramp_speeds.append(speed)
    
    import numpy as np
    avg_main = np.mean(main_speeds) if main_speeds else 0.0
    avg_ramp = np.mean(ramp_speeds) if ramp_speeds else 0.0
    
    return abs(avg_main - avg_ramp)



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
                # 修复：精确计算间隙
                gap_size = compute_gap_size(main_vehicles_info, ramp_vehicles_info)
                gap_speed_diff = compute_gap_speed_diff(main_vehicles_info, ramp_vehicles_info)

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
                    # 将连续动作转换为速度调整（使用SUMO真实限速）
                    speed_limit = 13.89  # 50 km/h - SUMO配置
                    target_speed = speed_limit * (0.3 + 0.9 * action_value)  # [4.17, 15.67] m/s
                    target_speed = max(0.0, min(target_speed, speed_limit * 1.2))  # 限制在合理范围
                    try:
                        traci.vehicle.setSpeed(veh_id, target_speed)
                        action_dict[junc_id][veh_id] = target_speed
                    except Exception as e:
                        print(f"设置主路车辆 {veh_id} 速度失败: {e}")

            if controlled['ramp'] and 'ramp' in action:
                for veh_id in controlled['ramp'][:1]:
                    action_value = action['ramp'].item()
                    # 调整匝道车辆速度（使用SUMO真实限速）
                    speed_limit = 13.89  # 50 km/h - SUMO配置
                    target_speed = speed_limit * (0.3 + 0.9 * action_value)  # [4.17, 15.67] m/s
                    target_speed = max(0.0, min(target_speed, speed_limit * 1.2))  # 限制在合理范围
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
