"""
收集车辆级专家演示数据（基于26分规则）

关键改进：
- 为每辆CV车收集独立的action（基于实际速度）
- 保留26分规则的控制逻辑
- 生成可训练车辆级网络的数据
"""
import os
import pickle
import numpy as np
import logging
from typing import Dict, List
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import traceback

# 尝试使用libsumo
try:
    import libsumo as traci
    USE_LIBSUMO = True
except ImportError:
    import traci
    USE_LIBSUMO = False


# ============================================================================
# 路口配置
# ============================================================================

JUNCTION_CONFIGS = {
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
        'main_edges': ['E10', 'E11'],
        'ramp_edges': ['E17'],
        'reverse_edges': ['-E11', '-E10'],
        'diverge_edges': ['E16'],
        'tl_id': 'J15',
        'num_phases': 2,
    },
    'J17': {
        'type': 'TYPE_B',
        'main_edges': ['E12', 'E13'],
        'ramp_edges': ['E19'],
        'reverse_edges': ['-E13', '-E12'],
        'diverge_edges': ['E18', 'E20'],
        'tl_id': 'J17',
        'num_phases': 2,
    },
}


# ============================================================================
# 专家策略（26分规则）
# ============================================================================

class ExpertPolicy:
    """专家策略：基于规则的控制（26分策略）"""

    def __init__(self):
        self.NEXT_EDGE = {
            '-E13': '-E12', '-E12': '-E11', '-E11': '-E10', '-E10': '-E9',
            '-E9': '-E8', '-E8': '-E7', '-E7': '-E6', '-E6': '-E5',
            '-E5': '-E3', '-E3': '-E2', '-E2': '-E1',
            'E1': 'E2', 'E2': 'E3', 'E3': 'E5', 'E5': 'E6',
            'E6': 'E7', 'E7': 'E8', 'E8': 'E9', 'E9': 'E10',
            'E10': 'E11', 'E11': 'E12', 'E12': 'E13',
            'E23': '-E2', 'E15': 'E10', 'E17': '-E10', 'E19': '-E12',
        }

        self.params = {
            'approach_dist': 50.0,
            'congest_speed': 5.0,
            'lookahead': 2,
            'speed_factor': 1.5,
            'speed_floor': 3.0,
            'speed_limit': 13.89,
        }

        self.vtype_configured = False
        self.controlled_cvs = set()
        self.edge_speeds = {}
        self.edge_speed_cache_step = -1

        # 记录专家策略的目标速度（用于数据收集）
        self.target_speeds = {}  # {veh_id: target_speed_ratio}

    def configure_vtype(self):
        """配置车辆类型参数（与26分脚本保持一致）"""
        if self.vtype_configured:
            return

        try:
            # CV车辆参数
            traci.vehicletype.setImperfection('CV', 0.0)
            traci.vehicletype.setTau('CV', 0.9)
            traci.vehicletype.setAccel('CV', 2.1)
            traci.vehicletype.setDecel('CV', 4.5)

            # HV车辆参数
            traci.vehicletype.setImperfection('HV', 0.0)
            traci.vehicletype.setTau('HV', 0.9)
            traci.vehicletype.setAccel('HV', 2.1)
            traci.vehicletype.setDecel('HV', 4.5)

            self.vtype_configured = True
        except:
            pass

    def apply_control(self, step: int):
        """应用主动速度引导控制（每5步执行一次）"""
        self.configure_vtype()

        if step % 5 != 0:
            return

        # 清空目标速度记录（每步重新计算）
        self.target_speeds.clear()

        # 更新边速度缓存
        if self.edge_speed_cache_step != step:
            all_edges = set(self.NEXT_EDGE.keys()) | set(self.NEXT_EDGE.values())
            self.edge_speeds = {}
            for eid in all_edges:
                try:
                    self.edge_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)
                except:
                    self.edge_speeds[eid] = self.params['speed_limit']
            self.edge_speed_cache_step = step

        new_controlled = set()

        for veh_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh_id) != 'CV':
                    continue

                road_id = traci.vehicle.getRoadID(veh_id)
                if road_id.startswith(':') or road_id not in self.NEXT_EDGE:
                    continue

                pos = traci.vehicle.getLanePosition(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_len = traci.lane.getLength(lane_id)
                dist_to_end = lane_len - pos

                # 默认：不控制，全速行驶（action = 1.0）
                self.target_speeds[veh_id] = 1.0

                # 只在靠近边末端时干预
                if dist_to_end > self.params['approach_dist']:
                    if veh_id in self.controlled_cvs:
                        traci.vehicle.setSpeed(veh_id, -1)
                    continue

                # 查看下游拥堵
                congested = False
                min_ds_speed = self.params['speed_limit']
                nxt = road_id
                for _ in range(self.params['lookahead']):
                    nxt = self.NEXT_EDGE.get(nxt)
                    if nxt is None:
                        break
                    ds_speed = self.edge_speeds.get(nxt, self.params['speed_limit'])
                    if ds_speed < self.params['congest_speed']:
                        congested = True
                        min_ds_speed = min(min_ds_speed, ds_speed)

                if congested:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    target = max(min_ds_speed * self.params['speed_factor'],
                                self.params['speed_floor'])
                    target = min(target, self.params['speed_limit'])

                    if target < current_speed:
                        traci.vehicle.slowDown(veh_id, target, 3.0)
                        new_controlled.add(veh_id)
                        # 记录专家策略的目标速度比例
                        self.target_speeds[veh_id] = target / self.params['speed_limit']
                elif veh_id in self.controlled_cvs:
                    traci.vehicle.setSpeed(veh_id, -1)

            except:
                continue

        self.controlled_cvs = new_controlled


# ============================================================================
# 工具函数：构建状态向量
# ============================================================================

def build_junction_state(junc_id: str, junc_config: dict, subscribed_vehicles: dict, edge_speeds: dict, SPEED_LIMIT: float = 13.89) -> np.ndarray:
    """
    构建23维路口状态向量

    Args:
        junc_id: 路口ID
        junc_config: 路口配置
        subscribed_vehicles: 订阅的车辆数据
        edge_speeds: 边速度缓存
        SPEED_LIMIT: 速度限制

    Returns:
        23维状态向量
    """
    state = np.zeros(23, dtype=np.float32)

    try:
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


# ============================================================================
# 车辆级数据收集
# ============================================================================

def collect_single_episode(args: tuple) -> Dict:
    """
    收集单个episode的车辆级专家数据

    Args:
        args: (episode_id, sumo_cfg, max_steps)

    Returns:
        episode_data字典
    """
    episode_id, sumo_cfg, max_steps = args

    # 启动SUMO
    seed = 42 + episode_id
    sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "true", "--seed", str(seed)]

    try:
        traci.start(sumo_cmd)
    except Exception as e:
        return {
            'episode_id': episode_id,
            'success': False,
            'error': str(e),
            'transitions': []
        }

    # 创建专家策略
    expert = ExpertPolicy()

    episode_data = {
        'episode_id': episode_id,
        'seed': seed,
        'transitions': [],
        'total_reward': 0.0,
    }

    subscribed_vehicles = {}
    episode_reward = 0.0

    try:
        # 订阅所有车辆
        for veh_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.subscribe(veh_id, [
                    traci.VAR_ROAD_ID,
                    traci.VAR_SPEED,
                    traci.VAR_WAITING_TIME,
                ])
                subscribed_vehicles[veh_id] = {'subscribed': True}
            except:
                pass

        for step in range(max_steps):
            # 应用专家控制（26分规则）
            expert.apply_control(step)

            # 仿真一步
            traci.simulationStep()

            # 更新订阅数据
            current_vehicles = traci.vehicle.getIDList()

            # 订阅新车辆
            for veh_id in current_vehicles:
                if veh_id not in subscribed_vehicles:
                    try:
                        traci.vehicle.subscribe(veh_id, [
                            traci.VAR_ROAD_ID,
                            traci.VAR_SPEED,
                            traci.VAR_WAITING_TIME,
                        ])
                        subscribed_vehicles[veh_id] = {'subscribed': True}
                    except:
                        pass

            # 更新订阅数据
            for veh_id in list(subscribed_vehicles.keys()):
                try:
                    result = traci.vehicle.getSubscriptionResults(veh_id)
                    if result:
                        subscribed_vehicles[veh_id] = {
                            'road_id': result.get(traci.VAR_ROAD_ID, ""),
                            'speed': result.get(traci.VAR_SPEED, 0.0),
                            'waiting_time': result.get(traci.VAR_WAITING_TIME, 0.0),
                            'is_cv': traci.vehicle.getTypeID(veh_id) == 'CV',
                        }
                except:
                    pass

            # 收集车辆级数据（每10步）
            if step % 10 == 0:
                for junc_id, junc_config in JUNCTION_CONFIGS.items():
                    # 获取该路口的所有边
                    main_edges = junc_config['main_edges']
                    ramp_edges = junc_config['ramp_edges']
                    reverse_edges = junc_config['reverse_edges']
                    all_edges = main_edges + ramp_edges + reverse_edges

                    if junc_config['type'] == 'TYPE_B':
                        all_edges.extend(junc_config['diverge_edges'])

                    # 找到该路口的所有CV车辆
                    junction_cvs = []
                    for veh_id, data in subscribed_vehicles.items():
                        if not data.get('is_cv', False):
                            continue
                        road_id = data.get('road_id', '')
                        if road_id in all_edges:
                            junction_cvs.append(veh_id)

                    if not junction_cvs:
                        continue

                    # 构建全局状态向量（使用真实数据而不是零向量）
                    state_vec = build_junction_state(
                        junc_id, junc_config, subscribed_vehicles,
                        expert.edge_speeds, expert.params['speed_limit']
                    )

                    # 为每辆CV车收集数据
                    for veh_id in junction_cvs:
                        try:
                            # 获取车辆状态
                            speed = subscribed_vehicles[veh_id]['speed']
                            road_id = subscribed_vehicles[veh_id]['road_id']
                            waiting_time = subscribed_vehicles[veh_id]['waiting_time']

                            # 获取更多信息
                            lane_id = traci.vehicle.getLaneID(veh_id)
                            lane_pos = traci.vehicle.getLanePosition(veh_id)
                            lane_idx = traci.vehicle.getLaneIndex(veh_id)
                            accel = traci.vehicle.getAcceleration(veh_id)
                            max_speed = traci.vehicle.getAllowedSpeed(veh_id)

                            # 关键特征：距离边末端的距离（专家策略的控制触发条件）
                            lane_len = traci.lane.getLength(lane_id)
                            dist_to_end = lane_len - lane_pos
                            dist_to_end_normalized = dist_to_end / 100.0  # 归一化（专家在<50m时控制）

                            # 确定车辆类别（main/ramp/diverge）
                            if road_id in main_edges or road_id in reverse_edges:
                                vehicle_type = 'main'
                            elif road_id in ramp_edges:
                                vehicle_type = 'ramp'
                            elif junc_config['type'] == 'TYPE_B' and road_id in junc_config['diverge_edges']:
                                vehicle_type = 'diverge'
                            else:
                                continue

                            # 使用专家策略记录的目标速度（而不是实际速度）
                            # 这反映了专家策略的意图：不控制=全速(1.0)，控制=减速(<1.0)
                            action_value = expert.target_speeds.get(veh_id, 1.0)

                            # 保存车辆级transition
                            transition = {
                                'step': step,
                                'junction_id': junc_id,
                                'vehicle_id': veh_id,  # 关键：包含车辆ID
                                'state': state_vec.copy(),
                                'action_main': np.array([action_value], dtype=np.float32),
                                'action_ramp': np.array([action_value], dtype=np.float32),
                                'vehicle_type': vehicle_type,
                                'vehicle_features': np.array([
                                    speed / 20.0,
                                    lane_pos / 500.0,
                                    lane_idx / 3.0,
                                    waiting_time / 60.0,
                                    accel / 5.0,
                                    1.0,  # is_cv
                                    dist_to_end_normalized,  # 距离边末端的距离（关键特征！）
                                    0.0   # padding
                                ], dtype=np.float32)
                            }
                            episode_data['transitions'].append(transition)

                            # 奖励（速度）
                            reward = speed / 20.0
                            episode_reward += reward

                        except Exception as e:
                            continue

            # 清理已离开的车辆
            left_vehicles = set(subscribed_vehicles.keys()) - set(current_vehicles)
            for veh_id in left_vehicles:
                del subscribed_vehicles[veh_id]

            # 检查是否结束
            if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                break

    except Exception as e:
        episode_data['error'] = str(e)
        episode_data['traceback'] = traceback.format_exc()

    finally:
        try:
            traci.close()
        except:
            pass

    episode_data['total_reward'] = episode_reward
    episode_data['steps'] = step
    episode_data['success'] = True
    episode_data['num_transitions'] = len(episode_data['transitions'])

    return episode_data


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='收集车辆级专家演示数据')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg',
                        help='SUMO配置文件')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='收集的episode数量')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='并行进程数（默认使用所有CPU核心）')
    parser.add_argument('--output-dir', type=str, default='expert_demos_vehicle',
                        help='输出目录')
    parser.add_argument('--max-steps', type=int, default=3600,
                        help='每个episode最大步数')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定并行进程数
    if args.num_workers is None:
        num_workers = min(cpu_count(), 32)
    else:
        num_workers = args.num_workers

    print("=" * 80)
    print("收集车辆级专家演示数据（基于26分规则）")
    print("=" * 80)
    print(f"SUMO配置: {args.sumo_cfg}")
    print(f"Episodes: {args.num_episodes}")
    print(f"并行进程: {num_workers}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # 准备任务参数
    task_args = [(i+1, args.sumo_cfg, args.max_steps) for i in range(args.num_episodes)]

    # 多进程收集
    all_episodes = []

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(collect_single_episode, task_args),
            total=args.num_episodes,
            desc="收集Episodes"
        ))

    # 处理结果
    success_count = 0
    total_transitions = 0

    for ep_data in results:
        if ep_data.get('success', False):
            success_count += 1
            total_transitions += ep_data['num_transitions']
            all_episodes.append(ep_data)
        else:
            print(f"Episode {ep_data['episode_id']} 失败: {ep_data.get('error', 'Unknown')}")

    print(f"\n{'='*80}")
    print(f"收集完成！")
    print(f"{'='*80}")
    print(f"成功Episodes: {success_count}/{args.num_episodes}")
    print(f"总Transitions: {total_transitions}")
    print(f"平均每个episode: {total_transitions/max(success_count, 1):.0f} transitions")

    # 保存数据
    output_file = os.path.join(args.output_dir, 'vehicle_expert_demos.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    print(f"\n数据已保存到: {output_file}")

    # 兼容格式（用于现有BC训练脚本）
    compatible_file = os.path.join(args.output_dir, 'expert_demonstrations.pkl')
    with open(compatible_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    print(f"兼容格式保存到: {compatible_file}")

    # 验证数据格式
    if all_episodes and all_episodes[0]['transitions']:
        sample = all_episodes[0]['transitions'][0]
        print(f"\n数据格式验证:")
        print(f"  Transition keys: {list(sample.keys())}")
        print(f"  Has vehicle_id: {'vehicle_id' in sample}")
        print(f"  Has action_main: {'action_main' in sample}")
        print(f"  Has action_ramp: {'action_ramp' in sample}")
        print(f"  Has vehicle_type: {'vehicle_type' in sample}")


if __name__ == '__main__':
    main()
