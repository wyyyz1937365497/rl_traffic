"""
简化版车辆级专家数据收集

使用26分规则的控制，收集每辆车的独立动作
"""
import os
import sys
import pickle
import numpy as np

# 导入必要的模块
sys.path.insert(0, '.')

# 路口配置
JUNCTION_CONFIGS = {
    'J5': {
        'main_edges': ['E2', 'E3'],
        'ramp_edges': ['E23'],
        'reverse_edges': ['-E3', '-E2'],
    },
    'J14': {
        'main_edges': ['E9', 'E10'],
        'ramp_edges': ['E15'],
        'reverse_edges': ['-E10', '-E9'],
    },
    'J15': {
        'main_edges': ['E10', 'E11'],
        'ramp_edges': ['E17'],
        'reverse_edges': ['-E11', '-E10'],
        'diverge_edges': ['E16'],
    },
    'J17': {
        'main_edges': ['E12', 'E13'],
        'ramp_edges': ['E19'],
        'reverse_edges': ['-E13', '-E12'],
        'diverge_edges': ['E18', 'E20'],
    },
}

# 26分规则的NEXT_EDGE映射
NEXT_EDGE = {
    '-E13': '-E12', '-E12': '-E11', '-E11': '-E10', '-E10': '-E9',
    '-E9': '-E8', '-E8': '-E7', '-E7': '-E6', '-E6': '-E5',
    '-E5': '-E3', '-E3': '-E2', '-E2': '-E1',
    'E1': 'E2', 'E2': 'E3', 'E3': 'E5', 'E5': 'E6',
    'E6': 'E7', 'E7': 'E8', 'E8': 'E9', 'E9': 'E10',
    'E10': 'E11', 'E11': 'E12', 'E12': 'E13',
    'E23': '-E2', 'E15': 'E10', 'E17': '-E10', 'E19': '-E12',
}

def apply_26point_control(step):
    """应用26分规则的控制（简化版）"""
    # 配置vType参数（仅执行一次）
    if not hasattr(apply_26point_control, 'vtype_configured'):
        try:
            traci.vehicletype.setImperfection('CV', 0.0)
            traci.vehicletype.setTau('CV', 0.9)
            traci.vehicletype.setAccel('CV', 2.1)
            traci.vehicletype.setDecel('CV', 4.5)

            traci.vehicletype.setImperfection('HV', 0.0)
            traci.vehicletype.setTau('HV', 0.9)
            traci.vehicletype.setAccel('HV', 2.1)
            traci.vehicletype.setDecel('HV', 4.5)

            apply_26point_control.vtype_configured = True
        except:
            pass

    # 主动速度引导（每5步执行一次）
    if step % 5 == 0:
        SPEED_LIMIT = 13.89
        CONGEST_SPEED = 5.0
        LOOKAHEAD = 2
        APPROACH_DIST = 50.0
        SPEED_FACTOR = 1.5
        SPEED_FLOOR = 3.0

        # 采集各边平均速度
        edge_speeds = {}
        for eid in list(NEXT_EDGE.keys()) + list(NEXT_EDGE.values()):
            try:
                edge_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)
            except:
                edge_speeds[eid] = SPEED_LIMIT

        controlled_cvs = set()

        for veh_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh_id) != 'CV':
                    continue

                road_id = traci.vehicle.getRoadID(veh_id)
                if road_id.startswith(':') or road_id not in NEXT_EDGE:
                    continue

                pos = traci.vehicle.getLanePosition(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_len = traci.lane.getLength(lane_id)
                dist_to_end = lane_len - pos

                # 只在靠近边末端时干预
                if dist_to_end > APPROACH_DIST:
                    if veh_id in controlled_cvs:
                        traci.vehicle.setSpeed(veh_id, -1)
                    continue

                # 查看下游拥堵
                congested = False
                min_ds_speed = SPEED_LIMIT
                nxt = road_id
                for _ in range(LOOKAHEAD):
                    nxt = NEXT_EDGE.get(nxt)
                    if nxt is None:
                        break
                    ds_speed = edge_speeds.get(nxt, SPEED_LIMIT)
                    if ds_speed < CONGEST_SPEED:
                        congested = True
                        min_ds_speed = min(min_ds_speed, ds_speed)

                if congested:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    target = max(min_ds_speed * SPEED_FACTOR, SPEED_FLOOR)
                    target = min(target, SPEED_LIMIT)

                    if target < current_speed:
                        traci.vehicle.slowDown(veh_id, target, 3.0)
                        controlled_cvs.add(veh_id)
                elif veh_id in controlled_cvs:
                    traci.vehicle.setSpeed(veh_id, -1)

            except:
                continue

        # 释放之前控制的车辆
        for veh_id in list(controlled_cvs):
            try:
                if veh_id not in traci.vehicle.getIDList():
                    controlled_cvs.remove(veh_id)
            except:
                controlled_cvs.discard(veh_id)


def collect_single_episode(sumo_cfg, episode_id, max_steps=3600):
    """收集单个episode"""
    # 启动SUMO
    import traci

    sumo_cmd = [
        'sumo',
        '-c', sumo_cfg,
        '--no-warnings', 'true',
        '--seed', '42'
    ]

    traci.start(sumo_cmd)

    episode_data = {
        'episode_id': episode_id,
        'transitions': []
    }

    try:
        for step in range(max_steps):
            # 应用26分规则控制
            apply_26point_control(step)

            # 仿真步
            traci.simulationStep()

            # 每10步收集一次数据
            if step % 10 == 0:
                current_vehicles = traci.vehicle.getIDList()

                for veh_id in current_vehicles:
                    try:
                        if traci.vehicle.getTypeID(veh_id) != 'CV':
                            continue

                        # 获取车辆状态
                        speed = traci.vehicle.getSpeed(veh_id)
                        lane_pos = traci.vehicle.getLanePosition(veh_id)
                        lane_idx = traci.vehicle.getLaneIndex(veh_id)
                        accel = traci.vehicle.getAcceleration(veh_id)

                        # 计算动作（从当前速度/最大速度）
                        max_speed = traci.vehicle.getAllowedSpeed(veh_id)
                        action_value = min(speed / max(max_speed, 0.1), 1.0)

                        # 保存车辆级数据
                        transition = {
                            'step': step,
                            'vehicle_id': veh_id,
                            'state': np.zeros(23, dtype=np.float32),  # 简化状态
                            'action': np.array([action_value], dtype=np.float32),
                            'vehicle_features': np.array([
                                speed / 20.0,
                                lane_pos / 500.0,
                                lane_idx / 3.0,
                                0.0,  # waiting_time
                                accel / 5.0,
                                1.0,  # is_cv
                                0.0,  # route_index
                                0.0
                            ], dtype=np.float32)
                        }
                        episode_data['transitions'].append(transition)

                    except Exception as e:
                        continue

            # 检查是否结束
            if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                break

    except Exception as e:
        print(f"Episode {episode_id} error: {e}")
    finally:
        traci.close()

    episode_data['num_transitions'] = len(episode_data['transitions'])
    return episode_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg')
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='expert_demos_vehicle_simple')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"收集 {args.num_episodes} 个episodes...")

    all_episodes = []
    for i in range(args.num_episodes):
        ep = collect_single_episode(args.sumo_cfg, i+1)
        all_episodes.append(ep)
        print(f"Episode {i+1}/{args.num_episodes}: {ep['num_transitions']} transitions")

    # 保存
    output_file = os.path.join(args.output_dir, 'vehicle_demos.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    print(f"\n保存到: {output_file}")
    print(f"总transitions: {sum(ep['num_transitions'] for ep in all_episodes)}")


if __name__ == '__main__':
    main()
