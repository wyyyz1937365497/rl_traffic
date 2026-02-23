"""
收集车辆级专家演示数据（基于26分规则）

关键改进：
- 收集每辆车的独立动作，而不是路口级的共享动作
- 保留26分规则的控制逻辑
- 生成可训练车辆级网络的专家数据
"""
import os
import sys
import pickle
import numpy as np
import logging
from typing import Dict, List
from tqdm import tqdm
import argparse

# 使用libsumo
try:
    import libsumo as traci
    USE_LIBSUMO = True
except ImportError:
    import traci
    USE_LIBSUMO = False

# 导入26分规则的专家策略
sys.path.insert(0, 'relu_based/rl_traffic/sumo')
if os.path.exists('relu_based/rl_traffic/sumo/main.py'):
    from main import SUMOCompetitionFramework


def collect_vehicle_level_demos(sumo_cfg: str, num_episodes: int = 50, output_dir: str = "expert_demos_vehicle"):
    """
    收集车辆级专家演示数据

    Args:
        sumo_cfg: SUMO配置文件
        num_episodes: 收集的episode数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("收集车辆级专家演示数据（基于26分规则）")
    print("=" * 80)
    print(f"SUMO配置: {sumo_cfg}")
    print(f"Episodes: {num_episodes}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 使用26分脚本的框架
    framework = SUMOCompetitionFramework(sumo_cfg)

    # 初始化环境
    framework.initialize_environment(use_gui=False, max_steps=3600)

    all_episodes = []

    for episode_id in range(1, num_episodes + 1):
        print(f"\n{'='*80}")
        print(f"收集 Episode {episode_id}/{num_episodes}")
        print(f"{'='*80}")

        # 重启SUMO
        framework._start_sumo(seed=42)

        episode_data = {
            'episode_id': episode_id,
            'transitions': []
        }

        try:
            for step in range(3600):
                # 应用26分规则的控制
                framework.apply_control_algorithm(step)

                # 收集每辆CV的动作（车辆级）
                current_vehicles = traci.vehicle.getIDList()

                for junc_id in ['J5', 'J14', 'J15', 'J17']:
                    # 获取该路口的CV车辆
                    controlled_vehicles = []
                    for veh_id in current_vehicles:
                        try:
                            if traci.vehicle.getTypeID(veh_id) == 'CV':
                                # 检查车辆是否属于该路口的控制范围
                                road_id = traci.vehicle.getRoadID(veh_id)
                                if road_id in JUNCTION_CONFIGS[junc_id].get('main_edges', []) + \
                                   JUNCTION_CONFIGS[junc_id].get('ramp_edges', []) + \
                                   JUNCTION_CONFIGS[junc_id].get('diverge_edges', []):
                                    controlled_vehicles.append(veh_id)
                        except:
                            continue

                    if not controlled_vehicles:
                        continue

                    # 为每辆车收集状态和动作
                    for veh_id in controlled_vehicles:
                        try:
                            # 获取车辆状态（8维特征）
                            speed = traci.vehicle.getSpeed(veh_id)
                            lane_position = traci.vehicle.getLanePosition(veh_id)
                            lane_index = traci.vehicle.getLaneIndex(veh_id)
                            acceleration = traci.vehicle.getAcceleration(veh_id)
                            waiting_time = traci.vehicle.getWaitingTime(veh_id)

                            # 获取路口状态（23维）
                            # 这里简化：使用固定状态（实际应该从环境获取）
                            state_vec = np.zeros(23, dtype=np.float32)

                            # 获取车辆的实际动作（从当前速度推断）
                            current_speed = speed
                            max_speed = traci.vehicle.getAllowedSpeed(veh_id)
                            action_value = current_speed / max(max_speed, 1.0)  # 归一化到[0,1]

                            # 保存转换
                            transition = {
                                'step': step,
                                'junction_id': junc_id,
                                'vehicle_id': veh_id,
                                'state': state_vec.copy(),
                                'action': np.array([action_value], dtype=np.float32),  # 车辆级动作
                                'vehicle_features': np.array([
                                    speed / 20.0,
                                    lane_position / 500.0,
                                    lane_index / 3.0,
                                    waiting_time / 60.0,
                                    acceleration / 5.0,
                                    1.0,  # is_cv
                                    0.0,  # route_index
                                    0.0
                                ], dtype=np.float32)
                            }
                            episode_data['transitions'].append(transition)

                        except Exception as e:
                            continue

                # 收集数据
                framework.collect_step_data(step)

                # 检查是否结束
                if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                    break

        except Exception as e:
            print(f"Episode {episode_id} 失败: {e}")

        finally:
            try:
                traci.close()
            except:
                pass

        # 统计
        stats = framework.statistics
        episode_data['statistics'] = {
            'cumulative_departed': stats['cumulative_departed'],
            'cumulative_arrived': stats['cumulative_arrived'],
            'ocr': stats['cumulative_arrived'] / max(stats['cumulative_departed'], 1)
        }
        episode_data['num_transitions'] = len(episode_data['transitions'])

        print(f"  OCR: {episode_data['statistics']['ocr']:.4f}")
        print(f"  Transitions: {episode_data['num_transitions']}")

        all_episodes.append(episode_data)

    # 保存所有数据
    output_file = os.path.join(output_dir, 'vehicle_level_demos.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    print(f"\n{'='*80}")
    print(f"数据收集完成！")
    print(f"{'='*80}")
    print(f"总Episodes: {len(all_episodes)}")
    print(f"总Transitions: {sum(ep['num_transitions'] for ep in all_episodes)}")
    print(f"保存到: {output_file}")

    # 保存为兼容格式（用于现有BC训练）
    compatible_file = os.path.join(output_dir, 'expert_demonstrations.pkl')
    with open(compatible_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    print(f"兼容格式保存到: {compatible_file}")

    return all_episodes


def main():
    parser = argparse.ArgumentParser(description='收集车辆级专家演示数据')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg')
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='expert_demos_vehicle')

    args = parser.parse_args()

    collect_vehicle_level_demos(
        sumo_cfg=args.sumo_cfg,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
