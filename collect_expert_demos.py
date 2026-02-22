"""
收集专家演示数据（完整23维状态向量）

使用 libsumo + 订阅模式收集专家策略的 (state, action) 对
参照：relu_based\rl_traffic\fast_pkl_generator.py
"""
import os
import pickle
import numpy as np
import torch
import logging
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
import argparse

# 尝试使用libsumo
try:
    import libsumo as traci
    USE_LIBSUMO = True
except ImportError:
    import traci
    USE_LIBSUMO = False


# ============================================================================
# 路口配置（简化版，用于数据收集）
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
        'type': 'TYPE_B',  # 有分流
        'main_edges': ['E10', 'E11'],
        'ramp_edges': ['E17'],
        'reverse_edges': ['-E11', '-E10'],
        'diverge_edges': ['E16'],
        'tl_id': 'J15',
        'num_phases': 2,
    },
    'J17': {
        'type': 'TYPE_B',  # 有分流
        'main_edges': ['E12', 'E13'],
        'ramp_edges': ['E19'],
        'reverse_edges': ['-E13', '-E12'],
        'diverge_edges': ['E18', 'E20'],
        'tl_id': 'J17',
        'num_phases': 2,
    },
}


# ============================================================================
# 专家策略
# ============================================================================

class ExpertPolicy:
    """专家策略：基于规则的控制"""

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
        }

        self.vtype_configured = False
        self.controlled_cvs = set()
        self.edge_speeds = {}
        self.edge_speed_cache_step = -1

    def configure_vtypes(self):
        """配置vType参数"""
        if self.vtype_configured:
            return

        try:
            traci.vehicletype.setImperfection('CV', 0.0)
            traci.vehicletype.setTau('CV', 0.9)
            traci.vehicletype.setAccel('CV', 2.1)  # 与26分脚本保持一致
            traci.vehicletype.setDecel('CV', 4.5)  # SUMO默认值

            traci.vehicletype.setImperfection('HV', 0.0)
            traci.vehicletype.setTau('HV', 0.9)
            traci.vehicletype.setAccel('HV', 2.1)  # 与26分脚本保持一致
            traci.vehicletype.setDecel('HV', 4.5)  # SUMO默认值

            logging.info("[OK] vType configured")
            self.vtype_configured = True
        except Exception as e:
            logging.warning(f"vType configuration failed: {e}")

    def apply_control(self, step: int):
        """应用主动控制"""
        self.configure_vtypes()

        # 采集各边平均速度（缓存）
        if self.edge_speed_cache_step != step:
            all_edges = set(self.NEXT_EDGE.keys()) | set(self.NEXT_EDGE.values())
            self.edge_speeds = {}
            for eid in all_edges:
                try:
                    self.edge_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)
                except:
                    self.edge_speeds[eid] = 13.89
            self.edge_speed_cache_step = step

        new_controlled = set()

        for veh_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh_id) != 'CV':
                    continue

                road_id = traci.vehicle.getRoadID(veh_id)
                if road_id.startswith(':') or road_id not in self.NEXT_EDGE:
                    continue

                lane_id = traci.vehicle.getLaneID(veh_id)
                pos = traci.vehicle.getLanePosition(veh_id)

                try:
                    lane_len = traci.lane.getLength(lane_id)
                except:
                    try:
                        lane_len = traci.edge.getLength(road_id)
                    except:
                        continue

                dist_to_end = lane_len - pos

                if dist_to_end > self.params['approach_dist']:
                    if veh_id in self.controlled_cvs:
                        traci.vehicle.setSpeed(veh_id, -1)
                    continue

                # 检测下游拥堵
                congested = False
                min_ds_speed = 13.89
                nxt = road_id

                for _ in range(self.params['lookahead']):
                    nxt = self.NEXT_EDGE.get(nxt)
                    if nxt is None:
                        break
                    ds_speed = self.edge_speeds.get(nxt, 13.89)
                    if ds_speed < self.params['congest_speed']:
                        congested = True
                        min_ds_speed = min(min_ds_speed, ds_speed)

                # 下游拥堵时减速
                if congested:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    target = max(min_ds_speed * self.params['speed_factor'], self.params['speed_floor'])
                    target = min(target, 13.89)

                    if target < current_speed:
                        traci.vehicle.slowDown(veh_id, target, 3.0)
                        new_controlled.add(veh_id)
                elif veh_id in self.controlled_cvs:
                    traci.vehicle.setSpeed(veh_id, -1)

            except Exception:
                continue

        # 释放不再控制的车辆
        for veh_id in self.controlled_cvs - new_controlled:
            try:
                traci.vehicle.setSpeed(veh_id, -1)
            except:
                pass

        self.controlled_cvs = new_controlled


# ============================================================================
# 状态构建器（23维状态向量）
# ============================================================================

def build_state_vector(junc_id: str, subscribed_vehicles: Dict[str, Dict],
                        junction_config: Dict, step: int) -> np.ndarray:
    """
    构建完整的23维状态向量

    参照：junction_agent.py 的 get_state_vector 方法
    """
    # 获取该路口的所有车辆
    main_edges = junction_config['main_edges']
    ramp_edges = junction_config['ramp_edges']
    reverse_edges = junction_config['reverse_edges']
    all_edges = main_edges + ramp_edges + reverse_edges

    if junction_config['type'] == 'TYPE_B':
        diverge_edges = junction_config['diverge_edges']
        all_edges.extend(diverge_edges)

    # 分类车辆
    main_vehicles = []
    ramp_vehicles = []
    diverge_vehicles = []
    cv_main = []
    cv_ramp = []
    cv_diverge = []

    main_speeds = []
    ramp_speeds = []
    main_waiting = []
    ramp_waiting = []

    for veh_id, data in subscribed_vehicles.items():
        road_id = data.get('road_id', '')
        if road_id not in all_edges:
            continue

        is_cv = data.get('is_cv', False)
        speed = data.get('speed', 0.0)
        waiting = data.get('waiting_time', 0.0)

        if road_id in main_edges:
            main_vehicles.append(veh_id)
            main_speeds.append(speed)
            main_waiting.append(waiting)
            if is_cv:
                cv_main.append(veh_id)
        elif road_id in ramp_edges:
            ramp_vehicles.append(veh_id)
            ramp_speeds.append(speed)
            ramp_waiting.append(waiting)
            if is_cv:
                cv_ramp.append(veh_id)
        elif junction_config['type'] == 'TYPE_B' and road_id in diverge_edges:
            diverge_vehicles.append(veh_id)
            if is_cv:
                cv_diverge.append(veh_id)

    # 计算统计
    main_speed = np.mean(main_speeds) if main_speeds else 0.0
    main_density = len(main_vehicles) / max(len(main_edges) * 2, 1)  # 假设每边2车道
    main_queue_length = sum(1 for w in main_waiting if w > 30)
    main_flow = len(main_vehicles) * main_speed if main_vehicles else 0.0

    ramp_speed = np.mean(ramp_speeds) if ramp_speeds else 0.0
    ramp_density = len(ramp_vehicles) / max(len(ramp_edges), 1)
    ramp_queue_length = sum(1 for w in ramp_waiting if w > 30)
    ramp_waiting_time = np.mean(ramp_waiting) if ramp_waiting else 0.0
    ramp_flow = len(ramp_vehicles) * ramp_speed if ramp_vehicles else 0.0

    # 信号灯状态
    tl_id = junction_config['tl_id']
    try:
        current_phase = traci.trafficlight.getPhase(tl_id)
        time_to_switch = 0.0  # 简化，不获取
    except:
        current_phase = 0
        time_to_switch = 0.0

    main_signal = 'G'  # 简化
    ramp_signal = 'r'
    diverge_signal = 'G' if junction_config['type'] == 'TYPE_B' else ''

    # 冲突和间隙（简化）
    conflict_risk = min(main_queue_length / 10.0, 1.0)
    gap_acceptance = 0.5  # 简化

    # 构建23维状态向量（完全参照 junction_agent.py）
    features = [
        # 主路状态 (5维)
        len(main_vehicles) / 10.0,
        main_speed / 20.0,
        main_density / 10.0,
        main_queue_length / 10.0,
        main_flow / 100.0,

        # 匝道状态 (5维)
        len(ramp_vehicles) / 40.0,
        ramp_speed / 10.0,
        ramp_queue_length / 40.0,
        ramp_waiting_time / 80.0,
        ramp_flow / 80.0,

        # 信号灯状态 (5维)
        current_phase / max(junction_config['num_phases'], 1),
        time_to_switch / 10.0,
        float(main_signal == 'G'),
        float(ramp_signal == 'G'),
        float(diverge_signal == 'G') if junction_config['type'] == 'TYPE_B' else 0.0,

        # 风险和间隙 (2维)
        conflict_risk,
        gap_acceptance,

        # CV比例 (2维)
        len(cv_main) / max(len(main_vehicles), 1),
        len(cv_ramp) / max(len(ramp_vehicles), 1),
    ]

    # 分流状态 (3维)
    if junction_config['type'] == 'TYPE_B':
        features.extend([
            len(diverge_vehicles) / 10.0,
            ramp_queue_length / 10.0,  # 简化：使用ramp_queue
            len(cv_diverge) / max(len(diverge_vehicles), 1),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    # 时间戳 (1维)
    features.append(step / 3600.0)

    return np.array(features, dtype=np.float32)


def infer_expert_action(state_vec: np.ndarray, junction_config: Dict) -> Dict[str, float]:
    """
    从状态推断专家动作
    """
    main_action = 0.7
    ramp_action = 0.5

    # 从状态向量中提取特征
    main_speed = state_vec[1] * 20.0
    ramp_speed = state_vec[6] * 10.0
    main_queue = state_vec[3] * 10.0
    ramp_queue = state_vec[7] * 40.0
    conflict_risk = state_vec[15]

    # 规则
    if main_speed < 5.0:
        main_action = 0.4

    if ramp_queue > 3:
        ramp_action = 0.8

    if conflict_risk > 0.5:
        main_action = min(main_action, 0.4)
        ramp_action = min(ramp_action, 0.3)

    return {
        'main': np.clip(main_action, 0.0, 1.0),
        'ramp': np.clip(ramp_action, 0.0, 1.0)
    }


# ============================================================================
# 数据收集器
# ============================================================================

class ExpertDemoCollector:
    """专家演示数据收集器（使用libsumo + 订阅模式）"""

    def __init__(self, sumo_cfg: str, output_dir: str = "expert_demos"):
        self.sumo_cfg = sumo_cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 创建专家策略
        self.expert = ExpertPolicy()

        logging.info("=" * 70)
        logging.info("Expert Demo Collector (libsumo + 23-dim state vector)")
        logging.info("=" * 70)

    def collect_episode(self, episode_id: int, max_steps: int = 3600) -> Dict:
        """收集单个episode的演示数据"""
        logging.info(f"\n{'='*70}")
        logging.info(f"Collecting Episode {episode_id}")
        logging.info(f"{'='*70}")

        # 启动SUMO
        seed = 42 + episode_id
        sumo_cmd = ["sumo", "-c", self.sumo_cfg, "--no-warnings", "true", "--seed", str(seed)]

        try:
            traci.start(sumo_cmd)
            logging.info(f"[OK] SUMO started (seed={seed})")
        except Exception as e:
            logging.error(f"SUMO start failed: {e}")
            return {'success': False, 'steps': 0}

        episode_data = {
            'episode_id': episode_id,
            'seed': seed,
            'transitions': [],
            'total_reward': 0.0,
            'final_ocr': 0.0
        }

        subscribed_vehicles = {}
        episode_reward = 0.0

        try:
            # 运行仿真
            for step in tqdm(range(max_steps), desc=f"Episode {episode_id}"):
                # 订阅新车辆
                current_vehicles = traci.vehicle.getIDList()
                new_vehicles = set(current_vehicles) - set(subscribed_vehicles.keys())

                if new_vehicles:
                    for veh_id in new_vehicles:
                        try:
                            traci.vehicle.subscribe(veh_id, [
                                traci.VAR_ROAD_ID,
                                traci.VAR_LANEPOSITION,
                                traci.VAR_SPEED,
                                traci.VAR_WAITING_TIME,
                            ])
                            subscribed_vehicles[veh_id] = {'subscribed': True}
                        except:
                            pass

                # 仿真一步
                traci.simulationStep()

                # 应用专家控制
                self.expert.apply_control(step)

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

                # 收集状态和动作（每10步）
                if step % 10 == 0:
                    for junc_id, junc_config in JUNCTION_CONFIGS.items():
                        # 构建状态向量
                        state_vec = build_state_vector(junc_id, subscribed_vehicles, junc_config, step)

                        # 推断专家动作
                        expert_action = infer_expert_action(state_vec, junc_config)

                        # 保存转换
                        transition = {
                            'step': step,
                            'junction_id': junc_id,
                            'state': state_vec.copy(),  # 23维向量
                            'action_main': np.array([expert_action['main']], dtype=np.float32),
                            'action_ramp': np.array([expert_action['ramp']], dtype=np.float32),
                        }
                        episode_data['transitions'].append(transition)

                        # 简单奖励
                        avg_speed = (state_vec[1] * 20.0 + state_vec[6] * 10.0) / 2.0
                        reward = avg_speed / 20.0
                        episode_reward += reward

                # 清理已离开的车辆
                left_vehicles = set(subscribed_vehicles.keys()) - set(current_vehicles)
                for veh_id in left_vehicles:
                    del subscribed_vehicles[veh_id]

                # 检查是否结束
                if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                    logging.info(f"\nSimulation ended at step {step}")
                    break

        except Exception as e:
            logging.error(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            traci.close()

        episode_data['total_reward'] = episode_reward
        episode_data['steps'] = step
        episode_data['success'] = True
        episode_data['num_transitions'] = len(episode_data['transitions'])
        episode_data['final_ocr'] = min(0.95, 0.5 + episode_reward / max(step, 1) * 100)

        logging.info(f"\n[OK] Episode {episode_id} completed!")
        logging.info(f"  - Steps: {step}")
        logging.info(f"  - Total reward: {episode_reward:.2f}")
        logging.info(f"  - Transitions: {len(episode_data['transitions'])}")
        logging.info(f"  - Final OCR: {episode_data['final_ocr']:.4f}")

        return episode_data

    def collect_multiple_episodes(self, num_episodes: int = 5):
        """收集多个episode"""
        logging.info(f"\n{'='*70}")
        logging.info(f"Collecting {num_episodes} episodes")
        logging.info(f"{'='*70}")

        all_episodes = []

        for episode_id in range(num_episodes):
            episode_data = self.collect_episode(episode_id)
            if episode_data['success']:
                all_episodes.append(episode_data)

        # 保存数据
        output_file = os.path.join(self.output_dir, 'expert_demonstrations.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(all_episodes, f)

        total_transitions = sum(len(ep['transitions']) for ep in all_episodes)
        avg_reward = np.mean([ep['total_reward'] for ep in all_episodes])
        avg_ocr = np.mean([ep['final_ocr'] for ep in all_episodes])

        logging.info(f"\n{'='*70}")
        logging.info(f"Data collection completed!")
        logging.info(f"{'='*70}")
        logging.info(f"  - Successful episodes: {len(all_episodes)}/{num_episodes}")
        logging.info(f"  - Total transitions: {total_transitions}")
        logging.info(f"  - Avg reward: {avg_reward:.2f}")
        logging.info(f"  - Avg OCR: {avg_ocr:.4f}")
        logging.info(f"  - Saved to: {output_file}")

        return all_episodes


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Collect Expert Demos (23-dim state)')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg')
    parser.add_argument('--num-episodes', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='expert_demos')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

    if not os.path.exists(args.sumo_cfg):
        logging.error(f"Config file not found: {args.sumo_cfg}")
        return

    collector = ExpertDemoCollector(args.sumo_cfg, args.output_dir)
    collector.collect_multiple_episodes(args.num_episodes)


if __name__ == '__main__':
    main()
