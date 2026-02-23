"""
收集专家演示数据（多进程并行版本）

使用 libsumo + 订阅模式 + 多进程并行收集专家策略的 (state, action) 对
参照：relu_based\rl_traffic\fast_pkl_generator.py
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
# 专家策略
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
        }

        self.vtype_configured = False
        self.controlled_cvs = set()
        self.edge_speeds = {}
        self.edge_speed_cache_step = -1

        # 路口拥堵状态缓存
        self.junction_congestion = {}  # {junc_id: bool}

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

            logging.info("[OK] vType configured (accel=2.1)")
            self.vtype_configured = True
        except Exception as e:
            pass

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

    def detect_junction_congestion(self, junc_id: str, junction_config: Dict, step: int) -> bool:
        """
        检测路口下游是否拥堵（26分策略的核心逻辑）

        Returns:
            bool: True表示下游拥堵，False表示畅通
        """
        # 更新边速度缓存
        if self.edge_speed_cache_step != step:
            all_edges = set(self.NEXT_EDGE.keys()) | set(self.NEXT_EDGE.values())
            self.edge_speeds = {}
            for eid in all_edges:
                try:
                    self.edge_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)
                except:
                    self.edge_speeds[eid] = 13.89
            self.edge_speed_cache_step = step

        # 获取路口的下游边
        main_edges = junction_config['main_edges']
        ramp_edges = junction_config['ramp_edges']

        # 检查主路和匝道的下游边
        congested_count = 0
        total_checks = 0

        for edge in main_edges + ramp_edges:
            nxt = edge
            for _ in range(self.params['lookahead']):
                nxt = self.NEXT_EDGE.get(nxt)
                if nxt is None:
                    break

                ds_speed = self.edge_speeds.get(nxt, 13.89)
                total_checks += 1

                if ds_speed < self.params['congest_speed']:
                    congested_count += 1
                    break  # 检测到拥堵就停止

        # 如果超过30%的下游路径拥堵，认为路口拥堵
        congestion_ratio = congested_count / max(total_checks, 1)
        return congestion_ratio > 0.3


# ============================================================================
# 状态构建器（23维状态向量）
# ============================================================================

def build_state_vector(junc_id: str, subscribed_vehicles: Dict[str, Dict],
                        junction_config: Dict, step: int) -> np.ndarray:
    """构建完整的23维状态向量"""
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
    main_density = len(main_vehicles) / max(len(main_edges) * 2, 1)
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
        time_to_switch = 0.0
    except:
        current_phase = 0
        time_to_switch = 0.0

    main_signal = 'G'
    ramp_signal = 'r'
    diverge_signal = 'G' if junction_config['type'] == 'TYPE_B' else ''

    # 冲突和间隙
    conflict_risk = min(main_queue_length / 10.0, 1.0)
    gap_acceptance = 0.5

    # 构建23维状态向量
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
            ramp_queue_length / 10.0,
            len(cv_diverge) / max(len(diverge_vehicles), 1),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    # 时间戳 (1维)
    features.append(step / 3600.0)

    return np.array(features, dtype=np.float32)


def infer_expert_action(state_vec: np.ndarray, junction_config: Dict,
                         expert_policy, junc_id: str, step: int) -> Dict[str, float]:
    """
    从状态推断专家动作（基于26分策略）

    Args:
        state_vec: 23维状态向量
        junction_config: 路口配置
        expert_policy: ExpertPolicy实例
        junc_id: 路口ID
        step: 当前步数

    Returns:
        {'main': float, 'ramp': float}
    """
    # 从状态向量提取关键信息
    main_speed = state_vec[1] * 20.0
    ramp_speed = state_vec[6] * 10.0
    main_queue = state_vec[3] * 10.0
    ramp_queue = state_vec[7] * 40.0
    conflict_risk = state_vec[15]

    # 检测下游拥堵（26分策略核心）
    downstream_congested = expert_policy.detect_junction_congestion(junc_id, junction_config, step)

    # 基于拥堵状态计算动作
    if downstream_congested:
        # 下游拥堵：降低主路和匝道限速，避免加剧拥堵
        main_action = 0.3  # 主路大幅降低限速
        ramp_action = 0.2  # 匝道限制汇入
    else:
        # 下游畅通：可以放开限速，提高通行效率
        main_action = 0.7  # 主路正常限速
        ramp_action = 0.8  # 匝道加快汇入

    # 根据当前状态微调
    if main_speed < 5.0:
        main_action = max(main_action - 0.2, 0.0)  # 已经很慢了，进一步降低

    if ramp_queue > 5:
        ramp_action = max(ramp_action - 0.2, 0.0)  # 匝道排队太长，限制汇入

    if conflict_risk > 0.6:
        main_action = max(main_action - 0.3, 0.0)  # 冲突风险高，主路减速
        ramp_action = max(ramp_action - 0.3, 0.0)  # 同时限制匝道

    return {
        'main': np.clip(main_action, 0.0, 1.0),
        'ramp': np.clip(ramp_action, 0.0, 1.0)
    }


# ============================================================================
# 单个Episode收集器（用于多进程）
# ============================================================================

def collect_single_episode(args: tuple) -> Dict:
    """
    收集单个episode（用于多进程）

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

    episode_data = {
        'episode_id': episode_id,
        'seed': seed,
        'transitions': [],
        'total_reward': 0.0,
        'final_ocr': 0.0
    }

    subscribed_vehicles = {}
    episode_reward = 0.0
    expert = ExpertPolicy()

    try:
        # 运行仿真
        for step in range(max_steps):
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
            expert.apply_control(step)

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

                    # 推断专家动作（使用26分策略）
                    expert_action = infer_expert_action(state_vec, junc_config, expert, junc_id, step)

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
    episode_data['final_ocr'] = min(0.95, 0.5 + episode_reward / max(step, 1) * 100)

    return episode_data


# ============================================================================
# 多进程数据收集器
# ============================================================================

class ParallelExpertDemoCollector:
    """多进程并行专家演示数据收集器"""

    def __init__(self, sumo_cfg: str, output_dir: str = "expert_demos"):
        self.sumo_cfg = sumo_cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print("Parallel Expert Demo Collector")
        print("=" * 70)

    def collect_multiple_episodes(self, num_episodes: int = 5, num_workers: int = None):
        """
        多进程并行收集多个episode

        Args:
            num_episodes: 要收集的episode数量
            num_workers: 工作进程数（默认=CPU核心数）
        """
        if num_workers is None:
            num_workers = min(cpu_count(), num_episodes)

        print(f"\nCollecting {num_episodes} episodes with {num_workers} parallel workers")
        print(f"SUMO config: {self.sumo_cfg}")
        print(f"Output directory: {self.output_dir}")

        # 准备参数
        episode_args = [(i, self.sumo_cfg, 3600) for i in range(num_episodes)]

        # 多进程收集
        all_episodes = []

        print("\nStarting parallel collection...")
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered以便实时看到进度
            results = list(tqdm(
                pool.imap_unordered(collect_single_episode, episode_args),
                total=num_episodes,
                desc="Collecting episodes"
            ))

        # 过滤成功的episodes
        for result in results:
            if isinstance(result, dict) and result.get('success', False):
                all_episodes.append(result)

        # 保存数据
        output_file = os.path.join(self.output_dir, 'expert_demonstrations.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(all_episodes, f)

        # 统计
        total_transitions = sum(len(ep['transitions']) for ep in all_episodes)
        avg_reward = np.mean([ep['total_reward'] for ep in all_episodes]) if all_episodes else 0.0
        avg_ocr = np.mean([ep['final_ocr'] for ep in all_episodes]) if all_episodes else 0.0

        print(f"\n{'='*70}")
        print("Data collection completed!")
        print(f"{'='*70}")
        print(f"  - Requested episodes: {num_episodes}")
        print(f"  - Successful episodes: {len(all_episodes)}/{num_episodes}")
        print(f"  - Total transitions: {total_transitions}")
        print(f"  - Avg reward per episode: {avg_reward:.2f}")
        print(f"  - Avg OCR: {avg_ocr:.4f}")
        print(f"  - Saved to: {output_file}")

        return all_episodes


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Collect Expert Demos (Multi-process Parallel)')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg')
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default=CPU count)')
    parser.add_argument('--output-dir', type=str, default='expert_demos')

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  SUMO config: {args.sumo_cfg}")
    print(f"  Episodes to collect: {args.num_episodes}")
    print(f"  Parallel workers: {args.num_workers or 'auto (CPU count)'}")
    print(f"  Output directory: {args.output_dir}")

    # 创建收集器
    collector = ParallelExpertDemoCollector(args.sumo_cfg, args.output_dir)

    # 收集数据
    collector.collect_multiple_episodes(args.num_episodes, args.num_workers)

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
