"""
多智能体路口交通控制系统 - 简化版
只支持CUDA训练 + 文件IO并行数据收集
"""
import sys
import io

# 将标准输出强制设置为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


import os
from vehicle_type_config import normalize_speed, get_vehicle_max_speed
import sys
import argparse
import json
import time
import shutil
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
from multiprocessing import Process
import multiprocessing
import subprocess
import threading
import traceback as tb

from junction_agent import JunctionAgent
from road_topology_hardcoded import (
    JUNCTION_CONFIG,
    get_junction_main_edges,
    get_junction_ramp_edges,
    get_junction_edges,
    create_junction_config_from_dict
)

from junction_network import create_vehicle_level_model, NetworkConfig
from junction_trainer import PPOConfig, MultiAgentPPOTrainer

# 直接使用 libsumo（参考 fast_pkl_generator.py）
import libsumo as traci


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_environment():
    """检查运行环境"""
    print("\n环境检查:")

    # libsumo 必须可用
    print("  ✓ libsumo 已加载（高速模式）")

    cuda_available = torch.cuda.is_available()
    print(f"  ✓ CUDA: {cuda_available}")
    if cuda_available:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    GPU数量: {torch.cuda.device_count()}")

    cpu_count = multiprocessing.cpu_count()
    print(f"  ✓ CPU核心数: {cpu_count}")

    # 检测WSL
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                print("  ✓ WSL 环境")

    print(f"\n推荐配置: --workers {min(8, cpu_count)}")


def start_async_evaluation(model_path, sumo_cfg, iteration, eval_dir='evaluations', device='cuda'):
    """
    启动异步评估进程（不阻塞主线程）

    Args:
        model_path: 模型路径
        sumo_cfg: SUMO配置文件
        iteration: 当前迭代次数
        eval_dir: 评估结果目录
        device: 设备
    """
    def run_in_thread():
        try:
            cmd = [
                sys.executable, 'evaluate_model.py',
                '--model-path', model_path,
                '--sumo-cfg', sumo_cfg,
                '--iteration', str(iteration),
                '--eval-dir', eval_dir,
                '--device', device
            ]

            # 启动评估进程（不等待）
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as e:
            print(f"  ⚠️  异步评估失败: {e}")

    # 在后台线程中运行
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return thread


def create_libsumo_environment(sumo_cfg: str, seed: int = 42):
    """创建libsumo环境"""
    import logging
    import traceback as tb

    # 为worker配置日志
    worker_logger = logging.getLogger(f'sumo_worker')
    if not worker_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        worker_logger.addHandler(handler)
        worker_logger.setLevel(logging.INFO)

    import junction_agent  # 导入模块本身（用于设置traci连接）
    from junction_agent import JunctionAgent, SubscriptionManager

    class Environment:
        def __init__(self, sumo_cfg: str, seed: int):
            self.sumo_cfg = sumo_cfg
            self.seed = seed
            self.agents = {}
            self.is_running = False
            self.current_step = 0
            self.logger = worker_logger

            # 创建订阅管理器（订阅模式优化）
            self.sub_manager = SubscriptionManager()

            # OCR追踪：episode累计计数（SUMO返回的是单步计数）
            self.episode_arrived = 0
            self.episode_departed = 0

            # ✅ 全局CV车辆分配缓存（确保每个CV只被一个路口控制）
            self._global_cv_assignment: Dict[str, str] = {}  # {veh_id: junction_id}

            try:
                for junc_id in JUNCTION_CONFIG.keys():
                    # 从简化的字典配置创建完整的 JunctionConfig 对象
                    full_config = create_junction_config_from_dict(junc_id, JUNCTION_CONFIG[junc_id])
                    self.agents[junc_id] = JunctionAgent(
                        full_config,
                        self.sub_manager
                    )
                self.logger.info(f"Environment初始化完成（订阅模式），种子={seed}")
            except Exception as e:
                self.logger.error(f"Environment初始化失败: {e}\n{tb.format_exc()}")
                raise

        def reset(self):
            """重置环境"""
            try:
                self._start_sumo()
                self.current_step = 0

                for agent in self.agents.values():
                    agent.state_history.clear()

                # 重置奖励计算器
                if hasattr(self, 'reward_calculator'):
                    self.reward_calculator.reset()

                # 1. 初始热身步进
                for _ in range(10):
                    traci.simulationStep()
                    self.current_step += 1

                # 2. 设置订阅（订阅模式优化）
                self._setup_subscriptions()

                # ========== 关键修复：刷新订阅数据 ==========
                # 订阅请求发出后，必须执行一次 simulationStep 才会有数据返回
                traci.simulationStep()
                self.current_step += 1

                # 然后必须调用 update_results 将数据从 traci 拉取到 SubscriptionManager 缓存中
                self.sub_manager.update_results()
                # ==========================================

                # 4. 观察状态（此时 edge_results 已有数据）
                observations = {}
                for junc_id in self.agents.keys():
                    state = self.agents[junc_id].observe()
                    observations[junc_id] = self.agents[junc_id].get_state_vector(state)  # ✅ 返回向量，不是 JunctionState

                # 重置episode累计计数（SUMO getArrivedNumber/getDepartedNumber 为单步计数）
                self.episode_arrived = 0
                self.episode_departed = 0

                self.logger.info(f"环境重置完成（订阅模式），current_step={self.current_step}, "
                               f"episode计数已清零")
                return observations

            except Exception as e:
                self.logger.error(f"环境reset失败: {e}\n{tb.format_exc()}")
                raise

        def _setup_subscriptions(self):
            """设置所有路口的订阅（订阅模式优化）"""
            try:
                for agent in self.agents.values():
                    agent.setup_subscriptions()
                self.logger.info(f"订阅设置完成，覆盖 {len(self.agents)} 个路口")
            except Exception as e:
                self.logger.error(f"设置订阅失败: {e}\n{tb.format_exc()}")
                raise

        def _assign_all_cv_vehicles(self):
            """
            全局分配所有CV车辆给各个路口

            基于road_topology_hardcoded.py的拓扑关系
            """
            try:
                from road_topology_hardcoded import JUNCTION_CONFIG, EDGE_TOPOLOGY

                # 获取所有CV车辆
                all_cv_vehicles = []
                for veh_id in traci.vehicle.getIDList():
                    try:
                        if traci.vehicle.getTypeID(veh_id) == 'CV':
                            all_cv_vehicles.append(veh_id)
                    except:
                        continue

                # 清空之前的分配
                self._global_cv_assignment.clear()

                # 为每个CV车辆分配路口
                for veh_id in all_cv_vehicles:
                    try:
                        current_edge = traci.vehicle.getRoadID(veh_id)

                        assigned_junction = None

                        # J5的影响范围：E2(主路上游) + E23(匝道) + -E3(反向冲突)
                        if current_edge in ['E2', 'E23', '-E3']:
                            assigned_junction = 'J5'

                        # J14的影响范围：E9(主路上游) + E15(匝道) + -E10(反向冲突)
                        elif current_edge in ['E9', 'E15', '-E10']:
                            assigned_junction = 'J14'

                        # J15的影响范围：E10(主路上游) + E17(匝道) + -E11(反向冲突) + E16(转出)
                        elif current_edge in ['E10', 'E17', '-E11', 'E16']:
                            assigned_junction = 'J15'

                        # J17的影响范围：E12(主路上游) + E19(匝道) + -E13(反向冲突) + E18/E20(转出)
                        elif current_edge in ['E12', 'E19', '-E13', 'E18', 'E20']:
                            assigned_junction = 'J17'

                        # 如果车辆不在任何路口的直接影响范围内，根据拓扑关系分配
                        if assigned_junction is None:
                            if current_edge in EDGE_TOPOLOGY:
                                edge_info = EDGE_TOPOLOGY[current_edge]
                                for downstream_edge in edge_info.downstream:
                                    for junc_id in JUNCTION_CONFIG.keys():
                                        affected_edges = get_junction_edges(junc_id)
                                        if downstream_edge in affected_edges:
                                            assigned_junction = junc_id
                                            break
                                    if assigned_junction:
                                        break

                        # 默认分配策略
                        if assigned_junction is None:
                            if current_edge.startswith('E') and not current_edge.startswith('-E'):
                                if 'E1' <= current_edge <= 'E9':
                                    assigned_junction = 'J14'
                                elif 'E10' <= current_edge <= 'E13':
                                    assigned_junction = 'J15'
                                else:
                                    assigned_junction = 'J14'
                            elif current_edge.startswith('-E'):
                                if '-E1' <= current_edge <= '-E3':
                                    assigned_junction = 'J5'
                                elif '-E4' <= current_edge <= '-E11':
                                    assigned_junction = 'J15'
                                elif '-E12' <= current_edge <= '-E13':
                                    assigned_junction = 'J17'
                                else:
                                    assigned_junction = 'J5'
                            else:
                                assigned_junction = list(self.agents.keys())[0] if self.agents else 'J14'

                        self._global_cv_assignment[veh_id] = assigned_junction

                    except Exception as e:
                        default_junction = list(self.agents.keys())[0] if self.agents else 'J14'
                        self._global_cv_assignment[veh_id] = default_junction

            except Exception as e:
                self.logger.warning(f"全局CV车辆分配失败: {e}")

        def get_controlled_vehicles_for_junction(self, junc_id: str) -> dict[str, list[str]]:
            """
            获取指定路口控制的所有CV车辆

            Args:
                junc_id: 路口ID

            Returns:
                {'main': [...], 'ramp': [...], 'diverge': [...]}
            """
            if junc_id not in self.agents:
                return {'main': [], 'ramp': [], 'diverge': []}

            agent = self.agents[junc_id]
            config = agent.config

            # 获取分配给这个路口的所有CV车辆
            assigned_cvs = [
                veh_id for veh_id, assigned_junc in self._global_cv_assignment.items()
                if assigned_junc == junc_id
            ]

            # 根据车辆所在的边，分类到main/ramp/diverge（使用简化的配置）
            main_cvs = []
            ramp_cvs = []
            diverge_cvs = []

            # 使用辅助函数获取各类边
            ramp_edges = get_junction_ramp_edges(junc_id)
            main_edges = get_junction_main_edges(junc_id)

            for veh_id in assigned_cvs:
                try:
                    current_edge = traci.vehicle.getRoadID(veh_id)

                    if current_edge in ramp_edges:
                        ramp_cvs.append(veh_id)
                    elif agent.junction_type == junction_agent.JunctionType.TYPE_B and \
                         (current_edge in ['E16', 'E18', 'E20']):
                        diverge_cvs.append(veh_id)
                    else:
                        main_cvs.append(veh_id)
                except:
                    main_cvs.append(veh_id)

            return {
                'main': main_cvs,
                'ramp': ramp_cvs,
                'diverge': diverge_cvs if agent.junction_type == junction_agent.JunctionType.TYPE_B else []
            }

        def step(self, actions):
            """执行一步"""
            import time
            try:
                step_start = time.time()

                # 应用动作
                self._apply_actions(actions)

                # 仿真一步
                traci.simulationStep()
                self.current_step += 1

                # 维护episode累计计数（单步 -> 累计）
                self.episode_arrived += traci.simulation.getArrivedNumber()
                self.episode_departed += traci.simulation.getDepartedNumber()

                # ========== 更新订阅结果 ==========
                self.sub_manager.update_results()

                # 为新车辆设置订阅
                current_vehicles = set(traci.vehicle.getIDList())
                new_vehicles = current_vehicles - self.sub_manager.subscribed_vehicles
                if new_vehicles:
                    self.sub_manager.setup_vehicle_subscription(list(new_vehicles))

                # 清理已离开的车辆
                self.sub_manager.cleanup_left_vehicles(current_vehicles)

                # ✅ 更新全局CV车辆分配
                self._assign_all_cv_vehicles()

                # ❌ 禁用主动控制，让RL模型完全接管车辆控制
                # 原主动控制会覆盖模型设置的速度，导致模型无法学习
                # self._active_cv_control()

                # 观察新状态（订阅模式优化）
                obs_start = time.time()
                observations = {}
                for junc_id in self.agents.keys():
                    state = self.agents[junc_id].observe()
                    observations[junc_id] = self.agents[junc_id].get_state_vector(state)  # ✅ 返回向量，不是 JunctionState
                obs_time = (time.time() - obs_start) * 1000  # ms

                # 计算奖励
                rewards = self._compute_rewards()
                done = self.current_step >= 3600

                # 性能监控（每100步记录一次）
                if self.current_step % 100 == 0:
                    step_time = (time.time() - step_start) * 1000  # ms
                    self.logger.debug(f"Step {self.current_step}: 总耗时={step_time:.1f}ms, 观察={obs_time:.1f}ms")

                return observations, rewards, done, {}

            except Exception as e:
                self.logger.error(f"环境step失败: {e}\n{tb.format_exc()}")
                raise

        def _start_sumo(self):
            """启动SUMO（使用 libsumo）"""
            try:
                if self.is_running:
                    try:
                        traci.close()
                        self.logger.debug("关闭旧的SUMO连接")
                    except Exception as e:
                        self.logger.warning(f"关闭SUMO连接时出错: {e}")

                # 使用 libsumo 启动（参考 fast_pkl_generator.py）
                sumo_cmd = [
                    "sumo",
                    "-c", self.sumo_cfg,
                    "--no-warnings", "true",
                    "--seed", str(self.seed)
                ]
                traci.start(sumo_cmd)

                self.is_running = True
                self.logger.info(f"SUMO已启动 (seed={self.seed})")

                # 设置 junction_agent 的 traci 连接
                junction_agent.traci = traci
                self.logger.debug("已设置traci连接（libsumo模式）")

                # 配置vType参数（关键优化！）
                self._configure_vtypes()

            except Exception as e:
                self.logger.error(f"启动SUMO失败: {e}\n{tb.format_exc()}")
                raise

        def _apply_actions(self, actions):
            """应用动作到车辆"""
            failed_count = 0

            for junc_id, action_dict in actions.items():
                for veh_id, action in action_dict.items():
                    try:
                        speed_limit = 13.89
                        target_speed = speed_limit * (0.3 + 0.9 * action)
                        traci.vehicle.setSpeed(veh_id, target_speed)
                    except Exception as e:
                        failed_count += 1
                        if failed_count <= 3:  # 只记录前3个错误
                            self.logger.debug(f"设置车辆 {veh_id} 速度失败: {e}")

            if failed_count > 3:
                self.logger.debug(f"总计 {failed_count} 个车辆速度设置失败")

        def _compute_rewards(self):
            """计算奖励（使用Baseline OCR比较）"""
            # 使用基于Baseline OCR比较的奖励计算器
            from baseline_ocr_rewards import BaselineOCRRewardCalculator

            if not hasattr(self, 'reward_calculator'):
                # 检查是否有baseline文件（优先级：BC模型 > Expert(10步) > Expert(100步)）
                baseline_file = 'baseline_ocr/bc_baseline_10step.pkl'
                if not os.path.exists(baseline_file):
                    # fallback到专家策略10步版本
                    baseline_file = 'baseline_ocr/expert_baseline_10step.pkl'
                    if not os.path.exists(baseline_file):
                        # fallback到专家策略100步版本
                        baseline_file = 'baseline_ocr/expert_baseline.pkl'
                        if not os.path.exists(baseline_file):
                            self.logger.warning(f"未找到baseline文件")
                            self.logger.warning(f"  将使用固定baseline OCR = 0.95")
                            self.logger.warning(f"  建议先运行: python baseline_ocr_rewards.py --model bc_checkpoints/best_model.pt")
                            baseline_file = None

                if baseline_file:
                    self.logger.info(f"加载baseline OCR文件: {baseline_file}")

                self.reward_calculator = BaselineOCRRewardCalculator(
                    baseline_file=baseline_file,
                    reward_weight=100.0  # OCR增量奖励权重
                )

            # 获取当前OCR
            try:
                current_ocr = self._compute_current_ocr()
            except:
                current_ocr = 0.0

            env_stats = {
                'ocr': current_ocr,
                'step': self.current_step
            }

            # 计算奖励（OCR增量 + 瞬时辅助）
            rewards = self.reward_calculator.compute_rewards(self.agents, env_stats)

            # 调试：每500步打印一次奖励详情
            if self.current_step % 500 == 0 and self.current_step > 0:
                for junc_id, reward in rewards.items():
                    agent = self.agents.get(junc_id)
                    if agent and hasattr(agent, 'reward_breakdown') and agent.reward_breakdown:
                        bd = agent.reward_breakdown
                        self.logger.info(
                            f"路口 {junc_id} 奖励: {reward:.4f} "
                            f"[OCR delta={bd.get('ocr_delta', 0):+.4f} "
                            f"(current={bd.get('current_ocr', 0):.4f}, "
                            f"baseline={bd.get('baseline_ocr', 0):.4f}), "
                            f"ocr_reward={bd.get('ocr_reward', 0):.2f}, "
                            f"speed={bd.get('speed_reward', 0):.3f}, "
                            f"throughput={bd.get('throughput_reward', 0):.3f}]"
                        )
                    else:
                        self.logger.info(f"路口 {junc_id} 奖励: {reward:.4f} (OCR={current_ocr:.4f})")

            return rewards

        def _compute_current_ocr(self) -> float:
            """
            计算当前OCR（符合官方评测公式）

            官方公式:
            OCR = (N_arrived + Σ(d_i_traveled / d_i_total)) / N_total

            其中:
            - N_arrived: 已到达车辆数（episode期间）
            - d_i_traveled: 在途车辆i已行驶的距离
            - d_i_total: 在途车辆i的OD路径总长度
            - N_total: 总车辆数（已到达 + 在途）
            """
            try:
                # episode累计到达数
                n_arrived = self.episode_arrived

                # 在途车辆完成度
                enroute_completion = 0.0
                enroute_count = 0

                for veh_id in traci.vehicle.getIDList():
                    try:
                        # 获取车辆已行驶距离
                        current_edge = traci.vehicle.getRoadID(veh_id)
                        current_position = traci.vehicle.getLanePosition(veh_id)
                        route_edges = traci.vehicle.getRoute(veh_id)

                        # 计算已行驶距离
                        traveled_distance = 0.0
                        for edge in route_edges:
                            if edge == current_edge:
                                # 当前边，加上当前位置
                                traveled_distance += current_position
                                break
                            else:
                                # 已通过的边，加上边全长
                                try:
                                    edge_length = traci.edge.getLength(edge)
                                    traveled_distance += edge_length
                                except:
                                    try:
                                        lane_id = f"{edge}_0"
                                        edge_length = traci.lane.getLength(lane_id)
                                        traveled_distance += edge_length
                                    except:
                                        traveled_distance += 100.0

                        # 计算总路径长度
                        total_distance = 0.0
                        for edge in route_edges:
                            try:
                                edge_length = traci.edge.getLength(edge)
                                total_distance += edge_length
                            except:
                                try:
                                    lane_id = f"{edge}_0"
                                    edge_length = traci.lane.getLength(lane_id)
                                    total_distance += edge_length
                                except:
                                    total_distance += 100.0

                        # 计算该车辆的完成度
                        if total_distance > 0:
                            completion_ratio = min(traveled_distance / total_distance, 1.0)
                            enroute_completion += completion_ratio
                            enroute_count += 1

                    except Exception as e:
                        continue

                # 总车辆数 = 已到达 + 在途
                n_total = n_arrived + enroute_count

                if n_total == 0:
                    return 0.0

                # OCR = (已到达 + 在途车辆完成度之和) / 总车辆数
                ocr = (n_arrived + enroute_completion) / n_total

                # 调试信息（仅在最后几步打印）
                if self.current_step >= 3590:
                    n_departed = self.episode_departed
                    self.logger.info(
                        f"OCR详情 [步{self.current_step}]: "
                        f"出发={n_departed}, 到达={n_arrived}, 在途={enroute_count}, "
                        f"完成度={enroute_completion:.2f}, OCR={ocr:.4f}"
                    )

                return min(ocr, 1.0)

            except Exception as e:
                self.logger.warning(f"计算OCR失败: {e}")
                return 0.0

        def _configure_vtypes(self):
            """配置vType参数（基于规则方法的核心优化）"""
            import os

            # 从环境变量读取参数（与评测脚本一致）
            sigma = float(os.environ.get('CTRL_SIGMA', '0.0'))
            tau = float(os.environ.get('CTRL_TAU', '0.9'))
            accel = os.environ.get('CTRL_ACCEL', '0.8')
            decel = os.environ.get('CTRL_DECEL', '1.5')

            try:
                # CV参数：消除随机减速，平滑跟车
                traci.vehicletype.setImperfection('CV', sigma)
                traci.vehicletype.setTau('CV', tau)
                traci.vehicletype.setAccel('CV', float(accel))
                traci.vehicletype.setDecel('CV', float(decel))

                # HV参数：同样优化
                traci.vehicletype.setImperfection('HV', sigma)
                traci.vehicletype.setTau('HV', tau)
                traci.vehicletype.setAccel('HV', float(accel))
                traci.vehicletype.setDecel('HV', float(decel))

                self.logger.info(f"vType配置: sigma={sigma}, tau={tau}, accel={accel}, decel={decel}")
            except Exception as e:
                self.logger.warning(f"vType配置失败: {e}")

        def _active_cv_control(self):
            """CV主动速度引导（每一步都执行）"""
            import os

            # 检查是否启用主动控制
            active = int(os.environ.get('CTRL_ACTIVE', '1'))
            if active < 1:
                return

            # 控制参数
            approach_dist = float(os.environ.get('CTRL_APPROACH_DIST', '50.0'))
            congest_speed = float(os.environ.get('CTRL_CONGEST_SPEED', '5.0'))
            speed_factor = float(os.environ.get('CTRL_SPEED_FACTOR', '1.5'))
            speed_floor = float(os.environ.get('CTRL_SPEED_FLOOR', '3.0'))
            lookahead = int(os.environ.get('CTRL_LOOKAHEAD', '2'))

            # 道路拓扑（下游边映射）
            NEXT_EDGE = {
                'E1': 'E2', 'E2': 'E3', 'E3': 'E4', 'E4': 'E5',
                'E5': 'E6', 'E6': 'E7', 'E7': 'E8', 'E8': 'E9',
                'E9': 'E10', 'E10': 'E11', 'E11': 'E12', 'E12': 'E13',
                'E13': 'E14', 'E14': 'E15', 'E15': 'E16', 'E16': 'E17',
                'E17': 'E18', 'E18': 'E19', 'E19': 'E20', 'E20': 'E21',
                'E21': 'E22', 'E22': 'E23', 'E23': 'E24'
            }

            try:
                for veh_id in traci.vehicle.getIDList():
                    # 只控制CV车辆
                    if traci.vehicle.getTypeID(veh_id) != 'CV':
                        continue

                    try:
                        # 获取当前位置
                        road_id = traci.vehicle.getRoadID(veh_id)
                        lane_id = traci.vehicle.getLaneID(veh_id)
                        lane_pos = traci.vehicle.getLanePosition(veh_id)

                        # 车换不在主路上，跳过
                        if road_id not in NEXT_EDGE:
                            continue

                        # 检查是否接近边末尾
                        lane_len = 0
                        try:
                            lane_len = traci.lane.getLength(lane_id)
                        except:
                            try:
                                edge_len = traci.edge.getLength(road_id)
                                lane_len = edge_len
                            except:
                                continue

                        dist_to_end = lane_len - lane_pos
                        if dist_to_end > approach_dist:
                            continue

                        # 检测下游拥堵
                        congested = False
                        min_ds_speed = 100.0
                        current_edge = road_id

                        for _ in range(lookahead):
                            if current_edge not in NEXT_EDGE:
                                break
                            next_edge = NEXT_EDGE[current_edge]

                            try:
                                ds_speed = traci.edge.getLastStepMeanSpeed(next_edge)
                                min_ds_speed = min(min_ds_speed, ds_speed)

                                if ds_speed < congest_speed:
                                    congested = True
                                    break
                            except:
                                break

                            current_edge = next_edge

                        # 如果下游拥堵且当前速度过快，温和减速
                        if congested:
                            current_speed = traci.vehicle.getSpeed(veh_id)
                            target_speed = max(min_ds_speed * speed_factor, speed_floor)

                            if current_speed > target_speed:
                                # 使用slowDown温和减速（3秒持续时间）
                                traci.vehicle.slowDown(veh_id, target_speed, 3.0)

                    except Exception as e:
                        # 单辆车控制失败不影响其他车辆
                        continue

            except Exception as e:
                # 整体控制失败只记录，不中断
                self.logger.debug(f"主动控制失败: {e}")

        def close(self):
            """关闭环境"""
            if self.is_running:
                try:
                    traci.close()
                    self.logger.info("SUMO连接已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭SUMO连接时出错: {e}")
                self.is_running = False

    return Environment(sumo_cfg, seed)


def worker_process(worker_id, sumo_cfg, output_dir, seed, model_state, use_cuda):
    """工作进程 - 文件IO版本"""
    import traceback
    import logging

    # 配置worker日志
    worker_logger = logging.getLogger(f'worker_{worker_id}')
    if not worker_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f'[Worker-{worker_id}] [%(levelname)s] %(message)s'))
        worker_logger.addHandler(handler)
        worker_logger.setLevel(logging.INFO)

    try:
        import time
        worker_logger.info(f"Worker {worker_id} 启动，seed={seed}")
        worker_start = time.time()

        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

        # 根据worker_id分配GPU（支持双GPU）
        if use_cuda and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                # 将workers均匀分配到两张GPU
                # 例如8个workers: 0,1,2,3 → cuda:0 | 4,5,6,7 → cuda:1
                device_id = worker_id % gpu_count
                device = f'cuda:{device_id}'
                worker_logger.info(f"使用设备: {device} (Worker {worker_id} → GPU {device_id})")
            else:
                device = 'cuda'
                worker_logger.info(f"使用设备: {device} (单GPU模式)")
        else:
            device = 'cpu'
            worker_logger.info(f"使用设备: {device}")

        # 创建环境
        env = create_libsumo_environment(sumo_cfg, seed)
        worker_logger.info("环境创建成功")

        # 创建模型
        model = create_vehicle_level_model(JUNCTION_CONFIG)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        worker_logger.info("模型加载成功")

        # 收集经验 - 运行完整的3600步episode
        episode_start = time.time()
        obs = env.reset()  # obs 是 {junc_id: state_vector}
        experiences = []
        total_rewards = {}
        step_count = 0

        # 运行完整的episode，直到环境done
        while True:
            # 准备观察 - env.reset() 和 env.step() 都返回状态向量
            obs_tensors = {}
            vehicle_obs = {}

            for junc_id, state_vec in obs.items():
                try:
                    obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

                    # ✅ 使用全局CV分配（控制所有车道上的CV车辆）
                    controlled = env.get_controlled_vehicles_for_junction(junc_id)
                    vehicle_obs[junc_id] = {
                        'main': _get_vehicle_features(controlled['main'], device) if controlled['main'] else None,
                        'ramp': _get_vehicle_features(controlled['ramp'], device) if controlled['ramp'] else None,
                        'diverge': _get_vehicle_features(controlled['diverge'], device) if controlled['diverge'] else None
                    }

                    # 调试：打印第一次的观察信息
                    if step_count == 0:
                        worker_logger.info(f"[{junc_id}] 状态向量: shape={state_vec.shape}, "
                                        f"CV: main={len(controlled['main'])}, "
                                        f"ramp={len(controlled['ramp'])}, "
                                        f"diverge={len(controlled['diverge'])}")

                except Exception as e:
                    worker_logger.error(f"路口 {junc_id} 观察失败: {e}\n{traceback.format_exc()}")

            if not obs_tensors:
                worker_logger.error(f"没有有效的观察，obs_tensors为空！已运行{step_count}步")
                break

            # 检查vehicle_obs是否也有效
            if not any(vehicle_obs.values()):
                worker_logger.error(f"没有有效的车辆观察！已运行{step_count}步")
                # 不要break，继续尝试

            # 获取动作
            try:
                with torch.no_grad():
                    actions, values, info = model(obs_tensors, vehicle_obs, deterministic=False)
            except Exception as e:
                worker_logger.error(f"模型推理失败: {e}")
                break

            # 转换动作（车辆级控制）
            action_dict = {}
            total_controlled = 0
            sample_actions = {}  # 收集第一个路口的动作值用于调试

            for junc_id, action in actions.items():
                action_dict[junc_id] = {}
                try:
                    # ✅ 使用全局CV分配（控制所有车道上的CV车辆）
                    controlled = env.get_controlled_vehicles_for_junction(junc_id)

                    # ✅ 车辆级控制：为每辆车分配独立动作
                    if controlled['main'] and 'main_actions' in action:
                        main_actions = action['main_actions']  # [batch, num_main]
                        num_main = min(len(controlled['main']), main_actions.size(1))
                        for i, veh_id in enumerate(controlled['main'][:num_main]):
                            action_dict[junc_id][veh_id] = main_actions[0, i].item()
                            total_controlled += 1

                        # 收集第一个路口的主路动作用于调试
                        if junc_id not in sample_actions and num_main > 0:
                            sample_actions[junc_id] = {}
                            sample_actions[junc_id]['main_actions'] = main_actions[0, :num_main].tolist()
                            sample_actions[junc_id]['main_count'] = num_main

                    if controlled['ramp'] and 'ramp_actions' in action:
                        ramp_actions = action['ramp_actions']  # [batch, num_ramp]
                        num_ramp = min(len(controlled['ramp']), ramp_actions.size(1))
                        for i, veh_id in enumerate(controlled['ramp'][:num_ramp]):
                            action_dict[junc_id][veh_id] = ramp_actions[0, i].item()
                            total_controlled += 1

                        # 收集第一个路口的匝道动作用于调试
                        if junc_id not in sample_actions and num_ramp > 0:
                            sample_actions[junc_id] = {}
                            sample_actions[junc_id]['ramp_actions'] = ramp_actions[0, :num_ramp].tolist()
                            sample_actions[junc_id]['ramp_count'] = num_ramp

                    if controlled.get('diverge') and 'diverge_actions' in action:
                        diverge_actions = action['diverge_actions']  # [batch, num_diverge]
                        num_diverge = min(len(controlled['diverge']), diverge_actions.size(1))
                        for i, veh_id in enumerate(controlled['diverge'][:num_diverge]):
                            action_dict[junc_id][veh_id] = diverge_actions[0, i].item()
                            total_controlled += 1

                        if junc_id not in sample_actions and num_diverge > 0:
                            sample_actions[junc_id] = {}
                            sample_actions[junc_id]['diverge_actions'] = diverge_actions[0, :num_diverge].tolist()
                            sample_actions[junc_id]['diverge_count'] = num_diverge

                except Exception as e:
                    worker_logger.debug(f"路口 {junc_id} 动作转换失败: {e}")

            # 调试：每100步打印一次控制统计和动作值
            if step_count % 100 == 0 and total_controlled > 0:
                # 计算全局CV统计和各路口分配
                total_cv_count = len(env._global_cv_assignment)

                # 统计每个路口的车辆分布
                junction_stats = {}
                for veh_id, junc_id in env._global_cv_assignment.items():
                    if junc_id not in junction_stats:
                        junction_stats[junc_id] = 0
                    junction_stats[junc_id] += 1

                # 获取第一个路口的动作值
                first_junc = next(iter(sample_actions.keys())) if sample_actions else None
                if first_junc:
                    info = sample_actions[first_junc]
                    action_str = f"main={info.get('main_action', 0):.3f} ({info.get('main_count', 0)}辆)"
                    if 'ramp_action' in info:
                        action_str += f", ramp={info['ramp_action']:.3f} ({info.get('ramp_count', 0)}辆)"
                    if 'diverge_action' in info:
                        action_str += f", diverge={info['diverge_action']:.3f} ({info.get('diverge_count', 0)}辆)"

                    # 构建路口分配字符串
                    junc_str = ", ".join([f"{j}={junction_stats[j]}" for j in sorted(junction_stats.keys())])

                    worker_logger.info(f"全局CV: {total_cv_count}辆 | 分配: [{junc_str}] | {first_junc} 动作: {action_str}")
                else:
                    junc_str = ", ".join([f"{j}={junction_stats[j]}" for j in sorted(junction_stats.keys())])
                    worker_logger.info(f"全局CV: {total_cv_count}辆 | 分配: [{junc_str}] | 模型控制: {total_controlled}辆")

            # 执行动作
            try:
                next_obs, rewards, done, info = env.step(action_dict)
            except Exception as e:
                worker_logger.error(f"环境step失败: {e}\n{traceback.format_exc()}")
                break

            # 存储经验（现在可以获取reward了）
            # 调试：第一次step时打印rewards字典
            if step_count == 0:
                worker_logger.info(f"rewards字典键: {list(rewards.keys())}")
                worker_logger.info(f"env.agents字典键: {list(env.agents.keys())}")
                worker_logger.info(f"obs_tensors字典键: {list(obs_tensors.keys())}")
                worker_logger.info(f"vehicle_obs字典键: {list(vehicle_obs.keys())}")

            for junc_id in env.agents.keys():
                try:
                    reward = rewards.get(junc_id, 0.0)

                    # 调试：第一次step时打印每个路口的奖励
                    if step_count == 0:
                        worker_logger.info(f"路口 {junc_id} 奖励: {reward:.6f}")

                    value = values.get(junc_id, torch.tensor(0.0))
                    log_prob = _compute_log_prob(info.get(junc_id, {}), actions.get(junc_id, {}))

                    # 检查obs_tensors和vehicle_obs是否包含该路口
                    if junc_id not in obs_tensors:
                        worker_logger.error(f"路口 {junc_id} 不在 obs_tensors 中！跳过经验存储")
                        continue
                    if junc_id not in vehicle_obs:
                        worker_logger.error(f"路口 {junc_id} 不在 vehicle_obs 中！跳过经验存储")
                        continue

                    experiences.append({
                        'junction_id': junc_id,
                        'state': obs_tensors[junc_id].squeeze(0).cpu().numpy(),
                        'vehicle_obs': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in vehicle_obs[junc_id].items()},
                        'action': {k: v.item() if torch.is_tensor(v) else v for k, v in actions.get(junc_id, {}).items()},
                        'reward': reward,
                        'value': value.item() if torch.is_tensor(value) else value,
                        'log_prob': log_prob
                    })

                    # 累计奖励
                    if junc_id not in total_rewards:
                        total_rewards[junc_id] = 0.0
                    total_rewards[junc_id] += reward

                    if step_count == 0:
                        worker_logger.info(f"路口 {junc_id} 经验已添加，当前experiences长度: {len(experiences)}")

                except Exception as e:
                    worker_logger.error(f"存储路口 {junc_id} 经验失败: {e}\n{traceback.format_exc()}")

            # 调试：每100步打印一次experiences长度
            if step_count % 100 == 0:
                worker_logger.info(f"步数 {step_count}, 已收集 {len(experiences)} 条经验")

            obs = next_obs
            step_count += 1

            # 每1000步记录一次进度
            if step_count % 1000 == 0:
                worker_logger.info(f"已运行 {step_count} 步")

            if done:
                break

        episode_time = time.time() - episode_start
        worker_logger.info(f"Worker {worker_id} 完成，收集 {len(experiences)} 步经验，耗时 {episode_time:.1f}秒")

        # 计算最终OCR（在关闭环境之前！）
        try:
            final_ocr = env._compute_current_ocr()
            worker_logger.info(f"最终OCR: {final_ocr:.4f}")
        except Exception as e:
            worker_logger.warning(f"计算OCR失败: {e}")
            final_ocr = 0.0

        try:
            env.close()
        except Exception as e:
            worker_logger.warning(f"关闭环境时出错: {e}")

        # 保存到文件
        output_file = os.path.join(output_dir, f'worker_{worker_id}.pkl')
        result_data = {
            'worker_id': worker_id,
            'experiences': experiences,
            'total_rewards': total_rewards,
            'steps': len(experiences),
            'ocr': final_ocr  # 添加OCR到结果中
        }

        try:
            with open(output_file, 'wb') as f:
                pickle.dump(result_data, f)

            with open(os.path.join(output_dir, f'worker_{worker_id}.done'), 'w') as f:
                f.write('done')

            worker_logger.info(f"结果已保存到 {output_file}")
        except Exception as e:
            worker_logger.error(f"保存结果失败: {e}\n{traceback.format_exc()}")
            raise

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        worker_logger.error(f"Worker {worker_id} 发生错误:\n{error_msg}")

        try:
            with open(os.path.join(output_dir, f'worker_{worker_id}.error'), 'w') as f:
                f.write(error_msg)
        except Exception as save_error:
            worker_logger.error(f"保存错误信息失败: {save_error}")


def _get_vehicle_features(vehicle_ids, device):
    """获取车辆特征"""
    if not vehicle_ids:
        return None

    MAX_VEHICLES = 350  # 最大车辆数
    features = []
    for veh_id in vehicle_ids[:MAX_VEHICLES]:
        try:
            features.append([
                normalize_speed(traci.vehicle.getSpeed(veh_id)),
                traci.vehicle.getLanePosition(veh_id) / 500.0,
                traci.vehicle.getLaneIndex(veh_id) / 3.0,
                traci.vehicle.getWaitingTime(veh_id) / 60.0,
                traci.vehicle.getAcceleration(veh_id) / 5.0,
                1.0 if traci.vehicle.getTypeID(veh_id) == 'CV' else 0.0,
                traci.vehicle.getRouteIndex(veh_id) / 10.0,
                0.0
            ])
        except Exception as e:
            print(f"获取车辆 {veh_id} 特征失败: {e}")
            continue

    if not features:
        return None

    # 填充到MAX_VEHICLES
    while len(features) < MAX_VEHICLES:
        features.append([0.0] * 8)

    # 返回2D张量 [N, 8]，让收集代码处理batch维度
    return torch.tensor(features, dtype=torch.float32, device=device)


def _compute_log_prob(info, actions):
    """计算对数概率"""
    log_prob = 0.0
    for key in ['main', 'ramp', 'diverge']:
        if f'{key}_probs' in info and key in actions:
            probs = info[f'{key}_probs']
            action = actions[key]
            if torch.is_tensor(probs) and torch.is_tensor(action):
                action_idx = int(action.item() * 10)
                action_idx = min(action_idx, probs.size(-1) - 1)
                log_prob += torch.log(probs[0, action_idx] + 1e-8).item()
    return log_prob


def train(args):
    """训练函数"""
    print_header("多智能体路口控制 - 训练")

    # 环境检查
    check_environment()

    # 配置
    net_config = NetworkConfig()
    ppo_config = PPOConfig()

    if args.lr:
        ppo_config.lr = args.lr
    if args.batch_size:
        ppo_config.batch_size = args.batch_size

    num_workers = args.workers or multiprocessing.cpu_count()

    print(f"\n训练配置:")
    print(f"  SUMO配置: {args.sumo_cfg}")
    print(f"  总步数: {args.total_timesteps}")
    print(f"  学习率: {ppo_config.lr}")
    print(f"  批大小: {ppo_config.batch_size}")
    print(f"  设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 显示GPU分配信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  GPU数量: {gpu_count}")
        if gpu_count >= 2 and num_workers > 1:
            workers_per_gpu = num_workers // gpu_count
            print(f"  Worker分配:")
            for i in range(gpu_count):
                start_worker = i * workers_per_gpu
                end_worker = (i + 1) * workers_per_gpu if i < gpu_count - 1 else num_workers
                worker_range = f"{start_worker}-{end_worker-1}" if end_worker - start_worker > 1 else str(start_worker)
                print(f"    cuda:{i}: Workers [{worker_range}] ({end_worker - start_worker}个)")

    print(f"  并行环境数 (Worker进程): {num_workers}")

    # 创建时间戳子目录
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'train_{timestamp}')
    log_dir = os.path.join(args.log_dir, f'train_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n输出目录:")
    print(f"  模型保存: {save_dir}")
    print(f"  日志保存: {log_dir}")

    # 创建模型
    model = create_vehicle_level_model(JUNCTION_CONFIG, net_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 加载预训练模型（如果有）
    if args.pretrained:
        print(f"\n加载预训练模型: {args.pretrained}")
        try:
            # PyTorch 2.6+ 需要设置 weights_only=False 以加载包含 numpy 数据的 checkpoint
            checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)

            # 支持两种checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  ✓ 从checkpoint加载模型")
                    if 'epoch' in checkpoint:
                        print(f"  ✓ 预训练epoch: {checkpoint['epoch']}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"  ✓ 直接加载state_dict")
            else:
                model.load_state_dict(checkpoint)
                print(f"  ✓ 直接加载模型")

            print(f"  ✓ 预训练模型加载成功")
        except Exception as e:
            print(f"  ✗ 预训练模型加载失败: {e}")
            print(f"  ℹ 将从头开始训练")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.lr)

    # 经验缓冲区
    from junction_trainer import ExperienceBuffer
    buffer = ExperienceBuffer()

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)

    # 临时目录
    temp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    print(f"  临时目录: {temp_dir}")

    # 计算总共需要的迭代次数
    num_iterations = (args.total_timesteps + args.update_frequency * num_workers - 1) // (args.update_frequency * num_workers)

    # 训练循环
    timesteps = 0
    best_ocr = 0.0
    entropy_coef = ppo_config.entropy_coef
    iteration_ocr_history = []  # 跟踪每次迭代的OCR

    print(f"\n开始训练...")
    print(f"预计迭代次数: {num_iterations}")
    print(f"每次迭代步数: ~{args.update_frequency * num_workers}")
    print("=" * 70)

    try:
        # 创建进度条
        pbar = tqdm(range(num_iterations), desc="训练进度", unit="iter",
                    ncols=120, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for iteration in pbar:
            start_time = time.time()

            # 清空临时目录
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except Exception as e:
                    print(f"删除临时文件 {f} 失败: {e}")

            # 启动工作进程（每个worker使用不同的种子）
            processes = []
            use_cuda = torch.cuda.is_available()  # 只要有CUDA就使用，workers可以共享GPU

            for worker_id in range(num_workers):
                worker_seed = 42 + worker_id + iteration * 100  # 每次迭代也使用不同的种子
                p = Process(
                    target=worker_process,
                    args=(worker_id, args.sumo_cfg, temp_dir, worker_seed,
                          model.state_dict(),
                          use_cuda)  # 传递use_cuda标志
                )
                p.start()
                processes.append(p)

            # 等待完成
            for p in processes:
                p.join(timeout=600)
                if p.is_alive():
                    p.terminate()

            # 读取结果（使用tqdm显示）
            total_rewards = {}
            total_steps = 0
            worker_stats = []
            worker_ocrs = []  # 收集所有worker的OCR

            for worker_id in tqdm(range(num_workers), desc="  收集数据", leave=False, ncols=100):
                result_file = os.path.join(temp_dir, f'worker_{worker_id}.pkl')
                error_file = os.path.join(temp_dir, f'worker_{worker_id}.error')

                if os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        error_msg = f.read()
                    tqdm.write(f"  ❌ Worker {worker_id} 错误: {error_msg[:50]}...")
                    continue

                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'rb') as f:
                            result_data = pickle.load(f)

                        exp_count = len(result_data.get('experiences', []))
                        ocr = result_data.get('ocr', 0.0)
                        worker_ocrs.append(ocr)
                        tqdm.write(f"  📦 Worker {worker_id}: OCR={ocr:.4f}, 读取 {exp_count} 条经验")

                        added_count = 0
                        for exp in result_data['experiences']:
                            try:
                                # 使用pin_memory加速CPU到GPU传输
                                state_tensor = torch.from_numpy(exp['state']).float().pin_memory().to(device, non_blocking=True)

                                # 处理vehicle_obs - 确保所有值都是正确的类型
                                vehicle_obs = {}
                                for k, v in exp['vehicle_obs'].items():
                                    if isinstance(v, np.ndarray):
                                        # 异步传输到GPU
                                        vehicle_obs[k] = torch.from_numpy(v).float().pin_memory().to(device, non_blocking=True)
                                    elif v is None:
                                        vehicle_obs[k] = None
                                    else:
                                        vehicle_obs[k] = v

                                # 确保action也是正确的格式
                                action = exp['action']
                                if not isinstance(action, dict):
                                    action = {}

                                buffer.add(
                                    exp['junction_id'], state_tensor, vehicle_obs,
                                    action, exp['reward'], exp['value'], exp['log_prob'], False
                                )
                                added_count += 1
                            except Exception as e:
                                tqdm.write(f"  ⚠️  添加经验失败: {e}")
                                continue

                        tqdm.write(f"  ✅ 成功添加 {added_count}/{exp_count} 条经验到缓冲区")

                        # 收集统计
                        worker_reward = sum(result_data['total_rewards'].values())
                        worker_steps = result_data['steps']
                        worker_stats.append({
                            'worker_id': worker_id,
                            'steps': worker_steps,
                            'reward': worker_reward
                        })

                        for junc_id, reward in result_data['total_rewards'].items():
                            if junc_id not in total_rewards:
                                total_rewards[junc_id] = 0.0
                            total_rewards[junc_id] += reward

                        total_steps += result_data['steps']

                        # 打印缓冲区状态
                        tqdm.write(f"  📊 当前缓冲区大小: {len(buffer)}")

                    except Exception as e:
                        tqdm.write(f"  ⚠️  Worker {worker_id} 读取失败: {e}\n{tb.format_exc()}")

            timesteps += total_steps
            collect_time = time.time() - start_time

            # ========== 保存训练前统计 ==========
            buffer_size_before = len(buffer)
            total_rewards_sum = sum(total_rewards.values()) if total_rewards else 0.0
            mean_reward_before = total_rewards_sum / len(total_rewards) if total_rewards else 0.0

            # 更新模型
            update_start = time.time()

            # 使用标准训练器更新
            trainer = MultiAgentPPOTrainer(model, ppo_config, device)
            trainer.buffer = buffer
            trainer.entropy_coef = entropy_coef
            update_result = trainer.update()
            entropy_coef = trainer.entropy_coef

            update_time = time.time() - update_start

            # 记录
            mean_reward = np.mean(list(total_rewards.values())) if total_rewards else 0.0

            writer.add_scalar('train/reward', mean_reward, timesteps)
            writer.add_scalar('train/loss', update_result['loss'], timesteps)
            writer.add_scalar('train/collect_time', collect_time, timesteps)
            writer.add_scalar('train/update_time', update_time, timesteps)
            writer.add_scalar('train/entropy_coef', entropy_coef, timesteps)

            # 记录OCR到TensorBoard
            if worker_ocrs:
                mean_ocr = np.mean(worker_ocrs)
                writer.add_scalar('train/ocr', mean_ocr, timesteps)
                # 记录预估得分
                estimated_score = max(0, (mean_ocr - 0.8812) / 0.8812 * 100)
                writer.add_scalar('train/estimated_score', estimated_score, timesteps)

            # ========== 模型更新完成日志 ==========
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"🔄 模型更新完成 - 迭代 {iteration + 1}/{num_iterations}")
            tqdm.write(f"{'='*70}")
            tqdm.write(f"📊 训练统计:")
            tqdm.write(f"  - 总步数: {timesteps:,} / {args.total_timesteps:,} ({timesteps/args.total_timesteps*100:.1f}%)")
            tqdm.write(f"  - 本次收集: {total_steps:,} 步")
            tqdm.write(f"  - 训练前缓冲区: {buffer_size_before:,} 样本")  # 使用训练前的大小
            tqdm.write(f"  - 训练后缓冲区: {len(buffer):,} 样本 (已清空)")
            tqdm.write(f"\n⏱️  时间统计:")
            tqdm.write(f"  - 数据收集: {collect_time:.1f}秒")
            tqdm.write(f"  - 模型更新: {update_time:.1f}秒")
            tqdm.write(f"  - 总耗时: {collect_time + update_time:.1f}秒")
            tqdm.write(f"\n🎯 性能指标:")

            # OCR统计
            if worker_ocrs:
                mean_ocr = np.mean(worker_ocrs)
                std_ocr = np.std(worker_ocrs)
                min_ocr = np.min(worker_ocrs)
                max_ocr = np.max(worker_ocrs)
                tqdm.write(f"  - 平均OCR: {mean_ocr:.4f} ± {std_ocr:.4f} (范围: {min_ocr:.4f} - {max_ocr:.4f})")
                # 使用固定的baseline OCR进行评分（0.94）
                baseline_ocr = 0.94
                score_estimate = (mean_ocr - baseline_ocr) / baseline_ocr * 100
                tqdm.write(f"  - 得分预估: {score_estimate:.2f} (基准OCR={baseline_ocr:.4f})")

            tqdm.write(f"  - 平均奖励: {mean_reward_before:.4f}")  # 使用训练前计算的奖励
            tqdm.write(f"  - 损失: {update_result['loss']:.4f}")
            tqdm.write(f"  - 熵系数: {entropy_coef:.6f}")
            tqdm.write(f"\n🏢 路口奖励详情:")
            for junc_id, reward in sorted(total_rewards.items()):
                tqdm.write(f"  - {junc_id}: {reward:.4f}")
            tqdm.write(f"{'='*70}\n")

            # 更新进度条后缀
            postfix_dict = {
                'steps': f'{timesteps:,}',
                'reward': f'{mean_reward:.2f}',
                'loss': f'{update_result["loss"]:.4f}',
                'col_t': f'{collect_time:.1f}s',
                'upd_t': f'{update_time:.1f}s'
            }
            # 添加OCR到进度条
            if worker_ocrs:
                mean_ocr = np.mean(worker_ocrs)
                postfix_dict['ocr'] = f'{mean_ocr:.4f}'

                # 更新最佳OCR
                if mean_ocr > best_ocr:
                    best_ocr = mean_ocr
                    tqdm.write(f"🏆 新的最佳OCR: {best_ocr:.4f} (预估得分: {(best_ocr - 0.8812) / 0.8812 * 100:.2f})")

                iteration_ocr_history.append(mean_ocr)

            pbar.set_postfix(postfix_dict)

            # ========== 保存检查点并启动异步评估 ==========
            # 每5次迭代保存一次检查点
            if (iteration + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration+1:04d}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                tqdm.write(f"💾 检查点已保存: {checkpoint_path}\n")

                # 启动异步评估
                tqdm.write(f"🚀 启动异步评估（后台运行，不阻塞训练）...")
                eval_thread = start_async_evaluation(
                    model_path=checkpoint_path,
                    sumo_cfg=args.sumo_cfg,
                    iteration=iteration + 1,
                    eval_dir=os.path.join(save_dir, 'evaluations'),
                    device=device
                )
                tqdm.write(f"✅ 评估进程已启动（迭代 {iteration + 1}）\n")

            # 每10次迭代打印详细信息
            if (iteration + 1) % 10 == 0:
                tqdm.write(f"  Worker统计:")
                for stat in worker_stats:
                    tqdm.write(f"    Worker {stat['worker_id']}: {stat['steps']:,} 步, 奖励: {stat['reward']:.2f}")

        # 关闭进度条
        pbar.close()

        # 训练完成总结
        print_header("训练完成")
        print(f"总训练步数: {timesteps:,}")
        print(f"总迭代次数: {iteration + 1}")

        if iteration_ocr_history:
            print(f"\nOCR统计:")
            print(f"  初始OCR: {iteration_ocr_history[0]:.4f}")
            print(f"  最终OCR: {iteration_ocr_history[-1]:.4f}")
            print(f"  最佳OCR: {best_ocr:.4f}")
            print(f"  平均OCR: {np.mean(iteration_ocr_history):.4f} ± {np.std(iteration_ocr_history):.4f}")
            # 使用固定的baseline OCR进行评分（0.94）
            baseline_ocr = 0.94
            print(f"\n得分预估 (基准OCR={baseline_ocr:.4f}):")
            print(f"  初始得分: {(iteration_ocr_history[0] - baseline_ocr) / baseline_ocr * 100:.2f}")
            print(f"  最终得分: {(iteration_ocr_history[-1] - baseline_ocr) / baseline_ocr * 100:.2f}")
            print(f"  最佳得分: {(best_ocr - baseline_ocr) / baseline_ocr * 100:.2f}")

            # OCR改进
            ocr_improvement = (iteration_ocr_history[-1] - iteration_ocr_history[0]) / iteration_ocr_history[0] * 100
            print(f"\nOCR改进: {ocr_improvement:+.2f}%")

        print("=" * 70)

    finally:
        # 清理临时文件
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except Exception as e:
                print(f"删除临时文件 {f} 失败: {e}")
        writer.close()

    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n模型已保存: {final_model_path}")
    print(f"所有训练文件保存在: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='多智能体路口控制 - 训练')

    parser.add_argument('--sumo-cfg', type=str, required=True, help='SUMO配置文件')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--batch-size', type=int, default=2048, help='批大小')
    parser.add_argument('--workers', type=int, help='工作进程数（默认=CPU核心数，每个进程=1个并行环境）')
    parser.add_argument('--update-frequency', type=int, default=2048, help='更新频率')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--pretrained', type=str, default=None, help='预训练模型路径（行为克隆模型）')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
