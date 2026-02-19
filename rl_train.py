"""
多智能体路口交通控制系统 - 简化版
只支持CUDA训练 + 文件IO并行数据收集
"""

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

# ============== 订阅模式优化 ==============
# 使用订阅模式提升数据收集速度 7-8x
from junction_agent_subscription import JUNCTION_CONFIGS
# ==========================================

from junction_network import create_junction_model, NetworkConfig
from junction_trainer import PPOConfig, MultiAgentPPOTrainer

# 尝试导入libsumo
try:
    import libsumo as traci_wrapper
    USE_LIBSUMO = True
except ImportError:
    import traci as traci_wrapper
    USE_LIBSUMO = False


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_environment():
    """检查运行环境"""
    print("\n环境检查:")

    try:
        import libsumo
        print("  ✓ libsumo 可用（高速模式）")
    except ImportError:
        print("  ⚠ libsumo 不可用，将使用 traci")

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

    print(f"\n推荐配置: --num-envs {min(4, cpu_count)}")


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

    import junction_agent_subscription  # 导入模块本身（用于设置traci连接）
    from junction_agent_subscription import JunctionAgent, SubscriptionManager

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

            try:
                for junc_id in JUNCTION_CONFIGS.keys():
                    self.agents[junc_id] = JunctionAgent(
                        JUNCTION_CONFIGS[junc_id],
                        self.sub_manager
                    )
                self.logger.info(f"Environment初始化完成（订阅模式），种子={seed}")
            except Exception as e:
                self.logger.error(f"Environment初始化失败: {e}\n{tb.format_exc()}")
                raise

        def reset(self):
            """重置环境并应用CACC参数"""
            try:
                self._start_sumo()
                self.current_step = 0

                for agent in self.agents.values():
                    agent.state_history.clear()

                for _ in range(10):
                    traci_wrapper.simulationStep()
                    self.current_step += 1

                # 设置订阅（订阅模式优化）
                self._setup_subscriptions()

                # 应用CACC参数优化（与推理环境完全一致）
                self._apply_cacc_parameters()

                observations = {junc_id: self.agents[junc_id].observe() for junc_id in self.agents.keys()}
                self.logger.info(f"环境重置完成（订阅模式），current_step={self.current_step}")
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

        def step(self, actions):
            """执行一步"""
            import time
            try:
                step_start = time.time()

                # 应用动作
                self._apply_actions(actions)

                # 仿真一步
                traci_wrapper.simulationStep()
                self.current_step += 1

                # 观察新状态（订阅模式优化）
                obs_start = time.time()
                observations = {junc_id: self.agents[junc_id].observe() for junc_id in self.agents.keys()}
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
            """启动SUMO"""
            import sys
            import traci as traci_global  # 导入全局traci模块

            try:
                if self.is_running:
                    try:
                        traci_wrapper.close()
                        self.logger.debug("关闭旧的SUMO连接")
                    except Exception as e:
                        self.logger.warning(f"关闭SUMO连接时出错: {e}")

                sumo_binary = "sumo"

                if USE_LIBSUMO:
                    sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--no-warnings", "true", "--seed", str(self.seed)]
                    traci_wrapper.start(sumo_cmd)
                else:
                    sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--remote-port", "0", "--no-warnings", "true", "--seed", str(self.seed)]
                    traci_wrapper.start(sumo_cmd)

                self.is_running = True
                self.logger.info(f"SUMO已启动 (seed={self.seed})")

                # 关键修复：设置订阅模式模块的traci连接
                # 1. 设置全局traci模块（sys.modules）
                sys.modules['traci'] = traci_wrapper
                # 2. 直接设置订阅模式模块的traci属性（因为模块级别引用已固定）
                junction_agent_subscription.traci = traci_wrapper
                self.logger.debug("已设置traci连接（订阅模式兼容）")

            except Exception as e:
                self.logger.error(f"启动SUMO失败: {e}\n{tb.format_exc()}")
                raise

        def _apply_cacc_parameters(self):
            """
            应用CACC参数优化

            核心策略：
            - sigma=0: 消除随机减速（完美驾驶），提高交通流稳定性
            - tau=1.12: 微增跟车时距（抵消sigma=0带来的容量增加，保持安全性）

            这个设置与推理环境完全一致，确保训练和推理的动作空间一致。
            """
            cacc_applied = set()  # 跟踪已设置的车辆，避免重复设置
            failed_vehicles = []  # 记录失败的车辆

            try:
                all_vehicles = traci_wrapper.vehicle.getIDList()
                self.logger.debug(f"开始应用CACC参数，车辆总数={len(all_vehicles)}")

                for veh_id in all_vehicles:
                    if veh_id in cacc_applied:
                        continue

                    try:
                        # 只对CV（Connected Vehicle）类型应用CACC参数
                        veh_type = traci_wrapper.vehicle.getTypeID(veh_id)
                        if veh_type == 'CV':
                            # 设置imperfection（sigma）为0，消除随机减速
                            traci_wrapper.vehicle.setImperfection(veh_id, 0.0)

                            # 设置tau（跟车时距）为1.12秒，略微增大以保持安全距离
                            traci_wrapper.vehicle.setTau(veh_id, 1.12)

                            cacc_applied.add(veh_id)
                    except Exception as e:
                        # 车辆可能在设置过程中离开路网，记录但不中断
                        failed_vehicles.append((veh_id, str(e)))

                self.logger.info(f"CACC参数应用完成: 成功={len(cacc_applied)}辆, 失败={len(failed_vehicles)}辆")
                if failed_vehicles and len(failed_vehicles) <= 5:
                    for veh_id, err in failed_vehicles[:5]:
                        self.logger.debug(f"  车辆 {veh_id} 设置失败: {err}")

            except Exception as e:
                self.logger.error(f"应用CACC参数时发生错误: {e}\n{tb.format_exc()}")

        def _apply_actions(self, actions):
            """应用动作到车辆"""
            failed_count = 0

            for junc_id, action_dict in actions.items():
                for veh_id, action in action_dict.items():
                    try:
                        speed_limit = 13.89
                        target_speed = speed_limit * (0.3 + 0.9 * action)
                        traci_wrapper.vehicle.setSpeed(veh_id, target_speed)
                    except Exception as e:
                        failed_count += 1
                        if failed_count <= 3:  # 只记录前3个错误
                            self.logger.debug(f"设置车辆 {veh_id} 速度失败: {e}")

            if failed_count > 3:
                self.logger.debug(f"总计 {failed_count} 个车辆速度设置失败")

        def _compute_rewards(self):
            """计算奖励"""
            rewards = {}
            for junc_id, agent in self.agents.items():
                try:
                    state = agent.current_state
                    if state is None:
                        rewards[junc_id] = 0.0
                        continue

                    throughput = -state.main_queue_length * 0.1 - state.ramp_queue_length * 0.2
                    waiting = -state.ramp_waiting_time * 0.05
                    conflict = -state.conflict_risk * 0.5
                    gap = state.gap_acceptance * 0.2 if state.ramp_vehicles else 0
                    speed_stability = -abs(state.main_speed - state.ramp_speed) * 0.02

                    rewards[junc_id] = throughput + waiting + conflict + gap + speed_stability

                except Exception as e:
                    self.logger.warning(f"计算路口 {junc_id} 奖励失败: {e}")
                    rewards[junc_id] = 0.0

            return rewards

        def close(self):
            """关闭环境"""
            if self.is_running:
                try:
                    traci_wrapper.close()
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

        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        worker_logger.info(f"使用设备: {device}")

        # 创建环境
        env = create_libsumo_environment(sumo_cfg, seed)
        worker_logger.info("环境创建成功")

        # 创建模型
        model = create_junction_model(JUNCTION_CONFIGS)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        worker_logger.info("模型加载成功")

        # 收集经验 - 运行完整的3600步episode
        episode_start = time.time()
        obs = env.reset()
        experiences = []
        total_rewards = {}
        step_count = 0

        # 运行完整的episode，直到环境done
        while True:
            # 准备观察
            obs_tensors = {}
            vehicle_obs = {}

            for junc_id, agent in env.agents.items():
                try:
                    state_vec = agent.get_state_vector()
                    obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

                    controlled = agent.get_controlled_vehicles()
                    vehicle_obs[junc_id] = {
                        'main': _get_vehicle_features(controlled['main'], device) if controlled['main'] else None,
                        'ramp': _get_vehicle_features(controlled['ramp'], device) if controlled['ramp'] else None,
                        'diverge': _get_vehicle_features(controlled['diverge'], device) if controlled['diverge'] else None
                    }
                except Exception as e:
                    worker_logger.debug(f"路口 {junc_id} 观察失败: {e}")

            if not obs_tensors:
                worker_logger.warning("没有有效的观察，跳过此步")
                break

            # 获取动作
            try:
                with torch.no_grad():
                    actions, values, info = model(obs_tensors, vehicle_obs, deterministic=False)
            except Exception as e:
                worker_logger.error(f"模型推理失败: {e}")
                break

            # 转换动作
            action_dict = {}
            for junc_id, action in actions.items():
                action_dict[junc_id] = {}
                try:
                    controlled = env.agents[junc_id].get_controlled_vehicles()

                    if controlled['main'] and 'main' in action:
                        for veh_id in controlled['main'][:1]:
                            action_dict[junc_id][veh_id] = action['main'].item()

                    if controlled['ramp'] and 'ramp' in action:
                        for veh_id in controlled['ramp'][:1]:
                            action_dict[junc_id][veh_id] = action['ramp'].item()
                except Exception as e:
                    worker_logger.debug(f"路口 {junc_id} 动作转换失败: {e}")

            # 执行动作
            try:
                next_obs, rewards, done, info = env.step(action_dict)
            except Exception as e:
                worker_logger.error(f"环境step失败: {e}\n{traceback.format_exc()}")
                break

            # 存储经验（现在可以获取reward了）
            for junc_id in env.agents.keys():
                try:
                    reward = rewards.get(junc_id, 0.0)
                    value = values.get(junc_id, torch.tensor(0.0))
                    log_prob = _compute_log_prob(info.get(junc_id, {}), actions.get(junc_id, {}))

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
                except Exception as e:
                    worker_logger.debug(f"存储路口 {junc_id} 经验失败: {e}")

            obs = next_obs
            step_count += 1

            # 每1000步记录一次进度
            if step_count % 1000 == 0:
                worker_logger.info(f"已运行 {step_count} 步")

            if done:
                break

        try:
            env.close()
        except Exception as e:
            worker_logger.warning(f"关闭环境时出错: {e}")

        episode_time = time.time() - episode_start
        worker_logger.info(f"Worker {worker_id} 完成，收集 {len(experiences)} 步经验，耗时 {episode_time:.1f}秒")

        # 保存到文件
        output_file = os.path.join(output_dir, f'worker_{worker_id}.pkl')
        result_data = {
            'worker_id': worker_id,
            'experiences': experiences,
            'total_rewards': total_rewards,
            'steps': len(experiences)
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

    features = []
    for veh_id in vehicle_ids[:10]:
        try:
            features.append([
                normalize_speed(traci_wrapper.vehicle.getSpeed(veh_id)),
                traci_wrapper.vehicle.getLanePosition(veh_id) / 500.0,
                traci_wrapper.vehicle.getLaneIndex(veh_id) / 3.0,
                traci_wrapper.vehicle.getWaitingTime(veh_id) / 60.0,
                traci_wrapper.vehicle.getAcceleration(veh_id) / 5.0,
                1.0 if traci_wrapper.vehicle.getTypeID(veh_id) == 'CV' else 0.0,
                traci_wrapper.vehicle.getRouteIndex(veh_id) / 10.0,
                0.0
            ])
        except:
            continue

    if not features:
        return None

    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


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
    num_envs = min(args.num_envs, num_workers)

    print(f"\n训练配置:")
    print(f"  SUMO配置: {args.sumo_cfg}")
    print(f"  总步数: {args.total_timesteps}")
    print(f"  学习率: {ppo_config.lr}")
    print(f"  批大小: {ppo_config.batch_size}")
    print(f"  设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  并行环境: {num_envs}")
    print(f"  工作进程: {num_workers}")

    # 创建模型
    model = create_junction_model(JUNCTION_CONFIGS, net_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.lr)

    # 经验缓冲区
    from junction_trainer import ExperienceBuffer
    buffer = ExperienceBuffer()

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.log_dir)

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
                except:
                    pass

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

                        for exp in result_data['experiences']:
                            # 使用pin_memory加速CPU到GPU传输
                            state_tensor = torch.from_numpy(exp['state']).float().pin_memory().to(device, non_blocking=True)
                            vehicle_obs = {}
                            for k, v in exp['vehicle_obs'].items():
                                if isinstance(v, np.ndarray):
                                    # 异步传输到GPU
                                    vehicle_obs[k] = torch.from_numpy(v).float().pin_memory().to(device, non_blocking=True)
                                else:
                                    vehicle_obs[k] = v

                            buffer.add(
                                exp['junction_id'], state_tensor, vehicle_obs,
                                exp['action'], exp['reward'], exp['value'], exp['log_prob'], False
                            )

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

                    except Exception as e:
                        tqdm.write(f"  ⚠️  Worker {worker_id} 读取失败: {e}")

            timesteps += total_steps
            collect_time = time.time() - start_time

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

            # ========== 模型更新完成日志 ==========
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"🔄 模型更新完成 - 迭代 {iteration + 1}/{num_iterations}")
            tqdm.write(f"{'='*70}")
            tqdm.write(f"📊 训练统计:")
            tqdm.write(f"  - 总步数: {timesteps:,} / {args.total_timesteps:,} ({timesteps/args.total_timesteps*100:.1f}%)")
            tqdm.write(f"  - 本次收集: {total_steps:,} 步")
            tqdm.write(f"  - 缓冲区大小: {len(buffer):,} 样本")
            tqdm.write(f"\n⏱️  时间统计:")
            tqdm.write(f"  - 数据收集: {collect_time:.1f}秒")
            tqdm.write(f"  - 模型更新: {update_time:.1f}秒")
            tqdm.write(f"  - 总耗时: {collect_time + update_time:.1f}秒")
            tqdm.write(f"\n🎯 性能指标:")
            tqdm.write(f"  - 平均奖励: {mean_reward:.4f}")
            tqdm.write(f"  - 损失: {update_result['loss']:.4f}")
            tqdm.write(f"  - 熵系数: {entropy_coef:.6f}")
            tqdm.write(f"\n🏢 路口奖励详情:")
            for junc_id, reward in sorted(total_rewards.items()):
                tqdm.write(f"  - {junc_id}: {reward:.4f}")
            tqdm.write(f"{'='*70}\n")

            # 更新进度条后缀
            pbar.set_postfix({
                'steps': f'{timesteps:,}',
                'reward': f'{mean_reward:.2f}',
                'loss': f'{update_result["loss"]:.4f}',
                'col_t': f'{collect_time:.1f}s',
                'upd_t': f'{update_time:.1f}s'
            })

            # ========== 保存检查点并启动异步评估 ==========
            # 每5次迭代保存一次检查点
            if (iteration + 1) % 5 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_iter_{iteration+1:04d}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                tqdm.write(f"💾 检查点已保存: {checkpoint_path}\n")

                # 启动异步评估
                tqdm.write(f"🚀 启动异步评估（后台运行，不阻塞训练）...")
                eval_thread = start_async_evaluation(
                    model_path=checkpoint_path,
                    sumo_cfg=args.sumo_cfg,
                    iteration=iteration + 1,
                    eval_dir=os.path.join(args.save_dir, 'evaluations'),
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

    finally:
        # 清理临时文件
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        writer.close()

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pt'))
    print(f"\n模型已保存: {args.save_dir}/final_model.pt")


def main():
    parser = argparse.ArgumentParser(description='多智能体路口控制 - 训练')

    parser.add_argument('--sumo-cfg', type=str, required=True, help='SUMO配置文件')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--num-envs', type=int, default=4, help='并行环境数量')
    parser.add_argument('--workers', type=int, help='工作进程数（默认=CPU核心数）')
    parser.add_argument('--update-frequency', type=int, default=2048, help='更新频率')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
