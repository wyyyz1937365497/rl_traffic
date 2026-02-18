"""
多智能体路口交通控制系统 - 简化版
只支持CUDA训练 + 文件IO并行数据收集
"""

import os
import sys
import argparse
import json
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pickle
from multiprocessing import Process
import multiprocessing

from junction_agent import JUNCTION_CONFIGS
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


def create_libsumo_environment(sumo_cfg: str, seed: int = 42):
    """创建libsumo环境"""
    from junction_agent import JunctionAgent

    class Environment:
        def __init__(self, sumo_cfg: str, seed: int):
            self.sumo_cfg = sumo_cfg
            self.seed = seed
            self.agents = {}
            self.is_running = False
            self.current_step = 0

            for junc_id in JUNCTION_CONFIGS.keys():
                self.agents[junc_id] = JunctionAgent(JUNCTION_CONFIGS[junc_id])

        def reset(self):
            self._start_sumo()
            self.current_step = 0

            for agent in self.agents.values():
                agent.state_history.clear()

            for _ in range(10):
                traci_wrapper.simulationStep()
                self.current_step += 1

            return {junc_id: self.agents[junc_id].observe() for junc_id in self.agents.keys()}

        def step(self, actions):
            self._apply_actions(actions)
            traci_wrapper.simulationStep()
            self.current_step += 1

            observations = {junc_id: self.agents[junc_id].observe() for junc_id in self.agents.keys()}
            rewards = self._compute_rewards()
            done = self.current_step >= 3600

            return observations, rewards, done, {}

        def _start_sumo(self):
            if self.is_running:
                try:
                    traci_wrapper.close()
                except:
                    pass

            sumo_binary = "sumo"

            if USE_LIBSUMO:
                sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--no-warnings", "true", "--seed", str(self.seed)]
                traci_wrapper.start(sumo_cmd)
            else:
                sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--remote-port", "0", "--no-warnings", "true", "--seed", str(self.seed)]
                traci_wrapper.start(sumo_cmd)

            self.is_running = True

        def _apply_actions(self, actions):
            for junc_id, action_dict in actions.items():
                for veh_id, action in action_dict.items():
                    try:
                        speed_limit = 13.89
                        target_speed = speed_limit * (0.3 + 0.9 * action)
                        traci_wrapper.vehicle.setSpeed(veh_id, target_speed)
                    except:
                        continue

        def _compute_rewards(self):
            rewards = {}
            for junc_id, agent in self.agents.items():
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

            return rewards

        def close(self):
            if self.is_running:
                try:
                    traci_wrapper.close()
                except:
                    pass
                self.is_running = False

    return Environment(sumo_cfg, seed)


def worker_process(worker_id, sumo_cfg, output_dir, seed, model_state, use_cuda):
    """工作进程 - 文件IO版本"""
    try:
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

        # 创建环境
        env = create_libsumo_environment(sumo_cfg, seed)

        # 创建模型
        model = create_junction_model(JUNCTION_CONFIGS)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        # 收集经验 - 运行完整的3600步episode
        obs = env.reset()
        experiences = []
        total_rewards = {}

        # 运行完整的episode，直到环境done
        while True:
            # 准备观察
            obs_tensors = {}
            vehicle_obs = {}

            for junc_id, agent in env.agents.items():
                state_vec = agent.get_state_vector()
                obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

                controlled = agent.get_controlled_vehicles()
                vehicle_obs[junc_id] = {
                    'main': _get_vehicle_features(controlled['main'], device) if controlled['main'] else None,
                    'ramp': _get_vehicle_features(controlled['ramp'], device) if controlled['ramp'] else None,
                    'diverge': _get_vehicle_features(controlled['diverge'], device) if controlled['diverge'] else None
                }

            # 获取动作
            with torch.no_grad():
                actions, values, info = model(obs_tensors, vehicle_obs, deterministic=False)

            # 转换动作
            action_dict = {}
            for junc_id, action in actions.items():
                action_dict[junc_id] = {}
                controlled = env.agents[junc_id].get_controlled_vehicles()

                if controlled['main'] and 'main' in action:
                    for veh_id in controlled['main'][:1]:
                        action_dict[junc_id][veh_id] = action['main'].item()

                if controlled['ramp'] and 'ramp' in action:
                    for veh_id in controlled['ramp'][:1]:
                        action_dict[junc_id][veh_id] = action['ramp'].item()

            # 执行动作
            next_obs, rewards, done, info = env.step(action_dict)

            # 存储经验（现在可以获取reward了）
            for junc_id in env.agents.keys():
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

            obs = next_obs

            if done:
                break

        env.close()

        # 保存到文件
        output_file = os.path.join(output_dir, f'worker_{worker_id}.pkl')
        result_data = {
            'worker_id': worker_id,
            'experiences': experiences,
            'total_rewards': total_rewards,
            'steps': len(experiences)
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result_data, f)

        with open(os.path.join(output_dir, f'worker_{worker_id}.done'), 'w') as f:
            f.write('done')

    except Exception as e:
        import traceback
        with open(os.path.join(output_dir, f'worker_{worker_id}.error'), 'w') as f:
            f.write(f"Error: {str(e)}\n{traceback.format_exc()}")


def _get_vehicle_features(vehicle_ids, device):
    """获取车辆特征"""
    if not vehicle_ids:
        return None

    features = []
    for veh_id in vehicle_ids[:10]:
        try:
            features.append([
                traci_wrapper.vehicle.getSpeed(veh_id) / 20.0,
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

    # 训练循环
    timesteps = 0
    best_ocr = 0.0
    entropy_coef = ppo_config.entropy_coef

    print(f"\n开始训练...")

    try:
        while timesteps < args.total_timesteps:
            start_time = time.time()

            # 清空临时目录
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass

            # 启动工作进程
            processes = []
            for worker_id in range(num_workers):
                p = Process(
                    target=worker_process,
                    args=(worker_id, args.sumo_cfg, temp_dir, 42 + worker_id,
                          model.state_dict(),
                          torch.cuda.is_available() and torch.cuda.device_count() >= num_workers)
                )
                p.start()
                processes.append(p)

            # 等待完成
            for p in processes:
                p.join(timeout=600)
                if p.is_alive():
                    p.terminate()

            # 读取结果
            total_rewards = {}
            total_steps = 0

            for worker_id in range(num_workers):
                result_file = os.path.join(temp_dir, f'worker_{worker_id}.pkl')
                error_file = os.path.join(temp_dir, f'worker_{worker_id}.error')

                if os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        print(f"Worker {worker_id} 错误:\n{f.read()}")
                    continue

                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'rb') as f:
                            result_data = pickle.load(f)

                        for exp in result_data['experiences']:
                            state_tensor = torch.from_numpy(exp['state']).float().to(device)
                            vehicle_obs = {}
                            for k, v in exp['vehicle_obs'].items():
                                if isinstance(v, np.ndarray):
                                    vehicle_obs[k] = torch.from_numpy(v).float().to(device)
                                else:
                                    vehicle_obs[k] = v

                            buffer.add(
                                exp['junction_id'], state_tensor, vehicle_obs,
                                exp['action'], exp['reward'], exp['value'], exp['log_prob'], False
                            )

                        for junc_id, reward in result_data['total_rewards'].items():
                            if junc_id not in total_rewards:
                                total_rewards[junc_id] = 0.0
                            total_rewards[junc_id] += reward

                        total_steps += result_data['steps']
                        print(f"  Worker {worker_id}: {result_data['steps']} 步")

                    except Exception as e:
                        print(f"  Worker {worker_id} 读取失败: {e}")

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

            # 打印进度
            if timesteps % (args.update_frequency * 2) == 0 or timesteps == total_steps:
                print(f"\n步数: {timesteps}/{args.total_timesteps}")
                print(f"  平均奖励: {mean_reward:.4f}")
                print(f"  损失: {update_result['loss']:.4f}")
                print(f"  收集时间: {collect_time:.2f}s")
                print(f"  更新时间: {update_time:.2f}s")

                for junc_id, reward in total_rewards.items():
                    print(f"    {junc_id}: {reward:.4f}")

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
