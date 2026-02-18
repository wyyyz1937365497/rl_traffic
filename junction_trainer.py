"""
多智能体PPO训练器
针对路口级多智能体系统设计
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from junction_agent import JUNCTION_CONFIGS, JunctionAgent, MultiAgentEnvironment
from junction_network import create_junction_model, NetworkConfig, MultiJunctionModel


@dataclass
class PPOConfig:
    """PPO配置"""
    # 学习率
    lr: float = 3e-4
    
    # PPO参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 训练参数
    batch_size: int = 64
    n_epochs: int = 10
    update_frequency: int = 1024
    
    # 探索
    entropy_decay: float = 0.999
    entropy_min: float = 0.001


@dataclass
class ExperienceBuffer:
    """经验缓冲区"""
    states: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    vehicle_states: Dict[str, List[Dict]] = field(default_factory=dict)
    actions: Dict[str, List[Dict]] = field(default_factory=dict)
    rewards: Dict[str, List[float]] = field(default_factory=dict)
    values: Dict[str, List[float]] = field(default_factory=dict)
    log_probs: Dict[str, List[float]] = field(default_factory=dict)
    dones: List[bool] = field(default_factory=list)
    
    def add(self, junction_id: str, state: torch.Tensor, vehicle_state: Dict,
            action: Dict, reward: float, value: float, log_prob: float, done: bool):
        """添加经验"""
        if junction_id not in self.states:
            self.states[junction_id] = []
            self.vehicle_states[junction_id] = []
            self.actions[junction_id] = []
            self.rewards[junction_id] = []
            self.values[junction_id] = []
            self.log_probs[junction_id] = []
        
        self.states[junction_id].append(state)
        self.vehicle_states[junction_id].append(vehicle_state)
        self.actions[junction_id].append(action)
        self.rewards[junction_id].append(reward)
        self.values[junction_id].append(value)
        self.log_probs[junction_id].append(log_prob)
        
        if len(self.dones) < len(self.states[junction_id]):
            self.dones.append(done)
        else:
            self.dones[-1] = done
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.vehicle_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        if not self.states:
            return 0
        return len(next(iter(self.states.values())))


class MultiAgentPPOTrainer:
    """多智能体PPO训练器"""
    
    def __init__(self, model: MultiJunctionModel, config: PPOConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config or PPOConfig()
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # 经验缓冲区
        self.buffer = ExperienceBuffer()
        
        # 统计
        self.stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_ocrs': deque(maxlen=100),
            'losses': deque(maxlen=1000)
        }
        
        # TensorBoard
        self.writer = None
        
        # 熵系数
        self.entropy_coef = self.config.entropy_coef
    
    def collect_experience(self, env: MultiAgentEnvironment, num_steps: int) -> Dict:
        """收集经验"""
        self.model.eval()
        
        obs = env.reset()
        history = {junc_id: [] for junc_id in env.agents.keys()}
        
        total_rewards = {junc_id: 0.0 for junc_id in env.agents.keys()}
        total_ocr = 0.0
        
        for step in range(num_steps):
            # 准备观察张量
            obs_tensors = {}
            vehicle_obs = {}
            
            for junc_id, agent in env.agents.items():
                state_vec = agent.get_state_vector()
                obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # 获取车辆观察
                controlled = agent.get_controlled_vehicles()
                vehicle_obs[junc_id] = {
                    'main': self._get_vehicle_features(controlled['main']) if controlled['main'] else None,
                    'ramp': self._get_vehicle_features(controlled['ramp']) if controlled['ramp'] else None,
                    'diverge': self._get_vehicle_features(controlled['diverge']) if controlled['diverge'] else None
                }
            
            # 获取动作
            with torch.no_grad():
                actions, values, info = self.model(obs_tensors, vehicle_obs, deterministic=False)
            
            # 转换动作格式
            action_dict = {}
            for junc_id, action in actions.items():
                action_dict[junc_id] = {}
                agent = env.agents[junc_id]
                controlled = agent.get_controlled_vehicles()
                
                # 主路动作
                if controlled['main'] and 'main' in action:
                    for i, veh_id in enumerate(controlled['main'][:1]):  # 只控制第一辆
                        action_dict[junc_id][veh_id] = action['main'].item() if torch.is_tensor(action['main']) else action['main']
                
                # 匝道动作
                if controlled['ramp'] and 'ramp' in action:
                    for i, veh_id in enumerate(controlled['ramp'][:1]):
                        action_dict[junc_id][veh_id] = action['ramp'].item() if torch.is_tensor(action['ramp']) else action['ramp']
            
            # 存储经验
            for junc_id in env.agents.keys():
                value = values.get(junc_id, torch.tensor(0.0))
                log_prob = self._compute_log_prob(info.get(junc_id, {}), actions.get(junc_id, {}))
                
                self.buffer.add(
                    junc_id,
                    obs_tensors[junc_id].squeeze(0),
                    vehicle_obs[junc_id],
                    actions.get(junc_id, {}),
                    0.0,  # 奖励稍后填充
                    value.item() if torch.is_tensor(value) else value,
                    log_prob,
                    False
                )
            
            # 执行动作
            next_obs, rewards, done, step_info = env.step(action_dict)
            
            # 更新奖励
            for junc_id, reward in rewards.items():
                if self.buffer.rewards[junc_id]:
                    self.buffer.rewards[junc_id][-1] = reward
                    total_rewards[junc_id] += reward
            
            total_ocr += step_info.get('global_stats', {}).get('total_ocr', 0)
            
            # 更新观察
            obs = next_obs
            
            if done:
                break
        
        mean_ocr = total_ocr / max(num_steps, 1)
        
        return {
            'total_rewards': total_rewards,
            'mean_ocr': mean_ocr,
            'steps': len(self.buffer.dones)
        }
    
    def _get_vehicle_features(self, vehicle_ids: List[str]) -> Optional[torch.Tensor]:
        """获取车辆特征张量"""
        if not vehicle_ids:
            return None

        features = []
        for veh_id in vehicle_ids[:10]:  # 最多10辆
            try:
                # 使用全局traci（已经在junction_agent中导入为libsumo或traci）
                from junction_agent import traci
                feat = [
                    traci.vehicle.getSpeed(veh_id) / 20.0,
                    traci.vehicle.getLanePosition(veh_id) / 500.0,
                    traci.vehicle.getLaneIndex(veh_id) / 3.0,
                    traci.vehicle.getWaitingTime(veh_id) / 60.0,
                    traci.vehicle.getAcceleration(veh_id) / 5.0,
                    1.0 if traci.vehicle.getTypeID(veh_id) == 'CV' else 0.0,
                    traci.vehicle.getRouteIndex(veh_id) / 10.0,
                    0.0  # 预留
                ]
                features.append(feat)
            except:
                continue

        if not features:
            return None

        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _compute_log_prob(self, info: Dict, actions: Dict) -> float:
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
    
    def compute_gae(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """计算GAE"""
        returns = {}
        advantages = {}
        
        for junc_id in self.buffer.states.keys():
            rewards = np.array(self.buffer.rewards[junc_id])
            values = np.array(self.buffer.values[junc_id])
            dones = np.array(self.buffer.dones[:len(rewards)])
            
            # 计算优势
            adv = np.zeros_like(rewards)
            last_gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
                adv[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            
            ret = adv + values
            
            returns[junc_id] = torch.tensor(ret, dtype=torch.float32, device=self.device)
            advantages[junc_id] = torch.tensor(adv, dtype=torch.float32, device=self.device)
        
        return returns, advantages
    
    def update(self) -> Dict[str, float]:
        """更新模型"""
        self.model.train()
        
        if len(self.buffer) < self.config.batch_size:
            return {'loss': 0.0}
        
        # 计算GAE
        returns, advantages = self.compute_gae()
        
        # 标准化优势
        for junc_id in advantages.keys():
            advantages[junc_id] = (advantages[junc_id] - advantages[junc_id].mean()) / (advantages[junc_id].std() + 1e-8)
        
        total_loss = 0.0
        n_updates = 0
        
        for epoch in range(self.config.n_epochs):
            # 遍历所有路口
            for junc_id in self.buffer.states.keys():
                states = self.buffer.states[junc_id]
                actions = self.buffer.actions[junc_id]
                old_log_probs = self.buffer.log_probs[junc_id]
                ret = returns[junc_id]
                adv = advantages[junc_id]
                
                # 批处理
                for start in range(0, len(states), self.config.batch_size):
                    end = min(start + self.config.batch_size, len(states))
                    
                    batch_states = torch.stack(states[start:end])
                    batch_returns = ret[start:end]
                    batch_advantages = adv[start:end]
                    batch_old_log_probs = torch.tensor(old_log_probs[start:end], dtype=torch.float32, device=self.device)
                    
                    # 前向传播
                    obs_dict = {junc_id: batch_states}
                    new_actions, new_values, new_info = self.model(obs_dict, deterministic=False)
                    
                    # 计算损失
                    # 策略损失
                    new_log_prob = self._compute_log_prob_batch(new_info.get(junc_id, {}), new_actions.get(junc_id, {}))
                    
                    ratio = torch.exp(new_log_prob - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值损失
                    value = new_values.get(junc_id, torch.zeros_like(batch_returns))
                    value_loss = F.mse_loss(value.squeeze(), batch_returns)
                    
                    # 熵损失
                    entropy = self._compute_entropy(new_info.get(junc_id, {}))
                    entropy_loss = -entropy
                    
                    # 总损失
                    loss = (policy_loss + 
                           self.config.value_coef * value_loss +
                           self.entropy_coef * entropy_loss)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    n_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 更新熵系数
        self.entropy_coef = max(self.entropy_coef * self.config.entropy_decay, self.config.entropy_min)
        
        return {
            'loss': total_loss / max(n_updates, 1),
            'entropy_coef': self.entropy_coef
        }
    
    def _compute_log_prob_batch(self, info: Dict, actions: Dict) -> torch.Tensor:
        """批量计算对数概率"""
        log_probs = []
        
        for key in ['main', 'ramp', 'diverge']:
            if f'{key}_probs' in info and key in actions:
                probs = info[f'{key}_probs']
                action = actions[key]
                
                if torch.is_tensor(probs) and torch.is_tensor(action):
                    batch_size = probs.size(0)
                    action_idx = (action * 10).long().clamp(0, probs.size(-1) - 1)
                    log_prob = torch.log(probs.gather(1, action_idx.unsqueeze(1)).squeeze(1) + 1e-8)
                    log_probs.append(log_prob)
        
        if log_probs:
            return torch.stack(log_probs).sum(dim=0)
        return torch.zeros(info.get('main_probs', torch.zeros(1)).size(0), device=self.device)
    
    def _compute_entropy(self, info: Dict) -> torch.Tensor:
        """计算熵"""
        entropies = []
        
        for key in ['main', 'ramp', 'diverge']:
            if f'{key}_probs' in info:
                probs = info[f'{key}_probs']
                if torch.is_tensor(probs):
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    entropies.append(entropy)
        
        if entropies:
            return torch.stack(entropies).mean()
        return torch.tensor(0.0, device=self.device)
    
    def train(self, env: MultiAgentEnvironment, total_timesteps: int,
              eval_env: MultiAgentEnvironment = None,
              save_dir: str = 'checkpoints',
              log_dir: str = 'logs') -> Dict:
        """训练"""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        timesteps = 0
        best_ocr = 0.0
        
        history = {
            'timesteps': [],
            'rewards': [],
            'ocrs': [],
            'losses': []
        }
        
        print(f"\n开始多智能体训练...")
        print(f"总步数: {total_timesteps}")
        print(f"路口数: {len(env.agents)}")
        
        while timesteps < total_timesteps:
            # 收集经验
            result = self.collect_experience(env, self.config.update_frequency)
            timesteps += result['steps']
            
            # 更新模型
            update_result = self.update()
            
            # 记录
            mean_reward = np.mean(list(result['total_rewards'].values()))
            self.stats['episode_rewards'].append(mean_reward)
            self.stats['episode_ocrs'].append(result['mean_ocr'])
            self.stats['losses'].append(update_result['loss'])
            
            history['timesteps'].append(timesteps)
            history['rewards'].append(mean_reward)
            history['ocrs'].append(result['mean_ocr'])
            history['losses'].append(update_result['loss'])
            
            # TensorBoard
            self.writer.add_scalar('train/reward', mean_reward, timesteps)
            self.writer.add_scalar('train/ocr', result['mean_ocr'], timesteps)
            self.writer.add_scalar('train/loss', update_result['loss'], timesteps)
            
            # 打印进度
            if timesteps % (self.config.update_frequency * 10) == 0:
                print(f"\n步数: {timesteps}/{total_timesteps}")
                print(f"  平均奖励: {mean_reward:.4f}")
                print(f"  平均OCR: {result['mean_ocr']:.4f}")
                print(f"  损失: {update_result['loss']:.4f}")
                
                # 打印各路口奖励
                for junc_id, reward in result['total_rewards'].items():
                    print(f"    {junc_id}: {reward:.4f}")
            
            # 评估
            if eval_env and timesteps % 100000 == 0:
                eval_ocr = self.evaluate(eval_env, 3)
                print(f"\n评估OCR: {eval_ocr:.4f}")
                
                if eval_ocr > best_ocr:
                    best_ocr = eval_ocr
                    self.save(os.path.join(save_dir, 'best_model.pt'))
        
        # 保存最终模型
        self.save(os.path.join(save_dir, 'final_model.pt'))
        
        print(f"\n训练完成!")
        print(f"最佳OCR: {best_ocr:.4f}")
        
        self.writer.close()
        
        return history
    
    def evaluate(self, env: MultiAgentEnvironment, n_episodes: int = 5) -> float:
        """评估"""
        self.model.eval()
        
        total_ocr = 0.0
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                # 准备观察
                obs_tensors = {}
                vehicle_obs = {}
                
                for junc_id, agent in env.agents.items():
                    state_vec = agent.get_state_vector()
                    obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    controlled = agent.get_controlled_vehicles()
                    vehicle_obs[junc_id] = {
                        'main': self._get_vehicle_features(controlled['main']),
                        'ramp': self._get_vehicle_features(controlled['ramp']),
                        'diverge': self._get_vehicle_features(controlled['diverge'])
                    }
                
                # 获取动作
                with torch.no_grad():
                    actions, _, _ = self.model(obs_tensors, vehicle_obs, deterministic=True)
                
                # 执行
                action_dict = {}
                for junc_id, action in actions.items():
                    action_dict[junc_id] = {}
                    agent = env.agents[junc_id]
                    controlled = agent.get_controlled_vehicles()
                    
                    if controlled['main'] and 'main' in action:
                        for veh_id in controlled['main'][:1]:
                            action_dict[junc_id][veh_id] = action['main'].item() if torch.is_tensor(action['main']) else action['main']
                    
                    if controlled['ramp'] and 'ramp' in action:
                        for veh_id in controlled['ramp'][:1]:
                            action_dict[junc_id][veh_id] = action['ramp'].item() if torch.is_tensor(action['ramp']) else action['ramp']
                
                obs, _, done, info = env.step(action_dict)
            
            # 计算OCR
            ocr = self._compute_episode_ocr(env)
            total_ocr += ocr
        
        return total_ocr / n_episodes
    
    def _compute_episode_ocr(self, env: MultiAgentEnvironment) -> float:
        """计算回合OCR（完整版）"""
        try:
            from junction_agent import traci

            # 1. 获取到达车辆数
            arrived = traci.simulation.getArrivedNumber()

            # 2. 获取出发车辆数
            departed = traci.simulation.getDepartedNumber()

            # 3. 计算在途车辆的完成度
            inroute_completion = 0.0
            total_vehicles = traci.vehicle.getCount()

            for veh_id in traci.vehicle.getIDList():
                try:
                    # 获取路径进度
                    route_idx = traci.vehicle.getRouteIndex(veh_id)
                    route = traci.vehicle.getRoute(veh_id)
                    route_len = len(route)

                    if route_len > 0:
                        # 完成度 = 当前路径索引 / 总路径长度
                        completion = route_idx / route_len
                        inroute_completion += completion
                except:
                    continue

            # 4. 计算OCR
            # OCR = (到达车辆数 + 在途车辆完成度之和) / 总出发车辆数
            if total_vehicles == 0 and arrived == 0:
                return 0.0

            ocr = (arrived + inroute_completion) / max(departed, 1)

            return min(ocr, 1.0)  # 限制在[0, 1]

        except Exception as e:
            print(f"OCR计算错误: {e}")
            return 0.0
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.entropy_coef = checkpoint.get('entropy_coef', self.config.entropy_coef)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多智能体路口控制训练')
    parser.add_argument('--sumo-cfg', type=str, required=True, help='SUMO配置文件')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("多智能体路口控制训练")
    print("=" * 70)
    
    # 创建环境
    env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=args.gui,
        seed=args.seed
    )
    
    eval_env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=False,
        seed=args.seed + 1000
    )
    
    # 创建模型
    model = create_junction_model(JUNCTION_CONFIGS)
    
    # 创建训练器
    trainer = MultiAgentPPOTrainer(model)
    
    # 训练
    save_dir = ' sumo/checkpoints_junction'
    log_dir = ' sumo/logs_junction'
    
    history = trainer.train(env, args.total_timesteps, eval_env, save_dir, log_dir)
    
    # 保存历史
    history_path = ' sumo/junction_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练历史已保存: {history_path}")
    
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
