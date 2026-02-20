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
from torch.amp import GradScaler, autocast

# 使用订阅模式提升数据收集速度 7-8x
from junction_agent import JUNCTION_CONFIGS, JunctionAgent, MultiAgentEnvironment, SubscriptionManager
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
    batch_size: int = 2048  # 增大到2048以充分利用GPU并行能力
    n_epochs: int = 5  # 减少epoch次数，5次通常足够
    update_frequency: int = 4096  # 增大更新频率，收集更多数据再更新
    
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
        """添加经验 - 使用pin_memory加速传输"""
        if junction_id not in self.states:
            self.states[junction_id] = []
            self.vehicle_states[junction_id] = []
            self.actions[junction_id] = []
            self.rewards[junction_id] = []
            self.values[junction_id] = []
            self.log_probs[junction_id] = []

        # 如果tensor在CPU上，使用pinned memory加速后续传输
        if state.device.type == 'cpu':
            state = state.pin_memory()

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

        # 调试：打印添加经验的信息
        import logging
        logger = logging.getLogger('buffer')
        logger.debug(f"添加经验: junction_id={junction_id}, state_shape={state.shape}, reward={reward}")
    
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
        """返回缓冲区总大小（所有junction的经验总数）"""
        if not self.states:
            return 0

        # 计算所有junction的经验总数
        total = sum(len(states) for states in self.states.values())
        return total


class MultiAgentPPOTrainer:
    """多智能体PPO训练器"""
    
    def __init__(self, model: MultiJunctionModel, config: PPOConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config or PPOConfig()
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        # 混合精度训练
        self.scaler = GradScaler() if device.startswith('cuda') else None

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

        MAX_VEHICLES = 300  # 最大车辆数
        features = []
        for veh_id in vehicle_ids[:MAX_VEHICLES]:
            try:
                # 使用全局traci（订阅模式）
                import traci
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
            except Exception as e:
                print(f"获取车辆 {veh_id} 特征失败: {e}")
                continue

        if not features:
            return None

        # 填充到MAX_VEHICLES
        while len(features) < MAX_VEHICLES:
            features.append([0.0] * 8)

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
    
    def compute_gae(self):
        """计算GAE - 完全GPU版本，最小化CPU-GPU通信"""
        # 预计算总样本数，预分配tensor
        junction_ids = list(self.buffer.states.keys())
        n_junctions = len(junction_ids)

        # 第一步：计算每个junction的GAE（完全在GPU上）
        all_returns_list = []
        all_advantages_list = []
        all_states_list = []
        all_log_probs_list = []
        junction_indices_list = []
        all_vehicle_obs_list = []  # 新增：预加载vehicle observations
        total_samples = 0

        for junc_idx, junc_id in enumerate(junction_ids):
            n_samples = len(self.buffer.rewards[junc_id])
            total_samples += n_samples

            # 直接在GPU上创建tensor（buffer中的数据已经在GPU上）
            rewards = torch.tensor(self.buffer.rewards[junc_id], dtype=torch.float32, device=self.device)
            values = torch.tensor(self.buffer.values[junc_id], dtype=torch.float32, device=self.device)
            dones = torch.tensor(self.buffer.dones[:n_samples], dtype=torch.float32, device=self.device)

            # 向量化GAE计算（完全在GPU上）
            # 计算TD residuals
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]

            deltas = rewards + self.config.gamma * next_values * (1 - dones) - values

            # 使用cumsum进行反向累积（GAE）
            # 反转序列
            deltas_reversed = torch.flip(deltas, [0])
            dones_reversed = torch.flip(1 - dones, [0])

            # 计算累积衰减系数
            gamma_lambda = self.config.gamma * self.config.gae_lambda
            decay_powers = torch.arange(n_samples, device=self.device)
            decay_factors = (gamma_lambda * dones_reversed).pow(decay_powers)

            # 加权累积和
            weighted_deltas = deltas_reversed * decay_factors
            advantages_reversed = torch.cumsum(weighted_deltas, dim=0)

            # 反转回来
            advantages = torch.flip(advantages_reversed, [0])
            returns = advantages + values

            all_returns_list.append(returns)
            all_advantages_list.append(advantages)
            all_states_list.extend(self.buffer.states[junc_id])  # 这些已经在GPU上
            all_log_probs_list.extend(self.buffer.log_probs[junc_id])
            junction_indices_list.extend([junc_idx] * n_samples)

            # 预加载vehicle observations到GPU
            junc_vehicle_obs = []
            for vehicle_obs_dict in self.buffer.vehicle_states[junc_id]:
                # 将每个vehicle_obs_dict转为GPU tensor
                gpu_dict = {}
                for key, value in vehicle_obs_dict.items():
                    if value is not None and torch.is_tensor(value):
                        gpu_dict[key] = value.to(self.device)
                junc_vehicle_obs.append(gpu_dict)
            all_vehicle_obs_list.append(junc_vehicle_obs)

        # 第二步：合并所有junction的数据（一次性的cat操作）
        all_returns = torch.cat(all_returns_list, dim=0)
        all_advantages = torch.cat(all_advantages_list, dim=0)

        # 标准化优势（原地操作）
        advantages_mean = all_advantages.mean()
        advantages_std = all_advantages.std()
        all_advantages.sub_(advantages_mean).div_(advantages_std + 1e-8)

        # Stack states（已经在GPU上）
        all_states = torch.stack(all_states_list)

        # log_probs转为tensor
        all_log_probs = torch.tensor(all_log_probs_list, dtype=torch.float32, device=self.device)

        # 创建junction索引tensor
        junction_indices = torch.tensor(junction_indices_list, dtype=torch.long, device=self.device)

        return all_states, all_returns, all_advantages, all_log_probs, junction_indices, junction_ids, all_vehicle_obs_list

    def update(self) -> Dict[str, float]:
        """更新模型 - 优化版本，使用统一批次避免嵌套循环"""
        self.model.train()

        total_samples = len(self.buffer)
        if total_samples < self.config.batch_size:
            return {'loss': 0.0}

        # 计算GAE（完全在GPU上，包括预加载vehicle observations）
        all_states, all_returns, all_advantages, all_log_probs, junction_indices, junction_ids, all_vehicle_obs = self.compute_gae()

        total_loss = 0.0
        n_batches = 0

        # 多个epoch的训练
        for epoch in range(self.config.n_epochs):
            # 在GPU上生成随机索引
            indices = torch.randperm(total_samples, device=self.device)

            # 批处理训练
            for start_idx in range(0, total_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]

                # 所有索引操作都在GPU上完成
                batch_states = all_states[batch_indices]
                batch_returns = all_returns[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_old_log_probs = all_log_probs[batch_indices]
                batch_junc_indices = junction_indices[batch_indices]

                # 按junction分组（使用GPU优化的masked_select）
                for junc_idx, junc_id in enumerate(junction_ids):
                    # 创建mask（在GPU上）
                    mask = (batch_junc_indices == junc_idx)

                    n_samples = mask.sum().item()
                    if n_samples == 0:
                        continue

                    # 使用masked_select（GPU优化操作，避免CPU-GPU传输）
                    junc_states = batch_states[mask]
                    junc_returns = batch_returns[mask]
                    junc_advantages = batch_advantages[mask]
                    junc_old_log_probs = batch_old_log_probs[mask]

                    # 准备vehicle observations（从预加载的GPU数据中获取）
                    # 首先获取该batch中属于当前junction的样本索引
                    selected_indices = batch_indices[mask]

                    # 计算该junction在总样本中的起始位置
                    start_pos = 0
                    for ji in range(junc_idx):
                        start_pos += len(self.buffer.states[junction_ids[ji]])

                    # 从预加载的vehicle_obs中提取对应的样本（直接在GPU上操作）
                    junc_vehicle_obs = {junc_id: {}}
                    for local_idx, global_idx in enumerate(selected_indices):
                        orig_idx = (global_idx - start_pos).item()
                        if orig_idx >= 0 and orig_idx < len(all_vehicle_obs[junc_idx]):
                            vo_dict = all_vehicle_obs[junc_idx][orig_idx]
                            for key, value in vo_dict.items():
                                if value is not None:
                                    if key not in junc_vehicle_obs[junc_id]:
                                        junc_vehicle_obs[junc_id][key] = []
                                    junc_vehicle_obs[junc_id][key].append(value)

                    # Stack为tensor（一次性，在GPU上）
                    for key in junc_vehicle_obs[junc_id]:
                        if junc_vehicle_obs[junc_id][key]:
                            junc_vehicle_obs[junc_id][key] = torch.stack(junc_vehicle_obs[junc_id][key])

                    # 前向传播
                    obs_dict = {junc_id: junc_states}
                    new_actions, new_values, new_info = self.model(obs_dict, junc_vehicle_obs, deterministic=False)

                    # 计算log_prob
                    new_log_prob = self._compute_log_prob_batch(new_info.get(junc_id, {}), new_actions.get(junc_id, {}))

                    # 形状对齐
                    min_size = min(new_log_prob.size(0), junc_old_log_probs.size(0))
                    if min_size > 0:
                        new_log_prob = new_log_prob[:min_size]
                        junc_old_log_probs = junc_old_log_probs[:min_size]
                        junc_advantages = junc_advantages[:min_size]
                        junc_returns = junc_returns[:min_size]

                        # PPO损失（所有操作在GPU上）
                        ratio = torch.exp(new_log_prob - junc_old_log_probs)
                        surr1 = ratio * junc_advantages
                        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * junc_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # 价值损失
                        value_pred = new_values.get(junc_id)
                        if value_pred is None:
                            # 如果模型没有返回该路口的价值预测，使用零值
                            value_pred = torch.zeros(junc_returns.size(0), 1, device=self.device)

                        # 确保 value_pred 的形状正确，并匹配 min_size
                        if value_pred.dim() > 1:
                            value_pred = value_pred.squeeze(-1)
                        if value_pred.size(0) != min_size:
                            value_pred = value_pred[:min_size]

                        value_loss = F.mse_loss(value_pred, junc_returns)

                        # 熵损失
                        entropy = self._compute_entropy(new_info.get(junc_id, {}))
                        entropy_loss = -entropy

                        # 总损失
                        loss = policy_loss + self.config.value_coef * value_loss + self.entropy_coef * entropy_loss

                        # 反向传播 - 使用混合精度训练
                        self.optimizer.zero_grad()

                        if self.scaler is not None:
                            # CUDA混合精度
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # CPU或标准精度
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            self.optimizer.step()

                        total_loss += loss.item()
                        n_batches += 1

        # 清空缓冲区
        self.buffer.clear()

        # 更新熵系数
        self.entropy_coef = max(self.entropy_coef * self.config.entropy_decay, self.config.entropy_min)

        return {
            'loss': total_loss / max(n_batches, 1),
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
        """
        计算回合OCR（符合官方评测公式）

        官方公式:
        OCR = (N_arrived + Σ(d_i_traveled / d_i_total)) / N_total

        其中:
        - N_arrived: 已到达车辆数
        - d_i_traveled: 在途车辆i已行驶的距离
        - d_i_total: 在途车辆i的OD路径总长度
        - N_total: 总车辆数（已到达 + 在途）
        """
        try:
            # 使用全局traci（订阅模式）
            import traci

            # 1. 获取到达车辆数
            n_arrived = traci.simulation.getArrivedNumber()

            # 2. 计算在途车辆的完成度
            enroute_completion = 0.0
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
                                # 如果边不存在，尝试获取车道长度
                                try:
                                    lane_id = f"{edge}_0"
                                    edge_length = traci.lane.getLength(lane_id)
                                    traveled_distance += edge_length
                                except:
                                    # 如果还是失败，使用默认值100m
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

                except Exception as e:
                    print(f"获取车辆 {veh_id} 路径完成度失败: {e}")
                    continue

            # 3. 总车辆数 = 已到达 + 在途
            n_total = n_arrived + len(traci.vehicle.getIDList())

            if n_total == 0:
                return 0.0

            # 4. OCR = (已到达 + 在途车辆完成度之和) / 总车辆数
            ocr = (n_arrived + enroute_completion) / n_total

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
