"""
PPO (Proximal Policy Optimization) 训练算法
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import PPOConfig, NetworkConfig, Config
from environment import TrafficEnvironment
from network import TrafficControlModel, create_model


@dataclass
class RolloutBuffer:
    """经验回放缓冲区"""
    observations: List[Dict]
    actions: List[Dict]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    history_batch: List[List[Dict]]
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.history_batch.clear()


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, config: Config, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            config: 配置
            device: 设备
        """
        self.config = config
        self.device = device
        
        print(f"使用设备: {device}")
        
        # 创建模型
        self.model = create_model(config.network).to(device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(),
            lr=config.ppo.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.model.critic.parameters(),
            lr=config.ppo.critic_lr
        )
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(
            observations=[],
            actions=[],
            rewards=[],
            values=[],
            log_probs=[],
            dones=[],
            history_batch=[]
        )
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'episode_ocrs': deque(maxlen=100),
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
        
        # TensorBoard
        self.writer = None
        
        # 熵系数（动态调整）
        self.entropy_coef = config.ppo.entropy_coef
    
    def collect_rollouts(self, env: TrafficEnvironment, num_steps: int) -> Tuple[float, float, int]:
        """
        收集经验
        
        Args:
            env: 环境
            num_steps: 收集步数
        
        Returns:
            total_reward: 总奖励
            mean_ocr: 平均OCR
            episodes: 完成的回合数
        """
        self.model.eval()
        
        obs = env.reset()
        history = []
        
        total_reward = 0.0
        total_ocr = 0.0
        episodes = 0
        steps = 0
        
        while steps < num_steps:
            # 获取动作
            with torch.no_grad():
                action_dict, value, log_prob = self.model(obs, history, deterministic=False)
            
            # 存储到缓冲区
            self.buffer.observations.append(obs)
            self.buffer.actions.append(action_dict)
            self.buffer.values.append(value.item())
            self.buffer.log_probs.append(log_prob.item())
            self.buffer.history_batch.append(history.copy())
            
            # 执行动作
            next_obs, reward, done, info = env.step(action_dict)
            
            # 存储奖励和done
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            
            # 更新历史
            history.append(obs)
            if len(history) > self.config.env.history_length:
                history.pop(0)
            
            # 更新统计
            total_reward += reward
            total_ocr += info.get('ocr', 0)
            steps += 1
            
            obs = next_obs
            
            if done:
                self.training_stats['episode_rewards'].append(total_reward)
                self.training_stats['episode_ocrs'].append(info.get('ocr', 0))
                episodes += 1
                
                # 重置
                obs = env.reset()
                history = []
                total_reward = 0.0
        
        mean_ocr = total_ocr / max(steps, 1)
        
        return total_reward, mean_ocr, episodes
    
    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)
        
        Returns:
            returns: 回报
            advantages: 优势
        """
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        gamma = self.config.ppo.gamma
        gae_lambda = self.config.ppo.gae_lambda
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return torch.tensor(returns, dtype=torch.float32, device=self.device), \
               torch.tensor(advantages, dtype=torch.float32, device=self.device)
    
    def update(self) -> Dict[str, float]:
        """
        更新模型
        
        Returns:
            训练统计
        """
        self.model.train()
        
        # 计算GAE
        returns, advantages = self.compute_gae()
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备数据
        observations = self.buffer.observations
        actions = self.buffer.actions
        history_batch = self.buffer.history_batch
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        n_epochs = self.config.ppo.n_epochs
        batch_size = self.config.ppo.batch_size
        n_samples = len(observations)
        
        for epoch in range(n_epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_history = [history_batch[i] for i in batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 评估动作
                values, log_probs, entropies = self.model.evaluate_actions(
                    batch_obs, batch_actions, batch_history
                )
                
                # 策略损失（PPO Clip）
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.ppo.clip_epsilon, 
                                   1 + self.config.ppo.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 熵损失
                entropy_loss = -entropies.mean()
                
                # 总损失
                loss = (policy_loss + 
                       self.config.ppo.value_coef * value_loss +
                       self.entropy_coef * entropy_loss)
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.config.ppo.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.config.ppo.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 更新熵系数
        self.entropy_coef = max(self.entropy_coef * self.config.ppo.entropy_decay,
                               self.config.ppo.entropy_min)
        
        # 返回统计
        n_updates = n_epochs * (n_samples // batch_size + 1)
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'total_loss': total_loss / n_updates,
            'entropy_coef': self.entropy_coef
        }
    
    def train(self, env: TrafficEnvironment, total_timesteps: int,
              eval_env: Optional[TrafficEnvironment] = None,
              save_dir: str = 'checkpoints',
              log_dir: str = 'logs') -> Dict[str, List]:
        """
        训练模型
        
        Args:
            env: 训练环境
            total_timesteps: 总训练步数
            eval_env: 评估环境
            save_dir: 保存目录
            log_dir: 日志目录
        
        Returns:
            训练历史
        """
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 训练循环
        timesteps = 0
        episode = 0
        best_ocr = 0.0
        no_improve_count = 0
        
        history = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_ocrs': [],
            'policy_losses': [],
            'value_losses': [],
            'eval_ocrs': []
        }
        
        update_frequency = self.config.ppo.update_frequency
        
        print("开始训练...")
        print(f"总步数: {total_timesteps}")
        print(f"更新频率: {update_frequency}")
        
        while timesteps < total_timesteps:
            # 收集经验
            total_reward, mean_ocr, episodes = self.collect_rollouts(env, update_frequency)
            timesteps += update_frequency
            episode += episodes
            
            # 更新模型
            update_stats = self.update()
            
            # 记录统计
            self.training_stats['policy_losses'].append(update_stats['policy_loss'])
            self.training_stats['value_losses'].append(update_stats['value_loss'])
            
            # TensorBoard记录
            self.writer.add_scalar('train/policy_loss', update_stats['policy_loss'], timesteps)
            self.writer.add_scalar('train/value_loss', update_stats['value_loss'], timesteps)
            self.writer.add_scalar('train/entropy_coef', update_stats['entropy_coef'], timesteps)
            self.writer.add_scalar('train/mean_ocr', mean_ocr, timesteps)
            
            # 保存历史
            history['timesteps'].append(timesteps)
            history['episode_ocrs'].append(mean_ocr)
            history['policy_losses'].append(update_stats['policy_loss'])
            history['value_losses'].append(update_stats['value_loss'])
            
            # 打印进度
            if timesteps % (update_frequency * 10) == 0:
                print(f"\n步数: {timesteps}/{total_timesteps}")
                print(f"  平均OCR: {mean_ocr:.4f}")
                print(f"  策略损失: {update_stats['policy_loss']:.4f}")
                print(f"  价值损失: {update_stats['value_loss']:.4f}")
            
            # 评估
            if eval_env and timesteps % self.config.training.eval_frequency == 0:
                eval_ocr = self.evaluate(eval_env, self.config.training.eval_episodes)
                history['eval_ocrs'].append(eval_ocr)
                
                self.writer.add_scalar('eval/ocr', eval_ocr, timesteps)
                
                print(f"\n评估OCR: {eval_ocr:.4f}")
                
                # 保存最佳模型
                if eval_ocr > best_ocr:
                    best_ocr = eval_ocr
                    no_improve_count = 0
                    self.save(os.path.join(save_dir, 'best_model.pt'))
                    print(f"保存最佳模型，OCR: {best_ocr:.4f}")
                else:
                    no_improve_count += 1
                
                # 早停
                if no_improve_count >= self.config.training.early_stop_patience:
                    print(f"\n早停: {no_improve_count} 次评估无改善")
                    break
            
            # 定期保存
            if timesteps % self.config.training.save_frequency == 0:
                self.save(os.path.join(save_dir, f'model_{timesteps}.pt'))
        
        # 保存最终模型
        self.save(os.path.join(save_dir, 'final_model.pt'))
        
        print("\n训练完成!")
        print(f"最佳OCR: {best_ocr:.4f}")
        
        self.writer.close()
        
        return history
    
    def evaluate(self, env: TrafficEnvironment, n_episodes: int = 5) -> float:
        """
        评估模型
        
        Args:
            env: 环境
            n_episodes: 评估回合数
        
        Returns:
            平均OCR
        """
        self.model.eval()
        
        total_ocr = 0.0
        
        for _ in range(n_episodes):
            obs = env.reset()
            history = []
            done = False
            
            while not done:
                with torch.no_grad():
                    action_dict, _, _ = self.model(obs, history, deterministic=True)
                
                obs, _, done, info = env.step(action_dict)
                
                history.append(obs)
                if len(history) > self.config.env.history_length:
                    history.pop(0)
            
            total_ocr += info.get('ocr', 0)
        
        return total_ocr / n_episodes
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'entropy_coef': self.entropy_coef
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.entropy_coef = checkpoint.get('entropy_coef', self.config.ppo.entropy_coef)


# 导入F用于损失计算
import torch.nn.functional as F


def train_model(config: Config = None, use_gui: bool = False):
    """
    训练模型入口函数
    
    Args:
        config: 配置
        use_gui: 是否使用GUI
    """
    if config is None:
        from config import get_default_config
        config = get_default_config()
    
    # 创建环境
    env = TrafficEnvironment(config.env, use_gui=use_gui, seed=config.training.seed)
    eval_env = TrafficEnvironment(config.env, use_gui=False, seed=config.training.seed + 1000)
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 训练
    save_dir = os.path.join(' sumo', config.training.save_dir)
    log_dir = os.path.join(' sumo', config.training.log_dir)
    
    history = trainer.train(
        env,
        total_timesteps=config.training.total_timesteps,
        eval_env=eval_env,
        save_dir=save_dir,
        log_dir=log_dir
    )
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return trainer, history


if __name__ == '__main__':
    train_model()
