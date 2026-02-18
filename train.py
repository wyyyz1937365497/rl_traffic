"""
完整训练脚本
整合高级模型和训练技术
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_default_config
from environment import TrafficEnvironment
from advanced_model import (
    AdvancedTrafficModel, 
    PrioritizedReplayBuffer,
    CurriculumScheduler,
    create_advanced_model
)


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = create_advanced_model(config.network).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.ppo.actor_lr,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2
        )
        
        # 经验回放
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        
        # 课程学习
        self.curriculum = CurriculumScheduler(config.training.total_timesteps)
        
        # 统计
        self.stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_ocrs': deque(maxlen=100),
            'losses': deque(maxlen=1000)
        }
        
        # TensorBoard
        self.writer = None
        
        # 熵系数
        self.entropy_coef = config.ppo.entropy_coef
    
    def collect_experience(self, env: TrafficEnvironment, num_steps: int) -> Dict:
        """收集经验"""
        self.model.eval()
        
        obs = env.reset()
        history = []
        
        experiences = []
        total_reward = 0
        total_ocr = 0
        
        for step in range(num_steps):
            # 获取动作
            with torch.no_grad():
                action_dict, value, log_prob = self.model(obs, history, deterministic=False)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action_dict)
            
            # 存储经验
            experience = {
                'observation': obs,
                'action': action_dict,
                'reward': reward,
                'value': value.item(),
                'log_prob': log_prob.item(),
                'done': done,
                'history': history.copy(),
                'info': info
            }
            
            # 计算TD误差作为优先级
            with torch.no_grad():
                _, next_value, _ = self.model(next_obs, history, deterministic=True)
                td_error = abs(reward + self.config.ppo.gamma * next_value.item() * (1 - done) - value.item())
            
            self.replay_buffer.push(experience, td_error + 1e-6)
            experiences.append(experience)
            
            # 更新历史
            history.append(obs)
            if len(history) > self.config.env.history_length:
                history.pop(0)
            
            total_reward += reward
            total_ocr += info.get('ocr', 0)
            
            obs = next_obs
            
            if done:
                break
        
        return {
            'experiences': experiences,
            'total_reward': total_reward,
            'mean_ocr': total_ocr / max(len(experiences), 1)
        }
    
    def update(self, batch_size: int = 64, n_epochs: int = 10) -> Dict:
        """更新模型"""
        self.model.train()
        
        if len(self.replay_buffer.buffer) < batch_size:
            return {'loss': 0}
        
        total_loss = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            # 采样
            samples, indices, weights = self.replay_buffer.sample(batch_size)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            
            # 计算损失
            losses = []
            new_priorities = []
            
            for i, exp in enumerate(samples):
                obs = exp['observation']
                action = exp['action']
                reward = exp['reward']
                old_value = exp['value']
                old_log_prob = exp['log_prob']
                done = exp['done']
                history = exp['history']
                
                # 前向传播
                action_dict, value, log_prob = self.model(obs, history, deterministic=False)
                
                # 计算优势
                with torch.no_grad():
                    _, next_value, _ = self.model(obs, history, deterministic=True)
                    advantage = reward + self.config.ppo.gamma * next_value.item() * (1 - done) - old_value
                    return_value = advantage + old_value
                
                # 价值损失
                value_loss = (value - return_value) ** 2
                
                # 策略损失
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config.ppo.clip_epsilon, 
                                   1 + self.config.ppo.clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2)
                
                # 熵奖励
                entropy = -log_prob * torch.exp(log_prob)
                
                # 总损失
                loss = (policy_loss + 
                       self.config.ppo.value_coef * value_loss -
                       self.entropy_coef * entropy)
                
                losses.append(loss)
                new_priorities.append(abs(advantage) + 1e-6)
            
            # 聚合损失
            batch_loss = torch.stack([l.mean() for l in losses])
            batch_loss = (batch_loss * weights).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.ppo.max_grad_norm)
            self.optimizer.step()
            
            # 更新优先级
            self.replay_buffer.update_priorities(indices, np.array(new_priorities))
            
            total_loss += batch_loss.item()
            n_updates += 1
        
        # 更新学习率
        self.lr_scheduler.step()
        
        # 更新熵系数
        self.entropy_coef = max(self.entropy_coef * self.config.ppo.entropy_decay,
                               self.config.ppo.entropy_min)
        
        return {
            'loss': total_loss / n_updates,
            'entropy_coef': self.entropy_coef,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def train(self, env: TrafficEnvironment, eval_env: TrafficEnvironment = None,
              save_dir: str = 'checkpoints', log_dir: str = 'logs') -> Dict:
        """训练"""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        total_timesteps = self.config.training.total_timesteps
        update_freq = self.config.ppo.update_frequency
        
        timesteps = 0
        best_ocr = 0
        no_improve = 0
        
        history = {
            'timesteps': [],
            'rewards': [],
            'ocrs': [],
            'losses': [],
            'eval_ocrs': []
        }
        
        print(f"\n开始训练...")
        print(f"总步数: {total_timesteps}")
        print(f"更新频率: {update_freq}")
        
        while timesteps < total_timesteps:
            # 获取课程配置
            curriculum_config = self.curriculum.get_config()
            
            # 收集经验
            result = self.collect_experience(env, update_freq)
            timesteps += len(result['experiences'])
            self.curriculum.step()
            
            # 更新模型
            update_result = self.update()
            
            # 记录
            self.stats['episode_rewards'].append(result['total_reward'])
            self.stats['episode_ocrs'].append(result['mean_ocr'])
            self.stats['losses'].append(update_result['loss'])
            
            history['timesteps'].append(timesteps)
            history['rewards'].append(result['total_reward'])
            history['ocrs'].append(result['mean_ocr'])
            history['losses'].append(update_result['loss'])
            
            # TensorBoard
            self.writer.add_scalar('train/reward', result['total_reward'], timesteps)
            self.writer.add_scalar('train/ocr', result['mean_ocr'], timesteps)
            self.writer.add_scalar('train/loss', update_result['loss'], timesteps)
            self.writer.add_scalar('train/entropy_coef', update_result['entropy_coef'], timesteps)
            self.writer.add_scalar('train/lr', update_result['lr'], timesteps)
            self.writer.add_scalar('curriculum/traffic_scale', curriculum_config['traffic_scale'], timesteps)
            self.writer.add_scalar('curriculum/cv_ratio', curriculum_config['cv_ratio'], timesteps)
            
            # 打印进度
            if timesteps % (update_freq * 10) == 0:
                mean_reward = np.mean(list(self.stats['episode_rewards']))
                mean_ocr = np.mean(list(self.stats['episode_ocrs']))
                mean_loss = np.mean(list(self.stats['losses']))
                
                print(f"\n步数: {timesteps}/{total_timesteps}")
                print(f"  阶段: {curriculum_config['stage_name']}")
                print(f"  平均奖励: {mean_reward:.4f}")
                print(f"  平均OCR: {mean_ocr:.4f}")
                print(f"  平均损失: {mean_loss:.4f}")
            
            # 评估
            if eval_env and timesteps % self.config.training.eval_frequency == 0:
                eval_ocr = self.evaluate(eval_env)
                history['eval_ocrs'].append(eval_ocr)
                
                self.writer.add_scalar('eval/ocr', eval_ocr, timesteps)
                
                print(f"\n评估OCR: {eval_ocr:.4f}")
                
                if eval_ocr > best_ocr:
                    best_ocr = eval_ocr
                    no_improve = 0
                    self.save(os.path.join(save_dir, 'best_model.pt'))
                    print(f"保存最佳模型: OCR = {best_ocr:.4f}")
                else:
                    no_improve += 1
                
                if no_improve >= self.config.training.early_stop_patience:
                    print(f"\n早停: {no_improve} 次无改善")
                    break
            
            # 定期保存
            if timesteps % self.config.training.save_frequency == 0:
                self.save(os.path.join(save_dir, f'model_{timesteps}.pt'))
        
        # 保存最终模型
        self.save(os.path.join(save_dir, 'final_model.pt'))
        
        print(f"\n训练完成!")
        print(f"最佳OCR: {best_ocr:.4f}")
        
        self.writer.close()
        
        return history
    
    def evaluate(self, env: TrafficEnvironment, n_episodes: int = 5) -> float:
        """评估"""
        self.model.eval()
        
        total_ocr = 0
        
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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'entropy_coef': self.entropy_coef,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.entropy_coef = checkpoint.get('entropy_coef', self.config.ppo.entropy_coef)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级强化学习交通控制训练')
    parser.add_argument('--max-steps', type=int, default=3600, help='每个回合最大步数')
    parser.add_argument('--total-timesteps', type=int, default=10000000, help='总训练步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gui', action='store_true', help='使用GUI')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    
    args = parser.parse_args()
    
    # 配置
    config = get_default_config()
    config.env.max_steps = args.max_steps
    config.training.total_timesteps = args.total_timesteps
    config.training.seed = args.seed
    
    print("=" * 70)
    print("高级强化学习交通控制训练")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  仿真步数: {config.env.max_steps}")
    print(f"  训练步数: {config.training.total_timesteps}")
    print(f"  随机种子: {config.training.seed}")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建环境
    env = TrafficEnvironment(config.env, use_gui=args.gui, seed=args.seed)
    eval_env = TrafficEnvironment(config.env, use_gui=False, seed=args.seed + 1000)
    
    # 创建训练器
    trainer = AdvancedTrainer(config)
    
    # 训练
    save_dir = os.path.join('/home/z/my-project/rl_traffic', args.save_dir)
    log_dir = os.path.join('/home/z/my-project/rl_traffic', args.log_dir)
    
    history = trainer.train(env, eval_env, save_dir, log_dir)
    
    # 保存历史
    history_path = os.path.join('/home/z/my-project/rl_traffic', 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                       for x in v] for k, v in history.items()}, f, indent=2)
    
    print(f"\n训练历史已保存: {history_path}")
    
    # 关闭环境
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
