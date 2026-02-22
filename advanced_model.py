"""
高级强化学习模型
包含更先进的技术：
1. 多智能体注意力机制
2. 课程学习
3. 奖励重塑
4. 经验优先级回放
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from config import NetworkConfig


class SpatialAttention(nn.Module):
    """空间注意力模块 - 捕捉车辆间的空间关系"""
    
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_vehicles, feature_dim]
            mask: [batch_size, num_vehicles] 有效车辆掩码
        
        Returns:
            [batch_size, num_vehicles, feature_dim]
        """
        batch_size, num_vehicles, feature_dim = x.shape
        
        # 投影
        q = self.q_proj(x).view(batch_size, num_vehicles, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_vehicles, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_vehicles, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_vehicles]
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_vehicles, feature_dim)
        
        return self.out_proj(out)


class TemporalAttention(nn.Module):
    """时序注意力模块 - 捕捉时间演化模式"""
    
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len]
        
        Returns:
            [batch_size, seq_len, feature_dim]
        """
        # 自注意力
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # 残差连接
        out = self.norm(x + self.dropout(attn_out))
        
        return out


class VehicleEncoder(nn.Module):
    """车辆特征编码器 - 使用残差连接"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(2)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_vehicles, input_dim]
        
        Returns:
            [batch_size, num_vehicles, output_dim]
        """
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = F.relu(x)
        
        return self.output_proj(x)


class EdgeEncoder(nn.Module):
    """道路边特征编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GlobalContextEncoder(nn.Module):
    """全局上下文编码器"""
    
    def __init__(self, vehicle_dim: int, edge_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.vehicle_aggregator = nn.Sequential(
            nn.Linear(vehicle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_aggregator = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 10, output_dim),  # +10 for global features
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, vehicle_features: torch.Tensor, edge_features: torch.Tensor,
                global_features: torch.Tensor, vehicle_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            vehicle_features: [batch_size, num_vehicles, vehicle_dim]
            edge_features: [batch_size, num_edges, edge_dim]
            global_features: [batch_size, 10]
            vehicle_mask: [batch_size, num_vehicles]
        
        Returns:
            [batch_size, output_dim]
        """
        # 聚合车辆特征
        if vehicle_mask is not None:
            vehicle_mask = vehicle_mask.unsqueeze(-1)
            vehicle_features = vehicle_features * vehicle_mask
            vehicle_agg = vehicle_features.sum(dim=1) / (vehicle_mask.sum(dim=1) + 1e-8)
        else:
            vehicle_agg = vehicle_features.mean(dim=1)
        
        vehicle_context = self.vehicle_aggregator(vehicle_agg)
        
        # 聚合道路特征
        edge_agg = edge_features.mean(dim=1)
        edge_context = self.edge_aggregator(edge_agg)
        
        # 合并
        combined = torch.cat([vehicle_context, edge_context, global_features], dim=-1)
        
        return self.combiner(combined)


class MultiAgentCoordinator(nn.Module):
    """多智能体协调模块"""
    
    def __init__(self, feature_dim: int, num_agents: int = 10, dropout: float = 0.1):
        super().__init__()
        
        self.num_agents = num_agents
        
        # 智能体间通信
        self.communication = nn.MultiheadAttention(feature_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # 协调网络
        self.coordination = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 优先级预测
        self.priority_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, agent_features: torch.Tensor, global_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_features: [batch_size, num_agents, feature_dim]
            global_context: [batch_size, feature_dim]
        
        Returns:
            coordinated_features: [batch_size, num_agents, feature_dim]
            priorities: [batch_size, num_agents]
        """
        # 智能体间通信
        comm_out, _ = self.communication(agent_features, agent_features, agent_features)
        
        # 与全局上下文协调
        global_expanded = global_context.unsqueeze(1).expand(-1, agent_features.size(1), -1)
        coord_input = torch.cat([comm_out, global_expanded], dim=-1)
        coordinated = self.coordination(coord_input)
        
        # 计算优先级
        priorities = self.priority_net(coordinated).squeeze(-1)
        
        return coordinated, priorities


class AdvancedActorNetwork(nn.Module):
    """高级Actor网络"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(config.transformer_hidden_dim + config.gnn_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 动作头
        self.action_head = nn.Linear(128, config.speed_action_bins)
        
        # 不确定性估计
        self.uncertainty_head = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, feature_dim]
        
        Returns:
            action_probs: [batch_size, action_bins]
            action_values: [batch_size]
            uncertainty: [batch_size]
        """
        features = self.shared(x)
        
        # 动作概率
        logits = self.action_head(features)
        action_probs = F.softmax(logits, dim=-1)
        
        # 期望动作值
        action_values = torch.sum(
            action_probs * torch.linspace(0, 1, self.config.speed_action_bins, device=x.device),
            dim=-1
        )
        
        # 不确定性
        uncertainty = torch.sigmoid(self.uncertainty_head(features)).squeeze(-1)
        
        return action_probs, action_values, uncertainty


class AdvancedCriticNetwork(nn.Module):
    """高级Critic网络"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        # 状态价值流
        self.state_value = nn.Sequential(
            nn.Linear(config.transformer_hidden_dim + 50, 256),  # +50 for global features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 优势流
        self.advantage = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state_features: torch.Tensor, agent_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_features: [batch_size, state_dim]
            agent_features: [batch_size, agent_dim]
        
        Returns:
            value: [batch_size, 1]
        """
        state_value = self.state_value(state_features)
        advantage = self.advantage(agent_features)
        
        return state_value + advantage


class AdvancedTrafficModel(nn.Module):
    """
    高级交通控制模型
    整合所有先进技术
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 编码器
        self.vehicle_encoder = VehicleEncoder(15, 128, config.vehicle_feature_dim)
        self.edge_encoder = EdgeEncoder(10, 64, config.edge_feature_dim)
        
        # 注意力模块
        self.spatial_attention = SpatialAttention(config.vehicle_feature_dim)
        self.temporal_attention = TemporalAttention(config.vehicle_feature_dim)
        
        # 全局上下文
        self.global_encoder = GlobalContextEncoder(
            config.vehicle_feature_dim,
            config.edge_feature_dim,
            128,
            config.transformer_hidden_dim
        )
        
        # 多智能体协调
        self.coordinator = MultiAgentCoordinator(config.vehicle_feature_dim)
        
        # Actor-Critic
        self.actor = AdvancedActorNetwork(config)
        self.critic = AdvancedCriticNetwork(config)
        
        # 特征融合
        self.feature_fusion = nn.Linear(
            config.vehicle_feature_dim + config.transformer_hidden_dim,
            config.gnn_hidden_dim + config.transformer_hidden_dim
        )
    
    def encode_observation(self, observation: Dict, history: List[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码观察"""
        device = next(self.parameters()).device
        
        # 编码车辆特征
        vehicle_features = torch.tensor(observation['vehicle_features'], dtype=torch.float32, device=device)
        vehicle_features = self.vehicle_encoder(vehicle_features)
        
        # 空间注意力
        vehicle_features = self.spatial_attention(vehicle_features.unsqueeze(0)).squeeze(0)
        
        # 时序注意力（如果有历史）
        if history and len(history) > 0:
            history_features = []
            for hist_obs in history:
                hist_feat = torch.tensor(hist_obs['vehicle_features'], dtype=torch.float32, device=device)
                hist_feat = self.vehicle_encoder(hist_feat)
                history_features.append(hist_feat)
            
            history_features = torch.stack(history_features, dim=0).unsqueeze(0)  # [1, seq_len, num_vehicles, feat_dim]
            temporal_out = self.temporal_attention(history_features.mean(dim=2))  # 聚合车辆维度
            temporal_features = temporal_out.squeeze(0).mean(dim=0)  # [feat_dim]
        else:
            temporal_features = torch.zeros(self.config.vehicle_feature_dim, device=device)
        
        # 编码道路特征
        edge_features = torch.tensor(observation['edge_features'], dtype=torch.float32, device=device)
        if edge_features.size(0) > 0:
            edge_features = self.edge_encoder(edge_features)
        
        # 全局上下文
        global_features = torch.tensor(observation['global_features'], dtype=torch.float32, device=device)
        global_context = self.global_encoder(
            vehicle_features.unsqueeze(0),
            edge_features.unsqueeze(0) if edge_features.size(0) > 0 else torch.zeros(1, 0, self.config.edge_feature_dim, device=device),
            global_features.unsqueeze(0)
        ).squeeze(0)
        
        return vehicle_features, global_context, temporal_features
    
    def forward(self, observation: Dict, history: List[Dict] = None,
                deterministic: bool = False) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """前向传播"""
        device = next(self.parameters()).device
        
        # 编码
        vehicle_features, global_context, temporal_features = self.encode_observation(observation, history)
        
        # 获取被控制的车辆
        controlled_vehicles = observation.get('controlled_vehicles', [])
        
        if len(controlled_vehicles) == 0:
            return {}, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # 多智能体协调
        agent_features = vehicle_features[:len(controlled_vehicles)].unsqueeze(0)
        coordinated_features, priorities = self.coordinator(agent_features, global_context)
        coordinated_features = coordinated_features.squeeze(0)
        
        # 生成动作
        action_dict = {}
        log_probs = []
        
        for i, veh_id in enumerate(controlled_vehicles):
            if i < coordinated_features.size(0):
                # 融合特征
                combined = torch.cat([coordinated_features[i], temporal_features], dim=-1)
                combined = self.feature_fusion(combined).unsqueeze(0)
                
                # Actor
                action_probs, action_value, uncertainty = self.actor(combined)
                action_probs = action_probs.squeeze(0)
                
                # 采样
                if deterministic:
                    action_idx = torch.argmax(action_probs)
                else:
                    dist = Categorical(action_probs)
                    action_idx = dist.sample()
                    log_probs.append(dist.log_prob(action_idx))
                
                # 转换为连续动作
                action_value = torch.linspace(0, 1, self.config.speed_action_bins, device=device)[action_idx]
                
                # 根据不确定性调整动作
                if uncertainty.item() > 0.5:
                    # 高不确定性时，倾向于保守策略
                    action_value = action_value * 0.8
                
                action_dict[veh_id] = action_value.item()
        
        # Critic
        state_features = torch.cat([global_context, temporal_features], dim=-1).unsqueeze(0)
        value = self.critic(state_features, coordinated_features.mean(dim=0, keepdim=True))
        
        # 总对数概率
        if log_probs:
            total_log_prob = torch.stack(log_probs).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=device)
        
        return action_dict, value.squeeze(), total_log_prob


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
    
    def push(self, experience: Dict, priority: float = None):
        """添加经验"""
        if len(self.buffer) >= self.capacity:
            idx = np.argmin(self.priorities)
            self.buffer[idx] = experience
            self.priorities[idx] = priority or self.max_priority
        else:
            self.buffer.append(experience)
            self.priorities.append(priority or self.max_priority)
        
        if priority and priority > self.max_priority:
            self.max_priority = priority
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """采样"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            if priority > self.max_priority:
                self.max_priority = priority


class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        
        # 定义课程阶段
        self.stages = [
            {'name': 'easy', 'traffic_scale': 0.5, 'cv_ratio': 0.5, 'end_step': 0.2},
            {'name': 'medium', 'traffic_scale': 0.75, 'cv_ratio': 0.35, 'end_step': 0.5},
            {'name': 'hard', 'traffic_scale': 1.0, 'cv_ratio': 0.25, 'end_step': 1.0}
        ]
    
    def step(self):
        """前进一步"""
        self.current_step += 1
    
    def get_config(self) -> Dict:
        """获取当前配置"""
        progress = self.current_step / self.total_steps
        
        for i, stage in enumerate(self.stages):
            if progress <= stage['end_step']:
                return {
                    'traffic_scale': stage['traffic_scale'],
                    'cv_ratio': stage['cv_ratio'],
                    'stage_name': stage['name']
                }
        
        return {
            'traffic_scale': 1.0,
            'cv_ratio': 0.25,
            'stage_name': 'hard'
        }


def create_advanced_model(config: NetworkConfig) -> AdvancedTrafficModel:
    """创建高级模型"""
    return AdvancedTrafficModel(config)
