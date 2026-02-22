"""
强化学习神经网络模型
包含GNN编码器、Transformer时序编码器和Actor-Critic网络
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from config import NetworkConfig


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.w_o(output)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        
        self.w_q = nn.Linear(in_features, out_features * num_heads)
        self.w_k = nn.Linear(in_features, out_features * num_heads)
        self.w_v = nn.Linear(in_features, out_features * num_heads)
        self.w_o = nn.Linear(out_features * num_heads, out_features)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feat_dim]
        """
        num_nodes = x.size(0)
        
        # 计算Q, K, V
        q = self.w_q(x).view(num_nodes, self.num_heads, self.out_features)
        k = self.w_k(x).view(num_nodes, self.num_heads, self.out_features)
        v = self.w_v(x).view(num_nodes, self.num_heads, self.out_features)
        
        # 如果没有边，返回变换后的特征
        if edge_index.size(1) == 0:
            return self.w_o(v.view(num_nodes, -1))
        
        # 计算注意力
        src, dst = edge_index[0], edge_index[1]
        
        # [num_edges, num_heads, out_features]
        q_edges = q[src]
        k_edges = k[dst]
        
        # 注意力分数
        scores = (q_edges * k_edges).sum(dim=-1) / math.sqrt(self.out_features)
        scores = self.leaky_relu(scores)
        
        # Softmax（按目标节点分组）
        # 修复：按目标节点分组计算softmax
        attn = self._softmax_by_target(scores, dst, num_nodes)
        attn = self.dropout(attn)
        
        # 聚合
        v_edges = v[src]  # [num_edges, num_heads, out_features]
        attn = attn.unsqueeze(-1)  # [num_edges, num_heads, 1]
        
        # 聚合到目标节点
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        out.index_add_(0, dst, v_edges * attn)
        
        # 合并多头
        out = out.view(num_nodes, -1)
        out = self.w_o(out)
        
        return out


class GNNEncoder(nn.Module):
    """图神经网络编码器"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 输入投影
        self.input_proj = nn.Linear(config.vehicle_feature_dim, config.gnn_hidden_dim)
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                config.gnn_hidden_dim, 
                config.gnn_hidden_dim,
                config.gnn_heads,
                config.dropout
            )
            for _ in range(config.gnn_num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.gnn_hidden_dim)
            for _ in range(config.gnn_num_layers)
        ])
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feat_dim]
        
        Returns:
            [num_nodes, gnn_hidden_dim]
        """
        # 输入投影
        x = self.input_proj(node_features)
        
        # GNN层
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            residual = x
            x = gnn_layer(x, edge_index, edge_attr)
            x = layer_norm(x + residual)
            x = F.relu(x)
        
        return x


class TemporalEncoder(nn.Module):
    """时序编码器（Transformer）"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 输入投影
        self.input_proj = nn.Linear(config.gnn_hidden_dim, config.transformer_hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.transformer_hidden_dim,
            dropout=config.transformer_dropout
        )
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.transformer_hidden_dim,
                config.transformer_heads,
                config.transformer_hidden_dim * 4,
                config.transformer_dropout
            )
            for _ in range(config.transformer_num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, gnn_hidden_dim]
            mask: [batch_size, seq_len]
        
        Returns:
            [batch_size, seq_len, transformer_hidden_dim]
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer层
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)
        
        return x


class VehicleFeatureEncoder(nn.Module):
    """车辆特征编码器"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 构建MLP
        layers = []
        input_dim = 15  # 车辆特征维度
        
        for hidden_dim in config.vehicle_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config.vehicle_feature_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_vehicles, 15]
        
        Returns:
            [batch_size, num_vehicles, vehicle_feature_dim]
        """
        return self.encoder(x)


class ActorNetwork(nn.Module):
    """Actor网络（策略网络）"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 构建MLP
        layers = []
        input_dim = config.transformer_hidden_dim + config.gnn_hidden_dim
        
        for hidden_dim in config.actor_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 动作头（速度控制）
        self.action_head = nn.Linear(input_dim, config.speed_action_bins)
        
        # 动作嵌入（用于生成连续动作）
        self.action_embedding = nn.Parameter(
            torch.linspace(0, 1, config.speed_action_bins).unsqueeze(0),
            requires_grad=False
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, feature_dim]
            mask: [batch_size, num_vehicles] 有效车辆掩码
        
        Returns:
            action_probs: [batch_size, num_vehicles, action_bins]
            action_values: [batch_size, num_vehicles] 连续动作值
        """
        # 共享层
        features = self.shared_layers(x)
        
        # 动作头
        logits = self.action_head(features)
        action_probs = F.softmax(logits, dim=-1)
        
        # 计算期望动作值
        action_values = torch.sum(action_probs * self.action_embedding, dim=-1)
        
        return action_probs, action_values
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            x: [batch_size, feature_dim]
            deterministic: 是否确定性选择
        
        Returns:
            action: [batch_size] 动作索引
            log_prob: [batch_size] 动作对数概率
        """
        action_probs, _ = self.forward(x)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic网络（价值网络）"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 构建MLP
        layers = []
        input_dim = config.transformer_hidden_dim + config.gnn_hidden_dim + 10  # 加上全局特征
        
        for hidden_dim in config.critic_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 价值头
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, feature_dim]
            global_features: [batch_size, global_dim]
        
        Returns:
            value: [batch_size, 1]
        """
        # 合并特征
        combined = torch.cat([x, global_features], dim=-1)
        
        # 共享层
        features = self.shared_layers(combined)
        
        # 价值头
        value = self.value_head(features)
        
        return value


class TrafficControlModel(nn.Module):
    """
    交通控制模型
    整合GNN、Transformer和Actor-Critic
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 车辆特征编码器
        self.vehicle_encoder = VehicleFeatureEncoder(config)
        
        # GNN编码器
        self.gnn_encoder = GNNEncoder(config)
        
        # 时序编码器
        self.temporal_encoder = TemporalEncoder(config)
        
        # Actor网络
        self.actor = ActorNetwork(config)
        
        # Critic网络
        self.critic = CriticNetwork(config)
        
        # 车辆级特征投影
        self.vehicle_proj = nn.Linear(config.gnn_hidden_dim, config.transformer_hidden_dim)
        
        # 全局特征聚合
        self.global_aggregator = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, config.transformer_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.transformer_hidden_dim, config.transformer_hidden_dim)
        )
    
    def encode_state(self, observation: Dict, history_observations: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码状态
        
        Args:
            observation: 当前观察
            history_observations: 历史观察列表
        
        Returns:
            vehicle_features: [num_vehicles, feature_dim]
            global_features: [global_dim]
            graph_data: 图数据
        """
        device = next(self.parameters()).device
        
        # 1. 编码车辆特征
        vehicle_features = torch.tensor(observation['vehicle_features'], dtype=torch.float32, device=device)
        vehicle_features = self.vehicle_encoder(vehicle_features)
        
        # 2. 构建图并编码
        graph_data = observation['graph']
        node_features = torch.tensor(graph_data['node_features'], dtype=torch.float32, device=device)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long, device=device)
        edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float32, device=device) if graph_data['edge_attr'].size > 0 else None
        
        # GNN编码
        gnn_features = self.gnn_encoder(node_features, edge_index, edge_attr)
        
        # 提取车辆节点特征
        num_vehicles = graph_data['num_vehicles']
        vehicle_gnn_features = gnn_features[:num_vehicles]
        
        # 3. 时序编码（如果有历史）
        if history_observations and len(history_observations) > 0:
            # 编码历史状态
            history_features = []
            for hist_obs in history_observations:
                hist_vehicle_features = torch.tensor(hist_obs['vehicle_features'], dtype=torch.float32, device=device)
                hist_vehicle_features = self.vehicle_encoder(hist_vehicle_features)
                history_features.append(hist_vehicle_features)
            
            # 堆叠历史特征
            history_features = torch.stack(history_features, dim=0)  # [seq_len, num_vehicles, feature_dim]
            history_features = history_features.unsqueeze(0)  # [1, seq_len, num_vehicles, feature_dim]
            
            # 时序编码
            temporal_features = self.temporal_encoder(history_features.mean(dim=2))  # 聚合车辆维度
            temporal_features = temporal_features.squeeze(0).mean(dim=0)  # [feature_dim]
        else:
            temporal_features = torch.zeros(config.transformer_hidden_dim, device=device)
        
        # 4. 全局特征
        global_features = torch.tensor(observation['global_features'], dtype=torch.float32, device=device)
        
        # 5. 聚合全局表示
        aggregated_global = self.global_aggregator(gnn_features.mean(dim=0))
        global_features = torch.cat([global_features, aggregated_global], dim=-1)
        
        return vehicle_gnn_features, global_features, temporal_features
    
    def forward(self, observation: Dict, history_observations: Optional[List[Dict]] = None,
                deterministic: bool = False) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        Args:
            observation: 当前观察
            history_observations: 历史观察
            deterministic: 是否确定性策略
        
        Returns:
            action_dict: 动作字典
            value: 状态价值
        """
        device = next(self.parameters()).device
        
        # 编码状态
        vehicle_features, global_features, temporal_features = self.encode_state(observation, history_observations)
        
        # 获取被控制的车辆
        controlled_vehicles = observation.get('controlled_vehicles', [])
        
        if len(controlled_vehicles) == 0:
            return {}, torch.tensor(0.0, device=device)
        
        # 为每个被控制车辆生成动作
        action_dict = {}
        log_probs = []
        
        for veh_idx, veh_id in enumerate(controlled_vehicles):
            if veh_idx < vehicle_features.size(0):
                # 获取车辆特征
                veh_feat = vehicle_features[veh_idx]
                
                # 合并特征
                combined_feat = torch.cat([veh_feat, temporal_features], dim=-1).unsqueeze(0)
                
                # Actor
                action_probs, action_value = self.actor(combined_feat)
                action_probs = action_probs.squeeze(0)
                action_value = action_value.squeeze(0)
                
                # 采样动作
                if deterministic:
                    action_idx = torch.argmax(action_probs)
                else:
                    dist = Categorical(action_probs)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    log_probs.append(log_prob)
                
                # 转换为连续动作
                action_value = torch.linspace(0, 1, self.config.speed_action_bins, device=device)[action_idx]
                
                action_dict[veh_id] = action_value.item()
        
        # Critic
        global_feat_expanded = global_features.unsqueeze(0)
        temporal_expanded = temporal_features.unsqueeze(0)
        combined_global = torch.cat([temporal_expanded, global_feat_expanded], dim=-1)
        value = self.critic(temporal_expanded, global_feat_expanded)
        
        # 计算总对数概率
        if log_probs:
            total_log_prob = torch.stack(log_probs).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=device)
        
        return action_dict, value, total_log_prob
    
    def evaluate_actions(self, observations: List[Dict], actions: List[Dict],
                         history_batch: Optional[List[List[Dict]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于PPO更新）
        
        Args:
            observations: 观察列表
            actions: 动作列表
            history_batch: 历史观察批次
        
        Returns:
            values: 价值
            log_probs: 对数概率
            entropy: 熵
        """
        device = next(self.parameters()).device
        
        values = []
        log_probs = []
        entropies = []
        
        for i, (obs, action_dict) in enumerate(zip(observations, actions)):
            history = history_batch[i] if history_batch else None
            
            # 编码状态
            vehicle_features, global_features, temporal_features = self.encode_state(obs, history)
            
            # 计算价值
            global_feat_expanded = global_features.unsqueeze(0)
            temporal_expanded = temporal_features.unsqueeze(0)
            value = self.critic(temporal_expanded, global_feat_expanded)
            values.append(value)
            
            # 计算动作概率
            controlled_vehicles = obs.get('controlled_vehicles', [])
            batch_log_probs = []
            batch_entropies = []
            
            for veh_id, action_value in action_dict.items():
                if veh_id in controlled_vehicles:
                    veh_idx = controlled_vehicles.index(veh_id)
                    if veh_idx < vehicle_features.size(0):
                        veh_feat = vehicle_features[veh_idx]
                        combined_feat = torch.cat([veh_feat, temporal_features], dim=-1).unsqueeze(0)
                        
                        action_probs, _ = self.actor(combined_feat)
                        action_probs = action_probs.squeeze(0)
                        
                        # 找到最近的离散动作
                        action_idx = int(action_value * (self.config.speed_action_bins - 1))
                        action_idx = min(action_idx, self.config.speed_action_bins - 1)
                        
                        dist = Categorical(action_probs)
                        log_prob = dist.log_prob(torch.tensor(action_idx, device=device))
                        entropy = dist.entropy()
                        
                        batch_log_probs.append(log_prob)
                        batch_entropies.append(entropy)
            
            if batch_log_probs:
                log_probs.append(torch.stack(batch_log_probs).sum())
                entropies.append(torch.stack(batch_entropies).mean())
            else:
                log_probs.append(torch.tensor(0.0, device=device))
                entropies.append(torch.tensor(0.0, device=device))
        
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        
        return values, log_probs, entropies


def create_model(config: NetworkConfig) -> TrafficControlModel:
    """创建模型"""
    return TrafficControlModel(config)
