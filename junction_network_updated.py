"""
路口级神经网络模型 - 更新版本
状态维度22维，包含信号灯特征
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# 导入路口类型
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junction_agent import JunctionType


@dataclass
class NetworkConfig:
    """网络配置"""
    # 类型A网络配置
    type_a_state_dim: int = 22  # 更新：从16维增加到22维
    type_a_hidden_dims: List[int] = None
    type_a_action_dim: int = 3
    
    # 类型B网络配置
    type_b_state_dim: int = 22  # 更新
    type_b_hidden_dims: List[int] = None
    type_b_action_dim: int = 4
    
    # 信号灯特征维度
    tl_feature_dim: int = 5
    
    # 共享配置
    attention_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.type_a_hidden_dims is None:
            self.type_a_hidden_dims = [128, 64]
        if self.type_b_hidden_dims is None:
            self.type_b_hidden_dims = [128, 64]


class TrafficLightEncoder(nn.Module):
    """
    信号灯特征编码器
    专门处理信号灯相位信息
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, output_dim: int = 16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, tl_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tl_features: [batch, 5] 
                - 当前相位索引
                - 距离切换时间
                - 主路信号状态
                - 匝道信号状态
                - 转出信号状态
        
        Returns:
            [batch, output_dim]
        """
        return self.encoder(tl_features)


class VehicleEncoder(nn.Module):
    """车辆级特征编码器"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialAttention(nn.Module):
    """空间注意力"""
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, main_features: torch.Tensor, ramp_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = main_features.size(0)
        
        # 主路车辆关注匝道车辆
        main_attended, _ = self.attention(main_features, ramp_features, ramp_features)
        main_attended = self.norm(main_features + main_attended)
        
        # 匝道车辆关注主路车辆
        ramp_attended, _ = self.attention(ramp_features, main_features, main_features)
        ramp_attended = self.norm(ramp_features + ramp_attended)
        
        # 聚合
        main_agg = main_attended.mean(dim=1)
        ramp_agg = ramp_attended.mean(dim=1)
        
        return main_agg, ramp_agg


class ConflictPredictor(nn.Module):
    """冲突预测模块"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 16, feature_dim),  # +16 for tl features
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, main_features: torch.Tensor, ramp_features: torch.Tensor,
                tl_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([main_features, ramp_features, tl_features], dim=-1)
        return self.predictor(combined)


class GapPredictor(nn.Module):
    """间隙预测模块"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + 16, feature_dim // 2),  # +16 for tl features
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, main_features: torch.Tensor, tl_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([main_features, tl_features], dim=-1)
        return self.predictor(combined)


class TypeAPolicyNetwork(nn.Module):
    """
    类型A路口策略网络（单纯匝道汇入）
    状态维度：22维（包含信号灯特征）
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 状态编码器（非信号灯部分）
        self.state_encoder = nn.Sequential(
            nn.Linear(config.type_a_state_dim - config.tl_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 信号灯编码器
        self.tl_encoder = TrafficLightEncoder(
            input_dim=config.tl_feature_dim,
            hidden_dim=32,
            output_dim=16
        )
        
        # 车辆编码器
        self.vehicle_encoder = VehicleEncoder(input_dim=8, output_dim=16)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(16, config.num_heads)
        
        # 冲突预测（考虑信号灯）
        self.conflict_predictor = ConflictPredictor(64)
        
        # 间隙预测（考虑信号灯）
        self.gap_predictor = GapPredictor(64)
        
        # 主路控制头
        self.main_control_head = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),  # 状态 + 注意力 + 信号灯
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 匝道控制头
        self.ramp_control_head = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(64 + 16 + 32, 32),  # 状态 + 信号灯 + 冲突/间隙
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state: torch.Tensor, main_vehicles: torch.Tensor = None, 
                ramp_vehicles: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, 22] 路口状态向量（包含信号灯特征）
            main_vehicles: [batch, num_main, 8] 主路车辆特征
            ramp_vehicles: [batch, num_ramp, 8] 匝道车辆特征
        
        Returns:
            main_action_probs: [batch, 11]
            ramp_action_probs: [batch, 11]
            value: [batch, 1]
            conflict_prob: [batch, 1]
        """
        batch_size = state.size(0)
        
        # 分离信号灯特征
        non_tl_features = state[:, :self.config.type_a_state_dim - self.config.tl_feature_dim]
        tl_features = state[:, self.config.type_a_state_dim - self.config.tl_feature_dim:]
        
        # 编码状态
        state_features = self.state_encoder(non_tl_features)  # [batch, 64]
        
        # 编码信号灯
        tl_encoded = self.tl_encoder(tl_features)  # [batch, 16]
        
        # 处理车辆特征
        if main_vehicles is not None and ramp_vehicles is not None:
            main_enc = self.vehicle_encoder(main_vehicles)
            ramp_enc = self.vehicle_encoder(ramp_vehicles)
            main_att, ramp_att = self.spatial_attention(main_enc, ramp_enc)
        else:
            main_att = torch.zeros(batch_size, 16, device=state.device)
            ramp_att = torch.zeros(batch_size, 16, device=state.device)
        
        # 预测冲突和间隙（考虑信号灯）
        conflict_prob = self.conflict_predictor(state_features, ramp_att, tl_encoded)
        gap_score = self.gap_predictor(main_att, tl_encoded)
        
        # 控制动作（融合信号灯特征）
        main_features = torch.cat([state_features, main_att, tl_encoded], dim=-1)
        ramp_features = torch.cat([state_features, ramp_att, tl_encoded], dim=-1)
        
        main_action_logits = self.main_control_head(main_features)
        ramp_action_logits = self.ramp_control_head(ramp_features)
        
        main_action_probs = F.softmax(main_action_logits, dim=-1)
        ramp_action_probs = F.softmax(ramp_action_logits, dim=-1)
        
        # 价值
        value_features = torch.cat([state_features, tl_encoded, conflict_prob, gap_score], dim=-1)
        value = self.value_head(value_features)
        
        return main_action_probs, ramp_action_probs, value, conflict_prob


class TypeBPolicyNetwork(nn.Module):
    """
    类型B路口策略网络（匝道汇入 + 主路转出）
    状态维度：22维（包含信号灯特征）
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(config.type_b_state_dim - config.tl_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 信号灯编码器
        self.tl_encoder = TrafficLightEncoder(
            input_dim=config.tl_feature_dim,
            hidden_dim=32,
            output_dim=16
        )
        
        # 车辆编码器
        self.vehicle_encoder = VehicleEncoder(input_dim=8, output_dim=16)
        
        # 三方注意力
        self.tri_attention = nn.ModuleDict({
            'main_to_ramp': nn.MultiheadAttention(16, 4, batch_first=True),
            'main_to_diverge': nn.MultiheadAttention(16, 4, batch_first=True),
            'ramp_to_main': nn.MultiheadAttention(16, 4, batch_first=True),
            'diverge_to_main': nn.MultiheadAttention(16, 4, batch_first=True)
        })
        
        # 冲突预测
        self.merge_conflict = ConflictPredictor(64)
        self.diverge_conflict = ConflictPredictor(64)
        
        # 协调模块
        self.coordinator = nn.Sequential(
            nn.Linear(64 * 3 + 16, 64),  # +16 for tl features
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 主路控制头
        self.main_control_head = nn.Sequential(
            nn.Linear(64 + 16 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 匝道控制头
        self.ramp_control_head = nn.Sequential(
            nn.Linear(64 + 16 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 转出引导头
        self.diverge_guide_head = nn.Sequential(
            nn.Linear(64 + 16 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(64 + 16 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state: torch.Tensor, main_vehicles: torch.Tensor = None,
                ramp_vehicles: torch.Tensor = None, diverge_vehicles: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size = state.size(0)
        device = state.device
        
        # 分离信号灯特征
        non_tl_features = state[:, :self.config.type_b_state_dim - self.config.tl_feature_dim]
        tl_features = state[:, self.config.type_b_state_dim - self.config.tl_feature_dim:]
        
        # 编码状态
        state_features = self.state_encoder(non_tl_features)
        
        # 编码信号灯
        tl_encoded = self.tl_encoder(tl_features)
        
        # 处理车辆特征
        main_att = torch.zeros(batch_size, 16, device=device)
        ramp_att = torch.zeros(batch_size, 16, device=device)
        diverge_att = torch.zeros(batch_size, 16, device=device)
        
        if main_vehicles is not None and ramp_vehicles is not None:
            main_enc = self.vehicle_encoder(main_vehicles)
            ramp_enc = self.vehicle_encoder(ramp_vehicles)
            
            main_att, _ = self.tri_attention['main_to_ramp'](main_enc, ramp_enc, ramp_enc)
            ramp_att, _ = self.tri_attention['ramp_to_main'](ramp_enc, main_enc, main_enc)
            
            main_att = main_att.mean(dim=1)
            ramp_att = ramp_att.mean(dim=1)
        
        if diverge_vehicles is not None and main_vehicles is not None:
            diverge_enc = self.vehicle_encoder(diverge_vehicles)
            main_enc = self.vehicle_encoder(main_vehicles)
            
            diverge_att, _ = self.tri_attention['diverge_to_main'](diverge_enc, main_enc, main_enc)
            diverge_att = diverge_att.mean(dim=1)
        
        # 冲突预测
        merge_conflict = self.merge_conflict(state_features, ramp_att, tl_encoded)
        diverge_conflict = self.diverge_conflict(state_features, diverge_att, tl_encoded)
        
        # 协调特征
        coord_features = self.coordinator(torch.cat([
            state_features, 
            ramp_att if ramp_att is not None else torch.zeros_like(state_features[:, :16]),
            diverge_att if diverge_att is not None else torch.zeros_like(state_features[:, :16]),
            tl_encoded
        ], dim=-1))
        
        # 控制动作
        control_features = torch.cat([state_features, main_att, coord_features], dim=-1)
        
        main_action_probs = F.softmax(self.main_control_head(control_features), dim=-1)
        ramp_action_probs = F.softmax(self.ramp_control_head(control_features), dim=-1)
        diverge_action_probs = F.softmax(self.diverge_guide_head(control_features), dim=-1)
        
        # 价值
        value_features = torch.cat([state_features, tl_encoded, coord_features], dim=-1)
        value = self.value_head(value_features)
        
        return main_action_probs, ramp_action_probs, diverge_action_probs, value, merge_conflict


class JunctionPolicyNetwork(nn.Module):
    """路口策略网络包装器"""
    
    def __init__(self, junction_type: JunctionType, config: NetworkConfig = None):
        super().__init__()
        
        if config is None:
            config = NetworkConfig()
        
        self.junction_type = junction_type
        self.config = config
        
        if junction_type == JunctionType.TYPE_A:
            self.policy_net = TypeAPolicyNetwork(config)
        else:
            self.policy_net = TypeBPolicyNetwork(config)
    
    def forward(self, state: torch.Tensor, main_vehicles: torch.Tensor = None,
                ramp_vehicles: torch.Tensor = None, diverge_vehicles: torch.Tensor = None,
                deterministic: bool = False) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict]:
        """前向传播"""
        if self.junction_type == JunctionType.TYPE_A:
            main_probs, ramp_probs, value, conflict = self.policy_net(state, main_vehicles, ramp_vehicles)
            
            if deterministic:
                main_action = torch.argmax(main_probs, dim=-1)
                ramp_action = torch.argmax(ramp_probs, dim=-1)
            else:
                main_dist = Categorical(main_probs)
                ramp_dist = Categorical(ramp_probs)
                main_action = main_dist.sample()
                ramp_action = ramp_dist.sample()
            
            actions = {
                'main': main_action.float() / 10.0,
                'ramp': ramp_action.float() / 10.0
            }
            
            info = {
                'conflict_prob': conflict,
                'main_probs': main_probs,
                'ramp_probs': ramp_probs
            }
        
        else:
            main_probs, ramp_probs, diverge_probs, value, conflict = self.policy_net(
                state, main_vehicles, ramp_vehicles, diverge_vehicles
            )
            
            if deterministic:
                main_action = torch.argmax(main_probs, dim=-1)
                ramp_action = torch.argmax(ramp_probs, dim=-1)
                diverge_action = torch.argmax(diverge_probs, dim=-1)
            else:
                main_dist = Categorical(main_probs)
                ramp_dist = Categorical(ramp_probs)
                diverge_dist = Categorical(diverge_probs)
                main_action = main_dist.sample()
                ramp_action = ramp_dist.sample()
                diverge_action = diverge_dist.sample()
            
            actions = {
                'main': main_action.float() / 10.0,
                'ramp': ramp_action.float() / 10.0,
                'diverge': diverge_action.float() / 10.0
            }
            
            info = {
                'conflict_prob': conflict,
                'main_probs': main_probs,
                'ramp_probs': ramp_probs,
                'diverge_probs': diverge_probs
            }
        
        return actions, value, info


class InterJunctionCoordinator(nn.Module):
    """路口间协调模块"""
    
    def __init__(self, feature_dim: int = 64, num_junctions: int = 4):
        super().__init__()
        
        self.junction_attention = nn.MultiheadAttention(
            feature_dim, num_heads=4, batch_first=True
        )
        
        self.coordination_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.adjacency = {
            'J5': ['J14'],
            'J14': ['J5', 'J15'],
            'J15': ['J14', 'J17'],
            'J17': ['J15']
        }
    
    def forward(self, junction_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        coordinated = {}
        
        for junc_id, features in junction_features.items():
            neighbors = self.adjacency.get(junc_id, [])
            
            if neighbors:
                neighbor_feats = [junction_features[n] for n in neighbors if n in junction_features]
                
                if neighbor_feats:
                    neighbor_stack = torch.stack(neighbor_feats, dim=1)
                    attended, _ = self.junction_attention(
                        features.unsqueeze(1), neighbor_stack, neighbor_stack
                    )
                    attended = attended.squeeze(1)
                    coordinated[junc_id] = self.coordination_net(features + attended)
                else:
                    coordinated[junc_id] = features
            else:
                coordinated[junc_id] = features
        
        return coordinated


class MultiJunctionModel(nn.Module):
    """多路口联合模型"""
    
    def __init__(self, junction_configs: Dict, config: NetworkConfig = None):
        super().__init__()
        
        if config is None:
            config = NetworkConfig()
        
        self.config = config
        
        self.junction_policies = nn.ModuleDict()
        for junc_id, junc_config in junction_configs.items():
            self.junction_policies[junc_id] = JunctionPolicyNetwork(
                junc_config.junction_type, config
            )
        
        self.coordinator = InterJunctionCoordinator()
        
        self.global_value = nn.Sequential(
            nn.Linear(64 * len(junction_configs), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor],
                vehicle_observations: Dict[str, Dict[str, torch.Tensor]] = None,
                deterministic: bool = False) -> Tuple[Dict[str, Dict], Dict[str, torch.Tensor], Dict]:
        """前向传播"""
        all_actions = {}
        all_values = {}
        all_info = {}
        junction_features = {}
        
        for junc_id, state in observations.items():
            veh_obs = vehicle_observations.get(junc_id, {}) if vehicle_observations else {}
            main_veh = veh_obs.get('main')
            ramp_veh = veh_obs.get('ramp')
            diverge_veh = veh_obs.get('diverge')
            
            policy = self.junction_policies[junc_id]
            actions, value, info = policy(
                state, main_veh, ramp_veh, diverge_veh, deterministic
            )
            
            all_actions[junc_id] = actions
            all_values[junc_id] = value
            all_info[junc_id] = info
            
            # 保存特征
            junction_features[junc_id] = policy.policy_net.state_encoder(
                state[:, :self.config.type_a_state_dim - self.config.tl_feature_dim]
            )
        
        # 路口间协调
        coordinated_features = self.coordinator(junction_features)
        
        # 全局价值
        if len(coordinated_features) > 0:
            global_feat = torch.cat(list(coordinated_features.values()), dim=-1)
            global_value = self.global_value(global_feat)
            all_info['global_value'] = global_value
        
        return all_actions, all_values, all_info


def create_junction_model(junction_configs: Dict, config: NetworkConfig = None) -> MultiJunctionModel:
    """创建多路口模型"""
    return MultiJunctionModel(junction_configs, config)
