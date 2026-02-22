"""
路口级神经网络模型
针对不同拓扑类型设计专门的网络结构
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

from junction_agent import JunctionType


@dataclass
class NetworkConfig:
    """网络配置"""
    # 类型A网络配置（单纯匝道汇入）
    type_a_state_dim: int = 23  # 订阅模式：基础16 + 时间1 + 匝道汇入特征6（实际需要重新计算）
    type_a_hidden_dims: List[int] = None
    type_a_action_dim: int = 3

    # 类型B网络配置（匝道汇入+主路转出）
    type_b_state_dim: int = 23  # 订阅模式：基础19 + 类型B特有3 + 时间1 = 23
    type_b_hidden_dims: List[int] = None
    type_b_action_dim: int = 4
    
    # 共享配置
    attention_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.type_a_hidden_dims is None:
            self.type_a_hidden_dims = [128, 64]
        if self.type_b_hidden_dims is None:
            self.type_b_hidden_dims = [128, 64]


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
    """空间注意力 - 捕捉主路与匝道车辆的空间关系"""
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, main_features: torch.Tensor, ramp_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            main_features: [batch, num_main, feature_dim]
            ramp_features: [batch, num_ramp, feature_dim]
        
        Returns:
            attended_main: [batch, feature_dim]
            attended_ramp: [batch, feature_dim]
        """
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
    """冲突预测模块 - 预测汇入冲突风险"""

    def __init__(self, feature_dim: int, other_dim: int = None):
        super().__init__()

        if other_dim is None:
            other_dim = feature_dim

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + other_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, main_features: torch.Tensor, ramp_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main_features: [batch, feature_dim]
            ramp_features: [batch, other_dim]

        Returns:
            conflict_prob: [batch, 1]
        """
        combined = torch.cat([main_features, ramp_features], dim=-1)
        return self.predictor(combined)


class GapPredictor(nn.Module):
    """间隙预测模块 - 预测主路可接受间隙"""

    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 8)

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, main_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main_features: [batch, input_dim]

        Returns:
            gap_score: [batch, 1] 可接受间隙评分
        """
        return self.predictor(main_features)


class TypeAPolicyNetwork(nn.Module):
    """
    类型A路口策略网络（单纯匝道汇入）
    
    关键决策：
    1. 主路CV车辆是否减速让行
    2. 匝道CV车辆何时加速汇入
    3. 汇入间隙选择
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(config.type_a_state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 车辆编码器
        self.vehicle_encoder = VehicleEncoder(input_dim=8, output_dim=16)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(16, config.num_heads)

        # 冲突预测 (64维状态特征 + 16维注意力特征)
        self.conflict_predictor = ConflictPredictor(64, 16)

        # 间隙预测 (16维注意力特征)
        self.gap_predictor = GapPredictor(16)
        
        # 主路控制头（控制主路CV车辆速度）
        self.main_control_head = nn.Sequential(
            nn.Linear(64 + 16, 32),  # 状态特征 + 注意力特征
            nn.ReLU(),
            nn.Linear(32, 11)  # 11个离散动作（速度比例0-1）
        )
        
        # 匝道控制头（控制匝道CV车辆汇入时机）
        self.ramp_control_head = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 价值头 (state_features:64 + conflict_prob:1 + gap_score:1 = 66)
        self.value_head = nn.Sequential(
            nn.Linear(66, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state: torch.Tensor, main_vehicles: torch.Tensor = None, 
                ramp_vehicles: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, state_dim] 路口状态向量
            main_vehicles: [batch, num_main, 8] 主路车辆特征
            ramp_vehicles: [batch, num_ramp, 8] 匝道车辆特征
        
        Returns:
            main_action_probs: [batch, 11] 主路动作概率
            ramp_action_probs: [batch, 11] 匝道动作概率
            value: [batch, 1] 状态价值
            conflict_prob: [batch, 1] 冲突概率
        """
        batch_size = state.size(0)
        
        # 编码状态
        state_features = self.state_encoder(state)  # [batch, 64]
        
        # 处理车辆特征
        if main_vehicles is not None and ramp_vehicles is not None:
            # 编码车辆
            main_enc = self.vehicle_encoder(main_vehicles)  # [batch, num_main, 16]
            ramp_enc = self.vehicle_encoder(ramp_vehicles)  # [batch, num_ramp, 16]
            
            # 空间注意力
            main_att, ramp_att = self.spatial_attention(main_enc, ramp_enc)
        else:
            main_att = torch.zeros(batch_size, 16, device=state.device)
            ramp_att = torch.zeros(batch_size, 16, device=state.device)
        
        # 预测冲突和间隙
        conflict_prob = self.conflict_predictor(state_features, ramp_att if ramp_att is not None else torch.zeros_like(state_features[:, :16]))
        gap_score = self.gap_predictor(main_att if main_att is not None else torch.zeros_like(state_features[:, :16]))
        
        # 控制动作
        main_features = torch.cat([state_features, main_att], dim=-1)
        ramp_features = torch.cat([state_features, ramp_att], dim=-1)
        
        main_action_logits = self.main_control_head(main_features)
        ramp_action_logits = self.ramp_control_head(ramp_features)
        
        main_action_probs = F.softmax(main_action_logits, dim=-1)
        ramp_action_probs = F.softmax(ramp_action_logits, dim=-1)
        
        # 价值
        value_features = torch.cat([state_features, conflict_prob, gap_score], dim=-1)
        value = self.value_head(value_features)
        
        return main_action_probs, ramp_action_probs, value, conflict_prob


class TypeBPolicyNetwork(nn.Module):
    """
    类型B路口策略网络（匝道汇入 + 主路转出）
    
    关键决策：
    1. 主路CV车辆是否减速让行
    2. 匝道CV车辆何时加速汇入
    3. 转出车辆与汇入车辆的协调
    4. 转出引导（帮助转出车辆选择时机）
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        self.config = config
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(config.type_b_state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 车辆编码器
        self.vehicle_encoder = VehicleEncoder(input_dim=8, output_dim=16)
        
        # 三方注意力（主路、匝道、转出）
        self.tri_attention = nn.ModuleDict({
            'main_to_ramp': nn.MultiheadAttention(16, 4, batch_first=True),
            'main_to_diverge': nn.MultiheadAttention(16, 4, batch_first=True),
            'ramp_to_main': nn.MultiheadAttention(16, 4, batch_first=True),
            'diverge_to_main': nn.MultiheadAttention(16, 4, batch_first=True)
        })
        
        # 冲突预测（汇入冲突 + 转出冲突）
        self.merge_conflict = ConflictPredictor(64, 16)
        self.diverge_conflict = ConflictPredictor(64, 16)

        # 协调模块（汇入与转出的协调）
        # 输入: state_features(64) + ramp_att(16) + diverge_att(16) = 96
        self.coordinator = nn.Sequential(
            nn.Linear(64 + 16 + 16, 64),  # 主路状态 + 匝道注意力 + 转出注意力
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 主路控制头
        self.main_control_head = nn.Sequential(
            nn.Linear(64 + 16 + 32, 32),  # 状态 + 注意力 + 协调
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
        
        # 价值头 (state_features:64 + coord_features:32 + merge_conflict:1 + diverge_conflict:1 = 98)
        self.value_head = nn.Sequential(
            nn.Linear(98, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state: torch.Tensor, main_vehicles: torch.Tensor = None,
                ramp_vehicles: torch.Tensor = None, diverge_vehicles: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            main_vehicles: [batch, num_main, 8]
            ramp_vehicles: [batch, num_ramp, 8]
            diverge_vehicles: [batch, num_diverge, 8]
        
        Returns:
            main_action_probs: [batch, 11]
            ramp_action_probs: [batch, 11]
            diverge_action_probs: [batch, 11]
            value: [batch, 1]
            merge_conflict_prob: [batch, 1]
        """
        batch_size = state.size(0)
        device = state.device
        
        # 编码状态
        state_features = self.state_encoder(state)  # [batch, 64]
        
        # 处理车辆特征
        main_att = torch.zeros(batch_size, 16, device=device)
        ramp_att = torch.zeros(batch_size, 16, device=device)
        diverge_att = torch.zeros(batch_size, 16, device=device)
        
        if main_vehicles is not None and ramp_vehicles is not None:
            main_enc = self.vehicle_encoder(main_vehicles)
            ramp_enc = self.vehicle_encoder(ramp_vehicles)
            
            # 主路-匝道注意力
            main_att, _ = self.tri_attention['main_to_ramp'](main_enc, ramp_enc, ramp_enc)
            ramp_att, _ = self.tri_attention['ramp_to_main'](ramp_enc, main_enc, main_enc)
            
            main_att = main_att.mean(dim=1)
            ramp_att = ramp_att.mean(dim=1)
        
        if diverge_vehicles is not None and main_vehicles is not None:
            diverge_enc = self.vehicle_encoder(diverge_vehicles)
            main_enc = self.vehicle_encoder(main_vehicles)
            
            # 主路-转出注意力
            diverge_att, _ = self.tri_attention['diverge_to_main'](diverge_enc, main_enc, main_enc)
            diverge_att = diverge_att.mean(dim=1)
        
        # 冲突预测
        merge_conflict = self.merge_conflict(state_features, ramp_att)
        diverge_conflict = self.diverge_conflict(state_features, diverge_att)
        
        # 协调特征
        coord_features = self.coordinator(torch.cat([
            state_features, 
            ramp_att if ramp_att is not None else torch.zeros_like(state_features[:, :16]),
            diverge_att if diverge_att is not None else torch.zeros_like(state_features[:, :16])
        ], dim=-1))
        
        # 控制动作
        control_features = torch.cat([state_features, main_att, coord_features], dim=-1)
        
        main_action_probs = F.softmax(self.main_control_head(control_features), dim=-1)
        ramp_action_probs = F.softmax(self.ramp_control_head(control_features), dim=-1)
        diverge_action_probs = F.softmax(self.diverge_guide_head(control_features), dim=-1)
        
        # 价值
        value_features = torch.cat([state_features, coord_features, merge_conflict, diverge_conflict], dim=-1)
        value = self.value_head(value_features)
        
        return main_action_probs, ramp_action_probs, diverge_action_probs, value, merge_conflict


class JunctionPolicyNetwork(nn.Module):
    """
    路口策略网络包装器
    根据路口类型选择对应的网络
    """
    
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
        """
        前向传播
        
        Returns:
            actions: 动作字典
            value: 状态价值
            info: 额外信息
        """
        if self.junction_type == JunctionType.TYPE_A:
            main_probs, ramp_probs, value, conflict = self.policy_net(state, main_vehicles, ramp_vehicles)
            
            # 采样动作
            if deterministic:
                main_action = torch.argmax(main_probs, dim=-1)
                ramp_action = torch.argmax(ramp_probs, dim=-1)
            else:
                main_dist = Categorical(main_probs)
                ramp_dist = Categorical(ramp_probs)
                main_action = main_dist.sample()
                ramp_action = ramp_dist.sample()
            
            actions = {
                'main': main_action.float() / 10.0,  # 归一化到0-1
                'ramp': ramp_action.float() / 10.0
            }
            
            info = {
                'conflict_prob': conflict,
                'main_probs': main_probs,
                'ramp_probs': ramp_probs
            }
        
        else:  # TYPE_B
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
    """
    路口间协调模块
    处理相邻路口之间的协调
    """
    
    def __init__(self, feature_dim: int = 64, num_junctions: int = 4):
        super().__init__()
        
        # 路口间注意力
        self.junction_attention = nn.MultiheadAttention(
            feature_dim, num_heads=4, batch_first=True
        )
        
        # 协调网络
        self.coordination_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 路口邻接关系（基于路网拓扑）
        self.adjacency = {
            'J5': ['J14'],      # J5下游是J14
            'J14': ['J5', 'J15'],  # J14上游J5，下游J15
            'J15': ['J14', 'J17'],  # J15上游J14，下游J17
            'J17': ['J15']      # J17上游J15
        }
    
    def forward(self, junction_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        路口间协调
        
        Args:
            junction_features: {路口ID: [batch, feature_dim]}
        
        Returns:
            coordinated_features: {路口ID: [batch, feature_dim]}
        """
        coordinated = {}
        
        for junc_id, features in junction_features.items():
            # 获取邻居特征
            neighbors = self.adjacency.get(junc_id, [])
            
            if neighbors:
                neighbor_feats = [junction_features[n] for n in neighbors if n in junction_features]
                
                if neighbor_feats:
                    # 堆叠邻居特征
                    neighbor_stack = torch.stack(neighbor_feats, dim=1)  # [batch, num_neighbors, dim]
                    
                    # 注意力
                    attended, _ = self.junction_attention(
                        features.unsqueeze(1), neighbor_stack, neighbor_stack
                    )
                    attended = attended.squeeze(1)
                    
                    # 协调
                    coordinated[junc_id] = self.coordination_net(features + attended)
                else:
                    coordinated[junc_id] = features
            else:
                coordinated[junc_id] = features
        
        return coordinated


class MultiJunctionModel(nn.Module):
    """
    多路口联合模型（改进版 - 使用差异化网络架构）
    为不同类型的路口使用专门的网络结构
    """

    def __init__(self, junction_configs: Dict, config: NetworkConfig = None):
        super().__init__()

        if config is None:
            config = NetworkConfig()

        self.config = config
        self.junction_configs = junction_configs

        # 使用自适应网络架构
        from adaptive_networks import AdaptiveJunctionNetwork
        self.adaptive_network = AdaptiveJunctionNetwork(junction_configs)

        # 路口间协调
        self.coordinator = InterJunctionCoordinator()

        # 全局价值网络
        self.global_value = nn.Sequential(
            nn.Linear(64 * len(junction_configs), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor],
                vehicle_observations: Dict[str, Dict[str, torch.Tensor]] = None,
                deterministic: bool = False) -> Tuple[Dict[str, Dict], Dict[str, torch.Tensor], Dict]:
        """
        前向传播（使用自适应网络架构）

        Args:
            observations: {路口ID: [batch, state_dim]}
            vehicle_observations: {路口ID: {'main': [...], 'ramp': [...], 'diverge': [...]}}
            deterministic: 是否确定性策略

        Returns:
            all_actions: {路口ID: {'main': action, 'ramp': action, ...}}
            all_values: {路口ID: value}
            all_info: 额外信息
        """
        from torch.distributions import Categorical

        all_actions = {}
        all_values = {}
        all_info = {}
        junction_features = {}

        for junc_id, state in observations.items():
            # 获取车辆观察
            veh_obs = vehicle_observations.get(junc_id, {}) if vehicle_observations else {}
            main_veh = veh_obs.get('main')
            ramp_veh = veh_obs.get('ramp')
            diverge_veh = veh_obs.get('diverge')

            # 调用自适应网络
            output = self.adaptive_network(
                junc_id,
                state,
                main_vehicles=main_veh,
                ramp_vehicles=ramp_veh,
                diverge_vehicles=diverge_veh
            )

            # 转换输出格式
            # 从action probabilities中采样
            main_probs = output['main_action']
            ramp_probs = output['ramp_action']

            if deterministic:
                main_action = torch.argmax(main_probs, dim=-1)
                ramp_action = torch.argmax(ramp_probs, dim=-1)
            else:
                main_dist = Categorical(main_probs)
                ramp_dist = Categorical(ramp_probs)
                main_action = main_dist.sample()
                ramp_action = ramp_dist.sample()

            # 构建actions字典
            actions = {
                'main': main_action.float() / 10.0,  # 归一化到0-1
                'ramp': ramp_action.float() / 10.0
            }

            # 如果有diverge_action（复杂网络）
            if 'diverge_action' in output:
                diverge_probs = output['diverge_action']
                if deterministic:
                    diverge_action = torch.argmax(diverge_probs, dim=-1)
                else:
                    diverge_dist = Categorical(diverge_probs)
                    diverge_action = diverge_dist.sample()
                actions['diverge'] = diverge_action.float() / 10.0

            all_actions[junc_id] = actions
            all_values[junc_id] = output['value']

            # 构建info
            info = {
                'main_probs': main_probs,
                'ramp_probs': ramp_probs
            }
            if 'diverge_action' in output:
                info['diverge_probs'] = output['diverge_action']
            if 'conflict_prob' in output:
                info['conflict_prob'] = output['conflict_prob']

            all_info[junc_id] = info

            # 保存状态编码特征用于协调（使用网络的state_encoder）
            network = self.adaptive_network.networks[junc_id]
            junction_features[junc_id] = network.state_encoder(state)

        # 路口间协调
        coordinated_features = self.coordinator(junction_features)

        # 全局价值（只在所有路口都有输入时计算）
        if len(coordinated_features) == len(self.adaptive_network.networks):
            global_feat = torch.cat(list(coordinated_features.values()), dim=-1)
            global_value = self.global_value(global_feat)
            all_info['global_value'] = global_value

        return all_actions, all_values, all_info


def create_junction_model(junction_configs: Dict, config: NetworkConfig = None) -> MultiJunctionModel:
    """创建多路口模型（旧版：路口级控制）"""
    return MultiJunctionModel(junction_configs, config)


# ============================================================================
# 车辆级控制模型（新版）
# ============================================================================

class VehicleLevelMultiJunctionModel(nn.Module):
    """
    车辆级控制的多路口模型
    为每辆CV车辆输出独立的连续动作
    """

    def __init__(self, junction_configs: Dict, config: NetworkConfig = None):
        super().__init__()

        if config is None:
            config = NetworkConfig()

        self.config = config
        self.junction_configs = junction_configs

        # 为每个路口创建车辆级控制器
        from vehicle_level_network import VehicleLevelJunctionNetwork
        self.networks = nn.ModuleDict({
            junc_id: VehicleLevelJunctionNetwork(
                state_dim=23,
                vehicle_feat_dim=8,
                hidden_dim=config.gnn_hidden_dim if hasattr(config, 'gnn_hidden_dim') else 64
            )
            for junc_id in junction_configs.keys()
        })

        # 路口间协调器（可选）
        self.coordinator = InterJunctionCoordinator()

        # 全局价值网络
        self.global_value = nn.Sequential(
            nn.Linear(64 * len(junction_configs), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observations: Dict[str, torch.Tensor],
                vehicle_observations: Dict[str, Dict[str, torch.Tensor]] = None,
                deterministic: bool = False) -> Tuple[Dict[str, Dict], Dict[str, torch.Tensor], Dict]:
        """
        前向传播（车辆级控制）

        Args:
            observations: {路口ID: [batch, state_dim]}
            vehicle_observations: {路口ID: {'main': [batch, N, 8], 'ramp': [batch, M, 8], 'diverge': [...]}}
            deterministic: 是否确定性策略

        Returns:
            all_actions: {路口ID: {'main_actions': [batch, N], 'ramp_actions': [batch, M], ...}}
            all_values: {路口ID: [batch, 1]}
            all_info: 额外信息
        """
        all_actions = {}
        all_values = {}
        all_info = {}

        for junc_id, state in observations.items():
            # 获取车辆观察
            veh_obs = vehicle_observations.get(junc_id, {}) if vehicle_observations else {}
            main_veh = veh_obs.get('main')
            ramp_veh = veh_obs.get('ramp')
            diverge_veh = veh_obs.get('diverge')

            # 调用车辆级网络
            network = self.networks[junc_id]
            output = network(state, main_veh, ramp_veh, diverge_veh)

            # 转换输出格式
            actions = {}
            if output['main_actions'] is not None:
                actions['main_actions'] = output['main_actions']
            if output['ramp_actions'] is not None:
                actions['ramp_actions'] = output['ramp_actions']
            if output['diverge_actions'] is not None:
                actions['diverge_actions'] = output['diverge_actions']

            all_actions[junc_id] = actions
            all_values[junc_id] = output['value']
            all_info[junc_id] = {}

        return all_actions, all_values, all_info


def create_vehicle_level_model(junction_configs: Dict, config: NetworkConfig = None) -> VehicleLevelMultiJunctionModel:
    """创建车辆级控制模型（新版）"""
    return VehicleLevelMultiJunctionModel(junction_configs, config)
