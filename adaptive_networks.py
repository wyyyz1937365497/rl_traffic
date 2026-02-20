"""
差异化路口网络架构
为不同类型的路口设计专门的网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class JunctionComplexity(Enum):
    """路口复杂度"""
    SIMPLE = "simple"           # 简单汇入：J5, J14
    COMPLEX = "complex"         # 复杂汇入+转出：J15
    HIGH_CONFLICT = "high_conflict"  # 高冲突：J17
    BOTTLENECK = "bottleneck"   # 瓶颈路口


@dataclass
class JunctionCharacteristics:
    """路口特征"""
    junction_id: str
    complexity: JunctionComplexity
    
    # 道路特征
    num_main_lanes: int
    num_ramp_lanes: int
    has_diverge: bool
    
    # 流量特征
    avg_traffic_volume: float
    peak_hour_factor: float
    
    # 冲突特征
    conflict_severity: float
    merge_difficulty: float


# 预定义路口特征
JUNCTION_CHARACTERISTICS = {
    'J5': JunctionCharacteristics(
        junction_id='J5',
        complexity=JunctionComplexity.SIMPLE,
        num_main_lanes=2,
        num_ramp_lanes=1,
        has_diverge=False,
        avg_traffic_volume=0.6,
        peak_hour_factor=1.2,
        conflict_severity=0.5,
        merge_difficulty=0.4
    ),
    'J14': JunctionCharacteristics(
        junction_id='J14',
        complexity=JunctionComplexity.SIMPLE,
        num_main_lanes=2,
        num_ramp_lanes=1,
        has_diverge=False,
        avg_traffic_volume=0.7,
        peak_hour_factor=1.3,
        conflict_severity=0.6,
        merge_difficulty=0.5
    ),
    'J15': JunctionCharacteristics(
        junction_id='J15',
        complexity=JunctionComplexity.COMPLEX,
        num_main_lanes=3,
        num_ramp_lanes=1,
        has_diverge=True,
        avg_traffic_volume=0.8,
        peak_hour_factor=1.5,
        conflict_severity=0.7,
        merge_difficulty=0.6
    ),
    'J17': JunctionCharacteristics(
        junction_id='J17',
        complexity=JunctionComplexity.HIGH_CONFLICT,
        num_main_lanes=3,
        num_ramp_lanes=2,
        has_diverge=True,
        avg_traffic_volume=0.9,
        peak_hour_factor=1.6,
        conflict_severity=0.9,
        merge_difficulty=0.8
    )
}


# ===== 简单汇入网络（J5, J14）=====

class SimpleMergeNetwork(nn.Module):
    """
    简单汇入网络
    适用于：2车道主路 + 1车道匝道
    特点：轻量级，快速响应
    """
    
    def __init__(self, state_dim: int = 23):
        super().__init__()

        # 轻量级状态编码器（输出64维以兼容其他网络）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # 简单注意力
        self.attention = nn.MultiheadAttention(16, 2, batch_first=True)

        # 控制头
        self.main_control = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )

        self.ramp_control = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )

        # 价值头
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, state, main_vehicles=None, ramp_vehicles=None, **kwargs):
        # 编码状态
        state_feat = self.state_encoder(state)
        
        # 简单注意力
        if main_vehicles is not None and ramp_vehicles is not None:
            combined = torch.cat([main_vehicles, ramp_vehicles], dim=1)
            attn_out, _ = self.attention(combined, combined, combined)
            attn_feat = attn_out.mean(dim=1)
        else:
            attn_feat = torch.zeros(state.size(0), 16, device=state.device)
        
        # 控制输出
        control_feat = torch.cat([state_feat, attn_feat], dim=-1)
        
        main_action = F.softmax(self.main_control(control_feat), dim=-1)
        ramp_action = F.softmax(self.ramp_control(control_feat), dim=-1)
        value = self.value_head(state_feat)
        
        return {
            'main_action': main_action,
            'ramp_action': ramp_action,
            'value': value
        }


# ===== 复杂汇入+转出网络（J15）=====

class ComplexMergeDivergeNetwork(nn.Module):
    """
    复杂汇入+转出网络
    适用于：3车道主路 + 汇入 + 转出
    特点：三方协调，精细控制
    """
    
    def __init__(self, state_dim: int = 23):
        super().__init__()
        
        # 深层状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 三方注意力
        self.tri_attention = nn.ModuleDict({
            'main_ramp': nn.MultiheadAttention(16, 4, batch_first=True),
            'main_diverge': nn.MultiheadAttention(16, 4, batch_first=True),
            'ramp_diverge': nn.MultiheadAttention(16, 4, batch_first=True)
        })
        
        # 协调模块
        self.coordinator = nn.Sequential(
            nn.Linear(64 + 16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 多控制头
        self.main_control = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        self.ramp_control = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        self.diverge_control = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state, main_vehicles=None, ramp_vehicles=None, diverge_vehicles=None, **kwargs):
        batch_size = state.size(0)
        device = state.device
        
        # 编码状态
        state_feat = self.state_encoder(state)
        
        # 三方注意力
        main_att = torch.zeros(batch_size, 16, device=device)
        ramp_att = torch.zeros(batch_size, 16, device=device)
        diverge_att = torch.zeros(batch_size, 16, device=device)
        
        if main_vehicles is not None and ramp_vehicles is not None:
            # 主路-匝道注意力
            combined = torch.cat([main_vehicles, ramp_vehicles], dim=1)
            attn_out, _ = self.tri_attention['main_ramp'](combined, combined, combined)
            main_att = attn_out[:, :main_vehicles.size(1), :].mean(dim=1)
            ramp_att = attn_out[:, main_vehicles.size(1):, :].mean(dim=1)
        
        if diverge_vehicles is not None:
            diverge_att = diverge_vehicles.mean(dim=1) if diverge_vehicles.dim() == 3 else diverge_vehicles
        
        # 协调
        coord_feat = self.coordinator(torch.cat([state_feat, ramp_att, diverge_att], dim=-1))
        
        # 控制输出
        control_feat = torch.cat([state_feat, coord_feat], dim=-1)
        
        main_action = F.softmax(self.main_control(control_feat), dim=-1)
        ramp_action = F.softmax(self.ramp_control(control_feat), dim=-1)
        diverge_action = F.softmax(self.diverge_control(control_feat), dim=-1)
        value = self.value_head(control_feat)
        
        return {
            'main_action': main_action,
            'ramp_action': ramp_action,
            'diverge_action': diverge_action,
            'value': value
        }


# ===== 高冲突网络（J17）=====

class HighConflictNetwork(nn.Module):
    """
    高冲突网络
    适用于：高冲突路口，多车道匝道
    特点：冲突预测，保守策略
    """
    
    def __init__(self, state_dim: int = 23):
        super().__init__()
        
        # 深层状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # 冲突预测模块
        self.conflict_predictor = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 多层注意力
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(16, 4, batch_first=True)
            for _ in range(3)
        ])
        
        # 保守策略模块
        self.conservative_policy = nn.Sequential(
            nn.Linear(64 + 16 + 1, 32),  # +1 for conflict prob
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 控制头
        self.main_control = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        self.ramp_control = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        self.diverge_control = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(64 + 16 + 16 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state, main_vehicles=None, ramp_vehicles=None, diverge_vehicles=None, **kwargs):
        batch_size = state.size(0)
        device = state.device
        
        # 编码状态
        state_feat = self.state_encoder(state)
        
        # 多层注意力
        attn_feat = torch.zeros(batch_size, 16, device=device)
        
        if main_vehicles is not None and ramp_vehicles is not None:
            combined = torch.cat([main_vehicles, ramp_vehicles], dim=1)
            
            for attn_layer in self.attention_layers:
                attn_out, _ = attn_layer(combined, combined, combined)
                combined = combined + attn_out  # 残差连接
            
            attn_feat = combined.mean(dim=1)
        
        # 冲突预测
        conflict_prob = self.conflict_predictor(torch.cat([state_feat, attn_feat], dim=-1))
        
        # 保守策略
        conservative_feat = self.conservative_policy(
            torch.cat([state_feat, attn_feat, conflict_prob], dim=-1)
        )
        
        # 控制输出
        control_feat = torch.cat([state_feat, attn_feat, conservative_feat], dim=-1)
        
        main_action = F.softmax(self.main_control(control_feat), dim=-1)
        ramp_action = F.softmax(self.ramp_control(control_feat), dim=-1)
        diverge_action = F.softmax(self.diverge_control(control_feat), dim=-1)
        
        # 价值（考虑冲突风险）
        value = self.value_head(torch.cat([control_feat, conflict_prob], dim=-1))
        
        return {
            'main_action': main_action,
            'ramp_action': ramp_action,
            'diverge_action': diverge_action,
            'value': value,
            'conflict_prob': conflict_prob
        }


# ===== 自适应网络选择器 =====

class AdaptiveJunctionNetwork(nn.Module):
    """
    自适应路口网络
    根据路口特征自动选择合适的网络架构
    """
    
    def __init__(self, junction_configs: Dict):
        super().__init__()
        
        # 为每个路口创建专门的网络
        self.networks = nn.ModuleDict()
        
        for junc_id, config in junction_configs.items():
            # 获取路口特征
            characteristics = JUNCTION_CHARACTERISTICS.get(junc_id)
            
            if characteristics is None:
                # 默认使用简单网络
                self.networks[junc_id] = SimpleMergeNetwork()
            else:
                # 根据复杂度选择网络
                if characteristics.complexity == JunctionComplexity.SIMPLE:
                    self.networks[junc_id] = SimpleMergeNetwork()
                elif characteristics.complexity == JunctionComplexity.COMPLEX:
                    self.networks[junc_id] = ComplexMergeDivergeNetwork()
                elif characteristics.complexity == JunctionComplexity.HIGH_CONFLICT:
                    self.networks[junc_id] = HighConflictNetwork()
                else:
                    self.networks[junc_id] = SimpleMergeNetwork()
    
    def forward(self, junction_id: str, state, **kwargs):
        """前向传播"""
        network = self.networks[junction_id]
        return network(state, **kwargs)
    
    def get_network_info(self) -> Dict:
        """获取网络信息"""
        info = {}
        for junc_id, network in self.networks.items():
            info[junc_id] = {
                'type': network.__class__.__name__,
                'num_parameters': sum(p.numel() for p in network.parameters())
            }
        return info


# ===== 使用示例 =====

def create_adaptive_network(junction_configs: Dict) -> AdaptiveJunctionNetwork:
    """创建自适应网络"""
    return AdaptiveJunctionNetwork(junction_configs)


# 打印网络信息
def print_network_info(network: AdaptiveJunctionNetwork):
    """打印网络信息"""
    info = network.get_network_info()
    
    print("=" * 70)
    print("差异化网络架构")
    print("=" * 70)
    
    for junc_id, net_info in info.items():
        print(f"\n{junc_id}:")
        print(f"  类型: {net_info['type']}")
        print(f"  参数量: {net_info['num_parameters']:,}")
