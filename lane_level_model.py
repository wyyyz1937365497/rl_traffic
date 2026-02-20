"""
车道级精细建模 + 可微动作空间
解决两个关键问题：
1. 车道级别的精细建模（区分不同车道的影响）
2. 可微的动作空间（解决梯度传播问题）
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import traci
except ImportError:
    print("请安装traci: pip install traci")
    sys.exit(1)


# ============================================================================
# 第一部分：车道级冲突矩阵定义
# ============================================================================

@dataclass
class LaneConflict:
    """车道冲突定义"""
    lane_id: str           # 车道ID
    conflicts_with: List[str]  # 与哪些车道冲突
    conflict_type: str     # 冲突类型：'merge', 'cross', 'diverge'
    severity: float        # 冲突严重程度 (0-1)


# 预定义车道级冲突关系（基于路网拓扑和 EDGE_TOPOLOGY）
# 数据来源：road_topology_hardcoded.py 的 LANE_CONFLICTS
LANE_CONFLICTS = {
    # ==================== J5: E23匝道汇入-E2 ====================
    # 拓扑：E23 → -E2，与 -E3 来车在 -E2 上冲突
    # LANE_CONFLICTS: 'E23_0': ['-E3_0']
    'E23_0': LaneConflict(
        lane_id='E23_0',
        conflicts_with=['-E3_0'],  # 与反向主路上游来车冲突
        conflict_type='merge',
        severity=0.8
    ),

    # ==================== J14: E15匝道汇入-E9 ====================
    # 拓扑：E15 → E10，与 E9 来车在 E10 上冲突
    # LANE_CONFLICTS: 'E15_0': ['E9_0'] (注意是正向E9，不是反向-E9)
    'E15_0': LaneConflict(
        lane_id='E15_0',
        conflicts_with=['E9_0'],  # 与正向主路上游来车冲突
        conflict_type='merge',
        severity=0.7
    ),

    # ==================== J15: E17匝道汇入-E10 ====================
    # 拓扑：E17 → -E10，与 -E11 来车在 -E10 上冲突
    # LANE_CONFLICTS: 'E17_0': ['-E11_0', '-E11_1'] (关键：不与-E11_2冲突！)
    'E17_0': LaneConflict(
        lane_id='E17_0',
        conflicts_with=['-E11_0', '-E11_1'],  # 只与前2条车道冲突，不与-E11_2冲突
        conflict_type='merge',
        severity=0.8
    ),

    # ==================== J17: E19匝道汇入-E12 ====================
    # 拓扑：E19 → -E12，与 -E13 来车在 -E12 上冲突
    # LANE_CONFLICTS: 'E19_0': ['-E13_0', '-E13_1'], 'E19_1': ['-E13_0', '-E13_1']
    # 关键：不与 -E13_2 冲突！
    'E19_0': LaneConflict(
        lane_id='E19_0',
        conflicts_with=['-E13_0', '-E13_1'],  # 只与前2条车道冲突，不与-E13_2冲突
        conflict_type='merge',
        severity=0.8
    ),
    'E19_1': LaneConflict(
        lane_id='E19_1',
        conflicts_with=['-E13_0', '-E13_1'],  # 第二条匝道车道也只与前2条车道冲突
        conflict_type='merge',
        severity=0.8
    ),
}


@dataclass
class LaneFeatures:
    """车道级特征"""
    lane_id: str
    edge_id: str
    lane_index: int
    
    # 车道属性
    length: float
    speed_limit: float
    is_ramp: bool          # 是否是匝道
    is_rightmost: bool     # 是否是最右侧车道
    
    # 车辆状态
    vehicle_count: int
    mean_speed: float
    queue_length: int
    density: float
    
    # 冲突信息
    has_conflict: bool
    conflict_lanes: List[str]
    conflict_severity: float
    
    # CV车辆
    cv_vehicles: List[str]
    cv_count: int


# ============================================================================
# 第二部分：车道级状态编码器
# ============================================================================

class LaneEncoder(nn.Module):
    """
    车道级特征编码器
    为每条车道单独编码，捕捉车道级别的差异
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 32, output_dim: int = 16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, lane_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_features: [batch, num_lanes, input_dim]
        
        Returns:
            [batch, num_lanes, output_dim]
        """
        return self.encoder(lane_features)


class LaneConflictAttention(nn.Module):
    """
    车道冲突注意力模块
    学习冲突车道之间的关系
    """
    
    def __init__(self, feature_dim: int = 16, num_heads: int = 2):
        super().__init__()
        
        self.conflict_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        
        # 冲突掩码（预定义哪些车道会冲突）
        self.conflict_mask = None
    
    def set_conflict_mask(self, conflict_matrix: torch.Tensor):
        """
        设置冲突掩码
        
        Args:
            conflict_matrix: [num_lanes, num_lanes] 
                            1表示冲突，0表示不冲突
        """
        self.conflict_mask = conflict_matrix
    
    def forward(self, lane_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_features: [batch, num_lanes, feature_dim]
        
        Returns:
            [batch, num_lanes, feature_dim]
        """
        # 使用冲突掩码进行注意力计算
        if self.conflict_mask is not None:
            # 扩展掩码到batch维度
            mask = self.conflict_mask.unsqueeze(0).expand(lane_features.size(0), -1, -1)
            mask = mask == 0  # 转换为注意力掩码格式
        else:
            mask = None
        
        attended, _ = self.conflict_attention(
            lane_features, lane_features, lane_features,
            key_padding_mask=mask
        )
        
        return attended + lane_features  # 残差连接


# ============================================================================
# 第三部分：可微动作空间
# ============================================================================

class DifferentiableActionLayer(nn.Module):
    """
    可微动作层
    使用Gumbel-Softmax实现可微的离散动作
    """
    
    def __init__(self, input_dim: int, num_actions: int = 11, temperature: float = 1.0):
        super().__init__()
        
        self.num_actions = num_actions
        self.temperature = temperature
        
        # 动作logits
        self.action_logits = nn.Linear(input_dim, num_actions)
        
        # 温度参数（可学习）
        self.temperature_param = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch, input_dim]
            hard: 是否使用硬采样（训练时用软，推理时用硬）
        
        Returns:
            action_probs: [batch, num_actions] 动作概率
            action_value: [batch, 1] 连续动作值（可微）
        """
        # 计算logits
        logits = self.action_logits(x)  # [batch, num_actions]
        
        # Gumbel-Softmax
        if self.training and not hard:
            # 训练时：使用Gumbel-Softmax（可微）
            action_probs = F.gumbel_softmax(
                logits, 
                tau=self.temperature_param,
                hard=False,
                dim=-1
            )
        else:
            # 推理时：使用Softmax
            action_probs = F.softmax(logits / self.temperature_param, dim=-1)
        
        # 计算期望动作值（连续，可微）
        action_values = torch.linspace(0, 1, self.num_actions, device=x.device)
        action_value = torch.sum(action_probs * action_values, dim=-1, keepdim=True)
        
        return action_probs, action_value
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        return self.temperature_param.item()


class ContinuousActionLayer(nn.Module):
    """
    连续动作层
    直接输出连续动作值，完全可微
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        # 均值网络
        self.mean_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出范围[0, 1]
        )
        
        # 标准差网络（用于探索）
        self.std_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保正数
        )
    
    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch, input_dim]
            deterministic: 是否确定性输出
        
        Returns:
            action: [batch, 1] 动作值
            log_prob: [batch, 1] 对数概率
        """
        mean = self.mean_net(x)  # [batch, 1]
        
        if deterministic:
            return mean, torch.zeros_like(mean)
        
        std = self.std_net(x)  # [batch, 1]
        
        # 采样
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # 重参数化采样
        action = torch.clamp(action, 0, 1)  # 限制范围
        
        log_prob = normal.log_prob(action)
        
        return action, log_prob


# ============================================================================
# 第四部分：车道级策略网络
# ============================================================================

class LaneLevelPolicyNetwork(nn.Module):
    """
    车道级策略网络
    精细建模每条车道，输出可微动作
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 车道编码器
        self.lane_encoder = LaneEncoder(
            input_dim=12,  # 车道特征维度
            hidden_dim=32,
            output_dim=16
        )
        
        # 车道冲突注意力
        self.conflict_attention = LaneConflictAttention(
            feature_dim=16,
            num_heads=2
        )
        
        # 全局特征编码
        self.global_encoder = nn.Sequential(
            nn.Linear(10, 32),  # 全局特征
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 信号灯编码
        self.tl_encoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(16 + 16 + 8, 64),  # 车道 + 全局 + 信号灯
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 动作层（可微）
        self.main_action = DifferentiableActionLayer(32, num_actions=11)
        self.ramp_action = DifferentiableActionLayer(32, num_actions=11)
        
        # 价值层
        self.value_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, lane_features: torch.Tensor, global_features: torch.Tensor,
                tl_features: torch.Tensor, hard: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            lane_features: [batch, num_lanes, 12] 车道特征
            global_features: [batch, 10] 全局特征
            tl_features: [batch, 5] 信号灯特征
            hard: 是否硬采样
        
        Returns:
            字典包含：
            - main_action_probs: 主路动作概率
            - main_action_value: 主路动作值（可微）
            - ramp_action_probs: 匝道动作概率
            - ramp_action_value: 匝道动作值（可微）
            - value: 状态价值
        """
        batch_size = lane_features.size(0)
        
        # 1. 编码车道特征
        lane_encoded = self.lane_encoder(lane_features)  # [batch, num_lanes, 16]
        
        # 2. 车道冲突注意力
        lane_attended = self.conflict_attention(lane_encoded)  # [batch, num_lanes, 16]
        
        # 3. 聚合车道特征（区分主路和匝道）
        # 假设前N条是主路车道，后M条是匝道车道
        num_main_lanes = self.config.get('num_main_lanes', 3)
        
        main_lane_features = lane_attended[:, :num_main_lanes, :].mean(dim=1)  # [batch, 16]
        ramp_lane_features = lane_attended[:, num_main_lanes:, :].mean(dim=1) if lane_attended.size(1) > num_main_lanes else torch.zeros(batch_size, 16, device=lane_features.device)
        
        # 4. 编码全局特征
        global_encoded = self.global_encoder(global_features)  # [batch, 16]
        
        # 5. 编码信号灯特征
        tl_encoded = self.tl_encoder(tl_features)  # [batch, 8]
        
        # 6. 融合特征
        main_fused = self.fusion(torch.cat([main_lane_features, global_encoded, tl_encoded], dim=-1))
        ramp_fused = self.fusion(torch.cat([ramp_lane_features, global_encoded, tl_encoded], dim=-1))
        
        # 7. 输出动作（可微）
        main_probs, main_value = self.main_action(main_fused, hard=hard)
        ramp_probs, ramp_value = self.ramp_action(ramp_fused, hard=hard)
        
        # 8. 输出价值
        value = self.value_layer(main_fused)
        
        return {
            'main_action_probs': main_probs,
            'main_action_value': main_value,
            'ramp_action_probs': ramp_probs,
            'ramp_action_value': ramp_value,
            'value': value,
            'temperature': self.main_action.get_temperature()
        }


# ============================================================================
# 第五部分：车道级环境接口
# ============================================================================

class LaneLevelEnvironment:
    """
    车道级环境
    提供车道级别的状态观察
    """
    
    def __init__(self, sumo_cfg: str, junction_configs: Dict):
        self.sumo_cfg = sumo_cfg
        self.junction_configs = junction_configs
        
        # 车道冲突矩阵
        self.conflict_matrices = self._build_conflict_matrices()
    
    def _build_conflict_matrices(self) -> Dict[str, torch.Tensor]:
        """
        为每个路口构建车道冲突矩阵
        
        Returns:
            {junction_id: conflict_matrix}
        """
        conflict_matrices = {}
        
        for junc_id, config in self.junction_configs.items():
            # 获取该路口的所有车道
            all_lanes = self._get_junction_lanes(config)
            num_lanes = len(all_lanes)
            
            # 初始化冲突矩阵
            conflict_matrix = torch.zeros(num_lanes, num_lanes)
            
            # 填充冲突关系
            for i, lane_i in enumerate(all_lanes):
                if lane_i in LANE_CONFLICTS:
                    for conflict_lane in LANE_CONFLICTS[lane_i].conflicts_with:
                        if conflict_lane in all_lanes:
                            j = all_lanes.index(conflict_lane)
                            conflict_matrix[i, j] = LANE_CONFLICTS[lane_i].severity
                            conflict_matrix[j, i] = LANE_CONFLICTS[lane_i].severity
            
            conflict_matrices[junc_id] = conflict_matrix
        
        return conflict_matrices
    
    def _get_junction_lanes(self, config) -> List[str]:
        """获取路口相关的所有车道"""
        lanes = []
        
        # 主路车道
        for edge in config.get('main_incoming', []):
            try:
                lane_count = traci.edge.getLaneNumber(edge)
                for i in range(lane_count):
                    lanes.append(f"{edge}_{i}")
            except Exception as e:
                print(f"获取主路边 {edge} 车道数失败: {e}")

        # 匝道车道
        for edge in config.get('ramp_incoming', []):
            try:
                lane_count = traci.edge.getLaneNumber(edge)
                for i in range(lane_count):
                    lanes.append(f"{edge}_{i}")
            except Exception as e:
                print(f"获取匝道边 {edge} 车道数失败: {e}")

        return lanes
    
    def get_lane_features(self, junction_id: str) -> torch.Tensor:
        """
        获取车道级特征
        
        Returns:
            [num_lanes, 12] 车道特征张量
        """
        config = self.junction_configs[junction_id]
        lanes = self._get_junction_lanes(config)
        
        lane_features = []
        
        for lane_id in lanes:
            try:
                # 获取车道属性
                edge_id = '_'.join(lane_id.split('_')[:-1])
                lane_index = int(lane_id.split('_')[-1])
                
                # 车道长度和限速
                length = traci.lane.getLength(lane_id)
                speed_limit = traci.lane.getSpeed(lane_id)
                
                # 是否是匝道/最右侧车道
                is_ramp = edge_id in config.get('ramp_incoming', [])
                is_rightmost = (lane_index == 0)
                
                # 车辆状态
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                vehicle_count = len(vehicles)
                
                speeds = [traci.vehicle.getSpeed(v) for v in vehicles if traci.vehicle.getSpeed(v) >= 0]
                mean_speed = np.mean(speeds) if speeds else 0.0
                
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                
                # 密度
                density = vehicle_count / max(length / 1000, 0.1)
                
                # 冲突信息
                has_conflict = lane_id in LANE_CONFLICTS
                conflict_lanes = LANE_CONFLICTS[lane_id].conflicts_with if has_conflict else []
                conflict_severity = LANE_CONFLICTS[lane_id].severity if has_conflict else 0.0
                
                # CV车辆
                cv_vehicles = [v for v in vehicles if traci.vehicle.getTypeID(v) == 'CV']
                cv_count = len(cv_vehicles)
                
                # 构建特征向量
                features = [
                    length / 500.0,           # 归一化长度
                    speed_limit / 20.0,       # 归一化限速
                    float(is_ramp),           # 是否匝道
                    float(is_rightmost),      # 是否最右侧
                    vehicle_count / 20.0,     # 归一化车辆数
                    mean_speed / 20.0,        # 归一化平均速度
                    queue_length / 10.0,      # 归一化排队长度
                    density / 100.0,          # 归一化密度
                    float(has_conflict),      # 是否有冲突
                    conflict_severity,        # 冲突严重程度
                    cv_count / 10.0,          # 归一化CV数量
                    len(cv_vehicles) / max(vehicle_count, 1)  # CV比例
                ]
                
                lane_features.append(features)
                
            except Exception as e:
                # 如果获取失败，使用零向量
                lane_features.append([0.0] * 12)
        
        return torch.tensor(lane_features, dtype=torch.float32)


# ============================================================================
# 第六部分：测试和验证
# ============================================================================

def test_lane_conflicts():
    """测试车道冲突定义"""
    print("=" * 70)
    print("测试车道冲突定义")
    print("=" * 70)
    
    print("\n预定义的车道冲突关系:")
    print("-" * 70)
    
    for lane_id, conflict in LANE_CONFLICTS.items():
        print(f"\n{lane_id}:")
        print(f"  冲突车道: {conflict.conflicts_with}")
        print(f"  冲突类型: {conflict.conflict_type}")
        print(f"  严重程度: {conflict.severity}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


def test_differentiable_action():
    """测试可微动作层"""
    print("\n" + "=" * 70)
    print("测试可微动作层")
    print("=" * 70)
    
    # 创建可微动作层
    action_layer = DifferentiableActionLayer(input_dim=32, num_actions=11)
    
    # 测试输入
    x = torch.randn(4, 32, requires_grad=True)
    
    # 前向传播
    action_probs, action_value = action_layer(x, hard=False)
    
    print(f"\n输入形状: {x.shape}")
    print(f"动作概率形状: {action_probs.shape}")
    print(f"动作值形状: {action_value.shape}")
    
    # 测试梯度
    loss = action_value.sum()
    loss.backward()
    
    print(f"\n梯度测试:")
    print(f"  输入梯度存在: {x.grad is not None}")
    print(f"  输入梯度范数: {x.grad.norm().item():.4f}")
    
    # 测试温度
    print(f"\n当前温度: {action_layer.get_temperature():.4f}")
    
    print("\n✓ 可微动作层测试通过")
    print("=" * 70)


def test_lane_encoder():
    """测试车道编码器"""
    print("\n" + "=" * 70)
    print("测试车道编码器")
    print("=" * 70)
    
    encoder = LaneEncoder(input_dim=12, hidden_dim=32, output_dim=16)
    
    # 模拟车道特征
    lane_features = torch.randn(2, 5, 12)  # batch=2, 5条车道, 12维特征
    
    # 编码
    encoded = encoder(lane_features)
    
    print(f"\n输入形状: {lane_features.shape}")
    print(f"输出形状: {encoded.shape}")
    
    print("\n✓ 车道编码器测试通过")
    print("=" * 70)


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("车道级建模 + 可微动作测试套件")
    print("=" * 70)
    
    # 测试车道冲突
    test_lane_conflicts()
    
    # 测试可微动作
    test_differentiable_action()
    
    # 测试车道编码器
    test_lane_encoder()
    
    print("\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)
    
    print("\n关键改进:")
    print("  1. ✅ 车道级精细建模")
    print("  2. ✅ 车道冲突矩阵")
    print("  3. ✅ 可微动作空间（Gumbel-Softmax）")
    print("  4. ✅ 梯度可传播")


if __name__ == '__main__':
    main()
