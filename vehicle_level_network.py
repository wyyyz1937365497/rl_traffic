"""
车辆级控制网络
为每辆CV车辆输出独立的连续动作（速度控制）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class VehicleLevelController(nn.Module):
    """
    车辆级控制器
    为每辆车输出独立的连续动作

    输入:
        - state: [batch, state_dim] 路口状态
        - vehicles: [batch, max_vehicles, vehicle_feat_dim] 车辆特征

    输出:
        - vehicle_actions: [batch, max_vehicles, 1] 每辆车的动作（0-1连续值）
        - value: [batch, 1] 状态价值
    """

    def __init__(self, state_dim: int = 23, vehicle_feat_dim: int = 8, hidden_dim: int = 64):
        super().__init__()

        self.state_dim = state_dim
        self.vehicle_feat_dim = vehicle_feat_dim
        self.hidden_dim = hidden_dim

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 车辆特征编码器
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 车辆间注意力机制（捕捉车辆之间的相互影响）
        self.vehicle_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # 车辆动作头（为每辆车输出动作）
        self.vehicle_action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 车辆特征 + 全局状态
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1的连续值
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, vehicles, vehicle_mask=None):
        """
        前向传播

        Args:
            state: [batch_size, state_dim] 路口状态
            vehicles: [batch_size, num_vehicles, vehicle_feat_dim] 车辆特征
            vehicle_mask: [batch_size, num_vehicles] 车辆掩码（标记真实车辆）

        Returns:
            actions: [batch_size, num_vehicles] 每辆车的动作（0-1）
            value: [batch_size, 1] 状态价值
        """
        batch_size, num_vehicles, _ = vehicles.shape
        device = vehicles.device

        # 编码状态
        state_feat = self.state_encoder(state)  # [batch, hidden]

        # 编码车辆特征
        vehicle_feat = self.vehicle_encoder(vehicles)  # [batch, num_vehicles, hidden]

        # 自注意力：让车辆之间相互感知
        attn_out, _ = self.vehicle_attention(
            vehicle_feat, vehicle_feat, vehicle_feat
        )  # [batch, num_vehicles, hidden]

        # 残差连接
        vehicle_feat = vehicle_feat + attn_out

        # 全局状态扩展到每辆车
        global_state_expanded = state_feat.unsqueeze(1).expand(-1, num_vehicles, -1)

        # 拼接车辆特征和全局状态
        combined_feat = torch.cat([vehicle_feat, global_state_expanded], dim=-1)
        # [batch, num_vehicles, hidden*2]

        # 为每辆车输出动作
        vehicle_actions = self.vehicle_action_head(combined_feat).squeeze(-1)
        # [batch, num_vehicles]

        # 如果有掩码，将无效位置的动作设为0
        if vehicle_mask is not None:
            vehicle_actions = vehicle_actions * vehicle_mask

        # 计算价值（使用全局特征）
        global_feat = torch.cat([state_feat, vehicle_feat.mean(dim=1)], dim=-1)
        value = self.value_head(global_feat)

        return vehicle_actions, value


class VehicleLevelJunctionNetwork(nn.Module):
    """
    车辆级路口网络
    分别处理主路、匝道、分流车辆
    """

    def __init__(self, state_dim: int = 23, vehicle_feat_dim: int = 8, hidden_dim: int = 64):
        super().__init__()

        # 主路车辆控制器
        self.main_controller = VehicleLevelController(
            state_dim=state_dim,
            vehicle_feat_dim=vehicle_feat_dim,
            hidden_dim=hidden_dim
        )

        # 匝道车辆控制器
        self.ramp_controller = VehicleLevelController(
            state_dim=state_dim,
            vehicle_feat_dim=vehicle_feat_dim,
            hidden_dim=hidden_dim
        )

        # 分流车辆控制器（如果存在）
        self.diverge_controller = VehicleLevelController(
            state_dim=state_dim,
            vehicle_feat_dim=vehicle_feat_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, state, main_vehicles=None, ramp_vehicles=None, diverge_vehicles=None):
        """
        前向传播

        Args:
            state: [batch, state_dim] 路口状态
            main_vehicles: [batch, num_main, vehicle_feat_dim] or None
            ramp_vehicles: [batch, num_ramp, vehicle_feat_dim] or None
            diverge_vehicles: [batch, num_diverge, vehicle_feat_dim] or None

        Returns:
            dict with keys:
                - 'main_actions': [batch, num_main] or None
                - 'ramp_actions': [batch, num_ramp] or None
                - 'diverge_actions': [batch, num_diverge] or None
                - 'value': [batch, 1]
        """
        batch_size = state.size(0)
        device = state.device

        results = {}
        values = []

        # 处理主路车辆
        if main_vehicles is not None and main_vehicles.size(1) > 0:
            num_main = main_vehicles.size(1)
            # 创建掩码（假设所有位置都有效，可以根据需要调整）
            main_mask = torch.ones(batch_size, num_main, device=device)

            main_actions, main_value = self.main_controller(
                state, main_vehicles, main_mask
            )
            results['main_actions'] = main_actions
            values.append(main_value)
        else:
            results['main_actions'] = None

        # 处理匝道车辆
        if ramp_vehicles is not None and ramp_vehicles.size(1) > 0:
            num_ramp = ramp_vehicles.size(1)
            ramp_mask = torch.ones(batch_size, num_ramp, device=device)

            ramp_actions, ramp_value = self.ramp_controller(
                state, ramp_vehicles, ramp_mask
            )
            results['ramp_actions'] = ramp_actions
            values.append(ramp_value)
        else:
            results['ramp_actions'] = None

        # 处理分流车辆
        if diverge_vehicles is not None and diverge_vehicles.size(1) > 0:
            num_diverge = diverge_vehicles.size(1)
            diverge_mask = torch.ones(batch_size, num_diverge, device=device)

            diverge_actions, diverge_value = self.diverge_controller(
                state, diverge_vehicles, diverge_mask
            )
            results['diverge_actions'] = diverge_actions
            values.append(diverge_value)
        else:
            results['diverge_actions'] = None

        # 聚合价值（取平均）
        if values:
            results['value'] = torch.stack(values, dim=0).mean(dim=0)
        else:
            results['value'] = torch.zeros(batch_size, 1, device=device)

        return results


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    batch_size = 2
    state_dim = 23
    vehicle_feat_dim = 8

    state = torch.randn(batch_size, state_dim)
    main_vehicles = torch.randn(batch_size, 5, vehicle_feat_dim)  # 5辆主路车
    ramp_vehicles = torch.randn(batch_size, 3, vehicle_feat_dim)  # 3辆匝道车
    diverge_vehicles = torch.randn(batch_size, 2, vehicle_feat_dim)  # 2辆分流车

    # 创建网络
    network = VehicleLevelJunctionNetwork()

    # 前向传播
    output = network(state, main_vehicles, ramp_vehicles, diverge_vehicles)

    print("车辆级控制网络测试:")
    print(f"  主路动作形状: {output['main_actions'].shape}")
    print(f"  匝道动作形状: {output['ramp_actions'].shape}")
    print(f"  分流动作形状: {output['diverge_actions'].shape}")
    print(f"  价值形状: {output['value'].shape}")
    print(f"  主路动作示例: {output['main_actions'][0]}")
    print(f"  匝道动作示例: {output['ramp_actions'][0]}")
