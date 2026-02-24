"""
车辆级控制网络组件
提供 VehicleLevelJunctionNetwork 供 BC / PPO 复用
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


class VehicleTypeController(nn.Module):
    """单类车辆控制器（main/ramp/diverge）"""

    def __init__(self, state_dim: int, vehicle_feat_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.vehicle_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.vehicle_action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 11)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _encode(self, state: torch.Tensor, vehicles: torch.Tensor) -> torch.Tensor:
        """编码状态与车辆特征，输出每辆车融合特征 [N, 2H]"""
        if vehicles is None or vehicles.numel() == 0:
            return torch.empty(0, 0, device=state.device)

        if vehicles.dim() == 3:
            vehicles = vehicles.squeeze(0)

        state_feat = self.state_encoder(state)  # [1, H]
        veh_feat = self.vehicle_encoder(vehicles)  # [N, H]

        q = veh_feat.unsqueeze(0)
        k = veh_feat.unsqueeze(0)
        v = veh_feat.unsqueeze(0)
        attn_out, _ = self.vehicle_attention(q, k, v)
        attn_out = attn_out.squeeze(0)  # [N, H]

        state_expand = state_feat.expand(attn_out.size(0), -1)
        return torch.cat([state_expand, attn_out], dim=-1)  # [N, 2H]

    def _encode_batched(self, state: torch.Tensor, vehicles: torch.Tensor) -> torch.Tensor:
        """编码批量状态与车辆特征，输出 [B, N, 2H]"""
        # state: [B, state_dim], vehicles: [B, N, feat]
        state_feat = self.state_encoder(state)      # [B, H]
        veh_feat = self.vehicle_encoder(vehicles)   # [B, N, H]

        attn_out, _ = self.vehicle_attention(veh_feat, veh_feat, veh_feat)
        state_expand = state_feat.unsqueeze(1).expand(-1, veh_feat.size(1), -1)
        return torch.cat([state_expand, attn_out], dim=-1)  # [B, N, 2H]

    def act(
        self,
        state: torch.Tensor,
        vehicles: Optional[torch.Tensor],
        deterministic: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        if vehicles is None:
            return {
                'actions': None,
                'action_indices': None,
                'log_probs': None,
                'entropies': None,
                'value': None,
                'logits': None,
            }

        fused = self._encode(state, vehicles)
        if fused.numel() == 0:
            return {
                'actions': None,
                'action_indices': None,
                'log_probs': None,
                'entropies': None,
                'value': None,
                'logits': None,
            }

        logits = self.vehicle_action_head(fused)
        dist = Categorical(logits=logits)

        if deterministic:
            action_idx = torch.argmax(logits, dim=-1)
        else:
            action_idx = dist.sample()

        actions = action_idx.float() / 10.0
        log_probs = dist.log_prob(action_idx)
        entropies = dist.entropy()
        values = self.value_head(fused).squeeze(-1)

        return {
            'actions': actions,
            'action_indices': action_idx,
            'log_probs': log_probs,
            'entropies': entropies,
            'value': values.mean(),
            'logits': logits,
        }

    def evaluate_actions(
        self,
        state: torch.Tensor,
        vehicles: Optional[torch.Tensor],
        action_values: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if vehicles is None or action_values is None:
            zero = torch.tensor(0.0, device=state.device)
            return {
                'log_prob_sum': zero,
                'entropy_mean': zero,
                'value': zero,
            }

        fused = self._encode(state, vehicles)
        if fused.numel() == 0:
            zero = torch.tensor(0.0, device=state.device)
            return {
                'log_prob_sum': zero,
                'entropy_mean': zero,
                'value': zero,
            }

        logits = self.vehicle_action_head(fused)
        dist = Categorical(logits=logits)

        action_idx = torch.clamp((action_values * 10.0).round().long(), 0, 10)
        if action_idx.dim() > 1:
            action_idx = action_idx.squeeze(0)

        log_prob = dist.log_prob(action_idx).sum()
        entropy = dist.entropy().mean()
        value = self.value_head(fused).mean()

        return {
            'log_prob_sum': log_prob,
            'entropy_mean': entropy,
            'value': value,
        }

    def evaluate_actions_batched(
        self,
        state: torch.Tensor,
        vehicles: Optional[torch.Tensor],
        action_values: Optional[torch.Tensor],
        vehicle_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """批量评估动作（向量化）

        Args:
            state: [B, state_dim]
            vehicles: [B, N, feat]
            action_values: [B, N]
            vehicle_mask: [B, N]，1为有效车辆
        """
        if vehicles is None or action_values is None:
            zero = torch.zeros(state.size(0), device=state.device)
            return {
                'log_prob_sum': zero,
                'entropy_mean': zero,
                'value': zero,
            }

        fused = self._encode_batched(state, vehicles)  # [B, N, 2H]
        logits = self.vehicle_action_head(fused)       # [B, N, 11]
        dist = Categorical(logits=logits)

        action_idx = torch.clamp((action_values * 10.0).round().long(), 0, 10)  # [B, N]
        log_prob = dist.log_prob(action_idx)  # [B, N]
        entropy = dist.entropy()              # [B, N]
        value = self.value_head(fused).squeeze(-1)  # [B, N]

        if vehicle_mask is not None:
            mask = vehicle_mask.float()
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            log_prob_sum = (log_prob * mask).sum(dim=1)
            entropy_mean = (entropy * mask).sum(dim=1) / denom
            value_mean = (value * mask).sum(dim=1) / denom
        else:
            log_prob_sum = log_prob.sum(dim=1)
            entropy_mean = entropy.mean(dim=1)
            value_mean = value.mean(dim=1)

        return {
            'log_prob_sum': log_prob_sum,
            'entropy_mean': entropy_mean,
            'value': value_mean,
        }


class VehicleLevelJunctionNetwork(nn.Module):
    """路口级车辆控制网络"""

    def __init__(self, state_dim: int, vehicle_feat_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.main_controller = VehicleTypeController(state_dim, vehicle_feat_dim, hidden_dim)
        self.ramp_controller = VehicleTypeController(state_dim, vehicle_feat_dim, hidden_dim)
        self.diverge_controller = VehicleTypeController(state_dim, vehicle_feat_dim, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        main_vehicles: Optional[torch.Tensor] = None,
        ramp_vehicles: Optional[torch.Tensor] = None,
        diverge_vehicles: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        main_out = self.main_controller.act(state, main_vehicles, deterministic)
        ramp_out = self.ramp_controller.act(state, ramp_vehicles, deterministic)
        diverge_out = self.diverge_controller.act(state, diverge_vehicles, deterministic)

        values = [v for v in [main_out['value'], ramp_out['value'], diverge_out['value']] if v is not None]
        if values:
            joint_value = torch.stack(values).mean()
        else:
            joint_value = torch.tensor(0.0, device=state.device)

        return {
            'main_actions': main_out['actions'],
            'ramp_actions': ramp_out['actions'],
            'diverge_actions': diverge_out['actions'],
            'main_log_probs': main_out['log_probs'],
            'ramp_log_probs': ramp_out['log_probs'],
            'diverge_log_probs': diverge_out['log_probs'],
            'main_entropies': main_out['entropies'],
            'ramp_entropies': ramp_out['entropies'],
            'diverge_entropies': diverge_out['entropies'],
            'main_logits': main_out['logits'],
            'ramp_logits': ramp_out['logits'],
            'diverge_logits': diverge_out['logits'],
            'value': joint_value,
        }

    def evaluate_actions(
        self,
        state: torch.Tensor,
        vehicle_observations: Dict[str, torch.Tensor],
        action_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        main_eval = self.main_controller.evaluate_actions(
            state,
            vehicle_observations.get('main'),
            action_dict.get('main_actions')
        )
        ramp_eval = self.ramp_controller.evaluate_actions(
            state,
            vehicle_observations.get('ramp'),
            action_dict.get('ramp_actions')
        )
        diverge_eval = self.diverge_controller.evaluate_actions(
            state,
            vehicle_observations.get('diverge'),
            action_dict.get('diverge_actions')
        )

        log_prob_sum = main_eval['log_prob_sum'] + ramp_eval['log_prob_sum'] + diverge_eval['log_prob_sum']
        entropy_mean = torch.stack([
            main_eval['entropy_mean'],
            ramp_eval['entropy_mean'],
            diverge_eval['entropy_mean']
        ]).mean()
        value = torch.stack([
            main_eval['value'],
            ramp_eval['value'],
            diverge_eval['value']
        ]).mean()

        return {
            'log_prob_sum': log_prob_sum,
            'entropy_mean': entropy_mean,
            'value': value,
        }

    def evaluate_actions_batched(
        self,
        state: torch.Tensor,
        vehicle_observations: Dict[str, torch.Tensor],
        action_dict: Dict[str, torch.Tensor],
        mask_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """批量评估路口动作

        Args:
            state: [B, state_dim]
            vehicle_observations: {'main': [B,N,8], ...}
            action_dict: {'main_actions': [B,N], ...}
            mask_dict: {'main': [B,N], ...}
        """
        mask_dict = mask_dict or {}

        main_eval = self.main_controller.evaluate_actions_batched(
            state,
            vehicle_observations.get('main'),
            action_dict.get('main_actions'),
            mask_dict.get('main')
        )
        ramp_eval = self.ramp_controller.evaluate_actions_batched(
            state,
            vehicle_observations.get('ramp'),
            action_dict.get('ramp_actions'),
            mask_dict.get('ramp')
        )
        diverge_eval = self.diverge_controller.evaluate_actions_batched(
            state,
            vehicle_observations.get('diverge'),
            action_dict.get('diverge_actions'),
            mask_dict.get('diverge')
        )

        log_prob_sum = main_eval['log_prob_sum'] + ramp_eval['log_prob_sum'] + diverge_eval['log_prob_sum']
        entropy_mean = (main_eval['entropy_mean'] + ramp_eval['entropy_mean'] + diverge_eval['entropy_mean']) / 3.0
        value = (main_eval['value'] + ramp_eval['value'] + diverge_eval['value']) / 3.0

        return {
            'log_prob_sum': log_prob_sum,
            'entropy_mean': entropy_mean,
            'value': value,
        }
