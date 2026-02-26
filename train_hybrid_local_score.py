"""
混合训练脚本：BC初始化 + 本地26分评分对齐微调

阶段1: 使用BC checkpoint初始化
阶段2: 使用本地评分公式为主奖励进行PPO微调，并保留少量原始奖励塑形
"""

import argparse
import logging
import os
import json
import numpy as np
import torch

from junction_trainer import PPOConfig
from local_score_calculator import LocalScoreCalculator
from train_ppo_finetune import PPOFinetuner


def resolve_training_seed(seed_arg: int, seed_base: int, seed_registry: str) -> int:
    """解析训练种子。

    规则：
    - seed_arg >= 0: 使用用户显式指定种子
    - seed_arg < 0 : 自动分配种子（基于seed_base + 运行计数），并写入registry保证可复现
    """
    if seed_arg >= 0:
        return int(seed_arg)

    os.makedirs(os.path.dirname(seed_registry) or '.', exist_ok=True)

    if os.path.exists(seed_registry):
        with open(seed_registry, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {'seed_base': int(seed_base), 'next_offset': 0, 'history': []}

    if int(data.get('seed_base', seed_base)) != int(seed_base):
        data['seed_base'] = int(seed_base)
        data['next_offset'] = 0

    offset = int(data.get('next_offset', 0))
    seed = int(seed_base) + offset
    data['next_offset'] = offset + 1
    data.setdefault('history', []).append({'offset': offset, 'seed': seed})

    with open(seed_registry, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return seed


class HybridLocalScoreFinetuner(PPOFinetuner):
    """本地评分对齐的混合PPO微调器"""

    def __init__(
        self,
        *args,
        baseline_pkl: str = None,
        local_score_weight: float = 2.0,
        shaping_weight: float = 0.4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.local_score_weight = float(local_score_weight)
        self.shaping_weight = float(shaping_weight)

        score_calc = LocalScoreCalculator(baseline_pkl)
        self.local_baseline = score_calc.baseline_metrics
        self.prev_local_score = 0.0

        self.W_efficiency = 0.5
        self.W_stability = 0.5
        self.k = 10.0

        logging.info(f"[混合奖励] local_score_weight={self.local_score_weight}")
        logging.info(f"[混合奖励] shaping_weight={self.shaping_weight}")
        logging.info(f"[混合奖励] baseline_ocr={self.local_baseline['ocr']:.4f}")
        logging.info(f"[混合奖励] baseline_speed_std={self.local_baseline['speed_std']:.4f}")
        logging.info(f"[混合奖励] baseline_mean_abs_accel={self.local_baseline['mean_abs_accel']:.4f}")

    def collect_episode(self, env, max_steps: int = 3600):
        self.prev_local_score = 0.0
        return super().collect_episode(env, max_steps)

    def _estimate_local_score(self, step_info):
        speeds = step_info.get('speeds', [])
        accelerations = step_info.get('accelerations', [])

        num_departed = max(step_info.get('num_departed', 0), 1)
        num_arrived = step_info.get('num_arrived', 0)

        current_ocr = num_arrived / num_departed
        baseline_ocr = max(self.local_baseline.get('ocr', 0.94), 1e-6)

        s_efficiency = 100.0 * max(0.0, (current_ocr - baseline_ocr) / baseline_ocr)

        if len(speeds) > 1:
            speed_std = float(np.std(speeds))
        else:
            speed_std = self.local_baseline.get('speed_std', 8.0)

        if len(accelerations) > 0:
            mean_abs_accel = float(np.mean(np.abs(accelerations)))
        else:
            mean_abs_accel = self.local_baseline.get('mean_abs_accel', 1.2)

        baseline_speed_std = max(self.local_baseline.get('speed_std', 8.0), 1e-6)
        baseline_abs_accel = max(self.local_baseline.get('mean_abs_accel', 1.2), 1e-6)

        i_speed_std = -(speed_std - baseline_speed_std) / baseline_speed_std
        i_abs_accel = -(mean_abs_accel - baseline_abs_accel) / baseline_abs_accel

        s_stability = 100.0 * (0.4 * max(0.0, i_speed_std) + 0.6 * max(0.0, i_abs_accel))

        control_effort = max(0.0, float(step_info.get('control_effort', 0.0)))
        c_int = control_effort
        p_int = float(np.exp(-self.k * c_int))

        local_score = (self.W_efficiency * s_efficiency + self.W_stability * s_stability) * p_int

        return {
            'local_score': float(local_score),
            's_efficiency': float(s_efficiency),
            's_stability': float(s_stability),
            'intervention_penalty': float(p_int),
            'control_effort': float(control_effort),
            'ocr': float(current_ocr),
            'speed_std': float(speed_std),
            'mean_abs_accel': float(mean_abs_accel),
        }

    def compute_reward_with_breakdown(self, step_info):
        base_reward, base_breakdown = super().compute_reward_with_breakdown(step_info)
        local_metrics = self._estimate_local_score(step_info)

        local_delta = local_metrics['local_score'] - self.prev_local_score
        self.prev_local_score = local_metrics['local_score']

        mixed_reward = self.local_score_weight * local_delta + self.shaping_weight * base_reward

        breakdown = dict(base_breakdown)
        breakdown.update({
            'local_score': local_metrics['local_score'],
            'local_score_delta': local_delta,
            'local_efficiency_score': local_metrics['s_efficiency'],
            'local_stability_score': local_metrics['s_stability'],
            'local_intervention_penalty': local_metrics['intervention_penalty'],
            'control_effort': local_metrics['control_effort'],
            'mixed_reward': mixed_reward,
        })

        return mixed_reward, breakdown


def main():
    parser = argparse.ArgumentParser(description='混合训练：BC初始化 + 本地26分对齐PPO微调')
    parser.add_argument('--bc-checkpoint', type=str, required=True, help='BC模型checkpoint路径')
    parser.add_argument('--baseline-pkl', type=str, default='', help='baseline pkl路径（用于本地评分基线）')
    parser.add_argument('--log-dir', type=str, default='./logs/ppo_hybrid_local_score', help='日志目录')
    parser.add_argument('--episodes', type=int, default=100, help='训练episodes')
    parser.add_argument('--max-steps', type=int, default=3600, help='每回合最大步数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=-1, help='随机种子；<0时自动分配')
    parser.add_argument('--seed-base', type=int, default=20260224, help='自动种子起点')
    parser.add_argument('--seed-registry', type=str, default='logs/seed_registry_hybrid.json', help='自动种子记录文件')

    parser.add_argument('--local-score-weight', type=float, default=2.0, help='本地评分增量奖励权重')
    parser.add_argument('--shaping-weight', type=float, default=0.4, help='原始奖励塑形权重')

    parser.add_argument('--anchor-coef', type=float, default=1e-5, help='BC参数锚定正则系数')
    parser.add_argument('--anchor-decay', type=float, default=0.999, help='BC参数锚定正则衰减')
    parser.add_argument('--early-stop-patience', type=int, default=15, help='早停耐心值')

    args = parser.parse_args()

    train_seed = resolve_training_seed(args.seed, args.seed_base, args.seed_registry)
    print(f"[Seed] 使用训练种子: {train_seed}")

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    config = PPOConfig(
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.1,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        batch_size=2048,
        n_epochs=8,
        update_frequency=2048,
        seed=train_seed
    )

    finetuner = HybridLocalScoreFinetuner(
        bc_checkpoint_path=args.bc_checkpoint,
        baseline_pkl=args.baseline_pkl if args.baseline_pkl else None,
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        anchor_coef=args.anchor_coef,
        anchor_decay=args.anchor_decay,
        early_stop_patience=args.early_stop_patience,
        local_score_weight=args.local_score_weight,
        shaping_weight=args.shaping_weight,
    )

    finetuner.train(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
    )


if __name__ == '__main__':
    main()
