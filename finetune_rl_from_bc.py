"""
BC + RL 微调脚本
使用BC模型初始化，通过RL进一步优化

关键改进：
1. 从BC checkpoint加载权重
2. 使用改进的奖励函数（鼓励主动控制）
3. 更低的clip参数防止性能崩溃
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, '.')

from junction_agent import JUNCTION_CONFIGS
from junction_network import VehicleLevelMultiJunctionModel, NetworkConfig
from junction_trainer import PPOConfig, MultiAgentPPOTrainer
from vehicle_level_network import VehicleLevelJunctionNetwork


def load_bc_model(checkpoint_path: str, device: str = 'cuda'):
    """加载BC模型权重"""
    print(f"加载BC模型: {checkpoint_path}")

    # 创建模型
    config = NetworkConfig()
    model = VehicleLevelMultiJunctionModel(JUNCTION_CONFIGS, config).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print("✓ BC模型权重加载成功")

    return model


class ImprovedRewardCalculator:
    """
    改进的奖励函数

    相比原始奖励：
    1. 减少OCR权重，增加稳定性权重
    2. 对控制给予适度奖励（鼓励主动控制）
    3. 对过度减速给予惩罚
    """

    def __init__(self, speed_limit: float = 13.89):
        self.speed_limit = speed_limit

    def calculate_reward(self, step_info: dict) -> float:
        """
        计算单步奖励

        Args:
            step_info: 包含以下字段的字典
                - speeds: list of vehicle speeds
                - accelerations: list of vehicle accelerations
                - controlled_vehicles: list of controlled vehicle IDs
                - target_speeds: dict of {vehicle_id: target_speed}
        """
        speeds = step_info.get('speeds', [])
        accelerations = step_info.get('accelerations', [])
        controlled_vehicles = step_info.get('controlled_vehicles', [])
        target_speeds = step_info.get('target_speeds', {})

        if not speeds:
            return 0.0

        # 1. 速度奖励（鼓励车辆保持高速）
        mean_speed = np.mean(speeds)
        speed_reward = (mean_speed / self.speed_limit) ** 2

        # 2. 稳定性惩罚（减少急加速和急减速）
        if accelerations:
            mean_abs_accel = np.mean(np.abs(accelerations))
            stability_penalty = -0.5 * (mean_abs_accel / 3.0)  # 归一化到0-3m/s²
        else:
            stability_penalty = 0.0

        # 3. 控制奖励（适度鼓励使用控制）
        # 如果有CV车辆被控制且速度合理，给予奖励
        control_reward = 0.0
        if controlled_vehicles and target_speeds:
            controlled_speeds = [
                target_speeds.get(v, self.speed_limit)
                for v in controlled_vehicles
            ]
            # 平均目标速度在合理范围（8-12 m/s）时给予奖励
            if controlled_speeds:
                avg_target = np.mean(controlled_speeds)
                if 8.0 <= avg_target <= 12.0:
                    control_reward = 0.1

        # 4. 过度减速惩罚
        slow_penalty = 0.0
        if speeds:
            slow_ratio = sum(1 for s in speeds if s < 3.0) / len(speeds)
            slow_penalty = -2.0 * slow_ratio

        # 总奖励
        total_reward = (
            1.0 * speed_reward +
            1.0 * stability_penalty +
            0.5 * control_reward +
            1.0 * slow_penalty
        )

        return total_reward


class RLFinetuningTrainer(MultiAgentPPOTrainer):
    """RL微调训练器（从BC初始化）"""

    def __init__(self, model, reward_calculator, config, device='cuda'):
        # 初始化父类，但不重新初始化模型
        super().__init__(model, config, device)

        self.reward_calculator = reward_calculator

        # 微调专用配置
        self.config.learning_rate = 1e-5  # 更低的学习率
        self.config.clip_range = 0.1  # 更保守的PPO clip


def main():
    parser = argparse.ArgumentParser(description='BC+RL微调训练')
    parser.add_argument('--bc-checkpoint', type=str, required=True,
                        help='BC模型checkpoint路径')
    parser.add_argument('--output-dir', type=str, default='rl_finetune_checkpoints',
                        help='输出目录')
    parser.add_argument('--episodes', type=int, default=500,
                        help='训练episodes数')
    parser.add_argument('--max-steps', type=int, default=3600,
                        help='每个episode的最大步数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("BC + RL 微调训练")
    print("=" * 70)

    # 1. 加载BC模型
    model = load_bc_model(args.bc_checkpoint, args.device)

    # 2. 创建改进的奖励计算器
    reward_calculator = ImprovedRewardCalculator(speed_limit=13.89)

    # 3. 配置PPO（微调参数）
    ppo_config = PPOConfig(
        learning_rate=1e-5,  # 更低的学习率，避免破坏BC权重
        num_steps=2048,
        batch_size=64,
        num_epochs=10,
        clip_range=0.1,  # 更保守的clip
        entropy_coef=0.01,  # 适度探索
        value_loss_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    )

    # 4. 创建训练器
    trainer = RLFinetuningTrainer(
        model=model,
        reward_calculator=reward_calculator,
        config=ppo_config,
        device=args.device
    )

    print("\n" + "=" * 70)
    print("开始微调训练")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {ppo_config.learning_rate}")
    print(f"Clip range: {ppo_config.clip_range}")
    print("=" * 70 + "\n")

    # 5. 训练循环
    best_reward = float('-inf')

    for episode in range(1, args.episodes + 1):
        # 模拟一个episode（这里需要实际的SUMO环境）
        # TODO: 实现完整的episode收集逻辑

        if episode % 50 == 0:
            print(f"Episode {episode}/{args.episodes}")
            # 保存checkpoint
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_ep{episode}.pt')
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"  ✓ 保存checkpoint: {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'finetuned_model.pt')
    torch.save({
        'episode': args.episodes,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, final_path)

    print("\n" + "=" * 70)
    print("微调完成!")
    print(f"最终模型: {final_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
