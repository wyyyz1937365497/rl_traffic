"""
行为克隆 (Behavior Cloning) 训练脚本

从专家演示数据中学习策略
"""
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import argparse
import logging
from tqdm import tqdm

# 导入必要的模块
sys.path.insert(0, '.')

from junction_agent import JUNCTION_CONFIGS
from junction_network import create_junction_model, NetworkConfig


# ============================================================================
# 数据集定义
# ============================================================================

class ExpertDemoDataset(Dataset):
    """专家演示数据集"""

    def __init__(self, demonstrations: List[Dict]):
        """
        Args:
            demonstrations: List of episode数据，每个episode包含多个transitions
        """
        self.transitions = []

        # 展平所有episodes的transitions
        for episode in demonstrations:
            self.transitions.extend(episode['transitions'])

        logging.info(f"数据集加载完成: {len(self.transitions)} 个transitions")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        """
        返回:
            state: numpy array, shape=[state_dim]
            action_main_class: int, 类别索引 (0-10)
            action_ramp_class: int, 类别索引 (0-10)
        """
        transition = self.transitions[idx]

        state = transition['state']

        # 将连续动作转换为类别索引 (0-10)
        # 动作范围 [0, 1] -> 类别 [0, 1, ..., 10]
        action_main = transition['action_main']
        action_ramp = transition['action_ramp']

        # 转换为类别：action * 10，然后四舍五入到最近的整数
        action_main_class = int(np.round(action_main[0] * 10))
        action_ramp_class = int(np.round(action_ramp[0] * 10))

        # 裁剪到 [0, 10] 范围
        action_main_class = np.clip(action_main_class, 0, 10)
        action_ramp_class = np.clip(action_ramp_class, 0, 10)

        return {
            'state': torch.tensor(state, dtype=torch.float32),
            'action_main_class': torch.tensor(action_main_class, dtype=torch.long),
            'action_ramp_class': torch.tensor(action_ramp_class, dtype=torch.long),
            'junction_id': transition['junction_id']
        }


# ============================================================================
# 行为克隆模型
# ============================================================================

class BehaviorCloningTrainer:
    """行为克隆训练器"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

        # 优化器（提高学习率）
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # 损失函数（交叉熵 for 分类）
        self.ce_loss = nn.CrossEntropyLoss()

        logging.info(f"训练器初始化完成 (device={device})")

    def compute_bc_loss(self, batch, junction_id_mapping):
        """
        计算行为克隆损失（使用交叉熵）

        Args:
            batch: 包含 state, action_main_class, action_ramp_class, junction_id 的批次
            junction_id_mapping: {junc_id: model_index} 映射

        Returns:
            loss, metrics
        """
        states = batch['state'].to(self.device)  # [batch_size, state_dim]
        action_main_classes = batch['action_main_class']  # [batch_size]
        action_ramp_classes = batch['action_ramp_class']  # [batch_size]
        junction_ids = batch['junction_id']

        # 按路口分组
        obs_tensors = {}
        action_targets = {}

        for i, junc_id in enumerate(junction_ids):
            junc_str = str(junc_id)  # 确保是字符串
            if junc_str not in obs_tensors:
                obs_tensors[junc_str] = []
                action_targets[junc_str] = {'main': [], 'ramp': []}

            obs_tensors[junc_str].append(states[i])
            action_targets[junc_str]['main'].append(action_main_classes[i])
            action_targets[junc_str]['ramp'].append(action_ramp_classes[i])

        # 转换为tensors
        for junc_id in list(obs_tensors.keys()):
            if len(obs_tensors[junc_id]) == 0:
                continue
            obs_tensors[junc_id] = torch.stack(obs_tensors[junc_id])
            action_targets[junc_id]['main'] = torch.stack(action_targets[junc_id]['main']).to(self.device)
            action_targets[junc_id]['ramp'] = torch.stack(action_targets[junc_id]['ramp']).to(self.device)

        # 计算损失
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        main_losses = []
        ramp_losses = []
        main_correct = 0
        ramp_correct = 0
        total_samples = 0

        # 为每个路口单独调用模型
        for junc_id, target_actions in action_targets.items():
            # 单路口的观测
            single_junc_obs = {junc_id: obs_tensors[junc_id]}

            # 获取网络实例
            network = self.model.adaptive_network.networks[junc_id]

            # 前向传播获取 logits（不需要 junc_id 参数）
            output = network(
                obs_tensors[junc_id],
                main_vehicles=None,
                ramp_vehicles=None,
                diverge_vehicles=None
            )

            # output['main_action'] 和 output['ramp_action'] 是 logits [batch, 11]
            main_logits = output['main_action']  # [n, 11]
            ramp_logits = output['ramp_action']  # [n, 11]

            target_main = target_actions['main']  # [n]
            target_ramp = target_actions['ramp']  # [n]

            # 交叉熵损失
            main_loss = self.ce_loss(main_logits, target_main)
            ramp_loss = self.ce_loss(ramp_logits, target_ramp)

            loss = main_loss + ramp_loss
            total_loss = total_loss + loss

            main_losses.append(main_loss.item())
            ramp_losses.append(ramp_loss.item())

            # 计算准确率
            main_pred = torch.argmax(main_logits, dim=-1)
            ramp_pred = torch.argmax(ramp_logits, dim=-1)

            main_correct += (main_pred == target_main).sum().item()
            ramp_correct += (ramp_pred == target_ramp).sum().item()
            total_samples += target_main.size(0)

        if len(main_losses) == 0:
            return torch.tensor(0.0, requires_grad=True), {}

        avg_loss = total_loss / len(action_targets)

        # 计算总准确率
        main_acc = main_correct / max(total_samples, 1)
        ramp_acc = ramp_correct / max(total_samples, 1)

        metrics = {
            'loss': avg_loss.item(),
            'main_loss': np.mean(main_losses),
            'ramp_loss': np.mean(ramp_losses),
            'main_acc': main_acc,
            'ramp_acc': ramp_acc
        }

        return avg_loss, metrics

    def train_epoch(self, dataloader, junction_id_mapping):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_main_loss = 0.0
        total_ramp_loss = 0.0
        total_main_acc = 0.0
        total_ramp_acc = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()

            # 计算损失
            loss, metrics = self.compute_bc_loss(batch, junction_id_mapping)

            if loss.item() == 0.0:
                continue

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += metrics['loss']
            total_main_loss += metrics['main_loss']
            total_ramp_loss += metrics['ramp_loss']
            total_main_acc += metrics['main_acc']
            total_ramp_acc += metrics['ramp_acc']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'main_acc': f"{metrics['main_acc']:.2%}",
                'ramp_acc': f"{metrics['ramp_acc']:.2%}"
            })

        if num_batches == 0:
            return {'loss': 0.0, 'main_loss': 0.0, 'ramp_loss': 0.0, 'main_acc': 0.0, 'ramp_acc': 0.0}

        return {
            'loss': total_loss / num_batches,
            'main_loss': total_main_loss / num_batches,
            'ramp_loss': total_ramp_loss / num_batches,
            'main_acc': total_main_acc / num_batches,
            'ramp_acc': total_ramp_acc / num_batches,
        }

    def evaluate(self, dataloader, junction_id_mapping):
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        total_main_loss = 0.0
        total_ramp_loss = 0.0
        total_main_acc = 0.0
        total_ramp_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                loss, metrics = self.compute_bc_loss(batch, junction_id_mapping)

                if loss.item() == 0.0:
                    continue

                total_loss += metrics['loss']
                total_main_loss += metrics['main_loss']
                total_ramp_loss += metrics['ramp_loss']
                total_main_acc += metrics['main_acc']
                total_ramp_acc += metrics['ramp_acc']
                num_batches += 1

        if num_batches == 0:
            return {'loss': 0.0, 'main_loss': 0.0, 'ramp_loss': 0.0, 'main_acc': 0.0, 'ramp_acc': 0.0}

        return {
            'loss': total_loss / num_batches,
            'main_loss': total_main_loss / num_batches,
            'ramp_loss': total_ramp_loss / num_batches,
            'main_acc': total_main_acc / num_batches,
            'ramp_acc': total_ramp_acc / num_batches,
        }

    def save_checkpoint(self, filepath, epoch, metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }

        torch.save(checkpoint, filepath)
        logging.info(f"检查点已保存: {filepath}")


# ============================================================================
# 训练流程
# ============================================================================

def train_behavior_cloning(
    demo_file: str,
    output_dir: str = "bc_checkpoints",
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    训练行为克隆模型

    Args:
        demo_file: 专家演示数据文件路径
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 设备
    """
    logging.info("=" * 70)
    logging.info("行为克隆训练")
    logging.info("=" * 70)
    logging.info(f"演示数据: {demo_file}")
    logging.info(f"训练轮数: {num_epochs}")
    logging.info(f"批次大小: {batch_size}")
    logging.info(f"学习率: {learning_rate}")
    logging.info(f"设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载演示数据
    logging.info(f"\n加载演示数据...")
    with open(demo_file, 'rb') as f:
        demonstrations = pickle.load(f)

    logging.info(f"成功加载 {len(demonstrations)} 个episodes")

    # 创建数据集
    dataset = ExpertDemoDataset(demonstrations)

    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logging.info(f"训练集: {train_size} samples")
    logging.info(f"验证集: {val_size} samples")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    logging.info(f"\n创建模型...")
    model = create_junction_model(JUNCTION_CONFIGS, NetworkConfig())

    # 创建训练器
    trainer = BehaviorCloningTrainer(model, device=device)

    # 路口ID映射
    junction_id_mapping = {junc_id: i for i, junc_id in enumerate(JUNCTION_CONFIGS.keys())}

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # 训练
        train_metrics = trainer.train_epoch(train_loader, junction_id_mapping)
        train_losses.append(train_metrics['loss'])

        logging.info(f"训练损失: {train_metrics['loss']:.4f} "
                    f"(main={train_metrics['main_loss']:.4f}, "
                    f"ramp={train_metrics['ramp_loss']:.4f})")
        logging.info(f"训练准确率: main={train_metrics['main_acc']:.2%}, "
                    f"ramp={train_metrics['ramp_acc']:.2%}")

        # 验证
        val_metrics = trainer.evaluate(val_loader, junction_id_mapping)
        val_losses.append(val_metrics['loss'])

        logging.info(f"验证损失: {val_metrics['loss']:.4f} "
                    f"(main={val_metrics['main_loss']:.4f}, "
                    f"ramp={val_metrics['ramp_loss']:.4f})")
        logging.info(f"验证准确率: main={val_metrics['main_acc']:.2%}, "
                    f"ramp={val_metrics['ramp_acc']:.2%}")

        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            trainer.save_checkpoint(best_model_path, epoch, val_metrics)
            logging.info(f"[OK] 保存最佳模型 (val_loss={best_val_loss:.4f})")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)

    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    trainer.save_checkpoint(final_model_path, num_epochs, val_losses[-1])

    logging.info(f"\n{'='*70}")
    logging.info(f"训练完成!")
    logging.info(f"{'='*70}")
    logging.info(f"最佳验证损失: {best_val_loss:.4f}")
    logging.info(f"最终模型: {final_model_path}")
    logging.info(f"最佳模型: {os.path.join(output_dir, 'best_model.pt')}")

    return model, best_val_loss


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='行为克隆训练')
    parser.add_argument('--demo-file', type=str, required=True,
                        help='专家演示数据文件路径')
    parser.add_argument('--output-dir', type=str, default='bc_checkpoints',
                        help='输出目录')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # 检查演示文件
    if not os.path.exists(args.demo_file):
        logging.error(f"演示数据文件不存在: {args.demo_file}")
        return

    # 训练
    train_behavior_cloning(
        demo_file=args.demo_file,
        output_dir=args.output_dir,
        num_epochs=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
