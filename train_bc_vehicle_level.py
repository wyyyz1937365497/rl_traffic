"""
使用车辆级控制网络的BC训练脚本
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
from junction_network import NetworkConfig, VehicleLevelMultiJunctionModel


class ExpertDemoDataset(Dataset):
    """专家演示数据集"""

    def __init__(self, demonstrations: List[Dict]):
        self.transitions = []
        for episode in demonstrations:
            self.transitions.extend(episode['transitions'])
        logging.info(f"数据集加载完成: {len(self.transitions)} 个transitions")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]
        state = transition['state']
        action_main = transition['action_main']
        action_ramp = transition['action_ramp']

        # 直接使用连续动作值，不进行离散化
        action_main_value = float(action_main[0])
        action_ramp_value = float(action_ramp[0])

        return {
            'state': torch.tensor(state, dtype=torch.float32),
            'action_main': torch.tensor(action_main_value, dtype=torch.float32),
            'action_ramp': torch.tensor(action_ramp_value, dtype=torch.float32),
            'junction_id': transition['junction_id'],
            'vehicle_features': torch.tensor(transition['vehicle_features'], dtype=torch.float32),
            'vehicle_type': transition.get('vehicle_type', 'main')  # main/ramp/diverge
        }


class VehicleLevelBCModel(nn.Module):
    """车辆级BC模型包装器"""

    def __init__(self, junction_configs: Dict):
        super().__init__()
        self.network = VehicleLevelMultiJunctionModel(
            junction_configs=junction_configs,
            config=NetworkConfig()
        )

    def forward(self, observations, vehicle_observations, deterministic=False):
        return self.network(observations, vehicle_observations, deterministic)


class VehicleLevelBCTrainer:
    """车辆级行为克隆训练器"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.mse_loss = nn.MSELoss()
        logging.info(f"训练器初始化完成 (device={device})")

    def compute_bc_loss(self, batch):
        """计算BC损失（使用MSE损失，因为网络输出连续动作）"""
        states = batch['state'].to(self.device)
        action_main_values = batch['action_main'].to(self.device)  # 连续值
        action_ramp_values = batch['action_ramp'].to(self.device)  # 连续值
        junction_ids = batch['junction_id']
        vehicle_features = batch['vehicle_features'].to(self.device)
        vehicle_types = batch['vehicle_type']

        # 车辆级网络需要：每个路口的所有车辆一起输入，并按类型分组
        observations = {}  # {junc_id: tensor[batch, N, state_dim]}
        vehicle_observations = {}  # {junc_id: {'main': [batch, N_main, 8], 'ramp': [...], 'diverge': [...]}}
        action_targets = {}  # {junc_id: {'main': tensor, 'ramp': tensor, ...}}

        batch_size = states.size(0)

        for i in range(batch_size):
            junc_id = junction_ids[i]
            if isinstance(junc_id, torch.Tensor):
                junc_id = junc_id.item()
            junc_id = str(junc_id)

            veh_type = vehicle_types[i]
            if isinstance(veh_type, torch.Tensor):
                veh_type = veh_type.item()

            if junc_id not in observations:
                observations[junc_id] = []
                vehicle_observations[junc_id] = {'main': [], 'ramp': [], 'diverge': []}
                action_targets[junc_id] = {'main': [], 'ramp': [], 'diverge': []}

            observations[junc_id].append(states[i])

            # 按车辆类型分组
            if veh_type in ['main', 'ramp', 'diverge']:
                vehicle_observations[junc_id][veh_type].append(vehicle_features[i])

                # 直接使用连续动作值
                action_main_value = action_main_values[i]
                action_ramp_value = action_ramp_values[i]

                action_targets[junc_id][veh_type].append((action_main_value, action_ramp_value))

        # 转换为tensors
        for junc_id in observations:
            # 状态向量：使用第一个状态作为全局状态 [1, state_dim]
            # 注意：车辆级网络期望一个全局状态，所有车辆共享
            # 这里的简化是：一个batch中的同一路口车辆共享状态
            observations[junc_id] = observations[junc_id][0].unsqueeze(0).to(self.device)

            # 车辆特征：按类型分组 [1, N, 8]
            veh_obs_dict = {}
            for veh_type in ['main', 'ramp', 'diverge']:
                if vehicle_observations[junc_id][veh_type]:
                    veh_obs_dict[veh_type] = torch.stack(vehicle_observations[junc_id][veh_type]).unsqueeze(0).to(self.device)
                else:
                    veh_obs_dict[veh_type] = None
            vehicle_observations[junc_id] = veh_obs_dict

            # 动作目标：按类型分组
            action_dict = {}
            for veh_type in ['main', 'ramp', 'diverge']:
                if action_targets[junc_id][veh_type]:
                    actions = action_targets[junc_id][veh_type]
                    main_actions = torch.tensor([a[0] for a in actions], dtype=torch.float32).unsqueeze(0).to(self.device)
                    ramp_actions = torch.tensor([a[1] for a in actions], dtype=torch.float32).unsqueeze(0).to(self.device)
                    action_dict[veh_type] = {'main': main_actions, 'ramp': ramp_actions}
                else:
                    action_dict[veh_type] = None
            action_targets[junc_id] = action_dict

        # 前向传播
        all_actions, _, _ = self.model.network(observations, vehicle_observations, deterministic=False)

        # 计算MSE损失
        losses = []
        total_samples = 0

        for junc_id, output in all_actions.items():
            if junc_id not in action_targets:
                continue

            junc_targets = action_targets[junc_id]

            # main actions
            if 'main_actions' in output and output['main_actions'] is not None:
                pred_main = output['main_actions']  # [1, N]

                # 找到对应的target
                for veh_type in ['main', 'ramp', 'diverge']:
                    if junc_targets[veh_type] is not None:
                        target_main = junc_targets[veh_type]['main']  # [1, N_type]
                        if pred_main.size(1) == target_main.size(1):
                            main_loss = self.mse_loss(pred_main, target_main)
                            losses.append(main_loss)
                            total_samples += target_main.size(1)
                            break

            # ramp actions
            if 'ramp_actions' in output and output['ramp_actions'] is not None:
                pred_ramp = output['ramp_actions']  # [1, N]

                for veh_type in ['main', 'ramp', 'diverge']:
                    if junc_targets[veh_type] is not None:
                        target_ramp = junc_targets[veh_type]['ramp']  # [1, N_type]
                        if pred_ramp.size(1) == target_ramp.size(1):
                            ramp_loss = self.mse_loss(pred_ramp, target_ramp)
                            losses.append(ramp_loss)
                            total_samples += target_ramp.size(1)
                            break

        # 汇总损失
        if losses:
            total_loss = torch.stack(losses).sum()
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss, total_samples

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            self.optimizer.zero_grad()

            loss, num_samples = self.compute_bc_loss(batch)

            if num_samples > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, num_samples = self.compute_bc_loss(batch)
                if num_samples > 0:
                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss


def load_expert_demos(demo_dir: str) -> List[Dict]:
    """加载专家演示数据"""
    demonstrations = []

    # 加载pickle文件
    pkl_file = os.path.join(demo_dir, 'expert_demonstrations.pkl')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            demonstrations = pickle.load(f)
        logging.info(f"加载了 {len(demonstrations)} 个episodes")
    else:
        # 尝试加载目录中的多个pkl文件
        import glob
        pkl_files = glob.glob(os.path.join(demo_dir, '*.pkl'))
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    demonstrations.extend(data)
                else:
                    demonstrations.append(data)
        logging.info(f"加载了 {len(demonstrations)} 个episodes (从多个文件)")

    return demonstrations


def main():
    parser = argparse.ArgumentParser(description='车辆级BC训练')
    parser.add_argument('--train-demos', type=str, required=True,
                        help='专家演示数据目录')
    parser.add_argument('--output-dir', type=str, default='bc_checkpoints_vehicle',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print(f"加载数据: {args.train_demos}")
    demonstrations = load_expert_demos(args.train_demos)

    # 创建数据集和dataloader
    dataset = ExpertDemoDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"训练集: {len(train_dataset)} transitions")
    print(f"验证集: {len(val_dataset)} transitions")

    # 创建模型（车辆级）
    model = VehicleLevelBCModel(JUNCTION_CONFIGS)

    # 创建训练器
    trainer = VehicleLevelBCTrainer(model, device=args.device)

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = trainer.train_epoch(train_loader, epoch)

        # 验证
        val_loss = trainer.evaluate(val_loader)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")

        # 保存checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_path)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终模型: {final_path}")
    print(f"最佳模型: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
