"""
车辆级BC训练 - 平衡版本（对control样本过采样）
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
from junction_network import VehicleLevelMultiJunctionModel
from config import NetworkConfig


class BalancedExpertDemoDataset(Dataset):
    """专家演示数据集 - 对control样本过采样"""

    def __init__(self, demonstrations: List[Dict], control_sample_weight: int = 100):
        """
        Args:
            demonstrations: 原始演示数据
            control_sample_weight: 对control样本的过采样倍数
        """
        self.transitions = []
        self.control_sample_weight = control_sample_weight

        # 分离normal和control样本
        normal_transitions = []
        control_transitions = []

        for episode in demonstrations:
            for trans in episode['transitions']:
                action = trans['action_main'][0]
                if action < 0.95:
                    # Control样本（需要减速）
                    control_transitions.append(trans)
                else:
                    # Normal样本（不控制）
                    normal_transitions.append(trans)

        logging.info(f"原始数据: {len(normal_transitions)} normal, {len(control_transitions)} control")

        # 对control样本过采样
        self.transitions = normal_transitions.copy()
        for _ in range(control_sample_weight):
            self.transitions.extend(control_transitions)

        logging.info(f"过采样后: {len(normal_transitions)} normal, {len(control_transitions) * control_sample_weight} control (总样本: {len(self.transitions)})")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]
        state = transition['state']
        action_main = transition['action_main']
        action_ramp = transition['action_ramp']

        # 直接使用连续动作值
        action_main_value = float(action_main[0])
        action_ramp_value = float(action_ramp[0])

        return {
            'state': torch.tensor(state, dtype=torch.float32),
            'action_main': torch.tensor(action_main_value, dtype=torch.float32),
            'action_ramp': torch.tensor(action_ramp_value, dtype=torch.float32),
            'junction_id': transition['junction_id'],
            'vehicle_features': torch.tensor(transition['vehicle_features'], dtype=torch.float32),
            'vehicle_type': transition.get('vehicle_type', 'main')
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
        """计算BC损失（使用MSE损失）"""
        states = batch['state'].to(self.device)
        action_main_values = batch['action_main'].to(self.device)
        action_ramp_values = batch['action_ramp'].to(self.device)
        junction_ids = batch['junction_id']
        vehicle_features = batch['vehicle_features'].to(self.device)
        vehicle_types = batch['vehicle_type']

        observations = {}
        vehicle_observations = {}
        action_targets = {}

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

            if veh_type in ['main', 'ramp', 'diverge']:
                vehicle_observations[junc_id][veh_type].append(vehicle_features[i])
                action_main_value = action_main_values[i]
                action_ramp_value = action_ramp_values[i]
                action_targets[junc_id][veh_type].append((action_main_value, action_ramp_value))

        # 转换为tensors
        for junc_id in observations:
            observations[junc_id] = observations[junc_id][0].unsqueeze(0).to(self.device)

            veh_obs_dict = {}
            for veh_type in ['main', 'ramp', 'diverge']:
                if vehicle_observations[junc_id][veh_type]:
                    # 直接堆叠tensor，不通过numpy转换
                    veh_obs_dict[veh_type] = torch.stack(
                        vehicle_observations[junc_id][veh_type]
                    ).unsqueeze(0).to(self.device)

            vehicle_observations[junc_id] = veh_obs_dict

        # 模型推理
        all_actions, _, _ = self.model(observations, vehicle_observations, deterministic=True)

        # 计算损失
        total_loss = 0.0
        num_samples = 0

        for junc_id, actions in all_actions.items():
            targets = action_targets[junc_id]

            for veh_type in ['main', 'ramp', 'diverge']:
                if f'{veh_type}_actions' in actions and actions[f'{veh_type}_actions'] is not None:
                    pred_actions = actions[f'{veh_type}_actions'][0]  # [N]
                    target_actions = torch.tensor([t[0] for t in targets[veh_type]], dtype=torch.float32).to(self.device)

                    if len(pred_actions) > 0 and len(target_actions) > 0:
                        loss = self.mse_loss(pred_actions, target_actions)
                        total_loss += loss
                        num_samples += 1

        return total_loss, num_samples

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

    pkl_file = os.path.join(demo_dir, 'expert_demonstrations.pkl')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            demonstrations = pickle.load(f)
        logging.info(f"加载了 {len(demonstrations)} 个episodes")
    else:
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
    parser = argparse.ArgumentParser(description='车辆级BC训练 - 平衡版本')
    parser.add_argument('--train-demos', type=str, required=True,
                        help='专家演示数据目录')
    parser.add_argument('--output-dir', type=str, default='bc_checkpoints_vehicle_balanced',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--control-weight', type=int, default=100,
                        help='对control样本的过采样倍数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载数据: {args.train_demos}")
    demonstrations = load_expert_demos(args.train_demos)

    # 创建平衡数据集（对control样本过采样）
    dataset = BalancedExpertDemoDataset(demonstrations, control_sample_weight=args.control_weight)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logging.info(f"训练集大小: {train_size}")
    logging.info(f"验证集大小: {val_size}")

    # 创建模型
    model = VehicleLevelBCModel(JUNCTION_CONFIGS)

    # 创建训练器
    trainer = VehicleLevelBCTrainer(model, device=args.device)

    # 训练
    best_val_loss = float('inf')
    print("=" * 70)
    print("开始训练")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    print("=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终模型: {os.path.join(args.output_dir, 'final_model.pt')}")
    print(f"最佳模型: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
