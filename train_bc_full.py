"""
完整BC训练脚本 - 带详细日志

基于train_bc_vehicle_level_balanced.py，添加详细的训练监控
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
from datetime import datetime
from tqdm import tqdm

# 设置控制台编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

sys.path.insert(0, '.')

from junction_agent import JUNCTION_CONFIGS
from junction_network import VehicleLevelMultiJunctionModel
from config import NetworkConfig


class BalancedExpertDemoDataset(Dataset):
    """专家演示数据集 - 对control样本过采样"""

    def __init__(self, demonstrations: List[Dict], control_sample_weight: int = 100):
        self.transitions = []
        self.control_sample_weight = control_sample_weight

        # 分离normal和control样本
        normal_transitions = []
        control_transitions = []

        for episode in demonstrations:
            for trans in episode['transitions']:
                action = trans['action_main'][0]
                if action < 0.95:
                    control_transitions.append(trans)
                else:
                    normal_transitions.append(trans)

        logging.info(f"[数据集] 原始数据: {len(normal_transitions)} normal, {len(control_transitions)} control")

        # 对control样本过采样
        self.transitions = normal_transitions.copy()
        for _ in range(control_sample_weight):
            self.transitions.extend(control_transitions)

        total_control = len(control_transitions) * control_sample_weight
        total_samples = len(self.transitions)

        logging.info(f"[数据集] 过采样后: {len(normal_transitions)} normal, {total_control} control")
        logging.info(f"[数据集] 总样本: {total_samples}")
        logging.info(f"[数据集] Control比例: {total_control/total_samples*100:.2f}%")

        # 统计action分布
        all_actions = np.array([t['action_main'][0] for t in self.transitions])
        logging.info(f"[数据集] Action统计: min={all_actions.min():.4f}, max={all_actions.max():.4f}, mean={all_actions.mean():.4f}, std={all_actions.std():.4f}")

        # 统计state向量
        states = np.array([t['state'] for t in self.transitions])
        logging.info(f"[数据集] State向量: shape={states.shape}, mean={states.mean():.4f}, std={states.std():.4f}")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]
        state = transition['state']
        action_main = transition['action_main']
        action_ramp = transition['action_ramp']

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
    """车辆级行为克隆训练器 - 完整版"""

    def __init__(self, model, device='cuda', log_dir='./logs'):
        self.model = model.to(device)
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.mse_loss = nn.MSELoss()

        # 训练统计
        self.epoch_stats = []

        logging.info(f"[训练器] 设备: {device}")
        logging.info(f"[训练器] 优化器: Adam(lr=1e-3)")

        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"[训练器] 总参数: {total_params:,}, 可训练: {trainable_params:,}")

    def compute_bc_loss(self, batch):
        """计算BC损失"""
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
                    veh_obs_dict[veh_type] = torch.stack(
                        vehicle_observations[junc_id][veh_type]
                    ).unsqueeze(0).to(self.device)

            vehicle_observations[junc_id] = veh_obs_dict

        # 模型推理
        all_actions, _, _ = self.model(observations, vehicle_observations, deterministic=True)

        # 计算损失
        total_loss = 0.0
        num_samples = 0

        # 收集预测和目标用于统计
        all_preds = []
        all_targets = []

        for junc_id, actions in all_actions.items():
            targets = action_targets[junc_id]

            for veh_type in ['main', 'ramp', 'diverge']:
                if f'{veh_type}_actions' in actions and actions[f'{veh_type}_actions'] is not None:
                    pred_actions = actions[f'{veh_type}_actions'][0]
                    target_actions = torch.tensor([t[0] for t in targets[veh_type]], dtype=torch.float32).to(self.device)

                    if len(pred_actions) > 0 and len(target_actions) > 0:
                        loss = self.mse_loss(pred_actions, target_actions)
                        total_loss += loss
                        num_samples += len(pred_actions)

                        # 收集统计信息
                        all_preds.extend(pred_actions.detach().cpu().numpy())
                        all_targets.extend(target_actions.detach().cpu().numpy())

        return total_loss, num_samples, all_preds, all_targets

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_preds = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            loss, num_samples, preds, targets = self.compute_bc_loss(batch)

            if num_samples > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                epoch_preds.extend(preds)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'samples': num_samples
                })

                # 每100个batch记录一次
                if batch_idx % 100 == 0:
                    logging.debug(f"[训练] Batch {batch_idx}, loss={loss.item():.4f}, samples={num_samples}")

        avg_loss = total_loss / max(num_batches, 1)

        # 统计预测值范围
        if epoch_preds:
            epoch_preds = np.array(epoch_preds)
            logging.info(f"[训练] Epoch {epoch} 预测action: min={epoch_preds.min():.4f}, max={epoch_preds.max():.4f}, mean={epoch_preds.mean():.4f}, std={epoch_preds.std():.4f}")
            logging.info(f"[训练] Epoch {epoch} Action分布: <0.5={np.sum(epoch_preds<0.5)}, 0.5-0.8={np.sum((epoch_preds>=0.5)&(epoch_preds<0.8))}, >=0.8={np.sum(epoch_preds>=0.8)}")

        return avg_loss

    def evaluate(self, dataloader, epoch):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                loss, num_samples, preds, targets = self.compute_bc_loss(batch)
                if num_samples > 0:
                    total_loss += loss.item()
                    num_batches += 1
                    all_preds.extend(preds)
                    all_targets.extend(targets)

        avg_loss = total_loss / max(num_batches, 1)

        # 详细统计
        if all_preds:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            logging.info(f"[验证] Epoch {epoch} 预测action: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}, std={all_preds.std():.4f}")
            logging.info(f"[验证] Epoch {epoch} 目标action: min={all_targets.min():.4f}, max={all_targets.max():.4f}, mean={all_targets.mean():.4f}, std={all_targets.std():.4f}")

            # MSE和MAE
            mse = np.mean((all_preds - all_targets) ** 2)
            mae = np.mean(np.abs(all_preds - all_targets))
            logging.info(f"[验证] Epoch {epoch} MSE={mse:.6f}, MAE={mae:.6f}")

            # 预测分布统计
            logging.info(f"[验证] Epoch {epoch} 预测分布: <0.5={np.sum(all_preds<0.5)}, 0.5-0.8={np.sum((all_preds>=0.5)&(all_preds<0.8))}, >=0.8={np.sum(all_preds>=0.8)}")
            logging.info(f"[验证] Epoch {epoch} 目标分布: <0.5={np.sum(all_targets<0.5)}, 0.5-0.8={np.sum((all_targets>=0.5)&(all_targets<0.8))}, >=0.8={np.sum(all_targets>=0.8)}")

        return avg_loss


def load_expert_demos(demo_dir: str) -> List[Dict]:
    """加载专家演示数据"""
    demonstrations = []

    pkl_file = os.path.join(demo_dir, 'expert_demonstrations.pkl')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            demonstrations = pickle.load(f)
        logging.info(f"[数据] 加载了 {len(demonstrations)} 个episodes")
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
        logging.info(f"[数据] 加载了 {len(demonstrations)} 个episodes (从多个文件)")

    return demonstrations


def main():
    parser = argparse.ArgumentParser(description='完整BC训练脚本')
    parser.add_argument('--train-demos', type=str, default='expert_demos_vehicle_v4',
                        help='专家演示数据目录')
    parser.add_argument('--output-dir', type=str, default='bc_checkpoints_full',
                        help='输出目录')
    parser.add_argument('--log-dir', type=str, default='./logs/bc_train',
                        help='日志目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--control-weight', type=int, default=100,
                        help='对control样本的过采样倍数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 配置日志
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )