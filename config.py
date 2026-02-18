"""
强化学习交通控制配置文件
定义所有超参数和环境配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os


@dataclass
class NetworkConfig:
    """神经网络配置"""
    # 车辆特征编码器
    vehicle_feature_dim: int = 32  # 车辆特征维度
    vehicle_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    
    # 道路特征编码器
    edge_feature_dim: int = 16  # 道路特征维度
    edge_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # 图神经网络配置
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_heads: int = 4  # 注意力头数
    
    # Transformer配置（时序建模）
    transformer_hidden_dim: int = 128
    transformer_num_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    
    # Actor网络
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # Critic网络
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # 动作空间
    speed_action_bins: int = 21  # 速度动作离散化数量
    min_speed_ratio: float = 0.3  # 最小速度比例（相对于限速）
    max_speed_ratio: float = 1.2  # 最大速度比例
    
    # Dropout
    dropout: float = 0.1


@dataclass
class PPOConfig:
    """PPO算法配置"""
    # 学习率
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    
    # PPO参数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    clip_epsilon: float = 0.2  # PPO裁剪参数
    entropy_coef: float = 0.01  # 熵正则化系数
    value_coef: float = 0.5  # 价值损失系数
    max_grad_norm: float = 0.5  # 梯度裁剪
    
    # 训练参数
    batch_size: int = 64
    n_epochs: int = 10  # 每次更新的epoch数
    update_frequency: int = 2048  # 更新频率（步数）
    
    # 探索参数
    entropy_decay: float = 0.999
    entropy_min: float = 0.001


@dataclass
class EnvironmentConfig:
    """环境配置"""
    # SUMO配置
    sumo_cfg: str = ""
    net_file: str = ""
    route_file: str = ""
    
    # 仿真参数
    max_steps: int = 3600
    delta_time: float = 1.0  # 仿真步长（秒）
    
    # 控制参数
    control_interval: int = 5  # 控制间隔（步数）
    cv_ratio: float = 0.25  # 智能网联车比例
    
    # 状态参数
    max_vehicles: int = 200  # 最大车辆数
    history_length: int = 10  # 历史状态长度
    detection_range: float = 500.0  # 检测范围（米）
    
    # 奖励参数
    ocr_weight: float = 1.0  # OCR奖励权重
    speed_weight: float = 0.1  # 速度奖励权重
    collision_penalty: float = -10.0  # 碰撞惩罚
    waiting_penalty: float = -0.1  # 等待惩罚
    
    # 关键路段（需要重点关注的路段）
    critical_edges: List[str] = field(default_factory=lambda: [
        'E7', 'E8', 'E9', 'E10', 'E11', 'E12',  # 主路
        '-E10', '-E11', '-E12',  # 反向主路
        'E15', 'E17', 'E19', 'E23',  # 匝道汇入
    ])
    
    # 关键交叉口
    critical_junctions: List[str] = field(default_factory=lambda: [
        'J5', 'J14', 'J15', 'J17'
    ])


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    total_timesteps: int = 10_000_000
    n_envs: int = 4  # 并行环境数
    seed: int = 42
    
    # 评估参数
    eval_frequency: int = 100_000  # 评估频率
    eval_episodes: int = 5  # 评估回合数
    
    # 保存参数
    save_frequency: int = 500_000
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 早停
    early_stop_patience: int = 10
    early_stop_threshold: float = 0.001


@dataclass
class Config:
    """总配置"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """初始化路径"""
        # 设置默认路径
        base_path = "sumo"
        if not self.env.sumo_cfg:
            self.env.sumo_cfg = os.path.join(base_path, "sumo.sumocfg")
        if not self.env.net_file:
            self.env.net_file = os.path.join(base_path, "net.xml")
        if not self.env.route_file:
            self.env.route_file = os.path.join(base_path, "routes.xml")


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()
