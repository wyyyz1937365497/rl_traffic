"""
强化学习交通控制主入口
提供训练、评估和推理功能
"""

import os
import sys
import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional

# ===== OD信息辅助函数 =====

def get_vehicle_origin(veh_id: str, veh_state) -> str:
    """获取车辆起点"""
    try:
        import traci
        route = traci.vehicle.getRoute(veh_id)
        if route and len(route) > 0:
            return route[0]
    except:
        pass
    return veh_state.edge_id if veh_state else ''

def get_vehicle_destination(veh_id: str, veh_state) -> str:
    """获取车辆终点"""
    try:
        import traci
        route = traci.vehicle.getRoute(veh_id)
        if route and len(route) > 0:
            return route[-1]
    except:
        pass
    return ''



import numpy as np
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_default_config
from environment import TrafficEnvironment
from network import TrafficControlModel, create_model
from ppo import PPOTrainer, train_model


class RLTrafficController:
    """
    强化学习交通控制器
    用于在SUMO仿真中应用训练好的策略
    """
    
    def __init__(self, model_path: str = None, config: Config = None, device: str = None):
        """
        初始化控制器
        
        Args:
            model_path: 模型路径
            config: 配置
            device: 设备
        """
        self.config = config or get_default_config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_model(self.config.network).to(self.device)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"已加载模型: {model_path}")
        
        # 状态追踪
        self.history_observations = []
        self.current_step = 0
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
    
    def get_actions(self, observation: Dict) -> Dict[str, float]:
        """
        获取动作
        
        Args:
            observation: 环境观察
        
        Returns:
            动作字典 {车辆ID: 速度比例}
        """
        with torch.no_grad():
            action_dict, _, _ = self.model(
                observation, 
                self.history_observations,
                deterministic=True
            )
        
        # 更新历史
        self.history_observations.append(observation)
        if len(self.history_observations) > self.config.env.history_length:
            self.history_observations.pop(0)
        
        self.current_step += 1
        
        return action_dict
    
    def reset(self):
        """重置控制器状态"""
        self.history_observations = []
        self.current_step = 0


def run_training(args):
    """运行训练"""
    print("=" * 70)
    print("强化学习交通控制 - 训练模式")
    print("=" * 70)
    
    # 加载配置
    config = get_default_config()
    
    # 修改配置
    if args.max_steps:
        config.env.max_steps = args.max_steps
    if args.total_timesteps:
        config.training.total_timesteps = args.total_timesteps
    if args.seed:
        config.training.seed = args.seed
    
    print(f"\n配置信息:")
    print(f"  仿真步数: {config.env.max_steps}")
    print(f"  训练步数: {config.training.total_timesteps}")
    print(f"  随机种子: {config.training.seed}")
    print(f"  设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 训练
    trainer, history = train_model(config, use_gui=args.gui)
    
    # 保存训练历史
    history_path = os.path.join(' sumo', 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                       for x in v] for k, v in history.items()}, f, indent=2)
    
    print(f"\n训练历史已保存: {history_path}")


def run_evaluation(args):
    """运行评估"""
    print("=" * 70)
    print("强化学习交通控制 - 评估模式")
    print("=" * 70)
    
    # 加载配置
    config = get_default_config()
    
    # 创建控制器
    controller = RLTrafficController(args.model, config)
    
    # 创建环境
    env = TrafficEnvironment(config.env, use_gui=args.gui)
    
    # 评估
    print(f"\n开始评估 ({args.episodes} 回合)...")
    
    results = []
    for episode in range(args.episodes):
        obs = env.reset()
        controller.reset()
        done = False
        episode_reward = 0
        
        while not done:
            actions = controller.get_actions(obs)
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
        
        results.append({
            'episode': episode + 1,
            'ocr': info.get('ocr', 0),
            'total_arrived': info.get('total_arrived', 0),
            'total_departed': info.get('total_departed', 0),
            'episode_reward': episode_reward
        })
        
        print(f"  回合 {episode + 1}: OCR = {info.get('ocr', 0):.4f}, "
              f"到达 = {info.get('total_arrived', 0)}, "
              f"出发 = {info.get('total_departed', 0)}")
    
    # 统计
    ocrs = [r['ocr'] for r in results]
    print(f"\n评估结果:")
    print(f"  平均OCR: {np.mean(ocrs):.4f}")
    print(f"  标准差: {np.std(ocrs):.4f}")
    print(f"  最大OCR: {np.max(ocrs):.4f}")
    print(f"  最小OCR: {np.min(ocrs):.4f}")
    
    # 保存结果
    results_path = os.path.join(' sumo', 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估结果已保存: {results_path}")
    
    env.close()


def run_inference(args):
    """运行推理（生成提交文件）"""
    print("=" * 70)
    print("强化学习交通控制 - 推理模式")
    print("=" * 70)
    
    # 加载配置
    config = get_default_config()
    
    # 创建控制器
    controller = RLTrafficController(args.model, config)
    
    # 创建环境
    env = TrafficEnvironment(config.env, use_gui=args.gui)
    
    # 运行仿真
    print("\n开始仿真...")
    
    obs = env.reset()
    controller.reset()
    done = False
    
    # 数据收集
    vehicle_data = []
    step_data = []
    
    while not done:
        actions = controller.get_actions(obs)
        obs, reward, done, info = env.step(actions)
        
        # 收集数据
        step_data.append({
            'step': info['step'],
            'active_vehicles': len(env.vehicle_states),
            'arrived_vehicles': info['total_arrived'],
            'departed_vehicles': info['total_departed'],
            'ocr': info['ocr']
        })
        
        for veh_id, veh_state in env.vehicle_states.items():
            vehicle_data.append({
                'step': info['step'],
                'vehicle_id': veh_id,
                'speed': veh_state.speed,
                'position': veh_state.position.tolist(),
                'edge_id': veh_state.edge_id,
                'completion_rate': veh_state.completion_rate,
                'origin': get_vehicle_origin(veh_id, veh_state),
                'destination': get_vehicle_destination(veh_id, veh_state)
                'vehicle_type': 'CV' if veh_state.is_cv else 'HV'
            })
        
        if info['step'] % 100 == 0:
            print(f"  步数: {info['step']}, OCR: {info['ocr']:.4f}")
    
    # 保存结果
    output_dir = 'data_output/competition_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存pickle
    data_package = {
        'parameters': {
            'simulation_time': config.env.max_steps,
            'step_length': config.env.delta_time,
            'total_steps': len(step_data),
            'final_departed': info['total_departed'],
            'final_arrived': info['total_arrived'],
            'collection_timestamp': timestamp
        },
        'step_data': step_data,
        'vehicle_data': vehicle_data,
        'statistics': {
            'final_ocr': info['ocr']
        }
    }
    
    pickle_path = os.path.join(output_dir, 'submit.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_package, f)
    
    print(f"\n仿真完成!")
    print(f"  最终OCR: {info['ocr']:.4f}")
    print(f"  到达车辆: {info['total_arrived']}")
    print(f"  出发车辆: {info['total_departed']}")
    print(f"\n提交文件已保存: {pickle_path}")
    
    env.close()


def run_baseline(args):
    """运行基线（无控制）"""
    print("=" * 70)
    print("基线仿真（无控制）")
    print("=" * 70)
    
    # 加载配置
    config = get_default_config()
    
    # 创建环境
    env = TrafficEnvironment(config.env, use_gui=args.gui)
    
    # 运行仿真（不执行任何控制）
    print("\n开始仿真...")
    
    obs = env.reset()
    done = False
    
    while not done:
        # 不执行任何控制动作
        obs, reward, done, info = env.step({})
        
        if info['step'] % 100 == 0:
            print(f"  步数: {info['step']}, OCR: {info['ocr']:.4f}")
    
    print(f"\n仿真完成!")
    print(f"  最终OCR: {info['ocr']:.4f}")
    print(f"  到达车辆: {info['total_arrived']}")
    print(f"  出发车辆: {info['total_departed']}")
    
    env.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='强化学习交通控制')
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--max-steps', type=int, help='每个回合最大步数')
    train_parser.add_argument('--total-timesteps', type=int, help='总训练步数')
    train_parser.add_argument('--seed', type=int, help='随机种子')
    train_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    # 评估模式
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--model', type=str, required=True, help='模型路径')
    eval_parser.add_argument('--episodes', type=int, default=5, help='评估回合数')
    eval_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    # 推理模式
    infer_parser = subparsers.add_parser('infer', help='运行推理')
    infer_parser.add_argument('--model', type=str, required=True, help='模型路径')
    infer_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    # 基线模式
    baseline_parser = subparsers.add_parser('baseline', help='运行基线')
    baseline_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'eval':
        run_evaluation(args)
    elif args.mode == 'infer':
        run_inference(args)
    elif args.mode == 'baseline':
        run_baseline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
