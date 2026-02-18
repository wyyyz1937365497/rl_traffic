"""
路口级多智能体交通控制系统
主入口文件
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from junction_agent import JUNCTION_CONFIGS, JunctionAgent, MultiAgentEnvironment
from junction_network import create_junction_model, NetworkConfig
from junction_trainer import MultiAgentPPOTrainer, PPOConfig


def print_junction_info():
    """打印路口信息"""
    print("=" * 70)
    print("路口拓扑分析")
    print("=" * 70)
    
    for junc_id, config in JUNCTION_CONFIGS.items():
        print(f"\n【{junc_id}】- 类型: {config.junction_type.value}")
        print(f"  主路入边: {config.main_incoming}")
        print(f"  主路出边: {config.main_outgoing}")
        print(f"  匝道入边: {config.ramp_incoming}")
        print(f"  匝道出边: {config.ramp_outgoing}")
        print(f"  信号灯: {'是' if config.has_traffic_light else '否'}")
        
        if config.junction_type.value == "type_a":
            print("  >>> 单纯匝道汇入：控制主路让行 + 匝道汇入时机")
        else:
            print("  >>> 匝道汇入+主路转出：协调汇入与转出")


def run_training(args):
    """运行训练"""
    print("=" * 70)
    print("多智能体路口控制 - 训练模式")
    print("=" * 70)
    
    # 打印路口信息
    print_junction_info()
    
    # 配置
    net_config = NetworkConfig()
    ppo_config = PPOConfig()
    
    if args.lr:
        ppo_config.lr = args.lr
    if args.batch_size:
        ppo_config.batch_size = args.batch_size
    
    print(f"\n训练配置:")
    print(f"  学习率: {ppo_config.lr}")
    print(f"  批大小: {ppo_config.batch_size}")
    print(f"  总步数: {args.total_timesteps}")
    print(f"  设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 创建环境
    env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=args.gui,
        seed=args.seed
    )
    
    eval_env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=False,
        seed=args.seed + 1000
    ) if args.eval else None
    
    # 创建模型
    model = create_junction_model(JUNCTION_CONFIGS, net_config)
    
    # 创建训练器
    trainer = MultiAgentPPOTrainer(model, ppo_config)
    
    # 训练
    save_dir = os.path.join(' sumo', args.save_dir)
    log_dir = os.path.join(' sumo', args.log_dir)
    
    history = trainer.train(
        env, 
        args.total_timesteps,
        eval_env,
        save_dir,
        log_dir
    )
    
    # 保存历史
    history_path = os.path.join(' sumo', 'junction_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练历史已保存: {history_path}")
    
    env.close()
    if eval_env:
        eval_env.close()


def run_evaluate(args):
    """运行评估"""
    print("=" * 70)
    print("多智能体路口控制 - 评估模式")
    print("=" * 70)
    
    # 创建环境
    env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=args.gui,
        seed=args.seed
    )
    
    # 创建模型
    model = create_junction_model(JUNCTION_CONFIGS)
    
    # 加载模型
    if args.model and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {args.model}")
    
    # 创建训练器（用于评估）
    trainer = MultiAgentPPOTrainer(model)
    
    # 评估
    print(f"\n开始评估 ({args.episodes} 回合)...")
    mean_ocr = trainer.evaluate(env, args.episodes)
    
    print(f"\n评估结果:")
    print(f"  平均OCR: {mean_ocr:.4f}")
    
    env.close()


def run_inference(args):
    """运行推理"""
    print("=" * 70)
    print("多智能体路口控制 - 推理模式")
    print("=" * 70)
    
    # 创建环境
    env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=args.gui,
        seed=args.seed
    )
    
    # 创建模型
    model = create_junction_model(JUNCTION_CONFIGS)
    
    # 加载模型
    if args.model and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {args.model}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # 运行仿真
    print("\n开始仿真...")
    
    obs = env.reset()
    done = False
    step = 0
    
    while not done:
        # 准备观察
        obs_tensors = {}
        vehicle_obs = {}
        
        for junc_id, agent in env.agents.items():
            state_vec = agent.get_state_vector()
            obs_tensors[junc_id] = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            
            controlled = agent.get_controlled_vehicles()
            vehicle_obs[junc_id] = {
                'main': None,
                'ramp': None,
                'diverge': None
            }
        
        # 获取动作
        with torch.no_grad():
            actions, _, _ = model(obs_tensors, vehicle_obs, deterministic=True)
        
        # 执行
        action_dict = {}
        for junc_id, action in actions.items():
            action_dict[junc_id] = {}
            agent = env.agents[junc_id]
            controlled = agent.get_controlled_vehicles()
            
            if controlled['main'] and 'main' in action:
                for veh_id in controlled['main'][:1]:
                    action_dict[junc_id][veh_id] = action['main'].item() if torch.is_tensor(action['main']) else action['main']
            
            if controlled['ramp'] and 'ramp' in action:
                for veh_id in controlled['ramp'][:1]:
                    action_dict[junc_id][veh_id] = action['ramp'].item() if torch.is_tensor(action['ramp']) else action['ramp']
        
        obs, _, done, info = env.step(action_dict)
        step += 1
        
        if step % 100 == 0:
            print(f"  步数: {step}")
    
    print("\n仿真完成!")
    
    env.close()


def run_baseline(args):
    """运行基线"""
    print("=" * 70)
    print("基线仿真（无控制）")
    print("=" * 70)
    
    # 创建环境
    env = MultiAgentEnvironment(
        sumo_cfg=args.sumo_cfg,
        use_gui=args.gui,
        seed=args.seed
    )
    
    # 运行仿真（不执行控制）
    print("\n开始仿真...")
    
    obs = env.reset()
    done = False
    step = 0
    
    while not done:
        # 不执行任何控制
        obs, _, done, info = env.step({})
        step += 1
        
        if step % 100 == 0:
            print(f"  步数: {step}")
    
    print("\n仿真完成!")
    
    env.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多智能体路口交通控制')
    parser.add_argument('--sumo-cfg', type=str, 
                       default='sumo/sumo.sumocfg',
                       help='SUMO配置文件路径')
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--total-timesteps', type=int, default=1000000, help='总训练步数')
    train_parser.add_argument('--lr', type=float, help='学习率')
    train_parser.add_argument('--batch-size', type=int, help='批大小')
    train_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    train_parser.add_argument('--gui', action='store_true', help='使用GUI')
    train_parser.add_argument('--eval', action='store_true', help='启用评估')
    train_parser.add_argument('--save-dir', type=str, default='checkpoints_junction', help='保存目录')
    train_parser.add_argument('--log-dir', type=str, default='logs_junction', help='日志目录')
    
    # 评估模式
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--model', type=str, required=True, help='模型路径')
    eval_parser.add_argument('--episodes', type=int, default=5, help='评估回合数')
    eval_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    eval_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    # 推理模式
    infer_parser = subparsers.add_parser('infer', help='运行推理')
    infer_parser.add_argument('--model', type=str, help='模型路径')
    infer_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    infer_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    # 基线模式
    baseline_parser = subparsers.add_parser('baseline', help='运行基线')
    baseline_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    baseline_parser.add_argument('--gui', action='store_true', help='使用GUI')
    
    # 信息模式
    info_parser = subparsers.add_parser('info', help='显示路口信息')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'eval':
        run_evaluate(args)
    elif args.mode == 'infer':
        run_inference(args)
    elif args.mode == 'baseline':
        run_baseline(args)
    elif args.mode == 'info':
        print_junction_info()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
