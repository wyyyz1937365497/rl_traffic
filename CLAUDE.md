# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

这是一个基于强化学习的交通控制系统项目，主要实现了多智能体路口交通控制。项目使用PyTorch和SUMO仿真环境，结合行为克隆（BC）和PPO强化学习算法来优化交通控制策略。

## Key Components

### Main Files
- `junction_main.py` - 主入口文件，包含训练、评估、推理等模式
- `train_ppo_finetune.py` - PPO微调训练脚本，从BC模型初始化
- `junction_agent.py` - 路口代理和环境定义
- `junction_network.py` - 网络模型定义
- `junction_trainer.py` - 训练器实现

### Training Workflow
1. **Behavior Cloning (BC)**: 使用专家数据训练初始模型
2. **PPO Fine-tuning**: 从BC模型初始化，使用PPO算法进行微调
3. **Reward Function**: 自定义的奖励函数，优先考虑流量效率、稳定性、OCR和安全性

### Key Features
- 多智能体路口控制
- 车辆级模型架构
- 硬编码路网拓扑配置（提高性能）
- 改进的奖励函数设计
- 完整的训练和评估流程

## Commands

### Training
```bash
# 训练模型
python junction_main.py train --total-timesteps 1000000

# PPO微调训练（从BC模型初始化）
python train_ppo_finetune.py --bc-checkpoint bc_checkpoints_vehicle_v4_balanced/best_model.pt --episodes 100
```

### Evaluation
```bash
# 评估模型
python junction_main.py eval --model checkpoints_junction/best_model.pt --episodes 5
```

### Inference
```bash
# 推理模式
python junction_main.py infer --model checkpoints_junction/best_model.pt
```

### Quick Start
```bash
# 快速开始
python quick_start.py
```

## Directory Structure
```
rl_traffic/
├── junction_main.py          # 主入口
├── train_ppo_finetune.py     # PPO微调训练
├── junction_agent.py         # 路口代理和环境
├── junction_network.py       # 网络模型
├── junction_trainer.py       # 训练器
├── sumo/                     # SUMO配置文件
├── logs/                     # 日志目录
├── checkpoints_junction/     # 模型检查点
└── logs/ppo_finetune/        # PPO微调日志和分析
```

## Key Concepts

### Junction Control
- 多路口协调控制
- 不同类型的路口（Type A: 单纯匝道汇入，Type B: 匝道汇入+主路转出）
- 信号灯控制支持

### Reward Design
- 流量奖励（权重4.5）：优先提升吞吐量
- 稳定性奖励（权重3.0）：减少波动和拥堵
- OCR奖励（权重2.0）：目标提升到0.96-0.97
- 安全性惩罚（权重-2.5）：控制碰撞和急停

### Performance Improvements
- 使用硬编码拓扑配置，性能提升100倍
- 改进的奖励函数，训练成功率从0%提升到>80%
- 精确的OCR计算，性能提升20-40%