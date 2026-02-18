# 多智能体路口控制训练系统

## 快速开始

```bash
python rl_train.py --sumo-cfg sumo/sumo.sumocfg --total-timesteps 1000000
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sumo-cfg` | 必需 | SUMO配置文件路径 |
| `--total-timesteps` | 1000000 | 总训练步数 |
| `--lr` | 3e-4 | 学习率 |
| `--batch-size` | 64 | 批大小 |
| `--num-envs` | 4 | 并行环境数量 |
| `--workers` | CPU核心数 | 工作进程数 |
| `--update-frequency` | 2048 | 更新频率 |
| `--save-dir` | checkpoints | 模型保存目录 |
| `--log-dir` | logs | 日志目录 |

## 示例

```bash
# 基础训练
python rl_train.py --sumo-cfg sumo/sumo.sumocfg

# 自定义参数
python rl_train.py --sumo-cfg sumo/sumo.sumocfg --lr 0.0001 --num-envs 8

# 监控训练
tensorboard --logdir logs
```

## 特性

✅ CUDA训练加速
✅ 文件IO并行数据收集
✅ 自动检测WSL环境
✅ 支持libsumo高速模式
✅ 无CUDA跨进程问题

## 文件说明

- `rl_train.py` - 训练入口（唯一需要的文件）
- `junction_agent.py` - 智能体和环境
- `junction_network.py` - 神经网络
- `junction_trainer.py` - PPO训练器

## 环境要求

- Python 3.8+
- PyTorch
- SUMO
- libsumo（可选，推荐）
