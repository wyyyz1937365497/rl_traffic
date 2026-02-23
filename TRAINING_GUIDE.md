# 完整训练指南 - BC + PPO微调

## 文件说明

### 训练脚本

1. **train_bc_full.py** - 完整BC训练脚本
   - 支持control样本过采样
   - 详细的训练日志
   - TensorBoard支持
   - 早停机制

2. **train_ppo_finetune.py** - 完整PPO微调脚本
   - 从BC checkpoint初始化
   - 完整的PPO训练循环
   - GAE优势函数计算
   - 详细的奖励监控

## 完整训练流程

### 阶段1: BC预训练

#### 步骤1.1: 准备数据

如果已有数据，跳过此步骤：

```bash
python collect_vehicle_expert_demos.py \
  --output expert_demos_vehicle_v5 \
  --episodes 200 \
  --workers 4
```

#### 步骤1.2: BC训练（100x过采样）

```bash
python train_bc_full.py \
  --train-demos expert_demos_vehicle_v4 \
  --output-dir bc_checkpoints_100x \
  --log-dir ./logs/bc_100x \
  --epochs 50 \
  --batch-size 64 \
  --control-weight 100 \
  --device cuda \
  --seed 42
```

**预期输出**：
- 训练集/验证集分割
- 每个epoch的详细统计
- Action分布预测
- MSE/MAE指标
- 最佳模型保存

#### 步骤1.3: BC训练（500x过采样，更强的控制信号）

```bash
python train_bc_full.py \
  --train-demos expert_demos_vehicle_v4 \
  --output-dir bc_checkpoints_500x \
  --log-dir ./logs/bc_500x \
  --epochs 50 \
  --batch-size 64 \
  --control-weight 500 \
  --device cuda \
  --seed 42
```

### 阶段2: PPO微调

从最佳BC模型开始微调：

```bash
python train_ppo_finetune.py \
  --bc-checkpoint bc_checkpoints_100x/best_model.pt \
  --output-dir ppo_finetune_checkpoints \
  --log-dir ./logs/ppo_finetune \
  --episodes 100 \
  --max-steps 3600 \
  --lr 1e-5 \
  --device cuda \
  --seed 42
```

**关键参数**：
- `--lr 1e-5`: 微调使用更小的学习率（BC训练时是1e-3）
- `--episodes 100`: 训练episodes数
- `--max-steps 3600`: 每个episode最大步数

### 阶段3: 生成提交

#### 步骤3.1: 生成BC提交

```bash
python generate_submit_bc.py \
  --checkpoint bc_checkpoints_100x/best_model.pt \
  --output submit_bc_100x.pkl
```

#### 步骤3.2: 生成PPO微调后的提交

```bash
python generate_submit_bc.py \
  --checkpoint ppo_finetune_checkpoints/best_model.pt \
  --output submit_ppo_finetune.pkl
```

#### 步骤3.3: 本地评分

```bash
python local_score_calculator.py submit_bc_100x.pkl
python local_score_calculator.py submit_ppo_finetune.pkl
```

## 日志和监控

### TensorBoard

查看训练曲线：

```bash
# BC训练
tensorboard --logdir ./logs/bc_100x

# PPO微调
tensorboard --logdir ./logs/ppo_finetune
```

### 关键指标

**BC训练**：
- 训练损失（MSE）
- 验证损失
- Action预测分布
- MAE指标

**PPO微调**：
- Episode总奖励
- 平均奖励
- Policy loss
- Value loss
- Entropy（探索程度）

## 日志文件位置

```
logs/
├── bc_100x/
│   └── train_YYYYMMDD_HHMMSS.log
├── bc_500x/
│   └── train_YYYYMMDD_HHMMSS.log
└── ppo_finetune/
    ├── finetune_YYYYMMDD_HHMMSS.log
    └── events.out.tfevents...
```

## 快速测试命令

### 测试BC模型加载

```bash
python -c "
import torch
from junction_network import VehicleLevelMultiJunctionModel, NetworkConfig
from junction_agent import JUNCTION_CONFIGS

model = VehicleLevelMultiJunctionModel(JUNCTION_CONFIGS, NetworkConfig())
checkpoint = torch.load('bc_checkpoints_100x/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print('✓ BC模型加载成功')
print(f'Checkpoint: epoch={checkpoint.get(\"epoch\")}, val_loss={checkpoint.get(\"val_loss\")}')
"
```

### 测试推理（单步）

```bash
python -c "
import torch
from generate_submit_bc import BCSubmissionGenerator

generator = BCSubmissionGenerator(
    'sumo/sumo.cfg',
    'bc_checkpoints_100x/best_model.pt',
    device='cuda'
)
print('✓ 模型推理初始化成功')
"
```

## 故障排查

### 问题1: CUDA out of memory

**解决**：减小batch size

```bash
python train_bc_full.py --batch-size 32 ...
```

### 问题2: BC模型输出全1.0

**原因**：数据不平衡，control样本太少

**解决**：
1. 增加control-weight（500x或1000x）
2. 使用PPO微调
3. 检查数据收集时的action分布

### 问题3: PPO微调不收敛

**原因**：学习率太大或太小

**解决**：
```bash
# 尝试不同的学习率
--lr 5e-6  # 更小
--lr 5e-5  # 更大
```

### 问题4: 模型加载失败

**原因**：架构不匹配

**解决**：
```python
# 使用strict=False允许部分加载
model.load_state_dict(state_dict, strict=False)
```

## 预期训练时间

| 阶段 | 任务 | 时间（RTX 3090） |
|------|------|------------------|
| 数据收集 | 200 episodes | 30-60分钟 |
| BC训练 | 50 epochs | 10-15分钟 |
| PPO微调 | 100 episodes | 2-3小时 |

## 最佳实践

1. **数据准备**：收集至少200个episodes的专家演示
2. **BC预训练**：使用100x过采样作为起点
3. **PPO微调**：使用更小的学习率（1e-5）
4. **定期评估**：每10个episode生成一次提交文件
5. **备份模型**：保存所有checkpoint，便于回滚

## 进阶技巧

### 技巧1: 课程学习

先训练简单场景，再逐步增加难度：

```bash
# 阶段1: 低流量
python train_bc_full.py --train-demos expert_demos_low_traffic ...

# 阶段2: 高流量
python train_bc_full.py --train-demos expert_demos_high_traffic ...
```

### 技巧2: 集成方法

结合BC规则和RL：

```python
# 在推理时结合BC输出和规则
rule_action = expert_policy.get_action(state)
bc_action = model(state)
final_action = 0.7 * bc_action + 0.3 * rule_action
```

### 技巧3: 数据增强

对现有数据进行变换：

```python
# 添加噪声到状态
augmented_state = state + np.random.normal(0, 0.01, state.shape)
```

## 联系和支持

如果遇到问题：
1. 检查日志文件
2. 查看TensorBoard曲线
3. 运行快速测试命令
4. 参考故障排查章节

---

**最后更新**: 2026-02-23
**版本**: v1.0
