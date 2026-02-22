# Imitation Learning + RL 微调训练流程

## 问题诊断

之前的训练失败原因：
1. **奖励函数问题**：奖励信号与OCR目标不相关，模型学到无意义的策略
2. **Credit Assignment问题**：OCR奖励只在最后一步给出，太稀疏
3. **训练不稳定**：从随机初始化开始训练，探索空间太大

## 解决方案：Imitation Learning

采用 **Behavior Cloning → RL Fine-tuning** 两阶段训练：

### 阶段1：Behavior Cloning (行为克隆)
- 使用专家策略收集演示数据
- 通过监督学习让模型模仿专家行为
- 快速获得一个合理的初始策略

### 阶段2：RL Fine-tuning (强化学习微调)
- 从预训练模型开始训练
- 使用改进的OCR增量奖励函数
- 在专家策略基础上进一步优化

---

## 使用流程

### 步骤1：收集专家演示数据

使用规则专家策略（来自 `relu_based\rl_traffic`）收集演示数据：

```bash
python collect_expert_demos.py \
    --sumo-cfg sumo/sumo.sumocfg \
    --num-episodes 5 \
    --output-dir expert_demos
```

**参数说明**：
- `--sumo-cfg`: SUMO配置文件路径
- `--num-episodes`: 收集的episode数量（建议5-10个）
- `--output-dir`: 输出目录

**输出**：
- `expert_demos/expert_demonstrations.pkl`: 专家演示数据文件

---

### 步骤2：训练行为克隆模型

从演示数据中学习专家策略：

```bash
python behavior_cloning.py \
    --demo-file expert_demos/expert_demonstrations.pkl \
    --output-dir bc_checkpoints \
    --num-episodes 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --device cuda
```

**参数说明**：
- `--demo-file`: 专家演示数据文件路径
- `--output-dir`: 检查点保存目录
- `--num-episodes`: 训练轮数（建议50-100）
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备（cuda/cpu）

**输出**：
- `bc_checkpoints/best_model.pt`: 最佳验证损失的模型
- `bc_checkpoints/final_model.pt`: 最终模型

---

### 步骤3：RL微调（从预训练模型开始）

使用预训练的行为克隆模型作为初始点，进行RL训练：

```bash
python rl_train.py \
    --sumo-cfg sumo/sumo.sumocfg \
    --total-timesteps 1000000 \
    --lr 3e-4 \
    --batch-size 2048 \
    --workers 8 \
    --update-frequency 2048 \
    --save-dir checkpoints \
    --log-dir logs \
    --pretrained bc_checkpoints/best_model.pt
```

**关键参数**：
- `--pretrained`: 预训练模型路径（来自行为克隆）

---

## 改进的奖励函数

### OCR增量奖励

新奖励函数 (`ocr_increment_rewards.py`) 特点：

1. **密集的OCR信号**：每步计算OCR趋势，不只在最后一步
2. **直接相关性**：奖励与OCR目标直接相关
3. **趋势奖励**：基于OCR变化率给奖励，而不仅仅是绝对值

```python
# OCR趋势计算
recent_avg = mean(OCR最近5步)
old_avg = mean(OCR前5步)
ocr_trend = recent_avg - old_avg
reward = ocr_trend * 50.0  # 系数可调
```

### 辅助奖励

保持较小的权重，辅助训练稳定性：
- 速度奖励：`weight = 0.01`
- 排队惩罚：`weight = 0.001`
- 离开奖励：`weight = 0.1`

---

## 预期效果

### 训练曲线变化

**之前**（从头训练）：
- OCR在0.948-0.952之间随机波动
- 没有上升/下降趋势
- Loss正常收敛，但学不到有用的策略

**现在**（Imitation Learning）：
- 初始OCR = 专家策略OCR (~0.95)
- 微调后OCR逐步提升
- 奖励信号与OCR改善相关

### 基准对比

| 方法 | 初始OCR | 最终OCR | 训练时间 |
|------|---------|---------|----------|
| 从头训练 | 0.9489 | 0.9485 | 7M步 |
| IL + RL微调 | ~0.95 | >0.952 | 预期更快收敛 |

---

## 故障排查

### 问题1：收集专家数据时SUMO启动失败

**解决方案**：
```bash
# 检查libsumo是否安装
pip install libsumo

# 或使用traci模式
# 修改 collect_expert_demos.py 中的导入
```

### 问题2：行为克隆损失不下降

**可能原因**：
- 学习率太大 → 降低 `--lr` 到 1e-5
- 批次太小 → 增大 `--batch-size` 到 512
- 数据不足 → 增加专家episode数量

### 问题3：RL微调时OCR没有提升

**检查**：
1. 预训练模型是否正确加载：查看日志 "✓ 预训练模型加载成功"
2. 奖励函数是否正确：查看日志中的 "ocr_trend_reward"
3. 探索是否足够：调整 `entropy_coef`

---

## 文件说明

### 新增文件

1. **collect_expert_demos.py**
   - 专家演示数据收集脚本
   - 使用 libsumo + 订阅模式
   - 实现规则专家策略（来自 relu_based）

2. **behavior_cloning.py**
   - 行为克隆训练脚本
   - 从专家数据学习
   - 输出预训练模型

3. **ocr_increment_rewards.py**
   - OCR增量奖励函数
   - 解决奖励稀疏问题
   - 直接与OCR目标相关

### 修改文件

1. **rl_train.py**
   - 添加 `--pretrained` 参数支持
   - 更新奖励计算为OCR增量版本
   - 支持从行为克隆模型继续训练

---

## 进阶调整

### 奖励权重调整

编辑 `ocr_increment_rewards.py`：

```python
# 提高OCR奖励权重
self.cr_trend_bonus = 100.0  # 默认50.0

# 降低辅助奖励权重
self.speed_weight = 0.001  # 默认0.01
```

### 专家策略调整

编辑 `collect_expert_demos.py` 中的 `ExpertPolicy.get_action()`：

```python
# 调整专家规则
def get_action(self, state):
    # 自定义规则逻辑...
    return {'main': ..., 'ramp': ...}
```

---

## 总结

这个新的训练流程解决了之前的核心问题：

1. ✅ **奖励相关性**：OCR增量奖励与目标直接相关
2. ✅ **训练稳定性**：从专家策略开始，探索空间小
3. ✅ **收敛速度**：预训练模型加速收敛
4. ✅ **可解释性**：专家策略提供良好的baseline

开始训练吧！🚀
