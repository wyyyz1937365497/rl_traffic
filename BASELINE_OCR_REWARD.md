# 基于Baseline OCR比较的奖励函数

## 核心思想

通过**在相同步数比较OCR**，消除OCR自然上升的影响：

```
奖励 = (当前OCR - BaselineOCR在相同步数) × 权重
```

### 为什么这样做？

**问题**：OCR会自然上升
- 车辆到达数增加
- 行驶距离增加
- 即使策略不变，OCR也会上升

**解决方案**：相对比较
- 记录baseline（专家/前次训练）在每个步数的OCR
- 当前训练时，在**相同步数**比较
- 直接反馈相对于baseline的改进

---

## 使用流程

### 步骤1：生成Baseline OCR数据

运行专家策略（或任何baseline策略），记录OCR曲线：

```bash
python baseline_ocr_rewards.py \
    --sumo-cfg sumo/sumo.sumocfg \
    --output baseline_ocr/expert_baseline.pkl \
    --max-steps 3600
```

**输出示例**：
```
✓ SUMO已启动
开始收集baseline OCR数据...
  步骤 0: OCR = 0.0500
  步骤 500: OCR = 0.4500
  步骤 1000: OCR = 0.7200
  步骤 1500: OCR = 0.8500
  步骤 2000: OCR = 0.9000
  步骤 2500: OCR = 0.9250
  步骤 3000: OCR = 0.9400
  步骤 3500: OCR = 0.9500

✓ Baseline生成完成！
  最终OCR: 0.9500
  输出文件: baseline_ocr/expert_baseline.pkl
```

**说明**：
- 每100步记录一次OCR
- 使用专家策略（vType优化 + 主动速度引导）
- 约5-10分钟完成

---

### 步骤2：开始训练（自动使用Baseline比较）

训练时会自动加载baseline并计算增量奖励：

```bash
python rl_train.py \
    --sumo-cfg sumo/sumo.sumocfg \
    --total-timesteps 1000000 \
    --pretrained bc_checkpoints/best_model.pt
```

**日志输出示例**：
```
路口 J5 奖励: 2.3456
    [OCR delta=+0.0234 (current=0.9234, baseline=0.9000),
     ocr_reward=2.34, speed=0.120, throughput=1.000]
```

---

## 奖励函数详解

### OCR增量奖励（核心）

```python
baseline_ocr = get_baseline_ocr(current_step)  # 从文件加载
current_ocr = compute_current_ocr()
ocr_delta = current_ocr - baseline_ocr
ocr_reward = ocr_delta * 100.0  # 权重100
```

**示例**：
| 步数 | Baseline OCR | 当前OCR | 增量 | OCR奖励 |
|------|--------------|---------|------|---------|
| 500  | 0.4500       | 0.4520  | +0.0020 | +0.20 |
| 1000 | 0.7200       | 0.7250  | +0.0050 | +0.50 |
| 1500 | 0.8500       | 0.8400  | -0.0100 | **-1.00** |
| 2000 | 0.9000       | 0.9100  | +0.0100 | +1.00 |

### 瞬时辅助奖励（较小权重）

```python
# 速度奖励（降低权重）
speed_reward = (main_speed / 15.0) * 0.05

# 流量奖励（保持）
throughput_reward = departed_vehicles * 1.0

# 排队惩罚
queue_penalty = -(queue_length) * 0.02

# 等待惩罚
waiting_penalty = -max(0, waiting_time - 30) * 0.005

# 冲突惩罚
conflict_penalty = -conflict_risk * 0.05

# 生存奖励（完全移除！）
survival_reward = 0.0
```

### 总奖励

```python
total_reward = (
    ocr_reward +        # OCR增量（主，权重100）
    speed_reward +      # 速度（辅助，权重0.05）
    throughput_reward + # 流量（辅助，权重1.0）
    queue_penalty +     # 排队（惩罚，权重0.02）
    waiting_penalty +   # 等待（惩罚，权重0.005）
    conflict_penalty    # 冲突（惩罚，权重0.05）
)

# 裁剪到 [-10, 10]
total_reward = clip(total_reward, -10, 10)
```

---

## 奖励权重对比

| 奖励分量 | 权重 | 说明 |
|---------|------|------|
| **OCR增量** | **100.0** | **核心信号！直接反馈改进** |
| 速度 | 0.05 | 降低（从0.2→0.05） |
| 流量 | 1.0 | 保持 |
| 排队 | 0.02 | 保持 |
| 等待 | 0.005 | 保持 |
| 冲突 | 0.05 | 保持 |
| **生存** | **0.0** | **完全移除！** |

---

## 工作原理

### Baseline OCR数据结构

```python
# baseline_ocr/expert_baseline.pkl
{
    'ocr_history': {
        0: 0.0500,
        100: 0.1500,
        200: 0.2500,
        ...
        3600: 0.9500
    },
    'num_records': 37,
    'interval': 100
}
```

### 线性插值

如果当前步数没有精确记录，使用线性插值：

```python
# 例如：需要步骤 1234 的baseline
# 找到最近的记录：1200 和 1300
baseline_1200 = 0.88
baseline_1300 = 0.90

# 线性插值
ratio = (1234 - 1200) / (1300 - 1200) = 0.34
baseline_1234 = 0.88 + 0.34 * (0.90 - 0.88) = 0.8868
```

---

## 预期训练效果

### 奖励信号

**好的训练**（模型持续改进）：
```
步骤 500:  OCR delta = +0.002, reward = +0.2
步骤 1000: OCR delta = +0.005, reward = +0.5
步骤 1500: OCR delta = +0.008, reward = +0.8
步骤 2000: OCR delta = +0.012, reward = +1.2
```

**差的训练**（模型退化）：
```
步骤 500:  OCR delta = -0.001, reward = -0.1
步骤 1000: OCR delta = -0.003, reward = -0.3
步骤 1500: OCR delta = -0.005, reward = -0.5
```

### 训练曲线

理想情况下：
- **初期**：OCR增量接近0（从专家初始化）
- **中期**：OCR增量逐渐为正（模型开始改进）
- **后期**：OCR增量稳定在正值（持续改进）

---

## 高级用法

### 1. 使用不同的Baseline

可以比较不同baseline：

```python
# 专家策略baseline
calc = BaselineOCRRewardCalculator(
    baseline_file='baseline_ocr/expert_baseline.pkl'
)

# 之前训练的模型baseline
calc = BaselineOCRRewardCalculator(
    baseline_file='baseline_ocr/v1_baseline.pkl'
)

# 固定值baseline（fallback）
calc = BaselineOCRRewardCalculator(
    baseline_file=None  # 使用固定值0.95
)
```

### 2. 调整OCR奖励权重

编辑 `rl_train.py` 第490行：

```python
self.reward_calculator = BaselineOCRRewardCalculator(
    baseline_file=baseline_file,
    reward_weight=200.0  # 增加到200（默认100）
)
```

### 3. 从已有模型生成Baseline

如果已经有训练好的模型，可以用它生成baseline：

```python
# 修改 baseline_ocr_rewards.py
# 将 ExpertPolicy 替换为你的模型
from your_model import YourModel

model = YourModel()
model.load('checkpoints/your_model.pt')

# 在仿真循环中使用模型控制
actions = model.get_action(state)
apply_actions(actions)
```

---

## 监控训练

### 关键指标

训练时关注：

1. **OCR Delta** - 当前OCR相对于baseline的改进
   - 目标：持续为正
   - 警告：持续为负（模型在退化）

2. **OCR Reward** - OCR增量奖励
   - 目标：逐渐增大
   - 警告：震荡或下降

3. **Total Reward** - 总奖励
   - 目标：稳定上升
   - 警告：持续为负

### TensorBoard

```bash
tensorboard --logdir logs
```

查看指标：
- `train/ocr_delta` - OCR增量
- `train/ocr_reward` - OCR奖励
- `train/total_reward` - 总奖励

---

## 故障排查

### 问题1：未找到baseline文件

**日志**：
```
[WARNING] 未找到baseline文件: baseline_ocr/expert_baseline.pkl
[WARNING] 将使用固定baseline OCR = 0.95
```

**解决**：
```bash
python baseline_ocr_rewards.py --sumo-cfg sumo/sumo.sumocfg
```

### 问题2：OCR增量始终为0

**可能原因**：
- 当前模型与baseline性能相同
- 计算OCR时出错

**检查**：
- 查看日志中的 `current_ocr` 和 `baseline_ocr`
- 确认OCR计算正确

### 问题3：OCR增量持续为负

**可能原因**：
- 模型性能比baseline差
- 学习率太高，训练不稳定

**解决**：
- 降低学习率：`--lr 1e-4`
- 检查模型是否正确加载
- 增加训练时间

---

## 总结

### 关键改进

1. ✅ **消除OCR自然上升偏差** - 通过相同步数比较
2. ✅ **直接反馈改进** - OCR增量 = 相对于baseline的进步
3. ✅ **目标导向明确** - 奖励 = (当前 - baseline) × 100
4. ✅ **生存奖励移除** - 完全移除，不依赖生存
5. ✅ **辅助奖励保持** - 速度、流量等辅助信号

### 训练流程

```bash
# 步骤0：生成baseline（只需一次）
python baseline_ocr_rewards.py --sumo-cfg sumo/sumo.sumocfg

# 步骤1：收集专家数据（如果还没有）
python collect_expert_demos.py --num-episodes 5

# 步骤2：行为克隆（如果还没有）
python behavior_cloning.py --demo-file expert_demos/expert_demonstrations.pkl

# 步骤3：RL微调（自动使用baseline比较）
python rl_train.py --pretrained bc_checkpoints/best_model.pt
```

现在可以开始训练了！🎯

**核心优势**：模型能明确知道"我比专家/前一次好了多少"，并通过奖励信号得到反馈。
