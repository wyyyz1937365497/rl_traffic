# 奖励函数重新设计

## 问题分析

### 为什么OCR增量奖励不工作？

OCR公式：
```
OCR = (N_arrived + Σ(d_i_traveled / d_i_total)) / N_total
```

**问题**：随着仿真进行，即使策略没有任何改善，OCR也会自然上升：
- `N_arrived` 随时间增加（车辆到达）
- `d_i_traveled` 随时间增加（在途车辆行驶距离增加）
- 分母 `N_total` 相对固定

**结论**：不能使用OCR的绝对值或增量作为即时奖励信号。

---

## 新的奖励设计：基于瞬时性能指标

### 核心原则

1. ✅ **不依赖OCR绝对值** - 避免自然上升的偏差
2. ✅ **关注瞬时状态** - 速度、排队、等待时间
3. ✅ **事件驱动** - 车辆离开、成功汇入
4. ✅ **极小的生存奖励** - 防止模型学会"什么都不做"

---

## 奖励组成

### 1. 速度奖励（权重 0.2）

```python
# 使用sigmoid归一化，避免极端值
speed_score = sigmoid((main_speed - 8.0) / 5.0)  # 8m/s为中点
speed_reward = speed_score * 0.2
```

- 主路速度目标：13.89 m/s (50 km/h)
- 匝道速度目标：8-10 m/s
- 使用sigmoid平滑归一化

### 2. 流量奖励（权重 2.0）

```python
# 车辆离开控制区
departed_delta = previous_in_zone - current_in_zone
throughput_reward = departed_delta * 2.0
```

- 每辆离开控制区的车给正奖励
- 鼓励车辆快速通过

### 3. 排队惩罚（权重 0.02）

```python
# 平方惩罚，对长队更敏感
queue_penalty = -(
    main_queue_length ** 1.5 * 0.01 +
    ramp_queue_length ** 1.5 * 0.02
)
```

- 匝道排队惩罚更重要
- 使用指数（1.5）放大长队列的影响

### 4. 等待惩罚（权重 0.01）

```python
# 分段惩罚：等待>20s才惩罚
waiting_penalty = -((ramp_waiting_time - 20) ** 0.8) * 0.01
```

- 短暂等待是可接受的
- 长时间等待受惩罚

### 5. 冲突惩罚（权重 0.1）

```python
# 平方惩罚
conflict_penalty = -(conflict_risk ** 2) * 0.1
```

- 高冲突风险受严厉惩罚

### 6. 生存奖励（权重 0.0001）⭐

```python
survival_reward = 0.0001  # 极小！
```

**之前的错误**：
- 生存奖励 0.05 × 3600步 = 180分（占主导！）

**现在的改进**：
- 生存奖励 0.0001 × 3600步 = 0.36分（可忽略）
- 防止模型通过"什么都不做"获取高奖励

---

## 总奖励计算

```python
total_reward = (
    speed_reward +           # ~0.2
    throughput_reward +      # ~2.0 (每辆离开的车)
    queue_penalty +          # ~ -0.5 to -2.0
    waiting_penalty +        # ~ -0.1 to -0.5
    conflict_penalty +       # ~ -0.05
    survival_reward          # 0.0001 (几乎为0)
)

# 裁剪到合理范围
total_reward = clip(total_reward, -5.0, 5.0)
```

---

## 三种实现版本

### 1. `InstantRewardCalculator` - 基础版本

线性奖励，简单直观。

### 2. `NormalizedInstantRewardCalculator` - 归一化版本 ⭐ **推荐**

- 所有奖励分量归一化到相似尺度
- 使用sigmoid避免极端值
- 使用非线性惩罚（指数）
- 生存奖励极小（0.0001）

### 3. `ShapedInstantRewardCalculator` - Reward Shaping版本

使用势函数进行reward shaping，理论保证不改变最优策略：
```python
shaping_reward = -gamma * Phi(s') + Phi(s)
```

---

## 奖励尺度对比

| 奖励分量 | 之前 | 现在 | 变化 |
|---------|------|------|------|
| 速度奖励 | 0.5 | 0.2 | 减小 |
| 离开奖励 | 0.3 | 2.0 | 大幅增加 |
| 排队惩罚 | 0.005 | 0.02 | 增加 |
| 等待惩罚 | 0.002 | 0.01 | 增加 |
| 冲突惩罚 | 0.02 | 0.1 | 大幅增加 |
| **生存奖励** | **0.05** | **0.0001** | **大幅减小** |

---

## 预期效果

### 奖励分布

```
理想状态（快速流动）：
  speed: +0.2
  throughput: +2.0
  queue: -0.1
  conflict: -0.01
  survival: +0.0001
  总计: ~+2.1

拥堵状态（速度低、排队多）：
  speed: +0.05
  throughput: +0.0
  queue: -1.5
  waiting: -0.3
  conflict: -0.05
  survival: +0.0001
  总计: ~-1.8
```

### 训练行为

1. **初期（随机策略）**：奖励波动，可能为负
2. **中期（学会基本控制）**：奖励逐渐为正，速度提升
3. **后期（精细优化）**：奖励稳定，持续优化流量和减少排队

---

## 使用方法

`rl_train.py` 已更新为使用新的奖励函数：

```python
from instant_rewards import NormalizedInstantRewardCalculator
```

无需修改代码，直接运行训练即可：

```bash
python rl_train.py --pretrained bc_checkpoints/best_model.pt
```

---

## 监控指标

训练时关注：

1. **奖励趋势**：应该逐渐上升（从负到正）
2. **速度得分**：应该逐渐接近1.0
3. **排队长度**：应该逐渐减少
4. **冲突风险**：应该逐渐降低

如果奖励始终为负或没有上升趋势，检查：
- 模型是否正确加载
- 学习率是否合适
- 网络架构是否合适

---

## 超参数调整

如果训练效果不理想，可以调整权重：

编辑 `instant_rewards.py` 中的 `self.weights`：

```python
# 如果更关注速度，增加：
'speed': 0.3,  # 默认0.2

# 如果排队太严重，增加惩罚：
'queue': 0.03,  # 默认0.02

# 如果冲突太多，增加：
'conflict': 0.2,  # 默认0.1
```

---

## 总结

新奖励函数的关键改进：

1. ✅ 移除OCR依赖，避免自然上升偏差
2. ✅ 关注瞬时性能，更直接的反馈
3. ✅ 生存奖励减小500倍（0.05 → 0.0001）
4. ✅ 奖励分量归一化，尺度平衡
5. ✅ 事件驱动，鼓励正确行为

开始训练吧！🚀
