# 动作空间一致性说明

## ⚠️ 关键问题：训练-推理动作空间不一致

### 问题说明

强化学习模型在训练时学习的是特定动作映射下的最优策略。如果推理时使用不同的动作映射，会导致**严重的distribution shift**，模型性能大幅下降。

### 修复前后对比

#### ❌ 修复前（不一致）

**训练时**：
```python
speed_limit = 13.89
target_speed = speed_limit * (0.3 + 0.9 * action)
# 当 action=0.5: target_speed = 13.89 * (0.3 + 0.45) = 10.37 m/s
```

**推理时**：
```python
# 主路
target_speed = 15.0 + action_value * 10.0
# 当 action=0.5: target_speed = 15.0 + 5.0 = 20.0 m/s ❌

# 匝道
target_speed = 10.0 + action_value * 10.0
# 当 action=0.5: target_speed = 10.0 + 5.0 = 15.0 m/s ❌
```

**结果**：同样的动作值产生完全不同的速度，模型完全失效！

#### ✅ 修复后（一致）

**训练时和推理时完全一致**：
```python
speed_limit = 13.89  # 50 km/h
target_speed = speed_limit * (0.3 + 0.9 * action)

# 动作范围分析：
# action ∈ [0, 1]（假设使用sigmoid/tanh激活后映射）
# action=0.0: target_speed = 13.89 * 0.3 = 4.17 m/s (15 km/h)
# action=0.5: target_speed = 13.89 * 0.75 = 10.42 m/s (37.5 km/h)
# action=1.0: target_speed = 13.89 * 1.2 = 16.67 m/s (60 km/h)
```

## 动作映射详解

### 公式推导

```python
target_speed = speed_limit * (0.3 + 0.9 * action)
             = speed_limit * 0.3 + speed_limit * 0.9 * action
             = 4.167 + 12.501 * action
```

### 设计理念

1. **最低速度**：4.17 m/s (15 km/h)
   - 避免车辆完全停止
   - 保持交通流动

2. **最高速度**：16.67 m/s (60 km/h)
   - speed_limit * 1.2
   - 允许轻微超速以提高通行效率

3. **线性映射**：
   - 动作值直接线性映射到速度
   - 模型学习简单直观
   - 便于优化

### 动作值含义

| 动作值 | 目标速度 | 含义 |
|--------|----------|------|
| 0.0 | 4.17 m/s | 最低速度，保守控制 |
| 0.25 | 7.29 m/s | 较低速度 |
| 0.5 | 10.42 m/s | 中等速度 |
| 0.75 | 13.54 m/s | 较高速度 |
| 1.0 | 16.67 m/s | 最高速度，激进控制 |

## 代码验证

### 训练代码验证（rl_train.py）

```python
# 第125-133行
def _apply_actions(self, actions):
    for junc_id, action_dict in actions.items():
        for veh_id, action in action_dict.items():
            try:
                speed_limit = 13.89
                target_speed = speed_limit * (0.3 + 0.9 * action)
                traci_wrapper.vehicle.setSpeed(veh_id, target_speed)
            except:
                continue
```

### 推理代码验证（sumo/main.py）

```python
# 第211-237行（已修复）
# 控制主路车辆
if controlled['main'] and 'main' in action:
    for veh_id in controlled['main'][:1]:
        try:
            action_value = action['main'].item()
            # 与训练时完全一致的映射
            speed_limit = 13.89
            target_speed = speed_limit * (0.3 + 0.9 * action_value)
            # 确保速度在合理范围内
            target_speed = max(0.0, min(target_speed, speed_limit * 1.2))
            traci.vehicle.setSpeed(veh_id, target_speed)
        except:
            pass

# 控制匝道车辆（使用相同的映射逻辑）
if controlled['ramp'] and 'ramp' in action:
    for veh_id in controlled['ramp'][:1]:
        try:
            action_value = action['ramp'].item()
            # 与训练时完全一致的映射
            speed_limit = 13.89
            target_speed = speed_limit * (0.3 + 0.9 * action_value)
            # 确保速度在合理范围内
            target_speed = max(0.0, min(target_speed, speed_limit * 1.2))
            traci.vehicle.setSpeed(veh_id, target_speed)
        except:
            pass
```

## 影响分析

### 动作空间不一致的后果

1. **模型输出与实际控制脱节**
   - 模型认为 action=0.5 → 10.42 m/s
   - 实际执行 action=0.5 → 20.0 m/s
   - 完全不可预测！

2. **优化目标偏移**
   - 训练时优化的策略在推理时失效
   - 奖励函数基于特定动作-速度映射
   - 不同的映射导致奖励函数计算错误

3. **性能大幅下降**
   - 预期OCR可能从0.95降至0.80以下
   - 训练完全白费

### 修复后的效果

✅ **完全一致性**
- 训练和推理使用完全相同的动作映射
- 模型输出可以直接用于控制
- 发挥训练的最佳性能

✅ **可预测性**
- action=0.5 始终对应 10.42 m/s
- 模型学习到的策略可以直接应用
- 结果可复现

✅ **最佳性能**
- 模型在训练时达到的最佳OCR可以直接体现在推理中
- 无需额外微调
- 充分发挥训练效果

## 测试验证

### 验证步骤

1. **检查训练代码**
   ```bash
   grep -n "speed_limit.*0.3.*0.9" rl_train.py
   # 应该看到训练时的映射逻辑
   ```

2. **检查推理代码**
   ```bash
   grep -n "speed_limit.*0.3.*0.9" sumo/main.py
   # 应该看到相同的映射逻辑
   ```

3. **单元测试**
   ```python
   # 测试动作映射
   action = 0.5
   speed_limit = 13.89
   expected_speed = speed_limit * (0.3 + 0.9 * action)  # 10.42 m/s

   # 训练时
   train_speed = compute_training_speed(action)
   assert abs(train_speed - expected_speed) < 0.01

   # 推理时
   infer_speed = compute_inference_speed(action)
   assert abs(infer_speed - expected_speed) < 0.01
   ```

4. **端到端测试**
   ```bash
   # 运行推理并检查结果
   cd sumo
   python main.py

   # 检查输出日志
   # OCR应该接近训练时的验证OCR
   ```

## 最佳实践

### 开发时

1. **将动作映射定义为常量**
   ```python
   # 在 constants.py 中定义
   SPEED_LIMIT = 13.89
   ACTION_SCALE = 0.9
   ACTION_OFFSET = 0.3

   def action_to_speed(action):
       return SPEED_LIMIT * (ACTION_OFFSET + ACTION_SCALE * action)
   ```

2. **训练和推理共用同一函数**
   ```python
   # 在 utils.py 中定义
   from constants import action_to_speed

   # 训练时使用
   target_speed = action_to_speed(action)

   # 推理时使用
   target_speed = action_to_speed(action_value)
   ```

3. **添加单元测试**
   ```python
   def test_action_consistency():
       actions = [0.0, 0.25, 0.5, 0.75, 1.0]
       for action in actions:
           speed = action_to_speed(action)
           assert 4.17 <= speed <= 16.67
   ```

### 提交前检查清单

- [ ] 训练代码的动作映射已确认
- [ ] 推理代码使用相同的映射
- [ ] 已通过单元测试验证
- [ ] 已通过端到端测试
- [ ] OCR性能符合预期
- [ ] 代码中有明确的注释说明映射关系

## 总结

**关键要点**：
- ⚠️ 训练和推理的动作映射必须完全一致
- ✅ 已修复为相同的映射逻辑
- 📊 公式：`target_speed = 13.89 * (0.3 + 0.9 * action)`
- 🎯 范围：[4.17, 16.67] m/s ([15, 60] km/h)

这确保了模型能够发挥训练时的最佳效果，在比赛中获得最高的OCR！
