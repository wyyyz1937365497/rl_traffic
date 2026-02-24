# PPO微调快速启动指南

## 🎯 奖励函数设计

改进的奖励函数优化以下指标：

### 1. OCR奖励（权重: 3.0）
```python
ocr_reward = 3.0 * (current_ocr / 0.94) ** 2
```
- 目标OCR: 0.94
- 越接近目标，奖励越高
- 鼓励车辆完成旅程

### 2. 流量奖励（权重: 2.0）
```python
speed_reward = 2.0 * (mean_speed / 13.89) ** 2
traffic_reward = 0.5 * sqrt(num_active / 100.0)
```
- 速度奖励：鼓励高速（0-13.89 m/s）
- 活跃车辆奖励：鼓励系统内有更多车辆运行
- 反映交通吞吐量

### 3. 稳定性奖励（权重: 2.0）
```python
stability_speed = 1.0 * max(0, 1.0 - speed_std / 8.0)
stability_accel = 1.0 * max(0, 1.0 - mean_abs_accel / 1.2)
```
- 速度标准差：< 8 m/s（越低越好）
- 加速度绝对值：< 1.2 m/s²（越低越好）
- 减少急加减速

### 4. 安全性惩罚（权重: -2.0）
```python
collision_penalty = -0.5 * num_collisions
emergency_stop_penalty = -0.1 * num_emergency_stops
slow_penalty = -1.0 * slow_ratio
```
- 碰撞惩罚：每次碰撞-0.5分
- 急停惩罚：每次-0.1分
- 慢速惩罚：速度<3m/s的车辆比例

---

## 🚀 完整训练流程

### 步骤1: 准备BC模型

如果已有BC模型，跳过此步骤。否则：

```bash
python train_bc_full.py \
  --train-demos expert_demos_vehicle_v4 \
  --output-dir bc_pretrain \
  --control-weight 100 \
  --epochs 50 \
  --device cuda
```

### 步骤2: PPO微调（核心）

```bash
python train_ppo_finetune.py \
  --bc-checkpoint bc_pretrain/best_model.pt \
  --output-dir ppo_finetune_checkpoints \
  --log-dir ./logs/ppo_finetune \
  --episodes 100 \
  --max-steps 3600 \
  --lr 1e-5 \
  --device cuda \
  --seed 42
```

**参数说明**：
- `--bc-checkpoint`: BC模型路径（必需）
- `--episodes`: 训练episodes数（建议100-200）
- `--lr`: 学习率（微调建议1e-5，比BC小100倍）
- `--max-steps`: 每个episode最大步数（3600=1小时仿真）

**预期时间**：
- 每个episode约1-2分钟
- 100 episodes约2-3小时（RTX 3090）

### 步骤3: 监控训练

```bash
# 实时日志
tail -f ./logs/ppo_finetune/finetune_*.log

# TensorBoard（另开终端）
tensorboard --logdir ./logs/ppo_finetune
```

### 步骤4: 生成提交

```bash
# 等待训练完成后
python generate_submit_bc.py \
  --checkpoint ppo_finetune_checkpoints/best_model.pt \
  --output submit_ppo_finetune.pkl

# 本地评分
python local_score_calculator.py submit_ppo_finetune.pkl
```

---

## 📊 预期输出

### Episode日志示例

```
[Episode] 总奖励: 3234.56
[Episode] 平均奖励分解:
  OCR奖励: 2.8456        # 主要贡献
  速度奖励: 1.6543
  流量奖励: 0.2345
  稳定性(速度): 0.8765
  稳定性(加速度): 0.6543
  碰撞惩罚: -0.2500      # 碰撞少
  急停惩罚: -0.0500
  慢速惩罚: -0.1234
```

### TensorBoard曲线

- **Reward/total**: 总奖励曲线（应上升）
- **Reward/mean**: 平均奖励（应稳定）
- **Loss/policy**: 策略损失（应下降）
- **Loss/value**: 价值损失（应下降）
- **Entropy**: 熵（探索程度，缓慢下降）

---

## 🔧 超参数调优

### 如果奖励太低（< 1000）

**可能原因**：模型不控制，OCR太低

**解决方案**：
```bash
# 增大学习率，加快训练
--lr 5e-5

# 或增加训练episodes
--episodes 200
```

### 如果碰撞太多（> 100）

**可能原因**：控制过于激进

**解决方案**：
```bash
# 降低学习率
--lr 5e-6

# 减小clip范围（需要修改代码）
```

### 如果速度太慢（< 5 m/s）

**可能原因**：过度减速

**解决方案**：
```bash
# 可能需要重新训练BC模型
# 或调整奖励权重（需要修改代码）
```

---

## 📝 不同配置的训练命令

### 配置1: 快速测试（10 episodes）

```bash
python train_ppo_finetune.py \
  --bc-checkpoint bc_pretrain/best_model.pt \
  --episodes 10 \
  --lr 1e-5 \
  --log-dir ./logs/ppo_test
```

**时间**: 约10-20分钟

### 配置2: 标准训练（100 episodes）

```bash
python train_ppo_finetune.py \
  --bc-checkpoint bc_pretrain/best_model.pt \
  --episodes 100 \
  --lr 1e-5 \
  --log-dir ./logs/ppo_standard
```

**时间**: 约2-3小时

### 配置3: 长期训练（200 episodes）

```bash
python train_ppo_finetune.py \
  --bc-checkpoint bc_pretrain/best_model.pt \
  --episodes 200 \
  --lr 5e-5 \
  --log-dir ./logs/ppo_long
```

**时间**: 约5-6小时

### 配置4: 谨慎微调（50 episodes，小学习率）

```bash
python train_ppo_finetune.py \
  --bc-checkpoint bc_pretrain/best_model.pt \
  --episodes 50 \
  --lr 5e-6 \
  --log-dir ./logs/ppo_conservative
```

**时间**: 约1-1.5小时

---

## ⚠️ 常见问题

### Q1: 训练很慢怎么办？

**A**: 检查GPU使用率
```bash
nvidia-smi
```

如果GPU利用率<50%，可能需要：
- 增加batch size（需要修改代码）
- 检查是否有CPU瓶颈

### Q2: 奖励一直很低（< 500）

**A**: 可能BC模型太差，尝试：
```bash
# 重新训练BC，使用更高过采样
python train_bc_full.py --control-weight 500
```

### Q3: 训练中途崩溃？

**A**: 检查checkpoint：
```bash
ls -lh ppo_finetune_checkpoints/
```

从最近的checkpoint恢复：
```bash
# 需要修改代码支持resume功能
```

### Q4: 内存不足（CUDA OOM）

**A**: 减小batch size或max_steps：
```bash
--max-steps 1800  # 减半
```

---

## 📈 成功指标

### Episode 1-10（探索阶段）
- 总奖励: 500-1500
- OCR奖励: 1.5-2.5
- 碰撞惩罚: < -1.0

### Episode 10-50（学习阶段）
- 总奖励: 1500-3000（上升趋势）
- OCR奖励: 2.5-3.0
- 碰撞惩罚: < -0.5

### Episode 50-100（优化阶段）
- 总奖励: 3000-4000（稳定）
- OCR奖励: 2.8-3.2
- 碰撞惩罚: < -0.2
- 稳定性奖励: > 1.2

### 目标性能
- **本地评分**: > 24分
- **OCR**: > 0.93
- **完成车辆**: > 5800
- **碰撞**: < 50

---

## 🎓 进阶技巧

### 技巧1: 课程学习

先训练简单场景（低流量），再增加难度：
```bash
# 阶段1: 低流量配置
# 阶段2: 标准配置
# 阶段3: 高流量配置
```

### 技巧2: 奖励 shaping

如果某个指标不达标，调整权重：
```python
# 在compute_reward()中修改权重
ocr_reward = 5.0 * (current_ocr / 0.94) ** 2  # 从3.0增加到5.0
```

### 技巧3: 多次训练

训练多个模型，选择最佳：
```bash
# 训练3个不同随机种子
for seed in 42 123 456; do
  python train_ppo_finetune.py --seed $seed --episodes 50
done
```

---

## 🚦 训练检查清单

- [ ] BC模型已准备
- [ ] 数据目录正确
- [ ] GPU可用（nvidia-smi）
- [ ] 日志目录已创建
- [ ] 磁盘空间充足（>10GB）
- [ ] TensorBoard已启动
- [ ] 预计时间已确认

开始训练后：
- [ ] 每10 episodes检查日志
- [ ] 监控TensorBoard曲线
- [ ] 检查GPU使用率
- [ ] 观察奖励分解是否合理

---

**最后更新**: 2026-02-24
**版本**: v2.0 - 改进奖励函数
