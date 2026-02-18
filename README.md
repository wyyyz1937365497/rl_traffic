# 强化学习交通控制系统 - 完整架构

## 系统概述

本系统提供两种架构方案，可根据需求选择：

### 方案一：全局模型（原始方案）
- **文件**: `config.py`, `environment.py`, `network.py`, `ppo.py`, `train.py`, `main.py`
- **特点**: 全局状态，统一决策
- **适用**: 小规模路网，简单场景

### 方案二：路口级多智能体（推荐方案）
- **文件**: `junction_agent.py`, `junction_network.py`, `junction_trainer.py`, `junction_main.py`
- **特点**: 路口独立决策，专门化网络
- **适用**: 大规模路网，复杂拓扑

## 架构对比

| 维度 | 全局模型 | 路口级多智能体 |
|------|---------|---------------|
| 状态空间 | 全局（~3000维） | 局部（~160维/路口） |
| 决策方式 | 集中式 | 分布式 |
| 网络结构 | 统一网络 | 专门化网络 |
| 训练难度 | 高 | 中 |
| 可解释性 | 低 | 高 |
| 泛化能力 | 低 | 高 |

## 推荐使用路口级多智能体方案

### 核心优势

1. **精准建模冲突点**
   - 拥堵主要来源于路口汇入冲突
   - 每个路口独立建模，聚焦核心问题

2. **拓扑类型专门化**
   - 类型A：单纯匝道汇入
   - 类型B：匝道汇入 + 主路转出
   - 不同类型使用不同的网络结构

3. **降低状态空间维度**
   - 从全局3000维降到局部160维
   - 训练更稳定，收敛更快

4. **路口间协调**
   - 相邻路口通过注意力机制协调
   - 避免局部最优

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy traci sumolib tensorboard
```

### 2. 测试系统

```bash
# 测试全局模型
python test.py

# 测试路口级模型（推荐）
python test_junction.py
```

### 3. 查看路口信息

```bash
python junction_main.py info
```

输出：
```
【J5】- 类型: type_a
  主路入边: ['E2']
  匝道入边: ['E23']
  >>> 单纯匝道汇入

【J15】- 类型: type_b
  主路入边: ['E10']
  匝道入边: ['E17']
  匝道出边: ['E16']
  >>> 匝道汇入+主路转出
```

### 4. 训练模型

```bash
# 路口级训练（推荐）
python junction_main.py train --total-timesteps 1000000

# 全局训练
python train.py --total-timesteps 1000000
```

### 5. 评估模型

```bash
python junction_main.py eval --model checkpoints_junction/best_model.pt --episodes 10
```

### 6. 运行推理

```bash
python junction_main.py infer --model checkpoints_junction/best_model.pt
```

## 文件结构

```
rl_traffic/
├── README.md                  # 本文件
├── README_JUNCTION.md         # 路口级方案详细说明
│
├── junction_agent.py          # 路口智能体定义
├── junction_network.py        # 路口级神经网络
├── junction_trainer.py        # 多智能体训练器
├── junction_main.py           # 路口级主入口
├── test_junction.py           # 路口级测试
│
├── config.py                  # 全局模型配置
├── environment.py             # 全局环境
├── network.py                 # 全局网络
├── ppo.py                     # 全局PPO
├── train.py                   # 全局训练
├── main.py                    # 全局主入口
├── test.py                    # 全局测试
│
├── advanced_model.py          # 高级模型（可选）
├── requirements.txt           # 依赖
└── checkpoints/               # 模型保存目录
```

## 路口级方案详解

### 路口拓扑

```
J5 (类型A) ──→ J14 (类型A) ──→ J15 (类型B) ──→ J17 (类型B)
  │                │                │                │
匝道E23汇入     匝道E15汇入     匝道E17汇入      匝道E19汇入
                                  主路转E16       主路转E18/E20
```

### 类型A：单纯匝道汇入

**决策逻辑**:
1. 主路CV车辆是否减速让行
2. 匝道CV车辆何时加速汇入
3. 汇入间隙选择

**网络结构**:
```
状态编码 → 车辆编码 → 空间注意力 → 冲突预测 → 控制头
```

### 类型B：匝道汇入 + 主路转出

**决策逻辑**:
1. 主路CV车辆让行策略
2. 匝道CV车辆汇入时机
3. **转出车辆与汇入车辆的协调**
4. 转出引导

**网络结构**:
```
状态编码 → 车辆编码 → 三方注意力 → 协调模块 → 控制头
```

### 路口间协调

```
J5 ←→ J14 ←→ J15 ←→ J17
 ↓      ↓      ↓      ↓
路口间注意力
 ↓      ↓      ↓      ↓
全局价值网络
```

## 训练建议

### 1. 课程学习

```python
# 阶段1: 单路口训练
python junction_main.py train --junctions J5 --total-timesteps 100000

# 阶段2: 双路口协调
python junction_main.py train --junctions J5,J14 --total-timesteps 200000

# 阶段3: 全路口联合
python junction_main.py train --total-timesteps 500000
```

### 2. 超参数调优

```bash
# 学习率
python junction_main.py train --lr 1e-4

# 批大小
python junction_main.py train --batch-size 128
```

### 3. 监控训练

```bash
# TensorBoard
tensorboard --logdir logs_junction
```

## 评测指标

### 初赛阶段
- **主要指标**: OCR（OD完成率）
- **次要指标**: 稳定性、干预成本（仅供参考）

### OCR计算
```python
OCR = (到达车辆数 + 在途车辆完成度之和) / 总车辆数
```

## 注意事项

1. **初赛限制**
   - 只能控制CV类型车辆
   - 不能修改信号灯配置
   - 不能修改车辆OD路径

2. **训练技巧**
   - 使用课程学习
   - 监控过拟合
   - 定期保存检查点

3. **调试建议**
   - 使用GUI观察策略行为
   - 检查关键路口的决策
   - 分析冲突风险预测

## 下一步改进

1. **层次化强化学习**
   - 上层：路口间协调
   - 下层：路口内控制

2. **图神经网络**
   - 更好地建模路口间关系
   - 自动学习拓扑结构

3. **元学习**
   - 快速适应新路口
   - 少样本学习

4. **迁移学习**
   - 从简单路口迁移到复杂路口
   - 从仿真迁移到真实场景

## 常见问题

### Q1: 为什么推荐路口级方案？
A: 因为拥堵主要来源于路口冲突，路口级建模更精准，状态空间更小，训练更高效。

### Q2: 如何选择路口？
A: 系统自动识别信号灯控制的关键路口（J5, J14, J15, J17），也可以手动指定。

### Q3: 如何处理路口间协调？
A: 通过路口间注意力机制，相邻路口交换信息，实现协调决策。

### Q4: 训练需要多长时间？
A: 取决于硬件和配置，通常100万步需要2-4小时（GPU）。

### Q5: 如何提高OCR？
A: 
1. 增加训练步数
2. 调整奖励函数权重
3. 使用课程学习
4. 优化网络结构

## 联系方式

如有问题，请查看代码注释或提交issue。

---

**祝比赛顺利！** 🚦🚗🏆
