# 路口级多智能体系统 - 订阅模式版本

## 核心改进

### 1. 使用SUMO订阅模式

**优势**:
- 减少TraCI调用次数，提高效率
- 批量获取数据，降低通信开销
- 实时更新，数据一致性更好

**订阅内容**:
```python
# 车辆订阅
- 速度、位置、角度
- 车道索引、车道位置
- 道路ID、路径索引
- 等待时间、加速度
- 车辆类型

# 道路边订阅
- 车辆数量、平均速度
- 车辆ID列表、占用率

# 车道订阅
- 车辆数量、停止车辆数
- 平均速度

# 信号灯订阅（重要）
- 当前相位
- 信号状态字符串
- 下次切换时间
- 控制车道列表
```

### 2. 信号灯相位作为重要特征

**状态空间扩展** (22维):

```python
[
    # 主路特征 (5维)
    主路车辆数, 主路速度, 主路密度, 主路排队长度, 主路流量,
    
    # 匝道特征 (5维)
    匝道车辆数, 匝道速度, 匝道排队长度, 匝道等待时间, 匝道流量,
    
    # 信号灯特征 (5维) - 重要！
    当前相位索引,
    距离下次切换时间,
    主路信号状态(是否绿灯),
    匝道信号状态(是否绿灯),
    转出信号状态(是否绿灯),
    
    # 冲突特征 (2维)
    冲突风险,
    可接受间隙,
    
    # CV车辆 (2维)
    主路CV比例,
    匝道CV比例,
    
    # 类型B特有 (3维)
    转出车辆数,
    转出排队长度,
    转出CV比例,
    
    # 时间 (1维)
    时间戳
]
```

### 3. 信号灯相位的影响

**对决策的影响**:

1. **匝道绿灯时**:
   - 匝道车辆可以汇入
   - 冲突风险降低
   - 鼓励匝道CV加速汇入

2. **匝道红灯时**:
   - 匝道车辆必须等待
   - 主路车辆优先通行
   - 惩罚匝道排队

3. **主路绿灯时**:
   - 主路车辆正常通行
   - 可能需要为匝道车辆让行

4. **相位切换前**:
   - 预测即将发生的变化
   - 提前调整策略

**对奖励的影响**:

```python
# 信号灯协调奖励
if 匝道绿灯 and 匝道有车辆:
    signal_reward = 0.1  # 鼓励汇入
elif 匝道红灯 and 匝道有车辆:
    signal_reward = -0.1 * 排队数  # 惩罚排队
```

**对冲突风险的影响**:

```python
# 根据信号灯状态调整风险
if 匝道绿灯:
    signal_factor = 0.3  # 风险降低
elif 匝道红灯:
    signal_factor = 0.1  # 风险最低
```

## 文件结构

```
rl_traffic/
├── junction_agent_subscription.py  # 使用订阅模式的智能体
├── junction_network.py              # 神经网络（已更新状态维度）
├── junction_trainer.py              # 训练器
├── junction_main.py                 # 主入口
└── README_SUBSCRIPTION.md           # 本文件
```

## 使用方法

### 1. 使用订阅模式版本

```python
from junction_agent_subscription import MultiAgentEnvironment

# 创建环境（自动使用订阅模式）
env = MultiAgentEnvironment(
    sumo_cfg='path/to/sumo.sumocfg',
    use_gui=False,
    seed=42
)

# 重置环境（自动设置订阅）
obs = env.reset()

# 观察状态（自动更新订阅）
obs, rewards, done, info = env.step(actions)
```

### 2. 查看信号灯状态

```python
# 获取路口智能体
agent = env.get_agent('J15')

# 获取当前状态
state = agent.current_state

# 查看信号灯信息
print(f"当前相位: {state.current_phase}")
print(f"主路信号: {state.main_signal}")
print(f"匝道信号: {state.ramp_signal}")
print(f"距离切换: {state.time_to_switch}秒")
```

### 3. 训练模型

```bash
# 使用订阅模式训练
python junction_main.py train --total-timesteps 1000000
```

## 性能对比

| 指标 | 原方案（逐个查询） | 订阅模式 |
|------|------------------|---------|
| 数据收集效率 | 低 | 高 |
| TraCI调用次数 | 多 | 少 |
| 通信开销 | 大 | 小 |
| 数据一致性 | 一般 | 好 |
| 适用场景 | 小规模 | 大规模 |

## 信号灯相位详解

### 相位状态字符串

SUMO使用字符串表示信号灯状态，例如：
- `"GGrrGG"`: 
  - 前两个字符：主路信号
  - 中间两个：匝道信号
  - 后两个：转出信号

### 信号字符含义

- `G`: 绿灯（通行）
- `r`: 红灯（停止）
- `y`: 黄灯（警告）
- `g`: 绿灯（次要方向）

### 相位切换

```python
# 获取下次切换时间
next_switch = traci.trafficlight.getNextSwitch(tl_id)

# 获取当前相位
current_phase = traci.trafficlight.getPhase(tl_id)

# 预测下一个相位
next_phase = (current_phase + 1) % num_phases
```

## 网络结构更新

### 输入层

```python
# 状态维度从16维增加到22维
state_dim = 22

# 包含信号灯特征
state = [
    ...,
    当前相位,          # 新增
    距离切换时间,      # 新增
    主路信号状态,      # 新增
    匝道信号状态,      # 新增
    转出信号状态,      # 新增
    ...
]
```

### 信号灯编码器

```python
class TrafficLightEncoder(nn.Module):
    """信号灯特征编码器"""
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(5, 16),  # 5个信号灯特征
            nn.ReLU(),
            nn.Linear(16, 16)
        )
    
    def forward(self, tl_features):
        return self.encoder(tl_features)
```

### 融合策略

```python
# 将信号灯特征与其他特征融合
state_features = self.state_encoder(state[:, :17])  # 其他特征
tl_features = self.tl_encoder(state[:, 17:22])      # 信号灯特征

# 融合
combined = torch.cat([state_features, tl_features], dim=-1)
```

## 决策逻辑示例

### 场景1：匝道绿灯

```python
if state.ramp_signal == 'G':
    # 匝道绿灯，鼓励汇入
    if state.gap_acceptance > 0.5:
        # 有足够间隙，加速汇入
        action['ramp'] = 0.8  # 高速汇入
    else:
        # 间隙不足，等待
        action['ramp'] = 0.3
```

### 场景2：匝道红灯

```python
if state.ramp_signal == 'r':
    # 匝道红灯，必须等待
    action['ramp'] = 0.0  # 停止
    
    # 主路可以正常通行
    action['main'] = 0.7
```

### 场景3：即将切换

```python
if state.time_to_switch < 5:
    # 即将切换，预测下一个相位
    if state.next_phase == 匝道绿灯相位:
        # 匝道即将变绿，准备汇入
        action['ramp'] = 0.5
```

## 注意事项

1. **初始化订阅**: 在`reset()`时自动设置所有订阅
2. **更新订阅**: 每步自动更新订阅结果
3. **清理订阅**: 自动清理已离开车辆的订阅
4. **信号灯同步**: 确保信号灯状态与仿真同步

## 下一步改进

1. **自适应订阅**: 根据需要动态调整订阅内容
2. **预测模型**: 预测信号灯相位变化
3. **协调优化**: 基于信号灯相位的路口间协调
4. **紧急响应**: 处理信号灯异常情况

## 常见问题

### Q1: 订阅模式有什么优势？
A: 减少TraCI调用次数，提高数据收集效率，适合大规模仿真。

### Q2: 信号灯相位为什么重要？
A: 直接影响匝道车辆的汇入时机，是决策的关键因素。

### Q3: 如何处理信号灯异常？
A: 系统会自动检测信号灯状态，异常时使用默认值。

### Q4: 订阅数据如何更新？
A: 每步自动调用`update_results()`更新所有订阅数据。

## 联系方式

如有问题，请查看代码注释或提交issue。
