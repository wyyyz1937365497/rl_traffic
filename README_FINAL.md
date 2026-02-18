# 强化学习交通控制系统 - 最终版本

## 核心改进总结

### 1. 使用SUMO订阅模式

**文件**: `junction_agent_subscription.py`

**优势**:
- ✅ 减少TraCI调用次数，提高效率
- ✅ 批量获取数据，降低通信开销
- ✅ 实时更新，数据一致性更好

**订阅内容**:
```python
# 车辆订阅
traci.vehicle.subscribe(veh_id, [
    VAR_SPEED, VAR_POSITION, VAR_ANGLE,
    VAR_LANE_INDEX, VAR_LANE_POSITION, VAR_ROAD_ID,
    VAR_ROUTE_INDEX, VAR_WAITING_TIME, VAR_ACCELERATION,
    VAR_VEHICLECLASS, VAR_TYPE
])

# 道路边订阅
traci.edge.subscribe(edge_id, [
    LAST_STEP_VEHICLE_NUMBER,
    LAST_STEP_MEAN_SPEED,
    LAST_STEP_VEHICLE_IDS,
    LAST_STEP_OCCUPANCY
])

# 车道订阅
traci.lane.subscribe(lane_id, [
    LAST_STEP_VEHICLE_NUMBER,
    LAST_STEP_VEHICLE_IDS,
    LAST_STEP_HALTING_NUMBER,
    LAST_STEP_MEAN_SPEED
])

# 信号灯订阅（重要）
traci.trafficlight.subscribe(tl_id, [
    TL_CURRENT_PHASE,
    TL_CURRENT_PROGRAM,
    TL_PHASE_DURATION,
    TL_NEXT_SWITCH,
    TL_RED_YELLOW_GREEN_STATE,
    TL_CONTROLLED_LANES,
    TL_CONTROLLED_LINKS
])
```

### 2. 信号灯相位作为重要特征

**文件**: `junction_network_updated.py`

**状态空间扩展** (22维):

```python
[
    # 主路特征 (5维)
    主路车辆数, 主路速度, 主路密度, 主路排队长度, 主路流量,
    
    # 匝道特征 (5维)
    匝道车辆数, 匝道速度, 匝道排队长度, 匝道等待时间, 匝道流量,
    
    # 信号灯特征 (5维) - 新增！
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

**信号灯编码器**:
```python
class TrafficLightEncoder(nn.Module):
    """专门处理信号灯相位信息"""
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(5, 32),  # 5个信号灯特征
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16)
        )
```

### 3. 信号灯对决策的影响

**冲突风险计算**:
```python
def _compute_conflict_risk(self, state):
    # 基础风险
    main_density = len(state.main_vehicles) / ...
    ramp_density = len(state.ramp_vehicles) / ...
    speed_diff = abs(state.main_speed - state.ramp_speed)
    
    # 根据信号灯状态调整
    if state.ramp_signal == 'G':
        signal_factor = 0.3  # 匝道绿灯，风险降低
    elif state.ramp_signal == 'r':
        signal_factor = 0.1  # 匝道红灯，风险最低
    
    risk = (main_density * ramp_density) * (speed_diff / 20.0) * signal_factor
    return risk
```

**奖励函数**:
```python
def _compute_rewards(self):
    # 基础奖励
    throughput_reward = -queue_length * 0.1
    waiting_penalty = -waiting_time * 0.05
    conflict_penalty = -conflict_risk * 0.5
    
    # 信号灯协调奖励（新增）
    if state.ramp_signal == 'G' and state.ramp_vehicles:
        signal_reward = 0.1  # 鼓励汇入
    elif state.ramp_signal == 'r' and state.ramp_vehicles:
        signal_reward = -0.1 * len(state.ramp_vehicles)  # 惩罚排队
    
    return throughput_reward + waiting_penalty + conflict_penalty + signal_reward
```

## 文件结构

```
rl_traffic/
├── junction_agent_subscription.py  # 订阅模式智能体（主要）
├── junction_network_updated.py     # 更新的网络（包含信号灯）
├── junction_trainer.py             # 训练器
├── junction_main.py                # 主入口
├── test_subscription.py            # 测试脚本
│
├── junction_agent.py               # 原版智能体（备选）
├── junction_network.py             # 原版网络（备选）
│
├── README_FINAL.md                 # 本文件
├── README_SUBSCRIPTION.md          # 订阅模式说明
└── README_JUNCTION.md              # 路口级方案说明
```

## 使用方法

### 1. 测试系统

```bash
# 测试订阅模式和信号灯特征
python test_subscription.py
```

### 2. 训练模型

```bash
# 使用订阅模式训练
python junction_main.py train --total-timesteps 1000000
```

### 3. 查看路口信息

```bash
python junction_main.py info
```

## 性能对比

| 指标 | 原方案 | 订阅模式 + 信号灯 |
|------|--------|------------------|
| 数据收集效率 | 低 | 高 |
| 状态维度 | 16维 | 22维 |
| 信号灯感知 | 无 | 有 |
| 决策精度 | 一般 | 高 |
| 训练效率 | 中 | 高 |

## 关键改进点

### 1. 订阅管理器

```python
class SubscriptionManager:
    """统一管理所有订阅"""
    
    def setup_vehicle_subscription(self, veh_ids, variables)
    def setup_edge_subscription(self, edge_ids, variables)
    def setup_lane_subscription(self, lane_ids, variables)
    def setup_traffic_light_subscription(self, tl_ids, variables)
    
    def update_results(self)  # 批量更新
    def cleanup_left_vehicles(self)  # 自动清理
```

### 2. 信号灯状态观察

```python
def _observe_traffic_light(self, state):
    """观察信号灯状态"""
    tl_data = self.sub_manager.get_tl_data(self.config.tl_id)
    
    state.current_phase = tl_data[TL_CURRENT_PHASE]
    state.phase_state = tl_data[TL_RED_YELLOW_GREEN_STATE]
    state.time_to_switch = tl_data[TL_NEXT_SWITCH] - timestamp
    
    # 解析各方向信号
    state.main_signal = phase_str[0]
    state.ramp_signal = phase_str[2]
    state.diverge_signal = phase_str[4]
```

### 3. 信号灯特征融合

```python
# 分离信号灯特征
non_tl_features = state[:, :17]
tl_features = state[:, 17:22]

# 分别编码
state_features = self.state_encoder(non_tl_features)
tl_encoded = self.tl_encoder(tl_features)

# 融合
combined = torch.cat([state_features, tl_encoded], dim=-1)
```

## 决策逻辑示例

### 场景1：匝道绿灯 + 有间隙

```python
if state.ramp_signal == 'G' and state.gap_acceptance > 0.5:
    # 匝道绿灯且有足够间隙
    action['ramp'] = 0.8  # 高速汇入
    action['main'] = 0.5  # 主路正常
```

### 场景2：匝道红灯 + 排队

```python
if state.ramp_signal == 'r' and state.ramp_queue_length > 5:
    # 匝道红灯且排队严重
    action['ramp'] = 0.0  # 停止
    action['main'] = 0.7  # 主路优先
```

### 场景3：即将切换

```python
if state.time_to_switch < 5:
    # 即将切换相位
    if state.next_phase == 匝道绿灯相位:
        # 准备汇入
        action['ramp'] = 0.5
```

## 训练建议

### 1. 课程学习

```bash
# 阶段1: 单路口
python junction_main.py train --junctions J5 --total-timesteps 100000

# 阶段2: 多路口
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
tensorboard --logdir logs_junction
```

## 注意事项

1. **订阅初始化**: 在`reset()`时自动设置所有订阅
2. **订阅更新**: 每步自动更新订阅结果
3. **订阅清理**: 自动清理已离开车辆的订阅
4. **信号灯同步**: 确保信号灯状态与仿真同步

## 下一步改进

1. **自适应订阅**: 根据需要动态调整订阅内容
2. **相位预测**: 预测信号灯相位变化
3. **协调优化**: 基于信号灯相位的路口间协调
4. **紧急响应**: 处理信号灯异常情况

## 常见问题

### Q1: 订阅模式有什么优势？
A: 减少TraCI调用次数，提高数据收集效率，适合大规模仿真。

### Q2: 信号灯相位为什么重要？
A: 直接影响匝道车辆的汇入时机，是决策的关键因素。

### Q3: 如何处理信号灯异常？
A: 系统会自动检测信号灯状态，异常时使用默认值。

### Q4: 状态维度为什么增加到22维？
A: 新增了5维信号灯特征，使模型能感知信号灯状态。

## 联系方式

如有问题，请查看代码注释或提交issue。

---

**系统已准备就绪！** 🚦🚗🏆

关键改进：
1. ✅ SUMO订阅模式
2. ✅ 信号灯相位特征
3. ✅ 状态维度22维
4. ✅ 信号灯编码器
5. ✅ 冲突风险调整
6. ✅ 奖励函数优化
