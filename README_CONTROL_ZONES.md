# 控制区域划分版本说明

## 核心问题解决

### 原问题
- 多个路口可能控制同一辆车
- 例如：J14的下游车辆可能是J15的上游车辆
- 控制权冲突导致决策混乱

### 解决方案
- **控制区域划分**：每个路口只控制其上游的CV车辆
- **车辆注册表**：跟踪每辆车的控制权归属
- **不重叠分区**：确保一辆车只被一个路口控制

## 模型输出说明

### 每个路口模型的输出

```python
# 类型A路口（单纯匝道汇入）
output = {
    'main_action': float,   # 主路CV车辆速度控制（0-1）
    'ramp_action': float,   # 匝道CV车辆速度控制（0-1）
    'value': float          # 状态价值
}

# 类型B路口（匝道汇入 + 主路转出）
output = {
    'main_action': float,   # 主路CV车辆速度控制（0-1）
    'ramp_action': float,   # 匝道CV车辆速度控制（0-1）
    'diverge_action': float, # 转出引导（0-1）
    'value': float          # 状态价值
}
```

### 动作含义

**main_action (主路动作)**:
- 范围：0-1
- 映射到速度：`speed = speed_limit * (0.3 + 0.9 * action)`
- 含义：控制主路上游CV车辆的速度
  - 0 = 减速到30%限速（让行）
  - 0.5 = 保持50%限速
  - 1 = 加速到120%限速（正常通行）

**ramp_action (匝道动作)**:
- 范围：0-1
- 含义：控制匝道上游CV车辆的汇入时机
  - 0 = 停止等待
  - 0.5 = 准备汇入
  - 1 = 加速汇入

**diverge_action (转出动作)**:
- 范围：0-1
- 含义：引导转出车辆选择时机
  - 0 = 延迟转出
  - 1 = 立即转出

## 控制区域划分

### 划分原则

1. **上游控制**：每个路口只控制其上游的车辆
2. **不重叠**：控制区域不重叠
3. **连续性**：车辆离开一个区域后，进入下一个区域

### 具体划分

```
J5 控制区域:
├── 主路上游: E2 (范围: 200m)
└── 匝道上游: E23 (范围: 150m)

J14 控制区域:
├── 主路上游: E9 (范围: 200m)
└── 匝道上游: E15 (范围: 150m)

J15 控制区域:
├── 主路上游: E10 (范围: 200m)
├── 匝道上游: E17 (范围: 150m)
└── 转出引导: E16 (范围: 100m)

J17 控制区域:
├── 主路上游: E12 (范围: 200m)
├── 匝道上游: E19 (范围: 150m)
└── 转出引导: E18, E20 (范围: 100m)
```

### 控制链

```
主路控制链:
J5 (E2) → J14 (E9) → J15 (E10) → J17 (E12)
  ↓         ↓          ↓          ↓
控制上游  控制上游   控制上游   控制上游

匝道控制:
J5: E23   J14: E15   J15: E17   J17: E19
```

## 车辆注册表

### 功能

```python
class VehicleRegistry:
    """跟踪每辆车的控制权归属"""
    
    # 车辆 -> 控制路口
    vehicle_to_junction: Dict[str, str]
    
    # 路口 -> 控制车辆
    junction_to_vehicles: Dict[str, Set[str]]
```

### 工作流程

```python
# 1. 更新所有车辆位置
all_vehicles = get_all_vehicles()

# 2. 更新注册表
vehicle_registry.update(all_vehicles)

# 3. 为每辆车分配控制路口
for veh_id, veh_info in all_vehicles.items():
    # 根据车辆位置确定控制路口
    junction = assign_junction(veh_id, veh_info)
    
    # 更新映射
    vehicle_to_junction[veh_id] = junction
    junction_to_vehicles[junction].add(veh_id)
```

### 分配原则

```python
def assign_junction(veh_id, veh_info):
    """
    为车辆分配控制路口
    
    规则：
    1. 检查车辆是否在某个路口的主路上游区域
    2. 检查车辆是否在某个路口的匝道上游区域
    3. 检查车辆是否在某个路口的转出区域
    4. 如果不在任何控制区域，返回None
    """
    edge = veh_info['edge']
    position = veh_info['lane_position']
    
    for junc_id, zone in CONTROL_ZONES.items():
        # 主路上游
        if edge in zone.main_upstream_edges:
            distance_to_junction = edge_length - position
            if distance_to_junction <= zone.main_upstream_range:
                return junc_id
        
        # 匝道上游
        if edge in zone.ramp_upstream_edges:
            distance_to_junction = edge_length - position
            if distance_to_junction <= zone.ramp_upstream_range:
                return junc_id
        
        # 转出区域
        if edge in zone.diverge_edges:
            if position <= zone.diverge_range:
                return junc_id
    
    return None
```

## 动作应用流程

### 步骤

```python
# 1. 每个路口模型生成动作
for junc_id, model in models.items():
    state = get_state(junc_id)
    actions[junc_id] = model(state)

# 2. 验证控制权
for junc_id, action_dict in actions.items():
    # 获取该路口控制的车辆
    controlled = vehicle_registry.get_controlled_vehicles(junc_id)
    
    # 只对控制的车辆应用动作
    for veh_id, action in action_dict.items():
        if veh_id in controlled:
            apply_action(veh_id, action)
        else:
            # 警告：试图控制未授权的车辆
            log_warning(junc_id, veh_id)
```

### 验证机制

```python
def apply_actions_with_validation(actions):
    """应用动作并验证控制权"""
    for junc_id, action_dict in actions.items():
        # 获取该路口控制的车辆
        controlled = agent.get_controlled_vehicles()
        all_controlled = set(controlled['main'] + controlled['ramp'] + controlled['diverge'])
        
        for veh_id, action in action_dict.items():
            # 验证1：车辆在控制列表中
            if veh_id not in all_controlled:
                continue
            
            # 验证2：注册表确认
            controlling_junction = vehicle_registry.get_controlling_junction(veh_id)
            if controlling_junction != junc_id:
                continue
            
            # 应用动作
            apply_action(veh_id, action)
```

## 示例场景

### 场景1：车辆从J5到J14

```
时刻T1:
  车辆V1在E2（J5的主路上游）
  → J5控制V1
  → J5模型输出：main_action = 0.7
  → V1速度设为：13.89 * (0.3 + 0.9 * 0.7) = 12.5 m/s

时刻T2:
  车辆V1进入E3（J5的下游，J14的上游）
  → J5释放V1
  → J14获得V1控制权
  → J14模型输出：main_action = 0.5
  → V1速度设为：13.89 * (0.3 + 0.9 * 0.5) = 10.4 m/s
```

### 场景2：匝道车辆汇入

```
时刻T1:
  车辆V2在E23（J5的匝道上游）
  → J5控制V2
  → J5模型检测到主路间隙
  → J5模型输出：ramp_action = 0.8
  → V2加速汇入

时刻T2:
  车辆V2成功汇入主路E3
  → J5释放V2
  → J14获得V2控制权
```

## 优势

### 1. 避免控制权冲突
- 每辆车只被一个路口控制
- 明确的控制权转移机制

### 2. 提高决策效率
- 每个路口只关注自己的控制区域
- 减少状态空间维度

### 3. 增强可解释性
- 清晰的控制区域划分
- 明确的责任归属

### 4. 便于调试
- 可以追踪每辆车的控制历史
- 可以分析每个路口的决策效果

## 文件说明

- `junction_control_zones.py` - 控制区域划分实现
- `VehicleRegistry` - 车辆注册表
- `ControlZone` - 控制区域定义
- `JunctionAgentWithZone` - 带控制区域的智能体

## 使用方法

```python
# 创建环境（自动处理控制区域）
env = MultiAgentEnvironmentWithZones(sumo_cfg='...')

# 重置环境
obs = env.reset()

# 每步自动更新控制权
obs, rewards, done, info = env.step(actions)

# 查看控制权分配
summary = env.get_control_summary()
```

## 注意事项

1. **控制范围设置**：根据实际路网调整控制范围
2. **控制权转移**：确保车辆在区域边界平滑转移
3. **异常处理**：处理车辆突然离开的情况
4. **性能监控**：监控每个路口的控制效果
