# 模型输入输出详细说明

## 问题回答

### 问题1：模型是否直接操控SUMO中的CV车辆？

**答案：是的，模型直接操控SUMO中的CV车辆。**

#### 控制机制

```python
# 模型输出动作
action = model(state)  # action范围: 0-1

# 转换为速度
speed_limit = 13.89  # m/s
target_speed = speed_limit * (0.3 + 0.9 * action)

# 直接控制CV车辆
for cv_id in controlled_cv_vehicles:
    traci.vehicle.setSpeed(cv_id, target_speed)
```

#### 控制范围

- **只控制CV车辆**：模型输出只应用于CV类型车辆
- **不控制HV车辆**：HV车辆由SUMO自动驾驶模型控制
- **控制区域限制**：只控制控制区域内的CV车辆

#### 控制流程

```
1. 获取所有车辆 → 区分CV和HV
2. 筛选控制区域内的CV车辆
3. 模型生成动作（速度比例）
4. 应用动作到CV车辆
   └─ traci.vehicle.setSpeed(cv_id, target_speed)
5. HV车辆保持SUMO默认行为
```

### 问题2：模型输入的图是否包含CV与HV车辆的标识？

**答案：是的，模型输入包含CV和HV车辆的标识。**

#### 车辆特征向量（包含CV/HV标识）

```python
vehicle_features = [
    position_x,          # 位置x
    position_y,          # 位置y
    speed,               # 速度
    acceleration,        # 加速度
    lane_position,       # 车道位置
    lane_index,          # 车道索引
    route_progress,      # 路径进度
    waiting_time,        # 等待时间
    is_cv,              # ★ 是否是CV（1.0=CV, 0.0=HV）
    is_controlled,      # 是否被控制（只有CV可被控制）
    in_conflict_zone,   # 是否在冲突区域
    conflict_severity,  # 冲突严重程度
    ...
]
```

#### 车道特征向量（包含CV/HV统计）

```python
lane_features = [
    length,              # 长度
    speed_limit,         # 限速
    lane_index,          # 车道索引
    is_ramp,            # 是否匝道
    is_rightmost,       # 是否最右侧
    total_vehicles,     # 总车辆数
    cv_count,           # ★ CV车辆数
    hv_count,           # ★ HV车辆数
    cv_ratio,           # ★ CV比例
    mean_speed,         # 平均速度
    mean_speed_cv,      # ★ CV平均速度
    mean_speed_hv,      # ★ HV平均速度
    queue_length,       # 排队长度
    queue_cv,           # ★ CV排队数
    queue_hv,           # ★ HV排队数
    density,            # 密度
    has_conflict,       # 是否有冲突
    conflict_severity,  # 冲突严重程度
    ...
]
```

#### 图结构

```python
graph = {
    'nodes': [
        # 所有车辆节点（CV和HV）
        {
            'id': 'veh_001',
            'type': 'vehicle',
            'features': [..., is_cv=1.0, ...]  # CV车辆
        },
        {
            'id': 'veh_002',
            'type': 'vehicle',
            'features': [..., is_cv=0.0, ...]  # HV车辆
        },
        # 车道节点
        {
            'id': 'E11_0',
            'type': 'lane',
            'features': [..., cv_count=5, hv_count=10, ...]
        },
        ...
    ],
    'edges': [
        # 车辆-车道关系
        {
            'source': 'veh_001',
            'target': 'E11_0',
            'type': 'vehicle_on_lane'
        },
        # 车道-车道冲突关系
        {
            'source': 'E17_0',
            'target': '-E11_0',
            'type': 'conflict',
            'severity': 0.8
        },
        ...
    ]
}
```

## 完整的车道级冲突矩阵

### J5路口

```
匝道E23汇入主路

车道冲突：
E23_0 (匝道) → -E3_0 (主路最外侧) [严重度: 0.8]
E23_0 (匝道) → -E3_1 (主路内侧)   [严重度: 0.6]

说明：
- E23只有1条车道
- -E3有2条车道，都与匝道冲突
- 最外侧车道冲突更严重
```

### J14路口

```
匝道E15汇入主路

车道冲突：
E15_0 (匝道) → -E10_0 (主路最外侧) [严重度: 0.8]
E15_0 (匝道) → -E10_1 (主路内侧)   [严重度: 0.6]

说明：
- E15只有1条车道
- -E10有2条车道，都与匝道冲突
```

### J15路口（复杂：汇入+转出）

```
匝道E17汇入主路 + 主路转出E16

车道冲突：
E17_0 (匝道) → -E11_0 (主路最外侧) [严重度: 0.8]
E17_0 (匝道) → -E11_1 (主路中间)   [严重度: 0.6]
E17_0 (匝道) → -E11_2 (主路最内侧) [严重度: 0.0] ← 不冲突！

关键发现：
- -E11有3条车道
- 匝道只与前2条车道冲突
- 最内侧车道(-E11_2)不与匝道冲突！

转出冲突：
-E11_0 → E16_1 (转出匝道) [严重度: 0.7]
E10_1 → E16_1 (主路转出) [严重度: 0.5]
```

### J17路口（复杂：汇入+转出）

```
匝道E19汇入主路 + 主路转出E18/E20

车道冲突：
E19_0 (匝道第1条) → -E13_0 (主路最外侧) [严重度: 0.8]
E19_0 (匝道第1条) → -E13_1 (主路中间)   [严重度: 0.6]
E19_1 (匝道第2条) → -E13_0 (主路最外侧) [严重度: 0.7]

关键发现：
- E19有2条车道（其他匝道只有1条）
- -E13有3条车道
- 匝道只与前2条车道冲突
- 最内侧车道(-E13_2)不与匝道冲突！

转出冲突：
-E13_0 → E20_0 (转出匝道) [严重度: 0.7]
E12_0 → E18_0 (主路转出) [严重度: 0.5]
```

## 模型架构

### 输入层

```python
class InputLayer:
    """处理CV和HV车辆"""
    
    def forward(self, observation):
        # 分离CV和HV
        all_vehicles = observation['vehicles']
        cv_vehicles = [v for v in all_vehicles if v['is_cv']]
        hv_vehicles = [v for v in all_vehicles if not v['is_cv']]
        
        # 构建图
        graph = {
            'cv_nodes': self.encode_vehicles(cv_vehicles),
            'hv_nodes': self.encode_vehicles(hv_vehicles),
            'lane_nodes': self.encode_lanes(observation['lanes']),
            'edges': self.build_edges(observation)
        }
        
        return graph
```

### 编码层

```python
class VehicleEncoder:
    """车辆编码器（区分CV和HV）"""
    
    def forward(self, vehicle_features):
        # vehicle_features包含is_cv字段
        is_cv = vehicle_features[:, 8]  # 第9个特征
        
        # CV和HV使用不同的编码
        cv_features = self.cv_encoder(vehicle_features[is_cv == 1])
        hv_features = self.hv_encoder(vehicle_features[is_cv == 0])
        
        return cv_features, hv_features
```

### 输出层

```python
class OutputLayer:
    """输出层（只控制CV车辆）"""
    
    def forward(self, encoded_features):
        # 生成动作
        action = self.action_head(encoded_features)
        
        # 只返回CV车辆的控制
        return {
            'action': action,
            'controlled_vehicles': cv_vehicle_ids  # 只有CV
        }
```

## 控制示例

### 场景：J15路口

```python
# 1. 获取状态
observation = env.get_observation()

# 车辆信息（包含CV/HV标识）
vehicles = {
    'veh_001': {'is_cv': True, 'lane': 'E17_0', ...},   # CV，在匝道
    'veh_002': {'is_cv': False, 'lane': '-E11_0', ...}, # HV，在主路
    'veh_003': {'is_cv': True, 'lane': '-E11_2', ...},  # CV，在主路最内侧
    'veh_004': {'is_cv': True, 'lane': 'E10_0', ...},   # CV，在主路上游
}

# 车道信息（包含CV/HV统计）
lanes = {
    'E17_0': {'cv_count': 1, 'hv_count': 0, ...},
    '-E11_0': {'cv_count': 0, 'hv_count': 1, ...},
    '-E11_1': {'cv_count': 0, 'hv_count': 0, ...},
    '-E11_2': {'cv_count': 1, 'hv_count': 0, ...},  # 不与匝道冲突
}

# 2. 模型推理
output = model(observation)

# 输出
{
    'main_action': 0.7,      # 主路CV控制
    'ramp_action': 0.8,      # 匝道CV控制
    'controlled_vehicles': {
        'main': ['veh_003', 'veh_004'],  # 主路CV
        'ramp': ['veh_001']              # 匝道CV
    }
}

# 3. 应用控制
# 只控制CV车辆
traci.vehicle.setSpeed('veh_001', 13.89 * (0.3 + 0.9 * 0.8))  # 匝道CV
traci.vehicle.setSpeed('veh_003', 13.89 * (0.3 + 0.9 * 0.7))  # 主路CV
traci.vehicle.setSpeed('veh_004', 13.89 * (0.3 + 0.9 * 0.7))  # 主路CV

# HV车辆(veh_002)不控制，由SUMO自动驾驶
```

## 关键结论

### 1. 模型输入

✅ **包含CV和HV车辆的标识**
- 车辆特征中有 `is_cv` 字段
- 车道特征中有 `cv_count`, `hv_count`, `cv_ratio`
- 图节点包含所有车辆（CV和HV）

### 2. 模型输出

✅ **只控制CV车辆**
- 输出动作只应用于CV车辆
- HV车辆由SUMO自动控制
- 输出包含被控制的CV车辆ID列表

### 3. 车道级建模

✅ **精确定义了车道冲突**
- J15: 匝道只与-E11前2条车道冲突，不与-E11_2冲突
- J17: 匝道只与-E13前2条车道冲突，不与-E13_2冲突
- 区分了不同车道的冲突严重程度

### 4. 控制机制

✅ **直接控制SUMO中的CV车辆**
- 使用 `traci.vehicle.setSpeed()` 控制速度
- 只控制控制区域内的CV车辆
- HV车辆保持SUMO默认行为
