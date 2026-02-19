# 订阅数据访问修复报告

## 🐛 问题描述

### 症状
所有奖励都是 0.0000，状态值全是0：

```
[INFO] 路口 J5 奖励: 0.0000 (队列:0.0/0.0, 等待:0.0)
[INFO] 路口 J14 奖励: 0.0000 (队列:0.0/0.0, 等待:0.0)
```

---

## 🔍 根本原因

### 订阅数据访问不匹配

**问题**：订阅时使用常量**值**，访问时使用常量**名称**，导致键不匹配。

#### 订阅时（常量值）
```python
# junction_agent_subscription.py:260-265
def setup_edge_subscription(self, edge_ids: List[str], variables: List[int] = None):
    if variables is None:
        variables = [
            0x11,  # LAST_STEP_VEHICLE_NUMBER
            0x12,  # VAR_LAST_STEP_MEAN_SPEED
            0x13,  # VAR_LAST_STEP_VEHICLE_DATA ✅ 使用常量值
            0x14   # VAR_LAST_STEP_OCCUPANCY
        ]

    traci.edge.subscribe(edge_id, variables)
```

**订阅结果存储**：
```python
# junction_agent_subscription.py:369-371
results = traci.edge.getSubscriptionResults(edge_id)
# results = {0x11: value1, 0x12: value2, 0x13: value3, 0x14: value4}
#           ^^^^ 常量值作为键
self.edge_results[edge_id] = results
```

#### 访问时（常量名）❌
```python
# 修复前 - junction_agent_subscription.py:593
def _get_vehicles_from_edges(self, edge_ids: List[str]) -> List[Dict]:
    edge_data = self.sub_manager.get_edge_data(edge_id)

    if edge_data:
        # ❌ 使用常量名访问，但键是常量值！
        veh_ids = edge_data.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, [])
        #    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 返回 None/默认值 []！

    return []  # 因为 veh_ids = []
```

**结果**：
- `edge_data.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, [])` 返回 `[]`
- 所有状态值都是0
- 所有奖励都是0

---

## ✅ 解决方案

### 统一使用常量值

**原则**：订阅和访问都使用常量值

#### 修复1: 车辆ID获取
```python
# 修复后
def _get_vehicles_from_edges(self, edge_ids: List[str]) -> List[Dict]:
    edge_data = self.sub_manager.get_edge_data(edge_id)

    if edge_data:
        # ✅ 使用常量值 0x13
        veh_ids = edge_data.get(0x13, [])  # VAR_LAST_STEP_VEHICLE_DATA

        for veh_id in veh_ids:
            veh_data = self.sub_manager.get_vehicle_data(veh_id)

            if veh_data:
                # ✅ 所有车辆数据也使用常量值
                veh_info = {
                    'id': veh_id,
                    'speed': veh_data.get(0x40, 0),      # VAR_SPEED
                    'position': veh_data.get(0x42, (0, 0)),  # VAR_POSITION
                    'lane': veh_data.get(0x53, 0),       # VAR_LANE_INDEX
                    'lane_position': veh_data.get(0x43, 0),  # VAR_LANE_POSITION
                    'waiting_time': veh_data.get(0x4B, 0),   # VAR_WAITING_TIME
                    'accel': veh_data.get(0x4A, 0),      # VAR_ACCELERATION
                    'is_cv': veh_data.get(0x4D, '') == 'CV',  # VAR_VEHICLECLASS
                    'route_index': veh_data.get(0x56, 0)    # VAR_ROUTE_INDEX
                }
                vehicles.append(veh_info)

    return vehicles
```

#### 修复2: 平均速度获取
```python
# 修复后
def _get_mean_speed(self, edge_ids: List[str]) -> float:
    for edge_id in edge_ids:
        edge_data = self.sub_manager.get_edge_data(edge_id)
        if edge_data:
            # ✅ 使用常量值 0x12
            speed = edge_data.get(0x12, -1)  # VAR_LAST_STEP_MEAN_SPEED
            if speed >= 0:
                speeds.append(speed)

    return np.mean(speeds) if speeds else 0.0
```

#### 修复3: 排队长度获取
```python
# 修复后
def _get_queue_length(self, edge_ids: List[str]) -> int:
    for edge_id in edge_ids:
        lane_data = self.sub_manager.get_lane_data(lane_id)

        if lane_data:
            # ✅ 使用常量值 0x10
            halting = lane_data.get(0x10, 0)  # LAST_STEP_VEHICLE_NUMBER
            queue_length += halting

    return queue_length
```

#### 修复4: 车辆数获取
```python
# 修复后
def _compute_flow(self, edge_ids: List[str], mean_speed: float) -> float:
    for edge_id in edge_ids:
        edge_data = self.sub_manager.get_edge_data(edge_id)
        if edge_data:
            # ✅ 使用常量值 0x11
            total_vehicles += edge_data.get(0x11, 0)  # LAST_STEP_VEHICLE_NUMBER

    return mean_speed * total_vehicles
```

#### 修复5: 信号灯状态获取
```python
# 修复后
def _observe_traffic_light(self, state: JunctionState) -> JunctionState:
    tl_data = self.sub_manager.get_tl_data(self.config.tl_id)

    if tl_data:
        # ✅ 使用常量值
        state.current_phase = tl_data.get(0x50, 0)  # TL_CURRENT_PHASE
        state.phase_state = tl_data.get(0x59, "")  # VAR_TL_RED_YELLOW_GREEN_STATE
        next_switch = tl_data.get(0x5A, 0)  # VAR_TL_NEXT_SWITCH

    return state
```

---

## 📊 常量值对照表

### 边订阅常量

| 常量值 | 说明 | 位置 |
|--------|------|------|
| 0x11 | LAST_STEP_VEHICLE_NUMBER | 车辆数量 |
| 0x12 | VAR_LAST_STEP_MEAN_SPEED | 平均速度 |
| 0x13 | VAR_LAST_STEP_VEHICLE_DATA | 车辆ID列表 |
| 0x14 | VAR_LAST_STEP_OCCUPANCY | 占用率 |

### 车辆订阅常量

| 常量值 | 说明 | 位置 |
|--------|------|------|
| 0x40 | VAR_SPEED | 速度 |
| 0x41 | VAR_ANGLE | 角度 |
| 0x42 | VAR_POSITION | 位置 |
| 0x43 | VAR_LANE_POSITION | 车道位置 |
| 0x4A | VAR_ACCELERATION | 加速度 |
| 0x4B | VAR_WAITING_TIME | 等待时间 |
| 0x4D | VAR_VEHICLECLASS | 车辆类型 |
| 0x4E | VAR_TYPE | 类型 |
| 0x53 | VAR_LANE_INDEX | 车道索引 |
| 0x56 | VAR_ROUTE_INDEX | 路线索引 |

### 车道订阅常量

| 常量值 | 说明 | 位置 |
|--------|------|------|
| 0x10 | LAST_STEP_VEHICLE_NUMBER | 停车数量 |
| 0x11 | LAST_STEP_VEHICLE_NUMBER | 车辆数量 |
| 0x12 | VAR_LAST_STEP_MEAN_SPEED | 平均速度 |
| 0x13 | VAR_LAST_STEP_VEHICLE_DATA | 车辆数据 |

### 信号灯订阅常量

| 常量值 | 说明 | 位置 |
|--------|------|------|
| 0x50 | TL_CURRENT_PHASE | 当前相位 |
| 0x51 | VAR_TL_CURRENT_PROGRAM | 当前程序 |
| 0x59 | VAR_TL_RED_YELLOW_GREEN_STATE | 信号状态 |
| 0x5A | VAR_TL_NEXT_SWITCH | 下次切换 |

---

## 🧪 验证修复

### 测试命令

```bash
python rl_train.py --sumo-cfg sumo/sumo.sumocfg --total-timesteps 10000 --workers 1
```

### 预期日志（修复后）

**成功标志**：
```
[INFO] 路口 J5 奖励: -0.1234 (队列:2.0/1.0, 等待:5.2)  ← ✅ 非零值！
[INFO] 路口 J14 奖励: -0.0891 (队列:1.0/0.0, 等待:3.1)
[INFO] 路口 J15 奖励: -0.1456 (队列:3.0/2.0, 等待:6.7)
[INFO] 路口 J17 奖励: -0.0678 (队列:1.0/1.0, 等待:2.4)
```

**不应该再看到**：
```
[INFO] 路口 J5 奖励: 0.0000 (队列:0.0/0.0, 等待:0.0)  ← ❌ 全是0
```

---

## 💡 关键要点

1. **订阅和访问必须一致**
   - 订阅时用常量值 → 访问时也用常量值
   - 订阅时用常量名 → 访问时也用常量名

2. **推荐使用常量值**
   - ✅ 兼容所有SUMO版本
   - ✅ 更稳定（常量名可能变化）
   - ✅ 更清晰（直接看到数值）

3. **调试技巧**
   - 检查 `edge_data.keys()` 确认键的类型
   - 使用 `print(list(edge_data.keys()))` 查看实际键
   - 对比订阅常量和访问键是否匹配

---

## 🎯 总结

### 修复内容

✅ **修复所有订阅数据访问**
- `_get_vehicles_from_edges()`: 车辆数据获取
- `_get_mean_speed()`: 平均速度获取
- `_get_queue_length()`: 排队长度获取
- `_compute_flow()`: 流量计算
- `_observe_traffic_light()`: 信号灯状态

✅ **统一使用常量值**
- 订阅时：`0x13`, `0x12`, `0x11`, `0x10` 等
- 访问时：`0x13`, `0x12`, `0x11`, `0x10` 等
- 完全一致！

✅ **预期效果**
- 状态值不再是0
- 奖励不再是0
- 训练能够正常进行

---

## 🚀 立即测试

```bash
python rl_train.py --sumo-cfg sumo/sumo.sumocfg --total-timesteps 10000 --workers 1
```

预期看到：
```
[INFO] 路口 J5 奖励: -0.1234 (队列:2.0/1.0, 等待:5.2)  ← 非零值！
✅ 订阅数据访问正常
✅ 状态值正确获取
✅ 奖励计算正常
```

祝测试顺利！🎉
