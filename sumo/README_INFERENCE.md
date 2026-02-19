# SUMO比赛推理指南

## 文件说明

`sumo/main.py` 是集成了强化学习模型的比赛提交文件，用于：
1. 加载训练好的模型
2. 在SUMO仿真中应用模型控制
3. 生成比赛要求的 `submit.pkl` 文件

## 使用方法

### 1. 准备模型

确保模型文件存在：
```bash
# 模型应该训练完成并保存为
checkpoints/final_model.pt
```

### 2. 运行推理

在sumo目录下运行：

```bash
cd sumo
python main.py
```

### 3. 输出结果

运行完成后会生成：

```
competition_results/
└── submit.pkl    # 比赛提交文件
```

## 代码结构

### 第一部分：环境初始化
```python
framework.parse_config()         # 解析SUMO配置
framework.parse_routes()         # 解析路径文件
framework.initialize_environment()  # 启动SUMO
framework.load_rl_model()        # 加载RL模型
```

### 第二部分：仿真运行
```python
framework.run_simulation()       # 运行3600步仿真
# - 观察环境状态
# - 模型推理生成动作
# - 应用速度控制
# - 收集数据
```

### 第三部分：保存结果
```python
framework.save_to_pickle()       # 生成submit.pkl
```

## 模型控制逻辑

### 控制对象
- **主路车辆**：速度范围 [15, 25] m/s
- **匝道车辆**：速度范围 [10, 20] m/s

### 控制策略
1. 每个时间步观察路口状态
2. 提取状态特征（队列长度、速度、密度等）
3. 模型推理生成连续动作
4. 将动作映射到目标速度
5. 通过 `traci.vehicle.setSpeed()` 应用控制

### 状态特征（17维）
```python
[
    main_queue_length,      # 主路队列长度
    ramp_queue_length,      # 匝道队列长度
    main_speed / 20.0,      # 主路平均速度（归一化）
    ramp_speed / 20.0,      # 匝道平均速度（归一化）
    main_density / 0.5,     # 主路密度（归一化）
    ramp_density / 0.5,     # 匝道密度（归一化）
    ramp_waiting_time / 60, # 匝道等待时间（归一化）
    gap_size / 10.0,        # 车辆间隙（归一化）
    gap_speed_diff / 20.0,  # 速度差（归一化）
    has_cv,                 # 是否有CV车辆
    conflict_risk,          # 冲突风险
    main_stop_count / 10,   # 主路停车计数
    ramp_stop_count / 10,   # 匝道停车计数
    throughput / 100.0,     # 通过量（归一化）
    phase_main,             # 主路相位
    phase_ramp,             # 匝道相位
    time_step               # 时间步
]
```

## 输出文件格式

`submit.pkl` 包含以下数据：

```python
{
    'parameters': {
        'flow_rate': float,
        'simulation_time': float,
        'total_steps': int,
        'total_demand': int,
        'final_departed': int,
        'final_arrived': int,
        'unique_vehicles': int,
        'model_used': str  # 使用的模型路径
    },
    'step_data': [...],      # 每步统计数据
    'vehicle_data': [...],   # 每辆车每个时间步的数据
    'route_data': {...},     # 车辆路径信息
    'vehicle_od_data': {...},# 车辆OD信息
    'statistics': {
        'all_departed_vehicles': [...],
        'all_arrived_vehicles': [...],
        'cumulative_departed': int,
        'cumulative_arrived': int,
        'maxspeed_violations': [...]  # 速度违规记录
    }
}
```

## 自定义配置

### 修改模型路径

在 `main.py` 第 709 行修改：

```python
framework = SUMOCompetitionFramework(
    sumo_cfg_path="sumo.sumocfg",
    model_path="../checkpoints/best_model.pt"  # 修改这里
)
```

### 修改仿真步数

在 `run_simulation()` 方法第 500 行修改：

```python
while step < 7200:  # 修改为7200步
    # ...
```

### 修改输出目录

在 `main()` 第 722 行修改：

```python
framework.save_to_pickle(output_dir="my_results")  # 修改输出目录
```

## 常见问题

### Q1: 模型加载失败

**错误信息**：
```
⚠️  模型文件不存在: ../checkpoints/final_model.pt
```

**解决方法**：
1. 确保已经完成训练
2. 检查模型文件路径是否正确
3. 训练命令：`python rl_train.py --sumo-cfg sumo/sumo.sumocfg --total-timesteps 1000000 --workers 16`

### Q2: CUDA相关错误

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
在 `load_rl_model()` 方法中添加：
```python
self.device = 'cpu'  # 强制使用CPU
```

### Q3: SUMO启动失败

**错误信息**：
```
traci.exceptions.TraCIException: Could not start SUMO
```

**解决方法**：
1. 检查SUMO是否已安装：`sumo --version`
2. 检查配置文件路径：`sumo.sumocfg`
3. 确保在sumo目录下运行

### Q4: pkl文件位置

**默认位置**：
```
G:\TJ\rl_traffic\sumo\competition_results\submit.pkl
```

这是比赛提交的文件。

## 测试和验证

### 快速测试

修改仿真步数进行快速测试：

```python
# 第500行
while step < 100:  # 只运行100步
```

### 验证模型是否生效

查看输出日志：

```
正在加载RL模型: ../checkpoints/final_model.pt
✓ 模型已加载到 cuda
✓ 已创建 X 个RL智能体

[第二部分] 开始仿真...
设备: cuda
模型状态: 已加载  # 确认模型已加载
```

### 检查控制效果

在 `apply_control_algorithm()` 方法中添加调试输出：

```python
# 第220行后添加
print(f"控制车辆 {veh_id}: 速度 {target_speed:.2f} m/s")
```

## 性能优化

### 使用GPU推理

确保CUDA可用：
```python
import torch
print(torch.cuda.is_available())  # 应该输出True
```

### 加速模型推理

在 `load_rl_model()` 后添加：
```python
# 编译模型（PyTorch 2.0+）
self.model = torch.compile(self.model)
```

## 比赛提交清单

提交前确认：

- [ ] 模型已训练完成
- [ ] `submit.pkl` 文件已生成
- [ ] OCR（完成率）达到预期
- [ ] 无maxSpeed违规
- [ ] 仿真完整运行3600步
- [ ] 数据格式符合比赛要求

## 提交文件示例

```
submission.zip
├── sumo/
│   ├── main.py              # 提交的代码
│   ├── sumo.sumocfg         # SUMO配置
│   ├── net.xml              # 路网文件
│   └── routes.xml           # 路径文件
└── checkpoints/
    └── final_model.pt       # 训练好的模型
```

## 联系和支持

如有问题，检查：
1. 训练日志：`logs/`
2. 模型文件：`checkpoints/`
3. SUMO版本：`sumo --version`
4. Python环境：`python --version`
