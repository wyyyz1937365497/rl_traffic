# 车道级建模 + 可微动作空间

## 问题分析

### 问题1：车道级别建模缺失

**当前问题**：
- 只考虑边级别特征，忽略了车道差异
- 例如J15：匝道汇入只与-E11-1（最外侧车道）冲突，不与-E11-2冲突
- 冲突预测不够精准

**具体例子**：
```
J15路口：
匝道E17汇入主路E11

主路E11有3条车道：
- E11_0 (最外侧) ← 与匝道冲突！
- E11_1 (中间)   ← 与匝道冲突！
- E11_2 (最内侧) ← 不冲突

当前模型：将E11作为整体处理
问题：无法区分不同车道的影响
```

### 问题2：动作不可微

**当前问题**：
- 动作是离散的（0-1跳跃）
- 使用argmax或采样，梯度无法回传
- 对模型训练不利

**具体表现**：
```python
# 当前方法
action_idx = argmax(action_probs)  # 不可微！
action_value = action_idx / 10.0   # 离散跳跃

# 问题：
# 1. argmax操作梯度为0或未定义
# 2. 模型无法通过动作值学习
# 3. 训练不稳定
```

## 解决方案

### 方案1：车道级精细建模

#### 1.1 车道冲突矩阵

```python
# 预定义车道冲突关系
LANE_CONFLICTS = {
    'E17_0': LaneConflict(
        lane_id='E17_0',  # 匝道车道
        conflicts_with=['-E11_0', '-E11_1'],  # 只与最外侧两条车道冲突
        conflict_type='merge',
        severity=0.8
    ),
}
```

#### 1.2 车道级特征

```python
@dataclass
class LaneFeatures:
    lane_id: str
    edge_id: str
    lane_index: int
    
    # 车道属性
    length: float
    speed_limit: float
    is_ramp: bool          # 是否是匝道
    is_rightmost: bool     # 是否是最右侧车道
    
    # 车辆状态
    vehicle_count: int
    mean_speed: float
    queue_length: int
    density: float
    
    # 冲突信息
    has_conflict: bool
    conflict_lanes: List[str]
    conflict_severity: float
```

#### 1.3 车道编码器

```python
class LaneEncoder(nn.Module):
    """车道级特征编码器"""
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(12, 32),  # 12维车道特征
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16)   # 16维输出
        )
    
    def forward(self, lane_features):
        # lane_features: [batch, num_lanes, 12]
        return self.encoder(lane_features)
```

#### 1.4 车道冲突注意力

```python
class LaneConflictAttention(nn.Module):
    """学习冲突车道之间的关系"""
    
    def __init__(self):
        super().__init__()
        
        self.conflict_attention = nn.MultiheadAttention(
            16, 2, batch_first=True
        )
    
    def forward(self, lane_features, conflict_mask):
        # 使用冲突掩码进行注意力计算
        attended = self.conflict_attention(
            lane_features, lane_features, lane_features,
            key_padding_mask=conflict_mask
        )
        return attended + lane_features
```

### 方案2：可微动作空间

#### 2.1 Gumbel-Softmax方法

```python
class DifferentiableActionLayer(nn.Module):
    """使用Gumbel-Softmax实现可微的离散动作"""
    
    def forward(self, x, hard=False):
        # 计算logits
        logits = self.action_logits(x)
        
        # Gumbel-Softmax（可微！）
        if self.training:
            action_probs = F.gumbel_softmax(
                logits, 
                tau=self.temperature,
                hard=False  # 软采样，可微
            )
        else:
            action_probs = F.softmax(logits, dim=-1)
        
        # 计算期望动作值（连续，可微）
        action_values = torch.linspace(0, 1, self.num_actions)
        action_value = torch.sum(action_probs * action_values, dim=-1)
        
        return action_probs, action_value
```

**Gumbel-Softmax原理**：
```
标准Softmax:
    p_i = exp(logit_i) / sum(exp(logit_j))

Gumbel-Softmax:
    1. 采样Gumbel噪声: g_i ~ Gumbel(0, 1)
    2. 计算: y_i = exp((logit_i + g_i) / tau) / sum(...)
    3. 当tau→0时，y→one-hot
    4. 当tau>0时，y是连续的，梯度可传播
```

#### 2.2 连续动作方法

```python
class ContinuousActionLayer(nn.Module):
    """直接输出连续动作值，完全可微"""
    
    def forward(self, x, deterministic=False):
        # 均值网络
        mean = self.mean_net(x)  # [0, 1]
        
        # 标准差网络（用于探索）
        std = self.std_net(x)
        
        # 采样
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # 重参数化采样
        action = torch.clamp(action, 0, 1)
        
        return action, log_prob
```

## 完整架构

### 车道级策略网络

```python
class LaneLevelPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 车道编码器
        self.lane_encoder = LaneEncoder()
        
        # 车道冲突注意力
        self.conflict_attention = LaneConflictAttention()
        
        # 全局特征编码
        self.global_encoder = nn.Sequential(...)
        
        # 信号灯编码
        self.tl_encoder = nn.Sequential(...)
        
        # 融合层
        self.fusion = nn.Sequential(...)
        
        # 可微动作层
        self.main_action = DifferentiableActionLayer()
        self.ramp_action = DifferentiableActionLayer()
    
    def forward(self, lane_features, global_features, tl_features):
        # 1. 编码车道特征
        lane_encoded = self.lane_encoder(lane_features)
        
        # 2. 车道冲突注意力
        lane_attended = self.conflict_attention(lane_encoded)
        
        # 3. 聚合车道特征（区分主路和匝道）
        main_lane_features = lane_attended[:, :num_main_lanes, :].mean(dim=1)
        ramp_lane_features = lane_attended[:, num_main_lanes:, :].mean(dim=1)
        
        # 4. 融合全局和信号灯特征
        main_fused = self.fusion(torch.cat([
            main_lane_features, global_encoded, tl_encoded
        ], dim=-1))
        
        # 5. 输出可微动作
        main_probs, main_value = self.main_action(main_fused)
        ramp_probs, ramp_value = self.ramp_action(ramp_fused)
        
        return {
            'main_action_value': main_value,  # 可微！
            'ramp_action_value': ramp_value,  # 可微！
            ...
        }
```

## 对比分析

### 车道级建模对比

| 维度 | 边级建模 | 车道级建模 |
|------|---------|-----------|
| 粒度 | 粗粒度 | **细粒度** |
| 冲突预测 | 不精准 | **精准** |
| 状态维度 | 低 | **高** |
| 可解释性 | 中 | **高** |

### 动作空间对比

| 维度 | 离散动作 | 可微动作 |
|------|---------|---------|
| 梯度传播 | 不可微 | **可微** |
| 训练稳定性 | 低 | **高** |
| 动作平滑性 | 跳跃 | **平滑** |
| 收敛速度 | 慢 | **快** |

## 实验验证

### 梯度传播测试

```python
# 测试可微动作层的梯度传播
action_layer = DifferentiableActionLayer(input_dim=32)
x = torch.randn(4, 32, requires_grad=True)

# 前向传播
action_probs, action_value = action_layer(x)

# 反向传播
loss = action_value.sum()
loss.backward()

# 验证梯度
print(f"梯度存在: {x.grad is not None}")  # True
print(f"梯度范数: {x.grad.norm()}")       # 非零
```

### 车道冲突验证

```python
# 验证车道冲突矩阵
conflict_matrix = build_conflict_matrix()

# J15路口
# E17_0 (匝道) 应该与 -E11_0, -E11_1 冲突
assert conflict_matrix['E17_0', '-E11_0'] > 0  # 冲突
assert conflict_matrix['E17_0', '-E11_1'] > 0  # 冲突
assert conflict_matrix['E17_0', '-E11_2'] == 0  # 不冲突
```

## 使用方法

### 1. 测试车道级建模

```bash
python lane_level_model.py
```

### 2. 集成到训练

```python
from lane_level_model import LaneLevelPolicyNetwork

# 创建模型
model = LaneLevelPolicyNetwork(config)

# 训练时自动使用可微动作
for batch in dataloader:
    lane_features = batch['lane_features']
    global_features = batch['global_features']
    tl_features = batch['tl_features']
    
    # 前向传播（可微）
    output = model(lane_features, global_features, tl_features)
    
    # 计算损失
    loss = compute_loss(output, targets)
    
    # 反向传播（梯度可传播）
    loss.backward()
    optimizer.step()
```

## 关键优势

### 1. 精准的冲突预测
- ✅ 区分不同车道的影响
- ✅ 预定义冲突关系
- ✅ 学习车道间关系

### 2. 可微的动作空间
- ✅ 梯度可传播
- ✅ 训练稳定
- ✅ 动作平滑

### 3. 更好的可解释性
- ✅ 车道级决策
- ✅ 冲突可视化
- ✅ 动作可解释

## 文件说明

- `lane_level_model.py` - 车道级建模 + 可微动作实现
- `LaneEncoder` - 车道特征编码器
- `LaneConflictAttention` - 车道冲突注意力
- `DifferentiableActionLayer` - 可微动作层
- `ContinuousActionLayer` - 连续动作层

## 注意事项

1. **车道冲突定义**：需要根据实际路网预定义
2. **温度参数**：Gumbel-Softmax的温度需要调优
3. **计算开销**：车道级建模增加计算量
4. **数据需求**：需要车道级别的数据

## 下一步改进

1. **自动冲突检测**：从路网自动提取冲突关系
2. **动态温度调整**：训练过程中调整温度
3. **层次化建模**：车道→边→路口的层次结构
4. **多任务学习**：同时预测冲突和动作
