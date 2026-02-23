# 并行BC数据收集 - 快速开始

## 一分钟快速开始

```bash
# 1. 测试并行收集器（2个episodes）
python test_parallel_collection.py

# 2. 收集训练数据（50个episodes，8个并行worker）
python collect_expert_demos_parallel.py \
    --sumo-cfg sumo/sumo.sumocfg \
    --num-episodes 50 \
    --num-workers 8 \
    --output-dir expert_demos/accel_21

# 3. 训练BC模型
python behavior_cloning.py \
    --train-demos expert_demos/accel_21 \
    --output-dir bc_checkpoints/accel_21

# 4. 生成提交文件并评估
python generate_submit_from_model.py \
    --checkpoint bc_checkpoints/accel_21/best.pt \
    --output submit_bc.pkl

python simple_score_calculator.py submit_bc.pkl
```

## 关键修复

✓ **已修复accel参数**：从0.8更新为2.1（与26分脚本一致）

## 常用命令

### 不同规模的收集

```bash
# 测试（2 episodes）
python collect_expert_demos_parallel.py --num-episodes 2 --num-workers 2 --output-dir test

# 小规模（20 episodes）
python collect_expert_demos_parallel.py --num-episodes 20 --num-workers 4 --output-dir demos/small

# 中规模（50 episodes，推荐）
python collect_expert_demos_parallel.py --num-episodes 50 --num-workers 8 --output-dir demos/medium

# 大规模（100 episodes）
python collect_expert_demos_parallel.py --num-episodes 100 --num-workers 8 --output-dir demos/large
```

### 不同并行度

```bash
# 4 workers（笔记本）
python collect_expert_demos_parallel.py --num-episodes 50 --num-workers 4

# 8 workers（台式机）
python collect_expert_demos_parallel.py --num-episodes 50 --num-workers 8

# 12 workers（高性能工作站）
python collect_expert_demos_parallel.py --num-episodes 100 --num-workers 12
```

## 验证数据质量

```bash
# 方法1：查看生成的文件
ls -lh expert_demos/accel_21/

# 方法2：统计文件数量
ls expert_demos/accel_21/*.pkl | wc -l

# 方法3：检查数据内容
python -c "
import pickle
import glob
files = glob.glob('expert_demos/accel_21/*.pkl')
print(f'Total episodes: {len(files)}')
for f in sorted(files)[:3]:
    with open(f, 'rb') as file:
        data = pickle.load(file)
    print(f'{f}: {len(data[\"transitions\"])} transitions, OCR={data[\"final_ocr\"]:.3f}')
"
```

## 性能预期

| 配置 | Episode数 | Workers | 预计时间 | 数据量 |
|------|-----------|---------|----------|--------|
| 测试 | 2 | 2 | ~1分钟 | ~2,800 transitions |
| 小规模 | 20 | 4 | ~5分钟 | ~28,000 transitions |
| 中规模 | 50 | 8 | ~10分钟 | ~72,000 transitions |
| 大规模 | 100 | 8 | ~20分钟 | ~144,000 transitions |

## 故障排查

### SUMO进程未关闭
```bash
# Windows
taskkill /F /IM sumo.exe

# Linux/Mac
killall -9 sumo
```

### 内存不足
```bash
# 减少worker数量
python collect_expert_demos_parallel.py --num-episodes 50 --num-workers 4
```

### 端口被占用
```bash
# 修改端口（在脚本中设置环境变量）
export TRACI_PORT=XXXX  # Linux/Mac
set TRACI_PORT=XXXX     # Windows
```

## 下一步

收集数据后，参考以下文档：

1. **训练BC模型**：`behavior_cloning.py --help`
2. **生成分数**：`generate_submit_from_model.py --help`
3. **本地评估**：`simple_score_calculator.py --help`

## 完整文档

详细说明请参考：
- `PARALLEL_BC_GUIDE.md` - 完整使用指南
- `LOCAL_SCORE_README.md` - 本地分数计算器使用说明
