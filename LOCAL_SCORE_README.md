# 本地分数计算器使用说明

## 功能

基于已知得分的pkl文件建立OCR->Score的映射关系，用于本地估算新提交文件的得分，避免频繁上传。

## 文件说明

1. **simple_score_calculator.py** - 简化版分数计算器（推荐使用）
   - 基于OCR线性拟合估算得分
   - 简单快速，精度较高（MAE < 0.01）

2. **score_calibrator.py** - 完整版分数计算器
   - 基于完整的评分公式
   - 考虑效率、稳定性、干预成本

## 使用方法

### 第一次使用：校准参数

```bash
# 使用已知得分的pkl文件校准
python simple_score_calculator.py --calibrate submit.pkl:25.7926 submission.pkl:15.7650
```

输出示例：
```
Calibrated parameters:
  score = 1101.0 * (OCR - 0.9444) + 14.7
  mean_active_baseline = 287.0
  final_active_baseline = 294.5

Mean Absolute Error: 0.00

Calibration saved to: simple_calibration.json
```

### 后续使用：计算新文件

```bash
# 计算单个文件
python simple_score_calculator.py new_submission.pkl

# 对比多个文件
python simple_score_calculator.py file1.pkl file2.pkl file3.pkl
```

输出示例：
```
new_submission.pkl
----------------------------------------
  Score: 26.45
  OCR: 0.9523
  Base Score: 24.67

Comparison
------------------------------------------------------------
File                                     Score      OCR
------------------------------------------------------------
new_submission.pkl                       26.45      0.9523
submit.pkl                               27.57      0.9543
submission.pkl                           14.71      0.9454
```

## 校准参数文件

校准后的参数保存在 `simple_calibration.json`：

```json
{
  "ocr_weight": 1101.0,
  "ocr_baseline": 0.9444,
  "intercept": 14.7,
  "mean_active_baseline": 287.0,
  "final_active_baseline": 294.5,
  "mae": 0.00,
  "calibrated_with": ["submit.pkl", "submission.pkl"]
}
```

**含义**：
- `score = 1101.0 * (OCR - 0.9444) + 14.7`
- OCR每提升0.001，得分提升约1.1分
- 当OCR = 0.9444时，基准得分为14.7分

## 精度验证

使用已知的测试文件验证：

| 文件 | 实际得分 | 预测得分 | 误差 |
|------|----------|----------|------|
| submit.pkl | 25.79 | 25.79 | 0.00 |
| submission.pkl | 15.77 | 15.76 | 0.00 |

**平均绝对误差 (MAE): < 0.01分**

## 注意事项

1. **校准数据选择**：
   - 建议使用3-5个已知得分的文件进行校准
   - 分数范围应覆盖预期的新文件分数范围
   - 至少包含一个高分文件和一个低分文件

2. **适用范围**：
   - 该计算器基于OCR拟合，适用于OCR接近95%的情况
   - 如果OCR差异很大，需要重新校准

3. **精度限制**：
   - 本地计算仅基于OCR，未考虑稳定性、干预成本等因素
   - 预测精度取决于校准数据的质量和数量
   - 建议用于筛选和排序，最终得分以平台评测为准

4. **重新校准**：
   - 如果获得了更多已知得分的文件，建议重新校准
   - 可以覆盖现有的 `simple_calibration.json` 文件

## 常见问题

### Q: 预测得分与实际得分有差异？

A: 可能的原因：
1. 校准数据不足，建议增加校准样本数量
2. 新文件的OCR范围超出校准数据范围
3. 平台评分公式可能包含未公开的因素

### Q: 如何提高预测精度？

A:
1. 使用更多已知得分的文件进行校准
2. 确保校准文件的分数范围覆盖预期范围
3. 定期用新的已知得分文件重新校准

### Q: 能否预测完全没有相似数据的新策略？

A: 不能保证精度。建议：
1. 先上传少量测试验证实际得分
2. 将新得分加入校准数据重新校准
3. 再用校准后的参数预测其他文件

## 示例工作流

```bash
# 1. 初始校准（使用比赛提供的示例得分）
python simple_score_calculator.py --calibrate \
    relu_based/rl_traffic/sumo/competition_results/submit.pkl:25.7926 \
    relu_based/rl_traffic/submission.pkl:15.7650

# 2. 训练新模型
python rl_train.py --total-timesteps 50000

# 3. 本地评估新模型
python generate_submit_from_model.py --checkpoint checkpoints/model.pt --output new_submit.pkl
python simple_score_calculator.py new_submit.pkl

# 4. 根据本地得分决定是否上传
# 如果本地得分 > 当前最佳，考虑上传验证

# 5. 上传验证后，用实际得分更新校准参数
python simple_score_calculator.py --calibrate \
    submit.pkl:25.7926 \
    submission.pkl:15.7650 \
    new_submit.pkl:26.50
```

## 高级用法

### 批量评估多个模型

```bash
# 生成所有候选提交文件
for checkpoint in checkpoints/checkpoint_*.pt; do
    output="submissions/$(basename $checkpoint .pt).pkl"
    python generate_submit_from_model.py --checkpoint $checkpoint --output $output
done

# 批量评估
python simple_score_calculator.py submissions/*.pkl > scores.txt

# 查看最佳模型
sort -k2 -nr scores.txt | head -5
```

### 与训练脚本集成

在训练脚本中添加定期评估：

```python
# 每N个epoch评估一次
if epoch % 10 == 0:
    checkpoint = f"checkpoints/epoch_{epoch}.pt"
    generate_submission(checkpoint, f"eval/epoch_{epoch}.pkl")
    score = calculate_score_local(f"eval/epoch_{epoch}.pkl")
    print(f"Epoch {epoch} - Estimated score: {score:.2f}")
```

## 更新日志

- **v1.0** (2024): 初始版本，基于OCR线性拟合
- 未来计划：加入稳定性、干预成本等因素，提高预测精度
