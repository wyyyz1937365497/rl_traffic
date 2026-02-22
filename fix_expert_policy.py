"""
修复专家策略参数，使其与26分脚本一致
关键问题：BC专家策略使用了错误的accel参数
"""

import os

# 对比分析
print("=" * 80)
print("26分脚本 vs BC专家策略 - 参数对比")
print("=" * 80)

print("\n关键参数差异:")
print("-" * 80)
print(f"{'参数':<20} {'26分脚本':<20} {'BC专家策略':<20} {'是否一致'}")
print("-" * 80)
print(f"{'CV accel':<20} {'2.1':<20} {'0.8':<20} {'❌ 不一致'}")
print(f"{'CV tau':<20} {'0.9':<20} {'0.9':<20} {'✓ 一致'}")
print(f"{'CV sigma':<20} {'0.0':<20} {'0.0':<20} {'✓ 一致'}")
print(f"{'控制间隔':<20} {'5步':<20} {'5步':<20} {'✓ 一致'}")
print(f"{'接近距离':<20} {'50米':<20} {'50米':<20} {'✓ 一致'}")
print(f"{'拥堵速度阈值':<20} {'5.0 m/s':<20} {'5.0 m/s':<20} {'✓ 一致'}")
print(f"{'速度系数':<20} {'1.5':<20} {'1.5':<20} {'✓ 一致'}")

print("\n" + "=" * 80)
print("影响分析")
print("=" * 80)

print("""
accel=0.8 vs accel=2.1 的影响:

1. 加速性能差异
   - accel=0.8: 车辆加速缓慢，需要更长时间达到目标速度
   - accel=2.1: 车辆加速较快，能更快响应控制指令

2. 吞吐量影响
   - 较低的accel会导致车辆起步慢，减少路网吞吐量
   - 这可能解释了为什么BC克隆的性能不如26分脚本

3. 稳定性影响
   - accel=0.8会让速度变化更平缓（看似更好）
   - 但同时也降低了系统的响应能力和吞吐量

4. 行为不一致
   - BC模型学习的是accel=0.8的行为模式
   - 但部署时如果使用accel=2.1，会有巨大的行为gap
   - 这个gap会导致策略完全失效
""")

print("\n" + "=" * 80)
print("修复方案")
print("=" * 80)

print("""
需要修改 collect_expert_demos.py 中的ExpertPolicy.configure_vtypes():

修改前:
    traci.vehicletype.setAccel('CV', 0.8)
    traci.vehicletype.setAccel('HV', 0.8)

修改后:
    traci.vehicletype.setAccel('CV', 2.1)
    traci.vehicletype.setAccel('HV', 2.1)

同时建议添加decel参数（26分脚本也设置decel）:
    traci.vehicletype.setDecel('CV', 4.5)  # SUMO默认值
    traci.vehicletype.setDecel('HV', 4.5)
""")

# 实际修改代码
file_path = 'collect_expert_demos.py'

print("\n正在修改文件:", file_path)

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换accel参数
old_accel_cv = "traci.vehicletype.setAccel('CV', 0.8)"
new_accel_cv = "traci.vehicletype.setAccel('CV', 2.1)"

old_accel_hv = "traci.vehicletype.setAccel('HV', 0.8)"
new_accel_hv = "traci.vehicletype.setAccel('HV', 2.1)"

if old_accel_cv in content:
    content = content.replace(old_accel_cv, new_accel_cv)
    print(f"✓ 已修改: CV accel 0.8 -> 2.1")
else:
    print(f"⚠ 未找到: {old_accel_cv}")

if old_accel_hv in content:
    content = content.replace(old_accel_hv, new_accel_hv)
    print(f"✓ 已修改: HV accel 0.8 -> 2.1")
else:
    print(f"⚠ 未找到: {old_accel_hv}")

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n" + "=" * 80)
print("下一步操作")
print("=" * 80)

print("""
1. 重新收集专家演示数据:
   python collect_expert_demos.py --episodes 10 --output demos/accel_21/

2. 重新训练BC模型:
   python behavior_cloning.py --train-demos demos/accel_21/ --output bc_checkpoints/accel_21/

3. 测试新模型:
   python generate_submit_from_model.py --checkpoint bc_checkpoints/accel_21/best.pt --output submit_bc_accel21.pkl

4. 对比性能:
   - 原BC模型 (accel=0.8)
   - 新BC模型 (accel=2.1)
   - 26分脚本 (accel=2.1 + 规则控制)

预期结果:
   - 新BC模型应该更接近26分脚本的行为
   - OCR应该保持在95%左右
   - 但吞吐量可能会有所提升
""")

print("\n" + "=" * 80)
print("超越OCR的优化建议")
print("=" * 80)

print("""
当OCR已经达到95%时，优化重点应该转向:

1. 【稳定性优化】(权重0.5)
   - 目标: 降低速度标准差和加速度标准差
   - 方法:
     a) 使用更温和的控制策略
     b) 增加控制间隔（减少干预频率）
     c) 仅在关键时刻干预（如接近边末端时）

2. 【干预成本优化】(指数惩罚)
   - 目标: 减少 C_int = (α×acmd + β×δlc) / (T×N)
   - 方法:
     a) 减少控制频率（从每步改为每5步）
     b) 减少动作变化幅度
     c) 智能选择需要控制的车辆

3. 【参数调优】
   基于26分脚本的成功经验:
   - 较高的accel (2.1): 保证吞吐量
   - 温和的主动控制 (仅50米内干预): 保持速度平滑
   - 适当的控制间隔 (5步): 减少干预成本
   - vType参数优化 (sigma=0, tau=0.9): 改善跟驰

4. 【车辆级控制 vs 路口级控制】
   - 当前架构: 路口级控制（同一类车共享动作）
   - 优化方向: 车辆级控制（每辆车独立动作）
   - 潜在收益: 更精细的控制，可以进一步降低速度标准差

关键洞察:
26分脚本的成功 = 高OCR(95%) + 速度平滑(σv低) + 低干预成本
     NOT just 高OCR
""")

print("\n完成! 请重新收集数据和训练模型。\n")
