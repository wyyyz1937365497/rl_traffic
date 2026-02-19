"""
检查SUMO TraCI常量名称
找出正确的订阅变量常量
"""

try:
    import libsumo as traci
    print("✓ 使用 libsumo")
except ImportError:
    import traci
    print("✓ 使用 traci")

# 打印所有包含 LAST_STEP 的常量
print("\n=== LAST_STEP 相关常量 ===")
constants_with_last_step = []
for attr in dir(traci.constants):
    if 'LAST_STEP' in attr:
        constants_with_last_step.append(attr)
        print(f"  traci.constants.{attr}")

print(f"\n共找到 {len(constants_with_last_step)} 个 LAST_STEP 常量")

# 打印所有 VAR 相关常量
print("\n=== VAR_* 相关常量（用于vehicle订阅） ===")
var_constants = []
for attr in dir(traci.constants):
    if attr.startswith('VAR_'):
        var_constants.append(attr)
        print(f"  traci.constants.{attr}")

print(f"\n共找到 {len(var_constants)} 个 VAR_ 常量")

# 尝试启动SUMO并测试订阅
print("\n=== 测试订阅 ===")
try:
    import os
    import sys

    # 查找sumo配置文件
    sumo_cfg = None
    for cfg_path in ['sumo/sumo.sumocfg', 'sumo.sumocfg']:
        if os.path.exists(cfg_path):
            sumo_cfg = cfg_path
            break

    if sumo_cfg is None:
        print("⚠️  未找到SUMO配置文件，跳过订阅测试")
    else:
        print(f"✓ 找到配置文件: {sumo_cfg}")

        # 启动SUMO
        sumo_cmd = ["sumo", "-c", sumo_cfg, "--no-warnings", "true"]
        traci.start(sumo_cmd)
        print("✓ SUMO已启动")

        # 测试edge订阅
        print("\n测试 edge 订阅:")

        # 尝试不同的常量组合
        test_variables = [
            [traci.constants.LAST_STEP_VEHICLE_NUMBER],
            [traci.constants.LAST_STEP_MEAN_SPEED],
            [traci.constants.LAST_STEP_VEHICLE_DATA],  # 可能的正确名称
        ]

        for i, var in enumerate(test_variables):
            try:
                # 假设有一个边叫 'E2'
                traci.edge.subscribe('E2', var)
                result = traci.edge.getSubscriptionResults('E2')
                print(f"  ✓ 测试 {i} 成功: {var}")
                print(f"    结果类型: {type(result)}")
                if result:
                    print(f"    结果内容: {result}")
            except AttributeError as e:
                print(f"  ✗ 测试 {i} 失败: {var}")
                print(f"    错误: {e}")
            except Exception as e:
                print(f"  ✗ 测试 {i} 其他错误: {e}")

        traci.close()
        print("\n✓ SUMO已关闭")

except Exception as e:
    print(f"⚠️  订阅测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 完成 ===")
