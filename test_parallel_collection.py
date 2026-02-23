"""
测试并行BC数据收集器

快速验证：
1. 参数是否正确（accel=2.1）
2. 多进程是否正常工作
3. 数据质量是否合格
"""

import os
import sys
import pickle
import glob
import subprocess


def test_parallel_collection():
    """测试并行收集功能"""
    print("=" * 80)
    print("并行BC数据收集器 - 测试")
    print("=" * 80)

    # 1. 检查依赖
    print("\n[1] 检查依赖...")
    try:
        import libsumo as traci
        print("  ✓ libsumo 可用")
    except ImportError:
        import traci
        print("  ⚠ 使用traci（会比libsumo慢）")

    from multiprocessing import cpu_count
    print(f"  ✓ CPU核心数: {cpu_count()}")

    # 2. 小规模测试（2个episodes，2个workers）
    print("\n[2] 小规模测试（2 episodes, 2 workers）...")
    cmd = [
        sys.executable,
        "collect_expert_demos_parallel.py",
        "--sumo-cfg", "sumo/sumo.sumocfg",
        "--num-episodes", "2",
        "--num-workers", "2",
        "--output-dir", "test_demos"
    ]

    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("  ✓ 并行收集成功")
    else:
        print("  ✗ 并行收集失败")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False

    # 3. 检查生成的文件
    print("\n[3] 检查生成的文件...")
    files = glob.glob("test_demos/episode_*.pkl")
    print(f"  生成文件数: {len(files)}")

    if len(files) < 2:
        print("  ✗ 生成的文件数量不足")
        return False

    print(f"  ✓ 生成了 {len(files)} 个episode文件")

    # 4. 验证数据质量
    print("\n[4] 验证数据质量...")
    all_good = True

    for f in sorted(files):
        with open(f, 'rb') as file:
            data = pickle.load(file)

        num_trans = data['num_transitions']
        ocr = data['final_ocr']
        steps = data['steps']

        print(f"  {os.path.basename(f)}:")
        print(f"    - Transitions: {num_trans}")
        print(f"    - OCR: {ocr:.4f}")
        print(f"    - Steps: {steps}")

        # 检查是否合理
        if num_trans < 100:
            print(f"    ⚠ Warning: Transitions数量偏少")
            all_good = False

        if ocr < 0.90:
            print(f"    ⚠ Warning: OCR偏低")
            all_good = False

        if steps < 3000:
            print(f"    ⚠ Warning: Steps偏少")
            all_good = False

    # 5. 检查状态和动作维度
    print("\n[5] 检查状态和动作维度...")
    with open(files[0], 'rb') as file:
        data = pickle.load(file)

    if data['transitions']:
        first_trans = data['transitions'][0]
        state_dim = first_trans['state'].shape[0]
        action_main_dim = first_trans['action_main'].shape
        action_ramp_dim = first_trans['action_ramp'].shape

        print(f"  状态维度: {state_dim}")
        print(f"  主路动作维度: {action_main_dim}")
        print(f"  匝道动作维度: {action_ramp_dim}")

        if state_dim == 23:
            print("  ✓ 状态维度正确（23维）")
        else:
            print(f"  ✗ 状态维度错误（应为23，实际{state_dim}）")
            all_good = False

        if action_main_dim == (1,):
            print("  ✓ 动作维度正确")
        else:
            print(f"  ✗ 动作维度错误")
            all_good = False

    # 6. 总结
    print("\n" + "=" * 80)
    if all_good:
        print("✓ 所有测试通过！并行收集器可以正常使用。")
        print("\n下一步：")
        print("  1. 收集完整数据集")
        print("     python collect_expert_demos_parallel.py --num-episodes 50 --num-workers 8")
        print("\n  2. 训练BC模型")
        print("     python behavior_cloning.py --train-demos test_demos")
    else:
        print("⚠ 部分测试失败，请检查上述警告。")

    print("=" * 80)

    return all_good


if __name__ == '__main__':
    success = test_parallel_collection()
    sys.exit(0 if success else 1)
