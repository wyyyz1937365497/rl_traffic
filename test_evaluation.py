"""
快速测试评估脚本

用于快速验证模型和评估流程是否正常工作
"""

import os
import sys

def test_evaluation():
    """运行测试评估"""
    print("="*70)
    print("快速评估测试")
    print("="*70)

    # 检查是否有检查点文件
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print("\n[错误] 找不到 checkpoints 目录")
        print("请先运行训练：python rl_train.py --sumo-cfg sumo/sumo.sumocfg")
        return False

    # 查找最新的检查点
    checkpoint_files = list(checkpoints_dir.glob("checkpoint_iter_*.pt"))
    if not checkpoint_files:
        print("\n[错误] 在 checkpoints 目录中没有找到检查点文件")
        return False

    # 按迭代次数排序
    import re
    def get_iteration(file_path):
        match = re.search(r'checkpoint_iter_(\d+)\.pt', file_path.name)
        return int(match.group(1)) if match else 0

    latest = max(checkpoint_files, key=get_iteration)
    print(f"\n[找到] 最新检查点: {latest.name}")

    # 检查SUMO配置文件
    sumo_cfg = "sumo/sumo.sumocfg"
    if not os.path.exists(sumo_cfg):
        print(f"\n[错误] 找不到SUMO配置文件: {sumo_cfg}")
        return False

    print(f"[找到] SUMO配置文件: {sumo_cfg}")

    # 运行快速评估（较少步数）
    print("\n[开始] 运行快速评估（100步）...")
    print("="*70)

    import subprocess

    # 使用evaluate_model_compliant.py运行评估
    cmd = [
        'python', 'evaluate_model_compliant.py',
        '--model-path', str(latest),
        '--sumo-cfg', sumo_cfg,
        '--iteration', 0,  # 测试用迭代号
        '--eval-dir', 'test_eval',
        '--max-steps', '100'  # 只运行100步
    ]

    print(f"执行命令: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n" + "="*70)
        print("[成功] 快速评估完成!")
        print("="*70)
        print(f"\n测试结果保存在: test_eval/")
        print(f"\n如果一切正常，可以运行完整评估：")
        print(f"  python run_evaluation.py --checkpoint {latest}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 评估失败")
        print(f"返回码: {e.returncode}")
        return False

if __name__ == "__main__":
    from pathlib import Path

    success = test_evaluation()
    sys.exit(0 if success else 1)
