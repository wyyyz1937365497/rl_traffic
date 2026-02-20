"""
模型评估和比赛pkl文件生成脚本

使用方法:
1. 评估单个检查点:
   python run_evaluation.py --checkpoint checkpoints/checkpoint_iter_0020.pt

2. 评估最新检查点:
   python run_evaluation.py --checkpoint latest

3. 评估所有检查点:
   python run_evaluation.py --checkpoint all

4. 自定义评估:
   python run_evaluation.py --checkpoint checkpoints/checkpoint_iter_0020.pt --steps 3600 --device cuda
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import re

def find_latest_checkpoint():
    """查找最新的检查点文件"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print(f"错误: 找不到 checkpoints 目录")
        return None

    # 查找所有 .pt 文件
    checkpoint_files = list(checkpoints_dir.glob("checkpoint_iter_*.pt"))
    if not checkpoint_files:
        print(f"错误: 在 {checkpoints_dir} 中没有找到检查点文件")
        return None

    # 按迭代次数排序
    def get_iteration(file_path):
        match = re.search(r'checkpoint_iter_(\d+)\.pt', file_path.name)
        return int(match.group(1)) if match else 0

    latest = max(checkpoint_files, key=get_iteration)
    print(f"找到最新检查点: {latest}")
    return str(latest)

def list_all_checkpoints():
    """列出所有检查点文件"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print(f"错误: 找不到 checkpoints 目录")
        return []

    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_iter_*.pt"))

    if not checkpoint_files:
        print(f"错误: 在 {checkpoints_dir} 中没有找到检查点文件")
        return []

    print(f"\n找到 {len(checkpoint_files)} 个检查点:")
    for i, cp in enumerate(checkpoint_files, 1):
        print(f"  {i}. {cp.name}")

    return [str(cp) for cp in checkpoint_files]

def run_single_evaluation(model_path, iteration, eval_dir, device='cuda', max_steps=3600):
    """运行单个模型的评估"""
    print(f"\n{'='*70}")
    print(f"评估检查点: {model_path}")
    print(f"迭代次数: {iteration}")
    print(f"设备: {device}")
    print(f"{'='*70}\n")

    cmd = [
        'python', 'evaluate_model_compliant.py',
        '--model-path', model_path,
        '--sumo-cfg', 'sumo/sumo.sumocfg',
        '--iteration', str(iteration),
        '--eval-dir', eval_dir,
        '--device', device,
        '--max-steps', str(max_steps)
    ]

    print(f"执行命令: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 评估失败")
        print(f"返回码: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='RL模型评估和比赛pkl文件生成',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='检查点路径 (如: checkpoints/checkpoint_iter_0020.pt, 或 "latest", 或 "all")'
    )
    parser.add_argument('--steps', type=int, default=3600, help='仿真步数 (默认: 3600)')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--eval-dir', type=str, default='competition_results', help='评估结果目录')

    args = parser.parse_args()

    # 如果没有指定检查点，查找最新的
    if not args.checkpoint or args.checkpoint == 'latest':
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            sys.exit(1)
    elif args.checkpoint == 'all':
        checkpoint_paths = list_all_checkpoints()
        if not checkpoint_paths:
            sys.exit(1)

        print(f"\n将评估 {len(checkpoint_paths)} 个检查点...")
        success_count = 0

        for i, cp_path in enumerate(checkpoint_paths, 1):
            print(f"\n[{i}/{len(checkpoint_paths)}] 评估 {cp_path}")

            # 从文件名提取迭代次数
            match = re.search(r'checkpoint_iter_(\d+)\.pt', cp_path)
            iteration = int(match.group(1)) if match else 0

            if run_single_evaluation(
                model_path=cp_path,
                iteration=iteration,
                eval_dir=args.eval_dir,
                device=args.device,
                max_steps=args.steps
            ):
                success_count += 1

        print(f"\n{'='*70}")
        print(f"批量评估完成: {success_count}/{len(checkpoint_paths)} 成功")
        print(f"{'='*70}\n")

        sys.exit(0 if success_count == len(checkpoint_paths) else 1)
    else:
        checkpoint_path = args.checkpoint

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件: {checkpoint_path}")
        sys.exit(1)

    # 从文件名提取迭代次数
    match = re.search(r'checkpoint_iter_(\d+)\.pt', checkpoint_path)
    iteration = int(match.group(1)) if match else 0

    # 运行评估
    success = run_single_evaluation(
        model_path=checkpoint_path,
        iteration=iteration,
        eval_dir=args.eval_dir,
        device=args.device,
        max_steps=args.steps
    )

    if success:
        print(f"\n✓ 评估完成! 结果保存在: {args.eval_dir}")
        print(f"\n比赛提交文件位于: {args.eval_dir}/iter_{iteration:04d}/\n")
        sys.exit(0)
    else:
        print(f"\n✗ 评估失败\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
