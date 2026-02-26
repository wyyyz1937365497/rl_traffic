"""
混合方案PKL生成脚本

- 使用车辆级checkpoint生成提交pkl
- 可选调用本地分数计算器进行对齐评估
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generate_submit_bc import generate_submission_bc
from local_score_calculator import LocalScoreCalculator


def main():
    parser = argparse.ArgumentParser(description='生成混合方案提交PKL并可选本地评分')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint路径（BC或PPO微调）')
    parser.add_argument('--output', type=str, default='submit_hybrid.pkl', help='输出pkl路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg', help='SUMO配置路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--steps', type=int, default=3600, help='仿真步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    parser.add_argument('--score', action='store_true', help='生成后计算本地分数')
    parser.add_argument('--baseline-pkl', type=str, default='', help='baseline pkl路径（本地评分用）')

    args = parser.parse_args()

    output_path = generate_submission_bc(
        output_path=args.output,
        sumo_cfg=args.sumo_cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_steps=args.steps,
        seed=args.seed,
    )

    if args.score:
        baseline_pkl = args.baseline_pkl if args.baseline_pkl else None
        calculator = LocalScoreCalculator(baseline_pkl)
        score = calculator.calculate_score(output_path)

        print('\n' + '=' * 80)
        print('本地评分结果')
        print('=' * 80)
        print(f"文件: {os.path.abspath(output_path)}")
        print(f"总分: {score['total_score']:.4f}")
        print(f"效率得分: {score['efficiency_score']:.4f}")
        print(f"稳定性得分: {score['stability_score']:.4f}")
        print(f"干预惩罚: {score['intervention_penalty']:.4f}")


if __name__ == '__main__':
    main()
