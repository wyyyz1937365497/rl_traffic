"""
Baseline评估脚本 - 符合比赛要求的评估

不加载任何模型，不修改环境配置
只收集原始数据并保存为比赛要求的pkl格式
"""

import os
import sys
import argparse
import logging
import traceback as tb
from datetime import datetime

# 导入原始的竞赛框架
from sumo.main import SUMOCompetitionFramework

def setup_evaluation_logger(eval_dir):
    """配置评估日志"""
    log_file = os.path.join(eval_dir, f"baseline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('baseline_evaluation')


def run_baseline_evaluation(sumo_cfg, iteration, eval_dir, max_steps=3600):
    """
    运行Baseline评估（不加载模型，不修改环境）

    Args:
        sumo_cfg: SUMO配置文件路径
        iteration: 当前迭代次数
        eval_dir: 评估结果保存目录
        max_steps: 最大仿真步数
    """
    logger = logging.getLogger('baseline_evaluation')

    logger.info("=" * 70)
    logger.info(f"Baseline评估 - 迭代 {iteration}")
    logger.info("=" * 70)
    logger.info("注意: 此评估不加载任何模型，不修改环境配置")
    logger.info("      只收集原始的Baseline数据")
    logger.info("=" * 70)

    try:
        # 创建原始框架实例（不传模型路径）
        framework = SUMOCompetitionFramework(sumo_cfg)

        # 运行仿真（不使用GUI，加快速度）
        logger.info("\n开始运行Baseline仿真...")
        result = framework.run(max_steps=max_steps, use_gui=False)

        if result:
            logger.info("\n" + "=" * 70)
            logger.info("评估完成!")
            logger.info("=" * 70)
            logger.info(f"✓ Pickle文件已保存: {result['pickle_file']}")
            logger.info(f"✓ 文件大小: {result['file_size_mb']:.2f} MB")
            logger.info(f"✓ 总出发车辆: {framework.cumulative_departed}")
            logger.info(f"✓ 总到达车辆: {framework.cumulative_arrived}")
            logger.info(f"✓ 完成率: {framework.cumulative_arrived / max(framework.cumulative_departed, 1):.4f}")

            return result
        else:
            logger.error("评估失败")
            return None

    except Exception as e:
        logger.error(f"评估失败: {e}\n{tb.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Baseline评估（符合比赛要求）')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg', help='SUMO配置文件')
    parser.add_argument('--iteration', type=int, required=True, help='当前迭代次数')
    parser.add_argument('--eval-dir', type=str, default='checkpoints/baseline_evaluations', help='评估结果目录')
    parser.add_argument('--max-steps', type=int, default=3600, help='最大仿真步数')

    args = parser.parse_args()

    # 创建评估目录
    os.makedirs(args.eval_dir, exist_ok=True)

    # 配置日志
    logger = setup_evaluation_logger(args.eval_dir)

    # 运行评估
    run_baseline_evaluation(
        sumo_cfg=args.sumo_cfg,
        iteration=args.iteration,
        eval_dir=args.eval_dir,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()
