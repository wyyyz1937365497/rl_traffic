"""
异步评估脚本 - 比赛级别的仿真评估

用于在训练过程中对模型进行评估，输出OCR相关指标
异步执行，不阻塞主训练进程
"""

import os
import sys
import torch
import argparse
import logging
import traceback as tb
from datetime import datetime
import json

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .sumo.main import SUMOCompetitionFramework


def setup_evaluation_logger(eval_dir):
    """配置评估日志"""
    log_file = os.path.join(eval_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('evaluation')


def run_evaluation(model_path, sumo_cfg, iteration, eval_dir, device='cuda'):
    """
    运行比赛级别评估

    Args:
        model_path: 模型路径
        sumo_cfg: SUMO配置文件路径
        iteration: 当前迭代次数
        eval_dir: 评估结果保存目录
        device: 设备 ('cuda' or 'cpu')
    """
    logger = logging.getLogger('evaluation')

    logger.info("=" * 70)
    logger.info(f"开始评估 - 迭代 {iteration}")
    logger.info("=" * 70)

    try:
        # 创建框架实例
        framework = SUMOCompetitionFramework(
            sumo_cfg_path=sumo_cfg,
            model_path=model_path
        )

        # 初始化
        framework.parse_config()
        framework.parse_routes()
        framework.initialize_environment()
        framework.load_rl_model()

        # 运行仿真
        logger.info("\n[第二部分] 开始仿真...")
        framework.run_simulation()

        # 计算OCR指标
        logger.info("\n[第三部分] 计算评估指标...")
        ocr_metrics = framework.calculate_ocr_metrics()

        # 保存结果
        result_file = os.path.join(eval_dir, f"eval_iter_{iteration:04d}.json")

        result = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': ocr_metrics,
            'statistics': {
                'total_departed': framework.cumulative_departed,
                'total_arrived': framework.cumulative_arrived,
                'completion_rate': framework.cumulative_arrived / max(framework.cumulative_departed, 1)
            }
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 输出关键指标
        logger.info("\n" + "=" * 70)
        logger.info(f"评估完成 - 迭代 {iteration}")
        logger.info("=" * 70)
        logger.info(f"📊 OCR指标:")
        logger.info(f"  - 全局OCR: {ocr_metrics.get('global_ocr', 0):.4f}")
        logger.info(f"  - 主路OCR: {ocr_metrics.get('main_road_ocr', 0):.4f}")
        logger.info(f"  - 匝道OCR: {ocr_metrics.get('ramp_road_ocr', 0):.4f}")
        logger.info(f"  - 转出OCR: {ocr_metrics.get('diverge_road_ocr', 0):.4f}")
        logger.info(f"\n📈 统计信息:")
        logger.info(f"  - 总出发车辆: {framework.cumulative_departed}")
        logger.info(f"  - 总到达车辆: {framework.cumulative_arrived}")
        logger.info(f"  - 完成率: {result['statistics']['completion_rate']:.2%}")
        logger.info(f"\n💾 结果已保存: {result_file}")
        logger.info("=" * 70)

        # 关闭SUMO
        framework.close()

        return result

    except Exception as e:
        logger.error(f"评估失败: {e}\n{tb.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(description='异步模型评估')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg', help='SUMO配置文件')
    parser.add_argument('--iteration', type=int, required=True, help='当前迭代次数')
    parser.add_argument('--eval-dir', type=str, default='evaluations', help='评估结果目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 创建评估目录
    os.makedirs(args.eval_dir, exist_ok=True)

    # 配置日志
    logger = setup_evaluation_logger(args.eval_dir)

    # 运行评估
    result = run_evaluation(
        model_path=args.model_path,
        sumo_cfg=args.sumo_cfg,
        iteration=args.iteration,
        eval_dir=args.eval_dir,
        device=args.device
    )

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
