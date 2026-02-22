"""
统一的模型评估脚本

功能:
1. 评估单个或批量检查点
2. 生成比赛pkl文件
3. 自动计算OCR和得分
4. 支持latest/all快捷方式

使用方法:
# 评估单个检查点
python evaluate.py --checkpoint checkpoints/checkpoint_iter_0020.pt

# 评估最新检查点
python evaluate.py --checkpoint latest

# 评估所有检查点
python evaluate.py --checkpoint all

# 批量评估并自动评分
python evaluate.py --checkpoint all --auto-score
"""

import os
import sys

# 强制使用 traci 而不是 libsumo
os.environ["USE_LIBSUMO"] = "0"

# 设置控制台编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

import argparse
import logging
import traceback as tb
from datetime import datetime
import json
import pickle
import re
from pathlib import Path

# 导入原始的竞赛框架
from sumo.main import SUMOCompetitionFramework


def setup_logger(eval_dir):
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


def find_latest_checkpoint():
    """查找最新的检查点文件"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print(f"错误: 找不到 checkpoints 目录")
        return None

    checkpoint_files = list(checkpoints_dir.glob("checkpoint_iter_*.pt"))
    if not checkpoint_files:
        print(f"错误: 在 {checkpoints_dir} 中没有找到检查点文件")
        return None

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


def calculate_ocr_and_score(pkl_path, base_ocr=0.8812):
    """从pkl文件计算OCR和得分"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        stats = data['statistics']
        n_arrived = stats['total_arrived']
        n_total = stats['total_departed']

        # 在途车辆完成度
        enroute_completion = 0.0
        for vehicle_data in data.get('vehicle_data', []):
            if 'completion_rate' in vehicle_data:
                enroute_completion += vehicle_data['completion_rate']

        ocr = (n_arrived + enroute_completion) / n_total if n_total > 0 else 0.0
        delta_ocr = (ocr - base_ocr) / base_ocr
        score = 100 * max(0, delta_ocr)

        return {
            'ocr': ocr,
            'n_arrived': n_arrived,
            'n_total': n_total,
            'delta_ocr': delta_ocr,
            'score': score
        }
    except Exception as e:
        return {'error': str(e)}


def run_single_evaluation(model_path, iteration, eval_dir, device='cuda', max_steps=3600, auto_score=False):
    """运行单个模型的评估"""
    logger = logging.getLogger('evaluation')

    logger.info("=" * 70)
    logger.info(f"评估检查点: {model_path}")
    logger.info(f"迭代次数: {iteration}")
    logger.info(f"设备: {device}")
    logger.info("=" * 70)

    try:
        # 创建框架实例
        framework = SUMOCompetitionFramework("sumo/sumo.sumocfg")

        # 初始化
        logger.info("\n[第一步] 初始化环境...")
        framework.parse_config()
        framework.parse_routes()

        sumo_binary = "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", "sumo/sumo.sumocfg",
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]

        import traci
        traci.start(sumo_cmd)
        logger.info("✓ SUMO已启动")

        framework.initialize_traffic_lights()

        # 加载模型
        logger.info(f"\n[第二步] 加载RL模型...")
        framework.load_rl_model(model_path, device=device)

        if not framework.model_loaded:
            logger.warning("模型加载失败，将运行Baseline模式")

        # 运行仿真
        logger.info(f"\n[第三步] 开始仿真...")
        logger.info(f"最大步数: {max_steps}")

        step = 0
        while step < max_steps:
            traci.simulationStep()
            framework.apply_control_algorithm(step)
            framework.collect_step_data(step)
            step += 1

            if step % 100 == 0:
                logger.info(f"[步骤 {step}] 活跃: {len(traci.vehicle.getIDList())}, "
                           f"累计出发: {framework.cumulative_departed}, "
                           f"累计到达: {framework.cumulative_arrived}")

            if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                logger.info(f"\n仿真自然结束于步骤 {step}")
                break

        traci.close()

        # 保存pkl文件
        logger.info(f"\n[第四步] 保存比赛提交文件...")
        pkl_dir = os.path.join(eval_dir, f"iter_{iteration:04d}")
        result = framework.save_to_pickle(output_dir=pkl_dir)

        logger.info("\n" + "=" * 70)
        logger.info("评估完成!")
        logger.info("=" * 70)
        logger.info(f"✓ Pickle文件: {result['pickle_file']}")
        logger.info(f"✓ 总出发车辆: {framework.cumulative_departed}")
        logger.info(f"✓ 总到达车辆: {framework.cumulative_arrived}")

        # 自动评分
        score_result = None
        if auto_score:
            logger.info(f"\n[第五步] 计算得分...")
            score_result = calculate_ocr_and_score(result['pickle_file'])

            if 'error' not in score_result:
                logger.info(f"✓ OCR: {score_result['ocr']:.4f}")
                logger.info(f"✓ 得分: {score_result['score']:.2f}")
                logger.info(f"  (ΔOCR: {score_result['delta_ocr']:.4f}, 基准: 0.8812)")
            else:
                logger.warning(f"✗ 评分失败: {score_result['error']}")

        # 保存评估结果JSON
        result_file = os.path.join(eval_dir, f"eval_iter_{iteration:04d}.json")
        result_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'statistics': {
                'total_departed': framework.cumulative_departed,
                'total_arrived': framework.cumulative_arrived,
                'completion_rate': framework.cumulative_arrived / max(framework.cumulative_departed, 1)
            },
            'pickle_file': result['pickle_file']
        }

        if score_result and 'error' not in score_result:
            result_data['score'] = {
                'ocr': score_result['ocr'],
                'delta_ocr': score_result['delta_ocr'],
                'estimated_score': score_result['score']
            }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ 评估结果: {result_file}")
        logger.info("=" * 70)

        return result_data

    except Exception as e:
        logger.error(f"评估失败: {e}\n{tb.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='统一的模型评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 检查点参数
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='检查点路径 (如: checkpoints/checkpoint_iter_0020.pt, 或 "latest", 或 "all")'
    )

    # 评估参数
    parser.add_argument('--steps', type=int, default=3600, help='仿真步数 (默认: 3600)')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--eval-dir', type=str, default='competition_results', help='评估结果目录')

    # 评分参数
    parser.add_argument('--auto-score', action='store_true', help='自动计算OCR和得分')
    parser.add_argument('--base-ocr', type=float, default=0.8812, help='基准OCR (默认: 0.8812)')

    args = parser.parse_args()

    # 创建时间戳子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(args.eval_dir, f'eval_{timestamp}')

    # 如果没有指定检查点，查找最新的
    if not args.checkpoint or args.checkpoint == 'latest':
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            sys.exit(1)
    elif args.checkpoint == 'all':
        checkpoint_paths = list_all_checkpoints()
        if not checkpoint_paths:
            sys.exit(1)

        # 创建评估目录
        os.makedirs(eval_dir, exist_ok=True)
        logger = setup_logger(eval_dir)

        print(f"\n评估目录: {eval_dir}")
        print(f"将评估 {len(checkpoint_paths)} 个检查点...")
        success_count = 0
        all_results = []

        for i, cp_path in enumerate(checkpoint_paths, 1):
            print(f"\n[{i}/{len(checkpoint_paths)}] 评估 {cp_path}")

            match = re.search(r'checkpoint_iter_(\d+)\.pt', cp_path)
            iteration = int(match.group(1)) if match else 0

            result = run_single_evaluation(
                model_path=cp_path,
                iteration=iteration,
                eval_dir=eval_dir,
                device=args.device,
                max_steps=args.steps,
                auto_score=args.auto_score
            )

            if result:
                success_count += 1
                all_results.append(result)

        # 批量评估总结
        print(f"\n{'='*70}")
        print(f"批量评估完成: {success_count}/{len(checkpoint_paths)} 成功")
        print(f"{'='*70}")

        if args.auto_score and all_results:
            # 打印评分汇总
            scored_results = [r for r in all_results if 'score' in r]
            if scored_results:
                scored_results.sort(key=lambda x: x['score']['estimated_score'], reverse=True)

                print(f"\nTop 10 模型 (按得分排序):")
                for i, result in enumerate(scored_results[:10], 1):
                    score_info = result['score']
                    print(f"  {i}. iter_{result['iteration']:04d}")
                    print(f"     OCR: {score_info['ocr']:.4f}, 得分: {score_info['estimated_score']:.2f}")

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

    # 创建评估目录
    os.makedirs(eval_dir, exist_ok=True)
    logger = setup_logger(eval_dir)

    print(f"\n评估目录: {eval_dir}")

    # 运行单个评估
    result = run_single_evaluation(
        model_path=checkpoint_path,
        iteration=iteration,
        eval_dir=eval_dir,
        device=args.device,
        max_steps=args.steps,
        auto_score=args.auto_score
    )

    if result:
        print(f"\n✓ 评估完成! 结果保存在: {eval_dir}")
        if 'score' in result:
            print(f"✓ OCR: {result['score']['ocr']:.4f}, 得分: {result['score']['estimated_score']:.2f}")
        print(f"\n比赛提交文件位于: {eval_dir}/iter_{iteration:04d}/\n")
        sys.exit(0)
    else:
        print(f"\n✗ 评估失败\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
