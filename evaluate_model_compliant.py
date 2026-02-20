"""
符合比赛要求的模型评估脚本

关键特性:
1. 不修改SUMO基础配置（不修改车辆类型maxSpeed等）
2. 只通过TraCI命令进行控制
3. 保存符合比赛要求的pkl文件
"""

import os
import sys

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

# 导入原始的竞赛框架
from sumo.main import SUMOCompetitionFramework

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


def run_evaluation(model_path, sumo_cfg, iteration, eval_dir, device='cuda', max_steps=3600):
    """
    运行符合比赛要求的模型评估

    Args:
        model_path: 模型文件路径
        sumo_cfg: SUMO配置文件路径
        iteration: 当前迭代次数
        eval_dir: 评估结果保存目录
        device: 设备 ('cuda' or 'cpu')
        max_steps: 最大仿真步数
    """
    logger = logging.getLogger('evaluation')

    logger.info("=" * 70)
    logger.info(f"模型评估 - 迭代 {iteration}")
    logger.info("=" * 70)
    logger.info("⚠️  重要提醒: 此评估不修改SUMO配置，符合比赛要求")

    try:
        # 创建框架实例
        framework = SUMOCompetitionFramework(sumo_cfg)

        # 第一部分: 初始化Baseline环境
        logger.info("\n[第一步] 初始化环境...")
        framework.parse_config()
        framework.parse_routes()

        # 启动SUMO（不使用GUI，加快速度）
        sumo_binary = "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", sumo_cfg,
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]

        import traci
        traci.start(sumo_cmd)
        logger.info("✓ SUMO已启动")

        framework.initialize_traffic_lights()

        # 第二步: 加载RL模型（不修改配置）
        logger.info(f"\n[第二步] 加载RL模型...")
        logger.info(f"模型路径: {model_path}")
        framework.load_rl_model(model_path, device=device)

        if not framework.model_loaded:
            logger.warning("模型加载失败，将运行Baseline模式")

        # 第三步: 运行仿真
        logger.info(f"\n[第三步] 开始仿真...")
        logger.info(f"最大步数: {max_steps}")
        logger.info(f"模式: {'RL控制' if framework.model_loaded else 'Baseline'}")

        step = 0
        try:
            while step < max_steps:
                # 仿真一步
                traci.simulationStep()

                # 应用控制算法（如果模型加载成功，会使用RL控制）
                framework.apply_control_algorithm(step)

                # 收集数据
                framework.collect_step_data(step)

                step += 1

                # 进度报告
                if step % 100 == 0:
                    logger.info(f"[步骤 {step}] 活跃: {len(traci.vehicle.getIDList())}, "
                               f"累计出发: {framework.cumulative_departed}, "
                               f"累计到达: {framework.cumulative_arrived}")

                # 检查仿真是否结束
                if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                    logger.info(f"\n仿真自然结束于步骤 {step}")
                    break

        except Exception as e:
            logger.error(f"\n仿真过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            traci.close()

        # 第四步: 保存pkl文件
        logger.info(f"\n[第四步] 保存比赛提交文件...")
        pkl_dir = os.path.join(eval_dir, f"iter_{iteration:04d}")
        result = framework.save_to_pickle(output_dir=pkl_dir)

        logger.info("\n" + "=" * 70)
        logger.info("评估完成!")
        logger.info("=" * 70)
        logger.info(f"✓ Pickle文件: {result['pickle_file']}")
        logger.info(f"✓ 文件大小: {result['file_size_mb']:.2f} MB")
        logger.info(f"✓ 总出发车辆: {framework.cumulative_departed}")
        logger.info(f"✓ 总到达车辆: {framework.cumulative_arrived}")
        logger.info(f"✓ 完成率: {framework.cumulative_arrived / max(framework.cumulative_departed, 1):.4f}")
        logger.info("=" * 70)

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

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ 评估结果: {result_file}")
        logger.info("=" * 70)

        return result_data

    except Exception as e:
        logger.error(f"评估失败: {e}\n{tb.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(description='符合比赛要求的模型评估')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg', help='SUMO配置文件')
    parser.add_argument('--iteration', type=int, required=True, help='当前迭代次数')
    parser.add_argument('--eval-dir', type=str, default='checkpoints/evaluations', help='评估结果目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--max-steps', type=int, default=3600, help='最大仿真步数')

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
        device=args.device,
        max_steps=args.max_steps
    )

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
