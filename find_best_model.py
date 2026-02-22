"""
查找最佳模型工具

用于分析所有评估结果，找出完成率最高的模型
"""

import json
import glob
from pathlib import Path
import sys

def find_best_model(eval_dir='competition_results'):
    """
    查找最佳模型

    Args:
        eval_dir: 评估结果目录

    Returns:
        best_model: 最佳模型信息字典
    """
    # 查找所有评估结果JSON
    json_pattern = Path(eval_dir) / "eval_iter_*.json"
    json_files = list(glob.glob(str(json_pattern)))

    if not json_files:
        print(f"错误: 在 {eval_dir} 中没有找到评估结果文件")
        print(f"请先运行: python run_evaluation.py --checkpoint all")
        return None

    # 读取所有结果
    results = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'json_file': json_file,
                    'iteration': data.get('iteration', 0),
                    'completion_rate': data['statistics'].get('completion_rate', 0.0),
                    'total_departed': data['statistics'].get('total_departed', 0),
                    'total_arrived': data['statistics'].get('total_arrived', 0),
                    'pickle_file': data.get('pickle_file', ''),
                    'timestamp': data.get('timestamp', '')
                })
        except Exception as e:
            print(f"警告: 无法读取 {json_file}: {e}")

    if not results:
        print(f"错误: 没有有效的评估结果")
        return None

    # 按完成率排序
    results.sort(key=lambda x: x['completion_rate'], reverse=True)

    return results

def print_results_table(results):
    """打印结果表格"""
    print("\n" + "=" * 100)
    print(f"{'排名':<6} {'迭代':<8} {'完成率':<12} {'出发车辆':<12} {'到达车辆':<12} {'Pickle文件':<40}")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        # 提取文件名
        pickle_name = Path(r['pickle_file']).name if r['pickle_file'] else 'N/A'

        # 标记最佳模型
        marker = "🏆 " if i == 1 else "   "

        print(f"{marker}{i:<3} {r['iteration']:<8} {r['completion_rate']:<12.4f} "
              f"{r['total_departed']:<12} {r['total_arrived']:<12} {pickle_name:<40}")

    print("=" * 100)

def print_summary(results):
    """打印汇总信息"""
    if not results:
        return

    best = results[0]
    worst = results[-1]
    avg_rate = sum(r['completion_rate'] for r in results) / len(results)

    print("\n" + "=" * 100)
    print("📊 评估汇总")
    print("=" * 100)
    print(f"总评估次数: {len(results)}")
    print(f"\n🏆 最佳模型:")
    print(f"   迭代: {best['iteration']}")
    print(f"   完成率: {best['completion_rate']:.4f} ({best['completion_rate']*100:.2f}%)")
    print(f"   出发车辆: {best['total_departed']}")
    print(f"   到达车辆: {best['total_arrived']}")
    print(f"   Pickle文件: {best['pickle_file']}")

    print(f"\n⚠️  最差模型:")
    print(f"   迭代: {worst['iteration']}")
    print(f"   完成率: {worst['completion_rate']:.4f} ({worst['completion_rate']*100:.2f}%)")

    print(f"\n📈 平均完成率: {avg_rate:.4f} ({avg_rate*100:.2f}%)")

    # 改进空间
    improvement = (best['completion_rate'] - worst['completion_rate']) * 100
    print(f"\n📊 改进幅度: {improvement:.2f}%")

    print("=" * 100)

def check_pickle_exists(results):
    """检查pkl文件是否存在"""
    print("\n" + "=" * 100)
    print("📁 文件检查")
    print("=" * 100)

    for r in results[:5]:  # 只检查前5个
        pickle_path = Path(r['pickle_file'])
        exists = "✓" if pickle_path.exists() else "✗"
        size_mb = pickle_path.stat().st_size / (1024*1024) if pickle_path.exists() else 0

        print(f"{exists} 迭代 {r['iteration']:<4} {pickle_path.name:<50} "
              f"{'{:.2f} MB'.format(size_mb) if size_mb > 0 else '不存在'}")

    print("=" * 100)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='查找最佳训练模型')
    parser.add_argument('--eval-dir', type=str, default='competition_results',
                       help='评估结果目录 (默认: competition_results)')
    parser.add_argument('--top', type=int, default=10,
                       help='显示前N个模型 (默认: 10)')

    args = parser.parse_args()

    results = find_best_model(args.eval_dir)

    if not results:
        sys.exit(1)

    # 打印所有结果
    print_results_table(results[:args.top])

    # 打印汇总
    print_summary(results)

    # 检查文件
    check_pickle_exists(results)

    # 打印推荐命令
    best = results[0]
    print("\n" + "=" * 100)
    print("💡 推荐提交命令")
    print("=" * 100)
    print(f"\n最佳模型是迭代 {best['iteration']}，完成率 {best['completion_rate']*100:.2f}%")
    print(f"\n提交文件位于: {best['pickle_file']}")
    print(f"\n如需重新生成，运行:")
    print(f"  python run_evaluation.py --checkpoint checkpoints/checkpoint_iter_{best['iteration']:04d}.pt")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()
