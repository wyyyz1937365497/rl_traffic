"""
PKL文件评分工具 - 统一版本

支持单个文件评分和批量评分
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path


def calculate_ocr_from_pkl(pkl_path):
    """从pkl文件计算OCR"""
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

    return {
        'ocr': ocr,
        'n_arrived': n_arrived,
        'n_total': n_total,
        'enroute_completion': enroute_completion
    }


def calculate_score(ocr_ai, ocr_base=0.90):
    """计算初赛效率得分"""
    delta_ocr = (ocr_ai - ocr_base) / ocr_base
    score = 100 * max(0, delta_ocr)
    return {
        'delta_ocr': delta_ocr,
        'efficiency_score': score,
        'total_score': score
    }


def find_all_pkl_files(directory):
    """查找目录下所有pkl文件"""
    path = Path(directory)
    if not path.exists():
        print(f"错误: 目录不存在: {directory}")
        return []

    pkl_files = list(path.rglob("*.pkl"))
    return sorted(pkl_files)


def score_single_file(pkl_path, base_ocr=0.8812):
    """对单个pkl文件评分"""
    try:
        ocr_result = calculate_ocr_from_pkl(pkl_path)
        score_result = calculate_score(ocr_result['ocr'], base_ocr)

        return {
            'pkl_file': str(pkl_path),
            'ocr': ocr_result['ocr'],
            'n_arrived': ocr_result['n_arrived'],
            'n_total': ocr_result['n_total'],
            'delta_ocr': score_result['delta_ocr'],
            'efficiency_score': score_result['efficiency_score'],
            'total_score': score_result['total_score']
        }
    except Exception as e:
        return {
            'pkl_file': str(pkl_path),
            'error': str(e)
        }


def print_single_result(result):
    """打印单个文件的评分结果"""
    print(f"\n{'='*70}")
    print(f"分析文件: {result['pkl_file']}")
    print(f"{'='*70}")

    if 'error' in result:
        print(f"错误: {result['error']}")
        return

    print(f"\n[基础统计]")
    print(f"  总出发车辆: {result['n_total']}")
    print(f"  总到达车辆: {result['n_arrived']}")

    print(f"\n[OCR计算]")
    print(f"  OCR: {result['ocr']:.4f}")

    print(f"\n[得分计算] (基准OCR={0.8812})")
    print(f"  ΔOCR: {result['delta_ocr']:.4f} ({result['delta_ocr']*100:.2f}%)")
    print(f"  效率得分: {result['efficiency_score']:.2f}")
    print(f"  总得分: {result['total_score']:.2f}")

    print(f"\n{'='*70}\n")


def batch_score(directory, base_ocr=0.8812, output_file=None, top_n=10):
    """批量评分"""
    print(f"\n{'='*70}")
    print(f"批量评分pkl文件")
    print(f"{'='*70}")
    print(f"目录: {directory}")
    print(f"基准OCR: {base_ocr}")
    print(f"{'='*70}\n")

    # 查找所有pkl文件
    pkl_files = find_all_pkl_files(directory)

    if not pkl_files:
        print(f"错误: 在 {directory} 中没有找到pkl文件")
        return

    print(f"找到 {len(pkl_files)} 个pkl文件\n")

    # 批量评分
    results = []
    best_score = -1
    best_pkl = None

    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"[{i}/{len(pkl_files)}] 评分: {pkl_file.name}")

        result = score_single_file(pkl_file, base_ocr)
        results.append(result)

        if 'error' in result:
            print(f"  ✗ 错误: {result['error']}")
        else:
            print(f"  ✓ OCR: {result['ocr']:.4f}, 得分: {result['total_score']:.2f}")

            if result['total_score'] > best_score:
                best_score = result['total_score']
                best_pkl = result

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"评分汇总")
    print(f"{'='*70}")

    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        # 按得分排序
        successful_results.sort(key=lambda x: x['total_score'], reverse=True)

        print(f"\nTop {min(top_n, len(successful_results))} 模型:")
        for i, result in enumerate(successful_results[:top_n], 1):
            print(f"  {i}. {Path(result['pkl_file']).name}")
            print(f"     OCR: {result['ocr']:.4f}, 得分: {result['total_score']:.2f}")

        if best_pkl:
            print(f"\n最佳模型:")
            print(f"  文件: {best_pkl['pkl_file']}")
            print(f"  OCR: {best_pkl['ocr']:.4f}")
            print(f"  得分: {best_pkl['total_score']:.2f}")
            print(f"  到达车辆: {best_pkl['n_arrived']}/{best_pkl['n_total']}")

    # 保存结果到JSON
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='PKL文件评分工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 评分单个文件
  python score_pkl.py --pkl result.pkl --base-ocr 0.8812

  # 批量评分
  python score_pkl.py --dir competition_results --base-ocr 0.8812

  # 批量评分并保存结果
  python score_pkl.py --dir competition_results --output scores.json

  # 反推基准OCR
  python score_pkl.py --pkl result.pkl --target-score 6.2513
        """
    )

    # 单文件评分
    parser.add_argument('--pkl', type=str, help='单个pkl文件路径')

    # 批量评分
    parser.add_argument('--dir', type=str, help='pkl文件所在目录')

    # 评分参数
    parser.add_argument('--base-ocr', type=float, default=0.8812, help='基准OCR (默认: 0.8812)')
    parser.add_argument('--target-score', type=float, help='目标分数（用于反推基准OCR）')

    # 输出参数
    parser.add_argument('--output', type=str, help='输出JSON文件路径 (可选)')
    parser.add_argument('--top', type=int, default=10, help='显示前N个最佳模型 (默认: 10)')

    args = parser.parse_args()

    # 检查参数
    if not args.pkl and not args.dir:
        parser.print_help()
        print("\n错误: 必须指定 --pkl 或 --dir 参数")
        sys.exit(1)

    # 单文件评分
    if args.pkl:
        if args.target_score:
            # 反推基准OCR
            result = score_single_file(args.pkl, base_ocr=0.8812)
            if 'error' not in result:
                ocr_ai = result['ocr']
                # S = 100 * (OCR_ai - OCR_base) / OCR_base
                # OCR_base = OCR_ai / (1 + S/100)
                base_ocr = ocr_ai / (1 + args.target_score / 100)
                print(f"\n[反推基准OCR]")
                print(f"  AI OCR: {ocr_ai:.4f}")
                print(f"  目标分数: {args.target_score}")
                print(f"  反推基准OCR: {base_ocr:.4f}\n")
        else:
            result = score_single_file(args.pkl, args.base_ocr)
            print_single_result(result)

    # 批量评分
    elif args.dir:
        batch_score(args.dir, args.base_ocr, args.output, args.top)


if __name__ == '__main__':
    main()
