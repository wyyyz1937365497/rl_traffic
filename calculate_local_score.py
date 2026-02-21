"""
初赛本地评分计算脚本

根据评测公式计算本地得分：
- 初赛阶段：只计算效率得分 S_efficiency
- 稳定性和干预成本在初赛阶段不计入 (P_int=1, W_stability=0)
"""

import pickle
import json
import numpy as np
from collections import defaultdict
from pathlib import Path


def calculate_ocr_from_pkl(pkl_path):
    """
    从pkl文件计算OCR (OD完成率)

    公式: OCR = (N_arrived + Σ(d_i_traveled / d_i_total)) / N_total
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    stats = data.get('statistics', {})

    # 从statistics获取基础数据
    n_arrived = stats.get('cumulative_arrived', 0)
    n_total = len(stats.get('all_departed_vehicles', []))
    n_enroute = n_total - n_arrived

    # 计算在途车辆完成度
    vehicle_data = data.get('vehicle_data', [])
    enroute_completion = 0.0

    if vehicle_data:
        # 获取最后一帧（step=3600）的车辆数据
        last_step_data = [v for v in vehicle_data if v.get('step') == 3600]

        # 获取已到达车辆集合
        arrived_vehicles = set(stats.get('all_arrived_vehicles', []))

        for veh_info in last_step_data:
            veh_id = veh_info.get('vehicle_id')
            if veh_id not in arrived_vehicles:
                # 在途车辆：使用completion_rate
                completion_rate = veh_info.get('completion_rate', 0.0)
                enroute_completion += completion_rate

    # OCR计算
    if n_total > 0:
        ocr = (n_arrived + enroute_completion) / n_total
    else:
        ocr = 0.0

    return {
        'ocr': ocr,
        'n_arrived': n_arrived,
        'n_total': n_total,
        'n_enroute': n_enroute,
        'enroute_completion': enroute_completion
    }


def calculate_score(ocr_ai, ocr_base=0.90):
    """
    计算初赛效率得分

    公式:
    - ΔOCR = (OCR_AI - OCR_Base) / OCR_Base
    - S_efficiency = 100 × max(0, ΔOCR)

    参数:
        ocr_ai: AI方案的OCR
        ocr_base: 基准方案的OCR (默认0.90，需根据实际基准调整)
    """
    delta_ocr = (ocr_ai - ocr_base) / ocr_base if ocr_base > 0 else 0
    score = 100 * max(0, delta_ocr)

    return {
        'delta_ocr': delta_ocr,
        'efficiency_score': score,
        'total_score': score  # 初赛阶段 total_score = efficiency_score (P_int=1)
    }


def analyze_pkl(pkl_path, ocr_base=0.90):
    """
    分析pkl文件并计算得分
    """
    print(f"\n{'='*60}")
    print(f"分析文件: {pkl_path}")
    print(f"{'='*60}\n")

    # 读取基本信息
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 显示基础统计
    print("[基础统计]")
    print(f"  总出发车辆: {data.get('statistics', {}).get('total_departed', 'N/A')}")
    print(f"  总到达车辆: {data.get('statistics', {}).get('total_arrived', 'N/A')}")

    # 计算OCR
    ocr_result = calculate_ocr_from_pkl(pkl_path)
    print(f"\n[OCR计算]")
    print(f"  N_arrived: {ocr_result['n_arrived']}")
    print(f"  N_enroute: {ocr_result['n_enroute']}")
    print(f"  N_total: {ocr_result['n_total']}")
    print(f"  Enroute Completion: {ocr_result['enroute_completion']:.2f}")
    print(f"  OCR: {ocr_result['ocr']:.4f}")

    # 计算得分
    score_result = calculate_score(ocr_result['ocr'], ocr_base)
    print(f"\n[得分计算] (基准OCR={ocr_base:.4f})")
    print(f"  ΔOCR: {score_result['delta_ocr']:.4f} ({score_result['delta_ocr']*100:.2f}%)")
    print(f"  效率得分: {score_result['efficiency_score']:.2f}")
    print(f"  总得分: {score_result['total_score']:.2f}")

    # 显示对比
    print(f"\n[对比分析]")
    print(f"  基准OCR: {ocr_base:.4f}")
    print(f"  AI OCR: {ocr_result['ocr']:.4f}")
    print(f"  改进: {(ocr_result['ocr'] - ocr_base):.4f} ({((ocr_result['ocr'] / ocr_base - 1) * 100):.2f}%)")

    return ocr_result, score_result


def load_ocr_baseline(json_path):
    """
    从JSON文件加载基准OCR
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('ocr_baseline', 0.90)
    except:
        return 0.90


def reverse_baseline_ocr(ocr_ai, target_score):
    """
    根据AI的OCR和目标分数反推基准OCR

    从公式: score = 100 × (ocr_ai - ocr_base) / ocr_base
    反推: ocr_base = ocr_ai / (1 + score/100)
    """
    if target_score <= 0:
        return ocr_ai
    return ocr_ai / (1 + target_score / 100.0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='初赛本地评分计算')
    parser.add_argument('--pkl', type=str, required=True, help='pkl文件路径')
    parser.add_argument('--base-ocr', type=float, help='基准OCR (如果不指定，将从target-score反推)')
    parser.add_argument('--target-score', type=float, help='已知的目标分数(用于反推基准OCR)')
    parser.add_argument('--json', type=str, help='JSON结果文件路径')

    args = parser.parse_args()

    # 如果提供了目标分数，反推基准OCR
    if args.target_score:
        # 先计算当前OCR
        ocr_result = calculate_ocr_from_pkl(args.pkl)
        base_ocr = reverse_baseline_ocr(ocr_result['ocr'], args.target_score)
        print(f"[反推基准OCR]")
        print(f"  AI OCR: {ocr_result['ocr']:.4f}")
        print(f"  目标分数: {args.target_score:.4f}")
        print(f"  反推基准OCR: {base_ocr:.4f}\n")
    elif args.base_ocr is not None:
        base_ocr = args.base_ocr
    elif args.json:
        base_ocr = load_ocr_baseline(args.json)
        print(f"从JSON加载基准OCR: {base_ocr:.4f}")
    else:
        # 默认基准OCR
        base_ocr = 0.90
        print(f"使用默认基准OCR: {base_ocr:.4f}")

    # 分析pkl文件
    analyze_pkl(args.pkl, base_ocr)

    print(f"\n{'='*60}")
    print(f"说明:")
    print(f"  初赛评分公式: S_efficiency = 100 × max(0, ΔOCR)")
    print(f"  其中: ΔOCR = (OCR_AI - OCR_Base) / OCR_Base")
    print(f"  初赛: P_int=1, W_stability=0 (无惩罚)")
    print(f"\n使用方法:")
    print(f"  1. python calculate_local_score.py --pkl xxx.pkl --base-ocr 0.88")
    print(f"  2. python calculate_local_score.py --pkl xxx.pkl --target-score 6.25")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
