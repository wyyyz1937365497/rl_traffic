"""
初赛评分计算器（严格按初赛标准）

依据评测公式（初赛）：
- W_stability = 0
- P_int = 1
- S_total = S_efficiency
- S_efficiency = 100 * max(0, (OCR_AI - OCR_Base) / OCR_Base)

OCR 计算采用文档定义：
OCR = (N_arrived + sum_{i in enroute}(d_traveled_i / d_total_i)) / N_total
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Any


def _load_pkl(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _get_total_vehicles(data: Dict[str, Any]) -> int:
    params = data.get('parameters', {})
    stats = data.get('statistics', {})

    total_demand = int(params.get('total_demand', 0) or 0)
    departed = int(stats.get('cumulative_departed', 0) or 0)

    route_data = data.get('route_data', {})
    route_count = len(route_data) if isinstance(route_data, dict) else 0

    # 优先使用总需求，其次使用观测到的车辆规模
    n_total = max(total_demand, departed, route_count)
    return max(n_total, 1)


def _get_arrived_set(data: Dict[str, Any]) -> set:
    stats = data.get('statistics', {})
    arrived = stats.get('all_arrived_vehicles', [])
    if isinstance(arrived, list):
        return set(arrived)
    return set()


def _build_completion_map(data: Dict[str, Any]) -> Dict[str, float]:
    vehicle_data = data.get('vehicle_data', [])
    completion_map: Dict[str, float] = {}

    for record in vehicle_data:
        veh_id = record.get('vehicle_id')
        if not veh_id:
            continue

        if 'completion_rate' in record and record['completion_rate'] is not None:
            completion = float(record['completion_rate'])
        else:
            traveled = float(record.get('traveled_distance', 0.0) or 0.0)
            route_len = float(record.get('route_length', 0.0) or 0.0)
            completion = traveled / route_len if route_len > 0 else 0.0

        completion = max(0.0, min(1.0, completion))

        # 保留该车最高完成度（通常越靠后越大）
        prev = completion_map.get(veh_id, 0.0)
        if completion > prev:
            completion_map[veh_id] = completion

    return completion_map


def calculate_ocr(data: Dict[str, Any]) -> Dict[str, float]:
    n_total = _get_total_vehicles(data)
    arrived_set = _get_arrived_set(data)
    completion_map = _build_completion_map(data)

    n_arrived = len(arrived_set)

    enroute_completion_sum = 0.0
    for veh_id, completion in completion_map.items():
        if veh_id in arrived_set:
            continue
        enroute_completion_sum += completion

    ocr = (n_arrived + enroute_completion_sum) / n_total

    return {
        'ocr': float(ocr),
        'n_total': int(n_total),
        'n_arrived': int(n_arrived),
        'enroute_completion_sum': float(enroute_completion_sum),
        'n_observed_vehicles': int(len(completion_map)),
    }


def calculate_prelim_score(ai_data: Dict[str, Any], baseline_data: Dict[str, Any]) -> Dict[str, Any]:
    ai_ocr_info = calculate_ocr(ai_data)
    base_ocr_info = calculate_ocr(baseline_data)

    ocr_ai = ai_ocr_info['ocr']
    ocr_base = base_ocr_info['ocr']

    if ocr_base <= 0:
        delta_ocr = 0.0
    else:
        delta_ocr = (ocr_ai - ocr_base) / ocr_base

    s_efficiency = 100.0 * max(0.0, delta_ocr)

    # 初赛口径：W_stability=0, P_int=1 => S_total=S_efficiency
    s_total = s_efficiency

    return {
        'S_total': float(s_total),
        'S_efficiency': float(s_efficiency),
        'Delta_OCR': float(delta_ocr),
        'OCR_AI': float(ocr_ai),
        'OCR_Base': float(ocr_base),
        'prelim_assumption': {
            'W_stability': 0,
            'P_int': 1,
            'final_formula': 'S_total = S_efficiency',
        },
        'detail': {
            'ai': ai_ocr_info,
            'baseline': base_ocr_info,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='初赛评分计算器（按OCR增益计算）')
    parser.add_argument('ai_pkls', nargs='+', help='待评估AI pkl文件（可多个）')
    parser.add_argument('--baseline', '-b', required=True, help='基准方案pkl文件路径')
    parser.add_argument('--output', '-o', default='', help='保存JSON结果到文件')
    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        raise FileNotFoundError(f'baseline文件不存在: {args.baseline}')

    baseline_data = _load_pkl(args.baseline)

    results: List[Dict[str, Any]] = []
    for ai_path in args.ai_pkls:
        if not os.path.exists(ai_path):
            print(f'⚠ 文件不存在，跳过: {ai_path}')
            continue

        ai_data = _load_pkl(ai_path)
        score = calculate_prelim_score(ai_data, baseline_data)
        score['ai_pkl'] = ai_path
        results.append(score)

        print('\n' + '=' * 90)
        print(f'文件: {ai_path}')
        print(f"OCR_AI={score['OCR_AI']:.6f}, OCR_Base={score['OCR_Base']:.6f}, Delta_OCR={score['Delta_OCR']:+.6f}")
        print(f"S_efficiency={score['S_efficiency']:.4f}")
        print(f"S_total={score['S_total']:.4f}  (初赛口径)")

    if len(results) > 1:
        results.sort(key=lambda x: x['S_total'], reverse=True)
        print('\n' + '=' * 90)
        print('对比排名（按初赛S_total降序）')
        print('=' * 90)
        for idx, item in enumerate(results, 1):
            print(f"{idx:02d}. {os.path.basename(item['ai_pkl'])} -> S_total={item['S_total']:.4f}, OCR={item['OCR_AI']:.6f}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存: {args.output}")


if __name__ == '__main__':
    main()
