"""
提交数据分析脚本（不计算分数）

目标：分析 pkl 中的交通运行数据，定位策略差异与短板。
支持：
1) 单文件分析
2) 双文件对比分析（如 baseline vs RL）
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Any

import numpy as np


def load_pkl(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _safe_std(arr):
    return float(np.std(arr)) if len(arr) > 0 else 0.0


def analyze_one(data: Dict[str, Any]) -> Dict[str, Any]:
    step_data = data.get('step_data', [])
    vehicle_data = data.get('vehicle_data', [])
    stats = data.get('statistics', {})
    route_data = data.get('route_data', {})

    total_steps = len(step_data)
    departed = int(stats.get('cumulative_departed', 0) or 0)
    arrived = int(stats.get('cumulative_arrived', 0) or 0)

    # OCR（按文档公式近似：arrived + enroute completion）
    completion_map = {}
    for rec in vehicle_data:
        vid = rec.get('vehicle_id')
        if not vid:
            continue
        completion = rec.get('completion_rate')
        if completion is None:
            traveled = float(rec.get('traveled_distance', 0.0) or 0.0)
            route_len = float(rec.get('route_length', 0.0) or 0.0)
            completion = traveled / route_len if route_len > 0 else 0.0
        completion = max(0.0, min(1.0, float(completion)))
        if completion > completion_map.get(vid, 0.0):
            completion_map[vid] = completion

    arrived_ids = set(stats.get('all_arrived_vehicles', [])) if isinstance(stats.get('all_arrived_vehicles', []), list) else set()
    total_vehicles = max(int(data.get('parameters', {}).get('total_demand', 0) or 0), departed, len(completion_map), 1)
    enroute_completion_sum = sum(v for k, v in completion_map.items() if k not in arrived_ids)
    ocr_doc = (len(arrived_ids) + enroute_completion_sum) / total_vehicles

    # 速度与加速度
    speeds = [float(rec.get('speed', 0.0) or 0.0) for rec in vehicle_data if rec.get('speed') is not None]

    vehicle_series = defaultdict(list)
    for rec in vehicle_data:
        vid = rec.get('vehicle_id')
        if not vid:
            continue
        step = int(rec.get('step', 0) or 0)
        speed = float(rec.get('speed', 0.0) or 0.0)
        vehicle_series[vid].append((step, speed))

    accels = []
    for arr in vehicle_series.values():
        arr.sort(key=lambda x: x[0])
        for i in range(1, len(arr)):
            dt = max(arr[i][0] - arr[i - 1][0], 1)
            accels.append((arr[i][1] - arr[i - 1][1]) / dt)

    # 活跃车辆时序
    active_series = [int(s.get('active_vehicles', 0) or 0) for s in step_data]
    arrived_series = [int(s.get('arrived_vehicles', 0) or 0) for s in step_data]

    # 拥堵代理：低速比例
    low_speed_ratio = float(np.mean(np.array(speeds) < 3.0)) if speeds else 0.0
    jam_speed_ratio = float(np.mean(np.array(speeds) < 5.0)) if speeds else 0.0

    # 路段统计
    edge_counter = defaultdict(int)
    for rec in vehicle_data:
        edge = rec.get('edge_id')
        if edge:
            edge_counter[edge] += 1

    top_edges = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    # OD统计
    od_counter = defaultdict(int)
    for rec in vehicle_data:
        o = rec.get('origin', 'unknown')
        d = rec.get('destination', 'unknown')
        od_counter[(o, d)] += 1
    top_ods = sorted(od_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        'summary': {
            'total_steps': total_steps,
            'departed': departed,
            'arrived': arrived,
            'arrive_ratio_simple': float(arrived / departed) if departed > 0 else 0.0,
            'ocr_doc_formula': float(ocr_doc),
            'total_vehicles_for_ocr': int(total_vehicles),
            'observed_unique_vehicles': int(len(vehicle_series)),
        },
        'flow': {
            'active_mean': _safe_mean(active_series),
            'active_max': int(max(active_series)) if active_series else 0,
            'active_p95': float(np.percentile(active_series, 95)) if active_series else 0.0,
            'final_arrived': int(arrived_series[-1]) if arrived_series else 0,
        },
        'speed': {
            'mean': _safe_mean(speeds),
            'std': _safe_std(speeds),
            'p10': float(np.percentile(speeds, 10)) if speeds else 0.0,
            'p50': float(np.percentile(speeds, 50)) if speeds else 0.0,
            'p90': float(np.percentile(speeds, 90)) if speeds else 0.0,
            'low_speed_ratio_lt3': low_speed_ratio,
            'jam_speed_ratio_lt5': jam_speed_ratio,
        },
        'accel': {
            'mean_abs': _safe_mean(np.abs(accels)) if accels else 0.0,
            'std': _safe_std(accels),
            'p90_abs': float(np.percentile(np.abs(accels), 90)) if accels else 0.0,
        },
        'top_edges': top_edges,
        'top_od_pairs': [
            {'origin': k[0], 'destination': k[1], 'count': v}
            for k, v in top_ods
        ],
    }


def diff_report(base: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
    def d(a, b):
        return float(a - b)

    return {
        'ocr_doc_delta': d(ai['summary']['ocr_doc_formula'], base['summary']['ocr_doc_formula']),
        'arrive_ratio_delta': d(ai['summary']['arrive_ratio_simple'], base['summary']['arrive_ratio_simple']),
        'active_mean_delta': d(ai['flow']['active_mean'], base['flow']['active_mean']),
        'speed_mean_delta': d(ai['speed']['mean'], base['speed']['mean']),
        'speed_std_delta': d(ai['speed']['std'], base['speed']['std']),
        'mean_abs_accel_delta': d(ai['accel']['mean_abs'], base['accel']['mean_abs']),
        'low_speed_ratio_lt3_delta': d(ai['speed']['low_speed_ratio_lt3'], base['speed']['low_speed_ratio_lt3']),
        'jam_speed_ratio_lt5_delta': d(ai['speed']['jam_speed_ratio_lt5'], base['speed']['jam_speed_ratio_lt5']),
    }


def main():
    parser = argparse.ArgumentParser(description='提交数据分析（非打分）')
    parser.add_argument('ai_pkl', help='待分析AI pkl')
    parser.add_argument('--baseline', '-b', default='', help='可选：baseline pkl，用于对比')
    parser.add_argument('--output', '-o', default='analysis_report.json', help='输出JSON')
    args = parser.parse_args()

    if not os.path.exists(args.ai_pkl):
        raise FileNotFoundError(f'文件不存在: {args.ai_pkl}')

    ai_data = load_pkl(args.ai_pkl)
    ai_report = analyze_one(ai_data)

    result = {
        'ai_file': args.ai_pkl,
        'ai_report': ai_report,
    }

    if args.baseline:
        if not os.path.exists(args.baseline):
            raise FileNotFoundError(f'baseline文件不存在: {args.baseline}')
        base_data = load_pkl(args.baseline)
        base_report = analyze_one(base_data)
        result['baseline_file'] = args.baseline
        result['baseline_report'] = base_report
        result['delta_vs_baseline'] = diff_report(base_report, ai_report)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'✓ 数据分析已完成: {args.output}')

    s = ai_report['summary']
    print('\n[AI数据摘要]')
    print(f"  OCR(doc): {s['ocr_doc_formula']:.6f}")
    print(f"  到达率(simple): {s['arrive_ratio_simple']:.6f}")
    print(f"  出发/到达: {s['departed']}/{s['arrived']}")
    print(f"  观测车辆数: {s['observed_unique_vehicles']}")

    if 'delta_vs_baseline' in result:
        dlt = result['delta_vs_baseline']
        print('\n[相对Baseline差异]')
        for k, v in dlt.items():
            print(f'  {k}: {v:+.6f}')


if __name__ == '__main__':
    main()
