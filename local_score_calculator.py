"""
本地分数计算器
基于比赛评分公式计算提交文件的得分，避免频繁上传

评分公式:
S_total = (W_efficiency × S_efficiency + W_stability × S_stability) × e^(-k × C_int)

其中:
- S_efficiency = 100 × max(0, (OCR_AI - OCR_Base) / OCR_Base)
- S_stability = 100 × [0.4 × I_σv + 0.6 × I_|a|_avg]
- I_σv = -(σv_AI - σv_Base) / σv_Base
- I_|a| = -(|a|_AI - |a|_Base) / |a|_Base
- C_int = (1/(T × N)) × Σ(α × acmd + β × δlc)
"""

import pickle
import numpy as np
import argparse
import os
from typing import Dict, Tuple
import json


class LocalScoreCalculator:
    """本地分数计算器"""

    def __init__(self, baseline_pkl: str = None):
        """
        Args:
            baseline_pkl: baseline的pkl文件路径，用于计算baseline指标
        """
        self.baseline_data = None
        self.baseline_metrics = None

        if baseline_pkl and os.path.exists(baseline_pkl):
            print(f"加载baseline数据: {baseline_pkl}")
            self.baseline_data = self._load_pkl(baseline_pkl)
            self.baseline_metrics = self._calculate_metrics(self.baseline_data)
            print(f"Baseline OCR: {self.baseline_metrics['ocr']:.4f}")
            print(f"Baseline speed_std: {self.baseline_metrics['speed_std']:.4f} m/s")
            print(f"Baseline mean_abs_accel: {self.baseline_metrics['mean_abs_accel']:.4f} m/s^2")
        else:
            print("⚠ 未提供baseline文件，将使用默认值")
            self.baseline_metrics = {
                'ocr': 0.94,
                'speed_std': 8.0,
                'mean_abs_accel': 1.2
            }

        # 评分权重
        self.W_efficiency = 0.5
        self.W_stability = 0.5

        # 干预成本参数
        self.alpha = 1.0  # 加速度指令权重
        self.beta = 5.0   # 换道指令权重
        self.k = 10.0     # 惩罚系数（需要标定）

    def _load_pkl(self, pkl_path: str) -> Dict:
        """加载pkl文件"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def _calculate_metrics(self, data: Dict) -> Dict:
        """计算pkl文件的所有指标"""
        metrics = {}

        # 基本统计
        step_data = data.get('step_data', [])
        vehicle_data = data.get('vehicle_data', [])

        if not step_data:
            raise ValueError("pkl文件中没有step_data")

        # 时间步数
        metrics['total_steps'] = len(step_data)

        # 车辆统计
        stats = data.get('statistics', {})
        metrics['total_departed'] = stats.get('cumulative_departed', 0)
        metrics['total_arrived'] = stats.get('cumulative_arrived', 0)

        # OCR计算
        # OCR = (N_arrived + Σ(d_traveled / d_total)) / N_total
        # 由于pkl中没有详细的行驶距离，我们使用简化公式
        if metrics['total_departed'] > 0:
            metrics['ocr'] = metrics['total_arrived'] / metrics['total_departed']
        else:
            metrics['ocr'] = 0.0

        # 从vehicle_data计算速度和加速度统计
        if vehicle_data:
            speeds = []
            accels = []

            for record in vehicle_data:
                if 'speed' in record:
                    speeds.append(record['speed'])
                # acceleration字段可能不存在，需要计算
                # 这里先跳过，后面从step_data计算

            if speeds:
                metrics['speed_mean'] = np.mean(speeds)
                metrics['speed_std'] = np.std(speeds)
                metrics['speed_min'] = np.min(speeds)
                metrics['speed_max'] = np.max(speeds)
            else:
                metrics['speed_mean'] = 0.0
                metrics['speed_std'] = 0.0
                metrics['speed_min'] = 0.0
                metrics['speed_max'] = 0.0

        # 从step_data计算活动车辆和平均速度
        active_vehicles = []
        arrived_per_step = []

        for step in step_data:
            active_vehicles.append(step.get('active_vehicles', 0))
            arrived_per_step.append(step.get('arrived_vehicles', 0))

        metrics['mean_active_vehicles'] = np.mean(active_vehicles) if active_vehicles else 0
        metrics['max_active_vehicles'] = np.max(active_vehicles) if active_vehicles else 0

        # 计算加速度统计（需要相邻时间步的速度差）
        # 由于vehicle_data包含所有车辆的所有时间步，我们需要按车辆分组
        if vehicle_data and len(vehicle_data) > 0:
            try:
                # 按车辆ID分组
                vehicle_records = {}
                for record in vehicle_data:
                    veh_id = record.get('vehicle_id')
                    if veh_id:
                        if veh_id not in vehicle_records:
                            vehicle_records[veh_id] = []
                        vehicle_records[veh_id].append(record)

                # 计算每辆车的加速度
                all_accels = []
                for veh_id, records in vehicle_records.items():
                    # 按时间步排序
                    sorted_records = sorted(records, key=lambda x: x.get('step', 0))
                    # 计算相邻时间步的速度差（加速度）
                    for i in range(1, len(sorted_records)):
                        prev_speed = sorted_records[i-1].get('speed', 0)
                        curr_speed = sorted_records[i].get('speed', 0)
                        # 加速度 = (v_t - v_{t-1}) / dt
                        dt = 1.0  # 时间步长
                        accel = (curr_speed - prev_speed) / dt
                        all_accels.append(accel)

                if all_accels:
                    metrics['mean_abs_accel'] = np.mean(np.abs(all_accels))
                    metrics['accel_std'] = np.std(all_accels)
                    metrics['max_accel'] = np.max(np.abs(all_accels))
                else:
                    metrics['mean_abs_accel'] = 0.0
                    metrics['accel_std'] = 0.0
                    metrics['max_accel'] = 0.0

            except Exception as e:
                print(f"⚠ 计算加速度时出错: {e}")
                metrics['mean_abs_accel'] = 0.0
                metrics['accel_std'] = 0.0
                metrics['max_accel'] = 0.0
        else:
            metrics['mean_abs_accel'] = 0.0
            metrics['accel_std'] = 0.0
            metrics['max_accel'] = 0.0

        return metrics

    def calculate_score(self, pkl_path: str) -> Dict:
        """
        计算pkl文件的得分

        Args:
            pkl_path: pkl文件路径

        Returns:
            包含各项得分和指标的字典
        """
        print(f"\n{'=' * 80}")
        print(f"计算得分: {pkl_path}")
        print(f"{'=' * 80}")

        # 加载数据
        data = self._load_pkl(pkl_path)
        metrics = self._calculate_metrics(data)

        # 打印基本统计
        print(f"\n[基本统计]")
        print(f"  总时间步: {metrics['total_steps']}")
        print(f"  总发出车辆: {metrics['total_departed']}")
        print(f"  总到达车辆: {metrics['total_arrived']}")
        print(f"  平均活动车辆: {metrics['mean_active_vehicles']:.1f}")
        print(f"  最大活动车辆: {metrics['max_active_vehicles']}")

        # 1. 效率得分
        print(f"\n[效率得分]")
        print(f"  OCR (完成率): {metrics['ocr']:.4f}")
        print(f"  Baseline OCR: {self.baseline_metrics['ocr']:.4f}")

        ocr_improvement = (metrics['ocr'] - self.baseline_metrics['ocr']) / self.baseline_metrics['ocr']
        s_efficiency = 100 * max(0, ocr_improvement)

        print(f"  OCR改善: {ocr_improvement:+.2%}")
        print(f"  效率得分 S_efficiency: {s_efficiency:.2f}")

        # 2. 稳定性得分
        print(f"\n[稳定性得分]")
        print(f"  速度标准差: {metrics['speed_std']:.4f} m/s")
        print(f"  Baseline 速度标准差: {self.baseline_metrics['speed_std']:.4f} m/s")

        I_speed_std = -(metrics['speed_std'] - self.baseline_metrics['speed_std']) / self.baseline_metrics['speed_std']
        print(f"  I_σv (速度标准差改善): {I_speed_std:+.4f}")

        print(f"  平均绝对加速度: {metrics['mean_abs_accel']:.4f} m/s^2")
        print(f"  Baseline 平均绝对加速度: {self.baseline_metrics['mean_abs_accel']:.4f} m/s^2")

        I_mean_accel = -(metrics['mean_abs_accel'] - self.baseline_metrics['mean_abs_accel']) / self.baseline_metrics['mean_abs_accel']
        print(f"  I_|a| (加速度改善): {I_mean_accel:+.4f}")

        s_stability = 100 * (0.4 * max(0, I_speed_std) + 0.6 * max(0, I_mean_accel))

        print(f"  稳定性得分 S_stability: {s_stability:.2f}")
        print(f"    = 100 × (0.4 × {max(0, I_speed_std):.4f} + 0.6 × {max(0, I_mean_accel):.4f})")

        # 3. 干预成本惩罚
        # 由于pkl中没有记录控制指令，我们使用估计值
        # 假设每步控制所有CV车辆，每辆车每步发出一次加速度指令
        print(f"\n[干预成本惩罚]")

        # 估计干预成本
        # C_int = (1/(T × N)) × Σ(α × acmd + β × δlc)
        T = metrics['total_steps']
        N = metrics['total_departed']

        # 如果有action数据，使用实际数据；否则使用估计
        if 'action_data' in data:
            # 使用实际的控制数据
            action_data = data['action_data']
            total_accel_commands = sum(1 for actions in action_data.values()
                                       for action in actions.values()
                                       if action.get('acceleration', 0) != 0)
            total_lane_changes = sum(1 for actions in action_data.values()
                                     for action in actions.values()
                                     if action.get('lane_change', None) is not None)
        else:
            # 估计：假设每5步控制一次，每次控制10%的CV车辆
            control_interval = 5
            cv_ratio = 0.1
            controlled_vehicles_per_step = int(N * cv_ratio)
            control_steps = T // control_interval

            total_accel_commands = control_steps * controlled_vehicles_per_step
            total_lane_changes = 0  # 假设不换道

            print(f"  ⚠ 未找到控制数据，使用估计值:")
            print(f"    控制间隔: {control_interval} 步")
            print(f"    CV比例: {cv_ratio:.1%}")
            print(f"    估计控制步数: {control_steps}")
            print(f"    估计每步控制车辆数: {controlled_vehicles_per_step}")

        C_int = (self.alpha * total_accel_commands + self.beta * total_lane_changes) / (T * N)

        print(f"  总加速度指令数: {total_accel_commands}")
        print(f"  总换道指令数: {total_lane_changes}")
        print(f"  干预成本 C_int: {C_int:.6f}")

        P_int = np.exp(-self.k * C_int)

        print(f"  惩罚因子 P_int = e^(-{self.k} × {C_int:.6f}) = {P_int:.4f}")
        print(f"  得分保留率: {P_int * 100:.1f}%")

        # 4. 总分
        s_total = (self.W_efficiency * s_efficiency + self.W_stability * s_stability) * P_int

        print(f"\n[最终得分]")
        print(f"  S_total = ({self.W_efficiency} × {s_efficiency:.2f} + {self.W_stability} × {s_stability:.2f}) × {P_int:.4f}")
        print(f"  S_total = {s_total:.4f}")

        return {
            'total_score': s_total,
            'efficiency_score': s_efficiency,
            'stability_score': s_stability,
            'intervention_penalty': P_int,
            'intervention_cost': C_int,
            'metrics': metrics,
            'improvements': {
                'ocr_improvement': ocr_improvement,
                'I_speed_std': I_speed_std,
                'I_mean_accel': I_mean_accel
            }
        }


def compare_pkl_files(pkl_files: list, baseline_pkl: str = None):
    """对比多个pkl文件的得分"""
    calculator = LocalScoreCalculator(baseline_pkl)

    results = []

    for pkl_file in pkl_files:
        if os.path.exists(pkl_file):
            result = calculator.calculate_score(pkl_file)
            result['pkl_file'] = pkl_file
            results.append(result)
        else:
            print(f"⚠ 文件不存在: {pkl_file}")

    # 打印对比表格
    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("得分对比")
        print(f"{'=' * 80}")

        print(f"\n{'文件':<40} {'总分':<10} {'效率得分':<10} {'稳定性得分':<10} {'惩罚因子':<10}")
        print("-" * 80)

        # 按总分排序
        results.sort(key=lambda x: x['total_score'], reverse=True)

        for result in results:
            pkl_name = os.path.basename(result['pkl_file'])
            print(f"{pkl_name:<40} "
                  f"{result['total_score']:<10.2f} "
                  f"{result['efficiency_score']:<10.2f} "
                  f"{result['stability_score']:<10.2f} "
                  f"{result['intervention_penalty']:<10.4f}")

        # 打印详细对比
        print(f"\n{'=' * 80}")
        print("详细指标对比")
        print(f"{'=' * 80}")

        print(f"\n{'文件':<40} {'OCR':<10} {'速度标准差':<12} {'平均绝对加速度':<15}")
        print("-" * 80)

        for result in results:
            pkl_name = os.path.basename(result['pkl_file'])
            metrics = result['metrics']
            print(f"{pkl_name:<40} "
                  f"{metrics['ocr']:<10.4f} "
                  f"{metrics['speed_std']:<12.4f} "
                  f"{metrics['mean_abs_accel']:<15.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='本地分数计算器')
    parser.add_argument('pkl_files', nargs='+', help='pkl文件路径（可多个）')
    parser.add_argument('--baseline', '-b', default=None,
                        help='baseline pkl文件路径（用于计算baseline指标）')
    parser.add_argument('--output', '-o', default=None,
                        help='保存结果到JSON文件')

    args = parser.parse_args()

    # 计算得分
    results = compare_pkl_files(args.pkl_files, args.baseline)

    # 保存结果
    if args.output and results:
        output_data = []
        for result in results:
            output_data.append({
                'pkl_file': result['pkl_file'],
                'total_score': round(result['total_score'], 4),
                'efficiency_score': round(result['efficiency_score'], 4),
                'stability_score': round(result['stability_score'], 4),
                'intervention_penalty': round(result['intervention_penalty'], 4),
                'intervention_cost': round(result['intervention_cost'], 6),
                'ocr': round(result['metrics']['ocr'], 4),
                'speed_std': round(result['metrics']['speed_std'], 4),
                'mean_abs_accel': round(result['metrics']['mean_abs_accel'], 4),
                'ocr_improvement': round(result['improvements']['ocr_improvement'], 4),
                'I_speed_std': round(result['improvements']['I_speed_std'], 4),
                'I_mean_accel': round(result['improvements']['I_mean_accel'], 4),
            })

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
