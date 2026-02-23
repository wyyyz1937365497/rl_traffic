"""
本地分数计算器 - 改进版
使用已知得分来校准参数，提供更准确的分数估算

使用方法:
1. 先用已知得分的文件校准参数:
   python local_score_calculator.py --calibrate submit.pkl:25.7926 submission.pkl:15.7650

2. 然后用校准后的参数计算新文件:
   python local_score_calculator.py new_submission.pkl
"""

import pickle
import numpy as np
import argparse
import os
import json
from typing import Dict, Tuple


class ImprovedScoreCalculator:
    """改进的本地分数计算器 - 支持校准"""

    def __init__(self, calibration_file: str = None):
        """
        Args:
            calibration_file: 校准参数文件路径
        """
        # 默认参数（将根据校准数据调整）
        self.baseline_ocr = 0.94
        self.baseline_speed_std = 6.5
        self.baseline_mean_accel = 0.3

        self.W_efficiency = 0.5
        self.W_stability = 0.5
        self.k = 10.0

        # 干预成本参数
        self.control_interval = 5
        self.cv_ratio = 0.1

        # 加载校准参数
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)

    def _load_pkl(self, pkl_path: str) -> Dict:
        """加载pkl文件"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def _calculate_metrics(self, data: Dict) -> Dict:
        """计算pkl文件的所有指标"""
        metrics = {}

        step_data = data.get('step_data', [])
        vehicle_data = data.get('vehicle_data', [])
        stats = data.get('statistics', {})

        # 基本统计
        metrics['total_steps'] = len(step_data)
        metrics['total_departed'] = stats.get('cumulative_departed', 0)
        metrics['total_arrived'] = stats.get('cumulative_arrived', 0)
        metrics['ocr'] = metrics['total_arrived'] / metrics['total_departed'] if metrics['total_departed'] > 0 else 0

        # 速度统计
        if vehicle_data:
            speeds = [r.get('speed', 0) for r in vehicle_data if 'speed' in r]
            if speeds:
                metrics['speed_std'] = np.std(speeds)
                metrics['speed_mean'] = np.mean(speeds)
            else:
                metrics['speed_std'] = 0
                metrics['speed_mean'] = 0
        else:
            metrics['speed_std'] = 0
            metrics['speed_mean'] = 0

        # 加速度统计
        if vehicle_data:
            vehicle_records = {}
            for record in vehicle_data:
                veh_id = record.get('vehicle_id')
                if veh_id:
                    if veh_id not in vehicle_records:
                        vehicle_records[veh_id] = []
                    vehicle_records[veh_id].append(record)

            all_accels = []
            for records in vehicle_records.values():
                sorted_records = sorted(records, key=lambda x: x.get('step', 0))
                for i in range(1, len(sorted_records)):
                    prev_speed = sorted_records[i-1].get('speed', 0)
                    curr_speed = sorted_records[i].get('speed', 0)
                    accel = (curr_speed - prev_speed) / 1.0
                    all_accels.append(accel)

            if all_accels:
                metrics['mean_abs_accel'] = np.mean(np.abs(all_accels))
            else:
                metrics['mean_abs_accel'] = 0
        else:
            metrics['mean_abs_accel'] = 0

        return metrics

    def calculate_score(self, pkl_path: str) -> Dict:
        """计算pkl文件的得分"""
        data = self._load_pkl(pkl_path)
        metrics = self._calculate_metrics(data)

        # 效率得分
        ocr_improvement = (metrics['ocr'] - self.baseline_ocr) / self.baseline_ocr
        s_efficiency = 100 * max(0, ocr_improvement)

        # 稳定性得分
        I_speed_std = -(metrics['speed_std'] - self.baseline_speed_std) / self.baseline_speed_std
        I_mean_accel = -(metrics['mean_abs_accel'] - self.baseline_mean_accel) / self.baseline_mean_accel
        s_stability = 100 * (0.4 * max(0, I_speed_std) + 0.6 * max(0, I_mean_accel))

        # 干预成本（估计）
        T = metrics['total_steps']
        N = metrics['total_departed']
        controlled_vehicles = int(N * self.cv_ratio)
        control_steps = T // self.control_interval
        total_accel_commands = control_steps * controlled_vehicles
        C_int = total_accel_commands / (T * N)

        P_int = np.exp(-self.k * C_int)

        # 总分
        s_total = (self.W_efficiency * s_efficiency + self.W_stability * s_stability) * P_int

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

    def calibrate(self, calibration_data: Dict):
        """
        根据已知得分校准参数

        Args:
            calibration_data: {pkl_file: actual_score}
        """
        print("=" * 80)
        print("Calibrating parameters using known scores...")
        print("=" * 80)

        # 收集所有文件的指标和实际得分
        file_data = []
        for pkl_file, actual_score in calibration_data.items():
            data = self._load_pkl(pkl_file)
            metrics = self._calculate_metrics(data)
            file_data.append({
                'file': pkl_file,
                'actual_score': actual_score,
                'metrics': metrics
            })
            print(f"\n{os.path.basename(pkl_file)}: {actual_score:.2f}")
            print(f"  OCR: {metrics['ocr']:.4f}")
            print(f"  speed_std: {metrics['speed_std']:.4f}")
            print(f"  mean_abs_accel: {metrics['mean_abs_accel']:.4f}")

        # 简化校准策略：
        # 假设 baseline_ocr = min(OCR) - 0.01
        # 假设 baseline_speed_std = max(speed_std)
        # 假设 baseline_mean_accel = max(mean_abs_accel)

        all_ocr = [d['metrics']['ocr'] for d in file_data]
        all_speed_std = [d['metrics']['speed_std'] for d in file_data]
        all_mean_accel = [d['metrics']['mean_abs_accel'] for d in file_data]

        self.baseline_ocr = min(all_ocr) - 0.005
        self.baseline_speed_std = max(all_speed_std)
        self.baseline_mean_accel = max(all_mean_accel)

        print(f"\n calibrated baseline values:")
        print(f"  baseline_ocr: {self.baseline_ocr:.4f}")
        print(f"  baseline_speed_std: {self.baseline_speed_std:.4f}")
        print(f"  baseline_mean_accel: {self.baseline_mean_accel:.4f}")

        # 计算理论得分并调整k值
        print(f"\n{'File':<30} {'Actual':<10} {'Calculated':<10} {'Ratio':<10}")
        print("-" * 60)

        ratios = []
        for d in file_data:
            m = d['metrics']

            # 效率得分
            ocr_imp = (m['ocr'] - self.baseline_ocr) / self.baseline_ocr
            s_eff = 100 * max(0, ocr_imp)

            # 稳定性得分
            I_std = -(m['speed_std'] - self.baseline_speed_std) / self.baseline_speed_std
            I_acc = -(m['mean_abs_accel'] - self.baseline_mean_accel) / self.baseline_mean_accel
            s_stab = 100 * (0.4 * max(0, I_std) + 0.6 * max(0, I_acc))

            # 干预惩罚（暂时忽略，设为1.0）
            P_int = 1.0

            s_calc = (self.W_efficiency * s_eff + self.W_stability * s_stab) * P_int

            ratio = d['actual_score'] / max(s_calc, 0.01)
            ratios.append(ratio)

            print(f"{os.path.basename(d['file']):<30} {d['actual_score']:<10.2f} {s_calc:<10.2f} {ratio:<10.2f}")

        # 使用平均ratio作为缩放因子
        avg_ratio = np.mean(ratios)

        print(f"\nScaling factor: {avg_ratio:.2f}")
        print(f"This will be used to adjust the final scores.")

        # 保存校准参数
        calibration_params = {
            'baseline_ocr': self.baseline_ocr,
            'baseline_speed_std': self.baseline_speed_std,
            'baseline_mean_accel': self.baseline_mean_accel,
            'W_efficiency': self.W_efficiency,
            'W_stability': self.W_stability,
            'scaling_factor': avg_ratio,
            'calibrated_with': [os.path.basename(d['file']) for d in file_data]
        }

        self.save_calibration('score_calibration.json', calibration_params)

        return calibration_params

    def save_calibration(self, filename: str, params: Dict):
        """保存校准参数"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"\nCalibration saved to: {filename}")

    def load_calibration(self, filename: str):
        """加载校准参数"""
        with open(filename, 'r', encoding='utf-8') as f:
            params = json.load(f)

        self.baseline_ocr = params.get('baseline_ocr', self.baseline_ocr)
        self.baseline_speed_std = params.get('baseline_speed_std', self.baseline_speed_std)
        self.baseline_mean_accel = params.get('baseline_mean_accel', self.baseline_mean_accel)
        self.W_efficiency = params.get('W_efficiency', self.W_efficiency)
        self.W_stability = params.get('W_stability', self.W_stability)
        self.scaling_factor = params.get('scaling_factor', 1.0)

        print(f"Calibration loaded from: {filename}")

    def calculate_score_calibrated(self, pkl_path: str) -> Dict:
        """使用校准后的参数计算得分"""
        result = self.calculate_score(pkl_path)

        # 应用缩放因子
        if hasattr(self, 'scaling_factor'):
            result['total_score'] *= self.scaling_factor

        return result


def main():
    parser = argparse.ArgumentParser(description='Local Score Calculator - Improved')
    parser.add_argument('pkl_files', nargs='+', help='pkl文件路径')
    parser.add_argument('--calibrate', '-c', action='store_true',
                        help='校准模式（需要提供实际得分，格式: file.pkl:score）')
    parser.add_argument('--use-calibration', '-uc', default='score_calibration.json',
                        help='使用指定的校准文件')

    args = parser.parse_args()

    calculator = ImprovedScoreCalculator(args.use_calibration)

    # 校准模式
    if args.calibrate:
        calibration_data = {}
        for item in args.pkl_files:
            if ':' in item:
                pkl_file, score = item.rsplit(':', 1)
                if os.path.exists(pkl_file):
                    calibration_data[pkl_file] = float(score)
                else:
                    print(f"Warning: File not found: {pkl_file}")

        if calibration_data:
            calculator.calibrate(calibration_data)
        else:
            print("Error: No valid calibration data provided")
            print("Usage: python local_score_calculator.py --calibrate file1.pkl:25.79 file2.pkl:15.77")
        return

    # 计算模式
    print("=" * 80)
    print("Local Score Calculator")
    print("=" * 80)

    results = []
    for pkl_file in args.pkl_files:
        if os.path.exists(pkl_file):
            result = calculator.calculate_score_calibrated(pkl_file)
            result['pkl_file'] = pkl_file
            results.append(result)

            print(f"\n{os.path.basename(pkl_file)}")
            print("-" * 40)
            print(f"  Total Score: {result['total_score']:.2f}")
            print(f"  Efficiency: {result['efficiency_score']:.2f}")
            print(f"  Stability: {result['stability_score']:.2f}")
            print(f"  Penalty: {result['intervention_penalty']:.4f}")
            print(f"\n  Metrics:")
            print(f"    OCR: {result['metrics']['ocr']:.4f}")
            print(f"    speed_std: {result['metrics']['speed_std']:.4f} m/s")
            print(f"    mean_abs_accel: {result['metrics']['mean_abs_accel']:.4f} m/s^2")

    # 对比多个文件
    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("Comparison")
        print(f"{'=' * 80}")
        print(f"\n{'File':<40} {'Score':<10}")
        print("-" * 50)

        results.sort(key=lambda x: x['total_score'], reverse=True)
        for r in results:
            print(f"{os.path.basename(r['pkl_file']):<40} {r['total_score']:<10.2f}")


if __name__ == '__main__':
    main()
