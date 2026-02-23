"""
本地分数计算器 - 简化实用版
基于OCR估算得分，适用于OCR接近95%的情况

核心思路：
1. 当OCR接近95%时，效率得分的提升空间有限
2. 使用已知得分建立OCR -> Score的映射关系
3. 考虑其他因素（稳定性、干预成本）的修正

使用方法：
1. 校准（使用已知得分）:
   python simple_score_calculator.py --calibrate file1.pkl:25.79 file2.pkl:15.77

2. 计算新文件:
   python simple_score_calculator.py new_file.pkl
"""

import pickle
import numpy as np
import argparse
import os
import json
from typing import Dict


class SimpleScoreCalculator:
    """简化的本地分数计算器"""

    def __init__(self, calibration_file: str = None):
        self.calibration_data = []
        self.scaling_params = {
            'ocr_weight': 1000,  # OCR对得分的影响权重
            'ocr_baseline': 0.94,
            'intercept': -930,   # 截距
        }

        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)

    def _load_pkl(self, pkl_path: str) -> Dict:
        """加载pkl文件"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def _calculate_ocr(self, data: Dict) -> float:
        """计算OCR"""
        stats = data.get('statistics', {})
        total_departed = stats.get('cumulative_departed', 0)
        total_arrived = stats.get('cumulative_arrived', 0)

        if total_departed > 0:
            return total_arrived / total_departed
        return 0.0

    def _get_additional_metrics(self, data: Dict) -> Dict:
        """获取额外的修正指标"""
        step_data = data.get('step_data', [])
        vehicle_data = data.get('vehicle_data', [])

        metrics = {}

        # 平均活动车辆数（路网负载）
        active_vehicles = [s.get('active_vehicles', 0) for s in step_data]
        metrics['mean_active'] = np.mean(active_vehicles) if active_vehicles else 0

        # 结束时在途车辆（拥堵程度）
        if step_data:
            metrics['final_active'] = step_data[-1].get('active_vehicles', 0)
        else:
            metrics['final_active'] = 0

        # 速度标准差
        if vehicle_data:
            speeds = [r.get('speed', 0) for r in vehicle_data if 'speed' in r]
            metrics['speed_std'] = np.std(speeds) if speeds else 0
        else:
            metrics['speed_std'] = 0

        return metrics

    def calculate_score(self, pkl_path: str) -> Dict:
        """计算得分"""
        data = self._load_pkl(pkl_path)
        ocr = self._calculate_ocr(data)
        metrics = self._get_additional_metrics(data)

        # 基于OCR的基准得分
        base_score = self.scaling_params['ocr_weight'] * (ocr - self.scaling_params['ocr_baseline']) + self.scaling_params['intercept']

        # 修正因子（基于其他指标）
        # 路网负载修正：活动车辆越少越好
        if 'mean_active_baseline' in self.scaling_params:
            load_factor = 1 - (metrics['mean_active'] - self.scaling_params['mean_active_baseline']) / 500
            load_factor = np.clip(load_factor, 0.95, 1.05)
        else:
            load_factor = 1.0

        # 拥堵修正：结束时在途车辆越少越好
        if 'final_active_baseline' in self.scaling_params:
            congestion_factor = 1 - (metrics['final_active'] - self.scaling_params['final_active_baseline']) / 100
            congestion_factor = np.clip(congestion_factor, 0.95, 1.05)
        else:
            congestion_factor = 1.0

        # 总分
        total_score = base_score * load_factor * congestion_factor

        return {
            'total_score': total_score,
            'ocr': ocr,
            'base_score': base_score,
            'metrics': metrics
        }

    def calibrate(self, calibration_data: Dict):
        """使用已知得分校准参数"""
        print("=" * 80)
        print("Simple Score Calculator - Calibration")
        print("=" * 80)

        # 收集数据
        samples = []
        for pkl_file, actual_score in calibration_data.items():
            data = self._load_pkl(pkl_file)
            ocr = self._calculate_ocr(data)
            metrics = self._get_additional_metrics(data)
            samples.append({
                'file': os.path.basename(pkl_file),
                'actual_score': actual_score,
                'ocr': ocr,
                'metrics': metrics
            })

        # 打印数据
        print(f"\n{'File':<30} {'Actual':<10} {'OCR':<10} {'MeanActive':<12} {'FinalActive':<12}")
        print("-" * 74)
        for s in samples:
            print(f"{s['file']:<30} {s['actual_score']:<10.2f} {s['ocr']:<10.4f} "
                  f"{s['metrics']['mean_active']:<12.1f} {s['metrics']['final_active']:<12}")

        # 简单线性回归：score = a * (ocr - baseline) + b
        # 使用最小二乘法
        ocr_vals = np.array([s['ocr'] for s in samples])
        score_vals = np.array([s['actual_score'] for s in samples])

        # 设定baseline为最小OCR
        ocr_baseline = ocr_vals.min() - 0.001

        # 构造特征矩阵 X = [ocr - baseline]
        X = (ocr_vals - ocr_baseline).reshape(-1, 1)

        # 添加截距项
        X_with_intercept = np.column_stack([X, np.ones(len(X))])

        # 最小二乘拟合
        result = np.linalg.lstsq(X_with_intercept, score_vals, rcond=None)
        params = result[0]

        ocr_weight = params[0]
        intercept = params[1]

        # 保存参数
        self.scaling_params = {
            'ocr_weight': ocr_weight,
            'ocr_baseline': ocr_baseline,
            'intercept': intercept,
        }

        # 计算修正因子（使用平均活动车辆数）
        mean_active_vals = np.array([s['metrics']['mean_active'] for s in samples])
        final_active_vals = np.array([s['metrics']['final_active'] for s in samples])

        self.scaling_params['mean_active_baseline'] = mean_active_vals.mean()
        self.scaling_params['final_active_baseline'] = final_active_vals.mean()

        # 验证拟合
        print(f"\nCalibrated parameters:")
        print(f"  score = {ocr_weight:.1f} * (OCR - {ocr_baseline:.4f}) + {intercept:.1f}")
        print(f"  mean_active_baseline = {mean_active_vals.mean():.1f}")
        print(f"  final_active_baseline = {final_active_vals.mean():.1f}")

        print(f"\n{'File':<30} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
        print("-" * 60)

        errors = []
        for s in samples:
            # 预测得分
            ocr = s['ocr']
            predicted = ocr_weight * (ocr - ocr_baseline) + intercept

            # 修正
            load_correction = 1.0  # 简化版本暂不使用修正
            congestion_correction = 1.0
            predicted_final = predicted * load_correction * congestion_correction

            error = abs(predicted_final - s['actual_score'])
            errors.append(error)

            print(f"{s['file']:<30} {s['actual_score']:<10.2f} {predicted_final:<10.2f} {error:<10.2f}")

        mae = np.mean(errors)
        print(f"\nMean Absolute Error: {mae:.2f}")

        # 保存校准
        calibration = {
            'ocr_weight': float(ocr_weight),
            'ocr_baseline': float(ocr_baseline),
            'intercept': float(intercept),
            'mean_active_baseline': float(mean_active_vals.mean()),
            'final_active_baseline': float(final_active_vals.mean()),
            'mae': float(mae),
            'calibrated_with': [s['file'] for s in samples]
        }

        self.save_calibration('simple_calibration.json', calibration)

        return calibration

    def save_calibration(self, filename: str, params: Dict):
        """保存校准参数"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
        print(f"\nCalibration saved to: {filename}")

    def load_calibration(self, filename: str):
        """加载校准参数"""
        with open(filename, 'r', encoding='utf-8') as f:
            params = json.load(f)
        self.scaling_params = params
        print(f"Calibration loaded from: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Simple Local Score Calculator')
    parser.add_argument('pkl_files', nargs='+', help='pkl文件路径')
    parser.add_argument('--calibrate', '-c', action='store_true',
                        help='校准模式（格式: file.pkl:score）')
    parser.add_argument('--use-calibration', '-uc', default='simple_calibration.json',
                        help='使用指定的校准文件')

    args = parser.parse_args()

    calculator = SimpleScoreCalculator(args.use_calibration)

    # 校准模式
    if args.calibrate:
        calibration_data = {}
        for item in args.pkl_files:
            if ':' in item:
                pkl_file, score = item.rsplit(':', 1)
                if os.path.exists(pkl_file):
                    calibration_data[pkl_file] = float(score)

        if calibration_data:
            calculator.calibrate(calibration_data)
        else:
            print("Error: No valid calibration data")
        return

    # 计算模式
    if not os.path.exists(args.use_calibration):
        print(f"Warning: Calibration file not found: {args.use_calibration}")
        print("Please run with --calibrate first")
        return

    print("=" * 80)
    print("Simple Local Score Calculator")
    print("=" * 80)

    results = []
    for pkl_file in args.pkl_files:
        if os.path.exists(pkl_file):
            result = calculator.calculate_score(pkl_file)
            result['pkl_file'] = pkl_file
            results.append(result)

            print(f"\n{os.path.basename(pkl_file)}")
            print("-" * 40)
            print(f"  Score: {result['total_score']:.2f}")
            print(f"  OCR: {result['ocr']:.4f}")
            print(f"  Base Score: {result['base_score']:.2f}")

    # 对比
    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("Comparison")
        print(f"{'=' * 80}")
        print(f"\n{'File':<40} {'Score':<10} {'OCR':<10}")
        print("-" * 60)

        results.sort(key=lambda x: x['total_score'], reverse=True)
        for r in results:
            print(f"{os.path.basename(r['pkl_file']):<40} {r['total_score']:<10.2f} {r['ocr']:<10.4f}")


if __name__ == '__main__':
    main()
