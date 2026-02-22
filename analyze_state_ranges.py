"""
状态变量范围分析脚本
从实际SUMO仿真中收集所有状态变量的统计信息，用于确定合理的归一化参数
"""

import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from sumo.main import SUMOCompetitionFramework
from junction_agent import JunctionAgent, JunctionType


class StateRangeAnalyzer:
    """状态变量范围分析器"""

    def __init__(self):
        self.stats = defaultdict(lambda: {
            'values': [],
            'name': ''
        })

        # 定义要追踪的状态变量
        self.variables = [
            # 主路变量
            'main_vehicles_count',      # len(state.main_vehicles)
            'main_speed',               # state.main_speed
            'main_density',             # state.main_density
            'main_queue_length',        # state.main_queue_length
            'main_flow',                # state.main_flow

            # 匝道变量
            'ramp_vehicles_count',      # len(state.ramp_vehicles)
            'ramp_speed',               # state.ramp_speed
            'ramp_queue_length',        # state.ramp_queue_length
            'ramp_waiting_time',        # state.ramp_waiting_time
            'ramp_flow',                # state.ramp_flow

            # 分流变量（TYPE_B）
            'diverge_vehicles_count',   # len(state.diverge_vehicles)
            'diverge_queue_length',     # state.diverge_queue_length

            # 信号变量
            'current_phase',            # state.current_phase
            'time_to_switch',           # state.time_to_switch

            # 风险和间隙
            'conflict_risk',            # state.conflict_risk
            'gap_acceptance',           # state.gap_acceptance

            # 时间
            'timestamp',                # state.timestamp
        ]

        # 初始化统计字典
        for var in self.variables:
            self.stats[var]['name'] = var
            self.stats[var]['values'] = []

    def collect_state_data(self, state, junction_type):
        """收集单个状态的数据"""
        data = {
            'main_vehicles_count': len(state.main_vehicles),
            'main_speed': state.main_speed,
            'main_density': state.main_density,
            'main_queue_length': state.main_queue_length,
            'main_flow': state.main_flow,

            'ramp_vehicles_count': len(state.ramp_vehicles),
            'ramp_speed': state.ramp_speed,
            'ramp_queue_length': state.ramp_queue_length,
            'ramp_waiting_time': state.ramp_waiting_time,
            'ramp_flow': state.ramp_flow,

            'current_phase': state.current_phase,
            'time_to_switch': state.time_to_switch,
            'conflict_risk': state.conflict_risk,
            'gap_acceptance': state.gap_acceptance,
            'timestamp': state.timestamp,
        }

        # TYPE_B路口的分流变量
        if junction_type == JunctionType.TYPE_B:
            data.update({
                'diverge_vehicles_count': len(state.diverge_vehicles),
                'diverge_queue_length': state.diverge_queue_length,
            })
        else:
            data.update({
                'diverge_vehicles_count': 0,
                'diverge_queue_length': 0,
            })

        # 存储数据
        for var, value in data.items():
            self.stats[var]['values'].append(value)

    def collect_state_data_direct(self, agent):
        """直接使用TraCI查询收集状态数据（不依赖订阅）"""
        import traci
        try:
            config = agent.config

            # 主路数据
            main_veh_ids = []
            for edge_id in config.main_incoming:
                main_veh_ids.extend(traci.edge.getLastStepVehicleIDs(edge_id))
            main_speeds = [traci.vehicle.getSpeed(v) for v in main_veh_ids] if main_veh_ids else [0]
            main_speed = sum(main_speeds) / len(main_speeds) if main_speeds else 0.0
            main_halting = sum(1 for v in main_veh_ids if traci.vehicle.getSpeed(v) < 0.1) if main_veh_ids else 0

            # 匝道数据
            ramp_veh_ids = []
            for edge_id in config.ramp_incoming:
                ramp_veh_ids.extend(traci.edge.getLastStepVehicleIDs(edge_id))
            ramp_speeds = [traci.vehicle.getSpeed(v) for v in ramp_veh_ids] if ramp_veh_ids else [0]
            ramp_speed = sum(ramp_speeds) / len(ramp_speeds) if ramp_speeds else 0.0
            ramp_waiting = sum(traci.vehicle.getWaitingTime(v) for v in ramp_veh_ids) if ramp_veh_ids else 0.0
            ramp_halting = sum(1 for v in ramp_veh_ids if traci.vehicle.getSpeed(v) < 0.1) if ramp_veh_ids else 0

            # 分流数据
            diverge_veh_ids = []
            if agent.junction_type == JunctionType.TYPE_B:
                for edge_id in config.ramp_outgoing:
                    diverge_veh_ids.extend(traci.edge.getLastStepVehicleIDs(edge_id))
                diverge_halting = sum(1 for v in diverge_veh_ids if traci.vehicle.getSpeed(v) < 0.1) if diverge_veh_ids else 0
            else:
                diverge_halting = 0

            # 信号灯数据
            current_phase = 0
            time_to_switch = 0.0
            if config.has_traffic_light:
                try:
                    current_phase = traci.trafficlight.getPhase(config.tl_id)
                    time_to_switch = traci.trafficlight.getTimeToNextSwitch(config.tl_id)
                except:
                    pass

            data = {
                'main_vehicles_count': len(main_veh_ids),
                'main_speed': main_speed,
                'main_density': len(main_veh_ids) / max(len(config.main_incoming), 1),
                'main_queue_length': main_halting,
                'main_flow': len(main_veh_ids) * main_speed,

                'ramp_vehicles_count': len(ramp_veh_ids),
                'ramp_speed': ramp_speed,
                'ramp_queue_length': ramp_halting,
                'ramp_waiting_time': ramp_waiting / max(len(ramp_veh_ids), 1),
                'ramp_flow': len(ramp_veh_ids) * ramp_speed,

                'diverge_vehicles_count': len(diverge_veh_ids),
                'diverge_queue_length': diverge_halting,

                'current_phase': float(current_phase),
                'time_to_switch': time_to_switch,
                'conflict_risk': 0.0,  # 需要更复杂的计算
                'gap_acceptance': 1.0,  # 需要更复杂的计算
                'timestamp': traci.simulation.getTime(),
            }

            # 存储数据
            for var, value in data.items():
                self.stats[var]['values'].append(value)

        except Exception as e:
            print(f"[ERROR] collect_state_data_direct: {e}")
            import traceback
            traceback.print_exc()

    def compute_statistics(self):
        """计算统计信息"""
        results = {}

        for var, data in self.stats.items():
            values = np.array(data['values'])

            if len(values) == 0:
                results[var] = {
                    'name': data['name'],
                    'count': 0,
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'std': 0,
                    'median': 0,
                    'p95': 0,
                    'p99': 0,
                    'p999': 0,
                }
                continue

            results[var] = {
                'name': data['name'],
                'count': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99)),
                'p999': float(np.percentile(values, 99.9)),
            }

        return results

    def suggest_normalization(self, stats):
        """建议归一化参数"""
        suggestions = {}

        # 主路变量
        suggestions['main_vehicles_count'] = {
            'current_divisor': 20.0,
            'suggested_divisor': max(stats['main_vehicles_count']['p99'], 1.0),
            'reason': f'99分位数: {stats["main_vehicles_count"]["p95"]:.1f}'
        }

        suggestions['main_speed'] = {
            'current_divisor': 20.0,
            'suggested_divisor': max(stats['main_speed']['p99'], 1.0),
            'reason': f'99分位数: {stats["main_speed"]["p99"]:.1f} m/s'
        }

        suggestions['main_density'] = {
            'current_divisor': 50.0,
            'suggested_divisor': max(stats['main_density']['p99'], 1.0),
            'reason': f'99分位数: {stats["main_density"]["p99"]:.1f}'
        }

        suggestions['main_queue_length'] = {
            'current_divisor': 20.0,
            'suggested_divisor': max(stats['main_queue_length']['p99'], 1.0),
            'reason': f'99分位数: {stats["main_queue_length"]["p99"]:.1f}'
        }

        suggestions['main_flow'] = {
            'current_divisor': 1000.0,
            'suggested_divisor': max(stats['main_flow']['p99'], 1.0),
            'reason': f'99分位数: {stats["main_flow"]["p99"]:.1f}'
        }

        # 匝道变量
        suggestions['ramp_vehicles_count'] = {
            'current_divisor': 10.0,
            'suggested_divisor': max(stats['ramp_vehicles_count']['p99'], 1.0),
            'reason': f'99分位数: {stats["ramp_vehicles_count"]["p99"]:.1f}'
        }

        suggestions['ramp_speed'] = {
            'current_divisor': 20.0,
            'suggested_divisor': max(stats['ramp_speed']['p99'], 1.0),
            'reason': f'99分位数: {stats["ramp_speed"]["p99"]:.1f} m/s'
        }

        suggestions['ramp_queue_length'] = {
            'current_divisor': 10.0,
            'suggested_divisor': max(stats['ramp_queue_length']['p99'], 1.0),
            'reason': f'99分位数: {stats["ramp_queue_length"]["p99"]:.1f}'
        }

        suggestions['ramp_waiting_time'] = {
            'current_divisor': 60.0,
            'suggested_divisor': max(stats['ramp_waiting_time']['p99'], 1.0),
            'reason': f'99分位数: {stats["ramp_waiting_time"]["p99"]:.1f} s'
        }

        suggestions['ramp_flow'] = {
            'current_divisor': 500.0,
            'suggested_divisor': max(stats['ramp_flow']['p99'], 1.0),
            'reason': f'99分位数: {stats["ramp_flow"]["p99"]:.1f}'
        }

        # 分流变量
        suggestions['diverge_vehicles_count'] = {
            'current_divisor': 10.0,
            'suggested_divisor': max(stats['diverge_vehicles_count']['p99'], 1.0),
            'reason': f'99分位数: {stats["diverge_vehicles_count"]["p99"]:.1f}'
        }

        suggestions['diverge_queue_length'] = {
            'current_divisor': 10.0,
            'suggested_divisor': max(stats['diverge_queue_length']['p99'], 1.0),
            'reason': f'99分位数: {stats["diverge_queue_length"]["p99"]:.1f}'
        }

        # 信号变量
        suggestions['time_to_switch'] = {
            'current_divisor': 100.0,
            'suggested_divisor': max(stats['time_to_switch']['p99'], 1.0),
            'reason': f'99分位数: {stats["time_to_switch"]["p99"]:.1f} s'
        }

        # 时间
        suggestions['timestamp'] = {
            'current_divisor': 3600.0,
            'suggested_divisor': max(stats['timestamp']['max'], 1.0),
            'reason': f'最大值: {stats["timestamp"]["max"]:.1f} s'
        }

        return suggestions

    def print_report(self, stats, suggestions):
        """打印分析报告"""
        print("\n" + "="*80)
        print("状态变量范围分析报告")
        print("="*80)

        # 分类打印
        categories = {
            '主路变量': ['main_vehicles_count', 'main_speed', 'main_density',
                        'main_queue_length', 'main_flow'],
            '匝道变量': ['ramp_vehicles_count', 'ramp_speed', 'ramp_queue_length',
                        'ramp_waiting_time', 'ramp_flow'],
            '分流变量': ['diverge_vehicles_count', 'diverge_queue_length'],
            '信号变量': ['current_phase', 'time_to_switch'],
            '风险变量': ['conflict_risk', 'gap_acceptance'],
            '时间变量': ['timestamp'],
        }

        for category, variables in categories.items():
            print(f"\n## {category}")
            print("-"*80)

            for var in variables:
                if var not in stats:
                    continue

                s = stats[var]
                sug = suggestions.get(var, {})

                print(f"\n【{s['name']}】")
                print(f"  样本数: {s['count']:,}")
                print(f"  范围: [{s['min']:.2f}, {s['max']:.2f}]")
                print(f"  均值±标准差: {s['mean']:.2f} ± {s['std']:.2f}")
                print(f"  中位数: {s['median']:.2f}")
                print(f"  分位数: p95={s['p95']:.2f}, p99={s['p99']:.2f}, p99.9={s['p999']:.2f}")

                if sug:
                    print(f"  归一化参数:")
                    print(f"    当前: 除以 {sug['current_divisor']:.1f}")
                    print(f"    建议: 除以 {sug['suggested_divisor']:.1f}")
                    print(f"    理由: {sug['reason']}")

        print("\n" + "="*80)
        print("归一化参数建议代码")
        print("="*80)
        print("\n# 替换 junction_agent.py 中的归一化参数:")
        print("features = [")

        code_mapping = {
            'main_vehicles_count': ("len(state.main_vehicles)", 'main_vehicles_count'),
            'main_speed': ("state.main_speed", 'main_speed'),
            'main_density': ("state.main_density", 'main_density'),
            'main_queue_length': ("state.main_queue_length", 'main_queue_length'),
            'main_flow': ("state.main_flow", 'main_flow'),
            'ramp_vehicles_count': ("len(state.ramp_vehicles)", 'ramp_vehicles_count'),
            'ramp_speed': ("state.ramp_speed", 'ramp_speed'),
            'ramp_queue_length': ("state.ramp_queue_length", 'ramp_queue_length'),
            'ramp_waiting_time': ("state.ramp_waiting_time", 'ramp_waiting_time'),
            'ramp_flow': ("state.ramp_flow", 'ramp_flow'),
            'diverge_vehicles_count': ("len(state.diverge_vehicles)", 'diverge_vehicles_count'),
            'diverge_queue_length': ("state.diverge_queue_length", 'diverge_queue_length'),
            'time_to_switch': ("state.time_to_switch", 'time_to_switch'),
            'timestamp': ("state.timestamp", 'timestamp'),
        }

        for var, (expr, key) in code_mapping.items():
            if key in suggestions:
                divisor = suggestions[key]['suggested_divisor']
                print(f"    {expr} / {divisor:.1f},  # {suggestions[key]['reason']}")

        print("]")
        print("="*80 + "\n")

    def save_results(self, stats, suggestions, output_path="state_normalization_stats.pkl"):
        """保存分析结果"""
        results = {
            'statistics': stats,
            'suggestions': suggestions,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n结果已保存到: {output_path}")


def run_analysis(num_steps=3600, flow_rate=None):
    """
    运行状态范围分析

    Args:
        num_steps: 分析步数（默认3600，即一个完整episode）
        flow_rate: 流量率（None表示使用默认值）
    """
    print("="*80)
    print("状态变量范围分析")
    print("="*80)
    print(f"分析步数: {num_steps}")
    print(f"流量率: {flow_rate if flow_rate else '默认'}")
    print("="*80)

    # 创建分析器
    analyzer = StateRangeAnalyzer()

    # 创建SUMO框架
    print("\n[1/3] 初始化SUMO环境...")
    sumo_cfg = r".\sumo\sumo.sumocfg"
    framework = SUMOCompetitionFramework(sumo_cfg)

    # 设置流量率
    if flow_rate:
        framework.flow_rate = flow_rate

    # 初始化环境
    if not framework.initialize_environment(use_gui=False, max_steps=num_steps):
        print("[ERROR] 环境初始化失败")
        return None, None

    # 手动创建agents（用于收集状态数据）
    print("[INFO] 创建路口agents...")
    from junction_control_zones import JUNCTION_CONFIGS

    import traci
    for junc_id, junc_config in JUNCTION_CONFIGS.items():
        agent = JunctionAgent(junc_config)
        agent.setup_subscriptions()
        framework.agents[junc_id] = agent

    print(f"[INFO] 已创建 {len(framework.agents)} 个agents")

    # 运行仿真并收集数据
    print(f"[2/3] 运行仿真并收集数据 (0-{num_steps}步)...")

    try:
        # 订阅的车辆集合
        subscribed_vehicles = set()

        for step in range(num_steps):
            # 执行仿真步
            traci.simulationStep()

            # 订阅新出现的车辆
            current_vehicles = set(traci.vehicle.getIDList())
            new_vehicles = current_vehicles - subscribed_vehicles
            if new_vehicles:
                for agent in framework.agents.values():
                    agent.sub_manager.setup_vehicle_subscription(list(new_vehicles))
                subscribed_vehicles.update(new_vehicles)

            # 收集每个路口的状态数据（直接使用TraCI查询）
            for junc_id, agent in framework.agents.items():
                analyzer.collect_state_data_direct(agent)

            # 进度报告
            if step % 500 == 0 and step > 0:
                print(f"  进度: {step}/{num_steps} ({step/num_steps*100:.1f}%)")

            # 检查仿真是否结束
            if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                print(f"\n仿真自然结束于步骤 {step}")
                break

    except KeyboardInterrupt:
        print("\n用户中断，正在保存已收集的数据...")

    except Exception as e:
        print(f"\n[ERROR] 仿真过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        traci.close()

    # 计算统计信息
    print(f"[3/3] 计算统计信息...")
    stats = analyzer.compute_statistics()
    suggestions = analyzer.suggest_normalization(stats)

    # 打印报告
    analyzer.print_report(stats, suggestions)

    # 保存结果
    analyzer.save_results(stats, suggestions)

    return stats, suggestions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='状态变量范围分析')
    parser.add_argument('--steps', type=int, default=3600,
                        help='分析步数 (默认: 3600)')
    parser.add_argument('--flow-rate', type=float, default=None,
                        help='流量率 (默认: 使用配置文件默认值)')
    parser.add_argument('--output', type=str, default='state_normalization_stats.pkl',
                        help='输出文件路径')

    args = parser.parse_args()

    # 运行分析
    stats, suggestions = run_analysis(
        num_steps=args.steps,
        flow_rate=args.flow_rate
    )
