"""
分析SUMO仿真中的车辆类型
获取所有车型的maxspeed并生成归一化配置
"""

import os
import sys
import xml.etree.ElementTree as ET

try:
    import traci
except ImportError:
    print("请安装traci: pip install traci")
    sys.exit(1)


def analyze_vehicle_types_from_net(net_file):
    """从net.xml中分析车辆类型"""
    print("=" * 70)
    print("从路网文件分析车辆类型")
    print("=" * 70)

    tree = ET.parse(net_file)
    root = tree.getroot()

    vehicle_types = {}

    # 查找所有车辆类型定义
    for vtype in root.findall('.//vType'):
        vtype_id = vtype.get('id')
        max_speed = float(vtype.get('maxSpeed', 0.0))

        if vtype_id and max_speed > 0:
            vehicle_types[vtype_id] = {
                'max_speed': max_speed,
                'length': float(vtype.get('length', 5.0)),
                'width': float(vtype.get('width', 2.0)),
                'accel': float(vtype.get('accel', 3.0)),
                'decel': float(vtype.get('decel', 4.0)),
            }

            print(f"  类型: {vtype_id:15s} | 最大速度: {max_speed:5.2f} m/s ({max_speed*3.6:5.1f} km/h)")

    return vehicle_types


def analyze_vehicle_types_from_sumo(sumo_cfg_path):
    """通过启动SUMO获取车辆类型信息"""
    print("\n" + "=" * 70)
    print("通过SUMO运行时获取车辆类型")
    print("=" * 70)

    # 启动SUMO
    sumo_cmd = [
        "sumo",
        "-c", sumo_cfg_path,
        "--no-warnings", "true",
        "--seed", "42"
    ]

    traci.start(sumo_cmd)

    vehicle_types = {}

    try:
        # 预运行几步让车辆出现
        for _ in range(10):
            traci.simulationStep()

        # 获取所有车辆类型
        vtype_list = traci.vehicletype.getIDList()

        print(f"  发现 {len(vtype_list)} 种车辆类型:")

        for vtype_id in vtype_list:
            try:
                max_speed = traci.vehicletype.getMaxSpeed(vtype_id)
                length = traci.vehicletype.getLength(vtype_id)
                width = traci.vehicletype.getWidth(vtype_id)

                vehicle_types[vtype_id] = {
                    'max_speed': max_speed,
                    'length': length,
                    'width': width,
                }

                print(f"  类型: {vtype_id:15s} | 最大速度: {max_speed:5.2f} m/s ({max_speed*3.6:5.1f} km/h)")

            except Exception as e:
                print(f"  ⚠️  获取类型 {vtype_id} 失败: {e}")

    finally:
        traci.close()

    return vehicle_types


def generate_normalization_config(vehicle_types, output_file="vehicle_type_config.py"):
    """生成归一化配置文件"""
    print("\n" + "=" * 70)
    print("生成归一化配置")
    print("=" * 70)

    # 计算全局最大速度
    global_max_speed = max(vt['max_speed'] for vt in vehicle_types.values())

    config_lines = [
        '"""',
        '车辆类型归一化配置',
        '自动生成，请勿手动编辑',
        '*/',
        '',
        'import torch',
        '',
        '# 车辆类型及其最大速度',
        'VEHICLE_TYPE_MAXSPEED = {',
    ]

    for vtype_id, vtype_data in sorted(vehicle_types.items()):
        max_speed = vtype_data['max_speed']
        config_lines.append(f"    '{vtype_id}': {max_speed:.4f},  # {max_speed*3.6:.1f} km/h")

    config_lines.extend([
        '}',
        '',
        '# 全局最大速度（用于归一化）',
        f'GLOBAL_MAX_SPEED = {global_max_speed:.4f}  # {global_max_speed*3.6:.1f} km/h',
        '',
        '# 速度归一化函数',
        'def normalize_speed(speed, max_speed=None):',
        '    """',
        '    归一化速度到 [0, 1]',
        '    ',
        '    参数:',
        '        speed: 速度值 (m/s)',
        '        max_speed: 该车辆类型的最大速度，如果为None则使用全局最大速度',
        '    ',
        '    返回:',
        '        归一化后的速度 [0, 1]',
        '    """',
        '    if max_speed is None:',
        '        max_speed = GLOBAL_MAX_SPEED',
        '    ',
        '    return torch.clamp(speed / max_speed, 0.0, 1.0)',
        '',
        '',
        'def denormalize_speed(normalized_speed, max_speed=None):',
        '    """',
        '    反归一化速度',
        '    ',
        '    参数:',
        '        normalized_speed: 归一化速度 [0, 1]',
        '        max_speed: 该车辆类型的最大速度，如果为None则使用全局最大速度',
        '    ',
        '    返回:',
        '        实际速度 (m/s)',
        '    """',
        '    if max_speed is None:',
        '        max_speed = GLOBAL_MAX_SPEED',
        '    ',
        '    return torch.clamp(normalized_speed * max_speed, 0.0, max_speed)',
        '',
        '',
        'def get_vehicle_max_speed(vehicle_type):',
        '    """',
        '    获取车辆类型的最大速度',
        '    ',
        '    参数:',
        '        vehicle_type: 车辆类型ID',
        '    ',
        '    返回:',
        '        该类型的最大速度 (m/s)，如果类型不存在则返回全局最大速度',
        '    """',
        '    return VEHICLE_TYPE_MAXSPEED.get(vehicle_type, GLOBAL_MAX_SPEED)',
        '',
        '',
        '# 批量归一化函数',
        'def normalize_batch_speeds(speeds, max_speeds):',
        '    """',
        '    批量归一化速度',
        '    ',
        '    参数:',
        '        speeds: 速度张量 [N]',
        '        max_speeds: 最大速度张量 [N]',
        '    ',
        '    返回:',
        '        归一化速度 [N]',
        '    """',
        '    max_speeds = torch.tensor(max_speeds, device=speeds.device, dtype=speeds.dtype)',
        '    return torch.clamp(speeds / max_speeds, 0.0, 1.0)',
        '',
        '',
        'def denormalize_batch_speeds(normalized_speeds, max_speeds):',
        '    """',
        '    批量反归一化速度',
        '    ',
        '    参数:',
        '        normalized_speeds: 归一化速度张量 [N]',
        '        max_speeds: 最大速度列表或张量 [N]',
        '    ',
        '    返回:',
        '        实际速度张量 [N]',
        '    """',
        '    max_speeds = torch.tensor(max_speeds, device=normalized_speeds.device, dtype=normalized_speeds.dtype)',
        '    return torch.clamp(normalized_speeds * max_speeds, 0.0, max_speeds)',
        '',
    ])

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(config_lines))

    print(f"✓ 配置文件已保存: {output_file}")
    print(f"\n配置摘要:")
    print(f"  - 车辆类型数: {len(vehicle_types)}")
    print(f"  - 全局最大速度: {global_max_speed:.2f} m/s ({global_max_speed*3.6:.1f} km/h)")

    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description='分析SUMO车辆类型并生成归一化配置')
    parser.add_argument('--net-file', type=str, help='路网文件路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg', help='SUMO配置文件路径')
    parser.add_argument('--output', type=str, default='vehicle_type_config.py', help='输出配置文件路径')
    parser.add_argument('--method', type=str, choices=['net', 'sumo', 'both'], default='both',
                       help='分析方法: net=从路网文件, sumo=从运行时, both=两者结合')

    args = parser.parse_args()

    vehicle_types = {}

    # 方法1: 从net.xml分析
    if args.method in ['net', 'both'] and args.net_file:
        if os.path.exists(args.net_file):
            vehicle_types.update(analyze_vehicle_types_from_net(args.net_file))
        else:
            print(f"⚠️  路网文件不存在: {args.net_file}")

    # 方法2: 从SUMO运行时获取
    if args.method in ['sumo', 'both']:
        if os.path.exists(args.sumo_cfg):
            sumo_types = analyze_vehicle_types_from_sumo(args.sumo_cfg)
            vehicle_types.update(sumo_types)
        else:
            print(f"⚠️  SUMO配置文件不存在: {args.sumo_cfg}")

    if not vehicle_types:
        print("\n❌ 未能获取到车辆类型信息")
        return

    # 生成配置文件
    generate_normalization_config(vehicle_types, args.output)

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print("\n下一步:")
    print(f"1. 检查生成的配置文件: {args.output}")
    print("2. 在代码中使用归一化函数:")
    print("   from vehicle_type_config import normalize_speed, denormalize_speed")
    print("3. 替换现有的速度归一化操作")


if __name__ == "__main__":
    main()
