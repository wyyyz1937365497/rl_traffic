"""
更新代码中的速度归一化操作
使用基于车辆类型的动态归一化
"""

import os
import re


def find_speed_normalizations(file_path):
    """查找文件中的速度归一化操作"""
    patterns = [
        r'/\s*20\.0\s*',  # / 20.0
        r'/\s*20\.\s*',   # / 20.
        r'speed\s*/\s*20\.?\d*',  # speed / 20
        r'\.getSpeed\([^)]*\)\s*/\s*20\.?\d*',  # getSpeed() / 20
        r'traci_wrapper\.vehicle\.getSpeed\([^)]*\)\s*/\s*20\.?\d*',
        r'traci\.vehicle\.getSpeed\([^)]*\)\s*/\s*20\.?\d*',
    ]

    matches = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        for pattern in patterns:
            if re.search(pattern, line):
                matches.append({
                    'line_num': line_num,
                    'line': line.strip(),
                    'pattern': pattern
                })
                break

    return matches


def update_rl_train():
    """更新rl_train.py中的速度归一化"""
    file_path = 'rl_train.py'

    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return

    print(f"\n正在处理: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 添加import
    if 'from vehicle_type_config import' not in content:
        import_section = re.search(r'(import.*?\n)\n', content)
        if import_section:
            insert_pos = import_section.end()
            content = content[:insert_pos] + 'from vehicle_type_config import normalize_speed, get_vehicle_max_speed\n' + content[insert_pos:]

    # 更新速度归一化
    # 模式1: traci_wrapper.vehicle.getSpeed(veh_id) / 20.0
    content = re.sub(
        r'traci_wrapper\.vehicle\.getSpeed\([^)]+\)\s*/\s*20\.0',
        'normalize_speed(traci_wrapper.vehicle.getSpeed(veh_id))',
        content
    )

    # 模式2: / 20.0 在特定上下文中
    content = re.sub(
        r'(traci_wrapper\.vehicle\.getSpeed\([^)]+\))\s*/\s*(20\.0|20)',
        r'normalize_speed(\1)',
        content
    )

    if content != original_content:
        # 备份原文件
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  ✓ 已备份原文件: {backup_path}")

        # 写入更新后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已更新: {file_path}")
    else:
        print(f"  - 无需更新")


def update_junction_agent():
    """更新junction_agent.py中的速度归一化"""
    file_path = 'junction_agent.py'

    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return

    print(f"\n正在处理: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 添加import
    if 'from vehicle_type_config import' not in content:
        import_section = re.search(r'(import.*?\n)\n', content)
        if import_section:
            insert_pos = import_section.end()
            content = content[:insert_pos] + 'from vehicle_type_config import normalize_speed, get_vehicle_max_speed\n' + content[insert_pos:]

    # 更新速度归一化
    content = re.sub(
        r'traci\.vehicle\.getSpeed\([^)]+\)\s*/\s*20\.0',
        'normalize_speed(traci.vehicle.getSpeed(veh_id))',
        content
    )

    if content != original_content:
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  ✓ 已备份原文件: {backup_path}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已更新: {file_path}")
    else:
        print(f"  - 无需更新")


def update_sumo_main():
    """更新sumo/main.py中的速度归一化"""
    file_path = 'sumo/main.py'

    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return

    print(f"\n正在处理: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 添加import
    if 'from vehicle_type_config import' not in content:
        # 找到sys.path.insert之后
        pattern = r'(sys\.path\.insert\(.*?\n)'
        match = re.search(pattern, content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + 'from vehicle_type_config import normalize_speed, get_vehicle_max_speed\n' + content[insert_pos:]

    # 更新速度归一化
    content = re.sub(
        r'traci\.vehicle\.getSpeed\([^)]+\)\s*/\s*20\.0',
        'normalize_speed(traci.vehicle.getSpeed(veh_id))',
        content
    )

    if content != original_content:
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  ✓ 已备份原文件: {backup_path}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已更新: {file_path}")
    else:
        print(f"  - 无需更新")


def create_updated_normalize_functions():
    """创建新的归一化函数示例"""
    content = '''"""
车辆特征归一化工具函数
使用基于车辆类型的动态归一化
"""

from vehicle_type_config import normalize_speed, get_vehicle_max_speed
import torch
import numpy as np


def get_vehicle_features_normalized(vehicle_ids, traci_conn, device):
    """
    获取归一化的车辆特征

    参数:
        vehicle_ids: 车辆ID列表
        traci_conn: TraCI连接
        device: torch设备

    返回:
        归一化特征张量 [N, 8]
    """
    if not vehicle_ids:
        return None

    features = []
    for veh_id in vehicle_ids[:10]:
        try:
            # 获取车辆类型
            vehicle_type = traci_conn.vehicle.getTypeID(veh_id)
            max_speed = get_vehicle_max_speed(vehicle_type)

            # 速度归一化（使用车辆类型的maxspeed）
            speed = normalize_speed(
                traci_conn.vehicle.getSpeed(veh_id),
                max_speed=max_speed
            )

            # 位置归一化 (0-500m)
            position = traci_conn.vehicle.getLanePosition(veh_id) / 500.0

            # 车道索引归一化 (0-3)
            lane_index = traci_conn.vehicle.getLaneIndex(veh_id) / 3.0

            # 等待时间归一化 (0-60s)
            waiting_time = traci_conn.vehicle.getWaitingTime(veh_id) / 60.0

            # 加速度归一化 (-5 to 5 m/s²)
            acceleration = traci_conn.vehicle.getAcceleration(veh_id) / 5.0

            # CV标识
            is_cv = 1.0 if vehicle_type == 'CV' else 0.0

            # 路线索引归一化 (0-10)
            route_index = traci_conn.vehicle.getRouteIndex(veh_id) / 10.0

            features.append([
                speed.item() if torch.is_tensor(speed) else speed,
                position,
                lane_index,
                waiting_time,
                acceleration,
                is_cv,
                route_index,
                0.0  # 占位符
            ])
        except Exception as e:
            # 车辆可能已离开网络，使用零填充
            features.append([0.0] * 8)

    # 补齐到10个车辆
    while len(features) < 10:
        features.append([0.0] * 8)

    return torch.tensor(features, dtype=torch.float32, device=device)


def get_state_vector_normalized(agent_state):
    """
    获取归一化的状态向量

    参数:
        agent_state: 智能体状态字典

    返回:
        归一化状态向量 (17维)
    """
    import torch
    from vehicle_type_config import GLOBAL_MAX_SPEED

    state_vec = torch.tensor([
        # 队列长度归一化
        agent_state.get('main_queue_length', 0) / 50.0,
        agent_state.get('ramp_queue_length', 0) / 50.0,

        # 速度归一化（使用全局最大速度）
        agent_state.get('main_speed', 0) / GLOBAL_MAX_SPEED,
        agent_state.get('ramp_speed', 0) / GLOBAL_MAX_SPEED,

        # 密度归一化
        agent_state.get('main_density', 0) / 0.5,
        agent_state.get('ramp_density', 0) / 0.5,

        # 等待时间归一化
        agent_state.get('ramp_waiting_time', 0) / 60.0,

        # 间隙归一化
        agent_state.get('gap_size', 0) / 10.0,

        # 速度差归一化
        agent_state.get('gap_speed_diff', 0) / GLOBAL_MAX_SPEED,

        # CV标识
        float(agent_state.get('has_cv', False)),

        # 冲突风险
        float(agent_state.get('conflict_risk', 0)),

        # 停车计数归一化
        agent_state.get('main_stop_count', 0) / 10.0,
        agent_state.get('ramp_stop_count', 0) / 10.0,

        # 通过量归一化
        agent_state.get('throughput', 0) / 100.0,

        # 相位标识
        0.0,  # phase_main
        0.0,  # phase_ramp
        0.0   # time_step
    ], dtype=torch.float32)

    return state_vec


# 使用示例
if __name__ == "__main__":
    print("归一化工具函数已加载")
    print("\\n使用方法:")
    print("  from normalization_utils import get_vehicle_features_normalized")
    print("  features = get_vehicle_features_normalized(vehicle_ids, traci, device)")
'''

    with open('normalization_utils.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ 已创建归一化工具文件: normalization_utils.py")


def main():
    print("=" * 70)
    print("更新速度归一化操作")
    print("=" * 70)

    # 检查vehicle_type_config.py是否存在
    if not os.path.exists('vehicle_type_config.py'):
        print("\n⚠️  警告: vehicle_type_config.py 不存在")
        print("请先运行: python analyze_vehicle_types.py --sumo-cfg sumo/sumo.sumocfg")
        print("\n是否继续使用默认归一化？(y/n): ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("已取消")
            return

    # 创建归一化工具文件
    create_updated_normalize_functions()

    # 更新各个文件
    files_to_update = [
        ('rl_train.py', update_rl_train),
        ('junction_agent.py', update_junction_agent),
        ('sumo/main.py', update_sumo_main),
    ]

    print("\n开始更新文件...")

    for file_name, update_func in files_to_update:
        try:
            update_func()
        except Exception as e:
            print(f"  ❌ 更新失败 {file_name}: {e}")

    print("\n" + "=" * 70)
    print("更新完成！")
    print("=" * 70)

    print("\n下一步:")
    print("1. 检查.backup文件确保更新正确")
    print("2. 如果有问题，可以恢复备份:")
    print("   mv rl_train.py.backup rl_train.py")
    print("3. 测试更新后的代码:")
    print("   python rl_train.py --sumo-cfg sumo/sumo.sumocfg --total-timesteps 1000")
    print("4. 确认无误后删除备份文件:")
    print("   rm *.backup")


if __name__ == "__main__":
    main()
