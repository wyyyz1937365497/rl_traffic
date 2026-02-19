"""
更新代码中的速度归一化操作
使用车辆类型特定的maxspeed
"""

import os
import re


def update_file(file_path, replacements):
    """更新文件中的归一化操作"""
    if not os.path.exists(file_path):
        print(f"  ⚠️  文件不存在: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 检查是否已经导入
    has_import = 'from vehicle_type_config import' in content

    # 添加import（如果还没有）
    if not has_import:
        # 找到第一个import语句后在后面添加
        import_match = re.search(r'(import .+?\n)', content)
        if import_match:
            insert_pos = import_match.end()
            import_line = 'from vehicle_type_config import normalize_speed, get_vehicle_max_speed\n'
            content = content[:insert_pos] + import_line + content[insert_pos:]

    # 应用替换规则
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original:
        # 备份
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original)

        # 写入更新后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✓ 已更新: {file_path}")
        print(f"    备份: {backup_path}")
        return True
    else:
        print(f"  - 无需更新: {file_path}")
        return False


def main():
    print("=" * 70)
    print("更新速度归一化操作")
    print("=" * 70)

    # 检查配置文件
    if not os.path.exists('vehicle_type_config.py'):
        print("\n❌ vehicle_type_config.py 不存在")
        print("请先运行: python analyze_vehicle_types.py --sumo-cfg sumo/sumo.sumocfg")
        return

    print("\n✓ 配置文件已就绪: vehicle_type_config.py")

    # 定义各文件的替换规则
    updates = {
        'rl_train.py': [
            # 车辆特征提取中的速度归一化
            (r'traci_wrapper\.vehicle\.getSpeed\((\w+)\)\s*/\s*(?:20\.0|20)',
             r'normalize_speed(traci_wrapper.vehicle.getSpeed(\1))'),
        ],
        'junction_agent.py': [
            # 观察函数中的速度归一化
            (r'traci\.vehicle\.getSpeed\((\w+)\)\s*/\s*(?:20\.0|20)',
             r'normalize_speed(traci.vehicle.getSpeed(\1))'),
        ],
        'sumo/main.py': [
            # 推理时的速度归一化
            (r'traci\.vehicle\.getSpeed\((\w+)\)\s*/\s*(?:20\.0|20)',
             r'normalize_speed(traci.vehicle.getSpeed(\1))'),
        ],
    }

    print("\n开始更新文件...")
    updated_count = 0

    for file_path, replacements in updates.items():
        print(f"\n处理: {file_path}")
        if update_file(file_path, replacements):
            updated_count += 1

    print("\n" + "=" * 70)
    print(f"更新完成！已更新 {updated_count} 个文件")
    print("=" * 70)

    print("\n验证更新:")
    print("  grep -n 'normalize_speed' rl_train.py junction_agent.py sumo/main.py")

    print("\n如果需要回退:")
    print("  mv rl_train.py.backup rl_train.py")
    print("  mv junction_agent.py.backup junction_agent.py")
    print("  mv sumo/main.py.backup sumo/main.py")

    print("\n确认无误后删除备份:")
    print("  rm *.backup")


if __name__ == "__main__":
    main()
