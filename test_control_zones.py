"""
测试控制区域划分
验证每个路口的控制区域不重叠
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junction_control_zones import (
    CONTROL_ZONES, 
    JUNCTION_CONFIGS,
    VehicleRegistry,
    ControlZone
)


def test_control_zones():
    """测试控制区域划分"""
    print("=" * 70)
    print("测试控制区域划分")
    print("=" * 70)
    
    # 1. 检查控制区域是否重叠
    print("\n1. 检查控制区域重叠")
    print("-" * 70)
    
    all_edges = {}
    overlap_found = False
    
    for junc_id, zone in CONTROL_ZONES.items():
        zone_edges = zone.main_upstream_edges + zone.ramp_upstream_edges + zone.diverge_edges
        
        for edge in zone_edges:
            if edge in all_edges:
                print(f"  ✗ 重叠发现: {edge} 被 {all_edges[edge]} 和 {junc_id} 同时控制")
                overlap_found = True
            else:
                all_edges[edge] = junc_id
    
    if not overlap_found:
        print("  ✓ 没有发现控制区域重叠")
    
    # 2. 检查控制链完整性
    print("\n2. 检查控制链完整性")
    print("-" * 70)
    
    # 主路控制链
    main_chain = ['E2', 'E9', 'E10', 'E12']
    print(f"  主路控制链: {' → '.join(main_chain)}")
    
    for i, edge in enumerate(main_chain):
        # 找到控制这个边的路口
        controlling_junction = None
        for junc_id, zone in CONTROL_ZONES.items():
            if edge in zone.main_upstream_edges:
                controlling_junction = junc_id
                break
        
        if controlling_junction:
            print(f"    {edge} 由 {controlling_junction} 控制")
        else:
            print(f"    ✗ {edge} 没有被任何路口控制")
    
    # 3. 检查匝道控制
    print("\n3. 检查匝道控制")
    print("-" * 70)
    
    ramp_edges = {
        'J5': 'E23',
        'J14': 'E15',
        'J15': 'E17',
        'J17': 'E19'
    }
    
    for junc_id, expected_edge in ramp_edges.items():
        zone = CONTROL_ZONES[junc_id]
        if expected_edge in zone.ramp_upstream_edges:
            print(f"  ✓ {junc_id} 正确控制匝道 {expected_edge}")
        else:
            print(f"  ✗ {junc_id} 未控制匝道 {expected_edge}")
    
    # 4. 检查转出控制
    print("\n4. 检查转出控制")
    print("-" * 70)
    
    diverge_edges = {
        'J15': ['E16'],
        'J17': ['E18', 'E20']
    }
    
    for junc_id, expected_edges in diverge_edges.items():
        zone = CONTROL_ZONES[junc_id]
        for edge in expected_edges:
            if edge in zone.diverge_edges:
                print(f"  ✓ {junc_id} 正确控制转出 {edge}")
            else:
                print(f"  ✗ {junc_id} 未控制转出 {edge}")
    
    # 5. 检查排除区域
    print("\n5. 检查排除区域")
    print("-" * 70)
    
    for junc_id, zone in CONTROL_ZONES.items():
        if zone.excluded_edges:
            print(f"  {junc_id} 排除区域: {zone.excluded_edges}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    return not overlap_found


def test_vehicle_registry():
    """测试车辆注册表"""
    print("\n" + "=" * 70)
    print("测试车辆注册表")
    print("=" * 70)
    
    registry = VehicleRegistry()
    
    # 模拟车辆数据
    all_vehicles = {
        'veh_001': {
            'id': 'veh_001',
            'edge': 'E2',
            'lane_position': 450,  # 距离路口50米
            'edge_length': 500,
            'is_cv': True
        },
        'veh_002': {
            'id': 'veh_002',
            'edge': 'E9',
            'lane_position': 400,  # 距离路口100米
            'edge_length': 500,
            'is_cv': True
        },
        'veh_003': {
            'id': 'veh_003',
            'edge': 'E23',
            'lane_position': 250,  # 距离路口50米
            'edge_length': 300,
            'is_cv': True
        },
        'veh_004': {
            'id': 'veh_004',
            'edge': 'E10',
            'lane_position': 350,  # 距离路口150米
            'edge_length': 500,
            'is_cv': True
        },
        'veh_005': {
            'id': 'veh_005',
            'edge': 'E1',  # 不在任何控制区域
            'lane_position': 100,
            'edge_length': 500,
            'is_cv': True
        }
    }
    
    # 更新注册表
    registry.update(all_vehicles)
    
    # 检查分配结果
    print("\n车辆控制权分配:")
    print("-" * 70)
    
    expected_assignments = {
        'veh_001': 'J5',   # E2, 距离50m < 200m
        'veh_002': 'J14',  # E9, 距离100m < 200m
        'veh_003': 'J5',   # E23, 距离50m < 150m
        'veh_004': 'J15',  # E10, 距离150m < 200m
        'veh_005': None    # E1, 不在控制区域
    }
    
    all_correct = True
    for veh_id, expected_junction in expected_assignments.items():
        actual_junction = registry.get_controlling_junction(veh_id)
        
        if actual_junction == expected_junction:
            print(f"  ✓ {veh_id}: 正确分配给 {actual_junction}")
        else:
            print(f"  ✗ {veh_id}: 期望 {expected_junction}, 实际 {actual_junction}")
            all_correct = False
    
    # 检查每个路口控制的车辆
    print("\n每个路口控制的车辆:")
    print("-" * 70)
    
    for junc_id in CONTROL_ZONES.keys():
        vehicles = registry.get_controlled_vehicles(junc_id)
        print(f"  {junc_id}: {vehicles if vehicles else '无'}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    return all_correct


def test_control_zone_ranges():
    """测试控制范围设置"""
    print("\n" + "=" * 70)
    print("测试控制范围设置")
    print("=" * 70)
    
    print("\n各路口控制范围:")
    print("-" * 70)
    
    for junc_id, zone in CONTROL_ZONES.items():
        print(f"\n{junc_id}:")
        print(f"  主路上游范围: {zone.main_upstream_range}m")
        print(f"  匝道上游范围: {zone.ramp_upstream_range}m")
        if zone.diverge_edges:
            print(f"  转出引导范围: {zone.diverge_range}m")
    
    # 检查范围是否合理
    print("\n范围合理性检查:")
    print("-" * 70)
    
    reasonable = True
    for junc_id, zone in CONTROL_ZONES.items():
        if zone.main_upstream_range > 300:
            print(f"  ⚠ {junc_id} 主路控制范围过大: {zone.main_upstream_range}m")
            reasonable = False
        
        if zone.ramp_upstream_range > 200:
            print(f"  ⚠ {junc_id} 匝道控制范围过大: {zone.ramp_upstream_range}m")
            reasonable = False
    
    if reasonable:
        print("  ✓ 所有控制范围设置合理")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    return reasonable


def test_control_transfer():
    """测试控制权转移"""
    print("\n" + "=" * 70)
    print("测试控制权转移")
    print("=" * 70)
    
    registry = VehicleRegistry()
    
    # 初始状态：车辆在J5控制区域
    vehicles_t1 = {
        'veh_001': {
            'id': 'veh_001',
            'edge': 'E2',
            'lane_position': 450,  # 距离J5路口50m
            'edge_length': 500,
            'is_cv': True
        }
    }
    
    registry.update(vehicles_t1)
    junc_t1 = registry.get_controlling_junction('veh_001')
    print(f"\nT1: 车辆在E2，由 {junc_t1} 控制")
    
    # 车辆移动到J14控制区域
    vehicles_t2 = {
        'veh_001': {
            'id': 'veh_001',
            'edge': 'E9',  # 进入J14控制区域
            'lane_position': 400,  # 距离J14路口100m
            'edge_length': 500,
            'is_cv': True
        }
    }
    
    registry.update(vehicles_t2)
    junc_t2 = registry.get_controlling_junction('veh_001')
    print(f"T2: 车辆移动到E9，由 {junc_t2} 控制")
    
    # 验证转移
    if junc_t1 == 'J5' and junc_t2 == 'J14':
        print("\n✓ 控制权转移正确")
        return True
    else:
        print(f"\n✗ 控制权转移错误: J5 → J14 期望，实际 {junc_t1} → {junc_t2}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("控制区域划分测试套件")
    print("=" * 70)
    
    results = {}
    
    # 运行所有测试
    try:
        results['zones'] = test_control_zones()
    except Exception as e:
        print(f"✗ 控制区域测试失败: {e}")
        results['zones'] = False
    
    try:
        results['registry'] = test_vehicle_registry()
    except Exception as e:
        print(f"✗ 车辆注册表测试失败: {e}")
        results['registry'] = False
    
    try:
        results['ranges'] = test_control_zone_ranges()
    except Exception as e:
        print(f"✗ 控制范围测试失败: {e}")
        results['ranges'] = False
    
    try:
        results['transfer'] = test_control_transfer()
    except Exception as e:
        print(f"✗ 控制权转移测试失败: {e}")
        results['transfer'] = False
    
    # 总结
    print("\n" + "=" * 70)
    print("测试结果总结")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！控制区域划分正确。")
        print("\n关键特性:")
        print("  1. 控制区域不重叠")
        print("  2. 车辆注册表正确分配控制权")
        print("  3. 控制范围设置合理")
        print("  4. 控制权转移机制正确")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    main()
