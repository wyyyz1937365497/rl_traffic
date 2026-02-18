"""
测试订阅模式和更新的网络结构
"""

import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from junction_agent import JUNCTION_CONFIGS, JunctionType
from junction_network_updated import (
    create_junction_model, 
    NetworkConfig, 
    TypeAPolicyNetwork, 
    TypeBPolicyNetwork,
    TrafficLightEncoder
)


def test_traffic_light_encoder():
    """测试信号灯编码器"""
    print("=" * 60)
    print("测试信号灯编码器")
    print("=" * 60)
    
    encoder = TrafficLightEncoder(input_dim=5, hidden_dim=32, output_dim=16)
    
    # 创建假数据
    batch_size = 4
    tl_features = torch.tensor([
        [0, 50.0, 1.0, 0.0, 0.0],  # 相位0，距离切换50秒，主路绿灯
        [1, 30.0, 0.0, 1.0, 0.0],  # 相位1，距离切换30秒，匝道绿灯
        [0, 10.0, 1.0, 0.0, 1.0],  # 相位0，距离切换10秒，主路和转出绿灯
        [1, 5.0, 0.0, 1.0, 1.0],   # 相位1，距离切换5秒，匝道和转出绿灯
    ], dtype=torch.float32)
    
    output = encoder(tl_features)
    
    print(f"\n输入形状: {tl_features.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n输入示例:")
    for i, feat in enumerate(tl_features):
        print(f"  样本{i+1}: 相位={int(feat[0])}, 切换时间={feat[1]:.1f}s, "
              f"主路={'绿' if feat[2] else '红'}, "
              f"匝道={'绿' if feat[3] else '红'}, "
              f"转出={'绿' if feat[4] else '红'}")
    
    print("\n✓ 信号灯编码器测试通过")
    return True


def test_network_with_tl():
    """测试包含信号灯特征的网络"""
    print("\n" + "=" * 60)
    print("测试包含信号灯特征的网络")
    print("=" * 60)
    
    config = NetworkConfig()
    
    # 测试类型A网络
    print("\n测试类型A网络...")
    net_a = TypeAPolicyNetwork(config)
    
    batch_size = 2
    state = torch.randn(batch_size, 22)  # 22维状态
    
    # 设置信号灯特征（最后5维）
    state[:, 17] = torch.tensor([0, 1])      # 相位
    state[:, 18] = torch.tensor([50, 30])    # 切换时间
    state[:, 19] = torch.tensor([1, 0])      # 主路信号
    state[:, 20] = torch.tensor([0, 1])      # 匝道信号
    state[:, 21] = torch.tensor([0, 0])      # 转出信号
    
    main_veh = torch.randn(batch_size, 5, 8)
    ramp_veh = torch.randn(batch_size, 3, 8)
    
    main_probs, ramp_probs, value, conflict = net_a(state, main_veh, ramp_veh)
    
    print(f"  主路动作概率: {main_probs.shape}")
    print(f"  匝道动作概率: {ramp_probs.shape}")
    print(f"  状态价值: {value.shape}")
    print(f"  冲突概率: {conflict.shape}")
    
    assert main_probs.shape == (batch_size, 11)
    assert ramp_probs.shape == (batch_size, 11)
    
    print("  ✓ 类型A网络测试通过")
    
    # 测试类型B网络
    print("\n测试类型B网络...")
    net_b = TypeBPolicyNetwork(config)
    
    diverge_veh = torch.randn(batch_size, 2, 8)
    
    main_probs, ramp_probs, diverge_probs, value, conflict = net_b(
        state, main_veh, ramp_veh, diverge_veh
    )
    
    print(f"  主路动作概率: {main_probs.shape}")
    print(f"  匝道动作概率: {ramp_probs.shape}")
    print(f"  转出动作概率: {diverge_probs.shape}")
    print(f"  状态价值: {value.shape}")
    
    assert diverge_probs.shape == (batch_size, 11)
    
    print("  ✓ 类型B网络测试通过")
    
    return True


def test_multi_junction_model_with_tl():
    """测试多路口模型（包含信号灯特征）"""
    print("\n" + "=" * 60)
    print("测试多路口模型（包含信号灯特征）")
    print("=" * 60)
    
    config = NetworkConfig()
    model = create_junction_model(JUNCTION_CONFIGS, config)
    
    print(f"\n模型创建成功")
    print(f"  路口数: {len(JUNCTION_CONFIGS)}")
    print(f"  状态维度: {config.type_a_state_dim}")
    
    # 创建假观察（包含信号灯特征）
    observations = {}
    vehicle_observations = {}
    
    for junc_id in JUNCTION_CONFIGS.keys():
        # 22维状态
        state = torch.randn(1, 22)
        
        # 设置信号灯特征
        state[0, 17] = 0       # 相位
        state[0, 18] = 50.0    # 切换时间
        state[0, 19] = 1.0     # 主路绿灯
        state[0, 20] = 0.0     # 匝道红灯
        state[0, 21] = 0.0     # 转出红灯
        
        observations[junc_id] = state
        vehicle_observations[junc_id] = {
            'main': torch.randn(1, 5, 8),
            'ramp': torch.randn(1, 3, 8),
            'diverge': torch.randn(1, 2, 8) if JUNCTION_CONFIGS[junc_id].junction_type == JunctionType.TYPE_B else None
        }
    
    # 前向传播
    actions, values, info = model(observations, vehicle_observations, deterministic=True)
    
    print(f"\n前向传播成功:")
    for junc_id in actions.keys():
        print(f"  {junc_id}:")
        print(f"    主路动作: {actions[junc_id]['main'].item():.3f}")
        print(f"    匝道动作: {actions[junc_id]['ramp'].item():.3f}")
        print(f"    价值: {values[junc_id].item():.3f}")
    
    print("\n✓ 多路口模型测试通过")
    return True


def test_state_vector_with_tl():
    """测试状态向量（包含信号灯特征）"""
    print("\n" + "=" * 60)
    print("测试状态向量（包含信号灯特征）")
    print("=" * 60)
    
    from junction_agent_subscription import JunctionAgent, JunctionConfig
    
    # 创建类型A智能体
    config_a = JunctionConfig(
        junction_id='TEST_A',
        junction_type=JunctionType.TYPE_A,
        main_incoming=['E1'],
        main_outgoing=['E2'],
        ramp_incoming=['R1'],
        has_traffic_light=True,
        tl_id='TEST_TL'
    )
    
    agent_a = JunctionAgent(config_a)
    
    print(f"\n类型A智能体:")
    print(f"  状态维度: {agent_a.get_state_dim()}")
    print(f"  动作维度: {agent_a.get_action_dim()}")
    
    # 创建假状态
    from junction_agent_subscription import JunctionState
    
    state = JunctionState(
        junction_id='TEST_A',
        timestamp=100.0,
        main_vehicles=[{'id': 'v1', 'speed': 10, 'lane_position': 50, 'waiting_time': 0, 'is_cv': True}],
        main_speed=10.0,
        main_density=0.5,
        main_queue_length=2,
        main_flow=100.0,
        ramp_vehicles=[{'id': 'v2', 'speed': 5, 'lane_position': 30, 'waiting_time': 10, 'is_cv': True}],
        ramp_speed=5.0,
        ramp_queue_length=3,
        ramp_waiting_time=10.0,
        ramp_flow=50.0,
        current_phase=0,
        phase_state="GGrrGG",
        time_in_phase=50.0,
        time_to_switch=40.0,
        next_phase=1,
        main_signal='G',
        ramp_signal='r',
        diverge_signal='r',
        conflict_risk=0.3,
        gap_acceptance=0.7,
        cv_vehicles_main=['v1'],
        cv_vehicles_ramp=['v2']
    )
    
    state_vec = agent_a.get_state_vector(state)
    
    print(f"\n状态向量 (维度={len(state_vec)}):")
    print(f"  主路特征: {state_vec[:5]}")
    print(f"  匝道特征: {state_vec[5:10]}")
    print(f"  信号灯特征: {state_vec[10:15]}")
    print(f"  冲突特征: {state_vec[15:17]}")
    print(f"  CV特征: {state_vec[17:19]}")
    print(f"  类型B特征: {state_vec[19:22]}")
    print(f"  时间: {state_vec[22]}")
    
    assert len(state_vec) == 22, f"状态维度错误: {len(state_vec)}"
    
    print("\n✓ 状态向量测试通过")
    return True


def test_model_save_load():
    """测试模型保存/加载"""
    print("\n" + "=" * 60)
    print("测试模型保存/加载")
    print("=" * 60)
    
    config = NetworkConfig()
    model = create_junction_model(JUNCTION_CONFIGS, config)
    
    # 保存
    save_path = '/tmp/test_junction_model_tl.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, save_path)
    print(f"\n模型已保存: {save_path}")
    
    # 加载
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载")
    
    print("\n✓ 模型保存/加载测试通过")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("路口级多智能体系统测试（订阅模式 + 信号灯特征）")
    print("=" * 60)
    
    results = {}
    
    # 运行测试
    try:
        results['tl_encoder'] = test_traffic_light_encoder()
    except Exception as e:
        print(f"✗ 信号灯编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['tl_encoder'] = False
    
    try:
        results['network_tl'] = test_network_with_tl()
    except Exception as e:
        print(f"✗ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['network_tl'] = False
    
    try:
        results['multi_model'] = test_multi_junction_model_with_tl()
    except Exception as e:
        print(f"✗ 多路口模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['multi_model'] = False
    
    try:
        results['state_vector'] = test_state_vector_with_tl()
    except Exception as e:
        print(f"✗ 状态向量测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['state_vector'] = False
    
    try:
        results['save_load'] = test_model_save_load()
    except Exception as e:
        print(f"✗ 保存/加载测试失败: {e}")
        results['save_load'] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！系统准备就绪。")
        print("\n关键改进:")
        print("  1. 使用SUMO订阅模式提高数据收集效率")
        print("  2. 信号灯相位作为重要特征（5维）")
        print("  3. 状态维度从16维增加到22维")
        print("  4. 信号灯特征编码器专门处理相位信息")
        print("\n开始训练:")
        print("  python junction_main.py train --total-timesteps 1000000")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    main()
