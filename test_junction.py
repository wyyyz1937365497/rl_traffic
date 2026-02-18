"""
路口级多智能体系统测试脚本
"""

import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from junction_agent import JUNCTION_CONFIGS, JunctionType, JunctionAgent, JunctionConfig
from junction_network import create_junction_model, NetworkConfig, TypeAPolicyNetwork, TypeBPolicyNetwork


def test_junction_config():
    """测试路口配置"""
    print("=" * 60)
    print("测试路口配置")
    print("=" * 60)
    
    assert len(JUNCTION_CONFIGS) == 4, "应该有4个关键路口"
    
    for junc_id, config in JUNCTION_CONFIGS.items():
        print(f"\n{junc_id}:")
        print(f"  类型: {config.junction_type.value}")
        print(f"  主路入边: {config.main_incoming}")
        print(f"  匝道入边: {config.ramp_incoming}")
        
        if config.junction_type == JunctionType.TYPE_A:
            assert len(config.ramp_outgoing) == 0, "类型A不应该有匝道出边"
            print("  ✓ 类型A验证通过")
        else:
            assert len(config.ramp_outgoing) > 0, "类型B应该有匝道出边"
            print("  ✓ 类型B验证通过")
    
    print("\n✓ 路口配置测试通过")
    return True


def test_junction_agent():
    """测试路口智能体"""
    print("\n" + "=" * 60)
    print("测试路口智能体")
    print("=" * 60)
    
    # 创建类型A智能体
    config_a = JunctionConfig(
        junction_id='TEST_A',
        junction_type=JunctionType.TYPE_A,
        main_incoming=['E1'],
        main_outgoing=['E2'],
        ramp_incoming=['R1']
    )
    
    agent_a = JunctionAgent(config_a)
    print(f"\n类型A智能体创建成功")
    print(f"  状态维度: {agent_a.get_state_dim()}")
    print(f"  动作维度: {agent_a.get_action_dim()}")
    
    # 创建类型B智能体
    config_b = JunctionConfig(
        junction_id='TEST_B',
        junction_type=JunctionType.TYPE_B,
        main_incoming=['E1'],
        main_outgoing=['E2'],
        ramp_incoming=['R1'],
        ramp_outgoing=['D1']
    )
    
    agent_b = JunctionAgent(config_b)
    print(f"\n类型B智能体创建成功")
    print(f"  状态维度: {agent_b.get_state_dim()}")
    print(f"  动作维度: {agent_b.get_action_dim()}")
    
    print("\n✓ 路口智能体测试通过")
    return True


def test_network():
    """测试网络"""
    print("\n" + "=" * 60)
    print("测试神经网络")
    print("=" * 60)
    
    config = NetworkConfig()
    
    # 测试类型A网络
    print("\n测试类型A网络...")
    net_a = TypeAPolicyNetwork(config)
    
    # 创建假数据
    batch_size = 2
    state = torch.randn(batch_size, config.type_a_state_dim)
    main_veh = torch.randn(batch_size, 5, 8)  # 5辆主路车
    ramp_veh = torch.randn(batch_size, 3, 8)  # 3辆匝道车
    
    main_probs, ramp_probs, value, conflict = net_a(state, main_veh, ramp_veh)
    
    print(f"  主路动作概率: {main_probs.shape}")
    print(f"  匝道动作概率: {ramp_probs.shape}")
    print(f"  状态价值: {value.shape}")
    print(f"  冲突概率: {conflict.shape}")
    
    assert main_probs.shape == (batch_size, 11), "主路动作维度错误"
    assert ramp_probs.shape == (batch_size, 11), "匝道动作维度错误"
    
    print("  ✓ 类型A网络测试通过")
    
    # 测试类型B网络
    print("\n测试类型B网络...")
    net_b = TypeBPolicyNetwork(config)
    
    diverge_veh = torch.randn(batch_size, 2, 8)  # 2辆转出车
    
    main_probs, ramp_probs, diverge_probs, value, conflict = net_b(
        state, main_veh, ramp_veh, diverge_veh
    )
    
    print(f"  主路动作概率: {main_probs.shape}")
    print(f"  匝道动作概率: {ramp_probs.shape}")
    print(f"  转出动作概率: {diverge_probs.shape}")
    print(f"  状态价值: {value.shape}")
    
    assert diverge_probs.shape == (batch_size, 11), "转出动作维度错误"
    
    print("  ✓ 类型B网络测试通过")
    
    return True


def test_multi_junction_model():
    """测试多路口模型"""
    print("\n" + "=" * 60)
    print("测试多路口联合模型")
    print("=" * 60)
    
    config = NetworkConfig()
    model = create_junction_model(JUNCTION_CONFIGS, config)
    
    print(f"\n模型创建成功")
    print(f"  路口数: {len(JUNCTION_CONFIGS)}")
    
    # 创建假观察
    observations = {}
    vehicle_observations = {}
    
    for junc_id in JUNCTION_CONFIGS.keys():
        observations[junc_id] = torch.randn(1, 16)
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


def test_model_save_load():
    """测试模型保存/加载"""
    print("\n" + "=" * 60)
    print("测试模型保存/加载")
    print("=" * 60)
    
    config = NetworkConfig()
    model = create_junction_model(JUNCTION_CONFIGS, config)
    
    # 保存
    save_path = '/tmp/test_junction_model.pt'
    torch.save({
        'model_state_dict': model.state_dict()
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
    print("路口级多智能体系统测试")
    print("=" * 60)
    
    results = {}
    
    # 运行测试
    try:
        results['config'] = test_junction_config()
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        results['config'] = False
    
    try:
        results['agent'] = test_junction_agent()
    except Exception as e:
        print(f"✗ 智能体测试失败: {e}")
        results['agent'] = False
    
    try:
        results['network'] = test_network()
    except Exception as e:
        print(f"✗ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['network'] = False
    
    try:
        results['multi_model'] = test_multi_junction_model()
    except Exception as e:
        print(f"✗ 多路口模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['multi_model'] = False
    
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
        print("\n开始训练:")
        print("  python junction_main.py train --total-timesteps 1000000")
        print("\n查看路口信息:")
        print("  python junction_main.py info")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    main()
