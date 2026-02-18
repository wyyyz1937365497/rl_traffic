"""
快速测试脚本
验证环境和模型是否正常工作
"""

import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from config import get_default_config
from environment import TrafficEnvironment
from network import create_model
from advanced_model import create_advanced_model


def test_config():
    """测试配置"""
    print("测试配置...")
    config = get_default_config()
    
    assert config.env.max_steps > 0
    assert config.network.gnn_hidden_dim > 0
    assert config.ppo.gamma > 0
    
    print("  ✓ 配置测试通过")
    return config


def test_environment(config):
    """测试环境"""
    print("\n测试环境...")
    
    try:
        env = TrafficEnvironment(config.env, use_gui=False, seed=42)
        print("  ✓ 环境创建成功")
        
        # 测试重置
        obs = env.reset()
        assert 'vehicle_features' in obs
        assert 'edge_features' in obs
        assert 'global_features' in obs
        print("  ✓ 环境重置成功")
        
        # 测试步进
        action_dict = {}
        for veh_id in obs.get('controlled_vehicles', [])[:5]:
            action_dict[veh_id] = 0.5
        
        next_obs, reward, done, info = env.step(action_dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        print("  ✓ 环境步进成功")
        
        # 测试统计
        stats = env.get_statistics()
        assert 'ocr' in stats
        print(f"  ✓ 当前OCR: {stats['ocr']:.4f}")
        
        env.close()
        print("  ✓ 环境关闭成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 环境测试失败: {e}")
        return False


def test_basic_model(config):
    """测试基础模型"""
    print("\n测试基础模型...")
    
    try:
        model = create_model(config.network)
        print("  ✓ 基础模型创建成功")
        
        # 测试前向传播
        dummy_obs = {
            'vehicle_features': np.random.randn(50, 15).astype(np.float32),
            'edge_features': np.random.randn(20, 10).astype(np.float32),
            'global_features': np.random.randn(10).astype(np.float32),
            'graph': {
                'node_features': np.random.randn(70, 15).astype(np.float32),
                'edge_index': np.zeros((2, 0), dtype=np.int64),
                'edge_attr': np.zeros((0, 2), dtype=np.float32),
                'num_vehicles': 50,
                'num_edges': 20
            },
            'controlled_vehicles': ['veh_0', 'veh_1', 'veh_2'],
            'cv_vehicles': ['veh_0', 'veh_1', 'veh_2', 'veh_3']
        }
        
        with torch.no_grad():
            action_dict, value, log_prob = model(dummy_obs, [], deterministic=True)
        
        print(f"  ✓ 前向传播成功，动作数: {len(action_dict)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 基础模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_model(config):
    """测试高级模型"""
    print("\n测试高级模型...")
    
    try:
        model = create_advanced_model(config.network)
        print("  ✓ 高级模型创建成功")
        
        # 测试前向传播
        dummy_obs = {
            'vehicle_features': np.random.randn(50, 15).astype(np.float32),
            'edge_features': np.random.randn(20, 10).astype(np.float32),
            'global_features': np.random.randn(10).astype(np.float32),
            'graph': {
                'node_features': np.random.randn(70, 15).astype(np.float32),
                'edge_index': np.zeros((2, 0), dtype=np.int64),
                'edge_attr': np.zeros((0, 2), dtype=np.float32),
                'num_vehicles': 50,
                'num_edges': 20
            },
            'controlled_vehicles': ['veh_0', 'veh_1', 'veh_2'],
            'cv_vehicles': ['veh_0', 'veh_1', 'veh_2', 'veh_3']
        }
        
        with torch.no_grad():
            action_dict, value, log_prob = model(dummy_obs, [], deterministic=True)
        
        print(f"  ✓ 前向传播成功，动作数: {len(action_dict)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 高级模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(config):
    """测试训练步骤"""
    print("\n测试训练步骤...")
    
    try:
        from ppo import PPOTrainer
        
        trainer = PPOTrainer(config)
        print("  ✓ 训练器创建成功")
        
        # 测试模型保存/加载
        test_path = '/tmp/test_model.pt'
        trainer.save(test_path)
        trainer.load(test_path)
        print("  ✓ 模型保存/加载成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("强化学习交通控制系统 - 测试")
    print("=" * 60)
    
    # 测试配置
    config = test_config()
    
    # 测试环境
    env_success = test_environment(config)
    
    # 测试模型
    basic_model_success = test_basic_model(config)
    advanced_model_success = test_advanced_model(config)
    
    # 测试训练
    training_success = test_training_step(config)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"  环境: {'✓ 通过' if env_success else '✗ 失败'}")
    print(f"  基础模型: {'✓ 通过' if basic_model_success else '✗ 失败'}")
    print(f"  高级模型: {'✓ 通过' if advanced_model_success else '✗ 失败'}")
    print(f"  训练步骤: {'✓ 通过' if training_success else '✗ 失败'}")
    
    all_success = all([env_success, basic_model_success, advanced_model_success, training_success])
    
    if all_success:
        print("\n🎉 所有测试通过！系统准备就绪。")
        print("\n开始训练:")
        print("  python train.py --total-timesteps 1000000")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_success


if __name__ == '__main__':
    main()
