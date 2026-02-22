"""
快速测试CV车辆分配和经验收集
"""
import sys
sys.path.insert(0, '.')

from junction_agent import MultiAgentEnvironment
import torch

def test_reset_return_type():
    """测试env.reset()返回值类型"""
    print("="*70)
    print("测试env.reset()返回值")
    print("="*70)

    # 创建环境
    env = MultiAgentEnvironment(
        junction_ids=['J5', 'J14'],
        sumo_cfg='sumo/sumo.sumocfg',
        use_gui=False,
        seed=42
    )

    # 重置环境
    print("\n重置环境...")
    obs = env.reset()

    print(f"\n观察类型: {type(obs)}")
    print(f"观察键: {list(obs.keys())}")

    for junc_id, obs_value in obs.items():
        print(f"\n{junc_id}:")
        print(f"  类型: {type(obs_value)}")
        print(f"  内容: {obs_value}")
        if hasattr(obs_value, 'shape'):
            print(f"  形状: {obs_value.shape}")
        elif hasattr(obs_value, '__class__'):
            print(f"  类名: {obs_value.__class__.__name__}")

    # 关闭环境
    env.close()

def test_cv_allocation():
    """测试CV车辆分配"""
    print("\n" + "="*70)
    print("测试CV车辆分配")
    print("="*70)

    # 创建环境
    env = MultiAgentEnvironment(
        junction_ids=['J5', 'J14', 'J15', 'J17'],
        sumo_cfg='sumo/sumo.sumocfg',
        use_gui=False,
        seed=42
    )

    # 重置环境
    print("\n重置环境...")
    obs = env.reset()

    print(f"\n观察数量: {len(obs)}")
    print(f"观察键: {list(obs.keys())}")

    # 检查CV分配
    print(f"\n全局CV分配数量: {len(env._global_cv_assignment)}")

    # 统计每个路口的分配
    junction_stats = {}
    for veh_id, junc_id in env._global_cv_assignment.items():
        if junc_id not in junction_stats:
            junction_stats[junc_id] = 0
        junction_stats[junc_id] += 1

    print(f"\n路口分配统计:")
    for junc_id in sorted(junction_stats.keys()):
        controlled = env.get_controlled_vehicles_for_junction(junc_id)
        print(f"  {junc_id}: {junction_stats[junc_id]}辆 (main={len(controlled['main'])}, "
              f"ramp={len(controlled['ramp'])}, diverge={len(controlled.get('diverge', []))})")

    # 测试一步
    print("\n执行一步...")
    actions = {}
    for junc_id in env.agents.keys():
        controlled = env.get_controlled_vehicles_for_junction(junc_id)
        actions[junc_id] = {}
        for veh_id in controlled['main'][:1]:
            actions[junc_id][veh_id] = 0.5

    next_obs, rewards, done, info = env.step(actions)

    print(f"next_obs类型: {type(next_obs)}")
    print(f"奖励: {rewards}")
    print(f"Done: {done}")

    # 关闭环境
    env.close()

    print("\n✅ 测试完成！")

if __name__ == '__main__':
    test_reset_return_type()
    test_cv_allocation()
