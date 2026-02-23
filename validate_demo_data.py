"""
验证专家演示数据的质量
"""
import pickle
import numpy as np
from collections import defaultdict


def validate_demo_data(demo_file):
    """验证演示数据"""
    print("=" * 80)
    print("专家演示数据质量验证")
    print("=" * 80)

    with open(demo_file, 'rb') as f:
        episodes = pickle.load(f)

    print(f"\n总Episodes: {len(episodes)}")

    # 统计信息
    all_transitions = []
    all_ocr = []
    all_rewards = []
    all_steps = []

    junction_stats = defaultdict(int)
    state_dim = None
    action_dims = []

    for ep in episodes:
        transitions = ep['transitions']
        all_transitions.append(len(transitions))
        all_ocr.append(ep['final_ocr'])
        all_rewards.append(ep['total_reward'])
        all_steps.append(ep.get('steps', 0))

        for trans in transitions:
            junc_id = trans['junction_id']
            junction_stats[junc_id] += 1

            if state_dim is None:
                state_dim = trans['state'].shape[0]
                action_dims.append(trans['action_main'].shape)
                action_dims.append(trans['action_ramp'].shape)

    # 打印统计
    print(f"\n基本统计:")
    print(f"  总transitions: {sum(all_transitions)}")
    print(f"  平均每episode: {np.mean(all_transitions):.1f} transitions")
    print(f"  平均steps: {np.mean(all_steps):.1f}")

    print(f"\n质量指标:")
    print(f"  平均OCR: {np.mean(all_ocr):.4f}")
    print(f"  OCR范围: {np.min(all_ocr):.4f} - {np.max(all_ocr):.4f}")
    print(f"  平均奖励: {np.mean(all_rewards):.2f}")

    print(f"\n状态和动作维度:")
    print(f"  状态维度: {state_dim}")
    print(f"  主路动作维度: {action_dims[0] if action_dims else 'N/A'}")
    print(f"  匝道动作维度: {action_dims[1] if len(action_dims) > 1 else 'N/A'}")

    print(f"\n路口分布:")
    for junc_id, count in sorted(junction_stats.items()):
        print(f"  {junc_id}: {count} transitions ({count/sum(all_transitions)*100:.1f}%)")

    # 数据质量检查
    print(f"\n质量检查:")

    issues = []

    # 检查OCR
    if np.mean(all_ocr) < 0.90:
        issues.append("⚠ 平均OCR偏低 (<0.90)")
    else:
        print("  ✓ OCR质量良好")

    # 检查transitions数量
    if np.mean(all_transitions) < 1000:
        issues.append("⚠ 平均transitions数量偏少")
    else:
        print("  ✓ Transitions数量充足")

    # 检查状态维度
    if state_dim == 23:
        print("  ✓ 状态维度正确 (23维)")
    else:
        issues.append(f"✗ 状态维度错误 (应为23, 实际{state_dim})")

    # 检查动作维度
    if action_dims and action_dims[0] == (1,):
        print("  ✓ 动作维度正确")
    else:
        issues.append(f"✗ 动作维度错误")

    # 检查一致性
    if len(set(all_transitions)) == 1:
        print("  ✓ 所有episodes的transitions数量一致")
    else:
        print(f"  ℹ Episodes的transitions数量有差异 (范围: {min(all_transitions)}-{max(all_transitions)})")

    # 总结
    print("\n" + "=" * 80)
    if issues:
        print("发现问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ 所有检查通过！数据质量良好。")

    print("\n建议:")
    if np.mean(all_ocr) >= 0.94:
        print("  ✓ OCR优秀，可以直接用于训练BC模型")
    elif np.mean(all_ocr) >= 0.90:
        print("  ℹ OCR良好，可以用于训练，但可能需要更多episodes")
    else:
        print("  ⚠ OCR偏低，建议检查专家策略参数")

    print("=" * 80)

    return {
        'num_episodes': len(episodes),
        'total_transitions': sum(all_transitions),
        'avg_ocr': np.mean(all_ocr),
        'avg_reward': np.mean(all_rewards),
        'state_dim': state_dim,
    }


if __name__ == '__main__':
    import sys

    demo_file = sys.argv[1] if len(sys.argv) > 1 else 'expert_demos/expert_demonstrations.pkl'

    stats = validate_demo_data(demo_file)
