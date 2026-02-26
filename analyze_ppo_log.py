#!/usr/bin/env python3
"""
PPO微调日志分析脚本
从日志文件中提取训练指标并生成可视化图表
"""

import re
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_file_path):
    """
    解析PPO微调日志文件，提取训练指标
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 存储每个episode的指标
    episodes_data = []

    # 当前episode的指标
    current_episode = None
    episode_data = {}

    for line in lines:
        # 检查是否是episode开始行
        episode_match = re.search(r'Episode (\d+)/(\d+)', line)
        if episode_match:
            if current_episode is not None:
                # 保存上一个episode的数据
                episodes_data.append(episode_data.copy())

            current_episode = int(episode_match.group(1))
            episode_data = {
                'episode': current_episode,
                'total_reward': 0,
                'avg_reward': 0,
                'length': 0,
                'policy_loss': 0,
                'value_loss': 0,
                'entropy': 0,
                'entropy_coef': 0,
                'traffic_reward': 0,
                'stability_reward': 0,
                'ocr_reward': 0,
                'safety_penalty': 0
            }

        # 提取各种指标
        if current_episode is not None:
            # 总奖励
            reward_match = re.search(r'\[Episode ' + str(current_episode) + r'\] 总奖励: ([\d.]+)', line)
            if reward_match:
                episode_data['total_reward'] = float(reward_match.group(1))

            # 平均奖励
            avg_reward_match = re.search(r'\[Episode ' + str(current_episode) + r'\] 平均奖励: ([\d.]+)', line)
            if avg_reward_match:
                episode_data['avg_reward'] = float(avg_reward_match.group(1))

            # 长度
            length_match = re.search(r'\[Episode ' + str(current_episode) + r'\] 长度: (\d+)', line)
            if length_match:
                episode_data['length'] = int(length_match.group(1))

            # Policy loss
            policy_loss_match = re.search(r'\[Episode ' + str(current_episode) + r'\] Policy loss: ([\d.-]+)', line)
            if policy_loss_match:
                episode_data['policy_loss'] = float(policy_loss_match.group(1))

            # Value loss
            value_loss_match = re.search(r'\[Episode ' + str(current_episode) + r'\] Value loss: ([\d.-]+)', line)
            if value_loss_match:
                episode_data['value_loss'] = float(value_loss_match.group(1))

            # Entropy
            entropy_match = re.search(r'\[Episode ' + str(current_episode) + r'\] Entropy: ([\d.-]+)', line)
            if entropy_match:
                episode_data['entropy'] = float(entropy_match.group(1))

            # Entropy Coef
            entropy_coef_match = re.search(r'\[Episode ' + str(current_episode) + r'\] Entropy coef: ([\d.eE+-]+)', line)
            if entropy_coef_match:
                episode_data['entropy_coef'] = float(entropy_coef_match.group(1))

            # 流量奖励
            traffic_match = re.search(r'【流量奖励】总计: ([\d.]+)', line)
            if traffic_match:
                episode_data['traffic_reward'] = float(traffic_match.group(1))

            # 稳定性奖励
            stability_match = re.search(r'【稳定性奖励】总计: ([\d.]+)', line)
            if stability_match:
                episode_data['stability_reward'] = float(stability_match.group(1))

            # OCR奖励
            ocr_match = re.search(r'【OCR奖励】: ([\d.]+)', line)
            if ocr_match:
                episode_data['ocr_reward'] = float(ocr_match.group(1))

            # 安全性惩罚
            safety_match = re.search(r'【安全性惩罚】总计: (-[\d.]+)', line)
            if safety_match:
                episode_data['safety_penalty'] = float(safety_match.group(1))

    # 添加最后一个episode的数据
    if current_episode is not None:
        episodes_data.append(episode_data)

    return episodes_data

def save_metrics_to_json(episodes_data, output_dir):
    """
    将提取的指标保存为JSON文件
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 提取每个指标的列表
    metrics = {
        'episode': [ep['episode'] for ep in episodes_data],
        'total_reward': [ep['total_reward'] for ep in episodes_data],
        'avg_reward': [ep['avg_reward'] for ep in episodes_data],
        'length': [ep['length'] for ep in episodes_data],
        'policy_loss': [ep['policy_loss'] for ep in episodes_data],
        'value_loss': [ep['value_loss'] for ep in episodes_data],
        'entropy': [ep['entropy'] for ep in episodes_data],
        'entropy_coef': [ep['entropy_coef'] for ep in episodes_data],
        'traffic_reward': [ep['traffic_reward'] for ep in episodes_data],
        'stability_reward': [ep['stability_reward'] for ep in episodes_data],
        'ocr_reward': [ep['ocr_reward'] for ep in episodes_data],
        'safety_penalty': [ep['safety_penalty'] for ep in episodes_data]
    }

    # 保存每个指标为单独的JSON文件
    for metric_name, metric_values in metrics.items():
        output_file = os.path.join(output_dir, f'{metric_name}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metric_values, f, indent=2)

    print(f"指标已保存到 {output_dir} 目录")

def plot_metrics(episodes_data, output_dir):
    """
    绘制训练指标变化曲线
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 准备数据
    episodes = [ep['episode'] for ep in episodes_data]

    # 总奖励曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(episodes, [ep['total_reward'] for ep in episodes_data], marker='o')
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # 平均奖励曲线
    plt.subplot(2, 3, 2)
    plt.plot(episodes, [ep['avg_reward'] for ep in episodes_data], marker='s')
    plt.title('Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # Policy Loss曲线
    plt.subplot(2, 3, 3)
    plt.plot(episodes, [ep['policy_loss'] for ep in episodes_data], marker='^')
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)

    # Value Loss曲线
    plt.subplot(2, 3, 4)
    plt.plot(episodes, [ep['value_loss'] for ep in episodes_data], marker='v')
    plt.title('Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)

    # Entropy曲线
    plt.subplot(2, 3, 5)
    plt.plot(episodes, [ep['entropy'] for ep in episodes_data], marker='d')
    plt.title('Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.grid(True)

    # 各种奖励分解
    plt.subplot(2, 3, 6)
    traffic_rewards = [ep['traffic_reward'] for ep in episodes_data]
    stability_rewards = [ep['stability_reward'] for ep in episodes_data]
    ocr_rewards = [ep['ocr_reward'] for ep in episodes_data]
    safety_penalties = [ep['safety_penalty'] for ep in episodes_data]

    plt.plot(episodes, traffic_rewards, label='Traffic Reward', marker='o')
    plt.plot(episodes, stability_rewards, label='Stability Reward', marker='s')
    plt.plot(episodes, ocr_rewards, label='OCR Reward', marker='^')
    plt.plot(episodes, safety_penalties, label='Safety Penalty', marker='v')
    plt.title('Reward Components')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

    print(f"图表已保存到 {output_dir}/training_metrics.png")


def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def analyze_degradation(episodes_data):
    """生成退化分析摘要"""
    if not episodes_data:
        return {
            'num_episodes': 0,
            'message': '无有效episode数据'
        }

    # 过滤未完成写入的异常尾点（全部或关键指标为0）
    cleaned = []
    for ep in episodes_data:
        if ep['episode'] <= 0:
            continue
        if ep['total_reward'] == 0 and ep['avg_reward'] == 0 and ep['length'] == 0:
            continue
        cleaned.append(ep)

    if not cleaned:
        return {
            'num_episodes': 0,
            'message': '清洗后无有效episode数据'
        }

    rewards = np.array([ep['total_reward'] for ep in cleaned], dtype=float)
    policy_loss = np.array([ep['policy_loss'] for ep in cleaned], dtype=float)
    value_loss = np.array([ep['value_loss'] for ep in cleaned], dtype=float)
    entropy = np.array([ep['entropy'] for ep in cleaned], dtype=float)
    entropy_coef = np.array([ep['entropy_coef'] for ep in cleaned], dtype=float)
    ocr_reward = np.array([ep['ocr_reward'] for ep in cleaned], dtype=float)
    safety_penalty = np.array([ep['safety_penalty'] for ep in cleaned], dtype=float)

    n = len(cleaned)
    best_idx = int(np.argmax(rewards))
    best_episode = int(cleaned[best_idx]['episode'])
    best_reward = float(rewards[best_idx])
    final_reward = float(rewards[-1])

    post_best = rewards[best_idx:]
    post_best_mean = _safe_mean(post_best)
    post_best_min = float(np.min(post_best))

    # 回撤定义：峰值到之后最低点
    max_drawdown = best_reward - post_best_min
    drawdown_ratio = max_drawdown / max(best_reward, 1e-8)

    # 前后半程对比
    mid = n // 2
    first_half = rewards[:mid] if mid > 0 else rewards
    second_half = rewards[mid:] if mid > 0 else rewards

    # 峰值后趋势斜率（最小二乘）
    if len(post_best) >= 2:
        x = np.arange(len(post_best), dtype=float)
        slope = float(np.polyfit(x, post_best, 1)[0])
    else:
        slope = 0.0

    return {
        'num_episodes': n,
        'best_episode': best_episode,
        'best_total_reward': best_reward,
        'final_total_reward': final_reward,
        'final_vs_best_delta': final_reward - best_reward,
        'post_best_mean_reward': post_best_mean,
        'max_drawdown': float(max_drawdown),
        'max_drawdown_ratio': float(drawdown_ratio),
        'post_best_trend_slope': slope,
        'first_half_mean_reward': _safe_mean(first_half),
        'second_half_mean_reward': _safe_mean(second_half),
        'policy_loss_mean': _safe_mean(policy_loss),
        'value_loss_mean': _safe_mean(value_loss),
        'entropy_mean': _safe_mean(entropy),
        'entropy_coef_last': float(entropy_coef[-1]) if len(entropy_coef) > 0 else 0.0,
        'ocr_reward_mean': _safe_mean(ocr_reward),
        'safety_penalty_mean': _safe_mean(safety_penalty),
    }


def save_summary(summary, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    text_path = os.path.join(output_dir, 'summary.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write('PPO训练退化分析摘要\n')
        f.write('=' * 40 + '\n')
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')

    print(f"摘要已保存: {summary_path}")
    print(f"文本摘要已保存: {text_path}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='PPO微调日志分析')
    parser.add_argument('--log-file', type=str, default='logs/ppo_finetune/finetune_20260224_141249.log', help='日志文件路径')
    parser.add_argument('--output-dir', type=str, default='logs/ppo_finetune/analyze', help='分析输出目录')
    args = parser.parse_args()

    log_file_path = args.log_file

    # 检查日志文件是否存在
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件不存在 - {log_file_path}")
        return

    # 解析日志文件
    print("正在解析日志文件...")
    episodes_data = parse_log_file(log_file_path)
    print(f"解析完成，共找到 {len(episodes_data)} 个episode的数据")

    # 指标输出目录
    output_dir = args.output_dir

    # 保存指标到JSON文件
    print("正在保存指标到JSON文件...")
    save_metrics_to_json(episodes_data, output_dir)

    # 绘制图表
    print("正在绘制训练指标图表...")
    plot_metrics(episodes_data, output_dir)

    # 退化分析摘要
    print("正在生成退化分析摘要...")
    summary = analyze_degradation(episodes_data)
    save_summary(summary, output_dir)

    print("\n关键结论:")
    for key in ['num_episodes', 'best_episode', 'best_total_reward', 'final_total_reward', 'final_vs_best_delta', 'max_drawdown', 'max_drawdown_ratio', 'post_best_trend_slope']:
        if key in summary:
            print(f"  {key}: {summary[key]}")

    print("分析完成！")

if __name__ == "__main__":
    main()