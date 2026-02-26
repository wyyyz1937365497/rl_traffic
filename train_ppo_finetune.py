"""
完整PPO微调脚本 - 从BC模型初始化

基于junction_trainer.py，实现完整的RL微调流程
"""
import os
import sys
import time
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 设置控制台编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

sys.path.insert(0, '.')

from junction_agent import JUNCTION_CONFIGS, JunctionAgent, MultiAgentEnvironment, traci
from junction_network import VehicleLevelMultiJunctionModel, NetworkConfig
from junction_trainer import PPOConfig, ExperienceBuffer


class PPOFinetuner:
    """
    PPO微调器 - 从BC模型初始化

    完整实现：
    1. 从BC checkpoint加载权重
    2. 使用车辆级模型架构
    3. 完整的PPO训练循环
    4. 详细日志和监控
    """

    def __init__(
        self,
        bc_checkpoint_path: str,
        config: PPOConfig = None,
        device: str = 'cuda',
        log_dir: str = './logs/ppo_finetune',
        anchor_coef: float = 1e-5,
        anchor_decay: float = 0.999,
        early_stop_patience: int = 0
    ):
        """
        初始化PPO微调器

        Args:
            bc_checkpoint_path: BC模型checkpoint路径
            config: PPO配置
            device: 设备
            log_dir: 日志目录
        """
        self.device = device
        self.log_dir = log_dir
        self.config = config or PPOConfig()
        self.bc_checkpoint_path = bc_checkpoint_path

        os.makedirs(log_dir, exist_ok=True)

        # 设置日志
        self._setup_logging()

        logging.info("=" * 70)
        logging.info("PPO微调器初始化")
        logging.info("=" * 70)

        # 步骤1: 加载BC模型
        self.model = self._load_bc_model(bc_checkpoint_path)

        # 步骤2: 配置优化器（使用更小的学习率进行微调）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            eps=1e-5
        )

        logging.info(f"[优化器] 学习率: {self.config.lr}")

        # 步骤4: TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 步骤5: 训练统计
        self.episode_stats = []
        self.best_reward = float('-inf')
        self.global_step = 0
        self.entropy_coef = self.config.entropy_coef
        self.cumulative_departed = 0
        self.cumulative_arrived = 0
        self.value_clip_epsilon = 0.2
        self.anchor_coef = float(anchor_coef)
        self.anchor_decay = float(anchor_decay)
        self.anchor_min = 0.0
        self.early_stop_patience = int(early_stop_patience)
        self.no_improve_count = 0
        self.bc_anchor_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        logging.info(f"[防漂移] Anchor coef: {self.anchor_coef}")
        logging.info(f"[防漂移] Anchor decay: {self.anchor_decay}")
        logging.info(f"[早停] Patience: {self.early_stop_patience}")

        self.min_speed_ratio = 0.88
        self.release_speed_ratio = 0.78
        self.near_merge_threshold = 0.45
        self.prev_ocr = 0.0

        logging.info("=" * 70)
        logging.info("初始化完成\n")

    def _pin_if_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """对CPU tensor进行pin_memory，便于异步传输"""
        if torch.is_tensor(tensor) and tensor.device.type == 'cpu':
            return tensor.pin_memory()
        return tensor

    def _to_device_async(self, tensor: torch.Tensor) -> torch.Tensor:
        """异步传输到训练设备"""
        if not torch.is_tensor(tensor):
            return tensor
        return tensor.to(self.device, non_blocking=True)

    def _setup_logging(self):
        """配置日志"""
        log_file = os.path.join(self.log_dir, f'finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def _candidate_checkpoint_keys(self, model_key: str) -> List[str]:
        """为模型参数名生成可能的checkpoint键名（历史前缀/别名对齐）"""
        candidates = [model_key]

        # 常见前缀包装
        candidates.append(f"network.{model_key}")
        candidates.append(f"model.{model_key}")
        candidates.append(f"module.{model_key}")
        candidates.append(f"module.network.{model_key}")
        candidates.append(f"module.model.{model_key}")

        # 历史命名别名（低风险字符串替换）
        alias_pairs = [
            ("main_controller", "main"),
            ("ramp_controller", "ramp"),
            ("diverge_controller", "diverge"),
            ("vehicle_action_head", "action_head"),
        ]

        for src, dst in alias_pairs:
            if src in model_key:
                alias_key = model_key.replace(src, dst)
                candidates.append(alias_key)
                candidates.append(f"network.{alias_key}")
                candidates.append(f"model.{alias_key}")
                candidates.append(f"module.{alias_key}")

        # 去重并保持顺序
        seen = set()
        uniq = []
        for key in candidates:
            if key not in seen:
                uniq.append(key)
                seen.add(key)
        return uniq

    def _load_bc_model(self, checkpoint_path: str):
        """从BC checkpoint加载模型"""
        logging.info(f"[步骤1] 加载BC模型: {checkpoint_path}")

        # 创建车辆级模型
        config = NetworkConfig()
        model = VehicleLevelMultiJunctionModel(JUNCTION_CONFIGS, config)

        # 检测设备可用性
        device = self.device
        if device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA不可用，自动切换到CPU")
            device = 'cpu'
            self.device = 'cpu'  # 更新实际使用的设备

        # 加载权重（使用正确的map_location）
        map_location = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('val_loss', 'unknown')
            logging.info(f"  Checkpoint: epoch={epoch}, val_loss={val_loss}")
        else:
            state_dict = checkpoint

        # 原始checkpoint键集合（保留原始前缀，后续候选键匹配）
        checkpoint_state_dict = dict(state_dict)

        # 加载权重（允许部分加载）
        model_state = model.state_dict()

        # 调试：打印前5个key
        logging.info(f"  Checkpoint keys (前5个): {list(checkpoint_state_dict.keys())[:5]}")
        logging.info(f"  Model keys (前5个): {list(model_state.keys())[:5]}")
        logging.info(f"  Checkpoint keys 总数: {len(checkpoint_state_dict)}")

        # 导出完整checkpoint键名，便于排查
        key_dump_path = os.path.join(
            self.log_dir,
            f"bc_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(key_dump_path, 'w', encoding='utf-8') as dump_file:
            for key in sorted(checkpoint_state_dict.keys()):
                dump_file.write(f"{key}\n")
        logging.info(f"  [INFO] BC权重键名已导出: {key_dump_path}")

        pretrained_dict = {}
        matched_pairs = []
        shape_mismatch = {}
        missing_key_models = []
        for model_key, model_tensor in model_state.items():
            candidate_keys = self._candidate_checkpoint_keys(model_key)
            present_keys = []
            for ckpt_key in candidate_keys:
                if ckpt_key in checkpoint_state_dict:
                    present_keys.append(ckpt_key)
                if ckpt_key in checkpoint_state_dict and checkpoint_state_dict[ckpt_key].shape == model_tensor.shape:
                    pretrained_dict[model_key] = checkpoint_state_dict[ckpt_key]
                    matched_pairs.append((model_key, ckpt_key))
                    break
            else:
                if present_keys:
                    first_present = present_keys[0]
                    shape_mismatch[model_key] = (
                        first_present,
                        tuple(model_tensor.shape),
                        tuple(checkpoint_state_dict[first_present].shape)
                    )
                else:
                    missing_key_models.append(model_key)

        # 回退策略：若checkpoint缺少部分路口参数，使用J5参数做共享初始化
        # 适用于“单路口BC权重 -> 多路口PPO微调”场景
        shared_init_pairs = []
        for model_key, model_tensor in model_state.items():
            if model_key in pretrained_dict:
                continue

            if not model_key.startswith('networks.'):
                continue

            parts = model_key.split('.')
            if len(parts) < 3:
                continue

            target_junc = parts[1]
            if target_junc == 'J5':
                continue

            source_key = model_key.replace(f'networks.{target_junc}.', 'networks.J5.', 1)
            for ckpt_key in self._candidate_checkpoint_keys(source_key):
                if ckpt_key in checkpoint_state_dict and checkpoint_state_dict[ckpt_key].shape == model_tensor.shape:
                    pretrained_dict[model_key] = checkpoint_state_dict[ckpt_key]
                    shared_init_pairs.append((model_key, ckpt_key))
                    break

        model.load_state_dict(pretrained_dict, strict=False)
        model = model.to(device)

        loaded_keys = len(pretrained_dict)
        total_keys = len(model_state)
        logging.info(f"  [OK] 加载权重: {loaded_keys}/{total_keys} 参数")
        if shared_init_pairs:
            logging.info(f"  [INFO] 共享初始化参数: {len(shared_init_pairs)} (J5 -> 其他路口)")
            for model_key, ckpt_key in shared_init_pairs[:10]:
                logging.info(f"    共享映射: {ckpt_key} -> {model_key}")
        if loaded_keys < total_keys:
            logging.info(f"  [INFO] 未加载参数: {total_keys - loaded_keys}")
            for model_key, ckpt_key in matched_pairs[:10]:
                if model_key != ckpt_key:
                    logging.info(f"    对齐映射: {ckpt_key} -> {model_key}")
        if shape_mismatch:
            logging.warning(f"  [WARN] shape不匹配参数: {len(shape_mismatch)}")
            for model_key, (ckpt_key, model_shape, ckpt_shape) in list(shape_mismatch.items())[:20]:
                logging.warning(f"    {model_key} <= {ckpt_key} | model{model_shape} vs ckpt{ckpt_shape}")
        if missing_key_models:
            logging.warning(f"  [WARN] checkpoint中缺失参数键: {len(missing_key_models)}")
            for model_key in missing_key_models[:20]:
                logging.warning(f"    缺失: {model_key}")
        if shape_mismatch and not missing_key_models:
            logging.warning("  [ROOT CAUSE] 主要问题为网络结构维度不一致（非键名前缀问题）")
        logging.info(f"  设备: {device}")

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"  总参数: {total_params:,}")
        logging.info(f"  可训练参数: {trainable_params:,}")

        return model

    def compute_reward(self, step_info: Dict) -> float:
        """
        重新设计的奖励函数 - 以流量为核心

        优先级调整（基于BC基础OCR=0.93的观察）：
        1. 流量奖励（权重4.5）- 最重要，直接提升吞吐量
        2. 稳定性奖励（权重3.0）- 减少波动和拥堵
        3. OCR奖励（权重2.0）- 目标提升到0.96-0.97
        4. 安全性惩罚（权重-2.5）- 控制碰撞和急停

        Args:
            step_info: 包含以下字段的字典
                - speeds: list of vehicle speeds
                - accelerations: list of vehicle accelerations
                - num_departed: 累计出发车辆数
                - num_arrived: 累计到达车辆数
                - num_active: 当前活跃车辆数
                - num_collisions: 累计碰撞数
                - num_emergency_stops: 急停次数
                - controlled_vehicles: 受控车辆列表

        Returns:
            reward: 标量奖励值
        """
        reward, _ = self.compute_reward_with_breakdown(step_info)
        return reward

    def compute_reward_with_breakdown(self, step_info: Dict) -> Tuple[float, Dict]:
        """
        计算奖励并返回详细分解

        Returns:
            reward: 总奖励
            breakdown: 各奖励组成部分的字典
        """
        speeds = step_info.get('speeds', [])
        accelerations = step_info.get('accelerations', [])

        num_departed = step_info.get('num_departed', 0)
        num_arrived = step_info.get('num_arrived', 0)
        num_active = step_info.get('num_active', 0)
        num_collisions = step_info.get('num_collisions', 0)
        num_emergency_stops = step_info.get('num_emergency_stops', 0)
        step_departed = step_info.get('step_departed', 0)
        step_arrived = step_info.get('step_arrived', 0)

        speed_limit = 13.89

        # 1. 流量奖励（强化吞吐）
        mean_speed = np.mean(speeds) if speeds else 0.0
        speed_reward = 2.2 * (mean_speed / speed_limit)
        traffic_reward = 0.8 * min(1.0, num_active / 350.0) if num_active > 0 else 0.0
        arrival_reward = 1.8 * min(1.0, step_arrived / 3.0)
        throughput_reward = speed_reward + traffic_reward + arrival_reward

        # 2. 稳定性奖励
        stability_speed_reward = 0.0
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            stability_speed_reward = 1.2 * max(0, 1.0 - speed_std / 6.0)

        stability_accel_reward = 0.0
        if accelerations and len(accelerations) > 0:
            mean_abs_accel = np.mean(np.abs(accelerations))
            stability_accel_reward = 1.0 * max(0, 1.0 - mean_abs_accel / 1.0)

        stability_reward = stability_speed_reward + stability_accel_reward

        # 3. OCR奖励（同时优化OCR水平与OCR增量）
        ocr_reward = 0.0
        current_ocr = 0.0
        ocr_delta = 0.0
        if num_departed > 0:
            current_ocr = num_arrived / num_departed
            prev_ocr = float(getattr(self, 'prev_ocr', current_ocr))
            ocr_delta = current_ocr - prev_ocr

            ocr_level_reward = 2.0 * max(0.0, (current_ocr - 0.93) / (0.97 - 0.93))
            ocr_level_reward = min(ocr_level_reward, 2.5)
            ocr_delta_reward = float(np.clip(6.0 * ocr_delta, -1.5, 1.5))
            ocr_reward = ocr_level_reward + ocr_delta_reward

        backlog_penalty = 0.0
        if num_departed > 0:
            backlog_ratio = max(0.0, (num_departed - num_arrived) / num_departed)
            backlog_penalty = -0.8 * backlog_ratio

        # 4. 安全性惩罚
        collision_penalty = -1.0 * num_collisions
        emergency_stop_penalty = -0.2 * num_emergency_stops

        slow_penalty = 0.0
        if speeds:
            slow_ratio = sum(1 for s in speeds if s < 3.0) / len(speeds)
            slow_penalty = -2.0 * slow_ratio

        jam_penalty = 0.0
        if speeds:
            jam_ratio = sum(1 for s in speeds if s < 5.0) / len(speeds)
            jam_penalty = -0.8 * jam_ratio

        safety_penalty = collision_penalty + emergency_stop_penalty + slow_penalty + jam_penalty

        # 分解
        breakdown = {
            'throughput_reward': throughput_reward,  # 流量（最重要）
            'speed_reward': speed_reward,
            'traffic_reward': traffic_reward,
            'arrival_reward': arrival_reward,
            'stability_reward': stability_reward,
            'stability_speed': stability_speed_reward,
            'stability_accel': stability_accel_reward,
            'ocr_reward': ocr_reward,
            'ocr_delta': ocr_delta,
            'current_ocr': current_ocr,
            'backlog_penalty': backlog_penalty,
            'safety_penalty': safety_penalty,
            'collision_penalty': collision_penalty,
            'emergency_stop_penalty': emergency_stop_penalty,
            'slow_penalty': slow_penalty,
            'jam_penalty': jam_penalty,
            'total': throughput_reward + stability_reward + ocr_reward + backlog_penalty + safety_penalty
        }

        self.prev_ocr = current_ocr

        return breakdown['total'], breakdown

    def _extract_vehicle_features(self, veh_ids: List[str]) -> np.ndarray:
        """提取车辆特征 [N, 8]（与BC训练保持一致）"""
        features = []
        for veh_id in veh_ids:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                lane_pos = traci.vehicle.getLanePosition(veh_id)
                lane_idx = traci.vehicle.getLaneIndex(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                accel = traci.vehicle.getAcceleration(veh_id)
                veh_type = traci.vehicle.getTypeID(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)

                feat = [
                    speed / 20.0,
                    lane_pos / 500.0,
                    lane_idx / 3.0,
                    waiting_time / 60.0,
                    accel / 5.0,
                    1.0 if veh_type == 'CV' else 0.0,
                    route_index / 10.0,
                    0.0
                ]
                features.append(feat)
            except Exception:
                continue

        if not features:
            return np.zeros((0, 8), dtype=np.float32)

        return np.array(features, dtype=np.float32)

    def _build_step_info(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """构建奖励所需的环境统计信息"""
        try:
            vehicle_ids = list(traci.vehicle.getIDList())
        except Exception:
            vehicle_ids = []

        speeds = []
        accelerations = []
        for veh_id in vehicle_ids:
            try:
                speeds.append(float(traci.vehicle.getSpeed(veh_id)))
                accelerations.append(float(traci.vehicle.getAcceleration(veh_id)))
            except Exception:
                continue

        # 使用仿真真实统计，累计到episode级别
        try:
            step_departed = int(traci.simulation.getDepartedNumber())
        except Exception:
            step_departed = 0

        try:
            step_arrived = int(traci.simulation.getArrivedNumber())
        except Exception:
            step_arrived = 0

        try:
            # 当前步发生碰撞的车辆数（非累计）
            num_collisions = int(traci.simulation.getCollidingVehiclesNumber())
        except Exception:
            num_collisions = 0

        try:
            # 当前步急停车辆数（非累计）
            num_emergency_stops = int(traci.simulation.getEmergencyStoppingVehiclesNumber())
        except Exception:
            num_emergency_stops = 0

        self.cumulative_departed += step_departed
        self.cumulative_arrived += step_arrived

        num_departed = self.cumulative_departed
        num_arrived = self.cumulative_arrived

        return {
            'speeds': speeds,
            'accelerations': accelerations,
            'num_departed': num_departed,
            'num_arrived': num_arrived,
            'step_departed': step_departed,
            'step_arrived': step_arrived,
            'num_active': len(vehicle_ids),
            'num_collisions': num_collisions,
            'num_emergency_stops': num_emergency_stops,
        }

    def _build_model_inputs(self, env: MultiAgentEnvironment, state_vectors: Dict[str, np.ndarray]):
        """
        构建模型输入（junction_observations + vehicle_observations）

        Args:
            env: SUMO环境
            state_vectors: 路口状态向量字典

        Returns:
            junction_observations: Dict[junc_id, torch.Tensor]
            vehicle_observations: Dict[junc_id, Dict[str, torch.Tensor]]
            controlled_map: Dict[junc_id, Dict[str, List[str]]]
        """
        junction_observations = {}
        vehicle_observations = {}
        controlled_map = {}

        try:
            for junc_id, state_vec in state_vectors.items():
                # 转换状态向量为tensor
                junction_observations[junc_id] = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

                # 获取该路口的agent
                if junc_id not in env.agents:
                    continue

                agent = env.agents[junc_id]

                # 获取受控车辆
                try:
                    controlled = agent.get_controlled_vehicles()
                    controlled_map[junc_id] = controlled

                    # 构建vehicle_observations
                    veh_obs_dict = {}
                    for veh_type in ['main', 'ramp', 'diverge']:
                        veh_ids = controlled.get(veh_type, [])
                        feats = self._extract_vehicle_features(veh_ids)
                        if len(feats) > 0:
                            veh_obs_dict[veh_type] = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)

                    vehicle_observations[junc_id] = veh_obs_dict
                except Exception:
                    # 如果获取失败，使用空字典
                    vehicle_observations[junc_id] = {}
                    controlled_map[junc_id] = {'main': [], 'ramp': [], 'diverge': []}

        except Exception as e:
            logging.warning(f"构建模型输入时出错: {e}")

        return junction_observations, vehicle_observations, controlled_map

    def collect_episode(self, env: MultiAgentEnvironment, max_steps: int = 3600) -> Dict:
        """
        收集一个episode的经验

        Args:
            env: SUMO环境
            max_steps: 最大步数

        Returns:
            episode_data: episode数据
        """
        self.model.eval()

        buffer = ExperienceBuffer()
        episode_reward = 0.0
        episode_rewards = []
        reward_breakdowns = []

        # 重置环境
        state_vectors = env.reset()  # Dict[junc_id, np.ndarray]
        self.cumulative_departed = 0
        self.cumulative_arrived = 0
        self.prev_ocr = 0.0

        for step in tqdm(range(max_steps), desc="收集episode"):
            # 构建模型输入（包含车辆观测）
            junction_observations, vehicle_observations, controlled_map = self._build_model_inputs(env, state_vectors)

            # 选择动作
            with torch.no_grad():
                env_actions, policy_actions, log_probs, values = self._select_actions(
                    junction_observations,
                    vehicle_observations,
                    controlled_map
                )

            # 执行动作
            next_state_vectors, rewards, dones, info = env.step(env_actions)

            # 计算奖励（带分解）
            step_info = self._build_step_info(info)
            control_values = []
            for junc_actions in env_actions.values():
                control_values.extend(list(junc_actions.values()))
            if control_values:
                control_array = np.array(control_values, dtype=np.float32)
                step_info['num_controlled'] = int(control_array.size)
                step_info['control_effort'] = float(np.mean(np.abs(control_array - 1.0)))
            else:
                step_info['num_controlled'] = 0
                step_info['control_effort'] = 0.0

            reward, breakdown = self.compute_reward_with_breakdown(step_info)
            episode_reward += reward
            episode_rewards.append(reward)
            reward_breakdowns.append(breakdown)

            # 存储经验（包含车辆状态与真实log_prob）
            for junc_id, state_tensor in junction_observations.items():
                if junc_id in policy_actions:
                    vehicle_state = {
                        veh_type: self._pin_if_cpu(tensor.detach().cpu())
                        for veh_type, tensor in vehicle_observations.get(junc_id, {}).items()
                    }
                    buffer.add(
                        junction_id=junc_id,
                        state=self._pin_if_cpu(state_tensor.squeeze(0).detach().cpu()),
                        vehicle_state=vehicle_state,
                        action={
                            key: self._pin_if_cpu(value.detach().cpu())
                            for key, value in policy_actions[junc_id].items()
                        },
                        reward=reward,
                        value=values.get(junc_id, 0.0),
                        log_prob=log_probs.get(junc_id, 0.0),
                        done=dones
                    )

            state_vectors = next_state_vectors

            if dones:
                break

        # 统计平均奖励分解
        avg_breakdown = {}
        if reward_breakdowns:
            for key in reward_breakdowns[0].keys():
                avg_breakdown[key] = np.mean([b[key] for b in reward_breakdowns])

        # 日志输出
        logging.info(f"[Episode] 总奖励: {episode_reward:.2f}")
        logging.info(f"[Episode] 平均奖励分解（新版本）:")
        logging.info(f"  【流量奖励】总计: {avg_breakdown.get('throughput_reward', 0):.4f} ⭐")
        logging.info(f"    - 速度奖励: {avg_breakdown.get('speed_reward', 0):.4f}")
        logging.info(f"    - 活跃车辆: {avg_breakdown.get('traffic_reward', 0):.4f}")
        logging.info(f"  【稳定性奖励】总计: {avg_breakdown.get('stability_reward', 0):.4f} ⭐")
        logging.info(f"    - 速度标准差: {avg_breakdown.get('stability_speed', 0):.4f}")
        logging.info(f"    - 加速度稳定: {avg_breakdown.get('stability_accel', 0):.4f}")
        logging.info(f"  【OCR奖励】: {avg_breakdown.get('ocr_reward', 0):.4f} (目标0.965)")
        logging.info(f"  【安全性惩罚】总计: {avg_breakdown.get('safety_penalty', 0):.4f}")
        logging.info(f"    - 碰撞: {avg_breakdown.get('collision_penalty', 0):.4f}")
        logging.info(f"    - 急停: {avg_breakdown.get('emergency_stop_penalty', 0):.4f}")
        logging.info(f"    - 慢速车辆: {avg_breakdown.get('slow_penalty', 0):.4f}")
        logging.info(f"    - 拥堵: {avg_breakdown.get('jam_penalty', 0):.4f}")

        # 统计
        episode_data = {
            'buffer': buffer,
            'total_reward': episode_reward,
            'mean_reward': np.mean(episode_rewards),
            'rewards': episode_rewards,
            'reward_breakdowns': reward_breakdowns,
            'avg_breakdown': avg_breakdown,
            'length': len(episode_rewards)
        }

        return episode_data

    def _select_actions(
        self,
        junction_observations: Dict,
        vehicle_observations: Dict,
        controlled_map: Dict[str, Dict[str, List[str]]]
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        选择动作（车辆级模型）

        Args:
            junction_observations: Dict[junc_id, torch.Tensor]
            vehicle_observations: Dict[junc_id, Dict[str, torch.Tensor]]

        Returns:
            env_actions: {junction_id: {veh_id: speed_ratio}}
            policy_actions: {junction_id: model_action_tensors}
            log_probs: {junction_id: log_prob}
            values: {junction_id: value}
        """
        env_actions = {}
        policy_actions = {}
        log_probs = {}
        values = {}

        def _risk_score(feat: np.ndarray, veh_type: str) -> int:
            if feat is None or len(feat) < 7:
                return 0
            speed_norm = float(feat[0])
            waiting_norm = float(feat[3])
            dist_to_end_norm = float(feat[6])
            score = 0
            if dist_to_end_norm <= self.near_merge_threshold:
                score += 1
            if speed_norm <= 0.45:
                score += 1
            if waiting_norm >= 0.25:
                score += 1
            if veh_type in ('ramp', 'diverge'):
                if speed_norm <= 0.50:
                    score += 1
            else:
                if speed_norm <= 0.40:
                    score += 1
            return score

        def _is_high_conflict_risk(feat: np.ndarray, veh_type: str) -> bool:
            threshold = 3 if veh_type in ('ramp', 'diverge') else 4
            return _risk_score(feat, veh_type) >= threshold

        def _should_release(feat: np.ndarray) -> bool:
            if feat is None or len(feat) < 7:
                return True
            speed_norm = float(feat[0])
            dist_to_end_norm = float(feat[6])
            return (speed_norm >= self.release_speed_ratio and dist_to_end_norm > 0.6) or dist_to_end_norm > 1.0

        # 模型推理
        with torch.no_grad():
            all_actions, all_values, all_info = self.model(
                junction_observations,
                vehicle_observations,
                deterministic=False  # 采样
            )

        # 提取动作和values
        for junc_id, output in all_actions.items():
            junc_env_actions = {}
            controlled = controlled_map.get(junc_id, {'main': [], 'ramp': [], 'diverge': []})

            for veh_type in ['main', 'ramp', 'diverge']:
                action_key = f'{veh_type}_actions'
                veh_ids = controlled.get(veh_type, [])
                veh_actions = output.get(action_key)
                veh_feats = vehicle_observations.get(junc_id, {}).get(veh_type)

                if veh_actions is None:
                    continue

                if veh_actions.dim() > 1:
                    veh_actions = veh_actions.squeeze(0)

                if torch.is_tensor(veh_feats):
                    if veh_feats.dim() == 3:
                        veh_feats = veh_feats.squeeze(0)
                    veh_feats_np = veh_feats.detach().cpu().numpy()
                else:
                    veh_feats_np = None

                n = min(len(veh_ids), veh_actions.shape[0])
                for i in range(n):
                    feat = veh_feats_np[i] if veh_feats_np is not None and i < len(veh_feats_np) else None
                    if _should_release(feat) or (not _is_high_conflict_risk(feat, veh_type)):
                        continue

                    action_val = float(torch.clamp(veh_actions[i], 0.0, 1.0).item())
                    action_val = max(action_val, self.min_speed_ratio)
                    junc_env_actions[veh_ids[i]] = action_val

            env_actions[junc_id] = junc_env_actions
            policy_actions[junc_id] = {
                k: v for k, v in output.items()
                if k in ['main_actions', 'ramp_actions', 'diverge_actions'] and v is not None
            }

            values[junc_id] = all_values.get(junc_id, 0.0).item() if torch.is_tensor(all_values.get(junc_id, 0.0)) else all_values.get(junc_id, 0.0)
            log_prob_sum = 0.0
            info = all_info.get(junc_id, {})
            for key in ['main_log_probs', 'ramp_log_probs', 'diverge_log_probs']:
                val = info.get(key)
                if torch.is_tensor(val):
                    log_prob_sum += float(val.sum().item())
            log_probs[junc_id] = log_prob_sum

        return env_actions, policy_actions, log_probs, values

    def _build_padded_type_batch(
        self,
        samples: List[Dict],
        veh_type: str,
        action_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建某车辆类型的padding batch [B, N, ...] + mask"""
        veh_tensors = []
        act_tensors = []
        max_len = 0

        for sample in samples:
            veh = sample['vehicle_state'].get(veh_type)
            act = sample['action'].get(action_key)

            if veh is None or act is None:
                veh_len = 0
                veh = None
                act = None
            else:
                if veh.dim() == 3:
                    veh = veh.squeeze(0)
                if act.dim() > 1:
                    act = act.squeeze(0)
                veh_len = int(min(veh.size(0), act.size(0)))

            max_len = max(max_len, veh_len)
            veh_tensors.append((veh, veh_len))
            act_tensors.append((act, veh_len))

        if max_len == 0:
            return None, None, None

        batch_size = len(samples)
        feat_dim = 8

        vehicles_batch = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        actions_batch = torch.zeros(batch_size, max_len, dtype=torch.float32)
        mask_batch = torch.zeros(batch_size, max_len, dtype=torch.float32)

        for i in range(batch_size):
            veh, n_v = veh_tensors[i]
            act, n_a = act_tensors[i]
            n = min(n_v, n_a)
            if n <= 0:
                continue
            vehicles_batch[i, :n] = veh[:n].float()
            actions_batch[i, :n] = act[:n].float()
            mask_batch[i, :n] = 1.0

        if self.device.startswith('cuda'):
            vehicles_batch = vehicles_batch.pin_memory()
            actions_batch = actions_batch.pin_memory()
            mask_batch = mask_batch.pin_memory()

        return (
            self._to_device_async(vehicles_batch),
            self._to_device_async(actions_batch),
            self._to_device_async(mask_batch),
        )

    def _evaluate_batch(self, samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量评估一批sample（按junction分组，组内向量化）"""
        if not samples:
            zero = torch.zeros(0, device=self.device)
            return zero, zero, zero

        grouped = {}
        for idx, sample in enumerate(samples):
            grouped.setdefault(sample['junction_id'], []).append((idx, sample))

        batch_size = len(samples)
        logp_out = torch.zeros(batch_size, device=self.device)
        value_out = torch.zeros(batch_size, device=self.device)
        entropy_out = torch.zeros(batch_size, device=self.device)

        for junc_id, idx_samples in grouped.items():
            network = self.model.networks[junc_id]
            local_indices = [pair[0] for pair in idx_samples]
            local_samples = [pair[1] for pair in idx_samples]

            states_cpu = torch.stack([s['state'].float() for s in local_samples], dim=0)
            if self.device.startswith('cuda'):
                states_cpu = states_cpu.pin_memory()
            states = self._to_device_async(states_cpu)

            main_pack = self._build_padded_type_batch(local_samples, 'main', 'main_actions')
            ramp_pack = self._build_padded_type_batch(local_samples, 'ramp', 'ramp_actions')
            diverge_pack = self._build_padded_type_batch(local_samples, 'diverge', 'diverge_actions')

            veh_obs = {
                'main': main_pack[0] if main_pack[0] is not None else None,
                'ramp': ramp_pack[0] if ramp_pack[0] is not None else None,
                'diverge': diverge_pack[0] if diverge_pack[0] is not None else None,
            }
            act_dict = {
                'main_actions': main_pack[1] if main_pack[1] is not None else None,
                'ramp_actions': ramp_pack[1] if ramp_pack[1] is not None else None,
                'diverge_actions': diverge_pack[1] if diverge_pack[1] is not None else None,
            }
            mask_dict = {
                'main': main_pack[2] if main_pack[2] is not None else None,
                'ramp': ramp_pack[2] if ramp_pack[2] is not None else None,
                'diverge': diverge_pack[2] if diverge_pack[2] is not None else None,
            }

            eval_out = network.evaluate_actions_batched(states, veh_obs, act_dict, mask_dict)
            idx_tensor = torch.tensor(local_indices, dtype=torch.long, device=self.device)
            logp_out[idx_tensor] = eval_out['log_prob_sum']
            value_out[idx_tensor] = eval_out['value']
            entropy_out[idx_tensor] = eval_out['entropy_mean']

        return logp_out, value_out, entropy_out

    def update_policy(self, buffer: ExperienceBuffer) -> Dict:
        """
        更新策略（PPO）

        Args:
            buffer: 经验缓冲区

        Returns:
            metrics: 训练指标
        """
        self.model.train()

        if len(buffer) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
            }

        # 计算GAE
        advantages, returns = self._compute_gae(buffer)

        samples = []
        for junc_id in buffer.states:
            n = len(buffer.states[junc_id])
            for i in range(n):
                samples.append({
                    'junction_id': junc_id,
                    'state': buffer.states[junc_id][i],
                    'vehicle_state': buffer.vehicle_states[junc_id][i],
                    'action': buffer.actions[junc_id][i],
                    'old_log_prob': float(buffer.log_probs[junc_id][i]),
                    'old_value': float(buffer.values[junc_id][i]),
                    'advantage': float(advantages[junc_id][i]),
                    'return': float(returns[junc_id][i]),
                })

        if not samples:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
            }

        adv_tensor = torch.tensor([s['advantage'] for s in samples], dtype=torch.float32, device=self.device)
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std(unbiased=False) + 1e-8
        for idx, sample in enumerate(samples):
            sample['advantage'] = float(((adv_tensor[idx] - adv_mean) / adv_std).item())

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_anchor_loss = 0.0
        num_updates = 0

        # 多个epoch更新
        batch_size = min(self.config.batch_size, len(samples))
        for epoch in range(self.config.n_epochs):
            random.shuffle(samples)

            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]

                new_log_probs, new_values, entropies = self._evaluate_batch(batch)

                old_log_probs = torch.tensor(
                    [sample['old_log_prob'] for sample in batch],
                    dtype=torch.float32,
                    device=self.device
                )
                old_values = torch.tensor(
                    [sample['old_value'] for sample in batch],
                    dtype=torch.float32,
                    device=self.device
                )
                advantages_batch = torch.tensor(
                    [sample['advantage'] for sample in batch],
                    dtype=torch.float32,
                    device=self.device
                )
                returns_batch = torch.tensor(
                    [sample['return'] for sample in batch],
                    dtype=torch.float32,
                    device=self.device
                )

                # return标准化（批内）
                returns_batch = (returns_batch - returns_batch.mean()) / (returns_batch.std(unbiased=False) + 1e-8)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages_batch

                policy_loss = -torch.min(surr1, surr2).mean()
                # PPO式value clipping + Huber损失
                values_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon
                )
                value_loss_unclipped = nn.functional.smooth_l1_loss(new_values, returns_batch, reduction='none')
                value_loss_clipped = nn.functional.smooth_l1_loss(values_clipped, returns_batch, reduction='none')
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy = entropies.mean()
                if self.anchor_coef > 0.0:
                    anchor_loss = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        anchor_param = self.bc_anchor_params.get(name)
                        if anchor_param is None:
                            continue
                        anchor_loss = anchor_loss + nn.functional.mse_loss(param, anchor_param, reduction='mean')
                else:
                    anchor_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.anchor_coef * anchor_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_anchor_loss += float(anchor_loss.item())
                num_updates += 1

        metrics = {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'anchor_loss': total_anchor_loss / max(num_updates, 1),
        }

        return metrics

    def _compute_gae(self, buffer: ExperienceBuffer) -> Tuple[Dict, Dict]:
        """
        计算GAE优势函数

        Returns:
            advantages: {junction_id: advantages}
            returns: {junction_id: returns}
        """
        advantages = {}
        returns = {}

        for junc_id in buffer.states:
            rewards = buffer.rewards.get(junc_id, [])
            values = buffer.values.get(junc_id, [])

            if not rewards:
                advantages[junc_id] = []
                returns[junc_id] = []
                continue

            dones = [False] * len(rewards)
            dones[-1] = True

            values_ext = values + [0.0]
            gae = 0.0
            adv_junc = [0.0] * len(rewards)
            ret_junc = [0.0] * len(rewards)

            for i in reversed(range(len(rewards))):
                mask = 0.0 if dones[i] else 1.0
                delta = rewards[i] + self.config.gamma * values_ext[i + 1] * mask - values_ext[i]
                gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
                adv_junc[i] = gae
                ret_junc[i] = gae + values_ext[i]

            advantages[junc_id] = adv_junc
            returns[junc_id] = ret_junc

        return advantages, returns

    def train(self, n_episodes: int, max_steps: int = 3600):
        """
        训练循环

        Args:
            n_episodes: 训练episodes数
            max_steps: 每个episode最大步数
        """
        logging.info("=" * 70)
        logging.info("开始PPO微调训练")
        logging.info("=" * 70)
        logging.info(f"Episodes: {n_episodes}")
        logging.info(f"Max steps per episode: {max_steps}")
        logging.info(f"Learning rate: {self.config.lr}")
        logging.info(f"Gamma: {self.config.gamma}")
        logging.info(f"GAE lambda: {self.config.gae_lambda}")
        logging.info(f"Clip epsilon: {self.config.clip_epsilon}")
        logging.info("=" * 70 + "\n")

        # 创建环境
        env = MultiAgentEnvironment(
            junction_ids=list(JUNCTION_CONFIGS.keys()),
            sumo_cfg='sumo/sumo.sumocfg',
            use_gui=False,
            seed=self.config.seed
        )

        for episode in range(1, n_episodes + 1):
            logging.info(f"\n{'='*20} Episode {episode}/{n_episodes} {'='*20}")

            # 收集经验
            episode_data = self.collect_episode(env, max_steps)

            logging.info(f"[Episode {episode}] 总奖励: {episode_data['total_reward']:.2f}")
            logging.info(f"[Episode {episode}] 平均奖励: {episode_data['mean_reward']:.4f}")
            logging.info(f"[Episode {episode}] 长度: {episode_data['length']}")

            # 更新策略
            if len(episode_data['buffer']) > 0:
                metrics = self.update_policy(episode_data['buffer'])

                logging.info(f"[Episode {episode}] Policy loss: {metrics['policy_loss']:.4f}")
                logging.info(f"[Episode {episode}] Value loss: {metrics['value_loss']:.4f}")
                logging.info(f"[Episode {episode}] Entropy: {metrics['entropy']:.4f}")
                logging.info(f"[Episode {episode}] Anchor loss: {metrics['anchor_loss']:.6f}")

                # TensorBoard记录
                self.writer.add_scalar('Reward/total', episode_data['total_reward'], episode)
                self.writer.add_scalar('Reward/mean', episode_data['mean_reward'], episode)
                self.writer.add_scalar('Loss/policy', metrics['policy_loss'], episode)
                self.writer.add_scalar('Loss/value', metrics['value_loss'], episode)
                self.writer.add_scalar('Loss/anchor', metrics['anchor_loss'], episode)
                self.writer.add_scalar('Entropy', metrics['entropy'], episode)

            # 熵系数衰减（低风险：仅影响探索强度）
            self.entropy_coef = max(
                self.config.entropy_min,
                self.entropy_coef * self.config.entropy_decay
            )
            self.anchor_coef = max(self.anchor_min, self.anchor_coef * self.anchor_decay)
            self.writer.add_scalar('Entropy/coef', self.entropy_coef, episode)
            self.writer.add_scalar('Anchor/coef', self.anchor_coef, episode)
            logging.info(f"[Episode {episode}] Entropy coef: {self.entropy_coef:.6f}")
            logging.info(f"[Episode {episode}] Anchor coef: {self.anchor_coef:.8f}")

            # 保存最佳模型
            if episode_data['total_reward'] > self.best_reward:
                self.best_reward = episode_data['total_reward']
                self.no_improve_count = 0
                self._save_checkpoint(episode, episode_data['total_reward'], 'best_model.pt')
                logging.info(f"✓ 保存最佳模型 (reward={episode_data['total_reward']:.2f})")
            else:
                self.no_improve_count += 1

            # 定期保存
            if episode % 10 == 0:
                self._save_checkpoint(episode, episode_data['total_reward'], f'checkpoint_ep{episode}.pt')

            if self.early_stop_patience > 0 and self.no_improve_count >= self.early_stop_patience:
                logging.info(f"[早停] 连续{self.no_improve_count}个episode无提升，提前停止训练")
                break

        logging.info("\n" + "=" * 70)
        logging.info("训练完成!")
        logging.info("=" * 70)
        logging.info(f"最佳奖励: {self.best_reward:.2f}")

        # 清理环境
        try:
            env.close()
            logging.info("✓ 环境已关闭")
        except Exception as e:
            logging.warning(f"关闭环境时出错: {e}")

        self.writer.close()

    def _save_checkpoint(self, episode: int, reward: float, filename: str):
        """保存checkpoint"""
        save_path = os.path.join(self.log_dir, filename)

        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward': reward,
            'best_reward': self.best_reward,
            'config': self.config.__dict__,
        }, save_path)

        logging.info(f"✓ Checkpoint已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='完整PPO微调训练')
    parser.add_argument('--bc-checkpoint', type=str, required=True,
                        help='BC模型checkpoint路径')
    parser.add_argument('--output-dir', type=str, default='ppo_finetune_checkpoints',
                        help='输出目录')
    parser.add_argument('--log-dir', type=str, default='./logs/ppo_finetune',
                        help='日志目录')
    parser.add_argument('--episodes', type=int, default=100,
                        help='训练episodes数')
    parser.add_argument('--max-steps', type=int, default=3600,
                        help='每个episode最大步数')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='学习率（微调用较小值）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--anchor-coef', type=float, default=1e-5,
                        help='BC参数锚定正则系数')
    parser.add_argument('--anchor-decay', type=float, default=0.999,
                        help='BC参数锚定正则衰减')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='连续无提升的早停耐心值，<=0表示关闭')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 配置PPO（微调专用）
    config = PPOConfig(
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.1,  # 更保守的clip
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        batch_size=2048,
        n_epochs=8,
        update_frequency=2048,
        seed=args.seed
    )

    # 创建微调器
    finetuner = PPOFinetuner(
        bc_checkpoint_path=args.bc_checkpoint,
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        anchor_coef=args.anchor_coef,
        anchor_decay=args.anchor_decay,
        early_stop_patience=args.early_stop_patience,
    )

    # 开始训练
    finetuner.train(
        n_episodes=args.episodes,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
