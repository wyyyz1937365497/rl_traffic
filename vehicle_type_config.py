"""
车辆类型归一化配置
自动生成，请勿手动编辑
"""

import torch

# 车辆类型及其最大速度
VEHICLE_TYPE_MAXSPEED = {
    'CV': 30.0000,
    'DEFAULT_BIKETYPE': 13.8889,
    'DEFAULT_CONTAINERTYPE': 55.5556,
    'DEFAULT_PEDTYPE': 10.4389,
    'DEFAULT_RAILTYPE': 44.4444,
    'DEFAULT_TAXITYPE': 55.5556,
    'DEFAULT_VEHTYPE': 55.5556,
    'HV': 30.0000,
    'penetration0.05': 30.0000,
}

# 全局最大速度（用于归一化）
GLOBAL_MAX_SPEED = 55.5556


def normalize_speed(speed, max_speed=None):
    """
    归一化速度到 [0, 1]

    Args:
        speed: 速度值 (m/s)
        max_speed: 该车辆类型的最大速度，如果为None则使用全局最大速度

    Returns:
        归一化后的速度 [0, 1]
    """
    if max_speed is None:
        max_speed = GLOBAL_MAX_SPEED

    return torch.clamp(speed / max_speed, 0.0, 1.0)


def denormalize_speed(normalized_speed, max_speed=None):
    """
    反归一化速度

    Args:
        normalized_speed: 归一化速度 [0, 1]
        max_speed: 该车辆类型的最大速度，如果为None则使用全局最大速度

    Returns:
        实际速度 (m/s)
    """
    if max_speed is None:
        max_speed = GLOBAL_MAX_SPEED

    return torch.clamp(normalized_speed * max_speed, 0.0, max_speed)


def get_vehicle_max_speed(vehicle_type):
    """
    获取车辆类型的最大速度

    Args:
        vehicle_type: 车辆类型ID

    Returns:
        该类型的最大速度 (m/s)，如果类型不存在则返回全局最大速度
    """
    return VEHICLE_TYPE_MAXSPEED.get(vehicle_type, GLOBAL_MAX_SPEED)


def normalize_batch_speeds(speeds, max_speeds):
    """
    批量归一化速度

    Args:
        speeds: 速度张量 [N]
        max_speeds: 最大速度列表 [N]

    Returns:
        归一化速度 [N]
    """
    max_speeds = torch.tensor(max_speeds, device=speeds.device, dtype=speeds.dtype)
    return torch.clamp(speeds / max_speeds, 0.0, 1.0)


def denormalize_batch_speeds(normalized_speeds, max_speeds):
    """
    批量反归一化速度

    Args:
        normalized_speeds: 归一化速度张量 [N]
        max_speeds: 最大速度列表或张量 [N]

    Returns:
        实际速度张量 [N]
    """
    max_speeds = torch.tensor(max_speeds, device=normalized_speeds.device, dtype=normalized_speeds.dtype)
    return torch.clamp(normalized_speeds * max_speeds, 0.0, max_speeds)
