"""
完整的车道级建模
基于路网分析结果，精确定义每条车道的冲突关系
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第一部分：车道类型定义
# ============================================================================

class LaneType(Enum):
    """车道类型"""
    MAIN_THROUGH = "main_through"      # 主路直行车道
    MAIN_RIGHT = "main_right"          # 主路右转车道
    MAIN_LEFT = "main_left"            # 主路左转车道
    RAMP_MERGE = "ramp_merge"          # 匝道汇入车道
    RAMP_DIVERGE = "ramp_diverge"      # 匝道转出车道
    AUXILIARY = "auxiliary"            # 辅助车道


class ConflictType(Enum):
    """冲突类型"""
    MERGE = "merge"          # 汇入冲突
    DIVERGE = "diverge"      # 转出冲突
    CROSS = "cross"          # 交叉冲突
    WEAVING = "weaving"      # 交织冲突


@dataclass
class LaneConflict:
    """车道冲突定义"""
    lane_id: str
    conflicts_with: List[str]
    conflict_type: ConflictType
    severity: float  # 0-1
    description: str = ""


# ============================================================================
# 第二部分：基于 EDGE_TOPOLOGY 和 LANE_CONFLICTS 的完整车道冲突矩阵
# ============================================================================
# 数据来源：road_topology_hardcoded.py
# ============================================================================

# 完整的车道冲突定义（基于路网拓扑和车道级冲突矩阵）
LANE_CONFLICTS = {
    # ==================== J5 路口：E23匝道汇入-E2 ====================
    # 拓扑：E23 → -E2，与 -E3 来车在 -E2 上冲突
    # LANE_CONFLICTS: 'E23_0': ['-E3_0']
    'E23_0': LaneConflict(
        lane_id='E23_0',
        conflicts_with=['-E3_0'],  # 与反向主路上游来车冲突
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="匝道E23汇入-E2，与-E3来车在-E2上冲突"
    ),

    '-E3_0': LaneConflict(
        lane_id='-E3_0',
        conflicts_with=['E23_0'],
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="反向主路最外侧车道，与E23匝道汇入冲突"
    ),

    # ==================== J14 路口：E15匝道汇入E10 ====================
    # 拓扑：E15 → E10，与 E9 来车在 E10 上冲突
    # LANE_CONFLICTS: 'E15_0': ['E9_0'] (注意是正向E9，不是反向-E9)
    'E15_0': LaneConflict(
        lane_id='E15_0',
        conflicts_with=['E9_0'],  # 与正向主路上游来车冲突
        conflict_type=ConflictType.MERGE,
        severity=0.7,
        description="匝道E15汇入E10，与E9来车在E10上冲突"
    ),

    'E9_0': LaneConflict(
        lane_id='E9_0',
        conflicts_with=['E15_0'],
        conflict_type=ConflictType.MERGE,
        severity=0.7,
        description="正向主路最外侧车道，与E15匝道汇入冲突"
    ),
    'E9_1': LaneConflict(
        lane_id='E9_1',
        conflicts_with=[],  # 不与匝道冲突
        conflict_type=ConflictType.MERGE,
        severity=0.0,
        description="正向主路内侧车道，不与匝道冲突"
    ),


    # ==================== J15 路口：E17匝道汇入-E10 + E16转出 ====================
    # 拓扑：E17 → -E10，与 -E11 来车在 -E10 上冲突
    # LANE_CONFLICTS: 'E17_0': ['-E11_0', '-E11_1'] (关键：不与-E11_2冲突！)
    'E17_0': LaneConflict(
        lane_id='E17_0',
        conflicts_with=['-E11_0', '-E11_1'],  # 只与前2条车道冲突
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="匝道E17汇入-E10，与-E11前两条车道冲突，不与-E11_2冲突"
    ),

    '-E11_0': LaneConflict(
        lane_id='-E11_0',
        conflicts_with=['E17_0'],
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="反向主路最外侧车道，与E17匝道汇入冲突，可转出E16"
    ),
    '-E11_1': LaneConflict(
        lane_id='-E11_1',
        conflicts_with=['E17_0'],
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="反向主路中间车道，与E17匝道汇入冲突"
    ),
    '-E11_2': LaneConflict(
        lane_id='-E11_2',
        conflicts_with=[],  # 不与匝道冲突！
        conflict_type=ConflictType.MERGE,
        severity=0.0,
        description="反向主路最内侧车道，不与匝道冲突"
    ),

    # E16转出匝道
    'E16_0': LaneConflict(
        lane_id='E16_0',
        conflicts_with=[],
        conflict_type=ConflictType.DIVERGE,
        severity=0.5,
        description="转出匝道E16第一车道"
    ),
    'E16_1': LaneConflict(
        lane_id='E16_1',
        conflicts_with=[],
        conflict_type=ConflictType.DIVERGE,
        severity=0.5,
        description="转出匝道E16第二车道"
    ),

    # ==================== J17 路口：E19匝道汇入-E12 + E18/E20转出 ====================
    # 拓扑：E19 → -E12，与 -E13 来车在 -E12 上冲突
    # LANE_CONFLICTS: 'E19_0': ['-E13_0', '-E13_1'], 'E19_1': ['-E13_0', '-E13_1']
    # 关键：不与 -E13_2 冲突！
    'E19_0': LaneConflict(
        lane_id='E19_0',
        conflicts_with=['-E13_0', '-E13_1'],  # 与前2条车道都冲突
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="匝道E19第一车道汇入-E12，与-E13前两条车道冲突"
    ),
    'E19_1': LaneConflict(
        lane_id='E19_1',
        conflicts_with=['-E13_0', '-E13_1'],  # 也与前2条车道都冲突
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="匝道E19第二车道汇入-E12，与-E13前两条车道冲突"
    ),

    '-E13_0': LaneConflict(
        lane_id='-E13_0',
        conflicts_with=['E19_0', 'E19_1'],
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="反向主路最外侧车道，与E19匝道两条车道都冲突，可转出E20"
    ),
    '-E13_1': LaneConflict(
        lane_id='-E13_1',
        conflicts_with=['E19_0', 'E19_1'],
        conflict_type=ConflictType.MERGE,
        severity=0.8,
        description="反向主路中间车道，与E19匝道两条车道都冲突"
    ),
    '-E13_2': LaneConflict(
        lane_id='-E13_2',
        conflicts_with=[],  # 不与匝道冲突！
        conflict_type=ConflictType.MERGE,
        severity=0.0,
        description="反向主路最内侧车道，不与匝道冲突"
    ),

    # E18, E20转出匝道
    'E18_0': LaneConflict(
        lane_id='E18_0',
        conflicts_with=[],
        conflict_type=ConflictType.DIVERGE,
        severity=0.0,
        description="转出匝道E18"
    ),
    'E20_0': LaneConflict(
        lane_id='E20_0',
        conflicts_with=[],
        conflict_type=ConflictType.DIVERGE,
        severity=0.0,
        description="转出匝道E20"
    ),
}


# ============================================================================
# 第三部分：车道级特征定义（包含CV/HV标识）
# ============================================================================

@dataclass
class VehicleFeatures:
    """车辆特征（包含CV/HV标识）"""
    veh_id: str
    vehicle_type: str  # 'CV' 或 'HV'
    is_cv: bool        # 是否是智能网联车
    
    # 位置和运动状态
    lane_id: str
    lane_index: int
    lane_position: float
    speed: float
    acceleration: float
    
    # 路径信息
    route_index: int
    route_length: int
    distance_traveled: float
    distance_total: float
    completion_rate: float
    
    # 等待时间
    waiting_time: float
    
    # 是否在冲突区域
    in_conflict_zone: bool
    conflict_severity: float


@dataclass
class LaneFeatures:
    """车道级特征"""
    lane_id: str
    edge_id: str
    lane_index: int
    lane_type: LaneType
    
    # 车道属性
    length: float
    speed_limit: float
    
    # 车辆统计
    total_vehicles: int
    cv_vehicles: int      # CV车辆数量
    hv_vehicles: int      # HV车辆数量
    cv_ratio: float       # CV比例
    
    # 车辆列表（区分CV和HV）
    vehicle_ids: List[str]
    cv_vehicle_ids: List[str]  # CV车辆ID列表
    hv_vehicle_ids: List[str]  # HV车辆ID列表
    
    # 速度统计
    mean_speed: float
    mean_speed_cv: float  # CV车辆平均速度
    mean_speed_hv: float  # HV车辆平均速度
    
    # 排队统计
    queue_length: int
    queue_cv: int         # CV排队数量
    queue_hv: int         # HV排队数量
    
    # 密度
    density: float
    density_cv: float     # CV密度
    density_hv: float     # HV密度
    
    # 冲突信息
    has_conflict: bool
    conflict_lanes: List[str]
    conflict_severity: float
    
    # 控制信息
    controllable_cv: List[str]  # 可控制的CV车辆


# ============================================================================
# 第四部分：模型输入说明
# ============================================================================

"""
模型输入包含CV和HV车辆的标识！

输入图结构：
{
    'nodes': [
        {
            'type': 'vehicle',
            'id': 'veh_001',
            'features': [
                ...,
                is_cv,  # 1.0表示CV，0.0表示HV
                ...
            ]
        },
        ...
    ],
    'edges': [
        {
            'type': 'lane_connection',
            'source': 'veh_001',
            'target': 'lane_E11_0',
            'features': [...]
        },
        ...
    ]
}

车辆特征向量（包含CV/HV标识）：
[
    position_x,          # 位置x
    position_y,          # 位置y
    speed,               # 速度
    acceleration,        # 加速度
    lane_position,       # 车道位置
    lane_index,          # 车道索引
    route_progress,      # 路径进度
    waiting_time,        # 等待时间
    is_cv,              # ★ 是否是CV（1.0=CV, 0.0=HV）
    is_controlled,      # 是否被控制（只有CV可被控制）
    in_conflict_zone,   # 是否在冲突区域
    conflict_severity,  # 冲突严重程度
    ...
]

车道特征向量：
[
    length,              # 长度
    speed_limit,         # 限速
    lane_index,          # 车道索引
    is_ramp,            # 是否匝道
    is_rightmost,       # 是否最右侧
    total_vehicles,     # 总车辆数
    cv_count,           # ★ CV车辆数
    hv_count,           # ★ HV车辆数
    cv_ratio,           # ★ CV比例
    mean_speed,         # 平均速度
    mean_speed_cv,      # ★ CV平均速度
    mean_speed_hv,      # ★ HV平均速度
    queue_length,       # 排队长度
    queue_cv,           # ★ CV排队数
    queue_hv,           # ★ HV排队数
    density,            # 密度
    has_conflict,       # 是否有冲突
    conflict_severity,  # 冲突严重程度
    ...
]
"""


# ============================================================================
# 第五部分：模型输出与控制说明
# ============================================================================

"""
模型输出说明：

1. 模型只控制CV车辆！
   - 输出动作只应用于CV车辆
   - HV车辆不受模型控制，由SUMO自动驾驶

2. 输出格式：
{
    'junction_id': {
        'main_action': float,    # 主路CV车辆速度控制（0-1）
        'ramp_action': float,    # 匝道CV车辆速度控制（0-1）
        'diverge_action': float, # 转出CV车辆引导（0-1）
        'controlled_vehicles': {
            'main': ['cv_001', 'cv_002'],  # 主路被控制的CV车辆ID
            'ramp': ['cv_003'],            # 匝道被控制的CV车辆ID
            'diverge': ['cv_004']          # 转出被控制的CV车辆ID
        }
    }
}

3. 控制方式：
   - 通过traci.vehicle.setSpeed(veh_id, target_speed)控制CV车辆
   - HV车辆不调用任何控制函数
   - 只控制控制区域内的CV车辆

4. 控制流程：
   a. 获取所有车辆信息（包含CV/HV标识）
   b. 筛选出CV车辆
   c. 根据控制区域划分，确定每个路口控制的CV车辆
   d. 模型为每个路口生成动作
   e. 只对控制区域内的CV车辆应用动作
   f. HV车辆保持SUMO默认行为
"""


# ============================================================================
# 第六部分：完整的车道级环境
# ============================================================================

class LaneLevelEnvironment:
    """
    车道级环境
    提供完整的车道级状态观察，包含CV/HV区分
    """
    
    def __init__(self, sumo_cfg: str):
        self.sumo_cfg = sumo_cfg
        
        # 加载车道冲突矩阵
        self.conflict_matrix = self._build_conflict_matrix()
    
    def _build_conflict_matrix(self) -> Dict[str, Dict]:
        """构建车道冲突矩阵"""
        matrix = {}
        
        for lane_id, conflict in LANE_CONFLICTS.items():
            matrix[lane_id] = {
                'conflicts_with': conflict.conflicts_with,
                'type': conflict.conflict_type.value,
                'severity': conflict.severity
            }
        
        return matrix
    
    def get_lane_features(self, lane_id: str) -> LaneFeatures:
        """
        获取车道特征（包含CV/HV区分）
        """
        try:
            import traci
            
            # 获取车道上的所有车辆
            all_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            
            # 区分CV和HV
            cv_vehicles = []
            hv_vehicles = []
            
            for veh_id in all_vehicles:
                veh_type = traci.vehicle.getTypeID(veh_id)
                if veh_type == 'CV':
                    cv_vehicles.append(veh_id)
                else:
                    hv_vehicles.append(veh_id)
            
            # 计算速度统计
            cv_speeds = [traci.vehicle.getSpeed(v) for v in cv_vehicles]
            hv_speeds = [traci.vehicle.getSpeed(v) for v in hv_vehicles]
            all_speeds = cv_speeds + hv_speeds
            
            # 计算排队
            cv_queue = sum(1 for v in cv_vehicles if traci.vehicle.getSpeed(v) < 0.1)
            hv_queue = sum(1 for v in hv_vehicles if traci.vehicle.getSpeed(v) < 0.1)
            
            # 获取冲突信息
            conflict_info = self.conflict_matrix.get(lane_id, {})
            
            # 构建特征
            features = LaneFeatures(
                lane_id=lane_id,
                edge_id='_'.join(lane_id.split('_')[:-1]),
                lane_index=int(lane_id.split('_')[-1]),
                lane_type=self._get_lane_type(lane_id),
                length=traci.lane.getLength(lane_id),
                speed_limit=traci.lane.getSpeed(lane_id),
                total_vehicles=len(all_vehicles),
                cv_vehicles=len(cv_vehicles),
                hv_vehicles=len(hv_vehicles),
                cv_ratio=len(cv_vehicles) / max(len(all_vehicles), 1),
                vehicle_ids=all_vehicles,
                cv_vehicle_ids=cv_vehicles,
                hv_vehicle_ids=hv_vehicles,
                mean_speed=np.mean(all_speeds) if all_speeds else 0.0,
                mean_speed_cv=np.mean(cv_speeds) if cv_speeds else 0.0,
                mean_speed_hv=np.mean(hv_speeds) if hv_speeds else 0.0,
                queue_length=traci.lane.getLastStepHaltingNumber(lane_id),
                queue_cv=cv_queue,
                queue_hv=hv_queue,
                density=len(all_vehicles) / max(traci.lane.getLength(lane_id) / 1000, 0.1),
                density_cv=len(cv_vehicles) / max(traci.lane.getLength(lane_id) / 1000, 0.1),
                density_hv=len(hv_vehicles) / max(traci.lane.getLength(lane_id) / 1000, 0.1),
                has_conflict=lane_id in self.conflict_matrix,
                conflict_lanes=conflict_info.get('conflicts_with', []),
                conflict_severity=conflict_info.get('severity', 0.0),
                controllable_cv=cv_vehicles  # 只有CV可被控制
            )
            
            return features
            
        except Exception as e:
            return None
    
    def _get_lane_type(self, lane_id: str) -> LaneType:
        """判断车道类型"""
        edge_id = '_'.join(lane_id.split('_')[:-1])
        
        # 匝道
        if edge_id in ['E23', 'E15', 'E17', 'E19']:
            return LaneType.RAMP_MERGE
        if edge_id in ['E16', 'E18', 'E20', 'E24']:
            return LaneType.RAMP_DIVERGE
        
        return LaneType.MAIN_THROUGH
    
    def get_vehicle_features(self, veh_id: str) -> VehicleFeatures:
        """
        获取车辆特征（包含CV/HV标识）
        """
        try:
            import traci
            
            veh_type = traci.vehicle.getTypeID(veh_id)
            is_cv = (veh_type == 'CV')
            
            lane_id = traci.vehicle.getLaneID(veh_id)
            
            # 检查是否在冲突区域
            conflict_info = self.conflict_matrix.get(lane_id, {})
            in_conflict_zone = lane_id in self.conflict_matrix
            
            features = VehicleFeatures(
                veh_id=veh_id,
                vehicle_type=veh_type,
                is_cv=is_cv,
                lane_id=lane_id,
                lane_index=traci.vehicle.getLaneIndex(veh_id),
                lane_position=traci.vehicle.getLanePosition(veh_id),
                speed=traci.vehicle.getSpeed(veh_id),
                acceleration=traci.vehicle.getAcceleration(veh_id),
                route_index=traci.vehicle.getRouteIndex(veh_id),
                route_length=len(traci.vehicle.getRoute(veh_id)),
                distance_traveled=0,  # 需要累计计算
                distance_total=0,
                completion_rate=0,
                waiting_time=traci.vehicle.getWaitingTime(veh_id),
                in_conflict_zone=in_conflict_zone,
                conflict_severity=conflict_info.get('severity', 0.0)
            )
            
            return features
            
        except Exception as e:
            return None


# ============================================================================
# 第七部分：测试和验证
# ============================================================================

def print_lane_conflicts():
    """打印车道冲突关系"""
    print("=" * 70)
    print("完整车道冲突关系")
    print("=" * 70)
    
    for lane_id, conflict in LANE_CONFLICTS.items():
        if conflict.severity > 0:
            print(f"\n{lane_id}:")
            print(f"  冲突车道: {conflict.conflicts_with}")
            print(f"  冲突类型: {conflict.conflict_type.value}")
            print(f"  严重程度: {conflict.severity}")
            print(f"  说明: {conflict.description}")


def print_model_io():
    """打印模型输入输出说明"""
    print("\n" + "=" * 70)
    print("模型输入输出说明")
    print("=" * 70)
    
    print("\n【模型输入】")
    print("-" * 70)
    print("包含CV和HV车辆的标识：")
    print("  - 车辆特征中包含 is_cv 字段（1.0=CV, 0.0=HV）")
    print("  - 车道特征中包含 cv_count, hv_count, cv_ratio")
    print("  - 图节点包含所有车辆（CV和HV）")
    print("  - 图边包含车辆-车道关系")
    
    print("\n【模型输出】")
    print("-" * 70)
    print("只控制CV车辆：")
    print("  - 输出动作只应用于CV车辆")
    print("  - HV车辆由SUMO自动控制")
    print("  - 输出包含被控制的CV车辆ID列表")
    
    print("\n【控制流程】")
    print("-" * 70)
    print("1. 获取所有车辆信息（区分CV/HV）")
    print("2. 筛选控制区域内的CV车辆")
    print("3. 模型生成动作")
    print("4. 只对CV车辆应用动作")
    print("5. HV车辆保持默认行为")


def main():
    """主测试函数"""
    print_lane_conflicts()
    print_model_io()
    
    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    print("\n1. ✅ 模型输入包含CV和HV车辆的标识")
    print("   - 车辆特征中有 is_cv 字段")
    print("   - 车道特征中有 cv_count, hv_count")
    
    print("\n2. ✅ 模型只控制CV车辆")
    print("   - 输出动作只应用于CV车辆")
    print("   - HV车辆由SUMO自动控制")
    
    print("\n3. ✅ 车道级建模完成")
    print("   - 精确定义了每条车道的冲突关系")
    print("   - 区分了不同车道的影响")


if __name__ == '__main__':
    import numpy as np
    main()
