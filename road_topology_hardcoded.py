"""
硬编码的路网拓扑配置
从net.xml解析生成，无需运行时动态计算
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class EdgeInfo:
    """边信息"""
    edge_id: str
    from_junction: str
    to_junction: str
    num_lanes: int
    downstream: List[str]  # 下游边
    upstream: List[str]    # 上游边
    is_ramp: bool          # 是否是匝道
    is_main: bool          # 是否是主路


# ============================================================================
# 完整的边拓扑配置（硬编码）
# ============================================================================

EDGE_TOPOLOGY: Dict[str, EdgeInfo] = {
    # ==================== 主路正向 ====================
    'E1': EdgeInfo(
        edge_id='E1',
        from_junction='',
        to_junction='',
        num_lanes=2,
        downstream=['E2'],
        upstream=[],
        is_ramp=False,
        is_main=True
    ),
    'E2': EdgeInfo(
        edge_id='E2',
        from_junction='',
        to_junction='J5',
        num_lanes=2,
        downstream=['E3'],
        upstream=['E1'],
        is_ramp=False,
        is_main=True
    ),
    'E3': EdgeInfo(
        edge_id='E3',
        from_junction='J5',
        to_junction='',
        num_lanes=2,
        downstream=['E5'],
        upstream=['E2'],
        is_ramp=False,
        is_main=True
    ),
    'E5': EdgeInfo(
        edge_id='E5',
        from_junction='',
        to_junction='',
        num_lanes=2,
        downstream=['E6'],
        upstream=['E3'],
        is_ramp=False,
        is_main=True
    ),
    'E6': EdgeInfo(
        edge_id='E6',
        from_junction='',
        to_junction='J11',
        num_lanes=2,
        downstream=['E7'],
        upstream=['E5'],
        is_ramp=False,
        is_main=True
    ),
    'E7': EdgeInfo(
        edge_id='E7',
        from_junction='J11',
        to_junction='J12',
        num_lanes=3,
        downstream=['E8','E24'],
        upstream=['E6'],
        is_ramp=False,
        is_main=True
    ),
    'E8': EdgeInfo(
        edge_id='E8',
        from_junction='J12',
        to_junction='',
        num_lanes=2,
        downstream=['E9'],
        upstream=['E7'],
        is_ramp=False,
        is_main=True
    ),
    'E9': EdgeInfo(
        edge_id='E9',
        from_junction='',
        to_junction='J14',
        num_lanes=2,
        downstream=['E10'],
        upstream=['E8'],
        is_ramp=False,
        is_main=True
    ),
    'E10': EdgeInfo(
        edge_id='E10',
        from_junction='J14',
        to_junction='J15',
        num_lanes=2,
        downstream=['E11'],  # 可直行或转出
        upstream=['E9', 'E15'],
        is_ramp=False,
        is_main=True
    ),
    'E11': EdgeInfo(
        edge_id='E11',
        from_junction='J15',
        to_junction='',
        num_lanes=3,
        downstream=['E12'],
        upstream=['E10'],
        is_ramp=False,
        is_main=True
    ),
    'E12': EdgeInfo(
        edge_id='E12',
        from_junction='',
        to_junction='J17',
        num_lanes=3,
        downstream=['E13', 'E18'],  # 可直行或转出
        upstream=['E11'],
        is_ramp=False,
        is_main=True
    ),
    'E13': EdgeInfo(
        edge_id='E13',
        from_junction='J17',
        to_junction='J18',
        num_lanes=2,
        downstream=[],
        upstream=['E12'],
        is_ramp=False,
        is_main=True
    ),
    
    # ==================== 主路反向 ====================
    '-E1': EdgeInfo(
        edge_id='-E1',
        from_junction='',
        to_junction='',
        num_lanes=2,
        downstream=[''],
        upstream=['-E2'],  # E23匝道汇入
        is_ramp=False,
        is_main=True
    ),
    '-E2': EdgeInfo(
        edge_id='-E2',
        from_junction='J5',
        to_junction='',
        num_lanes=2,
        downstream=['-E1'],
        upstream=['-E3', 'E23'],  # E23匝道汇入
        is_ramp=False,
        is_main=True
    ),
    '-E3': EdgeInfo(
        edge_id='-E3',
        from_junction='',
        to_junction='J5',
        num_lanes=2,
        downstream=['-E2'],
        upstream=['-E5'],
        is_ramp=False,
        is_main=True
    ),
    '-E5': EdgeInfo(
        edge_id='-E5',
        from_junction='',
        to_junction='',
        num_lanes=2,
        downstream=['-E3'],
        upstream=['-E6'],
        is_ramp=False,
        is_main=True
    ),
    '-E6': EdgeInfo(
        edge_id='-E6',
        from_junction='J11',
        to_junction='',
        num_lanes=2,
        downstream=['-E5'],
        upstream=['-E7'],
        is_ramp=False,
        is_main=True
    ),
    '-E7': EdgeInfo(
        edge_id='-E7',
        from_junction='J12',
        to_junction='J11',
        num_lanes=2,
        downstream=['-E6'],
        upstream=['-E8'],
        is_ramp=False,
        is_main=True
    ),
    '-E8': EdgeInfo(
        edge_id='-E8',
        from_junction='',
        to_junction='J12',
        num_lanes=2,
        downstream=['-E7'],
        upstream=['-E9'],
        is_ramp=False,
        is_main=True
    ),
    '-E9': EdgeInfo(
        edge_id='-E9',
        from_junction='J14',
        to_junction='',
        num_lanes=2,
        downstream=['-E8'],
        upstream=['-E10'],  # E15匝道汇入
        is_ramp=False,
        is_main=True
    ),
    '-E10': EdgeInfo(
        edge_id='-E10',
        from_junction='J15',
        to_junction='J14',
        num_lanes=2,
        downstream=['-E9'],
        upstream=['-E11', 'E17'],  # E17匝道汇入
        is_ramp=False,
        is_main=True
    ),
    '-E11': EdgeInfo(
        edge_id='-E11',
        from_junction='',
        to_junction='J15',
        num_lanes=3,
        downstream=['-E10', 'E16'],  # 可直行或转出
        upstream=['-E12'],
        is_ramp=False,
        is_main=True
    ),
    '-E12': EdgeInfo(
        edge_id='-E12',
        from_junction='J17',
        to_junction='',
        num_lanes=3,
        downstream=['-E11'],
        upstream=['-E13', 'E19'],  # E19匝道汇入
        is_ramp=False,
        is_main=True
    ),
    '-E13': EdgeInfo(
        edge_id='-E13',
        from_junction='',
        to_junction='J17',
        num_lanes=3,
        downstream=['-E12', 'E20'],  # 可直行或转出
        upstream=[],
        is_ramp=False,
        is_main=True
    ),
    
    # ==================== 匝道（汇入）====================
    'E23': EdgeInfo(
        edge_id='E23',
        from_junction='',
        to_junction='J5',
        num_lanes=1,
        downstream=['-E2'],  # 汇入主路
        upstream=[],
        is_ramp=True,
        is_main=False
    ),
    'E15': EdgeInfo(
        edge_id='E15',
        from_junction='',
        to_junction='J14',
        num_lanes=1,
        downstream=['E10'],  # 汇入主路
        upstream=[],
        is_ramp=True,
        is_main=False
    ),
    'E17': EdgeInfo(
        edge_id='E17',
        from_junction='',
        to_junction='J15',
        num_lanes=1,
        downstream=['-E10'],  # 可转出或汇入
        upstream=[],
        is_ramp=True,
        is_main=False
    ),
    'E19': EdgeInfo(
        edge_id='E19',
        from_junction='',
        to_junction='J17',
        num_lanes=2,
        downstream=['-E12'],  # 可转出或汇入
        upstream=[],
        is_ramp=True,
        is_main=False
    ),
    
    # ==================== 匝道（转出）====================
    'E16': EdgeInfo(
        edge_id='E16',
        from_junction='J15',
        to_junction='',
        num_lanes=2,
        downstream=[],
        upstream=['-E11'],
        is_ramp=True,
        is_main=False
    ),
    'E18': EdgeInfo(
        edge_id='E18',
        from_junction='J17',
        to_junction='',
        num_lanes=1,
        downstream=[],
        upstream=['E12'],
        is_ramp=True,
        is_main=False
    ),
    'E20': EdgeInfo(
        edge_id='E20',
        from_junction='J17',
        to_junction='',
        num_lanes=1,
        downstream=[],
        upstream=['-E13'],
        is_ramp=True,
        is_main=False
    ),
}


# ============================================================================
# 车道级冲突矩阵（硬编码）
# ============================================================================

LANE_CONFLICTS: Dict[str, List[str]] = {
    # J5: E23匝道汇入-E2
    'E23_0': ['-E3_0'],  # 匝道与主路最外侧两条车道冲突
    
    # J14: E15匝道汇入-E9
    'E15_0': ['E9_0'],
    
    # J15: E17匝道汇入-E10（关键：只与前2条车道冲突）
    'E17_0': ['-E11_0', '-E11_1'],  # 不与-E11_2冲突！
    
    # J17: E19匝道汇入-E12（关键：只与前2条车道冲突）
    'E19_0': ['-E13_0', '-E13_1'],  # 不与-E13_2冲突！
    'E19_1': ['-E13_0', '-E13_1'],  # 第二条匝道车道只与最外侧冲突
}


# ============================================================================
# 交叉口配置（硬编码）
# ============================================================================

JUNCTION_CONFIG: Dict[str, Dict] = {
    'J5': {
        'type': 'simple_merge',
        'main_incoming': ['E2'],
        'main_outgoing': ['E3'],
        'ramp_incoming': ['E23'],
        'main_reverse': ['-E3'],
        'num_main_lanes': 2,
        'num_ramp_lanes': 1,
        'has_traffic_light': True,
        'conflict_edges': {
            'ramp': 'E23',
            'main': '-E3',
            'conflict_lanes': ['-E3_0', '-E3_1']
        }
    },
    'J14': {
        'type': 'simple_merge',
        'main_incoming': ['E9'],
        'main_outgoing': ['E10'],
        'ramp_incoming': ['E15'],
        'main_reverse': ['-E10'],
        'num_main_lanes': 2,
        'num_ramp_lanes': 1,
        'has_traffic_light': True,
        'conflict_edges': {
            'ramp': 'E15',
            'main': '-E10',
            'conflict_lanes': ['-E10_0', '-E10_1']
        }
    },
    'J15': {
        'type': 'complex_merge_diverge',
        'main_incoming': ['E10'],
        'main_outgoing': ['E11'],
        'ramp_incoming': ['E17'],
        'ramp_outgoing': ['E16'],
        'main_reverse': ['-E11'],
        'num_main_lanes': 3,
        'num_ramp_lanes': 1,
        'has_traffic_light': True,
        'conflict_edges': {
            'ramp': 'E17',
            'main': '-E11',
            'conflict_lanes': ['-E11_0', '-E11_1'],  # 不与-E11_2冲突
            'diverge': 'E16'
        }
    },
    'J17': {
        'type': 'high_conflict',
        'main_incoming': ['E12'],
        'main_outgoing': ['E13'],
        'ramp_incoming': ['E19'],
        'ramp_outgoing': ['E18', 'E20'],
        'main_reverse': ['-E13'],
        'num_main_lanes': 3,
        'num_ramp_lanes': 2,
        'has_traffic_light': True,
        'conflict_edges': {
            'ramp': 'E19',
            'main': '-E13',
            'conflict_lanes': ['-E13_0', '-E13_1'],  # 不与-E13_2冲突
            'diverge': ['E18', 'E20']
        }
    }
}


# ============================================================================
# 快速查询函数
# ============================================================================

def get_downstream_edges(edge_id: str) -> List[str]:
    """获取下游边"""
    if edge_id in EDGE_TOPOLOGY:
        return EDGE_TOPOLOGY[edge_id].downstream
    return []


def get_upstream_edges(edge_id: str) -> List[str]:
    """获取上游边"""
    if edge_id in EDGE_TOPOLOGY:
        return EDGE_TOPOLOGY[edge_id].upstream
    return []


def are_edges_connected(edge1: str, edge2: str) -> bool:
    """检查两条边是否连接"""
    if edge1 in EDGE_TOPOLOGY:
        return edge2 in EDGE_TOPOLOGY[edge1].downstream
    return False


def get_conflict_lanes(lane_id: str) -> List[str]:
    """获取冲突车道"""
    return LANE_CONFLICTS.get(lane_id, [])


def is_ramp_edge(edge_id: str) -> bool:
    """判断是否是匝道"""
    if edge_id in EDGE_TOPOLOGY:
        return EDGE_TOPOLOGY[edge_id].is_ramp
    return False


def get_junction_config(junction_id: str) -> Dict:
    """获取交叉口配置"""
    return JUNCTION_CONFIG.get(junction_id, {})


def get_all_edges() -> List[str]:
    """获取所有边"""
    return list(EDGE_TOPOLOGY.keys())


def get_main_edges() -> List[str]:
    """获取所有主路边"""
    return [eid for eid, info in EDGE_TOPOLOGY.items() if info.is_main]


def get_ramp_edges() -> List[str]:
    """获取所有匝道边"""
    return [eid for eid, info in EDGE_TOPOLOGY.items() if info.is_ramp]


# ============================================================================
# 图构建辅助函数
# ============================================================================

def build_edge_adjacency_matrix() -> Dict[str, Set[str]]:
    """
    构建边邻接矩阵
    
    Returns:
        {edge_id: set(connected_edge_ids)}
    """
    adjacency = defaultdict(set)
    
    for edge_id, info in EDGE_TOPOLOGY.items():
        # 下游连接
        for downstream in info.downstream:
            adjacency[edge_id].add(downstream)
        
        # 上游连接
        for upstream in info.upstream:
            adjacency[edge_id].add(upstream)
    
    return adjacency


def get_edge_connection_type(edge1: str, edge2: str) -> str:
    """
    获取边连接类型
    
    Returns:
        'through': 直行
        'merge': 汇入
        'diverge': 转出
        'none': 不连接
    """
    if not are_edges_connected(edge1, edge2):
        return 'none'
    
    info1 = EDGE_TOPOLOGY.get(edge1)
    info2 = EDGE_TOPOLOGY.get(edge2)
    
    if info1 and info2:
        # 匝道到主路：汇入
        if info1.is_ramp and info2.is_main:
            return 'merge'
        
        # 主路到匝道：转出
        if info1.is_main and info2.is_ramp:
            return 'diverge'
        
        # 主路到主路：直行
        if info1.is_main and info2.is_main:
            return 'through'
    
    return 'none'


# ============================================================================
# 打印拓扑信息
# ============================================================================

def print_topology_info():
    """打印拓扑信息"""
    print("=" * 70)
    print("硬编码路网拓扑配置")
    print("=" * 70)
    
    print(f"\n总边数: {len(EDGE_TOPOLOGY)}")
    print(f"主路边数: {len(get_main_edges())}")
    print(f"匝道边数: {len(get_ramp_edges())}")
    
    print("\n【关键交叉口配置】")
    for junc_id, config in JUNCTION_CONFIG.items():
        print(f"\n{junc_id} ({config['type']}):")
        print(f"  主路上游: {config['main_incoming']}")
        print(f"  匝道上游: {config['ramp_incoming']}")
        print(f"  冲突车道: {config['conflict_edges']['conflict_lanes']}")
    
    print("\n【车道冲突矩阵】")
    for lane_id, conflict_lanes in LANE_CONFLICTS.items():
        print(f"  {lane_id} → {conflict_lanes}")


if __name__ == '__main__':
    print_topology_info()
