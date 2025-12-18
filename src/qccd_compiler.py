"""
QCCD Compiler: DAG → Executable QCCD Native Circuit

作業フロー:
  Stim Circuit
       ↓
  QCCD Native (non-executable) ... 移動操作なし、位置情報あり
       ↓
  DAG (依存関係グラフ)
       ↓  ← このモジュール
  QCCD Native (executable) ... 移動操作が挿入済み

コンパイラの責務:
  1. DAGからFront Layerを取得
  2. 実行可能なゲートを選択（必要なイオンがLIZにいるか確認）
  3. 必要に応じて移動操作(SHUTTLE)を挿入
  4. 全ゲートが実行されるまで繰り返し
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from ast_def import (
    Location, QubitDecl, Operation,
    GateOp, ShuttleOp, MeasureOp, ResetOp, Program
)
from dag_builder import DAG, DAGNode


# ==========================================
# QCCD Architecture (入力として与える構成)
# ==========================================

@dataclass
class Zone:
    """ゾーンの定義"""
    id: int
    zone_type: str      # "LIZ" or "Memory"
    capacity: int       # 最大イオン数
    position: int       # 1D配置での位置 (0, 1, 2, ...)


@dataclass
class QCCDArchitecture:
    """
    QCCDアーキテクチャの構成定義
    
    将来的には外部ファイル(JSON/YAML)から読み込む想定
    """
    zones: List[Zone]
    
    def __post_init__(self):
        self._zone_by_id: Dict[int, Zone] = {z.id: z for z in self.zones}
        self._liz_zones: List[Zone] = [z for z in self.zones if z.zone_type == "LIZ"]
        self._memory_zones: List[Zone] = [z for z in self.zones if z.zone_type == "Memory"]
    
    def get_zone(self, zone_id: int) -> Optional[Zone]:
        return self._zone_by_id.get(zone_id)
    
    def get_liz(self) -> Zone:
        """LIZを取得（現在は単一LIZを想定）"""
        if not self._liz_zones:
            raise ValueError("No LIZ defined in architecture")
        return self._liz_zones[0]
    
    def get_memory_zones(self) -> List[Zone]:
        return self._memory_zones
    
    @classmethod
    def simple_1d(cls, num_memory_zones: int = 2, liz_capacity: int = 4, memory_capacity: int = 10):
        """
        シンプルな1Dアーキテクチャを生成
        
        Layout: [Memory0] - [LIZ] - [Memory1] - ...
        """
        zones = []
        # LIZを中央に配置
        zones.append(Zone(id=0, zone_type="LIZ", capacity=liz_capacity, position=num_memory_zones))
        # Memoryゾーンを両側に配置
        for i in range(num_memory_zones):
            zones.append(Zone(
                id=i + 1, 
                zone_type="Memory", 
                capacity=memory_capacity,
                position=i if i < num_memory_zones // 2 else num_memory_zones + 1 + (i - num_memory_zones // 2)
            ))
        return cls(zones=zones)


# ==========================================
# QCCD State (実行時に変化する状態)
# ==========================================

@dataclass
class QCCDState:
    """
    QCCDの実行時状態
    
    イオンの位置を追跡し、ゲート実行可能性を判定
    """
    architecture: QCCDArchitecture
    ion_locations: Dict[int, int] = field(default_factory=dict)  # ion_id -> zone_id
    zone_contents: Dict[int, List[int]] = field(default_factory=dict)  # zone_id -> [ion_ids]
    
    def __post_init__(self):
        # 各ゾーンの内容を初期化
        for zone in self.architecture.zones:
            if zone.id not in self.zone_contents:
                self.zone_contents[zone.id] = []
    
    def place_ion(self, ion_id: int, zone_id: int):
        """イオンを指定ゾーンに配置（初期配置用）"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} does not exist")
        if len(self.zone_contents[zone_id]) >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")
        
        self.ion_locations[ion_id] = zone_id
        self.zone_contents[zone_id].append(ion_id)
    
    def locate(self, ion_id: int) -> int:
        """イオンの現在のゾーンIDを取得"""
        if ion_id not in self.ion_locations:
            raise ValueError(f"Ion {ion_id} not found")
        return self.ion_locations[ion_id]
    
    def is_in_liz(self, ion_id: int) -> bool:
        """イオンがLIZにいるか"""
        zone_id = self.locate(ion_id)
        zone = self.architecture.get_zone(zone_id)
        return zone is not None and zone.zone_type == "LIZ"
    
    def can_execute(self, gate: GateOp) -> bool:
        """ゲートが実行可能か（必要な全イオンがLIZにいるか）"""
        return all(self.is_in_liz(ion_id) for ion_id in gate.targets)
    
    def liz_has_space(self) -> bool:
        """LIZに空きがあるか"""
        liz = self.architecture.get_liz()
        return len(self.zone_contents[liz.id]) < liz.capacity
    
    def get_liz_ions(self) -> List[int]:
        """LIZにいるイオンのリスト"""
        liz = self.architecture.get_liz()
        return self.zone_contents[liz.id].copy()
    
    def move_ion(self, ion_id: int, dst_zone_id: int):
        """イオンを移動（状態を更新）"""
        src_zone_id = self.locate(ion_id)
        dst_zone = self.architecture.get_zone(dst_zone_id)
        
        if dst_zone is None:
            raise ValueError(f"Destination zone {dst_zone_id} does not exist")
        if len(self.zone_contents[dst_zone_id]) >= dst_zone.capacity:
            raise ValueError(f"Destination zone {dst_zone_id} is at capacity")
        
        # 移動を実行
        self.zone_contents[src_zone_id].remove(ion_id)
        self.zone_contents[dst_zone_id].append(ion_id)
        self.ion_locations[ion_id] = dst_zone_id


# ==========================================
# Compiler
# ==========================================

@dataclass
class CompileResult:
    """コンパイル結果"""
    operations: List[Operation]  # 実行順の操作列
    stats: Dict[str, int]        # 統計情報


class QCCDCompiler:
    """
    DAG → Executable な操作列へコンパイル
    
    アルゴリズム (プロトタイプ版 - 最適化なし):
    1. Front Layerから実行可能なゲートを探す
    2. 実行可能なゲートがあれば実行
    3. なければ、必要なイオンをLIZへ移動（SHUTTLE挿入）
    4. 全ゲート実行まで繰り返し
    """
    
    def __init__(self, architecture: QCCDArchitecture):
        self.architecture = architecture
    
    def compile(self, dag: DAG, initial_placement: Dict[int, int]) -> CompileResult:
        """
        DAGをコンパイルしてexecutableな操作列を生成
        
        Args:
            dag: 依存関係グラフ
            initial_placement: ion_id -> zone_id の初期配置
        
        Returns:
            CompileResult: 操作列と統計情報
        """
        # 状態を初期化
        state = QCCDState(architecture=self.architecture)
        for ion_id, zone_id in initial_placement.items():
            state.place_ion(ion_id, zone_id)
        
        # DAGの実行状態をリセット
        dag.reset_execution()
        
        # 結果を格納
        operations: List[Operation] = []
        stats = {"gates": 0, "shuttles": 0, "iterations": 0}
        
        while not dag.is_complete():
            stats["iterations"] += 1
            front_layer = dag.get_front_layer()
            
            # Step 1: 実行可能なゲートを探す
            executable_node = self._find_executable_gate(dag, front_layer, state)
            
            if executable_node is not None:
                # Step 2: ゲートを実行
                operations.append(executable_node.operation)
                dag.execute(executable_node.id)
                stats["gates"] += 1
            else:
                # Step 3: 移動操作を挿入
                shuttle_ops = self._generate_shuttle_ops(dag, front_layer, state)
                for shuttle_op in shuttle_ops:
                    operations.append(shuttle_op)
                    state.move_ion(shuttle_op.target, self.architecture.get_liz().id)
                    stats["shuttles"] += 1
        
        return CompileResult(operations=operations, stats=stats)
    
    def _find_executable_gate(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> Optional[DAGNode]:
        """
        Front Layerから実行可能なゲートを探す
        
        プロトタイプ版: 最小IDのものを選択（決定的な動作のため）
        """
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if isinstance(node.operation, GateOp):
                if state.can_execute(node.operation):
                    return node
            elif isinstance(node.operation, ResetOp):
                # RESETもLIZで実行が必要と仮定
                if all(state.is_in_liz(ion_id) for ion_id in node.operation.targets):
                    return node
        return None
    
    def _generate_shuttle_ops(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> List[ShuttleOp]:
        """
        必要な移動操作を生成
        
        プロトタイプ版: Front Layerの最小IDノードに必要なイオンをLIZへ移動
        TODO: LIZの容量制限を考慮（イオンの退避が必要な場合）
        """
        # 最小IDのゲートを対象にする
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if isinstance(node.operation, (GateOp, ResetOp)):
                targets = node.operation.targets
                shuttle_ops = []
                
                for ion_id in targets:
                    if not state.is_in_liz(ion_id):
                        # LIZへのシャトル操作を生成
                        liz = self.architecture.get_liz()
                        # TODO: LIZが満杯の場合の処理
                        shuttle_ops.append(ShuttleOp(
                            target=ion_id,
                            dst=Location(trap=liz.id, site=0)  # site=0は仮
                        ))
                
                if shuttle_ops:
                    return shuttle_ops
        
        return []


# ==========================================
# 動作確認
# ==========================================

if __name__ == "__main__":
    from qccd_parser import parse
    from dag_builder import DAGBuilder
    
    # テスト用の簡単な回路
    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 2 AT (2, 0)

    RESET 0
    RESET 1
    RESET 2
    
    R 0 1.57 0.0
    R 1 1.57 0.0
    
    MS 0 1 0.25
    MS 1 2 0.25
    """
    
    print("=" * 60)
    print("QCCD Compiler Test (Prototype)")
    print("=" * 60)
    
    # パースとDAG構築
    program = parse(code)
    builder = DAGBuilder(program)
    dag = builder.build()
    
    print("\n--- Architecture ---")
    arch = QCCDArchitecture.simple_1d(num_memory_zones=2, liz_capacity=4)
    for zone in arch.zones:
        print(f"  {zone}")
    
    print("\n--- Initial Placement ---")
    # Ion 0, 1 をLIZ(zone 0)に、Ion 2 をMemory(zone 1)に配置
    initial_placement = {0: 0, 1: 0, 2: 1}
    for ion_id, zone_id in initial_placement.items():
        print(f"  Ion {ion_id} -> Zone {zone_id}")
    
    print("\n--- Compilation ---")
    compiler = QCCDCompiler(arch)
    result = compiler.compile(dag, initial_placement)
    
    print("\nGenerated Operations:")
    for i, op in enumerate(result.operations):
        print(f"  [{i}] {op}")
    
    print(f"\nStatistics: {result.stats}")