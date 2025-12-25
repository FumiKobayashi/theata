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
from typing import List, Dict, Set, Optional, Tuple, Tuple
from ast_def import (
    Location, QubitDecl, Operation,
    GateOp, ShuttleOp, SwapOp, MeasureOp, ResetOp, Program
)
from dag_builder import DAG, DAGNode


# ==========================================
# QCCD Architecture (入力として与える構成)
# ==========================================

@dataclass
class Zone:
    """ゾーンの定義"""
    id: int
    zone_type: str      # "L" (LIZ) or "M" (Memory)
    capacity: int       # 最大イオン数
    position: int       # 1D配置での位置 (リストのインデックス)


class QCCDArchitecture:
    """
    1D QCCDアーキテクチャの構成定義
    
    入力形式: [(type, capacity), ...]
      - type: "M" (Memory) or "L" (LIZ)
      - capacity: そのゾーンの最大イオン数
      - 位置はリストのインデックスで自動決定
      - 隣接関係は index ± 1 で自動決定
    
    例:
      QCCDArchitecture([("M", 15), ("L", 2), ("L", 2), ("M", 15)])
      → Memory - LIZ - LIZ - Memory の4ゾーン構成
    """
    
    def __init__(self, zone_specs: List[Tuple[str, int]]):
        """
        Args:
            zone_specs: [(type, capacity), ...] のリスト
                       type は "M" (Memory) or "L" (LIZ)
        """
        self.zones: List[Zone] = []
        for i, (zone_type, capacity) in enumerate(zone_specs):
            if zone_type not in ("M", "L"):
                raise ValueError(f"Invalid zone type '{zone_type}'. Must be 'M' or 'L'.")
            self.zones.append(Zone(
                id=i,
                zone_type=zone_type,
                capacity=capacity,
                position=i
            ))
        
        self._zone_by_id: Dict[int, Zone] = {z.id: z for z in self.zones}
        self._liz_zones: List[Zone] = [z for z in self.zones if z.zone_type == "L"]
        self._memory_zones: List[Zone] = [z for z in self.zones if z.zone_type == "M"]
        
        if not self._liz_zones:
            raise ValueError("Architecture must have at least one LIZ")
    
    def get_zone(self, zone_id: int) -> Optional[Zone]:
        return self._zone_by_id.get(zone_id)
    
    def get_liz_zones(self) -> List[Zone]:
        """全LIZを取得"""
        return self._liz_zones
    
    def get_memory_zones(self) -> List[Zone]:
        """全Memoryゾーンを取得"""
        return self._memory_zones
    
    def get_adjacent_zones(self, zone_id: int) -> List[int]:
        """隣接するゾーンIDのリストを取得（1Dなので最大2つ）"""
        adjacent = []
        if zone_id > 0:
            adjacent.append(zone_id - 1)
        if zone_id < len(self.zones) - 1:
            adjacent.append(zone_id + 1)
        return adjacent
    
    def is_liz(self, zone_id: int) -> bool:
        """指定ゾーンがLIZかどうか"""
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "L"
    
    def find_nearest_memory(self, zone_id: int) -> Optional[int]:
        """指定ゾーンから最も近いMemoryゾーンを探す"""
        # BFS で探索
        visited = {zone_id}
        queue = [zone_id]
        
        while queue:
            current = queue.pop(0)
            zone = self.get_zone(current)
            if zone and zone.zone_type == "M":
                return current
            
            for adj in self.get_adjacent_zones(current):
                if adj not in visited:
                    visited.add(adj)
                    queue.append(adj)
        
        return None
    
    def __repr__(self):
        layout = " - ".join(f"{z.zone_type}({z.capacity})" for z in self.zones)
        return f"QCCDArchitecture([{layout}])"
    
    @classmethod
    def simple_mlm(cls, liz_capacity: int = 4, memory_capacity: int = 15):
        """シンプルな Memory-LIZ-Memory 構成"""
        return cls([("M", memory_capacity), ("L", liz_capacity), ("M", memory_capacity)])
    
    @classmethod
    def dual_liz(cls, liz_capacity: int = 2, memory_capacity: int = 15):
        """Memory-LIZ-LIZ-Memory 構成"""
        return cls([("M", memory_capacity), ("L", liz_capacity), ("L", liz_capacity), ("M", memory_capacity)])


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
        """イオンがいずれかのLIZにいるか"""
        zone_id = self.locate(ion_id)
        return self.architecture.is_liz(zone_id)
    
    def can_execute(self, gate: GateOp) -> bool:
        """
        ゲートが実行可能か
        条件: 必要な全イオンが同じLIZにいる
        """
        if not gate.targets:
            return True
        
        first_zone = self.locate(gate.targets[0])
        if not self.architecture.is_liz(first_zone):
            return False
        
        # 全イオンが同じLIZにいるか確認
        return all(self.locate(ion_id) == first_zone for ion_id in gate.targets)
    
    def get_liz_with_space(self) -> Optional[int]:
        """空きのあるLIZを探す"""
        for liz in self.architecture.get_liz_zones():
            if len(self.zone_contents[liz.id]) < liz.capacity:
                return liz.id
        return None
    
    def get_liz_ions(self, liz_id: Optional[int] = None) -> List[int]:
        """指定LIZ（またはいずれかのLIZ）にいるイオンのリスト"""
        if liz_id is not None:
            return self.zone_contents[liz_id].copy()
        # 全LIZのイオンを返す
        ions = []
        for liz in self.architecture.get_liz_zones():
            ions.extend(self.zone_contents[liz.id])
        return ions
    
    def get_zone_free_space(self, zone_id: int) -> int:
        """ゾーンの空き容量"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            return 0
        return zone.capacity - len(self.zone_contents[zone_id])
    
    def find_eviction_destination(self, from_zone_id: int) -> Optional[int]:
        """
        退避先のMemoryゾーンを探す
        from_zone_id に隣接するMemoryゾーンで空きがあるものを優先
        """
        # 隣接ゾーンを優先
        for adj_id in self.architecture.get_adjacent_zones(from_zone_id):
            zone = self.architecture.get_zone(adj_id)
            if zone and zone.zone_type == "M" and self.get_zone_free_space(adj_id) > 0:
                return adj_id
        
        # 隣接になければ最寄りのMemoryを探す
        nearest = self.architecture.find_nearest_memory(from_zone_id)
        if nearest and self.get_zone_free_space(nearest) > 0:
            return nearest
        
        return None
    
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
    
    def get_ion_position_in_zone(self, ion_id: int) -> int:
        """ゾーン内でのイオンの位置（インデックス）を取得"""
        zone_id = self.locate(ion_id)
        return self.zone_contents[zone_id].index(ion_id)
    
    def swap_ions(self, ion_a: int, ion_b: int):
        """
        同一ゾーン内の隣接する2つのイオンの位置を交換（状態を更新）
        
        制約: 隣接イオン同士のみSWAP可能
        """
        zone_a = self.locate(ion_a)
        zone_b = self.locate(ion_b)
        
        if zone_a != zone_b:
            raise ValueError(f"Cannot swap ions in different zones: {ion_a} in {zone_a}, {ion_b} in {zone_b}")
        
        contents = self.zone_contents[zone_a]
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        
        # 隣接性チェック
        if abs(pos_a - pos_b) != 1:
            raise ValueError(
                f"Cannot swap non-adjacent ions: {ion_a} at pos {pos_a}, {ion_b} at pos {pos_b}. "
                f"Only adjacent swaps are allowed."
            )
        
        contents[pos_a], contents[pos_b] = contents[pos_b], contents[pos_a]
    
    def is_at_left_edge(self, ion_id: int) -> bool:
        """イオンがゾーンの左端にいるか"""
        return self.get_ion_position_in_zone(ion_id) == 0
    
    def is_at_right_edge(self, ion_id: int) -> bool:
        """イオンがゾーンの右端にいるか"""
        zone_id = self.locate(ion_id)
        return self.get_ion_position_in_zone(ion_id) == len(self.zone_contents[zone_id]) - 1
    
    def get_ion_at_edge(self, zone_id: int, left: bool) -> Optional[int]:
        """ゾーンの左端(left=True)または右端(left=False)のイオンを取得"""
        contents = self.zone_contents[zone_id]
        if not contents:
            return None
        return contents[0] if left else contents[-1]


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
       - LIZが満杯なら、まず退避操作を挿入
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
        stats = {"gates": 0, "shuttles_in": 0, "shuttles_out": 0, "swaps": 0, "iterations": 0}
        
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
                # 注意: _generate_shuttle_ops内で状態は既に更新される
                movement_ops = self._generate_shuttle_ops(dag, front_layer, state)
                for op, op_type in movement_ops:
                    operations.append(op)
                    # 統計のみ更新（状態は_generate_shuttle_ops内で更新済み）
                    if isinstance(op, ShuttleOp):
                        if op_type == "in":
                            stats["shuttles_in"] += 1
                        else:
                            stats["shuttles_out"] += 1
                    elif isinstance(op, SwapOp):
                        stats["swaps"] += 1
        
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
                # 全ターゲットが同じLIZにいるか確認
                targets = node.operation.targets
                if targets:
                    first_zone = state.locate(targets[0])
                    if self.architecture.is_liz(first_zone):
                        if all(state.locate(t) == first_zone for t in targets):
                            return node
        return None
    
    def _generate_shuttle_ops(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> List[Tuple[Operation, str]]:
        """
        必要な移動操作を生成
        
        Returns:
            List of (Operation, op_type) where op_type is "in", "out", or "swap"
        
        プロトタイプ版:
        - Front Layerの最小IDノードに必要なイオンをLIZへ移動
        - LIZが満杯なら、Front Layerで不要なイオンを退避
        - SHUTTLE前に必要ならSWAPを挿入
        """
        # 最小IDのゲートを対象にする
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if isinstance(node.operation, (GateOp, ResetOp)):
                targets = node.operation.targets
                movement_ops: List[Tuple[Operation, str]] = []
                
                # LIZにいないイオンを特定
                ions_to_bring = [ion_id for ion_id in targets if not state.is_in_liz(ion_id)]
                
                if not ions_to_bring:
                    continue
                
                # 目標LIZを決定（空きがあるLIZ、なければ最初のLIZ）
                target_liz_id = state.get_liz_with_space()
                if target_liz_id is None:
                    target_liz_id = self.architecture.get_liz_zones()[0].id
                
                target_liz = self.architecture.get_zone(target_liz_id)
                
                # 必要な空きスペース
                needed_space = len(ions_to_bring)
                current_space = state.get_zone_free_space(target_liz_id)
                
                # LIZが満杯または空きが足りない場合、退避が必要
                if current_space < needed_space:
                    eviction_ops = self._generate_eviction_ops(
                        state, target_liz_id, needed_space - current_space, set(targets)
                    )
                    movement_ops.extend(eviction_ops)
                    # 状態は_generate_eviction_ops内で既に更新済み
                
                # イオンをLIZへ移動
                for ion_id in ions_to_bring:
                    src_zone = state.locate(ion_id)
                    
                    # SHUTTLE前に必要ならSWAPを挿入
                    swap_ops = self._generate_swap_ops_for_shuttle(
                        state, ion_id, src_zone, target_liz_id
                    )
                    movement_ops.extend(swap_ops)
                    # SWAP操作を状態に反映
                    for op, _ in swap_ops:
                        state.swap_ions(op.targets[0], op.targets[1])
                    
                    # SHUTTLE操作を追加
                    movement_ops.append((
                        ShuttleOp(target=ion_id, dst=Location(trap=target_liz_id, site=0)),
                        "in"
                    ))
                    # SHUTTLE操作を状態に反映
                    state.move_ion(ion_id, target_liz_id)
                
                if movement_ops:
                    return movement_ops
        
        return []
    
    def _generate_eviction_ops(
        self, 
        state: QCCDState, 
        liz_id: int, 
        num_to_evict: int,
        protected_ions: Set[int]
    ) -> List[Tuple[Operation, str]]:
        """
        LIZからイオンを退避させる操作を生成
        
        Args:
            state: 現在の状態
            liz_id: 退避元のLIZ
            num_to_evict: 退避させる数
            protected_ions: 退避させてはいけないイオン（これから使うゲートのターゲット）
        """
        eviction_ops: List[Tuple[Operation, str]] = []
        liz_ions = state.get_liz_ions(liz_id)
        
        # 退避候補: protected_ions に含まれないイオン
        candidates = [ion for ion in liz_ions if ion not in protected_ions]
        
        if len(candidates) < num_to_evict:
            raise RuntimeError(
                f"Cannot evict {num_to_evict} ions from LIZ {liz_id}. "
                f"Only {len(candidates)} candidates available (protected: {protected_ions})"
            )
        
        # 退避先を探す
        evict_dst = state.find_eviction_destination(liz_id)
        if evict_dst is None:
            raise RuntimeError(f"No available memory zone for eviction from LIZ {liz_id}")
        
        # 退避操作を生成
        for i in range(num_to_evict):
            ion_to_evict = candidates[i]
            
            # SHUTTLE前に必要ならSWAPを挿入
            swap_ops = self._generate_swap_ops_for_shuttle(
                state, ion_to_evict, liz_id, evict_dst
            )
            eviction_ops.extend(swap_ops)
            # SWAP操作を状態に反映
            for op, _ in swap_ops:
                state.swap_ions(op.targets[0], op.targets[1])
            
            # SHUTTLE操作を追加
            eviction_ops.append((
                ShuttleOp(target=ion_to_evict, dst=Location(trap=evict_dst, site=0)),
                "out"
            ))
            # SHUTTLE操作を状態に反映
            state.move_ion(ion_to_evict, evict_dst)
        
        return eviction_ops
    
    def _generate_swap_ops_for_shuttle(
        self,
        state: QCCDState,
        ion_id: int,
        src_zone: int,
        dst_zone: int
    ) -> List[Tuple[SwapOp, str]]:
        """
        SHUTTLE前に必要なSWAP操作を生成
        
        制約: 隣接イオン同士のみSWAP可能
        
        ルール:
        - dst_zone < src_zone (左へ移動) → 左端(index=0)にいる必要
        - dst_zone > src_zone (右へ移動) → 右端にいる必要
        
        例: [ion_a, ion_b, ion_c, ion_d] で ion_a を右端に移動
            SWAP(ion_a, ion_b) → [ion_b, ion_a, ion_c, ion_d]
            SWAP(ion_a, ion_c) → [ion_b, ion_c, ion_a, ion_d]
            SWAP(ion_a, ion_d) → [ion_b, ion_c, ion_d, ion_a]
        
        Args:
            state: 現在の状態
            ion_id: 移動するイオン
            src_zone: 移動元ゾーン
            dst_zone: 移動先ゾーン
        
        Returns:
            List of (SwapOp, "swap")
        """
        zone_contents = state.zone_contents[src_zone]
        
        # ゾーンにイオンが1つしかない場合、SWAPは不要
        if len(zone_contents) <= 1:
            return []
        
        current_pos = zone_contents.index(ion_id)
        
        if dst_zone < src_zone:
            # 左へ移動 → 左端(index=0)に移動
            target_pos = 0
            direction = -1
        else:
            # 右へ移動 → 右端に移動
            target_pos = len(zone_contents) - 1
            direction = 1
        
        # 既に目標位置にいる場合
        if current_pos == target_pos:
            return []
        
        # 隣接SWAPを連続生成
        swap_ops = []
        # zone_contentsのコピーを作ってシミュレート
        simulated_contents = zone_contents.copy()
        sim_pos = current_pos
        
        while sim_pos != target_pos:
            # 隣接位置
            next_pos = sim_pos + direction
            neighbor_ion = simulated_contents[next_pos]
            
            # SWAP操作を生成
            swap_ops.append((SwapOp(targets=[ion_id, neighbor_ion]), "swap"))
            
            # シミュレート上の状態を更新
            simulated_contents[sim_pos], simulated_contents[next_pos] = \
                simulated_contents[next_pos], simulated_contents[sim_pos]
            sim_pos = next_pos
        
        return swap_ops


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
    print("QCCD Compiler Test (Prototype with Eviction and SWAP)")
    print("=" * 60)
    
    # パースとDAG構築
    program = parse(code)
    builder = DAGBuilder(program)
    dag = builder.build()
    
    # ========================================
    # Test 1: LIZ容量に余裕あり
    # ========================================
    print("\n" + "=" * 60)
    print("Test 1: Sufficient LIZ capacity (M-L-M, LIZ capacity=4)")
    print("=" * 60)
    
    arch1 = QCCDArchitecture.simple_mlm(liz_capacity=4)
    print(f"\nArchitecture: {arch1}")
    
    # Ion 0, 1 をLIZ(zone 1)に、Ion 2 をMemory(zone 0)に配置
    initial1 = {0: 1, 1: 1, 2: 0}
    print(f"Initial placement: {initial1}")
    
    compiler1 = QCCDCompiler(arch1)
    result1 = compiler1.compile(dag, initial1)
    
    print("\nGenerated Operations:")
    for i, op in enumerate(result1.operations):
        print(f"  [{i}] {op}")
    print(f"\nStatistics: {result1.stats}")
    
    # ========================================
    # Test 2: LIZ容量制限あり（退避が必要）
    # ========================================
    print("\n" + "=" * 60)
    print("Test 2: Limited LIZ capacity (M-L-M, LIZ capacity=2)")
    print("=" * 60)
    
    arch2 = QCCDArchitecture.simple_mlm(liz_capacity=2)
    print(f"\nArchitecture: {arch2}")
    
    # Ion 0, 1 をLIZ(zone 1)に、Ion 2 をMemory(zone 0)に配置
    # LIZ容量=2 なので、Ion 2を入れるにはIon 0か1を退避する必要
    initial2 = {0: 1, 1: 1, 2: 0}
    print(f"Initial placement: {initial2}")
    
    dag.reset_execution()  # DAGをリセット
    compiler2 = QCCDCompiler(arch2)
    result2 = compiler2.compile(dag, initial2)
    
    print("\nGenerated Operations:")
    for i, op in enumerate(result2.operations):
        op_str = str(op)
        # 操作をハイライト
        if isinstance(op, ShuttleOp):
            if arch2.is_liz(op.dst.trap):
                op_str += "  ← SHUTTLE IN"
            else:
                op_str += "  ← SHUTTLE OUT (eviction)"
        elif isinstance(op, SwapOp):
            op_str += "  ← SWAP"
        print(f"  [{i}] {op_str}")
    print(f"\nStatistics: {result2.stats}")
    
    # ========================================
    # Test 3: SWAPが必要なケース
    # ========================================
    print("\n" + "=" * 60)
    print("Test 3: SWAP required (multiple ions in Memory)")
    print("=" * 60)
    
    # 回路: Ion 0, 1, 2, 3 で MS 0 3 を実行
    code3 = """
    QUBIT 0 AT (0, 0)
    QUBIT 1 AT (0, 1)
    QUBIT 2 AT (0, 2)
    QUBIT 3 AT (0, 3)

    MS 0 3 0.25
    """
    
    program3 = parse(code3)
    builder3 = DAGBuilder(program3)
    dag3 = builder3.build()
    
    arch3 = QCCDArchitecture.simple_mlm(liz_capacity=4)
    print(f"\nArchitecture: {arch3}")
    
    # 全イオンをMemory(zone 0)に配置
    # zone 0 = Memory(左), zone 1 = LIZ, zone 2 = Memory(右)
    # LIZへ移動するには右端にいる必要（dst > src）
    # Ion 0 は左端、Ion 3 は右端
    initial3 = {0: 0, 1: 0, 2: 0, 3: 0}
    print(f"Initial placement: {initial3}")
    print("Zone 0 contents: [Ion0, Ion1, Ion2, Ion3] (left to right)")
    print("To move to LIZ (zone 1), ions must be at right edge")
    print("Ion 0 needs SWAP, Ion 3 doesn't")
    
    compiler3 = QCCDCompiler(arch3)
    result3 = compiler3.compile(dag3, initial3)
    
    print("\nGenerated Operations:")
    for i, op in enumerate(result3.operations):
        op_str = str(op)
        if isinstance(op, ShuttleOp):
            op_str += "  ← SHUTTLE IN"
        elif isinstance(op, SwapOp):
            op_str += "  ← SWAP (to reach edge)"
        print(f"  [{i}] {op_str}")
    print(f"\nStatistics: {result3.stats}")
    
    # ========================================
    # Test 4: 複数LIZ構成
    # ========================================
    print("\n" + "=" * 60)
    print("Test 4: Dual LIZ (M-L-L-M)")
    print("=" * 60)
    
    arch4 = QCCDArchitecture.dual_liz(liz_capacity=2)
    print(f"\nArchitecture: {arch4}")
    for zone in arch4.zones:
        print(f"  Zone {zone.id}: {zone.zone_type}, capacity={zone.capacity}, position={zone.position}")
    
    # カスタム構成のテスト
    print("\n" + "=" * 60)
    print("Test 5: Custom architecture")
    print("=" * 60)
    
    arch5 = QCCDArchitecture([("M", 15), ("L", 2), ("L", 2), ("M", 15)])
    print(f"\nArchitecture: {arch5}")
    print(f"Adjacent to zone 1: {arch5.get_adjacent_zones(1)}")
    print(f"Adjacent to zone 2: {arch5.get_adjacent_zones(2)}")