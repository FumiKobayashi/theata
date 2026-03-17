"""
QCCD Compiler: DAG → Parallel Executable QCCD Native Circuit

アーキテクチャ:
  DAG (ゲートのみ)
       ↓
  [MovementPlanner]  初期配置 + DAG → 平坦なAtomicOp列
       ↓              (Split/Swap/Merge/Gate、ゾーンアノテーション付き)
  [Scheduler]        AtomicOp列 → TimeStep列（ASAP並列化）
       ↓
  List[TimeStep]

各クラスの責務:
  MovementPlanner: 量子回路の実行順を守りながら移動操作をインライン展開して出力
  Scheduler:       ゾーン競合ルールのみを見てTimeStepに割り当て（配置を知らない）
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
from ast_def import (
    Location, QubitDecl, Operation, TimeStep,
    GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp, MeasureOp, ResetOp, Program
)
from dag_builder import DAG, DAGNode


# ==========================================
# Compiler Configuration
# ==========================================

@dataclass
class CompilerConfig:
    """
    コンパイラ最適化設定

    Attributes:
        gate_selection: ゲート選択戦略
            - "min_id": Front Layerから最小IDを選択（baseline）
    """
    gate_selection: str = "min_id"

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return f"CompilerConfig(gate={self.gate_selection})"

    @classmethod
    def baseline(cls) -> "CompilerConfig":
        return cls()

    def validate(self):
        valid_gate = {"min_id"}
        if self.gate_selection not in valid_gate:
            raise ValueError(f"Invalid gate_selection: {self.gate_selection}. Valid: {valid_gate}")


# ==========================================
# QCCD Architecture
# ==========================================

@dataclass
class Zone:
    """ゾーンの定義"""
    id: int
    zone_type: str   # "L" (LIZ), "M" (Memory), "T" (Temporary)
    capacity: int
    position: int


class QCCDArchitecture:
    """
    1D QCCDアーキテクチャの構成定義

    物理的な電極配置に基づく7ゾーン構成:
      T(20) - M(20) - T(20) - L(4) - T(20) - M(20) - T(20)

    入力形式: [(type, capacity), ...]
    """

    def __init__(self, zone_specs: List[Tuple[str, int]]):
        self.zones: List[Zone] = []
        for i, (zone_type, capacity) in enumerate(zone_specs):
            if zone_type not in ("M", "L", "T"):
                raise ValueError(f"Invalid zone type '{zone_type}'. Must be 'M', 'L', or 'T'.")
            self.zones.append(Zone(id=i, zone_type=zone_type, capacity=capacity, position=i))

        self._zone_by_id: Dict[int, Zone] = {z.id: z for z in self.zones}
        self._liz_zones: List[Zone] = [z for z in self.zones if z.zone_type == "L"]
        self._memory_zones: List[Zone] = [z for z in self.zones if z.zone_type == "M"]
        self._temp_zones: List[Zone] = [z for z in self.zones if z.zone_type == "T"]

        if not self._liz_zones:
            raise ValueError("Architecture must have at least one LIZ")

    def get_zone(self, zone_id: int) -> Optional[Zone]:
        return self._zone_by_id.get(zone_id)

    def get_liz_zones(self) -> List[Zone]:
        return self._liz_zones

    def get_memory_zones(self) -> List[Zone]:
        return self._memory_zones

    def get_temp_zones(self) -> List[Zone]:
        return self._temp_zones

    def get_adjacent_zones(self, zone_id: int) -> List[int]:
        adjacent = []
        if zone_id > 0:
            adjacent.append(zone_id - 1)
        if zone_id < len(self.zones) - 1:
            adjacent.append(zone_id + 1)
        return adjacent

    def is_liz(self, zone_id: int) -> bool:
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "L"

    def is_temp(self, zone_id: int) -> bool:
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "T"

    def is_memory(self, zone_id: int) -> bool:
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "M"

    def get_adjacent_temp(self, zone_id: int, direction: str) -> Optional[int]:
        """
        指定ゾーンの左("left")または右("right")の隣接Temporaryゾーンを取得

        例: T(0)-M(1)-T(2)-L(3)-T(4)-M(5)-T(6)
            get_adjacent_temp(1, "left")  → 0
            get_adjacent_temp(3, "left")  → 2
        """
        adj_id = zone_id - 1 if direction == "left" else zone_id + 1
        if adj_id < 0 or adj_id >= len(self.zones):
            return None
        if self.zones[adj_id].zone_type == "T":
            return adj_id
        return None

    def get_path(self, src_zone: int, dst_zone: int) -> List[int]:
        """srcからdstへの経路（両端含む）"""
        if src_zone <= dst_zone:
            return list(range(src_zone, dst_zone + 1))
        else:
            return list(range(src_zone, dst_zone - 1, -1))

    def find_nearest_memory(self, zone_id: int) -> Optional[int]:
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
    def simple_mlm(cls, liz_capacity: int = 4, memory_capacity: int = 20,
                   temp_capacity: int = 20):
        """T-M-T-L-T-M-T 構成"""
        return cls([
            ("T", temp_capacity),    # Zone 0
            ("M", memory_capacity),  # Zone 1
            ("T", temp_capacity),    # Zone 2
            ("L", liz_capacity),     # Zone 3
            ("T", temp_capacity),    # Zone 4
            ("M", memory_capacity),  # Zone 5
            ("T", temp_capacity),    # Zone 6
        ])


# ==========================================
# QCCD State
# ==========================================

@dataclass
class QCCDState:
    """
    QCCDの実行時状態

    不変条件:
    - Temporary Zoneは操作前後で空であること
    - move_ionは隣接ゾーン間のみ
    - SwapはTemporary Zoneでは実行不可
    """
    architecture: QCCDArchitecture
    ion_locations: Dict[int, int] = field(default_factory=dict)   # ion_id -> zone_id
    zone_contents: Dict[int, List[int]] = field(default_factory=dict)  # zone_id -> [ion_ids]

    def __post_init__(self):
        for zone in self.architecture.zones:
            if zone.id not in self.zone_contents:
                self.zone_contents[zone.id] = []

    def place_ion(self, ion_id: int, zone_id: int):
        """イオンをMemory Zoneに配置（初期配置用、Memory限定）"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} does not exist")
        if zone.zone_type != "M":
            raise ValueError(
                f"Initial placement only allowed in Memory zones. "
                f"Zone {zone_id} is type '{zone.zone_type}'."
            )
        if len(self.zone_contents[zone_id]) >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")
        self.ion_locations[ion_id] = zone_id
        self.zone_contents[zone_id].append(ion_id)

    def place_ion_any(self, ion_id: int, zone_id: int):
        """イオンを任意ゾーンに配置（Memory/LIZ/Temp対応、初期配置用）"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} does not exist")
        if len(self.zone_contents[zone_id]) >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")
        self.ion_locations[ion_id] = zone_id
        self.zone_contents[zone_id].append(ion_id)

    def locate(self, ion_id: int) -> int:
        if ion_id not in self.ion_locations:
            raise ValueError(f"Ion {ion_id} not found")
        return self.ion_locations[ion_id]

    def is_in_liz(self, ion_id: int) -> bool:
        return self.architecture.is_liz(self.locate(ion_id))

    def can_execute(self, gate: GateOp) -> bool:
        """ゲート実行可能か: 全イオンが同じLIZにいる"""
        if not gate.targets:
            return True
        first_zone = self.locate(gate.targets[0])
        if not self.architecture.is_liz(first_zone):
            return False
        return all(self.locate(ion_id) == first_zone for ion_id in gate.targets)

    def can_execute_reset_measure(self, op) -> bool:
        """RESET/MEASURE実行可能か: 全ターゲットが同じLIZ、かつそのLIZにターゲット以外がいない"""
        targets = op.targets
        if not targets:
            return True
        first_zone = self.locate(targets[0])
        if not self.architecture.is_liz(first_zone):
            return False
        if not all(self.locate(t) == first_zone for t in targets):
            return False
        liz_contents = set(self.zone_contents[first_zone])
        return liz_contents == set(targets)

    def get_liz_with_space(self) -> Optional[int]:
        for liz in self.architecture.get_liz_zones():
            if len(self.zone_contents[liz.id]) < liz.capacity:
                return liz.id
        return None

    def get_liz_ions(self, liz_id: Optional[int] = None) -> List[int]:
        if liz_id is not None:
            return self.zone_contents[liz_id].copy()
        ions = []
        for liz in self.architecture.get_liz_zones():
            ions.extend(self.zone_contents[liz.id])
        return ions

    def get_zone_free_space(self, zone_id: int) -> int:
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            return 0
        return zone.capacity - len(self.zone_contents[zone_id])

    def find_eviction_destination(self, from_zone_id: int) -> Optional[int]:
        """空きのある最寄りのMemoryゾーンを探す"""
        visited = {from_zone_id}
        queue = [from_zone_id]
        while queue:
            current = queue.pop(0)
            zone = self.architecture.get_zone(current)
            if zone and zone.zone_type == "M" and self.get_zone_free_space(current) > 0:
                return current
            for adj in self.architecture.get_adjacent_zones(current):
                if adj not in visited:
                    visited.add(adj)
                    queue.append(adj)
        return None

    def move_ion(self, ion_id: int, dst_zone_id: int):
        """イオンを隣接ゾーンへ移動"""
        src_zone_id = self.locate(ion_id)
        dst_zone = self.architecture.get_zone(dst_zone_id)
        if dst_zone is None:
            raise ValueError(f"Destination zone {dst_zone_id} does not exist")
        if abs(src_zone_id - dst_zone_id) != 1:
            raise ValueError(f"move_ion only allows adjacent zones. src={src_zone_id}, dst={dst_zone_id}")
        if len(self.zone_contents[dst_zone_id]) >= dst_zone.capacity:
            raise ValueError(f"Destination zone {dst_zone_id} is at capacity")
        self.zone_contents[src_zone_id].remove(ion_id)
        if src_zone_id < dst_zone_id:
            self.zone_contents[dst_zone_id].insert(0, ion_id)
        else:
            self.zone_contents[dst_zone_id].append(ion_id)
        self.ion_locations[ion_id] = dst_zone_id

    def swap_ions(self, ion_a: int, ion_b: int):
        """同一ゾーン内の隣接イオンをスワップ"""
        zone_a = self.locate(ion_a)
        zone_b = self.locate(ion_b)
        if zone_a != zone_b:
            raise ValueError(f"Cannot swap ions in different zones")
        if self.architecture.is_temp(zone_a):
            raise ValueError(f"Cannot swap ions in Temporary zone {zone_a}")
        contents = self.zone_contents[zone_a]
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        if abs(pos_a - pos_b) != 1:
            raise ValueError(f"Cannot swap non-adjacent ions")
        contents[pos_a], contents[pos_b] = contents[pos_b], contents[pos_a]

    def split_chain(self, ion_id: int, dst_zone_id: int):
        """
        イオンチェーンを分割し、片方をdst_zone_idへ移す

        方向はdst_zone_idとソースゾーンの位置関係から自動決定:
        - dst_zone_id > source → 右側チェーンがdstへ
        - dst_zone_id < source → 左側チェーンがdstへ
        """
        zone_id = self.locate(ion_id)
        contents = self.zone_contents[zone_id]
        pos = contents.index(ion_id)

        if len(contents) == 1:
            self.zone_contents[dst_zone_id] = [ion_id]
            self.zone_contents[zone_id] = []
            self.ion_locations[ion_id] = dst_zone_id
            return

        move_right = dst_zone_id > zone_id
        left_contents = contents[:pos + 1]
        right_contents = contents[pos + 1:]

        if move_right:
            if not right_contents:
                raise ValueError(
                    f"Cannot split: no right chain. ion {ion_id} is at right edge of zone {zone_id}"
                )
            self.zone_contents[zone_id] = left_contents
            self.zone_contents[dst_zone_id] = right_contents
            for ion in right_contents:
                self.ion_locations[ion] = dst_zone_id
        else:
            self.zone_contents[zone_id] = right_contents
            self.zone_contents[dst_zone_id] = left_contents
            for ion in left_contents:
                self.ion_locations[ion] = dst_zone_id

    def merge_chains(self, zone_left: int, zone_right: int):
        """
        2つの隣接イオンチェーンを結合

        結果はMemory/LIZゾーン側に保存、Temp Zoneは空になる。
        """
        left_contents = self.zone_contents.get(zone_left, [])
        right_contents = self.zone_contents.get(zone_right, [])
        merged = left_contents + right_contents

        left_is_temp = self.architecture.is_temp(zone_left)
        right_is_temp = self.architecture.is_temp(zone_right)

        if left_is_temp and not right_is_temp:
            result_zone, empty_zone = zone_right, zone_left
        elif right_is_temp and not left_is_temp:
            result_zone, empty_zone = zone_left, zone_right
        else:
            result_zone, empty_zone = zone_left, zone_right

        self.zone_contents[result_zone] = merged
        self.zone_contents[empty_zone] = []
        for ion in merged:
            self.ion_locations[ion] = result_zone

    def get_ion_position_in_zone(self, ion_id: int) -> int:
        zone_id = self.locate(ion_id)
        return self.zone_contents[zone_id].index(ion_id)

    def is_at_right_edge(self, ion_id: int) -> bool:
        zone_id = self.locate(ion_id)
        return self.get_ion_position_in_zone(ion_id) == len(self.zone_contents[zone_id]) - 1


# ==========================================
# Movement Planner
# ==========================================

class MovementPlanner:
    """
    DAG + 初期配置 → 平坦なAtomicOp列

    責務:
    - 量子回路の実行順を守りながら移動操作を生成
    - ShuttleOp/SwapOp を Split/Merge にインライン展開
    - 各操作にゾーンアノテーション(zone_id, src_zone_id等)を付与
    - Schedulerに渡す平坦なList[Operation]を出力

    Schedulerへの約束:
    - 各操作に zones_used() に必要なフィールドが設定されている
    - 出力の順序は論理的な依存関係を反映している（Schedulerはこれを並列化する）
    """

    def __init__(self, architecture: QCCDArchitecture, config: "CompilerConfig" = None):
        self.architecture = architecture
        self.config = config or CompilerConfig.baseline()

    def plan(self, dag: DAG, initial_placement: Dict[int, int]) -> List[Operation]:
        """
        メインエントリーポイント: DAG → 平坦なAtomicOp列

        Args:
            dag: 依存関係グラフ（事前にreset_execution済み想定）
            initial_placement: ion_id -> zone_id

        Returns:
            List[Operation]: Split/Swap/Merge/Gate の平坦な列（ゾーンアノテーション付き）
        """
        state = QCCDState(architecture=self.architecture)
        for ion_id, zone_id in initial_placement.items():
            state.place_ion_any(ion_id, zone_id)

        dag.reset_execution()
        ops: List[Operation] = []
        max_iterations = len(dag.nodes) * 200

        for _ in range(max_iterations):
            if dag.is_complete():
                break

            front_layer = dag.get_front_layer()

            node = self._find_executable_node(dag, front_layer, state)
            if node is not None:
                ops.append(self._annotate_gate(node.operation, state))
                dag.execute(node.id)
            else:
                movement_ops = self._generate_movements(dag, front_layer, state)
                if not movement_ops:
                    raise RuntimeError(
                        "MovementPlanner: deadlock — no executable gate and no movement possible.\n"
                        f"Front layer: {front_layer}\n"
                        f"State: {state.ion_locations}"
                    )
                ops.extend(movement_ops)
        else:
            if not dag.is_complete():
                raise RuntimeError("MovementPlanner: max iterations exceeded")

        return ops

    # ------------------------------------------
    # Gate Execution
    # ------------------------------------------

    def _find_executable_node(
        self, dag: DAG, front_layer: Set[int], state: QCCDState
    ) -> Optional[DAGNode]:
        """Front Layerから実行可能なノードを探す"""
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            op = node.operation
            if isinstance(op, GateOp) and state.can_execute(op):
                return node
            if isinstance(op, (ResetOp, MeasureOp)) and state.can_execute_reset_measure(op):
                return node
        return None

    def _annotate_gate(self, op: Operation, state: QCCDState) -> Operation:
        """GateOp/ResetOp/MeasureOpにzone_idを付与した新しいオペレーションを返す"""
        if isinstance(op, GateOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            return GateOp(type=op.type, targets=op.targets, params=op.params, zone_id=liz_id)
        if isinstance(op, ResetOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            return ResetOp(targets=op.targets, zone_id=liz_id)
        if isinstance(op, MeasureOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            return MeasureOp(targets=op.targets, zone_id=liz_id)
        return op

    # ------------------------------------------
    # Movement Generation
    # ------------------------------------------

    def _generate_movements(
        self, dag: DAG, front_layer: Set[int], state: QCCDState
    ) -> List[Operation]:
        """
        Front Layerの最優先ノードに必要な移動操作を生成

        優先順: node_idの昇順で最初に移動が必要なノードを処理
        """
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            op = node.operation

            if not isinstance(op, (GateOp, ResetOp, MeasureOp)):
                continue

            targets = op.targets
            target_set = set(targets)

            # 各イオンの現在地を確認
            ions_in_liz = [ion for ion in targets if state.is_in_liz(ion)]
            ions_not_in_liz = [ion for ion in targets if not state.is_in_liz(ion)]

            # 全イオンがLIZ上にいる場合
            if not ions_not_in_liz:
                liz_zones = set(state.locate(ion) for ion in targets)
                if len(liz_zones) == 1:
                    # 同じLIZ内にいる
                    target_liz = state.locate(targets[0])
                    if isinstance(op, (ResetOp, MeasureOp)):
                        # RESET/MEASURE: ターゲット以外のイオンを退避
                        extra_ions = [
                            ion for ion in state.zone_contents.get(target_liz, [])
                            if ion not in target_set
                        ]
                        if extra_ions:
                            result = []
                            for ion in extra_ions:
                                evict_dst = state.find_eviction_destination(target_liz)
                                if evict_dst is None:
                                    raise RuntimeError(
                                        f"Cannot evict ion {ion} from LIZ {target_liz}: "
                                        f"no memory space available"
                                    )
                                result.extend(self._move_ion_to_zone(ion, evict_dst, state))
                            return result
                    # Gate or RESET/MEASURE with no extra ions → 実行可能なはず
                    continue
                # 異なるLIZ → 一方のLIZに集める（バグ修正: 旧実装の無限ループを解消）
                target_liz = state.locate(targets[0])
                result = []
                for ion in targets:
                    if state.locate(ion) != target_liz:
                        src_liz = state.locate(ion)
                        evict_dst = state.find_eviction_destination(src_liz)
                        if evict_dst is None:
                            raise RuntimeError(
                                f"Cannot evict ion {ion} from LIZ {src_liz}: no memory space available"
                            )
                        result.extend(self._move_ion_to_zone(ion, evict_dst, state))
                        result.extend(self._move_ion_to_zone(ion, target_liz, state))
                if result:
                    return result
                continue

            # 目標LIZを決定
            if ions_in_liz:
                # 既にLIZにいるイオンのLIZを目標に
                target_liz = state.locate(ions_in_liz[0])
            else:
                target_liz = state.get_liz_with_space()
                if target_liz is None:
                    target_liz = self.architecture.get_liz_zones()[0].id

            result = []

            # RESET/MEASUREの場合: LIZ上のターゲット以外を全て退避
            if isinstance(op, (ResetOp, MeasureOp)):
                for ion in list(state.zone_contents.get(target_liz, [])):
                    if ion not in target_set:
                        evict_dst = state.find_eviction_destination(target_liz)
                        if evict_dst is None:
                            raise RuntimeError(
                                f"Cannot evict ion {ion} from LIZ {target_liz}: no memory space available"
                            )
                        result.extend(self._move_ion_to_zone(ion, evict_dst, state))

            # LIZに必要な空きを確保
            ions_to_bring = [ion for ion in targets if state.locate(ion) != target_liz]
            free = state.get_zone_free_space(target_liz)
            needed = len(ions_to_bring)

            if free < needed:
                liz_contents = list(state.zone_contents.get(target_liz, []))
                candidates = [ion for ion in liz_contents if ion not in target_set]
                n_to_evict = needed - free
                if len(candidates) < n_to_evict:
                    raise RuntimeError(
                        f"Cannot free {n_to_evict} spots in LIZ {target_liz}. "
                        f"Only {len(candidates)} evictable ions (protected: {target_set})"
                    )
                for i in range(n_to_evict):
                    # 退避先を毎回再探索（バグ修正: 旧実装は容量確認なしで同一先に複数送っていた）
                    evict_dst = state.find_eviction_destination(target_liz)
                    if evict_dst is None:
                        raise RuntimeError(
                            f"Cannot find eviction destination from LIZ {target_liz}"
                        )
                    result.extend(self._move_ion_to_zone(candidates[i], evict_dst, state))

            # ターゲットイオンをLIZへ移動
            for ion in ions_to_bring:
                if state.locate(ion) != target_liz:
                    result.extend(self._move_ion_to_zone(ion, target_liz, state))

            if result:
                return result

        return []

    # ------------------------------------------
    # Ion Movement (inline expansion)
    # ------------------------------------------

    def _move_ion_to_zone(
        self, ion_id: int, dst_zone: int, state: QCCDState
    ) -> List[Operation]:
        """
        イオンをdst_zoneへ移動する操作列を生成（Split+Shuttle+Mergeをインライン展開）

        物理的な1ホップ: src(M/L) → temp(T) → dst(M/L)
        """
        src_zone = state.locate(ion_id)
        if src_zone == dst_zone:
            return []

        is_right = src_zone < dst_zone
        direction = "right" if is_right else "left"
        temp_zone = self.architecture.get_adjacent_temp(src_zone, direction)

        if temp_zone is None:
            raise RuntimeError(
                f"No temp zone adjacent to zone {src_zone} (direction={direction}). "
                f"Cannot move ion {ion_id}."
            )

        ops = []

        # Step 1: イオンを端に移動（展開済みSwap列を生成）
        ops.extend(self._position_for_shuttle(ion_id, src_zone, is_right, state))

        # Step 2: Split（ion_idをtemp_zoneへ分離）
        src_contents = state.zone_contents[src_zone]
        if len(src_contents) == 1:
            split_op = SplitOp(ion_id=ion_id, dst_zone_id=temp_zone, src_zone_id=src_zone)
            ops.append(split_op)
            state.split_chain(ion_id, temp_zone)
        else:
            ion_pos = src_contents.index(ion_id)
            if is_right:
                # 右端にいるはず: 左隣で分割 → right chain=[ion_id]がtempへ
                assert ion_pos == len(src_contents) - 1, \
                    f"ion {ion_id} should be at right end (pos={ion_pos}, len={len(src_contents)})"
                split_ref = src_contents[ion_pos - 1]
                split_op = SplitOp(ion_id=split_ref, dst_zone_id=temp_zone, src_zone_id=src_zone)
                ops.append(split_op)
                state.split_chain(split_ref, temp_zone)
            else:
                # 左端にいるはず: ion_id自体で分割 → left chain=[ion_id]がtempへ
                assert ion_pos == 0, \
                    f"ion {ion_id} should be at left end (pos={ion_pos})"
                split_op = SplitOp(ion_id=ion_id, dst_zone_id=temp_zone, src_zone_id=src_zone)
                ops.append(split_op)
                state.split_chain(ion_id, temp_zone)

        # Step 3: Shuttle（物理的な移動）
        dst_contents = state.zone_contents.get(dst_zone, [])
        site = 0 if is_right else len(dst_contents)
        shuttle_op = ShuttleOp(
            target=ion_id,
            dst=Location(trap=dst_zone, site=site),
            dst_chain_length=len(dst_contents),
            src_zone=temp_zone
        )
        ops.append(shuttle_op)

        # Step 4: Merge
        left_z = min(temp_zone, dst_zone)
        right_z = max(temp_zone, dst_zone)
        merge_op = MergeOp(zone_left=left_z, zone_right=right_z)
        ops.append(merge_op)
        state.merge_chains(left_z, right_z)

        return ops

    def _position_for_shuttle(
        self, ion_id: int, src_zone: int, is_right: bool, state: QCCDState
    ) -> List[Operation]:
        """
        イオンをシャトル方向の端へ移動する展開済みSwap操作列を生成

        is_right=True  → 右端へ移動
        is_right=False → 左端へ移動
        """
        ops = []
        direction = 1 if is_right else -1

        while True:
            contents = state.zone_contents[src_zone]
            ion_pos = contents.index(ion_id)
            target_pos = len(contents) - 1 if is_right else 0

            if ion_pos == target_pos:
                break

            neighbor = contents[ion_pos + direction]
            # 左イオンと右イオンの順でexpand
            if direction == 1:
                ops.extend(self._expand_swap_inline(ion_id, neighbor, state))
            else:
                ops.extend(self._expand_swap_inline(neighbor, ion_id, state))

        return ops

    def _expand_swap_inline(
        self, ion_a: int, ion_b: int, state: QCCDState
    ) -> List[Operation]:
        """
        ion_a と ion_b を同一ゾーン内でスワップ（Split+Swap+Mergeに展開）

        ion_a と ion_b は隣接している必要がある。
        ion_a が ion_b の左にいることを前提とする（呼び出し側の責務）。

        パターン:
        1. 2イオンのみ: Swap のみ
        2. 左端ペア [A,B,...]: 右チェーンを右Tempへ退避
        3. 右端ペア [...,C,D]: 左チェーンを左Tempへ退避
        4. 中間ペア: 左を左Temp、右を右Tempへ退避
        """
        zone_id = state.locate(ion_a)
        assert state.locate(ion_b) == zone_id, "Ions must be in the same zone"

        contents = state.zone_contents[zone_id]
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        left_pos = min(pos_a, pos_b)
        right_pos = max(pos_a, pos_b)
        left_ion = contents[left_pos]
        right_ion = contents[right_pos]

        arch = self.architecture
        left_temp = arch.get_adjacent_temp(zone_id, "left")
        right_temp = arch.get_adjacent_temp(zone_id, "right")

        ops = []

        if len(contents) == 2:
            # パターン1: 2イオンのみ → そのままSwap
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)

        elif left_pos == 0:
            # パターン2: 左端ペア [A, B, C, D] の Swap(A, B)
            # 右チェーン[C,D]を右Tempへ退避
            assert right_temp is not None, f"No right temp zone for zone {zone_id}"
            ops.append(SplitOp(ion_id=right_ion, dst_zone_id=right_temp, src_zone_id=zone_id))
            state.split_chain(right_ion, right_temp)
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)
            ops.append(MergeOp(zone_left=zone_id, zone_right=right_temp))
            state.merge_chains(zone_id, right_temp)

        elif right_pos == len(contents) - 1:
            # パターン3: 右端ペア [..., C, D] の Swap(C, D)
            # 左チェーンを左Tempへ退避
            assert left_temp is not None, f"No left temp zone for zone {zone_id}"
            split_ref = contents[left_pos - 1]
            ops.append(SplitOp(ion_id=split_ref, dst_zone_id=left_temp, src_zone_id=zone_id))
            state.split_chain(split_ref, left_temp)
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)
            ops.append(MergeOp(zone_left=left_temp, zone_right=zone_id))
            state.merge_chains(left_temp, zone_id)

        else:
            # パターン4: 中間ペア [A, B, C, D] の Swap(B, C)
            # 左チェーンを左Temp、右チェーンを右Tempへ退避
            assert left_temp is not None, f"No left temp zone for zone {zone_id}"
            assert right_temp is not None, f"No right temp zone for zone {zone_id}"
            split_ion1 = contents[left_pos - 1]
            ops.append(SplitOp(ion_id=split_ion1, dst_zone_id=left_temp, src_zone_id=zone_id))
            state.split_chain(split_ion1, left_temp)
            ops.append(SplitOp(ion_id=right_ion, dst_zone_id=right_temp, src_zone_id=zone_id))
            state.split_chain(right_ion, right_temp)
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)
            ops.append(MergeOp(zone_left=zone_id, zone_right=right_temp))
            state.merge_chains(zone_id, right_temp)
            ops.append(MergeOp(zone_left=left_temp, zone_right=zone_id))
            state.merge_chains(left_temp, zone_id)

        return ops


# ==========================================
# Scheduler
# ==========================================

class Scheduler:
    """
    AtomicOp列 → TimeStep列（ASAP並列スケジューリング）

    責務:
    - 各操作が占有するゾーン（zones_used）を見て競合を検出
    - 競合しない操作をできるだけ同じTimeStepに詰める
    - 入力の順序制約（同じゾーンを使う操作は入力順を維持）を守る

    配置情報を持たない: zones_usedフィールドのみを参照
    """

    def schedule(self, ops: List[Operation]) -> List[TimeStep]:
        """
        Args:
            ops: MovementPlannerが生成したAtomicOp列

        Returns:
            List[TimeStep]: 並列実行可能な操作グループの列
        """
        if not ops:
            return []

        n = len(ops)
        op_times = [0] * n              # 各opのTimeStepインデックス
        ts_zones: Dict[int, Set[int]] = {}    # ts -> 使用中ゾーン集合
        zone_last_ts: Dict[int, int] = {}     # zone_id -> そのゾーンを最後に使ったTS

        for i, op in enumerate(ops):
            zones = self._zones_used(op)

            # このopより前にzoneを使ったopのTS+1が最早開始TS
            earliest = 0
            for z in zones:
                if z in zone_last_ts:
                    earliest = max(earliest, zone_last_ts[z] + 1)

            # 競合しない最早TSを探す
            t = earliest
            while True:
                if t not in ts_zones or not (zones & ts_zones[t]):
                    break
                t += 1

            op_times[i] = t
            if t not in ts_zones:
                ts_zones[t] = set()
            ts_zones[t] |= zones
            for z in zones:
                zone_last_ts[z] = max(zone_last_ts.get(z, -1), t)

        # TimeStepに集約
        max_t = max(op_times)
        result = []
        for t in range(max_t + 1):
            step_ops = [ops[i] for i in range(n) if op_times[i] == t]
            if step_ops:
                result.append(TimeStep(ops=step_ops))

        return result

    def _zones_used(self, op: Operation) -> Set[int]:
        """操作が占有するゾーンの集合を返す"""
        if isinstance(op, SplitOp):
            zones = {op.dst_zone_id}
            if op.src_zone_id != -1:
                zones.add(op.src_zone_id)
            return zones
        if isinstance(op, ShuttleOp):
            # ShuttleはTempゾーン（src_zone）と移動先ゾーンの両方をロック
            # 移動先をロックしないと、退避が完了する前に他のイオンがそのゾーンへ
            # シャトルされるタイムステップに早期スケジューリングされる可能性がある
            zones = set()
            if op.src_zone != -1:
                zones.add(op.src_zone)
            dst_zone = op.dst.trap if hasattr(op.dst, 'trap') else op.dst[0]
            if dst_zone != -1:
                zones.add(dst_zone)
            return zones
        if isinstance(op, MergeOp):
            zones = set()
            if op.zone_left != -1:
                zones.add(op.zone_left)
            if op.zone_right != -1:
                zones.add(op.zone_right)
            return zones
        if isinstance(op, SwapOp):
            if op.zone_id != -1:
                return {op.zone_id}
            return set()
        if isinstance(op, (GateOp, ResetOp, MeasureOp)):
            if op.zone_id != -1:
                return {op.zone_id}
            return set()
        return set()


# ==========================================
# Compile Result
# ==========================================

@dataclass
class CompileResult:
    """コンパイル結果"""
    timesteps: List[TimeStep]   # Scheduler出力（並列化済み）
    stats: Dict[str, int]       # 統計情報
    config: "CompilerConfig" = None

    @property
    def operations(self) -> List[Operation]:
        """後方互換: timestepsを平坦化した操作列"""
        return [op for ts in self.timesteps for op in ts.ops]


# ==========================================
# QCCD Compiler (MovementPlanner + Scheduler のラッパー)
# ==========================================

class QCCDCompiler:
    """
    MovementPlanner と Scheduler を組み合わせたコンパイラ

    使用方法:
        compiler = QCCDCompiler(architecture)
        result = compiler.compile(dag, initial_placement)
        # result.timesteps: 並列TimeStep列
        # result.operations: 平坦な操作列（後方互換）
    """

    def __init__(self, architecture: QCCDArchitecture, config: CompilerConfig = None):
        self.architecture = architecture
        self.config = config or CompilerConfig.baseline()
        self.config.validate()
        self._planner = MovementPlanner(architecture, self.config)
        self._scheduler = Scheduler()

    def compile(self, dag: DAG, initial_placement: Dict[int, int]) -> CompileResult:
        """
        Args:
            dag: 依存関係グラフ
            initial_placement: ion_id -> zone_id の初期配置

        Returns:
            CompileResult: timesteps（並列化済み）と統計情報
        """
        # Phase 1: 平坦なAtomicOp列を生成
        flat_ops = self._planner.plan(dag, initial_placement)

        # Phase 2: TimeStep列に並列化
        timesteps = self._scheduler.schedule(flat_ops)

        stats = self._calculate_stats(flat_ops)
        stats["timesteps"] = len(timesteps)

        return CompileResult(timesteps=timesteps, stats=stats, config=self.config)

    def _calculate_stats(self, ops: List[Operation]) -> Dict[str, int]:
        stats = {"gates": 0, "shuttles": 0, "swaps": 0, "splits": 0, "merges": 0}
        for op in ops:
            if isinstance(op, GateOp):
                stats["gates"] += 1
            elif isinstance(op, (ResetOp, MeasureOp)):
                stats["gates"] += 1
            elif isinstance(op, ShuttleOp):
                stats["shuttles"] += 1
            elif isinstance(op, SwapOp):
                stats["swaps"] += 1
            elif isinstance(op, SplitOp):
                stats["splits"] += 1
            elif isinstance(op, MergeOp):
                stats["merges"] += 1
        return stats
