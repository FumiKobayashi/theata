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
        memory_reorder: Measure/Reset中にMemory内イオンを先読み整列するか
        measure_batch_size: Measureのバッチサイズ
            - 1: 即時測定（デフォルト、従来動作）
            - 2+: 指定数のアンシラが測定待ちになるまで遅延し一括測定
            - LIZ容量を超える値は自動的にLIZ容量にクランプされる
        eviction_strategy: LIZからの退避戦略
            - "balanced": アンシラの左右で振り分け（複数ラウンド向き、配置バランス保持）
            - "nearest": 最寄りMemoryへBFS（1ラウンド最小ops）
    """
    gate_selection: str = "min_id"
    memory_reorder: bool = False
    measure_batch_size: int = 1
    interleave: bool = False
    eviction_strategy: str = "balanced"

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        parts = [f"gate={self.gate_selection}"]
        if self.measure_batch_size > 1:
            parts.append(f"batch={self.measure_batch_size}")
        if self.interleave:
            parts.append("interleave")
        return f"CompilerConfig({', '.join(parts)})"

    @classmethod
    def baseline(cls) -> "CompilerConfig":
        return cls()

    def validate(self):
        valid_gate = {"min_id"}
        if self.gate_selection not in valid_gate:
            raise ValueError(f"Invalid gate_selection: {self.gate_selection}. Valid: {valid_gate}")
        if not isinstance(self.memory_reorder, bool):
            raise ValueError(f"memory_reorder must be bool, got {type(self.memory_reorder)}")
        valid_eviction = {"balanced", "nearest"}
        if self.eviction_strategy not in valid_eviction:
            raise ValueError(f"Invalid eviction_strategy: {self.eviction_strategy}. Valid: {valid_eviction}")
        if not isinstance(self.measure_batch_size, int) or self.measure_batch_size < 1:
            raise ValueError(f"measure_batch_size must be int >= 1, got {self.measure_batch_size}")


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

    フォノン追跡:
    - 各イオンの axial / transversal フォノン数を管理
    - 同一チェイン内のイオンは同じフォノン数を持つ（不変条件）
    - phonon_params が None の場合はフォノン追跡を無効化
    """
    architecture: QCCDArchitecture
    ion_locations: Dict[int, int] = field(default_factory=dict)   # ion_id -> zone_id
    zone_contents: Dict[int, List[int]] = field(default_factory=dict)  # zone_id -> [ion_ids]
    phonon_axial: Dict[int, float] = field(default_factory=dict)       # ion_id -> axial phonon
    phonon_transversal: Dict[int, float] = field(default_factory=dict) # ion_id -> transversal phonon
    phonon_params: Optional[dict] = None  # フォノンパラメータ（Noneで無効化）
    # MSゲート実行時のフォノン記録
    phonon_at_gates: List[dict] = field(default_factory=list)

    def __post_init__(self):
        for zone in self.architecture.zones:
            if zone.id not in self.zone_contents:
                self.zone_contents[zone.id] = []

    def _init_phonon(self, ion_id: int):
        """イオンのフォノン数を初期化"""
        if ion_id not in self.phonon_axial:
            self.phonon_axial[ion_id] = 0.0
            self.phonon_transversal[ion_id] = 0.0

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
        self._init_phonon(ion_id)

    def place_ion_any(self, ion_id: int, zone_id: int):
        """イオンを任意ゾーンに配置（Memory/LIZ/Temp対応、初期配置用）"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} does not exist")
        if len(self.zone_contents[zone_id]) >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")
        self.ion_locations[ion_id] = zone_id
        self.zone_contents[zone_id].append(ion_id)
        self._init_phonon(ion_id)

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

    def find_eviction_destination(self, from_zone_id: int, ion_id: int = None) -> Optional[int]:
        """空きのある最寄りのMemoryゾーンを探す（方向指定可能）"""
        return self._find_memory_in_direction(from_zone_id, prefer_left=True)

    def _find_memory_in_direction(self, from_zone_id: int, prefer_left: bool = True) -> Optional[int]:
        """指定方向を優先してMemoryゾーンを探す"""
        if prefer_left:
            search_order_fn = lambda zid: self.architecture.get_adjacent_zones(zid)
        else:
            search_order_fn = lambda zid: list(reversed(self.architecture.get_adjacent_zones(zid)))

        visited = {from_zone_id}
        queue = [from_zone_id]
        while queue:
            current = queue.pop(0)
            zone = self.architecture.get_zone(current)
            if zone and zone.zone_type == "M" and self.get_zone_free_space(current) > 0:
                return current
            for adj in search_order_fn(current):
                if adj not in visited:
                    visited.add(adj)
                    queue.append(adj)
        return None

    def plan_evictions(self, zone_id: int, ions_to_evict: list, protected: set = None) -> list:
        """
        複数イオンの退避先をprotected（アンシラ）の左右で振り分ける

        LIZ内の配置は自然に [z1由来, アンシラ, z5由来] となっているため、
        アンシラより左のイオンは左方向(zone_idが小さいMemory)へ、
        アンシラより右のイオンは右方向(zone_idが大きいMemory)へ退避する。
        これによりLIZ内でのSwapが不要になる。

        protectedがない場合は左右交互にフォールバック。

        Returns:
            [(ion_id, evict_destination_zone_id), ...]
            退避順序: 左端→右端→左端→... の順（両端から内側へ）
        """
        if protected is None:
            protected = set()

        contents = list(self.zone_contents.get(zone_id, []))
        evict_set = set(ions_to_evict)

        # protectedイオンの位置を特定
        protected_positions = [i for i, ion in enumerate(contents) if ion in protected]

        if protected_positions:
            # protectedの最小位置より左 → 左方向へ退避
            # protectedの最大位置より右 → 右方向へ退避
            protected_min = min(protected_positions)
            protected_max = max(protected_positions)
        else:
            # protectedがなければ退避先の空き容量で振り分ける
            # 左Memory と右Memory の空きを比較して均等に分配
            left_mem = self._find_memory_in_direction(zone_id, prefer_left=True)
            right_mem = self._find_memory_in_direction(zone_id, prefer_left=False)
            left_free = self.get_zone_free_space(left_mem) if left_mem is not None else 0
            right_free = self.get_zone_free_space(right_mem) if right_mem is not None else 0

            plan = []
            evict_list = [ion for ion in contents if ion in evict_set]
            for ion in evict_list:
                if left_free > right_free:
                    plan.append((ion, left_mem))
                    left_free -= 1
                elif right_free > 0:
                    plan.append((ion, right_mem))
                    right_free -= 1
                else:
                    plan.append((ion, left_mem))
                    left_free -= 1
            return plan

        # 左側イオン（左端から順に）と右側イオン（右端から順に）に分類
        left_ions = []   # 左方向へ退避（左端から内側へ）
        right_ions = []  # 右方向へ退避（右端から内側へ）

        for i, ion in enumerate(contents):
            if ion not in evict_set:
                continue
            if i < protected_min:
                left_ions.append(ion)
            elif i > protected_max:
                right_ions.append(ion)
            else:
                # protectedの間にいる → 近い方へ
                if i - protected_min <= protected_max - i:
                    left_ions.append(ion)
                else:
                    right_ions.append(ion)

        # 退避計画を構築（両端から交互に実行）
        plan = []
        # 左側は左端から（リスト順 = 左端から内側へ）
        for ion in left_ions:
            dst = self._find_memory_in_direction(zone_id, prefer_left=True)
            if dst is not None:
                plan.append((ion, dst))

        # 右側は右端から（リスト逆順 = 右端から内側へ）
        for ion in reversed(right_ions):
            dst = self._find_memory_in_direction(zone_id, prefer_left=False)
            if dst is not None:
                plan.append((ion, dst))

        return plan

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
        # フォノン: +0.1 (axial, transversal)
        if self.phonon_params:
            ax = self.phonon_params.get('swap_axial', 0.1)
            tr = self.phonon_params.get('swap_transversal', 0.1)
            self.phonon_axial[ion_a] += ax
            self.phonon_transversal[ion_a] += tr
            self.phonon_axial[ion_b] += ax
            self.phonon_transversal[ion_b] += tr

    def split_chain(self, ion_id: int, dst_zone_id: int):
        """
        イオンチェーンを分割し、片方をdst_zone_idへ移す

        方向はdst_zone_idとソースゾーンの位置関係から自動決定:
        - dst_zone_id > source → 右側チェーンがdstへ
        - dst_zone_id < source → 左側チェーンがdstへ

        フォノン: 移動するチェインのイオンにのみ加算
        """
        zone_id = self.locate(ion_id)
        contents = self.zone_contents[zone_id]
        pos = contents.index(ion_id)

        if len(contents) == 1:
            self.zone_contents[dst_zone_id] = [ion_id]
            self.zone_contents[zone_id] = []
            self.ion_locations[ion_id] = dst_zone_id
            # フォノン: 移動したイオンに加算
            if self.phonon_params:
                self.phonon_axial[ion_id] += self.phonon_params.get('split_axial', 10.0)
                self.phonon_transversal[ion_id] += self.phonon_params.get('split_transversal', 0.05)
            return

        move_right = dst_zone_id > zone_id
        left_contents = contents[:pos + 1]
        right_contents = contents[pos + 1:]

        if move_right:
            if not right_contents:
                raise ValueError(
                    f"Cannot split: no right chain. ion {ion_id} is at right edge of zone {zone_id}"
                )
            moved = right_contents
            self.zone_contents[zone_id] = left_contents
            self.zone_contents[dst_zone_id] = right_contents
            for ion in right_contents:
                self.ion_locations[ion] = dst_zone_id
        else:
            moved = left_contents
            self.zone_contents[zone_id] = right_contents
            self.zone_contents[dst_zone_id] = left_contents
            for ion in left_contents:
                self.ion_locations[ion] = dst_zone_id

        # フォノン: 移動チェインにのみ加算
        if self.phonon_params:
            ax = self.phonon_params.get('split_axial', 10.0)
            tr = self.phonon_params.get('split_transversal', 0.05)
            for ion in moved:
                self.phonon_axial[ion] += ax
                self.phonon_transversal[ion] += tr

    def merge_chains(self, zone_left: int, zone_right: int):
        """
        2つの隣接イオンチェーンを結合

        結果はMemory/LIZゾーン側に保存、Temp Zoneは空になる。

        フォノン: 全イオンの平均を取ってからmergeコストを加算
        """
        left_contents = self.zone_contents.get(zone_left, [])
        right_contents = self.zone_contents.get(zone_right, [])
        merged = left_contents + right_contents

        # フォノン: 全イオンの平均 + merge cost
        if self.phonon_params and merged:
            n = len(merged)
            avg_ax = sum(self.phonon_axial.get(i, 0.0) for i in merged) / n
            avg_tr = sum(self.phonon_transversal.get(i, 0.0) for i in merged) / n
            merge_ax = self.phonon_params.get('merge_axial', 10.0)
            merge_tr = self.phonon_params.get('merge_transversal', 0.05)
            for ion in merged:
                self.phonon_axial[ion] = avg_ax + merge_ax
                self.phonon_transversal[ion] = avg_tr + merge_tr

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

    def apply_shuttle_phonon(self, ion_id: int, dst_chain_length: int = 0):
        """Shuttle操作のフォノン加算（ShuttleOp生成時に呼ぶ）

        Temp zone にいる全イオンに距離依存の axial + 固定 transversal を加算
        """
        if not self.phonon_params:
            return
        src_zone = self.locate(ion_id)
        ions_in_chain = self.zone_contents.get(src_zone, [])
        # 距離依存: shuttle_base_distance から計算
        base_dist = self.phonon_params.get('shuttle_base_distance', 1000.0)
        ion_factor = self.phonon_params.get('shuttle_ion_factor', 5.0)
        distance_um = base_dist - ion_factor * dst_chain_length
        axial_per_180 = self.phonon_params.get('shuttle_axial_per_180um', 0.5)
        axial_add = axial_per_180 * (distance_um / 180.0)
        tr_add = self.phonon_params.get('shuttle_transversal', 0.05)
        for ion in ions_in_chain:
            self.phonon_axial[ion] += axial_add
            self.phonon_transversal[ion] += tr_add

    def record_gate_phonon(self, gate_type: str, targets: List[int]):
        """ゲート実行時のフォノン数を記録"""
        if not self.phonon_params:
            return
        phonons = {ion: (self.phonon_axial.get(ion, 0.0), self.phonon_transversal.get(ion, 0.0))
                   for ion in targets}
        self.phonon_at_gates.append({
            "gate": f"{gate_type}({','.join(str(t) for t in targets)})",
            "phonons": phonons,
        })

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
        self._reorder_optimizer = None
        if self.config.memory_reorder:
            from memory_reorder_optimizer import MemoryReorderOptimizer
            self._reorder_optimizer = MemoryReorderOptimizer(architecture)
        # Interleave用: MS gate後にスタビライザを切り替える
        self._last_ms_stabilizer: Optional[int] = None
        self._preferred_stabilizer: Optional[int] = None

    def _evict_ions(self, zone_id: int, ions: list, state: QCCDState,
                    protected: set = None) -> List[Operation]:
        """退避戦略に基づいて複数イオンをLIZから退避する

        Returns:
            退避に必要な操作列
        """
        ops = []
        if self.config.eviction_strategy == "balanced":
            eviction_plan = state.plan_evictions(zone_id, ions, protected=protected)
            for ion, dst in eviction_plan:
                ops.extend(self._move_ion_to_zone(ion, dst, state))
        else:
            # nearest: 従来のBFS
            for ion in ions:
                evict_dst = state.find_eviction_destination(zone_id)
                if evict_dst is not None:
                    ops.extend(self._move_ion_to_zone(ion, evict_dst, state))
        return ops

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
        # フォノン追跡を有効化（デフォルトパラメータ）
        state.phonon_params = {
            'split_axial': 10.0, 'split_transversal': 0.05,
            'merge_axial': 10.0, 'merge_transversal': 0.05,
            'swap_axial': 0.1, 'swap_transversal': 0.1,
            'shuttle_axial_per_180um': 0.5, 'shuttle_transversal': 0.05,
            'shuttle_base_distance': 1000.0, 'shuttle_ion_factor': 5.0,
        }
        for ion_id, zone_id in initial_placement.items():
            state.place_ion_any(ion_id, zone_id)

        dag.reset_execution()
        ops: List[Operation] = []
        max_iterations = len(dag.nodes) * 200
        self._last_state = state  # フォノン結果取得用

        # Measure バッチ化: 遅延中の Measure ノード情報
        # batch_size > 1 の場合、Measure を即時実行せず溜めてからバッチ実行
        batch_size = self.config.measure_batch_size
        # LIZ容量でクランプ
        liz_cap = self.architecture.get_liz_zones()[0].capacity
        batch_size = min(batch_size, liz_cap)
        # バッチRESET済みアンシラ: flush時に自動RESETしたアンシラを記録し、
        # DAG上のRESETノードが後から出てきた場合にスキップする
        batch_reset_done: Set[int] = set()
        # 補助量子ビットのユニーク数でクランプ
        # reuse回路では同じアンシラが複数回MEASUREされるため、
        # 同時測定可能な数はユニークなアンシラ数が上限
        measure_ancillas = set()
        for n in dag.nodes.values():
            if isinstance(n.operation, MeasureOp):
                for t in n.operation.targets:
                    measure_ancillas.add(t)
        if len(measure_ancillas) < batch_size:
            batch_size = 1
        # 測定待ちリスト: [(ancilla_ion_id, MeasureOp, node_id), ...]
        pending_measures: List[tuple] = []

        for _ in range(max_iterations):
            if dag.is_complete() and not pending_measures:
                break

            # DAG完了だが測定待ちが残っている → フラッシュ
            if dag.is_complete() and pending_measures:
                ops.extend(self._flush_pending_measures(pending_measures, state, dag, batch_reset_done))
                pending_measures.clear()
                break

            front_layer = dag.get_front_layer()

            # バッチ化: Measureが実行可能でも遅延させるか判定
            node = self._find_executable_node(dag, front_layer, state)
            if node is not None:
                if isinstance(node.operation, MeasureOp) and batch_size > 1:
                    # Measureを遅延: DAG上では実行済みにするが、ops には追加しない
                    ancilla_id = node.operation.targets[0]
                    # 同じアンシラが既にpendingにいる場合はバッチ化せず先にフラッシュ
                    pending_ancillas = set(item[0] for item in pending_measures)
                    if ancilla_id in pending_ancillas:
                        ops.extend(self._flush_pending_measures(pending_measures, state, dag, batch_reset_done))
                        pending_measures.clear()
                    pending_measures.append((ancilla_id, node.operation, node.id))
                    dag.execute(node.id)

                    # アンシラをLIZからMemoryに退避（次のスタビライザに進むため）
                    if state.is_in_liz(ancilla_id):
                        liz_zone = state.locate(ancilla_id)
                        evict_dst = state.find_eviction_destination(liz_zone, ancilla_id)
                        if evict_dst is not None:
                            ops.extend(self._move_ion_to_zone(ancilla_id, evict_dst, state))

                    # バッチが溜まったら一括測定
                    if len(pending_measures) >= batch_size:
                        ops.extend(self._flush_pending_measures(pending_measures, state, dag, batch_reset_done))
                        pending_measures.clear()
                elif isinstance(node.operation, ResetOp) and batch_size > 1:
                    # バッチRESET済みのアンシラなら DAG上でスキップ
                    reset_ancilla = node.operation.targets[0]
                    if reset_ancilla in batch_reset_done:
                        dag.execute(node.id)
                        batch_reset_done.discard(reset_ancilla)
                        continue
                    else:
                        ops.append(self._annotate_gate(node.operation, state))
                        dag.execute(node.id)
                else:
                    ops.append(self._annotate_gate(node.operation, state))
                    dag.execute(node.id)

                    # Interleave: MS gate実行後にスタビライザを切り替え
                    if self.config.interleave and isinstance(node.operation, GateOp) and node.operation.type == 'MS':
                        current_stab = node.stabilizer_id
                        if current_stab is not None:
                            self._last_ms_stabilizer = current_stab

                    # Measure/Reset 直後にMemory内先読み整列を挿入
                    if self._reorder_optimizer and isinstance(node.operation, (MeasureOp, ResetOp)):
                        reorder_ops = self._reorder_optimizer.generate_reorder_ops(
                            dag, state, node.id
                        )
                        ops.extend(reorder_ops)
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

    def _flush_pending_measures(
        self,
        pending_measures: List[tuple],
        state: QCCDState,
        dag: DAG = None,
        batch_reset_done: Set[int] = None
    ) -> List[Operation]:
        """
        測定待ちアンシラを全てLIZに集めて一括Measure + 一括Resetする

        Interleaveモードでは、回路側のRESETに依存せず、MEASURE直後に
        コンパイラが自動的にRESETを挿入する。アンシラがまだLIZ上にいる
        ため搬送コストはゼロ。DAG上の対応するRESETノードはスキップする。

        Args:
            pending_measures: [(ancilla_id, MeasureOp, node_id), ...]
            state: 現在の QCCDState
            dag: DAG（RESETノードのスキップ用、省略可）

        Returns:
            移動 + Measure + Reset の操作列
        """
        ops: List[Operation] = []
        target_liz = self.architecture.get_liz_zones()[0].id

        # LIZ 上の非ターゲットイオンを退避
        ancilla_set = set(item[0] for item in pending_measures)
        extra_ions = [ion for ion in state.zone_contents.get(target_liz, [])
                      if ion not in ancilla_set]
        if extra_ions:
            ops.extend(self._evict_ions(target_liz, extra_ions, state, protected=ancilla_set))

        # アンシラを全てLIZに集める
        for ancilla_id, _, _ in pending_measures:
            if state.locate(ancilla_id) != target_liz:
                ops.extend(self._move_ion_to_zone(ancilla_id, target_liz, state))

        # バッチMeasure: 全アンシラを一括測定
        all_targets = [item[0] for item in pending_measures]
        batch_measure = MeasureOp(targets=all_targets, zone_id=target_liz)
        ops.append(batch_measure)

        # バッチReset: MEASURE直後に無条件で全アンシラをRESET
        # アンシラはまだLIZ上にいるので搬送不要
        # 最後のバッチ（DAG完了直前）ではRESET不要
        if dag is not None:
            # DAG上に残りのノードがある = まだスタビライザが残っている → RESET必要
            remaining_after = set(dag.nodes.keys()) - dag.get_executed()
            has_remaining_work = any(
                not isinstance(dag.nodes[nid].operation, ResetOp)
                for nid in remaining_after
            )
            if has_remaining_work:
                batch_reset = ResetOp(targets=all_targets, zone_id=target_liz)
                ops.append(batch_reset)
                # バッチRESET済みとして記録（後からDAGのRESETノードが出たらスキップ）
                if batch_reset_done is not None:
                    batch_reset_done.update(all_targets)

            # DAG上の対応するRESETノードをスキップ（実行済みとしてマーク）
            front_layer = dag.get_front_layer()
            for node_id in list(front_layer):
                node = dag.nodes[node_id]
                if isinstance(node.operation, ResetOp):
                    reset_ancilla = node.operation.targets[0]
                    if reset_ancilla in ancilla_set:
                        dag.execute(node_id)

        return ops

    # ------------------------------------------
    # Gate Execution
    # ------------------------------------------

    def _find_executable_node(
        self, dag: DAG, front_layer: Set[int], state: QCCDState
    ) -> Optional[DAGNode]:
        """Front Layerから実行可能なノードを探す

        interleaveモード時: MS gate実行後に他のスタビライザのノードを優先
        """
        def is_executable(node):
            op = node.operation
            if isinstance(op, GateOp) and state.can_execute(op):
                return True
            if isinstance(op, (ResetOp, MeasureOp)) and state.can_execute_reset_measure(op):
                return True
            return False

        if not self.config.interleave:
            # 従来通り: min_id
            for node_id in sorted(front_layer):
                node = dag.nodes[node_id]
                if is_executable(node):
                    return node
            return None

        # === Interleave モード ===
        # MS gate 実行直後 → 他のスタビライザを優先
        preferred = []   # 優先スタビライザのノード
        fallback = []    # それ以外

        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if not is_executable(node):
                continue
            if self._last_ms_stabilizer is not None and node.stabilizer_id != self._last_ms_stabilizer:
                preferred.append(node)
            else:
                fallback.append(node)

        # 優先グループから選択、なければフォールバック
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return None

    def _annotate_gate(self, op: Operation, state: QCCDState) -> Operation:
        """GateOp/ResetOp/MeasureOpにzone_idを付与した新しいオペレーションを返す"""
        if isinstance(op, GateOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            # MSゲート実行時のフォノン数を記録
            if op.type == 'MS':
                state.record_gate_phonon(op.type, op.targets)
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

        優先順: interleaveモード時は他のスタビライザを優先、通常はnode_id昇順
        """
        # Interleaveモード: 他のスタビライザのノードを先に処理
        sorted_ids = sorted(front_layer)
        if self.config.interleave and self._last_ms_stabilizer is not None:
            preferred = [nid for nid in sorted_ids
                         if dag.nodes[nid].stabilizer_id != self._last_ms_stabilizer]
            others = [nid for nid in sorted_ids
                      if dag.nodes[nid].stabilizer_id == self._last_ms_stabilizer]
            sorted_ids = preferred + others

        for node_id in sorted_ids:
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
                        # RESET/MEASURE: ターゲット以外のイオンを退避（左右交互）
                        extra_ions = [
                            ion for ion in state.zone_contents.get(target_liz, [])
                            if ion not in target_set
                        ]
                        if extra_ions:
                            return self._evict_ions(target_liz, extra_ions, state, protected=target_set)
                    # Gate or RESET/MEASURE with no extra ions → 実行可能なはず
                    continue
                # 異なるLIZ → 一方のLIZに集める（バグ修正: 旧実装の無限ループを解消）
                target_liz = state.locate(targets[0])
                result = []
                for ion in targets:
                    if state.locate(ion) != target_liz:
                        src_liz = state.locate(ion)
                        evict_dst = state.find_eviction_destination(src_liz, ion)
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
                extra_ions = [ion for ion in state.zone_contents.get(target_liz, [])
                              if ion not in target_set]
                if extra_ions:
                    result.extend(self._evict_ions(target_liz, extra_ions, state, protected=target_set))

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
                result.extend(self._evict_ions(
                    target_liz, candidates[:n_to_evict], state, protected=target_set))

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

        1ホップ（隣接M/L間）の場合: src → temp → dst
        複数ホップの場合: 経路上の中間ゾーンを1ホップずつ進む
        """
        src_zone = state.locate(ion_id)
        if src_zone == dst_zone:
            return []

        # 経路を計算: src → ... → dst
        path = self.architecture.get_path(src_zone, dst_zone)
        # 経路上の非Tempゾーン（M/L）を中間目標として抽出
        # 各中間目標に対して1ホップずつ移動する
        waypoints = [z for z in path if not self.architecture.is_temp(z)]

        ops = []
        for i in range(1, len(waypoints)):
            ops.extend(self._move_ion_one_hop(ion_id, waypoints[i], state))

        return ops

    def _move_ion_one_hop(
        self, ion_id: int, dst_zone: int, state: QCCDState
    ) -> List[Operation]:
        """
        イオンを隣接する非Tempゾーンへ1ホップで移動する

        物理的な1ホップ: src(M/L) → temp(T) → dst(M/L)

        dst_zone が LIZ で容量が足りない場合、既存イオンを一時退避して
        通過させた後に戻す。
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

        # Step 0: dst_zone に空きがない場合の処理
        # TODO: LIZ通過時の一時退避を実装（複数LIZ対応）
        # 現在は容量チェックのみ行い、超過時はそのまま進む（Validation で検出）

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
                assert ion_pos == len(src_contents) - 1, \
                    f"ion {ion_id} should be at right end (pos={ion_pos}, len={len(src_contents)})"
                split_ref = src_contents[ion_pos - 1]
                split_op = SplitOp(ion_id=split_ref, dst_zone_id=temp_zone, src_zone_id=src_zone)
                ops.append(split_op)
                state.split_chain(split_ref, temp_zone)
            else:
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
        state.apply_shuttle_phonon(ion_id, len(dst_contents))

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
    flat_ops: List[Operation] = field(default_factory=list)  # Scheduler前の元の生成順
    phonon_data: Optional[Dict] = None  # フォノン追跡結果
    final_placement: Optional[Dict[int, int]] = None  # ラウンド終了時のion配置 (ion_id -> zone_id)
    final_zone_contents: Optional[Dict[int, List[int]]] = None  # ゾーン内の並び順 (zone_id -> [ion_ids])

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

        # フォノン結果を抽出
        phonon_data = None
        if hasattr(self._planner, '_last_state') and self._planner._last_state.phonon_params:
            st = self._planner._last_state
            max_ax = max(st.phonon_axial.values()) if st.phonon_axial else 0.0
            max_tr = max(st.phonon_transversal.values()) if st.phonon_transversal else 0.0
            phonon_data = {
                'max_axial': max_ax,
                'max_transversal': max_tr,
                'per_ion_axial': dict(st.phonon_axial),
                'per_ion_transversal': dict(st.phonon_transversal),
                'at_gates': st.phonon_at_gates,
            }

        # 最終配置を抽出（zone_contents の順序を保持）
        final_placement = None
        final_zone_contents = None
        if hasattr(self._planner, '_last_state'):
            final_placement = dict(self._planner._last_state.ion_locations)
            final_zone_contents = {
                z: list(ions) for z, ions in self._planner._last_state.zone_contents.items()
                if ions
            }

        return CompileResult(timesteps=timesteps, stats=stats, config=self.config,
                             flat_ops=flat_ops, phonon_data=phonon_data,
                             final_placement=final_placement,
                             final_zone_contents=final_zone_contents)

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
