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

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
from ast_def import (
    Location, QubitDecl, Operation,
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
    
    各フィールドは戦略名を指定。
    アブレーションスタディのため、個別にON/OFF可能。
    
    新しい戦略追加時の変更箇所:
    1. validate() 内の valid_xxx セット
    2. 対応するディスパッチャメソッド（例: _find_executable_gate）
    3. この docstring
    
    Attributes:
        gate_selection: ゲート選択戦略
            - "min_id": Front Layerから最小IDを選択（baseline）
            - "locality_aware": LIZにいるイオンを使うゲートを優先
        
        eviction_policy: 退避イオン選択戦略
            - "first": 候補リストの先頭から選択（baseline）
            - "future_aware": 将来使われにくいイオンを優先
        
        target_liz_selection: 目標LIZ選択戦略
            - "first_available": 空きがあるLIZ（baseline）
            - "ion_aware": 関連イオンが多いLIZを優先
        
        batch_reset_measure: RESET/MEASUREバッチ処理
            - False: 1つずつ実行（baseline）
            - True: Front Layer内の複数RESET/MEASUREをまとめてLIZに運び実行
        
        batch_size: バッチサイズ
            - 0: LIZ capacityを使用（デフォルト）
            - 1以上: 指定したサイズでバッチ処理
    """
    gate_selection: str = "min_id"
    eviction_policy: str = "first"
    target_liz_selection: str = "first_available"
    batch_reset_measure: bool = False
    batch_size: int = 0  # 0 = LIZ capacityを使用
    
    def to_dict(self) -> dict:
        """JSON出力用に辞書化"""
        return asdict(self)
    
    def __repr__(self) -> str:
        return (f"CompilerConfig(gate={self.gate_selection}, "
                f"evict={self.eviction_policy}, "
                f"liz={self.target_liz_selection}, "
                f"batch={self.batch_reset_measure}, "
                f"batch_size={self.batch_size})")
    
    @classmethod
    def baseline(cls) -> "CompilerConfig":
        """全て愚直な戦略（比較のベースライン）"""
        return cls()
    
    @classmethod
    def optimized(cls) -> "CompilerConfig":
        """全ての最適化を有効化"""
        return cls(
            gate_selection="locality_aware",
            eviction_policy="future_aware",
            target_liz_selection="ion_aware",
        )
    
    # TODO: 新しい戦略を追加する際は以下も更新すること
    #   - _find_executable_gate() 内のディスパッチ分岐
    #   - CompilerConfig の docstring
    def validate(self):
        """設定値の検証"""
        valid_gate = {"min_id", "locality_aware"}
        valid_evict = {"first", "future_aware"}
        valid_liz = {"first_available", "ion_aware"}
        
        if self.gate_selection not in valid_gate:
            raise ValueError(f"Invalid gate_selection: {self.gate_selection}. Valid: {valid_gate}")
        if self.eviction_policy not in valid_evict:
            raise ValueError(f"Invalid eviction_policy: {self.eviction_policy}. Valid: {valid_evict}")
        if self.target_liz_selection not in valid_liz:
            raise ValueError(f"Invalid target_liz_selection: {self.target_liz_selection}. Valid: {valid_liz}")


# ==========================================
# QCCD Architecture (入力として与える構成)
# ==========================================

@dataclass
class Zone:
    """ゾーンの定義"""
    id: int
    zone_type: str      # "L" (LIZ), "M" (Memory), or "T" (Temporary)
    capacity: int       # 最大イオン数
    position: int       # 1D配置での位置 (リストのインデックス)


class QCCDArchitecture:
    """
    1D QCCDアーキテクチャの構成定義
    
    物理的な電極配置に基づく7ゾーン構成:
      T(20) - M(20) - T(20) - L(4) - T(20) - M(20) - T(20)
    
    入力形式: [(type, capacity), ...]
      - type: "M" (Memory), "L" (LIZ), or "T" (Temporary)
      - capacity: そのゾーンの最大イオン数
      - 位置はリストのインデックスで自動決定
      - 隣接関係は index ± 1 で自動決定
    
    Temporary Zoneの制約:
      - Split/Merge操作の一時置き場として使用
      - 使用前は常に空であること（不変条件）
      - 初期配置には使用しない
      - Swapは実行不可
    """
    
    def __init__(self, zone_specs: List[Tuple[str, int]]):
        """
        Args:
            zone_specs: [(type, capacity), ...] のリスト
                       type は "M" (Memory), "L" (LIZ), or "T" (Temporary)
        """
        self.zones: List[Zone] = []
        for i, (zone_type, capacity) in enumerate(zone_specs):
            if zone_type not in ("M", "L", "T"):
                raise ValueError(f"Invalid zone type '{zone_type}'. Must be 'M', 'L', or 'T'.")
            self.zones.append(Zone(
                id=i,
                zone_type=zone_type,
                capacity=capacity,
                position=i
            ))
        
        self._zone_by_id: Dict[int, Zone] = {z.id: z for z in self.zones}
        self._liz_zones: List[Zone] = [z for z in self.zones if z.zone_type == "L"]
        self._memory_zones: List[Zone] = [z for z in self.zones if z.zone_type == "M"]
        self._temp_zones: List[Zone] = [z for z in self.zones if z.zone_type == "T"]
        
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
    
    def get_temp_zones(self) -> List[Zone]:
        """全Temporaryゾーンを取得"""
        return self._temp_zones
    
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
    
    def is_temp(self, zone_id: int) -> bool:
        """指定ゾーンがTemporaryかどうか"""
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "T"
    
    def is_memory(self, zone_id: int) -> bool:
        """指定ゾーンがMemoryかどうか"""
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "M"
    
    def get_adjacent_temp(self, zone_id: int, direction: str) -> Optional[int]:
        """
        指定ゾーンの左("left")または右("right")の隣接Temporaryゾーンを取得
        
        Args:
            zone_id: 基準ゾーン
            direction: "left" or "right"
        
        Returns:
            隣接Tempゾーンのid。存在しない場合はNone
        
        例: T(0)-M(1)-T(2)-L(3)-T(4)-M(5)-T(6)
            get_adjacent_temp(1, "left")  → 0
            get_adjacent_temp(1, "right") → 2
            get_adjacent_temp(3, "left")  → 2
        """
        if direction == "left":
            adj_id = zone_id - 1
        elif direction == "right":
            adj_id = zone_id + 1
        else:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'left' or 'right'.")
        
        if adj_id < 0 or adj_id >= len(self.zones):
            return None
        
        if self.zones[adj_id].zone_type == "T":
            return adj_id
        return None
    
    def get_path(self, src_zone: int, dst_zone: int) -> List[int]:
        """
        srcからdstへの経路（通過するゾーンIDのリスト、両端含む）を返す
        1D配置なので単純に連番
        
        例: get_path(1, 3) → [1, 2, 3]
            get_path(5, 3) → [5, 4, 3]
        """
        if src_zone <= dst_zone:
            return list(range(src_zone, dst_zone + 1))
        else:
            return list(range(src_zone, dst_zone - 1, -1))
    
    def find_nearest_memory(self, zone_id: int) -> Optional[int]:
        """指定ゾーンから最も近いMemoryゾーンを探す"""
        # BFS で探索（Tempゾーンを通過する）
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
        """T-M-T-L-T-M-T 構成（物理的な電極配置に基づく7ゾーン）"""
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
# QCCD State (実行時に変化する状態)
# ==========================================

@dataclass
class QCCDState:
    """
    QCCDの実行時状態（7ゾーン対応: T-M-T-L-T-M-T）
    
    イオンの位置を追跡し、ゲート実行可能性を判定
    
    不変条件:
    - Temporary Zoneは操作前後で空であること
    - move_ionは隣接ゾーン間のみ
    - 初期配置はMemory Zoneのみ
    - SwapはTemporary Zoneでは実行不可
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
        """
        イオンを指定ゾーンに配置（初期配置用）
        
        制約: Memory Zoneのみ配置可能
        """
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
    
    def can_execute_reset_measure(self, op) -> bool:
        """
        RESET/MEASUREが実行可能か
        
        条件:
        1. ターゲットが全て同じLIZにいる
        2. そのLIZにはターゲット以外のイオンがいない
           (RESET/MEASUREはLIZ上の全イオンに影響するため)
        """
        targets = op.targets
        if not targets:
            return True
        
        first_zone = self.locate(targets[0])
        if not self.architecture.is_liz(first_zone):
            return False
        
        # 全ターゲットが同じLIZにいるか
        if not all(self.locate(t) == first_zone for t in targets):
            return False
        
        # LIZ上にターゲット以外のイオンがいないか
        liz_contents = set(self.zone_contents[first_zone])
        target_set = set(targets)
        return liz_contents == target_set
    
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
        退避先のMemoryゾーンを探す（Tempゾーンは対象外）
        from_zone_id に近いMemoryゾーンで空きがあるものを優先
        """
        # 最寄りのMemoryを探す（BFSでTempを通過）
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
        """
        イオンを隣接ゾーンへ移動（状態を更新）
        
        制約: 隣接ゾーン間のみ（|src - dst| == 1）
        
        シャトル方向によって挿入位置が決まる:
        - 右方向（src < dst）: 左端に入る → insert(0)
        - 左方向（src > dst）: 右端に入る → append
        """
        src_zone_id = self.locate(ion_id)
        dst_zone = self.architecture.get_zone(dst_zone_id)
        
        if dst_zone is None:
            raise ValueError(f"Destination zone {dst_zone_id} does not exist")
        
        # 隣接チェック
        if abs(src_zone_id - dst_zone_id) != 1:
            raise ValueError(
                f"move_ion only allows adjacent zones. "
                f"src={src_zone_id}, dst={dst_zone_id} (distance={abs(src_zone_id - dst_zone_id)}). "
                f"Use transfer_ion for multi-hop moves."
            )
        
        if len(self.zone_contents[dst_zone_id]) >= dst_zone.capacity:
            raise ValueError(f"Destination zone {dst_zone_id} is at capacity")
        
        # 移動を実行
        self.zone_contents[src_zone_id].remove(ion_id)
        
        # シャトル方向によって挿入位置を決定
        if src_zone_id < dst_zone_id:
            # 右方向: 左端に入る
            self.zone_contents[dst_zone_id].insert(0, ion_id)
        else:
            # 左方向: 右端に入る
            self.zone_contents[dst_zone_id].append(ion_id)
        
        self.ion_locations[ion_id] = dst_zone_id
    
    def transfer_ion(self, ion_id: int, dst_zone_id: int):
        """
        イオンを任意のゾーンへ移動（長距離対応）
        
        内部で隣接move_ionを繰り返し呼び出す。
        Compiler が ShuttleOp 生成時に状態を更新するために使用。
        （OperationExpander レベルでは move_ion を直接使用）
        """
        src_zone_id = self.locate(ion_id)
        if src_zone_id == dst_zone_id:
            return
        
        path = self.architecture.get_path(src_zone_id, dst_zone_id)
        # path[0] は src なのでスキップ
        for next_zone in path[1:]:
            self.move_ion(ion_id, next_zone)
    
    def get_ion_position_in_zone(self, ion_id: int) -> int:
        """ゾーン内でのイオンの位置（インデックス）を取得"""
        zone_id = self.locate(ion_id)
        return self.zone_contents[zone_id].index(ion_id)
    
    def swap_ions(self, ion_a: int, ion_b: int):
        """
        同一ゾーン内の隣接する2つのイオンの位置を交換（状態を更新）
        
        制約:
        - 隣接イオン同士のみSWAP可能
        - Temporary Zoneでは実行不可
        """
        zone_a = self.locate(ion_a)
        zone_b = self.locate(ion_b)
        
        if zone_a != zone_b:
            raise ValueError(f"Cannot swap ions in different zones: {ion_a} in {zone_a}, {ion_b} in {zone_b}")
        
        # Temp Zone チェック
        if self.architecture.is_temp(zone_a):
            raise ValueError(
                f"Cannot swap ions in Temporary zone {zone_a}. "
                f"Swap is only allowed in Memory and LIZ zones."
            )
        
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
    
    def validate_temp_empty(self, temp_zone_id: int):
        """Temp Zoneが空であることを検証（不変条件チェック）"""
        if not self.architecture.is_temp(temp_zone_id):
            raise ValueError(f"Zone {temp_zone_id} is not a Temporary zone")
        contents = self.zone_contents.get(temp_zone_id, [])
        if contents:
            raise RuntimeError(
                f"Invariant violation: Temp zone {temp_zone_id} should be empty "
                f"but contains {contents}"
            )
    
    def split_chain(self, ion_id: int, dst_zone_id: int):
        """
        イオンチェーンを分割し、片方をdst_zone_idへ移す
        
        方向はdst_zone_idとソースゾーンの位置関係から自動決定:
        - dst_zone_id > source → 右側チェーンがdstへ
        - dst_zone_id < source → 左側チェーンがdstへ
        
        特殊ケース: ゾーンに1イオンのみの場合、そのイオンをdstへ移動
        （物理的にはSplit操作でイオンをゾーンから分離する）
        
        Args:
            ion_id: このイオンが右端になる位置で分割
            dst_zone_id: 分割されたチェーンの移動先（Temporary Zone）
        """
        zone_id = self.locate(ion_id)
        contents = self.zone_contents[zone_id]
        pos = contents.index(ion_id)
        
        # 1イオンのみの場合: そのままdstへ移動
        if len(contents) == 1:
            self.zone_contents[dst_zone_id] = [ion_id]
            self.zone_contents[zone_id] = []
            self.ion_locations[ion_id] = dst_zone_id
            return
        
        # 方向を決定
        move_right = dst_zone_id > zone_id
        
        left_contents = contents[:pos + 1]
        right_contents = contents[pos + 1:]
        
        if move_right:
            # 右チェーンがdstへ
            if not right_contents:
                raise ValueError(
                    f"Cannot split: no right chain to move to dst. "
                    f"ion {ion_id} is at right edge of zone {zone_id}"
                )
            self.zone_contents[zone_id] = left_contents
            self.zone_contents[dst_zone_id] = right_contents
            for ion in right_contents:
                self.ion_locations[ion] = dst_zone_id
        else:
            # 左チェーンがdstへ
            self.zone_contents[zone_id] = right_contents
            self.zone_contents[dst_zone_id] = left_contents
            for ion in left_contents:
                self.ion_locations[ion] = dst_zone_id
    
    def merge_chains(self, zone_left: int, zone_right: int):
        """
        2つの隣接イオンチェーンを結合
        
        結合順序は物理配置に従い、左側が常に前になる。
        結果はMemory/LIZゾーン側に保存され、Temp Zoneは空になる。
        
        Args:
            zone_left: 左側のゾーン
            zone_right: 右側のゾーン
        """
        left_contents = self.zone_contents.get(zone_left, [])
        right_contents = self.zone_contents.get(zone_right, [])
        merged = left_contents + right_contents
        
        # どちらがTemp Zoneかを判定し、非Temp側に保存
        left_is_temp = self.architecture.is_temp(zone_left)
        right_is_temp = self.architecture.is_temp(zone_right)
        
        if left_is_temp and not right_is_temp:
            # 左がTemp → 右(M/L)に保存
            result_zone = zone_right
            empty_zone = zone_left
        elif right_is_temp and not left_is_temp:
            # 右がTemp → 左(M/L)に保存
            result_zone = zone_left
            empty_zone = zone_right
        else:
            # 両方Tempまたは両方非Temp（想定外だがフォールバック: 左に保存）
            result_zone = zone_left
            empty_zone = zone_right
        
        # 状態を更新
        self.zone_contents[result_zone] = merged
        self.zone_contents[empty_zone] = []
        
        # ion_locationsを更新
        for ion in merged:
            self.ion_locations[ion] = result_zone



# ==========================================
# Operation Expander
# ==========================================

class OperationExpander:
    """
    高レベル操作を低レベル操作に展開（7ゾーン対応）
    
    SwapOp → Split + Swap + Merge（Swapは元のM/Lゾーン内で実行、Tempでは不可）
    ShuttleOp → Split + Shuttle + Merge（物理Temp Zoneを経由）
    """
    
    def __init__(self):
        pass
    
    def expand(
        self, 
        operations: List[Operation], 
        architecture: QCCDArchitecture,
        initial_placement: Dict[int, int]
    ) -> List[Operation]:
        """
        高レベル操作を低レベル操作に展開
        
        Args:
            operations: 高レベル操作のリスト
            architecture: QCCDアーキテクチャ
            initial_placement: 初期配置 (ion_id -> zone_id)
        
        Returns:
            展開後の低レベル操作のリスト
        """
        # 状態を初期化
        state = QCCDState(architecture)
        for ion_id, zone_id in initial_placement.items():
            state.place_ion(ion_id, zone_id)
        
        expanded: List[Operation] = []
        
        for op in operations:
            if isinstance(op, SwapOp):
                ion_a, ion_b = op.targets
                zone_a = state.locate(ion_a)
                zone_b = state.locate(ion_b)
                
                if zone_a != zone_b:
                    # 異なるゾーンにいる場合、スキップ
                    continue
                
                # 同じゾーンにいる場合、Split + Swap + Mergeで展開
                expanded.extend(self._expand_swap(op, state))
            elif isinstance(op, ShuttleOp):
                expanded.extend(self._expand_shuttle(op, state))
            else:
                expanded.append(op)
        
        return expanded
    
    def _expand_swap(self, swap_op: SwapOp, state: QCCDState) -> List[Operation]:
        """
        SwapOpを展開（7ゾーン対応）
        
        Swapは元のM/Lゾーン内で実行し、Temp Zoneでは実行しない。
        不要なイオンを隣接Temp Zoneに退避させて2イオンに絞る。
        
        パターン:
        1. 2イオンのみ: Swap のみ
        2. 左端ペア [A,B,...]: 右チェインを右Tempへ退避
        3. 右端ペア [...,C,D]: 左チェインを左Tempへ退避
        4. 中間ペア [A,B,C,D]: 左を左Temp、右を右Tempへ退避
        """
        ion_a, ion_b = swap_op.targets
        zone_id = state.locate(ion_a)
        contents = state.zone_contents[zone_id]
        arch = state.architecture
        
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        
        # 左右を確定
        left_pos, right_pos = min(pos_a, pos_b), max(pos_a, pos_b)
        left_ion = contents[left_pos]
        right_ion = contents[right_pos]
        
        # 隣接Tempゾーンを取得
        left_temp = arch.get_adjacent_temp(zone_id, "left")
        right_temp = arch.get_adjacent_temp(zone_id, "right")
        
        ops: List[Operation] = []
        
        if len(contents) == 2:
            # パターン1: 2イオンのみ → そのままSwap
            ops.append(SwapOp(targets=[left_ion, right_ion]))
            state.swap_ions(left_ion, right_ion)
            
        elif left_pos == 0:
            # パターン2: 左端ペア [A, B, C, D] の Swap(A, B)
            # 右チェイン[C,D]を右Tempへ退避、zone_idでSwap
            assert right_temp is not None, f"No right temp zone for zone {zone_id}"
            
            # Split(B) → zone_id: [A, B], right_temp: [C, D]
            ops.append(SplitOp(ion_id=right_ion, dst_zone_id=right_temp))
            state.split_chain(right_ion, right_temp)
            
            # Swap in zone_id (M/L zone)
            ops.append(SwapOp(targets=[left_ion, right_ion]))
            state.swap_ions(left_ion, right_ion)
            
            # Merge: zone_id(M/L) + right_temp(T) → zone_idに保存
            ops.append(MergeOp(zone_left=zone_id, zone_right=right_temp))
            state.merge_chains(zone_id, right_temp)
            
        elif right_pos == len(contents) - 1:
            # パターン3: 右端ペア [A, B, C, D] の Swap(C, D)
            # 左チェイン[A,B]を左Tempへ退避、zone_idでSwap
            assert left_temp is not None, f"No left temp zone for zone {zone_id}"
            
            split_ion = contents[left_pos - 1]
            # Split(B) → left_temp: [A, B], zone_id: [C, D]
            ops.append(SplitOp(ion_id=split_ion, dst_zone_id=left_temp))
            state.split_chain(split_ion, left_temp)
            
            # Swap in zone_id (M/L zone)
            ops.append(SwapOp(targets=[left_ion, right_ion]))
            state.swap_ions(left_ion, right_ion)
            
            # Merge: left_temp(T) + zone_id(M/L) → zone_idに保存
            ops.append(MergeOp(zone_left=left_temp, zone_right=zone_id))
            state.merge_chains(left_temp, zone_id)
            
        else:
            # パターン4: 中間ペア [A, B, C, D] の Swap(B, C)
            # 左チェイン[A]を左Temp、右チェイン[D]を右Tempへ退避
            assert left_temp is not None, f"No left temp zone for zone {zone_id}"
            assert right_temp is not None, f"No right temp zone for zone {zone_id}"
            
            # Split1: [A]を左Tempへ
            split_ion1 = contents[left_pos - 1]
            ops.append(SplitOp(ion_id=split_ion1, dst_zone_id=left_temp))
            state.split_chain(split_ion1, left_temp)
            # Now: left_temp: [A], zone_id: [B, C, D]
            
            # Split2: [D]を右Tempへ
            ops.append(SplitOp(ion_id=right_ion, dst_zone_id=right_temp))
            state.split_chain(right_ion, right_temp)
            # Now: left_temp: [A], zone_id: [B, C], right_temp: [D]
            
            # Swap in zone_id (M/L zone)
            ops.append(SwapOp(targets=[left_ion, right_ion]))
            state.swap_ions(left_ion, right_ion)
            # Now: left_temp: [A], zone_id: [C, B], right_temp: [D]
            
            # Merge: zone_id(M/L) + right_temp(T) → zone_idに保存
            ops.append(MergeOp(zone_left=zone_id, zone_right=right_temp))
            state.merge_chains(zone_id, right_temp)
            # Now: left_temp: [A], zone_id: [C, B, D]
            
            # Merge: left_temp(T) + zone_id(M/L) → zone_idに保存
            ops.append(MergeOp(zone_left=left_temp, zone_right=zone_id))
            state.merge_chains(left_temp, zone_id)
            # Now: zone_id: [A, C, B, D]
        
        return ops
    
    def _expand_shuttle(self, shuttle_op: ShuttleOp, state: QCCDState) -> List[Operation]:
        """
        ShuttleOpを展開（7ゾーン対応）
        
        物理的な動作（1ホップ = 非Tゾーン → Tゾーン → 非Tゾーン）:
        1. srcからイオンを分離し隣接Temp Zoneへ (Split)
        2. Temp ZoneからdstへイオンがE極上を移動 (Shuttle)
        3. dstチェインと結合 (Merge)
        
        M(1) → L(3) の場合: Split(1→2), Shuttle(2→3), Merge(2→3)
        M(5) → L(3) の場合: Split(5→4), Shuttle(4→3), Merge(4→3)
        """
        ion_id = shuttle_op.target
        src_zone = state.locate(ion_id)
        dst_zone = shuttle_op.dst.trap
        is_right = src_zone < dst_zone
        
        # 方向に応じたTemp Zone決定
        if is_right:
            temp_zone = state.architecture.get_adjacent_temp(src_zone, "right")
        else:
            temp_zone = state.architecture.get_adjacent_temp(src_zone, "left")
        
        assert temp_zone is not None, \
            f"No adjacent temp zone for zone {src_zone} in direction {'right' if is_right else 'left'}"
        
        ops: List[Operation] = []
        
        src_contents = list(state.zone_contents[src_zone])
        dst_contents = list(state.zone_contents[dst_zone])
        
        # === Step 1: Split (srcからion_idを分離してTemp Zoneへ) ===
        
        if len(src_contents) == 1:
            # src_zoneに1イオンのみ: Split でイオンをTempへ分離
            ops.append(SplitOp(ion_id=ion_id, dst_zone_id=temp_zone))
            state.split_chain(ion_id, temp_zone)
        else:
            # 複数イオン
            ion_pos = src_contents.index(ion_id)
            
            if is_right:
                # 右方向: イオンは右端にいるはず
                assert ion_pos == len(src_contents) - 1, \
                    f"ion {ion_id} should be at right end for right-direction shuttle, but at pos {ion_pos}"
                
                # ion_idの左隣で分割 → src: [...], temp(right): [ion_id]
                split_ion = src_contents[ion_pos - 1]
                ops.append(SplitOp(ion_id=split_ion, dst_zone_id=temp_zone))
                state.split_chain(split_ion, temp_zone)
            else:
                # 左方向: イオンは左端にいるはず
                assert ion_pos == 0, \
                    f"ion {ion_id} should be at left end for left-direction shuttle, but at pos {ion_pos}"
                
                # ion_id自体で分割 → temp(left): [ion_id], src: [残り]
                ops.append(SplitOp(ion_id=ion_id, dst_zone_id=temp_zone))
                state.split_chain(ion_id, temp_zone)
        
        # === Step 2: Shuttle (Temp Zone → dst方向への物理的移動) ===
        ops.append(ShuttleOp(target=ion_id, dst=shuttle_op.dst, dst_chain_length=len(dst_contents)))
        
        # === Step 3: Merge (Temp Zone + dst → dstに保存) ===
        left_z = min(temp_zone, dst_zone)
        right_z = max(temp_zone, dst_zone)
        ops.append(MergeOp(zone_left=left_z, zone_right=right_z))
        state.merge_chains(left_z, right_z)
        
        return ops


# ==========================================
# Compiler
# ==========================================

@dataclass
class CompileResult:
    """コンパイル結果"""
    operations: List[Operation]  # 実行順の操作列
    stats: Dict[str, int]        # 統計情報
    config: CompilerConfig = None  # 使用した設定（オプション）


class QCCDCompiler:
    """
    DAG → Executable な操作列へコンパイル
    
    アルゴリズム:
    1. Front Layerから実行可能なゲートを探す
    2. 実行可能なゲートがあれば実行
    3. なければ、必要なイオンをLIZへ移動（SHUTTLE挿入）
       - LIZが満杯なら、まず退避操作を挿入
    4. 全ゲート実行まで繰り返し
    
    config により各決定ポイントの戦略を切り替え可能。
    """
    
    def __init__(self, architecture: QCCDArchitecture, config: CompilerConfig = None):
        self.architecture = architecture
        self.config = config or CompilerConfig.baseline()
        self.config.validate()
    
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
            
            # バッチ処理: RESET/MEASUREをまとめて処理
            if self.config.batch_reset_measure:
                batch_ops, batch_count = self._try_batch_reset_measure(dag, front_layer, state)
                if batch_count > 0:
                    operations.extend(batch_ops)
                    stats["gates"] += batch_count
                    continue
            
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
        
        # 高レベル操作を低レベル操作に展開
        expander = OperationExpander()
        expanded_operations = expander.expand(operations, self.architecture, initial_placement)
        
        # 展開後の統計を再計算
        expanded_stats = self._calculate_stats(expanded_operations)
        expanded_stats["iterations"] = stats["iterations"]
        
        return CompileResult(operations=expanded_operations, stats=expanded_stats, config=self.config)
    
    def _calculate_stats(self, operations: List[Operation]) -> Dict[str, int]:
        """展開後の操作列から統計情報を計算"""
        stats = {
            "gates": 0,
            "shuttles": 0,
            "swaps": 0,
            "splits": 0,
            "merges": 0,
        }
        for op in operations:
            if isinstance(op, GateOp):
                stats["gates"] += 1
            elif isinstance(op, (ResetOp, MeasureOp)):
                stats["gates"] += 1  # ゲートとしてカウント
            elif isinstance(op, ShuttleOp):
                stats["shuttles"] += 1
            elif isinstance(op, SwapOp):
                stats["swaps"] += 1
            elif isinstance(op, SplitOp):
                stats["splits"] += 1
            elif isinstance(op, MergeOp):
                stats["merges"] += 1
        return stats
    
    # ==========================================
    # Gate Selection Strategies
    # ==========================================
    # TODO: 新しいgate_selection戦略を追加する際は以下も更新すること
    #   - CompilerConfig.validate() 内の valid_gate
    #   - CompilerConfig の docstring
    
    def _find_executable_gate(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> Optional[DAGNode]:
        """
        Front Layerから実行可能なゲートを探す（ディスパッチャ）
        
        config.gate_selection に応じて適切な戦略を呼び出す
        """
        if self.config.gate_selection == "min_id":
            return self._find_gate_min_id(dag, front_layer, state)
        elif self.config.gate_selection == "locality_aware":
            return self._find_gate_locality_aware(dag, front_layer, state)
        else:
            raise ValueError(f"Unknown gate selection strategy: {self.config.gate_selection}")
    
    def _find_gate_min_id(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> Optional[DAGNode]:
        """
        ゲート選択戦略: min_id (baseline)
        
        Front Layerから最小IDの実行可能なゲートを選択。
        決定的な動作のため、デバッグや比較に適している。
        
        バッチ処理が有効な場合:
        - RESET/MEASUREがbatch_size個溜まるまで待機（他のゲートを優先）
        - ただし、他のゲートがない場合はRESET/MEASUREを実行
        """
        skip_reset = False
        skip_measure = False
        
        if self.config.batch_reset_measure:
            liz = self.architecture.get_liz_zones()[0]
            effective_batch_size = self.config.batch_size if self.config.batch_size > 0 else liz.capacity
            
            # Front LayerのRESET/MEASURE数をカウント
            reset_count = sum(1 for nid in front_layer 
                             if isinstance(dag.nodes[nid].operation, ResetOp))
            measure_count = sum(1 for nid in front_layer 
                               if isinstance(dag.nodes[nid].operation, MeasureOp))
            
            # 他に実行可能なゲートがあるか確認
            has_other_gate = False
            for node_id in front_layer:
                node = dag.nodes[node_id]
                if isinstance(node.operation, GateOp):
                    if state.can_execute(node.operation):
                        has_other_gate = True
                        break
            
            # RESET/MEASUREの待機判定
            # batch_size未満で、他のゲートがあればスキップ
            if 0 < reset_count < effective_batch_size and has_other_gate:
                skip_reset = True
            if 0 < measure_count < effective_batch_size and has_other_gate:
                skip_measure = True
        
        # ゲート選択
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if isinstance(node.operation, GateOp):
                if state.can_execute(node.operation):
                    return node
            elif isinstance(node.operation, ResetOp):
                if skip_reset:
                    continue
                if state.can_execute_reset_measure(node.operation):
                    return node
            elif isinstance(node.operation, MeasureOp):
                if skip_measure:
                    continue
                if state.can_execute_reset_measure(node.operation):
                    return node
        return None
    
    def _find_gate_locality_aware(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> Optional[DAGNode]:
        """
        ゲート選択戦略: locality_aware (optimized)
        
        LIZにいるイオンを使うゲートを優先的に選択。
        移動操作を減らすことを目的とする。
        
        TODO: 実装予定
        """
        # 暫定: min_idと同じ動作（後で実装）
        return self._find_gate_min_id(dag, front_layer, state)
    
    def _try_batch_reset_measure(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> Tuple[List[Operation], int]:
        """
        RESET/MEASUREをバッチ処理（複数ターゲットを1操作にまとめる）
        
        戦略:
        1. Front Layerから全RESET（またはMEASURE）を集める
        2. config.batch_sizeまたはLIZ容量でバッチサイズを決定
        3. バッチサイズに達していなければ待機（ただし他のゲートがない場合は処理）
        4. LIZ内の非ターゲットイオンを退避
        5. 全ターゲットをLIZに搬入
        6. 1つのRESET/MEASURE操作（複数targets）を発行
        
        Returns:
            (操作列, 実行したノード数)
        """
        operations: List[Operation] = []
        
        # Front LayerからRESETとMEASUREを集める
        reset_nodes = []
        measure_nodes = []
        gate_nodes = []
        
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if isinstance(node.operation, ResetOp):
                reset_nodes.append(node)
            elif isinstance(node.operation, MeasureOp):
                measure_nodes.append(node)
            elif isinstance(node.operation, GateOp):
                gate_nodes.append(node)
        
        # 他のゲート（RESET/MEASURE以外）があるか
        has_other_gate = len(gate_nodes) > 0
        
        # LIZ情報を取得
        liz = self.architecture.get_liz_zones()[0]
        liz_id = liz.id
        liz_capacity = liz.capacity
        
        # 有効なバッチサイズを計算
        effective_batch_size = self.config.batch_size if self.config.batch_size > 0 else liz_capacity
        
        # RESETのバッチ処理条件:
        # - batch_size以上ある、または
        # - 2つ以上あり、他のゲートがない（待機しても増えない）
        should_batch_reset = (
            len(reset_nodes) >= effective_batch_size or 
            (len(reset_nodes) >= 2 and not has_other_gate and len(measure_nodes) == 0)
        )
        
        if should_batch_reset:
            # 実際のバッチサイズ: effective_batch_sizeとLIZ容量の小さい方が上限
            max_batch = min(liz_capacity, effective_batch_size)
            # 重複するターゲットを除外してバッチを構築
            batch_targets = []
            batch_nodes = []
            for node in reset_nodes:
                target = node.operation.targets[0]
                if target not in batch_targets:
                    batch_targets.append(target)
                    batch_nodes.append(node)
                    if len(batch_targets) >= max_batch:
                        break
            
            # Step 1: LIZ内の非ターゲットイオンを全て退避
            liz_contents = state.zone_contents.get(liz_id, [])[:]
            for ion in liz_contents:
                if ion not in batch_targets:
                    evict_zone = 5 if ion >= 9 else 1
                    to_right = evict_zone > liz_id
                    self._swap_to_edge(ion, state, operations, to_right=to_right)
                    loc = self._make_shuttle_location(state, ion, evict_zone)
                    evict_op = ShuttleOp(target=ion, dst=loc, dst_chain_length=len(state.zone_contents.get(evict_zone, [])))
                    operations.append(evict_op)
                    state.transfer_ion(ion, evict_zone)
            
            # Step 2: 全ターゲットをLIZに搬入
            for target_id in batch_targets:
                if not state.is_in_liz(target_id):
                    src_zone = state.locate(target_id)
                    to_right = liz_id > src_zone
                    self._swap_to_edge(target_id, state, operations, to_right=to_right)
                    loc = self._make_shuttle_location(state, target_id, liz_id)
                    shuttle_op = ShuttleOp(target=target_id, dst=loc, dst_chain_length=len(state.zone_contents.get(liz_id, [])))
                    operations.append(shuttle_op)
                    state.transfer_ion(target_id, liz_id)
            
            # Step 3: 1つのRESET操作（複数targets）を発行
            batch_reset = ResetOp(targets=batch_targets)
            operations.append(batch_reset)
            
            # Step 4: DAGノードを実行済みにマーク
            for node in batch_nodes:
                dag.execute(node.id)
            
            return (operations, len(batch_nodes))
        
        # MEASUREのバッチ処理条件:
        # - batch_size以上ある、または
        # - 2つ以上あり、他のゲートがない（待機しても増えない）
        should_batch_measure = (
            len(measure_nodes) >= effective_batch_size or
            (len(measure_nodes) >= 2 and not has_other_gate and len(reset_nodes) == 0)
        )
        
        if should_batch_measure:
            # 実際のバッチサイズ: effective_batch_sizeとLIZ容量の小さい方が上限
            max_batch = min(liz_capacity, effective_batch_size)
            # 重複するターゲットを除外してバッチを構築
            batch_targets = []
            batch_nodes = []
            for node in measure_nodes:
                target = node.operation.targets[0]
                if target not in batch_targets:
                    batch_targets.append(target)
                    batch_nodes.append(node)
                    if len(batch_targets) >= max_batch:
                        break
            
            # Step 1: LIZ内の非ターゲットイオンを全て退避
            liz_contents = state.zone_contents.get(liz_id, [])[:]
            for ion in liz_contents:
                if ion not in batch_targets:
                    evict_zone = 5 if ion >= 9 else 1
                    to_right = evict_zone > liz_id
                    self._swap_to_edge(ion, state, operations, to_right=to_right)
                    loc = self._make_shuttle_location(state, ion, evict_zone)
                    evict_op = ShuttleOp(target=ion, dst=loc, dst_chain_length=len(state.zone_contents.get(evict_zone, [])))
                    operations.append(evict_op)
                    state.transfer_ion(ion, evict_zone)
            
            # Step 2: 全ターゲットをLIZに搬入
            for target_id in batch_targets:
                if not state.is_in_liz(target_id):
                    src_zone = state.locate(target_id)
                    to_right = liz_id > src_zone
                    self._swap_to_edge(target_id, state, operations, to_right=to_right)
                    loc = self._make_shuttle_location(state, target_id, liz_id)
                    shuttle_op = ShuttleOp(target=target_id, dst=loc, dst_chain_length=len(state.zone_contents.get(liz_id, [])))
                    operations.append(shuttle_op)
                    state.transfer_ion(target_id, liz_id)
            
            # Step 3: 1つのMEASURE操作（複数targets）を発行
            batch_measure = MeasureOp(targets=batch_targets)
            operations.append(batch_measure)
            
            # Step 4: DAGノードを実行済みにマーク
            for node in batch_nodes:
                dag.execute(node.id)
            
            return (operations, len(batch_nodes))
        
        # バッチ処理の条件を満たさない
        return ([], 0)
    
    def _swap_to_edge(self, ion_id: int, state: QCCDState, operations: List[Operation], to_right: bool = True):
        """
        イオンをzone内で端に移動するためのSwap操作を生成
        
        Args:
            ion_id: 移動するイオンのID
            state: 現在の状態
            operations: 操作を追加するリスト
            to_right: Trueなら右端、Falseなら左端に移動
        """
        zone_id = state.locate(ion_id)
        contents = state.zone_contents[zone_id]
        ion_pos = contents.index(ion_id)
        
        if to_right:
            # 右端に移動
            while ion_pos < len(contents) - 1:
                swap_with = contents[ion_pos + 1]
                swap_op = SwapOp(targets=[ion_id, swap_with])
                operations.append(swap_op)
                contents[ion_pos], contents[ion_pos + 1] = contents[ion_pos + 1], contents[ion_pos]
                ion_pos += 1
        else:
            # 左端に移動
            while ion_pos > 0:
                swap_with = contents[ion_pos - 1]
                swap_op = SwapOp(targets=[ion_id, swap_with])
                operations.append(swap_op)
                contents[ion_pos], contents[ion_pos - 1] = contents[ion_pos - 1], contents[ion_pos]
                ion_pos -= 1
    
    def _generate_shuttle_ops_for_target(self, node: DAGNode, state: QCCDState) -> List[Tuple[Operation, str]]:
        """
        特定のノード（RESET/MEASURE）のターゲットをLIZに持っていくための移動操作を生成
        """
        target_id = node.operation.targets[0]
        movement_ops: List[Tuple[Operation, str]] = []
        
        # LIZを取得
        liz = self.architecture.get_liz_zones()[0]
        liz_id = liz.id
        
        # ターゲットがLIZにいない場合、移動させる
        if not state.is_in_liz(target_id):
            # LIZが満杯なら退避
            liz_contents = state.zone_contents.get(liz_id, [])
            while len(liz_contents) >= liz.capacity:
                # LIZから1イオン退避
                evict_ion = liz_contents[0]
                nearest_memory = self.architecture.find_nearest_memory(liz_id)
                if nearest_memory is not None:
                    loc = self._make_shuttle_location(state, evict_ion, nearest_memory)
                    evict_op = ShuttleOp(target=evict_ion, dst=loc, dst_chain_length=len(state.zone_contents.get(nearest_memory, [])))
                    movement_ops.append((evict_op, "out"))
                    state.transfer_ion(evict_ion, nearest_memory)
                liz_contents = state.zone_contents.get(liz_id, [])
            
            # ターゲットをLIZに移動
            src_zone = state.locate(target_id)
            loc = self._make_shuttle_location(state, target_id, liz_id)
            shuttle_op = ShuttleOp(target=target_id, dst=loc, dst_chain_length=len(state.zone_contents.get(liz_id, [])))
            movement_ops.append((shuttle_op, "in"))
            state.transfer_ion(target_id, liz_id)
        
        # LIZにターゲット以外がいる場合、退避
        liz_contents = state.zone_contents.get(liz_id, [])
        for ion in liz_contents[:]:
            if ion != target_id:
                nearest_memory = self.architecture.find_nearest_memory(liz_id)
                if nearest_memory is not None:
                    loc = self._make_shuttle_location(state, ion, nearest_memory)
                    evict_op = ShuttleOp(target=ion, dst=loc, dst_chain_length=len(state.zone_contents.get(nearest_memory, [])))
                    movement_ops.append((evict_op, "out"))
                    state.transfer_ion(ion, nearest_memory)
        
        return movement_ops
    
    def _generate_shuttle_ops(self, dag: DAG, front_layer: Set[int], state: QCCDState) -> List[Tuple[Operation, str]]:
        """
        必要な移動操作を生成
        
        Returns:
            List of (Operation, op_type) where op_type is "in", "out", or "swap"
        
        プロトタイプ版:
        - Front Layerの最小IDノードに必要なイオンをLIZへ移動
        - LIZが満杯なら、Front Layerで不要なイオンを退避
        - SHUTTLE前に必要ならSWAPを挿入
        - RESET/MEASUREの場合、LIZ上のターゲット以外を退避
        
        バッチ処理が有効な場合:
        - RESET/MEASUREがbatch_size未満で他のゲートがあればスキップ
        """
        skip_reset = False
        skip_measure = False
        
        if self.config.batch_reset_measure:
            liz = self.architecture.get_liz_zones()[0]
            effective_batch_size = self.config.batch_size if self.config.batch_size > 0 else liz.capacity
            
            reset_count = sum(1 for nid in front_layer 
                             if isinstance(dag.nodes[nid].operation, ResetOp))
            measure_count = sum(1 for nid in front_layer 
                               if isinstance(dag.nodes[nid].operation, MeasureOp))
            
            # 他のゲート（GateOpのみ）があるか確認
            has_other_gate = any(
                isinstance(dag.nodes[nid].operation, GateOp)
                for nid in front_layer
            )
            
            if 0 < reset_count < effective_batch_size and has_other_gate:
                skip_reset = True
            if 0 < measure_count < effective_batch_size and has_other_gate:
                skip_measure = True
        
        # 最小IDのゲートを対象にする（RESET/MEASUREスキップ対応）
        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            
            # RESET/MEASUREスキップ
            if skip_reset and isinstance(node.operation, ResetOp):
                continue
            if skip_measure and isinstance(node.operation, MeasureOp):
                continue
            
            if isinstance(node.operation, (GateOp, ResetOp, MeasureOp)):
                targets = node.operation.targets
                movement_ops: List[Tuple[Operation, str]] = []
                
                is_reset_measure = isinstance(node.operation, (ResetOp, MeasureOp))
                
                # LIZにいないイオンを特定
                ions_to_bring = [ion_id for ion_id in targets if not state.is_in_liz(ion_id)]
                
                # RESET/MEASUREの場合: LIZ上のターゲット以外のイオンを退避
                if is_reset_measure and not ions_to_bring:
                    # ターゲットは既にLIZにいる
                    # LIZ上にターゲット以外のイオンがいるか確認
                    target_liz_id = state.locate(targets[0])
                    liz_contents = set(state.zone_contents[target_liz_id])
                    target_set = set(targets)
                    non_targets_in_liz = liz_contents - target_set
                    
                    if non_targets_in_liz:
                        # ターゲット以外のイオンを退避
                        eviction_ops = self._generate_eviction_ops_for_reset_measure(
                            state, target_liz_id, non_targets_in_liz
                        )
                        movement_ops.extend(eviction_ops)
                        
                        if movement_ops:
                            return movement_ops
                    
                    # ターゲット以外がいない場合は何もしない
                    continue
                
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
                
                # RESET/MEASUREの場合、LIZ上の全イオンを退避（ターゲットを除く）
                if is_reset_measure:
                    liz_contents = set(state.zone_contents[target_liz_id])
                    target_set = set(targets)
                    non_targets_in_liz = liz_contents - target_set
                    
                    if non_targets_in_liz:
                        # ターゲット以外の全イオンを退避
                        eviction_ops = self._generate_eviction_ops_for_reset_measure(
                            state, target_liz_id, non_targets_in_liz
                        )
                        movement_ops.extend(eviction_ops)
                        # 空きスペースを再計算
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
                        ShuttleOp(target=ion_id, dst=self._make_shuttle_location(state, ion_id, target_liz_id)),
                        "in"
                    ))
                    # SHUTTLE操作を状態に反映
                    state.transfer_ion(ion_id, target_liz_id)
                
                if movement_ops:
                    return movement_ops
        
        return []
    
    def _generate_eviction_ops_for_reset_measure(
        self, 
        state: QCCDState, 
        liz_id: int, 
        ions_to_evict: Set[int]
    ) -> List[Tuple[Operation, str]]:
        """
        RESET/MEASURE実行前にLIZからターゲット以外のイオンを退避
        
        Args:
            state: 現在の状態
            liz_id: 退避元のLIZ
            ions_to_evict: 退避するイオンの集合
        """
        eviction_ops: List[Tuple[Operation, str]] = []
        
        # 退避先を探す
        evict_dst = state.find_eviction_destination(liz_id)
        if evict_dst is None:
            raise RuntimeError(f"No eviction destination found for LIZ {liz_id}")
        
        for ion_id in ions_to_evict:
            # SHUTTLE前に必要ならSWAPを挿入
            swap_ops = self._generate_swap_ops_for_shuttle(
                state, ion_id, liz_id, evict_dst
            )
            eviction_ops.extend(swap_ops)
            # SWAP操作を状態に反映
            for op, _ in swap_ops:
                state.swap_ions(op.targets[0], op.targets[1])
            
            # SHUTTLE操作を追加
            eviction_ops.append((
                ShuttleOp(target=ion_id, dst=self._make_shuttle_location(state, ion_id, evict_dst)),
                "out"
            ))
            # 状態を更新
            state.transfer_ion(ion_id, evict_dst)
        
        return eviction_ops
    
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
                ShuttleOp(target=ion_to_evict, dst=self._make_shuttle_location(state, ion_to_evict, evict_dst)),
                "out"
            ))
            # SHUTTLE操作を状態に反映
            state.transfer_ion(ion_to_evict, evict_dst)
        
        return eviction_ops
    
    def _make_shuttle_location(self, state: QCCDState, ion_id: int, dst_zone_id: int) -> Location:
        """
        シャトル方向に基づいて、宛先ゾーン内の到着位置(site)を計算してLocationを生成
        
        物理的な制約:
        - 右方向（src < dst）: イオンは左端に入る → site=0
        - 左方向（src > dst）: イオンは右端に入る → site=現在のゾーン内イオン数
        
        これはstate.move_ion()の挿入ロジックと一致する。
        """
        src_zone_id = state.locate(ion_id)
        if src_zone_id < dst_zone_id:
            # 右方向: 左端に入る
            site = 0
        else:
            # 左方向: 右端に入る
            site = len(state.zone_contents.get(dst_zone_id, []))
        return Location(trap=dst_zone_id, site=site)
    
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
    print("Test 1: Sufficient LIZ capacity (T-M-T-L-T-M-T, LIZ capacity=4)")
    print("=" * 60)
    
    arch1 = QCCDArchitecture.simple_mlm(liz_capacity=4)
    print(f"\nArchitecture: {arch1}")
    
    # Ion 0, 1 をLIZ(zone 3)に、Ion 2 をMemory(zone 1)に配置
    initial1 = {0: 3, 1: 3, 2: 1}
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
    print("Test 2: Limited LIZ capacity (T-M-T-L-T-M-T, LIZ capacity=2)")
    print("=" * 60)
    
    arch2 = QCCDArchitecture.simple_mlm(liz_capacity=2)
    print(f"\nArchitecture: {arch2}")
    
    # Ion 0, 1 をLIZ(zone 3)に、Ion 2 をMemory(zone 1)に配置
    # LIZ容量=2 なので、Ion 2を入れるにはIon 0か1を退避する必要
    initial2 = {0: 3, 1: 3, 2: 1}
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
    
    # 全イオンをMemory(zone 1)に配置
    # zone 1 = Memory(左), zone 3 = LIZ, zone 5 = Memory(右)
    # LIZへ移動するには右端にいる必要（dst > src）
    initial3 = {0: 1, 1: 1, 2: 1, 3: 1}
    print(f"Initial placement: {initial3}")
    print("Zone 1 contents: [Ion0, Ion1, Ion2, Ion3] (left to right)")
    print("To move to LIZ (zone 3), ions must be at right edge")
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
    # Test 4: カスタム構成
    # ========================================
    print("\n" + "=" * 60)
    print("Test 4: Custom architecture")
    print("=" * 60)
    
    arch4 = QCCDArchitecture([("T", 20), ("M", 15), ("T", 20), ("L", 2), ("T", 20), ("L", 2), ("T", 20), ("M", 15), ("T", 20)])
    print(f"\nArchitecture: {arch4}")
    for zone in arch4.zones:
        print(f"  Zone {zone.id}: {zone.zone_type}, capacity={zone.capacity}, position={zone.position}")
    print(f"Adjacent to zone 3: {arch4.get_adjacent_zones(3)}")
    print(f"Path from zone 1 to zone 5: {arch4.get_path(1, 5)}")
    print(f"Adjacent temp of zone 1 (right): {arch4.get_adjacent_temp(1, 'right')}")
    print(f"Adjacent temp of zone 3 (left): {arch4.get_adjacent_temp(3, 'left')}")
    
    # ========================================
    # Test 5: Operation Expansion
    # ========================================
    print("\n" + "=" * 60)
    print("Test 5: Operation Expansion")
    print("=" * 60)
    
    # Swap展開テスト
    print("\n--- Swap Expansion Tests ---")
    
    arch6 = QCCDArchitecture.simple_mlm(liz_capacity=10)
    expander = OperationExpander()
    
    # テスト5-1: 2イオンのみ
    print("\n[5-1] 2 ions only: Swap(A, B)")
    initial_2ion = {0: 1, 1: 1}  # Zone1(M): [0, 1]
    ops_2ion = [SwapOp(targets=[0, 1])]
    expanded_2ion = expander.expand(ops_2ion, arch6, initial_2ion)
    print(f"  Input:  {ops_2ion}")
    print(f"  Output: {expanded_2ion}")
    
    # テスト5-2: 左端ペア
    print("\n[5-2] Left edge pair: [A, B, C, D] Swap(A, B)")
    initial_left = {0: 1, 1: 1, 2: 1, 3: 1}  # Zone1(M): [0, 1, 2, 3]
    ops_left = [SwapOp(targets=[0, 1])]
    expander = OperationExpander()
    expanded_left = expander.expand(ops_left, arch6, initial_left)
    print(f"  Input:  {ops_left}")
    print(f"  Output:")
    for op in expanded_left:
        print(f"    {op}")
    
    # テスト5-3: 右端ペア
    print("\n[5-3] Right edge pair: [A, B, C, D] Swap(C, D)")
    ops_right = [SwapOp(targets=[2, 3])]
    expander = OperationExpander()
    expanded_right = expander.expand(ops_right, arch6, initial_left)
    print(f"  Input:  {ops_right}")
    print(f"  Output:")
    for op in expanded_right:
        print(f"    {op}")
    
    # テスト5-4: 中間ペア
    print("\n[5-4] Middle pair: [A, B, C, D] Swap(B, C)")
    ops_mid = [SwapOp(targets=[1, 2])]
    expander = OperationExpander()
    expanded_mid = expander.expand(ops_mid, arch6, initial_left)
    print(f"  Input:  {ops_mid}")
    print(f"  Output:")
    for op in expanded_mid:
        print(f"    {op}")
    
    # Shuttle展開テスト
    print("\n--- Shuttle Expansion Tests ---")
    
    # テスト5-5: 右方向シャトル M(1) → L(3)
    print("\n[5-5] Shuttle right: Zone1(M)[A,B,C] -> Zone3(L), move C")
    initial_shuttle = {0: 1, 1: 1, 2: 1, 3: 3, 4: 3}  # Zone1(M): [0,1,2], Zone3(L): [3,4]
    ops_shuttle_r = [ShuttleOp(target=2, dst=Location(trap=3, site=0))]
    expander = OperationExpander()
    expanded_shuttle_r = expander.expand(ops_shuttle_r, arch6, initial_shuttle)
    print(f"  Input:  {ops_shuttle_r}")
    print(f"  Output:")
    for op in expanded_shuttle_r:
        print(f"    {op}")
    
    # テスト5-6: 左方向シャトル L(3) → M(1)
    print("\n[5-6] Shuttle left: Zone3(L)[X,Y] -> Zone1(M), move X")
    ops_shuttle_l = [ShuttleOp(target=3, dst=Location(trap=1, site=0))]
    expander = OperationExpander()
    expanded_shuttle_l = expander.expand(ops_shuttle_l, arch6, initial_shuttle)
    print(f"  Input:  {ops_shuttle_l}")
    print(f"  Output:")
    for op in expanded_shuttle_l:
        print(f"    {op}")