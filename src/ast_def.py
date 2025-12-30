from dataclasses import dataclass
from typing import List, Set, Optional

# ==========================================
# 基本型
# ==========================================

@dataclass(frozen=True)
class Location:
    """イオンの物理的位置を表す"""
    trap: int
    site: int
    
    def __repr__(self): 
        return f"({self.trap}, {self.site})"


# ==========================================
# 宣言
# ==========================================

@dataclass
class QubitDecl:
    """量子ビットの初期配置宣言"""
    id: int
    location: Location


# ==========================================
# 操作の基底クラス
# ==========================================

@dataclass
class Operation:
    """全操作の基底クラス"""
    pass


# ==========================================
# ゲート操作
# ==========================================

@dataclass
class GateOp(Operation):
    """量子ゲート操作 (MS, R)"""
    type: str           # "MS" or "R"
    targets: List[int]  # ion_id のリスト
    params: List[float] # 角度パラメータ


# ==========================================
# 移動操作
# ==========================================

@dataclass
class ShuttleOp(Operation):
    """シャトル操作: イオンを別の位置へ移動
    
    dst_chain_length: Shuttle直前のdstトラップ内のイオン数
                     （Shuttle時間の計算に使用）
    """
    target: int         # 移動するion_id
    dst: Location       # 移動先 (srcは状態から導出)
    dst_chain_length: int = 0  # dstトラップのイオン数（コンパイラが設定）
    type: str = "SHUTTLE"


@dataclass
class SwapOp(Operation):
    """スワップ操作: 同一trap内の2イオンの順序を入れ替え"""
    targets: List[int]  # 入れ替える2つのion_id
    type: str = "SWAP"


@dataclass
class SplitOp(Operation):
    """
    スプリット操作: イオンチェーンを分割
    
    指定したイオンが右端になる位置でチェーンを分割する。
    
    inheritパラメータでどちらのゾーンが元のIDを継承するか指定:
    - "left": 左側が元のzone_id、右側がright_zone_id（デフォルト）
    - "right": 右側が元のzone_id、左側がright_zone_id
    
    ion_id=None の場合、全イオンを右側のゾーンへ渡す（左側は空になる）。
    
    例1: Zone0: [A, B, C, D] で Split(ion_id=B, right_zone_id=3, inherit="left")
        → Zone0: [A, B], Zone3: [C, D]  （左が元ID）
    
    例2: Zone0: [A, B, C, D] で Split(ion_id=B, right_zone_id=3, inherit="right")
        → Zone3: [A, B], Zone0: [C, D]  （右が元ID）
    """
    ion_id: Optional[int]   # このイオンが右端になる位置で分割。Noneなら全部右へ
    right_zone_id: int      # 新しく付与するzone_id
    inherit: str = "left"   # "left" or "right" - どちらが元のIDを継承するか
    type: str = "SPLIT"


@dataclass
class MergeOp(Operation):
    """
    マージ操作: 2つの隣接イオンチェーンを結合
    
    inheritパラメータでどちらのゾーンIDを継承するか指定:
    - "left": 左側のゾーンIDが継承される（デフォルト）
    - "right": 右側のゾーンIDが継承される
    
    例: Zone0: [A, B], Zone1: [C, D] で Merge(zone_left=0, zone_right=1, inherit="left")
        → Zone0: [A, B, C, D]  (Zone1は消滅)
    
    例: Zone0: [A, B], Zone1: [C, D] で Merge(zone_left=0, zone_right=1, inherit="right")
        → Zone1: [A, B, C, D]  (Zone0は消滅)
    """
    zone_left: int       # 左側のゾーン
    zone_right: int      # 右側のゾーン
    inherit: str = "left"  # "left" or "right" - どちらのIDを継承するか
    type: str = "MERGE"


# ==========================================
# その他の操作
# ==========================================

@dataclass
class MeasureOp(Operation):
    """測定操作"""
    targets: List[int]  # 測定するion_idのリスト
    type: str = "MEASURE"


@dataclass
class ResetOp(Operation):
    """リセット操作"""
    targets: List[int]  # リセットするion_idのリスト
    type: str = "RESET"


@dataclass
class BarrierOp(Operation):
    """バリア操作: 同期点"""
    targets: List[int]  # 対象のion_idのリスト
    type: str = "BARRIER"


# ==========================================
# プログラム全体
# ==========================================

@dataclass
class Program:
    """パース結果のプログラム全体"""
    declarations: List[QubitDecl]
    instructions: List[Operation]
    
    def get_all_ion_ids(self) -> Set[int]:
        """宣言された全ion_idを取得"""
        return {decl.id for decl in self.declarations}
    
    def get_initial_positions(self) -> dict:
        """ion_id -> 初期位置 のマッピングを取得"""
        return {decl.id: decl.location for decl in self.declarations}