from dataclasses import dataclass
from typing import List, Set

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
    """シャトル操作: イオンを別の位置へ移動"""
    target: int         # 移動するion_id
    dst: Location       # 移動先 (srcは状態から導出)
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
    TODO: 実験系に確認後、引数の仕様を決定
    - 現状: 2つのtrap_idを指定する想定
    """
    trap_ids: List[int]  # 分割後の2つのtrap
    type: str = "SPLIT"


@dataclass
class MergeOp(Operation):
    """
    マージ操作: 2つのイオンチェーンを合体
    TODO: 実験系に確認後、引数の仕様を決定
    - 確認事項: 隣接trap限定か？合体後の順序は？
    """
    trap_ids: List[int]  # 合体する2つのtrap
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