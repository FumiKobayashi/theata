"""QCCD操作の時間パラメータ定義

2020年のQCCD論文に基づくパラメータ値をデフォルトとして使用。
"""

from dataclasses import dataclass
from enum import Enum


class QCCDOperationType(Enum):
    """QCCD操作の種類"""
    SPLIT = "split"
    MERGE = "merge"
    SWAP = "swap"
    SHUTTLE = "shuttle"


@dataclass
class QCCDParameters:
    """QCCD操作の時間パラメータ
    
    Attributes:
        split_us: Split操作の時間 [μs]
        merge_us: Merge操作の時間 [μs]
        swap_us: Swap操作の時間 [μs]
        shuttle_speed_m_s: Shuttle速度 [m/s]
    """
    split_us: float = 100.0
    merge_us: float = 100.0
    swap_us: float = 20.0
    shuttle_speed_m_s: float = 9.0
    
    def shuttle_time_us(self, distance_um: float) -> float:
        """シャトル距離から時間を計算
        
        Args:
            distance_um: シャトル距離 [μm]
            
        Returns:
            シャトル時間 [μs]
        """
        # distance [μm] → [m]
        distance_m = distance_um * 1e-6
        # time [s] = distance [m] / speed [m/s]
        time_s = distance_m / self.shuttle_speed_m_s
        # time [s] → [μs]
        return time_s * 1e6