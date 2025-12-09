# src/qccd_native.py
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto

# 操作タイプを定義して、後のシミュレータでの分岐を楽にします
class OpType(Enum):
    # Calculation (LIZ)
    R = auto()          # Single Qubit Rotation
    MS = auto()         # Two Qubit Entanglement (Molmer-Sorenson)
    MEASURE = auto()    # Measurement
    RESET = auto()      # Reset
    
    # Movement (Architecture)
    MOVE = auto()
    SPLIT = auto()
    MERGE = auto()
    SWAP = auto()
    
    # Barriers
    BARRIER = auto()    # Logic/Measurement Barrier

@dataclass
class NativeInstruction:
    """
    QCCDハードウェアが直接実行可能な物理命令。
    Source 1, 2 で定義されたネイティブゲートセットに対応。
    """
    type: OpType
    targets: List[int]          # 物理イオンID (Physical Ion ID)
    params: List[float] = field(default_factory=list) # 回転角など [theta, phi]
    duration: float = 0.0       # 実行にかかる時間 (us)

    def __repr__(self):
        # デバッグ表示用
        tgs = ",".join(map(str, self.targets))
        prms = ""
        if self.params:
            prms = f", p={self.params}"
        return f"[{self.type.name}] targets=[{tgs}]{prms} ({self.duration}us)"