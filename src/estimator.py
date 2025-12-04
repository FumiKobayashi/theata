"""QCCD回路の時間見積もりツール

Stim回路とQCCD操作を同時に管理し、コードサイクル時間を計算する。
"""

import stim
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

from .parameters import QCCDParameters, QCCDOperationType


@dataclass
class Operation:
    """回路内の1操作を表す"""
    name: str
    targets: List[int]
    duration_us: float
    is_qccd: bool  # True: QCCD操作, False: 量子ゲート


class QCCDCircuitBuilder:
    """QCCD回路を構築し、時間見積もりを行うクラス
    
    使用例:
        builder = QCCDCircuitBuilder()
        builder.shuttle(0, distance_um=2000)
        builder.shuttle(1, distance_um=2000)
        builder.cz(0, 1)
        builder.tick()
        builder.measure(0)
        
        print(builder.total_time_us())
        stim_circuit = builder.to_stim()
    """
    
    def __init__(self, params: Optional[QCCDParameters] = None):
        """
        Args:
            params: QCCD時間パラメータ。Noneならデフォルト値を使用。
        """
        self.params = params or QCCDParameters()
        self.operations: List[Operation] = []
        self.qubit_time_us: Dict[int, float] = {}  # 各qubitの経過時間
        self._current_tick_ops: List[Operation] = []  # 現在のTICK内の操作
    
    # =========================================
    # QCCD操作
    # =========================================
    
    def split(self, q1: int, q2: int) -> "QCCDCircuitBuilder":
        """Split操作: イオンチェーンを分割"""
        op = Operation(
            name="SPLIT",
            targets=[q1, q2],
            duration_us=self.params.split_us,
            is_qccd=True
        )
        self._add_operation(op)
        return self
    
    def merge(self, q1: int, q2: int) -> "QCCDCircuitBuilder":
        """Merge操作: イオンチェーンを結合"""
        op = Operation(
            name="MERGE",
            targets=[q1, q2],
            duration_us=self.params.merge_us,
            is_qccd=True
        )
        self._add_operation(op)
        return self
    
    def swap(self, q1: int, q2: int) -> "QCCDCircuitBuilder":
        """Swap操作: トラップ内のイオン位置を入れ替え"""
        op = Operation(
            name="SWAP",
            targets=[q1, q2],
            duration_us=self.params.swap_us,
            is_qccd=True
        )
        self._add_operation(op)
        return self
    
    def shuttle(self, qubit: int, distance_um: float) -> "QCCDCircuitBuilder":
        """Shuttle操作: イオンをゾーン間で移動"""
        op = Operation(
            name="SHUTTLE",
            targets=[qubit],
            duration_us=self.params.shuttle_time_us(distance_um),
            is_qccd=True
        )
        self._add_operation(op)
        return self
    
    # =========================================
    # 量子ゲート
    # =========================================
    
    def cz(self, q1: int, q2: int, duration_us: float = 100.0) -> "QCCDCircuitBuilder":
        """CZゲート（2量子ビット）"""
        op = Operation(
            name="CZ",
            targets=[q1, q2],
            duration_us=duration_us,
            is_qccd=False
        )
        self._add_operation(op)
        return self
    
    def h(self, qubit: int, duration_us: float = 10.0) -> "QCCDCircuitBuilder":
        """Hゲート（1量子ビット）"""
        op = Operation(
            name="H",
            targets=[qubit],
            duration_us=duration_us,
            is_qccd=False
        )
        self._add_operation(op)
        return self
    
    def measure(self, qubit: int, duration_us: float = 100.0) -> "QCCDCircuitBuilder":
        """測定"""
        op = Operation(
            name="M",
            targets=[qubit],
            duration_us=duration_us,
            is_qccd=False
        )
        self._add_operation(op)
        return self
    
    def reset(self, qubit: int, duration_us: float = 50.0) -> "QCCDCircuitBuilder":
        """リセット"""
        op = Operation(
            name="R",
            targets=[qubit],
            duration_us=duration_us,
            is_qccd=False
        )
        self._add_operation(op)
        return self
    
    # =========================================
    # タイミング制御
    # =========================================
    
    def tick(self) -> "QCCDCircuitBuilder":
        """TICKを挿入（時間ステップの区切り）"""
        op = Operation(
            name="TICK",
            targets=[],
            duration_us=0,
            is_qccd=False
        )
        self._add_operation(op)
        return self
    
    # =========================================
    # 内部メソッド
    # =========================================
    
    def _add_operation(self, op: Operation) -> None:
        """操作を追加し、qubitの経過時間を更新"""
        self.operations.append(op)
        
        # 各qubitの経過時間を更新
        for q in op.targets:
            if q not in self.qubit_time_us:
                self.qubit_time_us[q] = 0.0
            self.qubit_time_us[q] += op.duration_us
    
    # =========================================
    # 出力
    # =========================================
    
    def total_time_us(self) -> float:
        """全体の実行時間を返す（最も時間のかかるqubitの時間）"""
        if not self.qubit_time_us:
            return 0.0
        return max(self.qubit_time_us.values())
    
    def time_breakdown(self) -> Dict[str, float]:
        """操作カテゴリ別の時間内訳を返す"""
        breakdown = {
            "qccd_total": 0.0,
            "gate_total": 0.0,
            "split": 0.0,
            "merge": 0.0,
            "swap": 0.0,
            "shuttle": 0.0,
        }
        
        for op in self.operations:
            if op.is_qccd:
                breakdown["qccd_total"] += op.duration_us
                breakdown[op.name.lower()] += op.duration_us
            else:
                if op.name != "TICK":
                    breakdown["gate_total"] += op.duration_us
        
        return breakdown
    
    def to_stim(self) -> stim.Circuit:
        """Stim回路を生成（QCCD操作は除外）"""
        circuit = stim.Circuit()
        
        for op in self.operations:
            if op.is_qccd:
                # QCCD操作はStimに含めない
                continue
            
            if op.name == "TICK":
                circuit.append("TICK")
            elif op.name == "M":
                circuit.append("M", op.targets)
            elif op.name == "R":
                circuit.append("R", op.targets)
            else:
                circuit.append(op.name, op.targets)
        
        return circuit
    
    def __repr__(self) -> str:
        lines = ["QCCDCircuitBuilder:"]
        for op in self.operations:
            qccd_marker = "[QCCD] " if op.is_qccd else "       "
            lines.append(f"  {qccd_marker}{op.name} {op.targets} ({op.duration_us:.1f} μs)")
        lines.append(f"  Total: {self.total_time_us():.1f} μs")
        return "\n".join(lines)