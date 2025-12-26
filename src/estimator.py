"""
QCCD Time Estimator

展開済みの操作列から実行時間を推定する。

時間パラメータは論文の参照値を使用（調整可能）:
- Pino et al., "Demonstration of the trapped-ion quantum CCD computer architecture" (2021)
- Murali et al., "Architecting Noisy Intermediate-Scale Trapped Ion Quantum Computers" (2020)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from ast_def import (
    Operation, GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp,
    MeasureOp, ResetOp, BarrierOp
)


@dataclass
class TimeParameters:
    """
    各操作の時間パラメータ（マイクロ秒単位）
    
    デフォルト値は論文からの参照値。
    実際のハードウェアに合わせて調整可能。
    """
    # ゲート操作
    ms_gate: float = 200.0      # Mølmer-Sørensen gate
    r_gate: float = 10.0        # Single-qubit rotation
    
    # 測定・初期化
    reset: float = 100.0        # State preparation
    measure: float = 100.0      # Measurement
    
    # シャトリング操作
    split: float = 100.0        # Chain split
    merge: float = 100.0        # Chain merge
    swap: float = 20.0          # 2-ion trap内でのswap
    
    # Shuttle: 距離依存
    # 速度ベースで計算: time = distance / speed
    shuttle_speed: float = 9.0  # m/s
    zone_spacing: float = 100.0  # μm (ゾーン間の距離)
    
    def shuttle_time(self, zone_distance: int = 1) -> float:
        """
        シャトル時間を計算
        
        Args:
            zone_distance: 移動するゾーン数（隣接なら1）
        
        Returns:
            時間（マイクロ秒）
        """
        distance_m = self.zone_spacing * zone_distance * 1e-6  # μm → m
        time_s = distance_m / self.shuttle_speed
        return time_s * 1e6  # s → μs


@dataclass
class EstimationResult:
    """時間推定の結果"""
    total_time: float                    # 合計時間（μs）
    breakdown: Dict[str, float]          # 操作種別ごとの時間
    counts: Dict[str, int]               # 操作種別ごとのカウント
    
    def __repr__(self):
        lines = [
            f"Total Time: {self.total_time:.2f} μs ({self.total_time/1000:.2f} ms)",
            "",
            "Breakdown by operation type:",
        ]
        for op_type, time in sorted(self.breakdown.items(), key=lambda x: -x[1]):
            count = self.counts.get(op_type, 0)
            pct = (time / self.total_time * 100) if self.total_time > 0 else 0
            lines.append(f"  {op_type:12s}: {time:10.2f} μs ({pct:5.1f}%) × {count}")
        return "\n".join(lines)


class Estimator:
    """
    QCCD回路の実行時間を推定
    
    使用例:
        estimator = Estimator()
        result = estimator.estimate(operations)
        print(result)
    """
    
    def __init__(self, params: Optional[TimeParameters] = None):
        """
        Args:
            params: 時間パラメータ。Noneならデフォルト値を使用
        """
        self.params = params or TimeParameters()
    
    def estimate(self, operations: List[Operation]) -> EstimationResult:
        """
        操作列の実行時間を推定
        
        Args:
            operations: 展開済みの操作列
        
        Returns:
            EstimationResult: 時間推定結果
        """
        breakdown: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        
        for op in operations:
            op_type, time = self._get_operation_time(op)
            
            # 集計
            breakdown[op_type] = breakdown.get(op_type, 0.0) + time
            counts[op_type] = counts.get(op_type, 0) + 1
        
        total_time = sum(breakdown.values())
        
        return EstimationResult(
            total_time=total_time,
            breakdown=breakdown,
            counts=counts
        )
    
    def _get_operation_time(self, op: Operation) -> tuple:
        """
        操作の種類と時間を返す
        
        Returns:
            (操作種別名, 時間)
        """
        if isinstance(op, GateOp):
            if op.type == "MS":
                return ("MS", self.params.ms_gate)
            elif op.type == "R":
                return ("R", self.params.r_gate)
            else:
                # 未知のゲート: MSと同じ時間を仮定
                return (op.type, self.params.ms_gate)
        
        elif isinstance(op, ResetOp):
            # 複数ターゲットの場合も1回の操作として扱う
            return ("Reset", self.params.reset)
        
        elif isinstance(op, MeasureOp):
            return ("Measure", self.params.measure)
        
        elif isinstance(op, SplitOp):
            # ion_id=None (全イオン右へ) の場合も同じ時間
            # 実験家の確認後、変更の可能性あり
            return ("Split", self.params.split)
        
        elif isinstance(op, MergeOp):
            return ("Merge", self.params.merge)
        
        elif isinstance(op, SwapOp):
            return ("Swap", self.params.swap)
        
        elif isinstance(op, ShuttleOp):
            # 距離は常に1ゾーンと仮定（隣接ゾーン間のみ）
            # より正確にするにはアーキテクチャ情報が必要
            return ("Shuttle", self.params.shuttle_time(zone_distance=1))
        
        elif isinstance(op, BarrierOp):
            # バリアは同期点で、実行時間なし
            return ("Barrier", 0.0)
        
        else:
            # 未知の操作
            return ("Unknown", 0.0)


# ==========================================
# 動作確認
# ==========================================

if __name__ == "__main__":
    from qccd_parser import parse
    from dag_builder import DAGBuilder
    from qccd_compiler import QCCDArchitecture, QCCDCompiler
    
    print("=" * 60)
    print("QCCD Time Estimator Test")
    print("=" * 60)
    
    # テスト回路
    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 2 AT (0, 0)

    RESET 0
    RESET 1
    RESET 2
    
    R 0 1.57 0.0
    R 1 1.57 0.0
    
    MS 0 1 0.25
    MS 1 2 0.25
    """
    
    # パースとコンパイル
    program = parse(code)
    builder = DAGBuilder(program)
    dag = builder.build()
    
    # アーキテクチャとコンパイル
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 2: 0}
    
    compiler = QCCDCompiler(arch)
    result = compiler.compile(dag, initial)
    
    print("\n--- Compiled Operations ---")
    for i, op in enumerate(result.operations):
        print(f"  [{i}] {op}")
    
    # 時間推定
    print("\n--- Time Estimation (Default Parameters) ---")
    estimator = Estimator()
    estimation = estimator.estimate(result.operations)
    print(estimation)
    
    # カスタムパラメータでの推定
    print("\n--- Time Estimation (Custom Parameters) ---")
    custom_params = TimeParameters(
        ms_gate=150.0,    # より高速なMSゲート
        r_gate=5.0,
        split=80.0,
        merge=80.0,
    )
    estimator2 = Estimator(custom_params)
    estimation2 = estimator2.estimate(result.operations)
    print(estimation2)
    
    # 複数ケースの比較
    print("\n" + "=" * 60)
    print("Comparison: Different LIZ Capacities")
    print("=" * 60)
    
    for liz_cap in [2, 3, 4]:
        arch = QCCDArchitecture.simple_mlm(liz_capacity=liz_cap)
        compiler = QCCDCompiler(arch)
        result = compiler.compile(dag, initial)
        estimation = estimator.estimate(result.operations)
        
        print(f"\nLIZ Capacity = {liz_cap}:")
        print(f"  Total Time: {estimation.total_time:.2f} μs")
        print(f"  Operations: {sum(estimation.counts.values())}")
        print(f"  Shuttles: {estimation.counts.get('Shuttle', 0)}, "
              f"Splits: {estimation.counts.get('Split', 0)}, "
              f"Merges: {estimation.counts.get('Merge', 0)}")