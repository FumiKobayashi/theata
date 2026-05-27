"""Stim Recompiler: QCCD コンパイラ出力 → Stim 論理回路への変換。

Phase 1 (現状): 1-LIZ 系のみ対応。CompileResult を入力とし、ネイティブ命令列を
fold-up して Stim の基本論理ゲート (CX, H, M, R) に変換する。Transport op は
skip し、ノイズ挿入は別フェーズ。

Fold-up ロジック:
  - RESET on ancilla → Stim `R`
  - MEASURE on ancilla → Stim `M`
  - MS(ancilla, data) → CX(ancilla, data) + H ancilla の opening/closing（X-type）
  - MS(data, ancilla) → CX(data, ancilla)（Z-type、H なし）

各 stabilizer の構造は:
  RESET → [H_opening (X-type only)] → CX × n_data → [H_closing (X-type only)] → MEASURE

  R 回転は基底変換用のネイティブ命令で、compiler が並列性のために並べ替えるため、
  R の個数や配置から論理ゲートを再構成しようとはしない。MS の target 順 (= 入力
  Stim 生成側の convention) のみを参照して X/Z 基底を判定する。

公開 API:
  - `recompile_to_stim(compile_result)`: 単ラウンド, detector/observable なし
  - `recompile_multi_round(compile_results, observable_column=None)`:
      複数ラウンド連結 + cross-round DETECTOR + (オプション) 最終 data 測定 / observable

依存:
  - `CompileResult.flat_ops` を順に walk して RESET–MEASURE で stabilizer を切り出す。
  - DAG metadata には依存しない（stabilizer_parallel=False でも動く）。
"""

import stim
from typing import List, Dict, Optional, Tuple, Sequence, FrozenSet

from ast_def import (
    Operation, GateOp, ResetOp, MeasureOp,
    ShuttleOp, SwapOp, SplitOp, MergeOp, BarrierOp,
)


# ==========================================
# Stabilizer のグルーピング
# ==========================================

def _group_ops_by_stabilizer(
    compile_result,
) -> List[List[Operation]]:
    """flat_ops を RESET-MEASURE パターンで stabilizer 単位に分割する。

    DAG の stabilizer_id に依存しない（stabilizer_parallel=False で構築された DAG でも動く）。
    各 stabilizer は: RESET <ancilla> → ... → MEASURE <ancilla>（同じ ancilla）の連続列。

    transport op (Shuttle/Split/Merge/Swap) と BarrierOp は skip する。
    """
    transport_types = (ShuttleOp, SwapOp, SplitOp, MergeOp)
    stabilizers: List[List[Operation]] = []
    current: List[Operation] = []
    active_ancilla: Optional[int] = None

    for op in compile_result.flat_ops:
        if isinstance(op, transport_types) or isinstance(op, BarrierOp):
            continue

        if isinstance(op, ResetOp):
            if len(op.targets) != 1:
                raise ValueError(f"RESET must have exactly 1 target, got {op.targets}")
            if current:
                raise ValueError(
                    f"Unexpected RESET while stabilizer for ancilla {active_ancilla} "
                    f"not yet closed by MEASURE. New RESET target: {op.targets[0]}"
                )
            current = [op]
            active_ancilla = op.targets[0]
            continue

        if isinstance(op, MeasureOp):
            if active_ancilla is None:
                continue
            if len(op.targets) != 1 or op.targets[0] != active_ancilla:
                raise ValueError(
                    f"MEASURE target {op.targets} doesn't match active ancilla {active_ancilla}"
                )
            current.append(op)
            stabilizers.append(current)
            current = []
            active_ancilla = None
            continue

        if active_ancilla is not None:
            current.append(op)

    if current:
        raise ValueError(
            f"flat_ops ended with unclosed stabilizer (ancilla {active_ancilla}, "
            f"{len(current)} ops collected without MEASURE)"
        )

    return stabilizers


# ==========================================
# Stabilizer の分類: 基底, CX ペア, data qubit 集合
# ==========================================

def _classify_stabilizer(ops: List[Operation]) -> Tuple[int, str, List[Tuple[int, int]], List[int]]:
    """1 stabilizer 分の op 列から (ancilla, basis, cx_pairs, data_qubits) を返す。

    MS の target 順で X/Z 基底を判定:
      MS(ancilla, data) → X-type, CX(ancilla, data)、外側に H 配置
      MS(data, ancilla) → Z-type, CX(data, ancilla)、H 不要

    1 stabilizer 内の全 MS は同じ basis convention に従う前提。混在していれば error。
    """
    if len(ops) < 2:
        raise ValueError(f"Stabilizer ops too short: {len(ops)} ops")
    if not isinstance(ops[0], ResetOp):
        raise ValueError(f"First op of stabilizer must be RESET, got {type(ops[0]).__name__}")
    if not isinstance(ops[-1], MeasureOp):
        raise ValueError(f"Last op of stabilizer must be MEASURE, got {type(ops[-1]).__name__}")

    ancilla = ops[0].targets[0]
    body = ops[1:-1]

    ms_ops = [op for op in body if isinstance(op, GateOp) and op.type == 'MS']
    if not ms_ops:
        raise ValueError(f"Stabilizer (ancilla={ancilla}) has no MS gate in body")

    basis: Optional[str] = None
    cx_pairs: List[Tuple[int, int]] = []
    data_qubits: List[int] = []
    for ms in ms_ops:
        if len(ms.targets) != 2:
            raise ValueError(f"MS must have 2 targets, got {ms.targets}")
        q0, q1 = ms.targets
        if q0 == ancilla:
            ms_basis = 'X'
            cx_pairs.append((ancilla, q1))
            data_qubits.append(q1)
        elif q1 == ancilla:
            ms_basis = 'Z'
            cx_pairs.append((q0, ancilla))
            data_qubits.append(q0)
        else:
            raise ValueError(
                f"MS targets {ms.targets} don't include ancilla {ancilla}"
            )
        if basis is None:
            basis = ms_basis
        elif basis != ms_basis:
            raise ValueError(
                f"Mixed MS basis in stabilizer (ancilla={ancilla}): "
                f"previous={basis}, this={ms_basis} (targets={ms.targets})"
            )

    return ancilla, basis, cx_pairs, data_qubits


# ==========================================
# Recompile state: measurement カウンタ + stabilizer 履歴
# ==========================================

class _RecompileState:
    """recompile 中の状態:
    - 出力中の stim.Circuit
    - 累積 measurement 数 (rec[-k] 計算用)
    - stabilizer identity → 過去の M index 履歴
    - 全ラウンドで登場した data qubit 集合
    """
    def __init__(self):
        self.circuit = stim.Circuit()
        self.measurement_count = 0  # 累積 M 数 (0-indexed の次の値)
        self.stab_history: Dict[Tuple[str, FrozenSet[int]], List[int]] = {}
        self.data_qubits_seen: set = set()

    def append_gate(self, name: str, qubits: Sequence[int]) -> None:
        if name == 'TICK':
            self.circuit.append('TICK')
        else:
            self.circuit.append(name, list(qubits))

    def append_M(self, qubit: int) -> int:
        """M を 1 個 emit し、その 0-indexed M index を返す。"""
        self.circuit.append('M', [qubit])
        idx = self.measurement_count
        self.measurement_count += 1
        return idx

    def _rec_offset(self, m_idx: int) -> int:
        """0-indexed M index を rec[-k] の k 値に変換 (>0)。"""
        return self.measurement_count - m_idx

    def append_detector(self, m_indices: Sequence[int]) -> None:
        targets = [stim.target_rec(-self._rec_offset(m)) for m in m_indices]
        self.circuit.append('DETECTOR', targets)

    def append_observable(self, m_indices: Sequence[int], obs_idx: int = 0) -> None:
        targets = [stim.target_rec(-self._rec_offset(m)) for m in m_indices]
        self.circuit.append('OBSERVABLE_INCLUDE', targets, obs_idx)


# ==========================================
# 1 ラウンド分の emit
# ==========================================

def _emit_round(
    state: _RecompileState,
    compile_result,
    *,
    first_round: bool,
    with_detectors: bool,
) -> List[Tuple[str, FrozenSet[int], int]]:
    """1 ラウンド分の stabilizer 列を emit し、(basis, data_set, m_idx) のリストを返す。

    返り値はそのラウンドで測定した各 stabilizer の情報。最終ラウンドであれば、
    後段で Z-type stab × final data M の consistency detector を貼るのに使える。
    """
    stabilizers = _group_ops_by_stabilizer(compile_result)
    round_stabs: List[Tuple[str, FrozenSet[int], int]] = []

    for ops in stabilizers:
        try:
            ancilla, basis, cx_pairs, data_qubits = _classify_stabilizer(ops)
        except ValueError as e:
            raise ValueError(
                f"Failed to classify stabilizer: {e}\n"
                f"Stabilizer ops:\n  " +
                "\n  ".join(_describe_op(op) for op in ops)
            ) from e

        state.data_qubits_seen.update(data_qubits)

        # 論理ゲート列を emit (R, [H], CX..., [H], M)
        # TICK で各 op を区切る (Stim の auto-batching 抑制 + noise injection 用境界)
        state.append_gate('R', [ancilla])
        state.append_gate('TICK', [])
        if basis == 'X':
            state.append_gate('H', [ancilla])
            state.append_gate('TICK', [])
        for c, t in cx_pairs:
            state.append_gate('CX', [c, t])
            state.append_gate('TICK', [])
        if basis == 'X':
            state.append_gate('H', [ancilla])
            state.append_gate('TICK', [])

        # M (DETECTOR は M 直後に置く)
        m_idx = state.append_M(ancilla)
        data_set = frozenset(data_qubits)

        if with_detectors:
            key = (basis, data_set)
            history = state.stab_history.setdefault(key, [])
            if history:
                # 前ラウンドの同 stab M との XOR
                state.append_detector([m_idx, history[-1]])
            elif first_round and basis == 'Z':
                # 全 data |0⟩ 初期化前提: 初回 Z-stab 期待値 = 0
                state.append_detector([m_idx])
            # X-stab の初回はスキップ (data |0⟩ では値ランダム)
            history.append(m_idx)

        state.append_gate('TICK', [])
        round_stabs.append((basis, data_set, m_idx))

    return round_stabs


# ==========================================
# 最終 data 測定 + observable
# ==========================================

def _emit_final_data_measurements(
    state: _RecompileState,
    last_round_stabs: List[Tuple[str, FrozenSet[int], int]],
    observable_column: Sequence[int],
) -> None:
    """最終ラウンド後の data qubit Z-測定を emit する。

    - 全 data qubit を Z 測定
    - 各 Z-type 最終 stab に対し、その data qubit 集合の M XOR との consistency
      detector を貼る (transversal Z 観測としての一貫性)
    - observable_column の data M をまとめて OBSERVABLE_INCLUDE(0) として登録
      (列 Z parity = 論理 Z 観測値)
    """
    all_data = sorted(state.data_qubits_seen)
    data_m_idx: Dict[int, int] = {}
    for q in all_data:
        m_idx = state.append_M(q)
        data_m_idx[q] = m_idx
        state.append_gate('TICK', [])

    # Z-type 最終 stab consistency detector
    for basis, data_set, stab_m_idx in last_round_stabs:
        if basis != 'Z':
            continue
        refs = [stab_m_idx] + [data_m_idx[q] for q in data_set]
        state.append_detector(refs)

    # Observable
    if observable_column:
        missing = [q for q in observable_column if q not in data_m_idx]
        if missing:
            raise ValueError(
                f"observable_column references qubits not in data set: {missing}"
            )
        state.append_observable([data_m_idx[q] for q in observable_column], obs_idx=0)


# ==========================================
# 公開 API
# ==========================================

def recompile_to_stim(compile_result) -> stim.Circuit:
    """単ラウンド CompileResult → Stim 論理回路（detector/observable なし）。

    Args:
        compile_result: QCCDCompiler.compile() の戻り値。`flat_ops` を含む必要がある。

    Returns:
        stim.Circuit (Pauli error 等は未挿入, DETECTOR/OBSERVABLE_INCLUDE も挿入しない)
    """
    state = _RecompileState()
    _emit_round(state, compile_result, first_round=True, with_detectors=False)
    return state.circuit


def recompile_multi_round(
    compile_results: List,
    *,
    observable_column: Optional[Sequence[int]] = None,
) -> stim.Circuit:
    """複数ラウンドの CompileResult を連結し、DETECTOR / OBSERVABLE_INCLUDE 付き Stim 回路を返す。

    各ラウンドの M に対し、`(basis, frozenset(data_qubits))` で stabilizer identity を
    作り、前ラウンドの同 stab M との XOR を DETECTOR として貼る。Round 1 の Z-stab は
    data |0⟩ 初期化前提で `DETECTOR rec[-1]` (期待値 0)。X-stab の初回は detector 無し。

    observable_column を指定すると最終ラウンド後に全 data Z 測定 + Z-stab consistency
    detector + 指定 data qubit 列の OBSERVABLE_INCLUDE(0) を追加する。

    Args:
        compile_results: 各ラウンドの CompileResult のリスト。順序が時系列。
        observable_column: 論理 Z observable を構成する data qubit ID リスト
                            (例: d=3 で left column → [0, 3, 6])。
                            None なら最終 data 測定 / observable をスキップ。

    Returns:
        stim.Circuit (Pauli error 等は未挿入)
    """
    if not compile_results:
        raise ValueError("compile_results must contain at least one round")

    state = _RecompileState()
    last_round_stabs: List[Tuple[str, FrozenSet[int], int]] = []
    n = len(compile_results)
    for r_idx, cr in enumerate(compile_results):
        round_stabs = _emit_round(
            state, cr,
            first_round=(r_idx == 0),
            with_detectors=True,
        )
        if r_idx == n - 1:
            last_round_stabs = round_stabs

    if observable_column is not None:
        _emit_final_data_measurements(state, last_round_stabs, observable_column)

    return state.circuit


def _describe_op(op: Operation) -> str:
    """エラーメッセージ用の op の短い記述。"""
    if isinstance(op, GateOp):
        return f"{op.type}{tuple(op.targets)}"
    if isinstance(op, ResetOp):
        return f"RESET{tuple(op.targets)}"
    if isinstance(op, MeasureOp):
        return f"MEASURE{tuple(op.targets)}"
    return f"{type(op).__name__}"
