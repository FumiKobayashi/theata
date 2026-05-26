"""
Noise Injection Experiment

コンパイラのフォノン追跡結果を元に、Stim回路にゲートごとの
ノイズを挿入する。

パイプライン:
  1. 独自フォーマット回路 → コンパイル → MSゲートごとのフォノン数
  2. フォノン数 → エラー率 (ブラックボックスモジュール)
  3. Stim回路の各CXにDEPOLARIZE2を挿入 → ノイズ付きStim回路

使い方:
    python exp_noise_injection.py
    python exp_noise_injection.py --circuit snake_reuse1 --distance 3
    python exp_noise_injection.py --distance 5
"""

import sys
import math
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))

import stim
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator
from baseline.stim_snake_reuse import (
    SnakeReuse1Generator,
    SnakeReuse2Generator,
    SnakeReuse4Generator,
)


# ==========================================
# ブラックボックス: フォノン → エラー率
# ==========================================

def get_error_rate(axial_phonon: float, transversal_phonon: float) -> float:
    """
    フォノン数からMSゲートのエラー率を計算する。

    TODO: 他の人のモジュールに置き換え

    現在のplaceholder:
        基本エラー率 0.001 にフォノン数に比例した項を加算
    """
    base_error = 0.001
    # Placeholder: axial phonon 1000あたり +0.001 のエラー増加
    phonon_error = axial_phonon * 1e-6
    return min(base_error + phonon_error, 0.5)  # 上限50%


# ==========================================
# Step 1: コンパイル + フォノン取得
# ==========================================

def compile_with_phonon(circuit_module_name: str, placement: str = "split_even",
                        liz_capacity: int = 4, memory_capacity: int = 20):
    """
    独自フォーマット回路をコンパイルし、MSゲートごとのフォノン数を取得

    Returns:
        phonon_at_gates: [{'gate': 'MS(0,9)', 'phonons': {ion: (ax, tr)}}, ...]
    """
    mod = __import__(circuit_module_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()

    arch = QCCDArchitecture.simple_mlm(
        liz_capacity=liz_capacity,
        memory_capacity=memory_capacity,
        temp_capacity=memory_capacity,
    )

    # split_even 配置
    initial = {}
    data = mod.DATA_QUBITS
    half = len(data) // 2
    for i, qid in enumerate(data):
        initial[qid] = 1 if i < half else 5
    for qid in mod.ANCILLA_QUBITS:
        initial[qid] = 5

    compiler = QCCDCompiler(arch, config=CompilerConfig())
    result = compiler.compile(dag, initial)

    est = ParallelEstimator()
    r = est.estimate(result.timesteps)

    return result, r, initial


# ==========================================
# Step 2: フォノン → エラー率リスト
# ==========================================

def phonon_to_error_rates(phonon_at_gates: list) -> list:
    """
    各MSゲートのフォノン数からエラー率を計算

    Returns:
        [(gate_name, error_rate, max_axial, max_transversal), ...]
    """
    error_rates = []
    for gate_info in phonon_at_gates:
        phonons = gate_info['phonons']
        # phonons: {ion_id: (axial, transversal)}
        # 各値はタプルまたはリスト
        max_ax = max(p[0] for p in phonons.values())
        max_tr = max(p[1] for p in phonons.values())
        err = get_error_rate(max_ax, max_tr)
        error_rates.append((gate_info['gate'], err, max_ax, max_tr))
    return error_rates


# ==========================================
# Step 3: Stim回路にノイズ挿入
# ==========================================

def generate_stim_circuit(distance: int, rounds: int = 1) -> stim.Circuit:
    """ノイズなしStim回路を生成 (Stim 標準テンプレート、レガシー)。

    注意: Stim の `surface_code:rotated_memory_z` テンプレートは
    superconducting 風の配線・並列性を前提とした構造を生成する。
    QCCD コンパイラの入力（snake_reuseN, snake_per_stabilizer 等）とは
    qubit 数・ID 体系・ancilla 戦略が一致しないため、ノイズ挿入には
    `generate_baseline_stim_circuit()` の使用を推奨。
    """
    return stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        rounds=rounds,
        distance=distance,
    )


# QCCD 回路名 → 対応する Stim 回路生成クラス
_BASELINE_STIM_GENERATORS = {
    "snake_reuse1": SnakeReuse1Generator,
    "snake_reuse2": SnakeReuse2Generator,
    "snake_reuse4": SnakeReuse4Generator,
}


def generate_baseline_stim_circuit(circuit_name: str, distance: int = 3) -> stim.Circuit:
    """QCCD コンパイラ入力と構造一致する Stim 回路を生成する。

    QCCD 入力 (snake_reuse1, snake_reuse2, snake_reuse4) に対応する baseline
    generator を使い、qubit ID・ancilla 構造が一致した Stim 回路を返す。
    各 CX は個別命令として並んでおり、TICK 分割によるイオントラップ系の
    時系列再現に適した形式。
    """
    if circuit_name not in _BASELINE_STIM_GENERATORS:
        raise ValueError(
            f"No baseline Stim generator for circuit '{circuit_name}'. "
            f"Supported: {list(_BASELINE_STIM_GENERATORS.keys())}"
        )
    gen_cls = _BASELINE_STIM_GENERATORS[circuit_name]
    gen = gen_cls(distance=distance)
    return gen.generate_surface17()


# ==========================================
# Approach B: 5-step CX decomposition + idle dephasing
# ==========================================

def default_noise_params() -> dict:
    """Default noise parameters for Approach B.

    All values are placeholders for the dummy implementation; experimental
    values should override these when available. Durations come from §4.6
    of the paper draft (Table 1 timing parameters).
    """
    return {
        # Op durations (μs)
        't_R_us': 4.0,
        't_MS_us': 59.0,
        't_meas_us': 1000.0,
        't_reset_us': 41.0,
        # Gate error probabilities
        # 値は 2026/05 時点で先生から提示:
        #   - p_R = 1e-4 (1量子ビットゲート、確率値)
        #   - p_meas = 4e-3 (0.4%, 測定エラー)
        # p_reset, p_MS_default は未定（実験値要確認、placeholder）
        'p_R': 1e-4,
        'p_MS_default': 1e-3,   # fallback when phonon-based not available
        'p_meas': 4e-3,
        'p_reset': 1e-3,        # placeholder — 実験値要確認
        # Idle dephasing rate: spin dephasing Γ_φ from Hirai et al. Table 1
        'gamma_phi_per_s': 3.0,
    }


def compute_p_idle(duration_us: float, gamma_phi_per_s: float) -> float:
    """Z dephasing probability for `duration_us` of idle time.

    p_Z = (1 - exp(-2 Γ_φ t)) / 2, with t in seconds.
    Pure dephasing channel (Kraus: K_0 = √((1+e^{-2Γt})/2) I,
    K_1 = √((1-e^{-2Γt})/2) Z) → Z_ERROR(p_Z) in Stim.
    """
    t_sec = duration_us * 1e-6
    return (1.0 - math.exp(-2 * gamma_phi_per_s * t_sec)) / 2.0


def _append_idle_zerr(noisy: stim.Circuit, all_qubits: list,
                      busy_qubits: set, p_idle: float) -> None:
    """Append Z_ERROR on all qubits not in `busy_qubits`.

    `all_qubits` must be the set of actually-used qubits (from QUBIT_COORDS),
    not `range(num_qubits)`: Stim's num_qubits returns max_id + 1, which
    creates phantom qubits when there are gaps in qubit ID numbering.
    """
    if p_idle <= 0.0:
        return
    idle = [q for q in all_qubits if q not in busy_qubits]
    if idle:
        noisy.append("Z_ERROR", idle, p_idle)


def _get_declared_qubits(circuit: stim.Circuit) -> list:
    """Return sorted list of qubit IDs actually used in the circuit.

    Prefers `QUBIT_COORDS`-declared qubits if present; otherwise scans
    operation targets. Baseline-generated circuits don't include
    QUBIT_COORDS so the fallback is required.
    """
    coords = circuit.get_final_qubit_coordinates()
    if coords:
        return sorted(coords.keys())

    used = set()
    annotation_names = {'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE',
                        'QUBIT_COORDS', 'SHIFT_COORDS'}
    for instruction in circuit:
        if instruction.name in annotation_names:
            continue
        for t in instruction.targets_copy():
            if t.is_qubit_target:
                used.add(t.value)
    return sorted(used)


def _append_5step_cx(noisy: stim.Circuit, q1: int, q2: int, all_qubits: list,
                     params: dict, p_MS: float,
                     p_idle_R: float, p_idle_MS: float) -> None:
    """Insert 5-step error pattern after a logical CX(q1, q2).

    Physical decomposition of trapped-ion CNOT: R(q1) - MS(q1,q2) - R(q1) - R(q2) - R(q1).
    Pattern follows the user's spec (5/4 メモリ): 3 R on q1, 1 MS on (q1,q2), 1 R on q2.
    """
    p_R = params['p_R']

    # Step 1: R(q1), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q1], p_R)
    _append_idle_zerr(noisy, all_qubits, {q1}, p_idle_R)
    noisy.append("TICK")

    # Step 2: MS(q1, q2), t_MS
    if p_MS > 0:
        noisy.append("DEPOLARIZE2", [q1, q2], p_MS)
    _append_idle_zerr(noisy, all_qubits, {q1, q2}, p_idle_MS)
    noisy.append("TICK")

    # Step 3: R(q1), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q1], p_R)
    _append_idle_zerr(noisy, all_qubits, {q1}, p_idle_R)
    noisy.append("TICK")

    # Step 4: R(q2), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q2], p_R)
    _append_idle_zerr(noisy, all_qubits, {q2}, p_idle_R)
    noisy.append("TICK")

    # Step 5: R(q1), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q1], p_R)
    _append_idle_zerr(noisy, all_qubits, {q1}, p_idle_R)
    noisy.append("TICK")


def _append_h_2steps(noisy: stim.Circuit, h_targets: list, all_qubits: list,
                     params: dict, p_idle_R: float) -> None:
    """Insert 2-step error pattern (= R-R) after a logical H gate.

    H ≈ 2 single-qubit rotations physically, so emit the R gate error
    pattern twice for each H target.
    """
    p_R = params['p_R']
    busy = set(h_targets)
    for _ in range(2):
        if p_R > 0:
            noisy.append("DEPOLARIZE1", list(h_targets), p_R)
        _append_idle_zerr(noisy, all_qubits, busy, p_idle_R)
        noisy.append("TICK")


def inject_noise_approach_b(
    circuit: stim.Circuit,
    error_rates_ms: list,
    params: dict = None,
) -> stim.Circuit:
    """Inject noise per Approach B (5-step CX, 2-step H, target-only MEASURE/RESET).

    Spec summary (decided 5/5):
      - CX → CX + 5-step error block (R-MS-R-R-R)
      - H  → H + 2-step error block (R-R)
      - M/MR/MX → X_ERROR(p_meas) on target before measurement
      - R/RX/RZ → X_ERROR(p_reset) on target after reset
      - Idle qubits (= non-target) get Z_ERROR(p_idle) only during R / MS steps;
        MEASURE / RESET の他 qubit idle は無視（前回合意、不確実性のため）
      - DETECTOR / OBSERVABLE_INCLUDE / TICK は pass-through
    """
    if params is None:
        params = default_noise_params()

    noisy = stim.Circuit()
    all_qubits = _get_declared_qubits(circuit)
    ms_idx = 0  # index into error_rates_ms (one per MS gate / CX pair)

    p_idle_R = compute_p_idle(params['t_R_us'], params['gamma_phi_per_s'])
    p_idle_MS = compute_p_idle(params['t_MS_us'], params['gamma_phi_per_s'])

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()

        if name == 'CX':
            # Split into individual pairs (each gets its own phonon-based p_MS)
            for i in range(0, len(targets) - 1, 2):
                q1 = targets[i].value
                q2 = targets[i + 1].value

                # Per-pair MS error rate
                if ms_idx < len(error_rates_ms):
                    p_MS = error_rates_ms[ms_idx][1]
                    ms_idx += 1
                else:
                    p_MS = params['p_MS_default']

                noisy.append("CX", [q1, q2])
                _append_5step_cx(noisy, q1, q2, all_qubits, params,
                                 p_MS, p_idle_R, p_idle_MS)

        elif name == 'H':
            tq = [t.value for t in targets]
            noisy.append("H", tq)
            _append_h_2steps(noisy, tq, all_qubits, params, p_idle_R)

        elif name in ('M', 'MX', 'MY', 'MZ'):
            tq = [t.value for t in targets]
            if params['p_meas'] > 0:
                noisy.append("X_ERROR", tq, params['p_meas'])
            noisy.append(name, tq)
            noisy.append("TICK")

        elif name in ('R', 'RX', 'RY', 'RZ'):
            tq = [t.value for t in targets]
            noisy.append(name, tq)
            if params['p_reset'] > 0:
                noisy.append("X_ERROR", tq, params['p_reset'])
            noisy.append("TICK")

        elif name == 'MR':
            # Combined measure+reset: error before measure, error after reset
            tq = [t.value for t in targets]
            if params['p_meas'] > 0:
                noisy.append("X_ERROR", tq, params['p_meas'])
            noisy.append("MR", tq)
            if params['p_reset'] > 0:
                noisy.append("X_ERROR", tq, params['p_reset'])
            noisy.append("TICK")

        else:
            # TICK, DETECTOR, OBSERVABLE_INCLUDE, etc. — pass through
            noisy.append(instruction)

    return noisy


# ==========================================
# Approach B (annotated): emit as Stim text with comments
# ==========================================

def inject_noise_approach_b_annotated(
    circuit: stim.Circuit,
    error_rates_ms: list,
    params: dict = None,
) -> str:
    """Same logic as inject_noise_approach_b but emits Stim text with comments.

    Comments annotate each TICK / step with its physical meaning, e.g.,
    "# Step 2: MS(q0, q9), 59μs". The resulting text is still valid Stim
    syntax (Stim parser ignores `#` comment lines) so downstream tools
    can still read the file as a normal Stim circuit.
    """
    if params is None:
        params = default_noise_params()

    all_qubits = _get_declared_qubits(circuit)
    ms_idx = 0
    p_idle_R = compute_p_idle(params['t_R_us'], params['gamma_phi_per_s'])
    p_idle_MS = compute_p_idle(params['t_MS_us'], params['gamma_phi_per_s'])

    lines = []

    def fmt_targets(qb_iter):
        return " ".join(str(q) for q in qb_iter)

    def emit_idle_zerr(busy, p_idle):
        if p_idle <= 0.0:
            return
        idle = [q for q in all_qubits if q not in busy]
        if idle:
            lines.append(f"Z_ERROR({p_idle}) {fmt_targets(idle)}")

    def emit_step(label, t_us):
        lines.append(f"  # {label}  (t = {t_us:g} μs)")

    def emit_passthrough(instruction):
        # Convert a stim.CircuitInstruction to text via a temp circuit
        temp = stim.Circuit()
        temp.append(instruction)
        for line in str(temp).split("\n"):
            line = line.rstrip()
            if line:
                lines.append(line)

    # File header
    lines.append("# Noise-injected Stim circuit (Approach B, annotated)")
    lines.append(f"# Noise params: p_R={params['p_R']:.1e}, "
                 f"p_MS=phonon-dependent (placeholder), "
                 f"p_meas={params['p_meas']:.1e}, "
                 f"p_reset={params['p_reset']:.1e}, "
                 f"Γ_φ={params['gamma_phi_per_s']:.1f} s⁻¹")
    lines.append(f"# Idle Z dephasing: p_idle(t_R={params['t_R_us']:g}μs)={p_idle_R:.3e}, "
                 f"p_idle(t_MS={params['t_MS_us']:g}μs)={p_idle_MS:.3e}")
    lines.append("")

    cx_count = 0

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()

        if name == 'CX':
            for i in range(0, len(targets) - 1, 2):
                q1 = targets[i].value
                q2 = targets[i + 1].value
                cx_count += 1

                if ms_idx < len(error_rates_ms):
                    ms_label = error_rates_ms[ms_idx][0]
                    p_MS = error_rates_ms[ms_idx][1]
                    ms_idx += 1
                else:
                    ms_label = "MS(default)"
                    p_MS = params['p_MS_default']

                lines.append("")
                lines.append(f"# ==== CX #{cx_count}: q{q1} ↔ q{q2}  "
                             f"(QCCD compiler: {ms_label}) ====")
                lines.append(f"CX {q1} {q2}")

                # Step 1: R(q1)
                emit_step(f"Step 1: R(q{q1})", params['t_R_us'])
                if params['p_R'] > 0:
                    lines.append(f"DEPOLARIZE1({params['p_R']}) {q1}")
                emit_idle_zerr({q1}, p_idle_R)
                lines.append("TICK")

                # Step 2: MS(q1, q2)
                emit_step(f"Step 2: MS(q{q1}, q{q2})  [phonon-dependent rate]",
                          params['t_MS_us'])
                if p_MS > 0:
                    lines.append(f"DEPOLARIZE2({p_MS}) {q1} {q2}")
                emit_idle_zerr({q1, q2}, p_idle_MS)
                lines.append("TICK")

                # Step 3: R(q1)
                emit_step(f"Step 3: R(q{q1})", params['t_R_us'])
                if params['p_R'] > 0:
                    lines.append(f"DEPOLARIZE1({params['p_R']}) {q1}")
                emit_idle_zerr({q1}, p_idle_R)
                lines.append("TICK")

                # Step 4: R(q2)
                emit_step(f"Step 4: R(q{q2})", params['t_R_us'])
                if params['p_R'] > 0:
                    lines.append(f"DEPOLARIZE1({params['p_R']}) {q2}")
                emit_idle_zerr({q2}, p_idle_R)
                lines.append("TICK")

                # Step 5: R(q1)
                emit_step(f"Step 5: R(q{q1})", params['t_R_us'])
                if params['p_R'] > 0:
                    lines.append(f"DEPOLARIZE1({params['p_R']}) {q1}")
                emit_idle_zerr({q1}, p_idle_R)
                lines.append("TICK")

        elif name == 'H':
            tq = [t.value for t in targets]
            lines.append("")
            lines.append(f"# ==== H {fmt_targets(tq)}  (2-step R decomposition) ====")
            lines.append(f"H {fmt_targets(tq)}")
            for step_i in range(2):
                emit_step(f"H-Step {step_i+1}: R-equivalent on {fmt_targets(tq)}",
                          params['t_R_us'])
                if params['p_R'] > 0:
                    lines.append(f"DEPOLARIZE1({params['p_R']}) {fmt_targets(tq)}")
                emit_idle_zerr(set(tq), p_idle_R)
                lines.append("TICK")

        elif name in ('M', 'MX', 'MY', 'MZ'):
            tq = [t.value for t in targets]
            lines.append("")
            lines.append(f"# ==== {name} {fmt_targets(tq)}  (measurement, "
                         f"idle qubits ignored) ====")
            if params['p_meas'] > 0:
                lines.append(f"  # Pre-measurement bit-flip error")
                lines.append(f"X_ERROR({params['p_meas']}) {fmt_targets(tq)}")
            lines.append(f"{name} {fmt_targets(tq)}")
            lines.append("TICK")

        elif name in ('R', 'RX', 'RY', 'RZ'):
            tq = [t.value for t in targets]
            lines.append("")
            lines.append(f"# ==== {name} {fmt_targets(tq)}  (reset, "
                         f"idle qubits ignored) ====")
            lines.append(f"{name} {fmt_targets(tq)}")
            if params['p_reset'] > 0:
                lines.append(f"  # Post-reset bit-flip error")
                lines.append(f"X_ERROR({params['p_reset']}) {fmt_targets(tq)}")
            lines.append("TICK")

        elif name == 'MR':
            tq = [t.value for t in targets]
            lines.append("")
            lines.append(f"# ==== MR {fmt_targets(tq)}  (measure + reset, "
                         f"idle qubits ignored) ====")
            if params['p_meas'] > 0:
                lines.append(f"X_ERROR({params['p_meas']}) {fmt_targets(tq)}")
            lines.append(f"MR {fmt_targets(tq)}")
            if params['p_reset'] > 0:
                lines.append(f"X_ERROR({params['p_reset']}) {fmt_targets(tq)}")
            lines.append("TICK")

        else:
            # TICK, DETECTOR, OBSERVABLE_INCLUDE, etc. — pass through
            emit_passthrough(instruction)

    return "\n".join(lines) + "\n"


# ==========================================
# Approach A (legacy): 1 DEPOLARIZE2 per CX
# ==========================================

def inject_noise(circuit: stim.Circuit, error_rates: list) -> stim.Circuit:
    """
    Stim回路の各CXゲートにDEPOLARIZE2を挿入

    CXの出現順とerror_ratesの順序が対応する前提。
    各CX命令には複数ペアが含まれうるので、ペア単位で展開して
    個別のエラー率を設定する。

    Args:
        circuit: ノイズなしStim回路
        error_rates: [(gate_name, error_rate, ax, tr), ...]

    Returns:
        ノイズ付きStim回路
    """
    noisy = stim.Circuit()
    error_idx = 0

    for instruction in circuit:
        if instruction.name == 'CX':
            # CXをペアごとに分解して個別エラー率を挿入
            targets = instruction.targets_copy()
            for i in range(0, len(targets), 2):
                if i + 1 >= len(targets):
                    break
                q1 = targets[i].value
                q2 = targets[i + 1].value

                # CXペアを追加
                noisy.append("CX", [q1, q2])

                # エラー率を取得して DEPOLARIZE2 を挿入
                if error_idx < len(error_rates):
                    _, err_rate, _, _ = error_rates[error_idx]
                    if err_rate > 0:
                        noisy.append("DEPOLARIZE2", [q1, q2], err_rate)
                    error_idx += 1
            # TICKはCXの後に元の回路で来るので、ここでは追加しない
        else:
            # CX以外はそのまま追加
            noisy.append(instruction)

    return noisy


# ==========================================
# メイン
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Noise Injection Experiment')
    parser.add_argument('--circuit', default='snake_reuse1',
                        help='Circuit module name (default: snake_reuse1)')
    parser.add_argument('--distance', type=int, default=3,
                        help='Code distance for Stim circuit (default: 3)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of syndrome measurement rounds (default: 1)')
    parser.add_argument('--memory-capacity', type=int, default=20,
                        help='Memory zone capacity (default: 20)')
    parser.add_argument('--output-dir', default='results/noise_injection',
                        help='Output directory')
    parser.add_argument('--approach', choices=['a', 'b'], default='b',
                        help='Noise injection approach: a=1-DEPOLARIZE2-per-CX (legacy), '
                             'b=5-step CX + idle dephasing (default)')
    parser.add_argument('--stim-source', choices=['baseline', 'template'], default='baseline',
                        help='Source of input Stim circuit: baseline = QCCD-matched '
                             '(default), template = Stim built-in (legacy, qubit-id mismatch)')
    parser.add_argument('--annotate', action=argparse.BooleanOptionalAction, default=True,
                        help='Annotate the .stim file with # comments labelling each TICK '
                             '(default: True, use --no-annotate to disable)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Noise Injection Experiment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Circuit: {args.circuit} (d={args.distance}, rounds={args.rounds})")
    print("=" * 70)

    # Step 1: コンパイル + フォノン
    print("\n[Step 1] Compiling with phonon tracking...")
    result, par_result, initial = compile_with_phonon(
        args.circuit,
        memory_capacity=args.memory_capacity,
    )
    pd = result.phonon_data

    print(f"  Total ops: {len(result.operations)}")
    print(f"  Parallel time: {par_result.total_time/1000:.2f} ms")
    print(f"  Max axial phonon: {pd['max_axial']:.2f}")
    print(f"  Max transversal phonon: {pd['max_transversal']:.2f}")
    print(f"  MS gates: {len(pd['at_gates'])}")

    # Step 2: フォノン → エラー率
    print("\n[Step 2] Computing error rates from phonon numbers...")
    error_rates = phonon_to_error_rates(pd['at_gates'])

    print(f"  Error rate range: {min(e[1] for e in error_rates):.6f} - {max(e[1] for e in error_rates):.6f}")
    print(f"  Mean error rate: {sum(e[1] for e in error_rates)/len(error_rates):.6f}")
    print()
    print(f"  {'Gate':15s} {'Error Rate':>12s} {'Axial':>10s} {'Trans':>10s}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
    for gate_name, err, ax, tr in error_rates:
        print(f"  {gate_name:15s} {err:12.6f} {ax:10.2f} {tr:10.4f}")

    # Step 3: Stim回路にノイズ挿入
    print(f"\n[Step 3] Generating noisy Stim circuit (d={args.distance})...")
    if args.stim_source == 'baseline':
        print(f"  Stim source: baseline ({args.circuit} 対応のカスタム生成)")
        stim_circuit = generate_baseline_stim_circuit(args.circuit, distance=args.distance)
    else:
        print(f"  Stim source: template (Stim 標準テンプレート, qubit ID 不一致あり)")
        stim_circuit = generate_stim_circuit(args.distance, args.rounds)

    # CXペア数の確認
    stim_cx_count = sum(
        len(inst.targets_copy()) // 2
        for inst in stim_circuit
        if inst.name == 'CX'
    )
    print(f"  Stim CX pairs: {stim_cx_count}")
    print(f"  Compiler MS gates: {len(error_rates)}")

    if stim_cx_count != len(error_rates):
        print(f"  WARNING: CX count mismatch! Using min({stim_cx_count}, {len(error_rates)}) error rates")

    if args.approach == 'b':
        print(f"  Using Approach B (5-step CX decomposition + idle dephasing)")
        noise_params = default_noise_params()
        print(f"  Noise params: p_R={noise_params['p_R']:.1e}, "
              f"p_meas={noise_params['p_meas']:.1e}, "
              f"p_reset={noise_params['p_reset']:.1e}, "
              f"Γ_φ={noise_params['gamma_phi_per_s']:.1f}/s")
        p_idle_R = compute_p_idle(noise_params['t_R_us'], noise_params['gamma_phi_per_s'])
        p_idle_MS = compute_p_idle(noise_params['t_MS_us'], noise_params['gamma_phi_per_s'])
        print(f"  Idle Z_ERROR rates: p_idle(t_R={noise_params['t_R_us']}μs)={p_idle_R:.3e}, "
              f"p_idle(t_MS={noise_params['t_MS_us']}μs)={p_idle_MS:.3e}")
        noisy_circuit = inject_noise_approach_b(stim_circuit, error_rates, noise_params)
    else:
        print(f"  Using Approach A (1 DEPOLARIZE2 per CX, legacy)")
        noisy_circuit = inject_noise(stim_circuit, error_rates)

    # 出力保存
    # .stim ファイル (Stimシミュレーション入力)
    stim_path = output_dir / f"noisy_d{args.distance}_{args.circuit}_{timestamp}.stim"
    if args.annotate and args.approach == 'b':
        # Annotated text version (comments preserved)
        annotated = inject_noise_approach_b_annotated(stim_circuit, error_rates, noise_params)
        with open(stim_path, 'w', encoding='utf-8') as f:
            f.write(annotated)
        print(f"\n  Saved Stim circuit (annotated): {stim_path}")
    else:
        with open(stim_path, 'w', encoding='utf-8') as f:
            f.write(str(noisy_circuit))
        print(f"\n  Saved Stim circuit: {stim_path}")

    # JSON メタデータ
    json_path = output_dir / f"noisy_d{args.distance}_{args.circuit}_{timestamp}.json"
    metadata = {
        "generated_at": timestamp,
        "circuit_module": args.circuit,
        "distance": args.distance,
        "rounds": args.rounds,
        "compiler": {
            "total_ops": len(result.operations),
            "total_timesteps": len(result.timesteps),
            "par_time_ms": par_result.total_time / 1000,
            "seq_time_ms": par_result.sequential_time / 1000,
        },
        "phonon": {
            "max_axial": pd['max_axial'],
            "max_transversal": pd['max_transversal'],
        },
        "error_rates": [
            {"gate": name, "error_rate": err, "axial": ax, "transversal": tr}
            for name, err, ax, tr in error_rates
        ],
        "stim_cx_pairs": stim_cx_count,
        "stim_file": stim_path.name,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {json_path}")

    # Stim でデコード検証 (サンプル) — DETECTOR 必須なので baseline 回路では skip
    print(f"\n[Step 4] Quick verification with Stim sampler...")
    has_detectors = any(inst.name == 'DETECTOR' for inst in noisy_circuit)
    if not has_detectors:
        print(f"  Skipped: input circuit has no DETECTOR statements "
              f"(baseline generators don't add them yet; will be added in a later step)")
    else:
        try:
            sampler = noisy_circuit.compile_detector_sampler()
            shots = sampler.sample(shots=10000)
            detection_rate = shots.any(axis=1).mean()
            print(f"  Detection rate (10k shots): {detection_rate:.4f}")
            print(f"  (fraction of shots with at least one detection event)")
        except Exception as e:
            print(f"  Stim sampling failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
