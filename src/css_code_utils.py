"""Generic CSS code construction utilities.

古典符号のparity-check行列からCSS符号 (Hx, Hz) を構成し、QCCD DSL の
circuits/<name>.py モジュールまで一気に作るための汎用関数群。

GF(2) の rank / 正確な符号間距離は手組みせず `ldpc` パッケージ
(https://github.com/quantumgizmos/ldpc, `ldpc.mod2` / `ldpc.code_util`) に
委譲する。 numpy>=1.24 が必要なため、 メインの requirements.txt とは別の
venv (`venv/`, `requirements-ldpc.txt` 参照) で実行すること。

hypergraph product (HGP) 専用ではない: stabilizer (basis, [data qubits]) の
リストを作れるものなら何でも (将来の BB code 等も) `css_stabilizers` 以降の
関数で同じパイプラインに乗せられる。

パイプライン:
  classical H1, H2 --[hgp_check_matrices]--> Hx, Hz
  Hx, Hz            --[css_stabilizers]-----> [(basis, [data qubits]), ...]
  stabilizers        --[build_css_stim_circuit]--> stim.Circuit (1 ancilla reuse)
  stim.Circuit       --[save_css_circuit_module]--> circuits/<name>.py

`notebooks/qec_simulation_walkthrough.ipynb` の Steane code 例 (STEANE_STABS
-> stim.Circuit -> SerialConverter -> module) を一般化したもの。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import stim
from ldpc.mod2 import rank as gf2_rank, nullspace, pivot_rows
from ldpc.code_util import compute_exact_code_distance

sys.path.insert(0, str(Path(__file__).resolve().parent / 'baseline'))
from serial_converter import SerialConverter


def hgp_check_matrices(H1, H2) -> tuple:
    """Tillich-Zemor hypergraph product。 古典 parity-check 行列 H1 (m1 x n1),
    H2 (m2 x n2) から量子 CSS 符号の (Hx, Hz) を作る。

    Hx = [H1 (x) I_n2 | I_m1 (x) H2^T]
    Hz = [I_n1 (x) H2 | H1^T (x) I_m2]

    qubit 順序: [sector A (n1*n2 個, (i,j)->i*n2+j) | sector B (m1*m2 個,
    (i,j)->i*m2+j)]。 混合積公式 (kron(A,B)@kron(C,D)^T = kron(AC^T,BD^T)) から
    Hx @ Hz^T = kron(H1,H2^T) + kron(H1,H2^T) = 0 (mod 2) が保証される
    (呼び出し側で `css_code_params` がこれを実際に assert する)。
    """
    H1 = np.array(H1, dtype=np.uint8) % 2
    H2 = np.array(H2, dtype=np.uint8) % 2
    m1, n1 = H1.shape
    m2, n2 = H2.shape

    I_n1 = np.eye(n1, dtype=np.uint8)
    I_n2 = np.eye(n2, dtype=np.uint8)
    I_m1 = np.eye(m1, dtype=np.uint8)
    I_m2 = np.eye(m2, dtype=np.uint8)

    Hx = np.concatenate([
        np.kron(H1, I_n2),
        np.kron(I_m1, H2.T),
    ], axis=1) % 2
    Hz = np.concatenate([
        np.kron(I_n1, H2),
        np.kron(H1.T, I_m2),
    ], axis=1) % 2
    return Hx, Hz


def css_code_params(H1, H2, d1: int = None, d2: int = None) -> dict:
    """HGP(H1, H2) の n, k, Q, distance 下界をまとめて返す。

    d1, d2 (古典符号 H1, H2 の distance) が None の場合、
    `ldpc.code_util.compute_exact_code_distance` で実際に計算する
    (古典符号は小さい前提なので brute force で十分速い)。 既知の値を渡せば
    その再計算をスキップできる。

    d_lower_bound = min(d1, d2) は Tillich-Zemor の標準的な構成的下界。
    量子符号自体の正確な最小重みは一般に計算困難なため、 ここでは計算しない
    (下界のみ報告する)。

    Hx @ Hz^T = 0 (mod 2) を実際に検証し、 満たさなければ ValueError。
    """
    H1 = np.array(H1, dtype=np.uint8) % 2
    H2 = np.array(H2, dtype=np.uint8) % 2
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    r1, r2 = gf2_rank(H1), gf2_rank(H2)
    k1, k1t = n1 - r1, m1 - r1
    k2, k2t = n2 - r2, m2 - r2

    if d1 is None:
        d1 = compute_exact_code_distance(H1)
    if d2 is None:
        d2 = compute_exact_code_distance(H2)

    n = n1 * n2 + m1 * m2
    k = k1 * k2 + k1t * k2t

    Hx, Hz = hgp_check_matrices(H1, H2)
    if ((Hx @ Hz.T) % 2).any():
        raise ValueError("CSS commutation check failed: Hx @ Hz^T != 0 (mod 2)")

    return {
        'n': n, 'k': k, 'Q': n + 1,
        'd_lower_bound': min(d1, d2),
        'Hx': Hx, 'Hz': Hz,
    }


def compute_logical_z_operators(Hx, Hz) -> np.ndarray:
    """Z-type logical operator の代表元を k 個 (k = ker(Hx)/rowspan(Hz) の次元)
    計算する。 各行が 1 つの logical Z (Hx と可換、 Hz の row space には含まれ
    ない = 自明な stabilizer ではない) を表す長さ n の 0/1 ベクトル。

    手法: [Hz; nullspace(Hx)] を縦に積み、 pivot_rows で見つかる pivot のうち
    Hz 部分を超えるものが「Hz の row space に対して独立な ker(Hx) の元」=
    logical Z の代表元になる (典型的な CSS code の logical operator 計算法)。
    最小重みである保証はないが、 HGP のような構成では実用上低重みになる
    ことが多い (本プロジェクトの例でも確認済み)。
    """
    Hz = np.array(Hz, dtype=np.uint8) % 2
    ker_hx = np.array(nullspace(Hx).todense(), dtype=np.uint8) % 2
    stacked = np.vstack([Hz, ker_hx])
    pivots = pivot_rows(stacked)
    logical_idx = [p - Hz.shape[0] for p in pivots if p >= Hz.shape[0]]
    return ker_hx[logical_idx]


def css_stabilizers(Hx, Hz) -> list:
    """Hx の各行を X-type stabilizer、 Hz の各行を Z-type stabilizer として
    (basis, [data qubit indices]) のリストに変換する (全 0 行は無視)。"""
    stabilizers = []
    for row in np.atleast_2d(Hx):
        support = np.flatnonzero(row).tolist()
        if support:
            stabilizers.append(('X', support))
    for row in np.atleast_2d(Hz):
        support = np.flatnonzero(row).tolist()
        if support:
            stabilizers.append(('Z', support))
    return stabilizers


def build_css_stim_circuit(stabilizers: list, ancilla_id: int) -> stim.Circuit:
    """stabilizer リストから 1 ancilla reuse の stab-aware Stim 回路を作る
    (qec_simulation_walkthrough.ipynb の Steane code 例 Step A-2 を一般化)。

    X-type: R ancilla -> H -> CX(ancilla, data)x.. -> H -> M ancilla
    Z-type: R ancilla -> CX(data, ancilla)x..       -> M ancilla
    """
    circuit = stim.Circuit()
    for basis, data_qubits in stabilizers:
        circuit.append("R", [ancilla_id])
        if basis == 'X':
            circuit.append("H", [ancilla_id])
            for d in data_qubits:
                circuit.append("CX", [ancilla_id, d])
            circuit.append("H", [ancilla_id])
        else:
            for d in data_qubits:
                circuit.append("CX", [d, ancilla_id])
        circuit.append("M", [ancilla_id])
        circuit.append("TICK")
    return circuit


def save_css_circuit_module(stim_circuit: stim.Circuit, data_qubits: list,
                            ancilla_qubits: list, out_path, module_docstring: str,
                            logical_z_operators=None, d_lower_bound: int = None,
                            d_is_exact: bool = False) -> None:
    """`stim_circuit` を SerialConverter で QCCD DSL に変換し、
    circuits/d3_rep_reuse1.py と同じ形式の Python module として保存する
    (qec_simulation_walkthrough.ipynb Step A-3 を一般化)。

    `logical_z_operators` (compute_logical_z_operators の出力、 k x n 0/1
    配列) を渡すと、 各行の support (data qubit indices) を
    `LOGICAL_Z_OPERATORS` として module に埋め込む。 surface code の
    OBSERVABLE_QUBITS (notebook 側にハードコード) と異なり、 HGP のような
    符号では logical operator が自明に分からないため、 計算結果を回路と一緒
    に保存しておく。
    """
    dsl_lines = SerialConverter().convert(stim_circuit)

    qubit_decls = [f"QUBIT {q} AT (0, {q})" for q in data_qubits]
    qubit_decls += [f"QUBIT {q} AT (2, {i})" for i, q in enumerate(ancilla_qubits)]

    circuit_text = (
        "\n# ============================================================\n"
        "# Qubit Declarations (Dummy Coordinates for Parser)\n"
        "# ============================================================\n"
        + "\n".join(qubit_decls)
        + "\n\n# ============================================================\n"
        "# Instructions (auto-generated)\n"
        "# ============================================================\n"
        + "\n".join(dsl_lines)
        + "\n"
    )

    extra_lines = ""
    if logical_z_operators is not None:
        supports = [np.flatnonzero(row).tolist() for row in logical_z_operators]
        extra_lines += f"LOGICAL_Z_OPERATORS = {supports}\n"
    if d_lower_bound is not None:
        extra_lines += f"D_LOWER_BOUND = {d_lower_bound}\n"
        extra_lines += f"D_IS_EXACT = {d_is_exact}\n"
    if extra_lines:
        extra_lines += "\n"

    module_src = (
        f'"""\n{module_docstring}\n"""\n\n'
        f'QUBIT_COUNT = {len(data_qubits) + len(ancilla_qubits)}\n'
        f'DATA_QUBITS = {list(data_qubits)}\n'
        f'ANCILLA_QUBITS = {list(ancilla_qubits)}\n\n'
        f'{extra_lines}'
        f'CIRCUIT = """{circuit_text}"""\n'
    )

    Path(out_path).write_text(module_src, encoding='utf-8')
