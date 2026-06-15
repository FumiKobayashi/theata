"""
d3_rep_reuse1
Distance-3 Repetition Code (1 アンシラ再利用)

Bit-flip protection (Z-type stabilizers のみ):
  Data qubits: 0, 1, 2  (一直線に並ぶ 3 qubit)
  Ancilla:     3        (再利用)
  Stabilizers (2 個):
    Z01 = Z_0 Z_1
    Z12 = Z_1 Z_2

Surface code と比べた特徴:
  - X-type stabilizer が無いので H 開閉処理がなく、 native 命令列は最小構成
  - 重み 4 stab も無いので 1 stab あたり CX×2 (= MS×2) だけ
  - Logical Z = 任意の単 qubit Z (例: Z_0)、 Logical X = X_0 X_1 X_2

snake_reuse1 と同じ命令フォーマット (R/MS/MEASURE/RESET の DSL) で記述。
SerialConverter が Stim 回路から生成するのと同じ慣例。
"""

QUBIT_COUNT = 4
DATA_QUBITS = [0, 1, 2]
ANCILLA_QUBITS = [3]

CIRCUIT = """
# ============================================================
# Qubit Declarations (Dummy Coordinates for Parser)
# ============================================================
QUBIT 0 AT (0, 0)
QUBIT 1 AT (0, 1)
QUBIT 2 AT (0, 2)
QUBIT 3 AT (2, 0)

# ============================================================
# Instructions
# ============================================================
# Stabilizer 1/2: Z01 (Z_0 Z_1)
RESET 3
R 0 1.5708 1.5708
MS 0 3 0.25
R 0 1.5708 3.1416
R 3 1.5708 3.1416
R 0 1.5708 4.7124
R 1 1.5708 1.5708
MS 1 3 0.25
R 1 1.5708 3.1416
R 3 1.5708 3.1416
R 1 1.5708 4.7124
MEASURE 3

# Stabilizer 2/2: Z12 (Z_1 Z_2)
RESET 3
R 1 1.5708 1.5708
MS 1 3 0.25
R 1 1.5708 3.1416
R 3 1.5708 3.1416
R 1 1.5708 4.7124
R 2 1.5708 1.5708
MS 2 3 0.25
R 2 1.5708 3.1416
R 3 1.5708 3.1416
R 2 1.5708 4.7124
MEASURE 3
"""
