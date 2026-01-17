"""
XD_ZDパターンの実行順24通りの.pyファイルを生成する。

X Pattern D:
  X9:  D4→D1→D3→D0
  X11: D8→D5→D7→D4
Z Pattern D:
  Z14: D5→D4→D2→D1
  Z15: D7→D6→D4→D3

ペア:
  UL (左上): X9 + Z13
  UR (右上): Z14 + X10
  BL (左下): Z15 + X12
  BR (右下): X11 + Z16

使い方:
    python generate_optimal_order_24.py
"""

import stim
import sys
import itertools
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
circuits_dir = current_dir.parent / 'circuits'
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

from baseline.serial_converter import SerialConverter


# ==========================================
# XD_ZD パターン定義
# ==========================================

X_ORDER = {9: [4, 1, 3, 0], 11: [8, 5, 7, 4]}  # Pattern D
Z_ORDER = {14: [5, 4, 2, 1], 15: [7, 6, 4, 3]}  # Pattern D

# ペア定義: (w4_anc, w4_order, w2_anc, w2_data, w4_type)
PAIRS = {
    "BR": (11, X_ORDER[11], 16, [5, 8], 'X'),   # 右下: X11 + Z16
    "BL": (15, Z_ORDER[15], 12, [6, 7], 'Z'),   # 左下: Z15 + X12
    "UR": (14, Z_ORDER[14], 10, [1, 2], 'Z'),   # 右上: Z14 + X10
    "UL": (9, X_ORDER[9], 13, [0, 3], 'X'),     # 左上: X9 + Z13
}

PAIR_NAMES = {
    "BR": "右下",
    "BL": "左下", 
    "UR": "右上",
    "UL": "左上",
}


# ==========================================
# 回路生成
# ==========================================

def get_interleaved_sequence(w4_anc, w4_order, w2_anc, w2_data, w4_type):
    """Weight-4とWeight-2を並行実行するインターリーブ順序を生成"""
    sequence = []
    w2_set = set(w2_data)
    w2_remaining = list(w2_data)
    
    for data in w4_order:
        sequence.append((w4_type, w4_anc, data))
        if data in w2_set and data in w2_remaining:
            w2_type = 'Z' if w4_type == 'X' else 'X'
            sequence.append((w2_type, w2_anc, data))
            w2_remaining.remove(data)
    
    for data in w2_remaining:
        w2_type = 'Z' if w4_type == 'X' else 'X'
        sequence.append((w2_type, w2_anc, data))
    
    return sequence


def generate_circuit_with_order(pair_order):
    """指定された実行順で回路を生成"""
    circuit = stim.Circuit()
    
    for pair_name in pair_order:
        w4_anc, w4_order, w2_anc, w2_data, w4_type = PAIRS[pair_name]
        
        # RESET (並列)
        circuit.append("R", [w4_anc, w2_anc])
        
        # Hadamard on X ancillas
        if w4_type == 'X':
            circuit.append("H", [w4_anc])
        else:
            circuit.append("H", [w2_anc])
        
        # Interleaved CNOTs
        sequence = get_interleaved_sequence(w4_anc, w4_order, w2_anc, w2_data, w4_type)
        
        for op_type, anc, data in sequence:
            if op_type == 'X':
                circuit.append("CX", [anc, data])
            else:
                circuit.append("CX", [data, anc])
        
        # Hadamard on X ancillas
        if w4_type == 'X':
            circuit.append("H", [w4_anc])
        else:
            circuit.append("H", [w2_anc])
        
        # MEASURE (並列)
        circuit.append("M", [w4_anc, w2_anc])
    
    return circuit


def order_to_name(pair_order):
    """実行順を名前に変換"""
    return "_".join(pair_order)


def order_to_japanese(pair_order):
    """実行順を日本語に変換"""
    return "→".join(PAIR_NAMES[p] for p in pair_order)


# ==========================================
# ファイル出力
# ==========================================

def write_circuit_py_file(filename, dsl_lines, pair_order):
    """Python形式の回路定義ファイルを書き出す"""
    filepath = circuits_dir / f"{filename}.py"
    
    data_qubits = list(range(9))
    ancilla_qubits = list(range(9, 17))
    
    order_name = order_to_name(pair_order)
    order_japanese = order_to_japanese(pair_order)
    
    content = f'''"""
Optimal Circuit: {filename}

X Pattern D:
  X9:  D4→D1→D3→D0
  X11: D8→D5→D7→D4
Z Pattern D:
  Z14: D5→D4→D2→D1
  Z15: D7→D6→D4→D3

Pair execution order: {order_japanese}
  ({order_name})

Hook error free + System aware + Interleaved parallel execution
"""

# メタデータ (exp_detailed.py で必須)
QUBIT_COUNT = 17
DATA_QUBITS = {data_qubits}
ANCILLA_QUBITS = {ancilla_qubits}

CIRCUIT = """
# ============================================================
# Qubit Declarations (Dummy Coordinates for Parser)
# ============================================================
'''
    
    # QUBIT 宣言の生成
    for q in data_qubits:
        content += f"QUBIT {q} AT (0, {q})\n"
    for q in ancilla_qubits:
        content += f"QUBIT {q} AT (2, {q - 9})\n"

    content += '\n# ============================================================\n'
    content += '# Instructions\n'
    content += '# ============================================================\n'
    
    # 命令列の結合
    content += "\n".join(dsl_lines)
    content += '\n"""\n'

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[Generated] {filepath}")
    return filepath


# ==========================================
# メイン
# ==========================================

def main():
    # circuits_dir が存在しない場合は作成
    circuits_dir.mkdir(parents=True, exist_ok=True)
    
    converter = SerialConverter()
    
    # 24通りの実行順を生成
    all_orders = list(itertools.permutations(["BR", "BL", "UR", "UL"]))
    
    print("=" * 60)
    print("Generating 24 Execution Order Patterns (XD_ZD)")
    print("=" * 60)
    print()
    print("X Pattern D: X9: D4→D1→D3→D0, X11: D8→D5→D7→D4")
    print("Z Pattern D: Z14: D5→D4→D2→D1, Z15: D7→D6→D4→D3")
    print()
    print("Pairs:")
    print("  UL (左上): X9 + Z13")
    print("  UR (右上): Z14 + X10")
    print("  BL (左下): Z15 + X12")
    print("  BR (右下): X11 + Z16")
    print()
    
    generated_files = []
    
    for pair_order in all_orders:
        # Stim回路を生成
        circuit = generate_circuit_with_order(pair_order)
        
        # DSLに変換
        dsl_lines = converter.convert(circuit)
        
        # ファイル名
        order_name = order_to_name(pair_order)
        filename = f"optimal_XD_ZD_{order_name}"
        
        # 出力
        filepath = write_circuit_py_file(filename, dsl_lines, pair_order)
        generated_files.append(filename)
    
    print()
    print("=" * 60)
    print(f"Generated {len(generated_files)} circuit files in {circuits_dir}")
    print("=" * 60)
    
    # ファイル名一覧を出力
    print()
    print("Generated files:")
    for f in generated_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
