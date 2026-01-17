"""
16パターンの最適化回路を.pyファイルとして出力する。

generate_baselines.pyと同じ出力ディレクトリ（circuits/）に出力。

使い方:
    python generate_optimal_16.py
"""

import sys
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
circuits_dir = current_dir.parent / 'circuits'
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

from baseline.stim_optimal_16patterns import generate_all_patterns, X_PATTERNS, Z_PATTERNS
from baseline.serial_converter import SerialConverter


def write_circuit_py_file(filename, dsl_lines, qubit_count, data_qubits, ancilla_qubits, description=""):
    """
    Python形式の回路定義ファイルを書き出す
    """
    filepath = circuits_dir / f"{filename}.py"
    
    content = f'''"""
Optimal Circuit: {filename}
{description}
"""

# メタデータ (exp_detailed.py で必須)
QUBIT_COUNT = {qubit_count}
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


def main():
    # circuits_dir が存在しない場合は作成
    circuits_dir.mkdir(parents=True, exist_ok=True)
    
    # 16パターンを生成
    patterns = generate_all_patterns()
    converter = SerialConverter()
    
    data_qubits = list(range(9))
    ancilla_qubits = list(range(9, 17))
    
    print("=" * 60)
    print("Generating 16 Optimal Scheduling Patterns")
    print("=" * 60)
    print()
    
    for name, circuit in patterns.items():
        # パターン情報を抽出
        x_key = name[1]  # "XA_ZB" -> "A"
        z_key = name[4]  # "XA_ZB" -> "B"
        
        x_order_9 = X_PATTERNS[x_key][9]
        x_order_11 = X_PATTERNS[x_key][11]
        z_order_14 = Z_PATTERNS[z_key][14]
        z_order_15 = Z_PATTERNS[z_key][15]
        
        description = f"""
X Pattern {x_key}:
  X9:  D{x_order_9[0]}→D{x_order_9[1]}→D{x_order_9[2]}→D{x_order_9[3]}
  X11: D{x_order_11[0]}→D{x_order_11[1]}→D{x_order_11[2]}→D{x_order_11[3]}
Z Pattern {z_key}:
  Z14: D{z_order_14[0]}→D{z_order_14[1]}→D{z_order_14[2]}→D{z_order_14[3]}
  Z15: D{z_order_15[0]}→D{z_order_15[1]}→D{z_order_15[2]}→D{z_order_15[3]}

Pair execution order: 右下→左下→右上→左上
Hook error free + System aware + Interleaved parallel execution
"""
        
        # Stimを変換
        dsl_lines = converter.convert(circuit)
        
        # ファイル名
        filename = f"optimal_{name}"
        
        # 出力
        write_circuit_py_file(
            filename,
            dsl_lines,
            qubit_count=17,
            data_qubits=data_qubits,
            ancilla_qubits=ancilla_qubits,
            description=description
        )
    
    print()
    print("=" * 60)
    print(f"Generated {len(patterns)} circuit files in {circuits_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()