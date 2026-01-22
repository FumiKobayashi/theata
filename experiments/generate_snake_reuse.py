"""
巳型スケジューリング + アンシラ再利用の.pyファイル生成

3つの回路ファイルを生成:
  - snake_reuse4.py: 4アンシラ再利用
  - snake_reuse2.py: 2アンシラ再利用
  - snake_reuse1.py: 1アンシラ再利用

使い方:
    python generate_snake_reuse.py
"""

import sys
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
circuits_dir = current_dir.parent / 'circuits'
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

from baseline.stim_snake_reuse import (
    SnakeReuse4Generator,
    SnakeReuse2Generator,
    SnakeReuse1Generator,
    ANCILLA_PATTERN_4,
    ANCILLA_PATTERN_2,
    ANCILLA_PATTERN_1,
)
from baseline.serial_converter import SerialConverter


def write_circuit_py_file(filename, dsl_lines, ancilla_qubits, ancilla_pattern, description=""):
    """Python形式の回路定義ファイルを書き出す"""
    filepath = circuits_dir / f"{filename}.py"
    
    data_qubits = list(range(9))
    
    # アンシラパターンの文字列表現
    pattern_str = "->".join(f"A{p-8}" for p in ancilla_pattern)
    
    content = f'''"""
{filename}
{description}

巳型（蛇行）スケジューリング - 左上開始
実行順序: Z03 → X0134 → Z1245 → X12 → Z3467 → X67 → X4578 → Z58

アンシラ割り当てパターン: {pattern_str}
使用アンシラ: {ancilla_qubits}

Hook error avoiding順序:
  Xスタビライザ（Z型）: NW→NE→SW→SE
  Zスタビライザ（逆N字）: NW→SW→NE→SE
"""

# メタデータ (exp_detailed.py で必須)
QUBIT_COUNT = {9 + len(ancilla_qubits)}
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
    for i, q in enumerate(ancilla_qubits):
        content += f"QUBIT {q} AT (2, {i})\n"

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


def main():
    # circuits_dir が存在しない場合は作成
    circuits_dir.mkdir(parents=True, exist_ok=True)
    
    converter = SerialConverter()
    
    print("=" * 60)
    print("Generating Snake Schedule + Ancilla Reuse Circuits")
    print("=" * 60)
    print()
    
    # パターン1: 4アンシラ
    gen4 = SnakeReuse4Generator()
    circuit4 = gen4.generate_surface17()
    dsl4 = converter.convert(circuit4)
    
    write_circuit_py_file(
        "snake_reuse4",
        dsl4,
        gen4.ancilla_qubits,
        ANCILLA_PATTERN_4,
        description="4アンシラ再利用 (A1->A2->A3->A4->A4->A3->A2->A1)"
    )
    
    # パターン2: 2アンシラ
    gen2 = SnakeReuse2Generator()
    circuit2 = gen2.generate_surface17()
    dsl2 = converter.convert(circuit2)
    
    write_circuit_py_file(
        "snake_reuse2",
        dsl2,
        gen2.ancilla_qubits,
        ANCILLA_PATTERN_2,
        description="2アンシラ再利用 (A1->A2->A2->A1->A1->A2->A2->A1)"
    )
    
    # パターン3: 1アンシラ
    gen1 = SnakeReuse1Generator()
    circuit1 = gen1.generate_surface17()
    dsl1 = converter.convert(circuit1)
    
    write_circuit_py_file(
        "snake_reuse1",
        dsl1,
        gen1.ancilla_qubits,
        ANCILLA_PATTERN_1,
        description="1アンシラ再利用 (全部A1)"
    )
    
    print()
    print("=" * 60)
    print(f"Generated 3 circuit files in {circuits_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
