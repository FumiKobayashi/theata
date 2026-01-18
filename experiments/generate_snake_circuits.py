"""
巳型スケジューリングの.pyファイル生成

2つの回路ファイルを生成:
  - snake_batch_measure.py: 全CNOT後に一斉測定
  - snake_per_stabilizer.py: スタビライザごとに測定

使い方:
    python generate_snake_circuits.py
"""

import sys
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
circuits_dir = current_dir.parent / 'circuits'
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

from baseline.stim_snake_schedule import SnakeBatchMeasureGenerator, SnakePerStabilizerGenerator
from baseline.serial_converter import SerialConverter


def write_circuit_py_file(filename, dsl_lines, description=""):
    """Python形式の回路定義ファイルを書き出す"""
    filepath = circuits_dir / f"{filename}.py"
    
    data_qubits = list(range(9))
    ancilla_qubits = list(range(9, 17))
    
    content = f'''"""
{filename}
{description}

巳型（蛇行）スケジューリング - 左上開始
実行順序: Z03 → X0134 → Z1245 → X12 → Z3467 → X67 → X4578 → Z58

Hook error avoiding順序:
  Xスタビライザ（Z型）: NW→NE→SW→SE
  Zスタビライザ（逆N字）: NW→SW→NE→SE

補助量子ビットID（通過順）:
  Z03:9, X0134:10, Z1245:11, X12:12, Z3467:13, X67:14, X4578:15, Z58:16
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


def main():
    # circuits_dir が存在しない場合は作成
    circuits_dir.mkdir(parents=True, exist_ok=True)
    
    converter = SerialConverter()
    
    print("=" * 60)
    print("Generating Snake Schedule Circuits")
    print("=" * 60)
    print()
    
    # パターン1: 全CNOT後に一斉測定
    gen1 = SnakeBatchMeasureGenerator()
    circuit1 = gen1.generate_surface17()
    dsl1 = converter.convert(circuit1)
    
    write_circuit_py_file(
        "snake_batch_measure",
        dsl1,
        description="全CNOT後に一斉測定 (Batch Measure)"
    )
    
    # パターン2: スタビライザごとに測定
    gen2 = SnakePerStabilizerGenerator()
    circuit2 = gen2.generate_surface17()
    dsl2 = converter.convert(circuit2)
    
    write_circuit_py_file(
        "snake_per_stabilizer",
        dsl2,
        description="スタビライザごとに測定 (Per-Stabilizer Measure)"
    )
    
    print()
    print("=" * 60)
    print(f"Generated 2 circuit files in {circuits_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
