import os
import sys
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
circuits_dir = current_dir.parent / 'circuits'
sys.path.insert(0, str(src_dir))

from baseline.stim_naive_4step import Naive4StepGenerator
from baseline.serial_converter import SerialConverter

def write_circuit_py_file(filename, dsl_lines, qubit_count, data_qubits, ancilla_qubits):
    """
    Python形式の回路定義ファイルを書き出す
    """
    filepath = circuits_dir / f"{filename}.py"
    
    content = f'''"""
Baseline Circuit: {filename}
Generated from Naive Stim Circuit (Serial Conversion)
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
    
    # QUBIT 宣言の生成 (Parserを通すために必要)
    # 座標は exp_detailed.py で上書きされるため、パーサーが通る適当な値でOK
    # ここでは Dataを左、Ancillaを右に並べる
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
    # 1. Generate Naive Stim Circuit
    gen = Naive4StepGenerator()
    stim_circuit = gen.generate_surface17()
    
    # 2. Convert to QCCD DSL Strings
    converter = SerialConverter()
    dsl_lines = converter.convert(stim_circuit)
    
    # 3. Save as .py file
    write_circuit_py_file(
        "exp1_Naive_d3",
        dsl_lines,
        qubit_count=17,
        data_qubits=gen.data_qubits,
        ancilla_qubits=gen.ancilla_qubits
    )

if __name__ == "__main__":
    main()