"""
巳型（蛇行）スケジューリング - 2パターン生成

(1) 全CNOT後に一斉測定: stim_snake_batch_measure.py
(2) スタビライザごとに測定: stim_snake_per_stabilizer.py

巳型実行順序（左上開始）:
  Z03 → X0134 → Z1245 → X12 → Z58 → X4578 → Z3467 → X67

Hook error avoiding順序:
  Xスタビライザ（Z型）: NW→NE→SW→SE
  Zスタビライザ（逆N字）: NW→SW→NE→SE

補助量子ビットID（通過順に割り当て）:
  Z03: 9, X0134: 10, Z1245: 11, X12: 12,
  Z58: 13, X4578: 14, Z3467: 15, X67: 16

使い方:
    from stim_snake_schedule import SnakeBatchMeasureGenerator, SnakePerStabilizerGenerator
    
    gen1 = SnakeBatchMeasureGenerator()
    circuit1 = gen1.generate_surface17()  # 全CNOT後に一斉測定
    
    gen2 = SnakePerStabilizerGenerator()
    circuit2 = gen2.generate_surface17()  # スタビライザごとに測定
"""

import stim


# ==========================================
# スタビライザ定義（巳型順序）
# ==========================================

# (名前, アンシラID, タイプ, データ量子ビット, CNOT順序)
# 巳型実行順序: Z03 → X0134 → Z1245 → X12 → Z58 → X4578 → Z3467 → X67
STABILIZERS = [
    # 1. Z03 (weight-2)
    ("Z03", 9, "Z", [0, 3], [0, 3]),
    
    # 2. X0134 (weight-4, Z型: NW→NE→SW→SE)
    ("X0134", 10, "X", [0, 1, 3, 4], [0, 1, 3, 4]),
    
    # 3. Z1245 (weight-4, 逆N字: NW→SW→NE→SE)
    ("Z1245", 11, "Z", [1, 2, 4, 5], [1, 4, 2, 5]),
    
    # 4. X12 (weight-2)
    ("X12", 12, "X", [1, 2], [1, 2]),
    
    # 5. Z58 (weight-2)
    ("Z58", 13, "Z", [5, 8], [5, 8]),
    
    # 6. X4578 (weight-4, Z型: NW→NE→SW→SE)
    ("X4578", 14, "X", [4, 5, 7, 8], [4, 5, 7, 8]),
    
    # 7. Z3467 (weight-4, 逆N字: NW→SW→NE→SE)
    ("Z3467", 15, "Z", [3, 4, 6, 7], [3, 6, 4, 7]),
    
    # 8. X67 (weight-2)
    ("X67", 16, "X", [6, 7], [6, 7]),
]


# ==========================================
# パターン1: 全CNOT後に一斉測定
# ==========================================

class SnakeBatchMeasureGenerator:
    """
    巳型スケジューリング - 全CNOT後に一斉測定
    
    各スタビライザで RESET → CNOT を実行し、最後にまとめてMEASURE
    （per_stabilizerとの違いは測定タイミングのみ）
    """
    
    def __init__(self, distance=3):
        self.distance = distance
        self.data_qubits = list(range(9))
        self.ancilla_qubits = list(range(9, 17))
    
    def generate_surface17(self) -> stim.Circuit:
        circuit = stim.Circuit()
        
        # 各スタビライザで RESET → H(X) → CNOTs → H(X) を実行
        for name, ancilla, stype, data_qubits, cnot_order in STABILIZERS:
            # RESET
            circuit.append("R", [ancilla])
            
            # Hadamard (X stabilizer)
            if stype == "X":
                circuit.append("H", [ancilla])
            
            # CNOTs
            for data in cnot_order:
                if stype == "X":
                    circuit.append("CX", [ancilla, data])
                else:
                    circuit.append("CX", [data, ancilla])
            
            # Hadamard (X stabilizer)
            if stype == "X":
                circuit.append("H", [ancilla])
        
        # 最後に全アンシラをまとめてMEASURE
        all_ancillas = [s[1] for s in STABILIZERS]
        circuit.append("M", all_ancillas)
        
        return circuit


# ==========================================
# パターン2: スタビライザごとに測定
# ==========================================

class SnakePerStabilizerGenerator:
    """
    巳型スケジューリング - スタビライザごとに測定
    
    各スタビライザで RESET → CNOT → MEASURE を完結
    """
    
    def __init__(self, distance=3):
        self.distance = distance
        self.data_qubits = list(range(9))
        self.ancilla_qubits = list(range(9, 17))
    
    def generate_surface17(self) -> stim.Circuit:
        circuit = stim.Circuit()
        
        for name, ancilla, stype, data_qubits, cnot_order in STABILIZERS:
            # RESET
            circuit.append("R", [ancilla])
            
            # Hadamard (X stabilizer)
            if stype == "X":
                circuit.append("H", [ancilla])
            
            # CNOTs
            for data in cnot_order:
                if stype == "X":
                    circuit.append("CX", [ancilla, data])
                else:
                    circuit.append("CX", [data, ancilla])
            
            # Hadamard (X stabilizer)
            if stype == "X":
                circuit.append("H", [ancilla])
            
            # MEASURE
            circuit.append("M", [ancilla])
        
        return circuit


# ==========================================
# generate_baselines.py 互換エイリアス
# ==========================================

NaiveStimGenerator = SnakePerStabilizerGenerator  # デフォルト


# ==========================================
# テスト
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Snake Schedule - 2 Patterns")
    print("=" * 60)
    print()
    
    print("Stabilizer order (snake from top-left):")
    for i, (name, anc, stype, data, order) in enumerate(STABILIZERS, 1):
        print(f"  {i}. {name} (anc={anc}, type={stype})")
        print(f"      data={data}, CNOT order={order}")
    print()
    
    # パターン1
    print("-" * 60)
    print("Pattern 1: Batch Measure (all CNOTs then measure)")
    print("-" * 60)
    gen1 = SnakeBatchMeasureGenerator()
    circuit1 = gen1.generate_surface17()
    print(circuit1)
    print()
    
    # パターン2
    print("-" * 60)
    print("Pattern 2: Per-Stabilizer Measure")
    print("-" * 60)
    gen2 = SnakePerStabilizerGenerator()
    circuit2 = gen2.generate_surface17()
    print(circuit2)
    print()
    
    # 変換テスト
    print("-" * 60)
    print("QCCD DSL Conversion Test (Pattern 2, first 30 lines)")
    print("-" * 60)
    try:
        from serial_converter import SerialConverter
        converter = SerialConverter()
        dsl_lines = converter.convert(circuit2)
        for line in dsl_lines[:30]:
            print(line)
        print(f"... (total {len(dsl_lines)} lines)")
    except ImportError:
        print("(serial_converter.py not found)")