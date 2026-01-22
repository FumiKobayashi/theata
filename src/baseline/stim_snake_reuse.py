"""
巳型（蛇行）スケジューリング + アンシラ再利用パターン生成

巳型実行順序（左上開始）:
  Z03 → X0134 → Z1245 → X12 → Z3467 → X67 → X4578 → Z58

アンシラ再利用パターン:
  1. 4アンシラ: A1->A2->A3->A4->A4->A3->A2->A1
  2. 2アンシラ: A1->A2->A2->A1->A1->A2->A2->A1
  3. 1アンシラ: 全部A1

Hook error avoiding順序:
  Xスタビライザ（Z型）: NW→NE→SW→SE
  Zスタビライザ（逆N字）: NW→SW→NE→SE

使い方:
    from stim_snake_reuse import (
        SnakeReuse4Generator,
        SnakeReuse2Generator,
        SnakeReuse1Generator,
    )
    
    gen = SnakeReuse4Generator()
    circuit = gen.generate_surface17()
"""

import stim


# ==========================================
# スタビライザ定義（巳型順序）
# ==========================================

# (名前, タイプ, データ量子ビット, CNOT順序)
STABILIZERS = [
    ("Z03", "Z", [0, 3], [0, 3]),
    ("X0134", "X", [0, 1, 3, 4], [0, 1, 3, 4]),
    ("Z1245", "Z", [1, 2, 4, 5], [1, 4, 2, 5]),
    ("X12", "X", [1, 2], [1, 2]),
    ("Z3467", "Z", [3, 4, 6, 7], [3, 6, 4, 7]),
    ("X67", "X", [6, 7], [6, 7]),
    ("X4578", "X", [4, 5, 7, 8], [4, 5, 7, 8]),
    ("Z58", "Z", [5, 8], [5, 8]),
]

# アンシラ割り当てパターン
# 4アンシラ: A1->A2->A3->A4->A4->A3->A2->A1
ANCILLA_PATTERN_4 = [9, 10, 11, 12, 12, 11, 10, 9]

# 2アンシラ: A1->A2->A2->A1->A1->A2->A2->A1
ANCILLA_PATTERN_2 = [9, 10, 10, 9, 9, 10, 10, 9]

# 1アンシラ: 全部A1
ANCILLA_PATTERN_1 = [9, 9, 9, 9, 9, 9, 9, 9]


# ==========================================
# 回路生成の共通関数
# ==========================================

def generate_circuit_with_reuse(ancilla_pattern: list) -> stim.Circuit:
    """
    指定されたアンシラパターンで回路を生成
    スタビライザごとに RESET → H(X) → CNOTs → H(X) → MEASURE
    """
    circuit = stim.Circuit()
    
    for i, (name, stype, data_qubits, cnot_order) in enumerate(STABILIZERS):
        ancilla = ancilla_pattern[i]
        
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
# 各パターンのジェネレータクラス
# ==========================================

class SnakeReuse4Generator:
    """
    巳型スケジューリング - 4アンシラ再利用
    パターン: A1->A2->A3->A4->A4->A3->A2->A1
    """
    
    def __init__(self, distance=3):
        self.distance = distance
        self.data_qubits = list(range(9))
        self.ancilla_qubits = [9, 10, 11, 12]
    
    def generate_surface17(self) -> stim.Circuit:
        return generate_circuit_with_reuse(ANCILLA_PATTERN_4)


class SnakeReuse2Generator:
    """
    巳型スケジューリング - 2アンシラ再利用
    パターン: A1->A2->A2->A1->A1->A2->A2->A1
    """
    
    def __init__(self, distance=3):
        self.distance = distance
        self.data_qubits = list(range(9))
        self.ancilla_qubits = [9, 10]
    
    def generate_surface17(self) -> stim.Circuit:
        return generate_circuit_with_reuse(ANCILLA_PATTERN_2)


class SnakeReuse1Generator:
    """
    巳型スケジューリング - 1アンシラ再利用
    パターン: 全部A1
    """
    
    def __init__(self, distance=3):
        self.distance = distance
        self.data_qubits = list(range(9))
        self.ancilla_qubits = [9]
    
    def generate_surface17(self) -> stim.Circuit:
        return generate_circuit_with_reuse(ANCILLA_PATTERN_1)


# generate_baselines.py 互換エイリアス
NaiveStimGenerator = SnakeReuse4Generator


# ==========================================
# テスト
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Snake Schedule with Ancilla Reuse")
    print("=" * 60)
    print()
    
    print("Stabilizer order (snake from top-left):")
    for i, (name, stype, data, order) in enumerate(STABILIZERS):
        a4 = ANCILLA_PATTERN_4[i]
        a2 = ANCILLA_PATTERN_2[i]
        a1 = ANCILLA_PATTERN_1[i]
        print(f"  {i+1}. {name:8s} ({stype}) | 4anc:{a4} | 2anc:{a2} | 1anc:{a1}")
    print()
    
    # パターン1: 4アンシラ
    print("-" * 60)
    print("Pattern 1: 4 Ancilla Reuse (A1->A2->A3->A4->A4->A3->A2->A1)")
    print("-" * 60)
    gen4 = SnakeReuse4Generator()
    circuit4 = gen4.generate_surface17()
    print(circuit4)
    print()
    
    # パターン2: 2アンシラ
    print("-" * 60)
    print("Pattern 2: 2 Ancilla Reuse (A1->A2->A2->A1->A1->A2->A2->A1)")
    print("-" * 60)
    gen2 = SnakeReuse2Generator()
    circuit2 = gen2.generate_surface17()
    print(circuit2)
    print()
    
    # パターン3: 1アンシラ
    print("-" * 60)
    print("Pattern 3: 1 Ancilla Reuse (all A1)")
    print("-" * 60)
    gen1 = SnakeReuse1Generator()
    circuit1 = gen1.generate_surface17()
    print(circuit1)
