"""
Surface-17 (d=3) 最強スケジューリング - 16パターン生成

Weight-4のX/Zスタビライザの順序を4×4=16通り試す。
ペアの実行順は固定（右下→左下→右上→左上）。

X側の順序（X9, X11）:
  X_A: X9: D4→D1→D0→D3, X11: D8→D5→D4→D7
  X_B: X9: D4→D1→D3→D0, X11: D8→D5→D7→D4
  X_C: X9: D4→D0→D1→D3, X11: D8→D4→D5→D7
  X_D: X9: D4→D0→D3→D1, X11: D8→D4→D7→D5

Z側の順序（Z14, Z15）:
  Z_A: Z14: D5→D1→D2→D4, Z15: D7→D3→D4→D6
  Z_B: Z14: D5→D1→D4→D2, Z15: D7→D3→D6→D4
  Z_C: Z14: D5→D4→D1→D2, Z15: D7→D6→D3→D4
  Z_D: Z14: D5→D4→D2→D1, Z15: D7→D6→D4→D3

ペア（並行実行）:
  左上: X9 + Z13, 共有: D0, D3
  右上: Z14 + X10, 共有: D1, D2
  左下: Z15 + X12, 共有: D6, D7
  右下: X11 + Z16, 共有: D5, D8

使い方:
    from stim_optimal_16patterns import generate_all_patterns
    patterns = generate_all_patterns()
    # patterns["XA_ZA"], patterns["XA_ZB"], ... 等でアクセス
"""

import stim


# Weight-4 X スタビライザの順序パターン (hook error free)
X_PATTERNS = {
    "A": {9: [4, 0, 1, 3], 11: [8, 4, 5, 7]},
    "B": {9: [4, 0, 3, 1], 11: [8, 4, 7, 5]},
    "C": {9: [4, 1, 0, 3], 11: [8, 5, 4, 7]},
    "D": {9: [4, 1, 3, 0], 11: [8, 5, 7, 4]},
}

# Weight-4 Z スタビライザの順序パターン
Z_PATTERNS = {
    "A": {14: [5, 1, 2, 4], 15: [7, 3, 4, 6]},
    "B": {14: [5, 1, 4, 2], 15: [7, 3, 6, 4]},
    "C": {14: [5, 4, 1, 2], 15: [7, 6, 3, 4]},
    "D": {14: [5, 4, 2, 1], 15: [7, 6, 4, 3]},
}

# Weight-2 スタビライザ（共有データへの連続CNOTを考慮して動的に決定）
WEIGHT2_STABILIZERS = {
    13: [0, 3],  # Z13: 共有=D0,D3 (X9とペア)
    10: [1, 2],  # X10: 共有=D1,D2 (Z14とペア)
    12: [6, 7],  # X12: 共有=D6,D7 (Z15とペア)
    16: [5, 8],  # Z16: 共有=D5,D8 (X11とペア)
}


def get_interleaved_sequence(w4_anc, w4_order, w2_anc, w2_data, w4_type):
    """
    Weight-4とWeight-2を並行実行するインターリーブ順序を生成。
    共有データへの連続CNOTを実現する。
    
    Args:
        w4_anc: Weight-4のアンシラID
        w4_order: Weight-4のCNOT順序 [d1, d2, d3, d4]
        w2_anc: Weight-2のアンシラID
        w2_data: Weight-2のデータ [d1, d2]
        w4_type: 'X' or 'Z' (Weight-4側のタイプ)
    
    Returns:
        list of (type, ancilla, data): CNOTのシーケンス
    """
    sequence = []
    w2_set = set(w2_data)
    w2_remaining = list(w2_data)
    
    for data in w4_order:
        # Weight-4のCNOT
        sequence.append((w4_type, w4_anc, data))
        
        # 共有データなら、Weight-2のCNOTを連続で追加
        if data in w2_set and data in w2_remaining:
            w2_type = 'Z' if w4_type == 'X' else 'X'
            sequence.append((w2_type, w2_anc, data))
            w2_remaining.remove(data)
    
    # 残りのWeight-2（共有でないデータがあれば）
    for data in w2_remaining:
        w2_type = 'Z' if w4_type == 'X' else 'X'
        sequence.append((w2_type, w2_anc, data))
    
    return sequence


def generate_pattern(x_pattern_key, z_pattern_key):
    """
    指定されたX/Zパターンで回路を生成。
    
    ペア実行順: 右下 → 左下 → 右上 → 左上
    """
    x_orders = X_PATTERNS[x_pattern_key]
    z_orders = Z_PATTERNS[z_pattern_key]
    
    circuit = stim.Circuit()
    
    # ペア定義: (w4_anc, w4_order, w2_anc, w2_data, w4_type)
    # 右下: X11 + Z16, 共有: D5, D8
    # 左下: Z15 + X12, 共有: D6, D7
    # 右上: Z14 + X10, 共有: D1, D2
    # 左上: X9 + Z13, 共有: D0, D3
    
    pairs = [
        # 右下: X11 (weight-4) + Z16 (weight-2)
        (11, x_orders[11], 16, [5, 8], 'X'),
        # 左下: Z15 (weight-4) + X12 (weight-2)
        (15, z_orders[15], 12, [6, 7], 'Z'),
        # 右上: Z14 (weight-4) + X10 (weight-2)
        (14, z_orders[14], 10, [1, 2], 'Z'),
        # 左上: X9 (weight-4) + Z13 (weight-2)
        (9, x_orders[9], 13, [0, 3], 'X'),
    ]
    
    for w4_anc, w4_order, w2_anc, w2_data, w4_type in pairs:
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
                circuit.append("CX", [anc, data])  # X: ancilla -> data
            else:
                circuit.append("CX", [data, anc])  # Z: data -> ancilla
        
        # Hadamard on X ancillas
        if w4_type == 'X':
            circuit.append("H", [w4_anc])
        else:
            circuit.append("H", [w2_anc])
        
        # MEASURE (並列)
        circuit.append("M", [w4_anc, w2_anc])
    
    return circuit


def generate_all_patterns():
    """全16パターンを生成して辞書で返す"""
    patterns = {}
    
    for x_key in ["A", "B", "C", "D"]:
        for z_key in ["A", "B", "C", "D"]:
            name = f"X{x_key}_Z{z_key}"
            patterns[name] = generate_pattern(x_key, z_key)
    
    return patterns


class OptimalPatternGenerator:
    """generate_baselines.py互換のインターフェース"""
    
    def __init__(self, x_pattern="A", z_pattern="A", distance=3):
        self.distance = distance
        self.x_pattern = x_pattern
        self.z_pattern = z_pattern
        self.data_qubits = list(range(9))
        self.ancilla_qubits = list(range(9, 17))
    
    def generate_surface17(self):
        return generate_pattern(self.x_pattern, self.z_pattern)


# 便宜上のエイリアス
NaiveStimGenerator = OptimalPatternGenerator


if __name__ == "__main__":
    print("=" * 60)
    print("Optimal Scheduling - 16 Patterns")
    print("=" * 60)
    
    patterns = generate_all_patterns()
    
    for name, circuit in patterns.items():
        print(f"\n{name}:")
        print(circuit)
        print()
    
    # 1つだけ詳細表示
    print("=" * 60)
    print("Example: XA_ZA with serial_converter")
    print("=" * 60)
    
    try:
        from serial_converter import SerialConverter
        converter = SerialConverter()
        circuit = patterns["XA_ZA"]
        qccd_lines = converter.convert(circuit)
        for line in qccd_lines[:40]:
            print(line)
        print("...")
        print(f"(Total {len(qccd_lines)} lines)")
    except ImportError:
        print("(serial_converter.py not found)")