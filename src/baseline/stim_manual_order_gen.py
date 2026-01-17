"""
Surface-17 (d=3) Stim Circuit Generator - Manual Measurement Order

隣接するXスタビライザとZスタビライザを並行(interleaved)に実行。
RESETとMEASUREのみ並列、CNOTは順次実行。

データキュービット配置:
    0 --- 1 --- 2
    |     |     |
    3 --- 4 --- 5
    |     |     |
    6 --- 7 --- 8

4つの測定領域 (M位置) と対応するスタビライザペア:
  左上M: X0X1X3X4 (A9) + Z0Z3 (A13)
  右上M: X1X2 (A10) + Z1Z2Z4Z5 (A14)
  左下M: X6X7 (A12) + Z3Z4Z6Z7 (A15)
  右下:  X4X5X7X8 (A11) + Z5Z8 (A16)
"""

import stim


class ManualOrderStimGenerator:
    """
    Manual Measurement Order: 隣接スタビライザ並行実行
    8アンシラ版（再利用なし）
    """
    
    def __init__(self):
        self.data_qubits = list(range(9))         # 0-8
        self.ancilla_qubits = list(range(9, 17))  # 9-16
        
        # 4つの領域定義
        # 各領域: (x_ancilla, z_ancilla, interleaved_cnot_sequence)
        # interleaved_cnot_sequence: [(type, ancilla, data), ...]
        #   type: 'X' = ancilla->data, 'Z' = data->ancilla
        
        self.regions = [
            # 左上M: X0X1X3X4 (A9) + Z0Z3 (A13)
            # 順序: X(4) -> X(3) -> Z(3) -> X(1) -> X(0) -> Z(0)
            {
                'x_anc': 9,
                'z_anc': 13,
                'sequence': [
                    ('X', 9, 4),   # CNOT(9, 4)
                    ('X', 9, 3),   # CNOT(9, 3)
                    ('Z', 3, 13),  # CNOT(3, 13)
                    ('X', 9, 1),   # CNOT(9, 1)
                    ('X', 9, 0),   # CNOT(9, 0)
                    ('Z', 0, 13),  # CNOT(0, 13)
                ]
            },
            # 右上M: X1X2 (A10) + Z1Z2Z4Z5 (A14)
            # 順序: Z(5) -> Z(4) -> Z(2) -> X(2) -> Z(1) -> X(1)
            {
                'x_anc': 10,
                'z_anc': 14,
                'sequence': [
                    ('Z', 5, 14),  # CNOT(5, 14)
                    ('Z', 4, 14),  # CNOT(4, 14)
                    ('Z', 2, 14),  # CNOT(2, 14)
                    ('X', 10, 2),  # CNOT(10, 2)
                    ('Z', 1, 14),  # CNOT(1, 14)
                    ('X', 10, 1),  # CNOT(10, 1)
                ]
            },
            # 左下M: X6X7 (A12) + Z3Z4Z6Z7 (A15)
            # 順序: Z(7) -> X(7) -> Z(6) -> X(6) -> Z(4) -> Z(3)
            {
                'x_anc': 12,
                'z_anc': 15,
                'sequence': [
                    ('Z', 7, 15),  # CNOT(7, 15)
                    ('X', 12, 7),  # CNOT(12, 7)
                    ('Z', 6, 15),  # CNOT(6, 15)
                    ('X', 12, 6),  # CNOT(12, 6)
                    ('Z', 4, 15),  # CNOT(4, 15)
                    ('Z', 3, 15),  # CNOT(3, 15)
                ]
            },
            # 右下: X4X5X7X8 (A11) + Z5Z8 (A16)
            # 順序: X(8) -> Z(8) -> X(7) -> X(5) -> Z(5) -> X(4)
            {
                'x_anc': 11,
                'z_anc': 16,
                'sequence': [
                    ('X', 11, 8),  # CNOT(11, 8)
                    ('Z', 8, 16),  # CNOT(8, 16)
                    ('X', 11, 7),  # CNOT(11, 7)
                    ('X', 11, 5),  # CNOT(11, 5)
                    ('Z', 5, 16),  # CNOT(5, 16)
                    ('X', 11, 4),  # CNOT(11, 4)
                ]
            },
        ]

    def generate(self):
        """並行実行のStim回路を生成（TICKあり版）"""
        circuit = stim.Circuit()
        
        for region in self.regions:
            x_anc = region['x_anc']
            z_anc = region['z_anc']
            sequence = region['sequence']
            
            # RESET (並列)
            circuit.append("R", [x_anc, z_anc])
            
            # Hadamard on X ancilla
            circuit.append("H", [x_anc])
            circuit.append("TICK")
            
            # CNOTs (順次実行、interleavedパターン)
            for op_type, anc, data in sequence:
                if op_type == 'X':
                    circuit.append("CNOT", [anc, data])
                else:  # Z
                    circuit.append("CNOT", [data, anc])
                circuit.append("TICK")
            
            # Hadamard on X ancilla
            circuit.append("H", [x_anc])
            circuit.append("TICK")
            
            # MEASURE (並列)
            circuit.append("M", [x_anc, z_anc])
            circuit.append("TICK")
        
        return circuit

    def generate_for_converter(self):
        """
        serial_converter.py用のStim回路を生成
        （TICKなし、CNOTを順次実行）
        """
        circuit = stim.Circuit()
        
        for region in self.regions:
            x_anc = region['x_anc']
            z_anc = region['z_anc']
            sequence = region['sequence']
            
            # RESET (並列)
            circuit.append("R", [x_anc, z_anc])
            
            # Hadamard on X ancilla
            circuit.append("H", [x_anc])
            
            # CNOTs (順次実行、interleavedパターン)
            for op_type, anc, data in sequence:
                if op_type == 'X':
                    circuit.append("CNOT", [anc, data])
                else:  # Z
                    circuit.append("CNOT", [data, anc])
            
            # Hadamard on X ancilla
            circuit.append("H", [x_anc])
            
            # MEASURE (並列)
            circuit.append("M", [x_anc, z_anc])
        
        return circuit


# ========================================
# 使用例
# ========================================
if __name__ == "__main__":
    gen = ManualOrderStimGenerator()
    
    print("=" * 60)
    print("Manual Order - With TICK (for visualization)")
    print("=" * 60)
    circuit_tick = gen.generate()
    print(circuit_tick)
    print()
    
    print("=" * 60)
    print("Manual Order - For serial_converter")
    print("=" * 60)
    circuit_serial = gen.generate_for_converter()
    print(circuit_serial)
    print()
    
    # serial_converterとの連携
    try:
        from serial_converter import SerialConverter
        converter = SerialConverter()
        
        print("=" * 60)
        print("QCCD Intermediate Language Output")
        print("=" * 60)
        qccd_lines = converter.convert(circuit_serial)
        for line in qccd_lines:
            print(line)
    except ImportError:
        print("(serial_converter.py not found, skipping conversion)")
