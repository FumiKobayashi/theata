"""
Surface-17 (d=3) Hook Error Free Stim Generator - Interleaved
Reference: Tomita & Svore 2014

generate_baselines.py と同じインターフェースで使用可能。

使い方:
    from stim_hook_free_interleaved import HookFreeInterleavedGenerator
    gen = HookFreeInterleavedGenerator()
    circuit = gen.generate_surface17()
"""

import stim


class HookFreeInterleavedGenerator:
    """
    Hook error free な CNOT 順序で Surface-17 回路を生成。
    実行順序: X9 → Z13 → X10 → Z14 → X11 → Z15 → X12 → Z16
    （X と Z を交互に実行）
    """
    
    def __init__(self, distance=3):
        self.distance = distance
        # Surface-17 (d=3) 固定のマッピング定義
        # Data: 0-8, Ancilla: 9-16 (計17量子ビット)
        self.data_qubits = list(range(9))
        self.ancilla_qubits = list(range(9, 17))
        
        # Hook error free CNOT orders
        # X stabilizers: NE → NW → SE → SW for weight-4
        self.x_stabilizers = {
            9:  [1, 0, 4, 3],  # X0X1X3X4
            10: [1, 2],        # X1X2
            11: [5, 4, 8, 7],  # X4X5X7X8
            12: [6, 7],        # X6X7
        }
        
        # Z stabilizers: NW → NE → SW → SE for weight-4
        self.z_stabilizers = {
            13: [0, 3],        # Z0Z3
            14: [1, 2, 4, 5],  # Z1Z2Z4Z5
            15: [3, 4, 6, 7],  # Z3Z4Z6Z7
            16: [5, 8],        # Z5Z8
        }
        
        # Interleaved pairs: (X_anc, Z_anc)
        self.pairs = [
            (9, 13),   # X9 → Z13
            (10, 14),  # X10 → Z14
            (11, 15),  # X11 → Z15
            (12, 16),  # X12 → Z16
        ]

    def generate_surface17(self):
        """
        Hook error free な Surface-17 シンドローム測定回路を生成。
        X と Z を交互に実行（Interleaved）。
        """
        circuit = stim.Circuit()
        
        for x_anc, z_anc in self.pairs:
            # X stabilizer
            circuit.append("R", [x_anc])
            circuit.append("H", [x_anc])
            for data in self.x_stabilizers[x_anc]:
                circuit.append("CX", [x_anc, data])
            circuit.append("H", [x_anc])
            circuit.append("M", [x_anc])
            
            # Z stabilizer
            circuit.append("R", [z_anc])
            for data in self.z_stabilizers[z_anc]:
                circuit.append("CX", [data, z_anc])
            circuit.append("M", [z_anc])
        
        return circuit


# generate_baselines.py との互換性のためのエイリアス
NaiveStimGenerator = HookFreeInterleavedGenerator


if __name__ == "__main__":
    gen = HookFreeInterleavedGenerator()
    circuit = gen.generate_surface17()
    
    print("=" * 60)
    print("Hook Error Free - Interleaved")
    print("=" * 60)
    print(f"Data qubits: {gen.data_qubits}")
    print(f"Ancilla qubits: {gen.ancilla_qubits}")
    print()
    print("Stim Circuit:")
    print(circuit)
