"""
Surface-17 (d=3) Hook Error Free Stim Generator - X then Z
Reference: Tomita & Svore 2014

generate_baselines.py と同じインターフェースで使用可能。

使い方:
    from stim_hook_free_x_then_z import HookFreeXthenZGenerator
    gen = HookFreeXthenZGenerator()
    circuit = gen.generate_surface17()
"""

import stim


class HookFreeXthenZGenerator:
    """
    Hook error free な CNOT 順序で Surface-17 回路を生成。
    実行順序: X9 → X10 → X11 → X12 → Z13 → Z14 → Z15 → Z16
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

    def generate_surface17(self):
        """
        Hook error free な Surface-17 シンドローム測定回路を生成。
        X stabilizers を全て実行した後、Z stabilizers を実行。
        """
        circuit = stim.Circuit()
        
        # X stabilizers first
        for anc in [9, 10, 11, 12]:
            circuit.append("R", [anc])
            circuit.append("H", [anc])
            for data in self.x_stabilizers[anc]:
                circuit.append("CX", [anc, data])
            circuit.append("H", [anc])
            circuit.append("M", [anc])
        
        # Then Z stabilizers
        for anc in [13, 14, 15, 16]:
            circuit.append("R", [anc])
            for data in self.z_stabilizers[anc]:
                circuit.append("CX", [data, anc])
            circuit.append("M", [anc])
        
        return circuit


# generate_baselines.py との互換性のためのエイリアス
NaiveStimGenerator = HookFreeXthenZGenerator


if __name__ == "__main__":
    gen = HookFreeXthenZGenerator()
    circuit = gen.generate_surface17()
    
    print("=" * 60)
    print("Hook Error Free - X then Z")
    print("=" * 60)
    print(f"Data qubits: {gen.data_qubits}")
    print(f"Ancilla qubits: {gen.ancilla_qubits}")
    print()
    print("Stim Circuit:")
    print(circuit)
