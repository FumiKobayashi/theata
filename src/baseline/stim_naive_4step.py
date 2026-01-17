"""
Surface-17 (d=3) Naive Stim Generator - 4-Step Parallel Schedule
Reference: Tomita & Svore 2014

超伝導向けの4-step並列スケジュールをStim形式で記述。
Hook error freeなCNOT順序を使用。

serial_converterで直列化されると、超伝導ユーザーがそのまま入力した形になる。

4-Step Schedule:
  Step 1: 各スタビライザの1番目のCNOT
  Step 2: 各スタビライザの2番目のCNOT
  Step 3: 各スタビライザの3番目のCNOT (weight-4のみ)
  Step 4: 各スタビライザの4番目のCNOT (weight-4のみ)

使い方:
    from stim_naive_4step import Naive4StepGenerator
    gen = Naive4StepGenerator()
    circuit = gen.generate_surface17()
"""

import stim


class Naive4StepGenerator:
    """
    超伝導向け4-step並列スケジュールでSurface-17回路を生成。
    serial_converterで直列化される前提。
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
            9:  [1, 0, 4, 3],  # X0X1X3X4 (weight-4)
            10: [1, 2],        # X1X2 (weight-2)
            11: [5, 4, 8, 7],  # X4X5X7X8 (weight-4)
            12: [6, 7],        # X6X7 (weight-2)
        }
        
        # Z stabilizers: NW → NE → SW → SE for weight-4
        self.z_stabilizers = {
            13: [0, 3],        # Z0Z3 (weight-2)
            14: [1, 2, 4, 5],  # Z1Z2Z4Z5 (weight-4)
            15: [3, 4, 6, 7],  # Z3Z4Z6Z7 (weight-4)
            16: [5, 8],        # Z5Z8 (weight-2)
        }

    def generate_surface17(self):
        """
        4-step並列スケジュールでSurface-17シンドローム測定回路を生成。
        """
        circuit = stim.Circuit()
        
        x_ancillas = [9, 10, 11, 12]
        z_ancillas = [13, 14, 15, 16]
        all_ancillas = x_ancillas + z_ancillas
        
        # Reset all ancillas
        circuit.append("R", all_ancillas)
        circuit.append("TICK")
        
        # Hadamard on X ancillas
        circuit.append("H", x_ancillas)
        circuit.append("TICK")
        
        # Step 1: 各スタビライザの1番目のCNOT
        # X stabilizers
        for anc in x_ancillas:
            data = self.x_stabilizers[anc][0]
            circuit.append("CX", [anc, data])
        # Z stabilizers
        for anc in z_ancillas:
            data = self.z_stabilizers[anc][0]
            circuit.append("CX", [data, anc])
        circuit.append("TICK")
        
        # Step 2: 各スタビライザの2番目のCNOT
        # X stabilizers
        for anc in x_ancillas:
            if len(self.x_stabilizers[anc]) >= 2:
                data = self.x_stabilizers[anc][1]
                circuit.append("CX", [anc, data])
        # Z stabilizers
        for anc in z_ancillas:
            if len(self.z_stabilizers[anc]) >= 2:
                data = self.z_stabilizers[anc][1]
                circuit.append("CX", [data, anc])
        circuit.append("TICK")
        
        # Step 3: 各スタビライザの3番目のCNOT (weight-4のみ)
        # X stabilizers
        for anc in x_ancillas:
            if len(self.x_stabilizers[anc]) >= 3:
                data = self.x_stabilizers[anc][2]
                circuit.append("CX", [anc, data])
        # Z stabilizers
        for anc in z_ancillas:
            if len(self.z_stabilizers[anc]) >= 3:
                data = self.z_stabilizers[anc][2]
                circuit.append("CX", [data, anc])
        circuit.append("TICK")
        
        # Step 4: 各スタビライザの4番目のCNOT (weight-4のみ)
        # X stabilizers
        for anc in x_ancillas:
            if len(self.x_stabilizers[anc]) >= 4:
                data = self.x_stabilizers[anc][3]
                circuit.append("CX", [anc, data])
        # Z stabilizers
        for anc in z_ancillas:
            if len(self.z_stabilizers[anc]) >= 4:
                data = self.z_stabilizers[anc][3]
                circuit.append("CX", [data, anc])
        circuit.append("TICK")
        
        # Hadamard on X ancillas
        circuit.append("H", x_ancillas)
        circuit.append("TICK")
        
        # Measure all ancillas
        circuit.append("M", all_ancillas)
        
        return circuit


# generate_baselines.py との互換性のためのエイリアス
NaiveStimGenerator = Naive4StepGenerator


if __name__ == "__main__":
    gen = Naive4StepGenerator()
    circuit = gen.generate_surface17()
    
    print("=" * 60)
    print("Naive 4-Step Parallel Schedule (for superconducting)")
    print("=" * 60)
    print(f"Data qubits: {gen.data_qubits}")
    print(f"Ancilla qubits: {gen.ancilla_qubits}")
    print()
    print("Stim Circuit:")
    print(circuit)
    print()
    
    # serial_converterで直列化した結果を確認
    try:
        from serial_converter import SerialConverter
        converter = SerialConverter()
        qccd_lines = converter.convert(circuit)
        print("=" * 60)
        print("Serialized by serial_converter (QCCD DSL):")
        print("=" * 60)
        for line in qccd_lines[:30]:
            print(line)
        print("...")
        print(f"(Total {len(qccd_lines)} lines)")
    except ImportError:
        print("(serial_converter.py not found)")
