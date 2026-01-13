import stim

class NaiveStimGenerator:
    def __init__(self, distance=3):
        self.distance = distance
        # Surface-17 (d=3) 固定のマッピング定義
        # Data: 0-8, Ancilla: 9-16 (計17量子ビット)
        self.data_qubits = list(range(9))
        self.ancilla_qubits = list(range(9, 17))

    def generate_surface17(self):
        """
        d=3 の Surface Code シンドローム測定回路を生成する。
        QCCDの制約は考慮せず、アルゴリズム通りの順序でゲートを並べる。
        """
        circuit = stim.Circuit()

        # スタビライザ定義 (X: 9-12, Z: 13-16)
        x_stabilizers = {
            9:  [0, 1, 3, 4], 10: [1, 2],
            11: [4, 5, 7, 8], 12: [6, 7]
        }
        z_stabilizers = {
            13: [0, 3],       14: [1, 2, 4, 5],
            15: [3, 4, 6, 7], 16: [5, 8]
        }

        # 1. Reset & Tick
        circuit.append("R", self.ancilla_qubits)
        circuit.append("TICK")

        # 2. X Stabilizers (H -> CNOT -> H -> M)
        circuit.append("H", list(x_stabilizers.keys()))
        circuit.append("TICK")

        # CNOT列 (Naive: 定義順に実行)
        for ancilla, targets in x_stabilizers.items():
            for target in targets:
                # X check: Control=Ancilla, Target=Data
                circuit.append("CNOT", [ancilla, target])
        circuit.append("TICK")

        circuit.append("H", list(x_stabilizers.keys()))
        circuit.append("TICK")
        circuit.append("M", list(x_stabilizers.keys()))

        # 3. Z Stabilizers (CNOT -> M)
        # CNOT列
        for ancilla, targets in z_stabilizers.items():
            for target in targets:
                # Z check: Control=Data, Target=Ancilla
                circuit.append("CNOT", [target, ancilla])
        circuit.append("TICK")

        circuit.append("M", list(z_stabilizers.keys()))

        return circuit