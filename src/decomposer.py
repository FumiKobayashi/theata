# src/decomposer.py
from typing import List
from .parser import QuantumInstruction  # Step 2で作ったクラス
from .qccd_native import NativeInstruction, OpType
from .parameters import DURATION_R, DURATION_MS, DURATION_MEASURE, DURATION_RESET

class GateDecomposer:
    """
    論理ゲートをQCCDネイティブ命令列に変換するクラス。
    """

    def decompose(self, instruction: QuantumInstruction) -> List[NativeInstruction]:
        """
        1つの論理命令を受け取り、物理命令のリストを返す
        """
        name = instruction.name
        targets = instruction.targets
        
        # --- 1量子ビットゲート ---
        if name == "H":
            # H = Ry(pi/2) * Rx(pi)  (仮の分解)
            return [
                NativeInstruction(OpType.R, targets, params=[3.14159], duration=DURATION_R), # Rx
                NativeInstruction(OpType.R, targets, params=[1.5708], duration=DURATION_R)   # Ry
            ]
        
        elif name in ["X", "Y", "Z"]:
            # Pauliゲートは1回の回転で実装可能
            return [
                NativeInstruction(OpType.R, targets, params=[3.14159], duration=DURATION_R)
            ]

        # --- 2量子ビットゲート ---
        elif name == "CNOT":
            # CNOT = R + MS + R... (Source 1/2 Native Set)
            # 今は仮に「MSゲート1回 + 前後の回転」としておく
            control, target = targets[0], targets[1]
            return [
                NativeInstruction(OpType.R, [control], duration=DURATION_R), # Pre-rotation
                NativeInstruction(OpType.R, [target], duration=DURATION_R),
                NativeInstruction(OpType.MS, [control, target], duration=DURATION_MS), # Core MS
                NativeInstruction(OpType.R, [control], duration=DURATION_R), # Post-rotation
            ]

        # --- 測定 ---
        elif name == "M":
            return [
                NativeInstruction(OpType.MEASURE, targets, duration=DURATION_MEASURE)
            ]
        
        # --- 初期化 (Reset) ---
        elif name == "R":
            # Source 1 Table 1: (t6) Qubit reset -> 50us
            return [
                NativeInstruction(OpType.RESET, targets, duration=DURATION_RESET)
            ]

        else:
            raise ValueError(f"Unknown instruction: {name}")