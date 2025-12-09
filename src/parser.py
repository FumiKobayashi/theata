from lark import Lark, Transformer
from dataclasses import dataclass
from typing import List, Optional, Union

# --- データ構造の定義 ---
# 文字列やリストで管理するより、クラスにしておくと
# エディタの補完が効いて開発が楽になります。
@dataclass
class QuantumInstruction:
    name: str                   # ゲート名 (H, CNOT, M 等)
    targets: List[int]          # ターゲットとなる量子ビットのインデックス
    params: Optional[List[float]] = None  # (将来用) 回転角など

# ---------------------------------------------------------
# 1. 文法定義 (Step 1と同じ)
# ---------------------------------------------------------
stim_grammar = """
    start: instruction+

    instruction: gate | measurement | reset | comment

    // エイリアス(-> name)を付けることで、Transformerのメソッド名と対応させます
    gate: "H" qubit_idx          -> h_gate
        | "X" qubit_idx          -> x_gate
        | "Y" qubit_idx          -> y_gate
        | "Z" qubit_idx          -> z_gate
        | "CNOT" qubit_idx qubit_idx -> cnot_gate
        | "CX" qubit_idx qubit_idx   -> cnot_gate

    measurement: "M" qubit_idx   -> measure_op

    // リセット命令を追加 (Stimでは 'R' が標準的)
    reset: "R" qubit_idx         -> reset_op

    qubit_idx: INT
    comment: /#.*/

    %import common.INT
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
"""

# ---------------------------------------------------------
# 2. Transformer (ツリーをPythonオブジェクトに変換する)
# ---------------------------------------------------------
class StimTransformer(Transformer):
    """
    Larkがパースした結果を、QuantumInstructionオブジェクトに変換するクラス。
    メソッド名は文法定義のエイリアス(-> name)と一致させます。
    """

    def qubit_idx(self, items):
        # items[0] はLarkのTokenオブジェクトなので、intに変換して返す
        return int(items[0])

    def h_gate(self, items):
        # items = [target_idx]
        return QuantumInstruction(name="H", targets=[items[0]])
    
    def x_gate(self, items):
        return QuantumInstruction(name="X", targets=[items[0]])

    def y_gate(self, items):
        return QuantumInstruction(name="Y", targets=[items[0]])

    def z_gate(self, items):
        return QuantumInstruction(name="Z", targets=[items[0]])

    def cnot_gate(self, items):
        # items = [control_idx, target_idx]
        return QuantumInstruction(name="CNOT", targets=[items[0], items[1]])

    def measure_op(self, items):
        return QuantumInstruction(name="M", targets=[items[0]])
    
    def reset_op(self, items):
        return QuantumInstruction(name="R", targets=[items[0]])

    def comment(self, items):
        # コメントは処理に不要なので None を返して無視する
        return None

    def instruction(self, items):
        # gate, measurement, comment のいずれかが items[0] に入ってくる
        return items[0]

    def start(self, items):
        # items は instruction のリスト
        # None (コメント行) を除外して、命令リストだけを返す
        return [item for item in items if item is not None]


# ---------------------------------------------------------
# 3. 動作確認
# ---------------------------------------------------------
if __name__ == "__main__":
    # パーサー作成
    parser = Lark(stim_grammar, start='start')
    
    # トランスフォーマー作成
    transformer = StimTransformer()

    test_circuit = """
    # Syndrome Extraction Example
    H 0
    CNOT 0 1
    M 0
    # End
    """

    print("--- Input Text ---")
    print(test_circuit)

    try:
        # 1. パースしてツリーを作る
        tree = parser.parse(test_circuit)
        
        # 2. ツリーを変換してPythonリストにする
        circuit_data = transformer.transform(tree)

        print("\n--- Converted Python Objects ---")
        for i, cmd in enumerate(circuit_data):
            print(f"{i}: {cmd}")

        print(f"\nTotal instructions: {len(circuit_data)}")
        print("このように、Stimのテキストがプログラムで扱えるリストになりました。")

    except Exception as e:
        print(f"Error: {e}")