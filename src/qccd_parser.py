from lark import Lark, Transformer
from dataclasses import dataclass
from typing import List, Optional

# --- データ構造 (NativeInstruction) は qccd_native.py からインポートしても良いですが
# ここでは簡易化のため、Transformer内で辞書またはデータクラスに変換する想定で記述します ---

# ---------------------------------------------------------
# QCCD Native EBNF Definition
# ---------------------------------------------------------
qccd_grammar = """
    start: instruction+

    instruction: gate | movement | measure | reset | barrier | comment

    // --- 1. 量子ゲート命令 (LIZ内) ---
    // 文法: 命令 ターゲットリスト [パラメータ] 時間
    // 例: MS 0 1 40.0
    // 例: R 0 [3.14] 5.0
    
    gate: ms_gate | r_gate

    ms_gate: "MS" targets duration      -> ms_op
    r_gate:  "R"  targets params duration -> r_op

    // --- 2. 移動命令 (Architecture) ---
    // 文法: 命令 ターゲットリスト [from -> to] 時間
    // 例: MOVE 0 1 -> 2 5.0  (イオン0をノード1から2へ移動、5us)
    
    movement: move_op | split_op | merge_op | swap_op

    move_op:  "MOVE"  targets location_change duration
    split_op: "SPLIT" targets location_change duration
    merge_op: "MERGE" targets location_change duration
    swap_op:  "SWAP"  targets duration               // 場所変わらず順序のみ

    // --- 3. その他の命令 ---
    measure: "MEASURE" targets duration -> measure_op
    reset:   "RESET"   targets duration -> reset_op
    barrier: "BARRIER" targets          -> barrier_op

    // --- 共通要素の定義 ---
    targets: INT+
    params:  "[" FLOAT ("," FLOAT)* "]"
    duration: FLOAT "us"  // 単位(us)を明示的につける設計

    location_change: INT "->" INT  // src_node -> dst_node

    comment: /#.*/

    %import common.INT
    %import common.FLOAT
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
"""

# ---------------------------------------------------------
# Transformer (Text -> Python Object)
# ---------------------------------------------------------
class QCCDTransformer(Transformer):
    """
    パースされた木構造を、処理しやすい辞書やオブジェクトに変換する
    """
    def INT(self, token):
        return int(token)
    
    def FLOAT(self, token):
        return float(token)

    def targets(self, items):
        return list(items)  # [0, 1] etc.

    def params(self, items):
        return list(items)  # [3.14, 1.57] etc.

    def duration(self, items):
        return items[0]  # 時間(float)だけ返す

    def location_change(self, items):
        return {"src": items[0], "dst": items[1]}

    # --- 命令ごとの処理 ---
    def ms_op(self, items):
        # items: [targets, duration]
        return {"type": "MS", "targets": items[0], "duration": items[1]}

    def r_op(self, items):
        # items: [targets, params, duration]
        return {"type": "R", "targets": items[0], "params": items[1], "duration": items[2]}

    def move_op(self, items):
        # items: [targets, {src, dst}, duration]
        return {"type": "MOVE", "targets": items[0], "loc": items[1], "duration": items[2]}

    # ... (他の命令も同様に実装) ...
    
    def instruction(self, items):
        return items[0]

    def start(self, items):
        return [item for item in items if item is not None]

# ---------------------------------------------------------
# 動作確認
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = Lark(qccd_grammar, start='start')
    transformer = QCCDTransformer()

    # 将来コンパイラが出力するであろうテキスト形式の想定
    sample_qccd_code = """
    # Initialization
    RESET 0 1 50.0us
    
    # Gate Execution
    R 0 [1.57] 5.0us
    R 1 [3.14] 5.0us
    MS 0 1 40.0us
    
    # Shuttling (Data Evacuation)
    MOVE 2 10 -> 11 5.0us
    
    # Final Measurement
    MEASURE 0 1 400.0us
    """

    print("--- QCCD Source Code ---")
    print(sample_qccd_code)

    try:
        tree = parser.parse(sample_qccd_code)
        # print(tree.pretty()) # デバッグ時はツリー表示も有効
        result = transformer.transform(tree)
        
        print("\n--- Parsed Objects ---")
        for cmd in result:
            print(cmd)
            
    except Exception as e:
        print(e)