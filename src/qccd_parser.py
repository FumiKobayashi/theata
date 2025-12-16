from lark import Lark, Transformer
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------
# QCCD Native EBNF Definition
# ---------------------------------------------------------
qccd_grammar = """
    start: instruction+

    instruction: gate_op | movement_op | measure_op | reset_op | barrier_op | comment

    // --- 基本型の定義 ---
    ion_id:    INT
    trap_id:   INT
    site_id:   INT
    angle:     FLOAT
    
    // 先生の提案: (TrapId, SiteId)
    location: "(" trap_id "," site_id ")"

    // --- 1. 量子ゲート命令 ---
    gate_op: ms_op | r_op

    ms_op: "MS" ion_id ion_id angle
    r_op:  "R"  ion_id angle angle

    // --- 2. 移動命令 ---
    movement_op: shuttle_op | split_op | merge_op | swap_op

    shuttle_op: "SHUTTLE" ion_id location "->" location
    split_op: "SPLIT" ion_id location
    merge_op: "MERGE" ion_id location
    swap_op:  "SWAP"  ion_id ion_id "AT" location

    // --- 3. その他の命令 ---
    measure_op: "MEASURE" ion_id+
    reset_op:   "RESET"   ion_id+
    barrier_op: "BARRIER" ion_id+

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
    パースされた木構造を、処理しやすい辞書に変換する
    """
    def INT(self, token):
        return int(token)
    
    def FLOAT(self, token):
        return float(token)
    
    # --- 基本型をそのまま値として返す ---
    def ion_id(self, items):
        return items[0]
    
    def trap_id(self, items):
        return items[0]
    
    def site_id(self, items):
        return items[0]
    
    def angle(self, items):
        return items[0]
    
    def location(self, items):
        return {"trap": items[0], "site": items[1]}
    
    # --- 中間ノードを透過させる ---
    def gate_op(self, items):
        return items[0]
    
    def movement_op(self, items):
        return items[0]
    
    # --- コメントはNoneを返す ---
    def comment(self, items):
        return None

    # --- 命令ごとの処理 ---
    def shuttle_op(self, items):
        return {
            "type": "SHUTTLE",
            "targets": [items[0]],
            "src": items[1],
            "dst": items[2]
        }

    def ms_op(self, items):
        return {
            "type": "MS",
            "targets": [items[0], items[1]],
            "params": [items[2]]
        }

    def r_op(self, items):
        return {
            "type": "R",
            "targets": [items[0]],
            "angles": [items[1], items[2]]
        }
    
    def split_op(self, items):
        return {
            "type": "SPLIT",
            "targets": [items[0]],
            "location": items[1]
        }
    
    def merge_op(self, items):
        return {
            "type": "MERGE",
            "targets": [items[0]],
            "location": items[1]
        }
    
    def swap_op(self, items):
        return {
            "type": "SWAP",
            "targets": [items[0], items[1]],
            "location": items[2]
        }
    
    def measure_op(self, items):
        return {
            "type": "MEASURE",
            "targets": list(items)
        }
    
    def reset_op(self, items):
        return {
            "type": "RESET",
            "targets": list(items)
        }
    
    def barrier_op(self, items):
        return {
            "type": "BARRIER",
            "targets": list(items)
        }
    
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

    sample_qccd_code = """
    # Initialization
    RESET 0 1
    
    # Gate Execution
    R 0 1.57 0.0
    R 1 3.14 1.57
    MS 0 1 0.0
    
    # Shuttling (Data Evacuation)
    SHUTTLE 2 (1, 10) -> (2, 0) 
    
    # Final Measurement
    MEASURE 0 1 
    """

    print("--- QCCD Source Code ---")
    print(sample_qccd_code)

    try:
        tree = parser.parse(sample_qccd_code)
        result = transformer.transform(tree)
        
        print("\n--- Parsed Objects ---")
        for cmd in result:
            print(cmd)
            
    except Exception as e:
        print(e)