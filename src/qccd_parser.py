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

    instruction: gate_op | movement_op | measure_op | reset_op | barrier_op | comment

    // --- 基本型の定義 ---
    ion_id:    INT
    trap_id:   INT
    site_id:   INT
    angle:     FLOAT
    
    // 先生の提案: (TrapId, SiteId)
    // 例: (10, 2) -> トラップ10番の、2番目のサイト
    location: "(" trap_id "," site_id ")"

    // --- 1. 量子ゲート命令 ---
    gate_op: ms_op | r_op

    // ゲート操作も「場所」を明示したほうが親切かもしれません
    // (省略して ion_id だけでも良いですが、統一感のため入れるのもアリです)
    ms_op: "MS" ion_id ion_id angle
    r_op:  "R"  ion_id angle angle

    // --- 2. 移動命令 (ここが変更点) ---
    movement_op: shuttle_op | split_op | merge_op | swap_op

    // 先生の定義: SHUTTLE (Ion, Src, Dst)
    // EBNF風記述: SHUTTLE ion_id location "->" location
    // 例: SHUTTLE 5 (10, 1) -> (10, 2)
    shuttle_op: "SHUTTLE" ion_id location "->" location

    // その他の操作もLocationを使う形に更新
    split_op: "SPLIT" ion_id location
    merge_op: "MERGE" ion_id location
    swap_op:  "SWAP"  ion_id ion_id "AT" location

    // --- 3. その他の命令 ---
    // measure, resetはLIZ上全てのイオンに行うから1個以上のid入力を許す設計 
    // BARRIERに関してはAIの出力意図を理解できていない
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
    パースされた木構造を、処理しやすい辞書やオブジェクトに変換する
    """
    def INT(self, token):
        return int(token)
    
    def FLOAT(self, token):
        return float(token)
    
    def location(self, items):
        # items: [trap_id, site_id] -> タプルや辞書に変換
        return {"trap": items[0], "site": items[1]}

    # --- 命令ごとの処理 ---
    def shuttle_op(self, items):
        # items: [ion_id, src_loc, dst_loc]
        return {
            "type": "SHUTTLE",
            "targets": [items[0]],
            "src": items[1], # {"trap": 10, "site": 1}
            "dst": items[2]
        }

    def ms_op(self, items):
        # EBNF: "MS" ion_id ion_id angle
        # items: [ion1, ion2, angle]
        return {
            "type": "MS",
            "targets": [items[0], items[1]],
            "params": [items[2]]  # angleをparamsリストに入れる設計の場合
        }

    def r_op(self, items):
        # EBNF: "R" ion_id angle angle
        # items: [ion, theta, phi]
        return {
            "type": "R",
            "targets": [items[0]],
            "angles": [items[1], items[2]]
        }
    
    # TODO: ... measure, resetも同様に実装 ...
    def measure_op(self, items):
        # EBNF: "MEASURE" ion_id+
        # items: [ion1, ion2, ...]
        return{
            "type": "MEASURE",
            "targets": items
        }
    
    def reset_op(self, items):
        # EBNF: "RESET" ion_id+
        # items: [ion1, ion2, ...]
        return{
            "type": "RESET",
            "targets": items
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

    # 将来コンパイラが出力するであろうテキスト形式の想定
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
        # print(tree.pretty()) # デバッグ時はツリー表示も有効
        result = transformer.transform(tree)
        
        print("\n--- Parsed Objects ---")
        for cmd in result:
            print(cmd)
            
    except Exception as e:
        print(e)