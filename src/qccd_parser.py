from lark import Lark, Transformer
from ast_def import (
    Location, Operation, QubitDecl, GateOp, MovementOp, 
    MeasureOp, ResetOp, BarrierOp, Program
)

# ==========================================
# EBNF (文法定義)
# ==========================================
# 先生の要望通り、declaration (初期配置) を追加しました
qccd_grammar = """
    start: (declaration | instruction)+

    // --- 宣言部 (初期配置) ---
    declaration: "QUBIT" ion_id "AT" location

    // --- 命令部 ---
    instruction: gate_op | movement_op | measure_op | reset_op | barrier_op | comment

    // 基本型
    ion_id:    INT
    trap_id:   INT
    site_id:   INT
    angle:     FLOAT
    location: "(" trap_id "," site_id ")"

    // ゲート
    gate_op: ms_op | r_op
    ms_op: "MS" ion_id ion_id angle
    r_op:  "R"  ion_id angle angle

    // 移動
    movement_op: shuttle_op | split_op | merge_op | swap_op
    shuttle_op: "SHUTTLE" ion_id location "->" location
    split_op:   "SPLIT"   ion_id location
    merge_op:   "MERGE"   ion_id location
    swap_op:    "SWAP"    ion_id ion_id "AT" location

    // その他
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

# ==========================================
# 3. Transformer (Tree -> AST Objects)
# ==========================================
class QCCDASTTransformer(Transformer):
    """
    LarkのTreeを読み込み、上記で定義したデータクラスのインスタンスに変換する
    """
    def INT(self, token):
        return int(token)
    
    def FLOAT(self, token):
        return float(token)
    
    # 基本型の処理
    def ion_id(self, items): return items[0]
    def trap_id(self, items): return items[0]
    def site_id(self, items): return items[0]
    def angle(self, items): return items[0]
    
    def location(self, items):
        return Location(trap=items[0], site=items[1])

    # --- Declaration (初期配置) ---
    def declaration(self, items):
        return QubitDecl(id=items[0], location=items[1])

    # --- Gate Operations ---
    def ms_op(self, items):
        return GateOp(type="MS", targets=[items[0], items[1]], params=[items[2]])

    def r_op(self, items):
        return GateOp(type="R", targets=[items[0]], params=[items[1], items[2]])

    # --- Movement Operations ---
    def shuttle_op(self, items):
        return MovementOp(type="SHUTTLE", targets=[items[0]], src=items[1], dst=items[2])
    
    def split_op(self, items):
        # SPLITは場所を指定して実行(src/dstの概念は文脈によるが、ここではdst=locationとする例)
        return MovementOp(type="SPLIT", targets=[items[0]], dst=items[1])

    def merge_op(self, items):
        return MovementOp(type="MERGE", targets=[items[0]], dst=items[1])
    
    def swap_op(self, items):
        # SWAPは2つのイオン、1つの場所
        return MovementOp(type="SWAP", targets=[items[0], items[1]], dst=items[2])

    # --- Other Operations ---
    def measure_op(self, items):
        return MeasureOp(targets=list(items)) # items全体がion_idのリスト

    def reset_op(self, items):
        return ResetOp(targets=list(items))
    
    def barrier_op(self, items):
        return BarrierOp(targets=list(items))

    # --- Passthrough & Comment ---
    def gate_op(self, items): return items[0]
    def movement_op(self, items): return items[0]
    def instruction(self, items): return items[0]
    def comment(self, items): return None

    # --- Root (Start) ---
    def start(self, items):
        # None (コメント) を除去
        valid_items = [x for x in items if x is not None]
        
        # 宣言と命令を分別して Program オブジェクトに格納
        decls = [x for x in valid_items if isinstance(x, QubitDecl)]
        insts = [x for x in valid_items if isinstance(x, Operation)]
        
        return Program(declarations=decls, instructions=insts)


# ==========================================
# 動作確認
# ==========================================
if __name__ == "__main__":
    parser = Lark(qccd_grammar, start='start')
    transformer = QCCDASTTransformer()

    # サンプルコード（初期配置を追加）
    code = """
    # Initial Placement (Professor's Request)
    QUBIT 0 AT (1, 1)
    QUBIT 1 AT (1, 2)
    QUBIT 2 AT (1, 10)

    # Operations
    RESET 0 1
    R 0 1.57 0.0
    MS 0 1 0.0
    
    # Movement
    SHUTTLE 2 (1, 10) -> (2, 0)
    
    # Measurement
    MEASURE 0 1
    """

    print("--- Source Code ---")
    print(code)

    try:
        tree = parser.parse(code)
        program = transformer.transform(tree)
        
        print("\n--- AST Output (Program Object) ---")
        print(f"Declarations ({len(program.declarations)}):")
        for decl in program.declarations:
            print(f"  {decl}")
            
        print(f"\nInstructions ({len(program.instructions)}):")
        for inst in program.instructions:
            print(f"  {inst}")
            
            # クラスだからできること：プロパティアクセス
            if isinstance(inst, MovementOp):
                print(f"    -> Accessing property: Moving Ion {inst.targets[0]} to {inst.dst}")

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()