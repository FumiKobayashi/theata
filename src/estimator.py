# src/estimator.py
from .qccd_parser import Lark, QCCDTransformer, qccd_grammar
from .parameters import (
    DURATION_R, DURATION_MS, DURATION_MEASURE, DURATION_RESET,
    DURATION_SHUTTLE, DURATION_SPLIT, DURATION_MERGE, DURATION_SWAP
)

class QCCDEstimator:
    def __init__(self):
        self.parser = Lark(qccd_grammar, start='start')
        self.transformer = QCCDTransformer()

    def estimate(self, qccd_code: str) -> float:
        """
        QCCD Nativeコードを受け取り、総実行時間(us)を返す
        """
        # 1. パースして命令リストにする
        tree = self.parser.parse(qccd_code)
        instructions = self.transformer.transform(tree)
        
        total_time = 0.0
        
        # 2. 命令の種類を見て時間を足し合わせる
        for cmd in instructions:
            op_type = cmd["type"]
            
            if op_type == "R":
                total_time += DURATION_R
            elif op_type == "MS":
                total_time += DURATION_MS
            elif op_type == "SHUTTLE":
                # 移動距離に応じた計算を入れるならここでロジック分岐
                # Source 1 Table 1 では "Moving an ion... takes 5us" とあるので固定値でもOK
                total_time += DURATION_SHUTTLE
            elif op_type == "SPLIT":
                total_time += DURATION_SPLIT
            elif op_type == "MERGE":
                total_time += DURATION_MERGE
            elif op_type == "SWAP":
                total_time += DURATION_SWAP
            elif op_type == "MEASURE":
                total_time += DURATION_MEASURE
            elif op_type == "RESET":
                total_time += DURATION_RESET
            elif op_type == "BARRIER":
                pass # 時間はかからない
            
            # デバッグ用に経過を出しても良い
            # print(f"Op: {op_type}, Current Total: {total_time}us")

        return total_time

# --- 動作確認 ---
if __name__ == "__main__":
    # 手書きの "仮想コンパイル結果"
    mock_code = """
    RESET 0
    R 0 1.57 0.0
    R 1 3.14 0.0
    MS 0 1 1.57
    SHUTTLE 1 (10,1) -> (10,2)
    MEASURE 0
    """
    
    estimator = QCCDEstimator()
    result = estimator.estimate(mock_code)
    print(f"Total Execution Time: {result} us")