# main.py
from src.parser import Lark, StimTransformer, stim_grammar
from src.decomposer import GateDecomposer

def main():
    # 1. テスト回路の定義
    test_circuit = """
    # Test Circuit with Reset
    R 0
    H 0
    CNOT 0 1
    M 0
    """
    print("--- 1. Input Text ---")
    print(test_circuit)

    # 2. パース (Text -> QuantumInstruction)
    parser = Lark(stim_grammar, start='start')
    transformer = StimTransformer()
    
    tree = parser.parse(test_circuit)
    logical_ops = transformer.transform(tree)
    
    print("\n--- 2. Logical Instructions (Parser output) ---")
    for op in logical_ops:
        print(op)

    # 3. 分解 (QuantumInstruction -> NativeInstruction)
    decomposer = GateDecomposer()
    native_ops = []
    
    for op in logical_ops:
        # decomposeはリストを返すので extend で結合する
        native_ops.extend(decomposer.decompose(op))

    print("\n--- 3. Native Instructions (Decomposer output) ---")
    for i, op in enumerate(native_ops):
        print(f"{i:03}: {op}")

if __name__ == "__main__":
    main()