"""
QCCD Native Gate Set -> DAG 変換モジュール

目的:
    パースされたQCCD命令列を有向非巡回グラフ(DAG)に変換する。
    これにより:
    - 命令間の依存関係が明示化される
    - 並列実行可能な命令群が識別できる
    - 後段のスケジューリング最適化の基盤となる

依存関係の定義:
    命令Aと命令Bが同じイオンを操作する場合、
    先に現れたA → 後に現れたB という依存エッジを張る
"""

import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum, auto


# ---------------------------------------------------------
# データ構造
# ---------------------------------------------------------
class InstructionType(Enum):
    """命令の種類"""
    # ゲート操作
    MS = auto()      # Mølmer-Sørensen gate
    R = auto()       # Single qubit rotation
    
    # 移動操作
    SHUTTLE = auto()
    SPLIT = auto()
    MERGE = auto()
    SWAP = auto()
    
    # その他
    MEASURE = auto()
    RESET = auto()
    BARRIER = auto()


@dataclass
class DAGNode:
    """
    DAGの各ノードを表すデータクラス
    
    Attributes:
        node_id: ノードの一意識別子（命令の出現順）
        inst_type: 命令の種類
        targets: 操作対象のイオンIDリスト
        params: 命令固有のパラメータ（角度など）
        original: パーサーから受け取った元の辞書（デバッグ用）
    """
    node_id: int
    inst_type: InstructionType
    targets: List[int]
    params: Dict[str, Any] = field(default_factory=dict)
    original: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, DAGNode):
            return self.node_id == other.node_id
        return False


# ---------------------------------------------------------
# DAG Builder
# ---------------------------------------------------------
class QCCDDAGBuilder:
    """
    QCCD命令列からDAGを構築するクラス
    
    使用例:
        builder = QCCDDAGBuilder()
        dag = builder.build(parsed_instructions)
        
        # 可視化
        builder.visualize(dag, "circuit_dag.png")
        
        # トポロジカルソート（実行順序）
        order = list(nx.topological_sort(dag))
    """
    
    def __init__(self):
        self.dag: nx.DiGraph = nx.DiGraph()
        self.nodes: List[DAGNode] = []
        # 各イオンについて、最後に操作した命令のnode_idを追跡
        self.last_op_on_ion: Dict[int, int] = {}
        
    def _parse_instruction_type(self, type_str: str) -> InstructionType:
        """文字列からInstructionTypeへ変換"""
        mapping = {
            "MS": InstructionType.MS,
            "R": InstructionType.R,
            "SHUTTLE": InstructionType.SHUTTLE,
            "SPLIT": InstructionType.SPLIT,
            "MERGE": InstructionType.MERGE,
            "SWAP": InstructionType.SWAP,
            "MEASURE": InstructionType.MEASURE,
            "RESET": InstructionType.RESET,
            "BARRIER": InstructionType.BARRIER,
        }
        return mapping.get(type_str, InstructionType.R)
    
    def _extract_params(self, inst: Dict) -> Dict[str, Any]:
        """命令辞書からパラメータを抽出"""
        params = {}
        if "params" in inst:
            params["params"] = inst["params"]
        if "angles" in inst:
            params["angles"] = inst["angles"]
        if "src" in inst:
            params["src"] = inst["src"]
        if "dst" in inst:
            params["dst"] = inst["dst"]
        return params
    
    def build(self, instructions: List[Dict]) -> nx.DiGraph:
        """
        命令リストからDAGを構築する
        
        Args:
            instructions: QCCDTransformerが出力した命令辞書のリスト
            
        Returns:
            nx.DiGraph: 構築されたDAG
        """
        self.dag = nx.DiGraph()
        self.nodes = []
        self.last_op_on_ion = {}
        
        for idx, inst in enumerate(instructions):
            # DAGNodeを作成
            node = DAGNode(
                node_id=idx,
                inst_type=self._parse_instruction_type(inst["type"]),
                targets=inst.get("targets", []),
                params=self._extract_params(inst),
                original=inst
            )
            self.nodes.append(node)
            
            # ノードをDAGに追加（属性としてDAGNodeを保持）
            self.dag.add_node(idx, data=node)
            
            # 依存関係を追加
            # このノードが操作する各イオンについて、
            # 直前にそのイオンを操作したノードからエッジを張る
            predecessors: Set[int] = set()
            for ion_id in node.targets:
                if ion_id in self.last_op_on_ion:
                    predecessors.add(self.last_op_on_ion[ion_id])
                # このイオンの最後の操作を更新
                self.last_op_on_ion[ion_id] = idx
            
            # エッジを追加
            for pred_id in predecessors:
                self.dag.add_edge(pred_id, idx)
        
        return self.dag
    
    def get_node(self, node_id: int) -> Optional[DAGNode]:
        """ノードIDからDAGNodeを取得"""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None
    
    def get_parallel_groups(self) -> List[List[int]]:
        """
        並列実行可能な命令グループを取得
        
        Returns:
            各グループは同時実行可能な命令のnode_idリスト
        """
        # トポロジカルソートを使って層（レイヤー）ごとにグループ化
        # 同じ層の命令は並列実行可能
        
        if not self.dag.nodes():
            return []
        
        # 各ノードの「深さ」を計算
        depth: Dict[int, int] = {}
        
        for node_id in nx.topological_sort(self.dag):
            predecessors = list(self.dag.predecessors(node_id))
            if not predecessors:
                depth[node_id] = 0
            else:
                depth[node_id] = max(depth[p] for p in predecessors) + 1
        
        # 深さでグループ化
        max_depth = max(depth.values()) if depth else 0
        groups = [[] for _ in range(max_depth + 1)]
        for node_id, d in depth.items():
            groups[d].append(node_id)
        
        return groups
    
    def visualize(self, filename: str = "qccd_dag.png"):
        """
        DAGを画像として保存
        
        Args:
            filename: 出力ファイル名
        """
        try:
            import matplotlib.pyplot as plt
            
            # レイアウト計算
            pos = nx.spring_layout(self.dag, seed=42)
            
            # ノードラベルを作成
            labels = {}
            for node_id in self.dag.nodes():
                node = self.get_node(node_id)
                if node:
                    labels[node_id] = f"{node_id}: {node.inst_type.name}\n{node.targets}"
            
            # 描画
            plt.figure(figsize=(12, 8))
            nx.draw(
                self.dag,
                pos,
                labels=labels,
                node_color='lightblue',
                node_size=2000,
                font_size=8,
                arrows=True,
                arrowsize=20,
                edge_color='gray'
            )
            plt.title("QCCD Instruction DAG")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"DAG visualization saved to {filename}")
        except ImportError:
            print("matplotlib not available for visualization")
    
    def to_dot(self) -> str:
        """
        DOT形式の文字列を生成（Graphvizで可視化可能）
        """
        lines = ["digraph QCCD_DAG {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")
        
        for node_id in self.dag.nodes():
            node = self.get_node(node_id)
            if node:
                label = f"{node.inst_type.name}\\n{node.targets}"
                lines.append(f'    {node_id} [label="{label}"];')
        
        for src, dst in self.dag.edges():
            lines.append(f"    {src} -> {dst};")
        
        lines.append("}")
        return "\n".join(lines)
    
    def get_front_layer(self) -> List[int]:
        """
        Front Layerを取得する（MQT Ion-Shuttler論文のSection IV-B）
        
        Front Layer = 先行依存がないノード群
        これらのゲートは互いに可換であり、どれを次に実行しても良い
        
        Returns:
            先行依存のないノードIDのリスト
        """
        front_layer = []
        for node_id in self.dag.nodes():
            # 入次数が0 = 先行依存なし
            if self.dag.in_degree(node_id) == 0:
                front_layer.append(node_id)
        return front_layer
    
    def select_best_gate(self, ion_positions: Optional[Dict[int, int]] = None) -> Optional[int]:
        """
        Front Layerから最適なゲートを選択する（MQT Ion-Shuttler論文のSection IV-C）
        
        選択基準: Processing Zoneへの距離が最小のイオンを使うゲート
        
        Args:
            ion_positions: イオンID → Processing Zoneへの距離 のマッピング
                          Noneの場合はFront Layerの先頭を返す
        
        Returns:
            選択されたノードID、Front Layerが空ならNone
        """
        front_layer = self.get_front_layer()
        if not front_layer:
            return None
        
        if ion_positions is None:
            # 位置情報がない場合は単純に先頭を返す
            return front_layer[0]
        
        # 各ゲートの「コスト」= 対象イオンのProcessing Zoneへの距離の合計
        best_node = None
        best_cost = float('inf')
        
        for node_id in front_layer:
            node = self.get_node(node_id)
            if node is None:
                continue
            
            # ゲートが操作するイオンの距離の合計
            cost = sum(ion_positions.get(ion_id, 0) for ion_id in node.targets)
            
            if cost < best_cost:
                best_cost = cost
                best_node = node_id
        
        return best_node
    
    def execute_gate(self, node_id: int) -> Optional[DAGNode]:
        """
        ゲートを実行済みとしてDAGから削除する
        
        Args:
            node_id: 実行するノードID
            
        Returns:
            削除されたDAGNode、存在しなければNone
        """
        if node_id not in self.dag:
            return None
        
        node = self.get_node(node_id)
        self.dag.remove_node(node_id)
        return node
    
    def is_complete(self) -> bool:
        """全てのゲートが実行されたか"""
        return self.dag.number_of_nodes() == 0
    
    def get_executable_gates_for_ion(self, ion_id: int) -> List[int]:
        """
        特定のイオンに対して実行可能なゲートを取得
        
        Args:
            ion_id: イオンID
            
        Returns:
            そのイオンを対象とするFront Layer内のノードIDリスト
        """
        front_layer = self.get_front_layer()
        result = []
        for node_id in front_layer:
            node = self.get_node(node_id)
            if node and ion_id in node.targets:
                result.append(node_id)
        return result
    
    def summary(self) -> str:
        """DAGの概要を文字列で返す"""
        lines = [
            "=== DAG Summary ===",
            f"Total nodes: {self.dag.number_of_nodes()}",
            f"Total edges: {self.dag.number_of_edges()}",
            f"Is DAG: {nx.is_directed_acyclic_graph(self.dag)}",
        ]
        
        # Front Layer情報
        front_layer = self.get_front_layer()
        lines.append(f"\n--- Front Layer (executable now) ---")
        for nid in front_layer:
            node = self.get_node(nid)
            if node:
                lines.append(f"  Node {nid}: {node.inst_type.name} on ions {node.targets}")
        
        # 並列グループの情報
        groups = self.get_parallel_groups()
        lines.append(f"\n--- All Parallel Layers ({len(groups)} total) ---")
        for i, group in enumerate(groups):
            node_strs = []
            for nid in group:
                node = self.get_node(nid)
                if node:
                    node_strs.append(f"{node.inst_type.name}({node.targets})")
            lines.append(f"  Layer {i}: {', '.join(node_strs)}")
        
        return "\n".join(lines)


# ---------------------------------------------------------
# 統合: Parser + DAG Builder
# ---------------------------------------------------------
def parse_and_build_dag(qccd_code: str) -> nx.DiGraph:
    """
    QCCDコード文字列をパースしてDAGを構築する便利関数
    
    Args:
        qccd_code: QCCD native gate setの文字列
        
    Returns:
        nx.DiGraph: 構築されたDAG
    """
    from lark import Lark
    
    # qccd_parser.py からインポートする想定だが、
    # ここでは自己完結のため文法とTransformerを再定義
    from qccd_parser import qccd_grammar, QCCDTransformer
    
    parser = Lark(qccd_grammar, start='start')
    transformer = QCCDTransformer()
    
    tree = parser.parse(qccd_code)
    instructions = transformer.transform(tree)
    
    builder = QCCDDAGBuilder()
    dag = builder.build(instructions)
    
    return dag, builder


# ---------------------------------------------------------
# 動作確認
# ---------------------------------------------------------
if __name__ == "__main__":
    # qccd_parser.py が同じディレクトリにある想定
    from qccd_parser import qccd_grammar, QCCDTransformer
    from lark import Lark
    
    # サンプルコード
    sample_code = """
    # Initialization
    RESET 0 1
    
    # Parallel single-qubit gates (no dependency between ion 0 and 1)
    R 0 1.57 0.0
    R 1 3.14 1.57
    
    # Two-qubit gate (depends on both R gates above)
    MS 0 1 0.0
    
    # Another gate on ion 0 only
    R 0 0.78 0.0
    
    # Movement
    SHUTTLE 2 (1, 10) -> (2, 0)
    
    # Final measurement
    MEASURE 0 1
    """
    
    print("=== Input QCCD Code ===")
    print(sample_code)
    
    # パース
    parser = Lark(qccd_grammar, start='start')
    transformer = QCCDTransformer()
    tree = parser.parse(sample_code)
    instructions = transformer.transform(tree)
    
    print("\n=== Parsed Instructions ===")
    for i, inst in enumerate(instructions):
        print(f"{i}: {inst}")
    
    # DAG構築
    builder = QCCDDAGBuilder()
    dag = builder.build(instructions)
    
    print("\n" + builder.summary())
    
    # DOT形式出力（Graphvizで可視化可能）
    print("\n=== DOT Format ===")
    print(builder.to_dot())
    
    # ========================================
    # Gate Selection シミュレーション
    # (MQT Ion-Shuttler 論文 Section IV の実装)
    # ========================================
    print("\n" + "="*50)
    print("=== Gate Selection Simulation ===")
    print("="*50)
    
    # イオンの位置情報（Processing Zoneへの距離）
    # 実際のシステムでは、メモリゾーンでの位置から計算
    ion_positions = {
        0: 3,  # ion 0 は距離3
        1: 1,  # ion 1 は距離1（近い）
        2: 5,  # ion 2 は距離5（遠い）
    }
    print(f"\nIon positions (distance to Processing Zone): {ion_positions}")
    
    # DAGを再構築（シミュレーション用）
    builder2 = QCCDDAGBuilder()
    builder2.build(instructions)
    
    step = 0
    while not builder2.is_complete():
        step += 1
        print(f"\n--- Step {step} ---")
        
        # Front Layerを表示
        front_layer = builder2.get_front_layer()
        print(f"Front Layer: {front_layer}")
        for nid in front_layer:
            node = builder2.get_node(nid)
            if node:
                cost = sum(ion_positions.get(ion, 0) for ion in node.targets)
                print(f"  Node {nid}: {node.inst_type.name} on {node.targets}, cost={cost}")
        
        # 最適なゲートを選択
        best = builder2.select_best_gate(ion_positions)
        if best is not None:
            node = builder2.get_node(best)
            print(f"Selected: Node {best} ({node.inst_type.name} on {node.targets})")
            
            # ゲートを実行（DAGから削除）
            builder2.execute_gate(best)
        else:
            break
    
    print(f"\n=== All gates executed in {step} steps ===")