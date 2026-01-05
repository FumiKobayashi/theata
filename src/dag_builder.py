"""
DAG Builder: AST から Directed Acyclic Graph を構築

依存関係の種類:
1. データ依存 - 同じion_idに対する連続した操作
2. (将来) 位置依存 - 同じ物理位置を使う操作
3. (将来) リソース依存 - 同じハードウェアリソースを使う操作

現在の実装: データ依存のみを追跡

注意:
- 移動操作 (SHUTTLE, SWAP, SPLIT, MERGE) はDAGに含めない
  → 後段のスケジューリングで挿入
- MEASUREは別管理（アーキテクチャ依存のため）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from ast_def import (
    Program, Operation, QubitDecl, Location,
    GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp,
    MeasureOp, ResetOp, BarrierOp
)


# ==========================================
# DAG ノード
# ==========================================

@dataclass
class DAGNode:
    """DAGの各ノード（1つの操作に対応）"""
    id: int                          # ノードの一意識別子
    operation: Operation             # 対応するAST操作
    predecessors: Set[int] = field(default_factory=set)  # 依存先ノードID
    successors: Set[int] = field(default_factory=set)    # 後続ノードID
    
    def __repr__(self):
        return f"Node({self.id}, {self.operation.type}, pred={self.predecessors}, succ={self.successors})"


# ==========================================
# DAG
# ==========================================

@dataclass
class DAG:
    """Directed Acyclic Graph with Front Layer management"""
    nodes: Dict[int, DAGNode] = field(default_factory=dict)
    entry_nodes: Set[int] = field(default_factory=set)   # 依存なしノード（開始点）
    exit_nodes: Set[int] = field(default_factory=set)    # 後続なしノード（終了点）
    
    # Front Layer管理用
    _executed: Set[int] = field(default_factory=set)     # 実行済みノード
    _front_layer: Set[int] = field(default_factory=set)  # 現在のFront Layer
    
    def add_node(self, node: DAGNode):
        """ノードを追加"""
        self.nodes[node.id] = node
    
    def add_edge(self, from_id: int, to_id: int):
        """エッジを追加（from_id -> to_id の依存関係）"""
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].successors.add(to_id)
            self.nodes[to_id].predecessors.add(from_id)
    
    def compute_entry_exit(self):
        """entry_nodes と exit_nodes を計算"""
        self.entry_nodes = {
            nid for nid, node in self.nodes.items() 
            if len(node.predecessors) == 0
        }
        self.exit_nodes = {
            nid for nid, node in self.nodes.items() 
            if len(node.successors) == 0
        }
        # Front Layerを初期化
        self._front_layer = self.entry_nodes.copy()
        self._executed = set()
    
    # ==========================================
    # Front Layer Management
    # ==========================================
    
    def get_front_layer(self) -> Set[int]:
        """
        現在のFront Layer（実行可能なノードの集合）を取得
        
        Front Layer = 全ての先行ノードが実行済みのノード
        """
        return self._front_layer.copy()
    
    def execute(self, node_id: int) -> bool:
        """
        ノードを実行済みとしてマークし、Front Layerを更新
        
        Returns:
            True if successful, False if node not in front layer
        """
        if node_id not in self._front_layer:
            return False
        
        # Front Layerから削除
        self._front_layer.remove(node_id)
        self._executed.add(node_id)
        
        # 後続ノードをチェック：全先行が実行済みならFront Layerに追加
        node = self.nodes[node_id]
        for succ_id in node.successors:
            succ_node = self.nodes[succ_id]
            # 全ての先行ノードが実行済みか確認
            if succ_node.predecessors.issubset(self._executed):
                self._front_layer.add(succ_id)
        
        return True
    
    def is_complete(self) -> bool:
        """全ノードが実行済みか"""
        return len(self._executed) == len(self.nodes)
    
    def get_executed(self) -> Set[int]:
        """実行済みノードの集合"""
        return self._executed.copy()
    
    def get_remaining(self) -> Set[int]:
        """未実行ノードの集合"""
        return set(self.nodes.keys()) - self._executed
    
    def reset_execution(self):
        """実行状態をリセット"""
        self._executed = set()
        self._front_layer = self.entry_nodes.copy()
    
    # ==========================================
    # Analysis Methods
    # ==========================================
    
    def topological_sort(self) -> List[int]:
        """トポロジカルソート（Kahnのアルゴリズム）"""
        in_degree = {nid: len(node.predecessors) for nid, node in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # 同じin_degreeのノードはID順で処理（決定的な順序のため）
            queue.sort()
            nid = queue.pop(0)
            result.append(nid)
            
            for succ_id in self.nodes[nid].successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
        
        if len(result) != len(self.nodes):
            raise RuntimeError("DAG contains a cycle (should not happen)")
        
        return result
    
    def get_parallel_groups(self) -> List[List[int]]:
        """
        並列実行可能なグループを取得（静的解析用）
        各グループ内のノードは互いに依存がない
        
        注意: これは静的解析用。動的なスケジューリングには
        get_front_layer() + execute() を使用すること
        """
        in_degree = {nid: len(node.predecessors) for nid, node in self.nodes.items()}
        remaining = set(self.nodes.keys())
        groups = []
        
        while remaining:
            # 現在のin_degreeが0のノードを1グループに
            group = [nid for nid in remaining if in_degree[nid] == 0]
            if not group:
                raise RuntimeError("DAG contains a cycle")
            
            groups.append(sorted(group))
            
            # 処理済みとしてin_degreeを更新
            for nid in group:
                remaining.remove(nid)
                for succ_id in self.nodes[nid].successors:
                    in_degree[succ_id] -= 1
        
        return groups


# ==========================================
# DAG Builder
# ==========================================

class DAGBuilder:
    """ProgramからDAGを構築
    
    注意: 
    - 移動操作 (SHUTTLE, SWAP, SPLIT, MERGE) はDAGに含めない
    - MEASUREは別管理（アーキテクチャ依存のため）
    """
    
    def __init__(self, program: Program):
        self.program = program
        self.dag = DAG()
        self.node_counter = 0
        
        # 各ion_idの最後の操作を追跡
        self.last_op_for_ion: Dict[int, int] = {}
        
        # 除外された操作を保存（後段で使用可能）
        self.excluded_movements: List[Operation] = []
        self.measurements: List[MeasureOp] = []
    
    def _is_dag_operation(self, op: Operation) -> bool:
        """DAGに含める操作かどうかを判定"""
        # 移動操作は除外
        if isinstance(op, (ShuttleOp, SwapOp, SplitOp, MergeOp)):
            return False
        # MEASURE, RESETも通常のゲートとしてDAGに含める
        # TODO: バッチ処理最適化時には、複数RESET/MEASUREをまとめる処理を検討
        return True
    
    def _get_affected_ions(self, op: Operation) -> List[int]:
        """操作が影響するion_idのリストを取得"""
        if isinstance(op, GateOp):
            return op.targets
        elif isinstance(op, ResetOp):
            return op.targets
        elif isinstance(op, MeasureOp):
            return op.targets
        elif isinstance(op, BarrierOp):
            return op.targets
        else:
            return []
    
    def _create_node(self, op: Operation) -> DAGNode:
        """操作からDAGノードを作成"""
        node = DAGNode(id=self.node_counter, operation=op)
        self.node_counter += 1
        return node
    
    def build(self) -> DAG:
        """DAGを構築"""
        for op in self.program.instructions:
            # 移動操作は除外して別保存
            if isinstance(op, (ShuttleOp, SwapOp, SplitOp, MergeOp)):
                self.excluded_movements.append(op)
                continue
            
            node = self._create_node(op)
            self.dag.add_node(node)
            
            # データ依存関係を設定
            affected_ions = self._get_affected_ions(op)
            predecessor_ids = set()
            
            for ion_id in affected_ions:
                if ion_id in self.last_op_for_ion:
                    predecessor_ids.add(self.last_op_for_ion[ion_id])
            
            # エッジを追加
            for pred_id in predecessor_ids:
                self.dag.add_edge(pred_id, node.id)
            
            # 最後の操作を更新
            for ion_id in affected_ions:
                self.last_op_for_ion[ion_id] = node.id
        
        self.dag.compute_entry_exit()
        return self.dag


def build_dag(program: Program) -> DAG:
    """便利関数: ProgramからDAGを構築"""
    builder = DAGBuilder(program)
    return builder.build()


# ==========================================
# 可視化 (デバッグ用)
# ==========================================

def dag_to_dot(dag: DAG) -> str:
    """DAGをGraphviz DOT形式に変換（シンプル版）"""
    lines = ["digraph DAG {", "  rankdir=TB;"]
    
    for nid, node in dag.nodes.items():
        label = f"{nid}: {node.operation.type}"
        if hasattr(node.operation, 'targets'):
            targets = getattr(node.operation, 'targets', None)
            if targets:
                label += f"\\n{targets}"
        elif hasattr(node.operation, 'target'):
            label += f"\\n[{node.operation.target}]"
        
        lines.append(f'  {nid} [label="{label}"];')
    
    for nid, node in dag.nodes.items():
        for succ_id in node.successors:
            lines.append(f"  {nid} -> {succ_id};")
    
    lines.append("}")
    return "\n".join(lines)


def dag_to_dot_by_qubit(dag: DAG, num_qubits: int) -> str:
    """
    論文スタイルのDAG可視化（qubitごとに縦列配置）
    
    IonShuttler論文のFigure 4bのようなレイアウト:
    - 各qubitが縦の列を形成
    - 同じレベルの操作は横に並ぶ
    - 2-qubit gateは両方のqubitにまたがる
    """
    lines = [
        "digraph DAG {",
        "  rankdir=TB;",
        "  splines=ortho;",  # 直角の線
        "  nodesep=0.5;",
        "  ranksep=0.6;",
        "",
        "  // Qubit labels",
    ]
    
    # Qubitラベルを追加
    for q in range(num_qubits):
        lines.append(f'  q{q}_label [label="|q{q}⟩" shape=plaintext fontsize=14];')
    
    # Qubitラベルを同じrankに
    lines.append(f'  {{ rank=same; {" ".join(f"q{q}_label" for q in range(num_qubits))} }}')
    lines.append("")
    
    # ノードをレベル（並列グループ）ごとに配置
    groups = dag.get_parallel_groups()
    
    # 各ノードがどのqubitに属するかを決定
    node_qubits: Dict[int, List[int]] = {}
    for nid, node in dag.nodes.items():
        if hasattr(node.operation, 'targets'):
            node_qubits[nid] = node.operation.targets
        else:
            node_qubits[nid] = []
    
    lines.append("  // Operation nodes")
    
    # ゲートタイプに応じた色
    color_map = {
        "R": "#AED6F1",      # 薄い青
        "MS": "#ABEBC6",     # 薄い緑
        "RESET": "#F9E79F",  # 薄い黄
    }
    
    for nid, node in dag.nodes.items():
        op_type = node.operation.type
        color = color_map.get(op_type, "#D5D8DC")
        qubits = node_qubits.get(nid, [])
        
        if len(qubits) == 1:
            label = f"{nid}: {op_type}"
        elif len(qubits) == 2:
            label = f"{nid}: {op_type}"
        else:
            label = f"{nid}: {op_type}\\n{qubits}"
        
        lines.append(f'  {nid} [label="{label}" shape=box style=filled fillcolor="{color}"];')
    
    lines.append("")
    lines.append("  // Rank constraints (parallel groups)")
    
    # 各レベルを同じrankに
    for level, group in enumerate(groups):
        if group:
            nodes_str = " ".join(str(nid) for nid in group)
            lines.append(f'  {{ rank=same; {nodes_str} }}')
    
    lines.append("")
    lines.append("  // Edges (dependencies)")
    
    for nid, node in dag.nodes.items():
        for succ_id in sorted(node.successors):
            lines.append(f"  {nid} -> {succ_id};")
    
    # Qubitラベルから最初のノードへの不可視エッジ（整列用）
    lines.append("")
    lines.append("  // Invisible edges for alignment")
    
    if groups and groups[0]:
        first_level_nodes = groups[0]
        for nid in first_level_nodes:
            qubits = node_qubits.get(nid, [])
            if qubits:
                q = qubits[0]
                lines.append(f'  q{q}_label -> {nid} [style=invis];')
    
    lines.append("}")
    return "\n".join(lines)


# ==========================================
# 動作確認
# ==========================================

if __name__ == "__main__":
    from qccd_parser import parse
    
    # シンプルな回路で Front Layer の動作を確認
    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 2 AT (1, 2)

    # 独立したRESET (並列可能)
    RESET 0
    RESET 1
    RESET 2
    
    # 各qubitへのR gate
    R 0 1.57 0.0
    R 1 1.57 0.0
    R 2 1.57 0.0
    
    # 2-qubit gates
    MS 0 1 0.25
    MS 1 2 0.25
    """
    
    print("=" * 60)
    print("DAG Front Layer Test")
    print("=" * 60)
    
    program = parse(code)
    builder = DAGBuilder(program)
    dag = builder.build()
    
    print("\n--- DAG Structure ---")
    for nid in sorted(dag.nodes.keys()):
        node = dag.nodes[nid]
        op = node.operation
        targets = op.targets if hasattr(op, 'targets') else []
        print(f"  {nid}: {op.type} {targets}  pred={node.predecessors} succ={node.successors}")
    
    print("\n--- Front Layer Simulation ---")
    print("(Simulating scheduler selecting gates one by one)\n")
    
    step = 0
    while not dag.is_complete():
        front = dag.get_front_layer()
        print(f"Step {step}: Front Layer = {front}")
        
        # Front Layerの各ノードの詳細
        for nid in sorted(front):
            node = dag.nodes[nid]
            op = node.operation
            targets = op.targets if hasattr(op, 'targets') else []
            print(f"         -> {nid}: {op.type} on {targets}")
        
        # ここでSchedulerが最適なゲートを選ぶ
        # (今は単純に最小IDを選択)
        selected = min(front)
        print(f"         Selected: {selected}")
        
        dag.execute(selected)
        step += 1
        print()
    
    print(f"Execution complete! Total steps: {step}")
    print(f"Executed nodes: {sorted(dag.get_executed())}")
    
    # リセットして静的解析と比較
    print("\n--- Comparison with Static Parallel Groups ---")
    dag.reset_execution()
    groups = dag.get_parallel_groups()
    for i, group in enumerate(groups):
        ops = [f"{nid}:{dag.nodes[nid].operation.type}" for nid in group]
        print(f"  Level {i}: {ops}")