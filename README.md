# Transport-Aware Syndrome Measurement Circuit Compiler for 1D QCCD Architecture

1次元 QCCD アーキテクチャにおけるシンドローム測定回路のコンパイラ・シミュレータ。

## プロジェクト構成

```
src/                  コンパイラ本体
circuits/             入力回路定義
experiments/          実験スクリプト
results/              実験結果
plots/                グラフ出力
ThesisImages/         論文・ポスター素材
```

## コンパイラの使い方

### 基本的な使用フロー

```python
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator
from validator import validate_operations

# 1. 回路の読み込みとDAG構築
program = parse(circuit_module.CIRCUIT)
dag = DAGBuilder(program, stabilizer_parallel=False).build()

# 2. アーキテクチャと初期配置
arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=20, temp_capacity=20)
initial = {0: 1, 1: 1, 2: 5, ...}  # ion_id -> zone_id

# 3. コンパイル
config = CompilerConfig(eviction_strategy="balanced")
compiler = QCCDCompiler(arch, config=config)
result = compiler.compile(dag, initial)

# 4. 結果の利用（後述）
```

### CompileResult の構造

`compiler.compile()` が返す `CompileResult` は以下のデータを持つ:

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `timesteps` | `List[TimeStep]` | 並列化済みの命令列（Scheduler出力） |
| `flat_ops` | `List[Operation]` | 並列化前の逐次命令列（MovementPlanner出力） |
| `operations` | `List[Operation]` | timestepsを平坦化した命令列（後方互換プロパティ） |
| `stats` | `Dict[str, int]` | 操作種別ごとのカウント（gates, splits, merges等） |
| `phonon_data` | `Dict` | フォノン追跡結果 |
| `final_placement` | `Dict[int, int]` | ラウンド終了時のion配置 (ion_id -> zone_id) |
| `final_zone_contents` | `Dict[int, List[int]]` | ラウンド終了時のzone内イオン並び順 |
| `config` | `CompilerConfig` | 使用したコンパイラ設定 |

### CompileResult の使い方

#### 実行時間の推定

```python
from parallel_estimator import ParallelEstimator

est = ParallelEstimator()
r = est.estimate(result.timesteps)
print(f"Parallel time: {r.total_time / 1000:.2f} ms")
print(f"Sequential time: {r.sequential_time / 1000:.2f} ms")
print(f"Speedup: {r.speedup:.2f}x")
```

#### 命令列の検証

```python
from validator import validate_operations, validate_timesteps

# 逐次検証
seq_result = validate_operations(result.operations, initial, arch)

# 並列検証（zone競合・ion競合チェック付き）
par_result = validate_timesteps(result.timesteps, initial, arch)
```

#### テキスト出力（人間が読める命令列）

```python
from parallel_estimator import save_parallel_txt

save_parallel_txt(result.timesteps, "output.txt",
                  initial_placement=initial, arch=arch)
```

出力形式（案3: 1ops=1行、並列ops=ブロック展開）:
```
TS   0 [  100.00μs  @    100.00μs]  Split(ion8) z1->z2
TS   1 [  100.00μs  @    200.00μs]  [2P]
  | Split(ion9) z5->z4                    100.00μs
  | Swap(0,1) z1                           20.00μs
```

#### フォノン数の取得

```python
pd = result.phonon_data
print(f"Max axial phonon: {pd['max_axial']:.2f}")
print(f"Max transversal phonon: {pd['max_transversal']:.2f}")

# MSゲート実行時のフォノン数
for gate_info in pd['at_gates']:
    print(f"{gate_info['gate']}: {gate_info['phonons']}")
```

#### 複数ラウンドの配置引き継ぎ

```python
# Round 1
result1 = compiler.compile(dag1, initial)

# Round 2: final_zone_contents の順序を保持して配置
ordered_placement = {}
for zone_id in sorted(result1.final_zone_contents.keys()):
    for ion_id in result1.final_zone_contents[zone_id]:
        ordered_placement[ion_id] = zone_id

result2 = compiler.compile(dag2, ordered_placement)
```

### CompilerConfig のオプション

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `gate_selection` | str | `"min_id"` | ゲート選択戦略 |
| `memory_reorder` | bool | `False` | Measure中のMemory内先読み整列 |
| `measure_batch_size` | int | `1` | Measureのバッチサイズ（2+でバッチ化） |
| `interleave` | bool | `False` | スタビライザinterleave実行 |
| `eviction_strategy` | str | `"balanced"` | LIZからの退避戦略（`"balanced"` or `"nearest"`） |

## src/ の各ファイルの役割

| ファイル | 役割 |
|---------|------|
| `qccd_parser.py` | 独自DSLのパーサー（Lark使用） |
| `ast_def.py` | 操作の型定義（GateOp, SplitOp, ShuttleOp等） |
| `dag_builder.py` | 量子回路 → DAG変換、スタビライザ検出、hook error edge注入 |
| `qccd_compiler.py` | MovementPlanner + Scheduler + QCCDState（コンパイラ本体） |
| `estimator.py` | 逐次実行時間推定 + TimeParameters定義 |
| `parallel_estimator.py` | 並列実行時間推定 + テキスト出力 |
| `validator.py` | 命令列の制約検証（Sequential + Parallel） |
| `memory_reorder_optimizer.py` | Measure中のMemory内イオン整列最適化 |

## 回路ファイルの形式

`circuits/` 内の各 `.py` ファイルは以下のメタデータと回路文字列を持つ:

```python
QUBIT_COUNT = 10
DATA_QUBITS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ANCILLA_QUBITS = [9]

CIRCUIT = """
QUBIT 0 AT (0, 0)
...
RESET 9
R 0 1.5708 1.5708
MS 0 9 0.25
...
MEASURE 9
"""
```

`d5_generator.py` を使って d=5, d=7 の回路を自動生成可能:
```bash
cd circuits
python d5_generator.py 5 7    # d5/d7 の reuse1, reuse2, reverse_cnot を生成
```

## アーキテクチャ

7ゾーン1次元構成: `T(20) - M(20) - T(20) - L(4) - T(20) - M(20) - T(20)`

- T: Temporary zone（搬送時の一時領域）
- M: Memory zone（イオン格納）
- L: LIZ（ゲート実行領域）

## 実験スクリプト

実験スクリプトは `experiments/` 内にあり、コンパイラの出力を用いた各種分析を行う。
各スクリプトはコマンドライン引数で設定を変更可能（`--help` で確認）。

| ファイル | 目的 | 主な出力先 |
|---------|------|-----------|
| `exp_parallel.py` | 単一回路の並列分析 | `results/parallel/` |
| `exp_placement.py` | 初期配置戦略の比較 | `results/placement/` |
| `exp_multi_round.py` | 複数ラウンドQECシミュレーション | `results/multi_round/` |
| `exp_memory_reorder.py` | Memory整列最適化の効果検証 | `results/memory_reorder/` |
| `exp_interleave.py` | スタビライザinterleaveの効果検証 | `results/interleave/` |
| `exp_noise_injection.py` | フォノンベースのStimノイズ挿入 | `results/noise_injection/` |
