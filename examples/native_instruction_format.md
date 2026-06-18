# Native instruction format reference

QCCD コンパイラが出力する **TimeStep stream のテキスト表記** (例:
`examples/single_liz/d3_reuse1/timesteps.txt`) の文法と意味論の参照ドキュメント。

アニメーション化など外部ツールでこの表記を解釈する際に参照してください。

---

## 1. 文法を定義しているファイル

| 場所 | 役割 |
|------|------|
| [`src/parallel_estimator.py:261`](../src/parallel_estimator.py) `ParallelEstimator._op_summary` | **各 op の文字列表記の本体** (R(0) z3 等の整形ロジック) |
| [`src/parallel_estimator.py:289`](../src/parallel_estimator.py) `format_timesteps_text` | TS 行 (`TS N [crit @ cumu] [NP] ...`) のレイアウト |
| [`src/parallel_estimator.py:347`](../src/parallel_estimator.py) `save_parallel_txt` | ファイル全体 (header + TS 列) の構造 |
| [`src/formatter.py:33`](../src/formatter.py) `format_op_summary` | `_op_summary` の同形 mirror (新規コードの推奨入口) |
| [`src/ast_def.py`](../src/ast_def.py) | 各 op の **dataclass 定義** (フィールド = 表記に出る値) |

文法変更が必要なら `_op_summary` を編集 + `format_op_summary` も同期。

---

## 2. アーキテクチャの前提

`single_liz/dN_reuse1/timesteps.txt` の冒頭にゾーン構成が記載されている:

```
# Architecture: z0:T(20) - z1:M(20) - z2:T(20) - z3:L(4) - z4:T(20) - z5:M(20) - z6:T(20)
```

意味:
- ゾーンは `z0, z1, z2, ...` と一直線に並ぶ (1D アーキテクチャ)
- `T(20)` = **Temp zone**、 capacity 20 (= 最大 20 イオン格納可)
- `M(20)` = **Memory zone**、 容量 20、 イオンの主住居
- `L(4)` = **LIZ (Logical Ion Zone)**、 容量 4、 **ゲート実行はここでしか行わない**

イオンは普段 Memory に居て、 ゲートのとき LIZ に運ばれ、 終わったら Memory に戻る。 Temp は経由地。

初期配置:
```
# Initial placement: {0: 1, 1: 1, 2: 1, 3: 1, 9: 1, ...}
```
左辺 = ion ID、 右辺 = 初期に居るゾーン ID。

---

## 3. 各 op の文法と意味

### 3.1 ゲート関連 (LIZ で実行)

| 表記 | 例 | 意味 | 元クラス |
|------|----|------|---------|
| `Reset(ion) zN` | `Reset(9) z3` | ion を \|0⟩ に初期化 (LIZ z3 で実行) | `ResetOp` |
| `Measure(ion) zN` | `Measure(9) z3` | ion を Z 基底で測定 (LIZ z3 で実行) | `MeasureOp` |
| `R(ion) zN` | `R(0) z3` | 単 qubit 回転 (Mølmer-Sørensen 系列の構成要素) | `GateOp` (type="R") |
| `MS(a,b) zN` | `MS(0,9) z3` | 2 qubit ゲート (Mølmer-Sørensen) on ion a と b。 zN は LIZ | `GateOp` (type="MS") |

注: `R` ゲートの実 angle parameter は表記から省略 (compile output には保持されているが summary 形式では落とす)。

### 3.2 移動関連

| 表記 | 例 | 意味 | 元クラス |
|------|----|------|---------|
| `Split(ion_x) zA->zB` | `Split(ion1) z1->z2` | 元 zone A のイオンチェーンを分割。 ion_x が右端 (or 左端) になる位置でカット、 切り出した片方を zB へ動かす | `SplitOp` |
| `Merge zA+zB` | `Merge z2+z3` | 隣接 zone A と B のチェーンを結合 (どちらかに保存される。 通常 M or L 側) | `MergeOp` |
| `Swap(a,b) zN` | `Swap(3,2) z1` | 同一 zone 内で ion a と b の順序を入れ替え | `SwapOp` |
| `Shuttle(ion_x) zN->(trap, site)` | `Shuttle(ion0) z2->(3, 0)` | ion_x を zone N から、 (trap, site) の Location へ運ぶ。 dst は zone ID ではなく **Location** (trap=3, site=0 を意味) | `ShuttleOp` |

`Shuttle` の `(trap, site)` 表記は `Location` dataclass の `__repr__` から来ている (`src/ast_def.py` 参照)。 通常 `trap` は zone ID と一致するが、 必ずしも 1:1 ではない (確認したい場合は `Location` 定義参照)。

### 3.3 補助

| 表記 | 意味 |
|------|------|
| `Barrier(...)` | 同期点。 transport は通過できない | `BarrierOp` |

---

## 4. TimeStep 行のレイアウト

### 4.1 単独 op の場合
```
TS    2 [  100.00μs  @    311.11μs]  Merge z2+z3
```
- `TS    2` = ts index (0-indexed)
- `[ 100.00μs @ 311.11μs]` = この TS の critical path duration / その TS が終わった時点の累積時間
- 末尾 = op の summary

### 4.2 並列 op の場合
```
TS    6 [  100.00μs  @    721.67μs]  [3P]
  | Merge z0+z1                                100.00μs
  | Split(ion10) z5->z4                        100.00μs
  | MS(0,9) z3                                  59.00μs
```
- `[3P]` = この TS に 3 個の op が **並列に走る** ことを示す
- 続く行が `|` 始まりで各 op (右端は個別 duration、 critical path = max)
- 同 TS 内の op たちは互いに依存しない (異なる zone を使用)

---

## 5. ファイル全体の構造

`single_liz/d3_reuse1/timesteps.txt` の構成:
```
# Architecture: ...                ← Zone 構成
# Initial placement: ...           ← ion → zone の初期マップ

# TimeSteps: N | Operations: M | Avg parallelism: P     ← 全体統計
# Parallel time: ... | Sequential time: ... | Speedup: ...

TS    0 [ ...
TS    1 [ ...
...
```

---

## 6. dataclass 定義の参照

詳しいフィールドが知りたい場合は [`src/ast_def.py`](../src/ast_def.py):
- `GateOp(type, targets, params, zone_id)` — params は 表記から省略
- `ResetOp(targets, zone_id)`
- `MeasureOp(targets, zone_id)`
- `ShuttleOp(target, dst: Location, src_zone, dst_chain_length)`
- `SwapOp(targets, zone_id)`
- `SplitOp(ion_id, src_zone_id, dst_zone_id)`
- `MergeOp(zone_left, zone_right)`
- `BarrierOp(targets)`
- `Location(trap, site)` — `Shuttle` の dst で使う

---

## 7. 補足情報: 他にもある関連ファイル

| ファイル | 内容 |
|----------|------|
| [`flat_ops.txt`](single_liz/d3_reuse1/flat_ops.txt) | 並列化 **前** の生成順 (scheduler に入る前の native 命令列)。 各行 = 1 op、 同じ `_op_summary` 表記 |
| [`timesteps.txt`](single_liz/d3_reuse1/timesteps.txt) | 並列化 **後** の TimeStep 列 (上記の文法) |
| [`result.json`](single_liz/d3_reuse1/result.json) | 上記の数値だけ取り出した structured 表現 + フォノン track |

アニメーション化には `timesteps.txt` (TS 単位で並列実行が見える) が向いていると思われます。

---

## 8. 補足: 旧形式との違いがあるかもしれない場合

このリポジトリでは 2026-03-18 (`f9c5451 Pipeline 2`) 以降この表記が定着している。
3 月以前にアニメーションコードを書かれていた場合、 当時とは異なる文法に変更されている
可能性が高い。 git で `src/parallel_estimator.py` の履歴を遡ると変更点が見える。

具体的に旧表記との差分が知りたい場合は別途お問い合わせください。
