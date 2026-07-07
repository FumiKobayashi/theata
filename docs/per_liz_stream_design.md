# Per-core stream + 明示 sync — Phase 1 設計 (v2)

2026-07-06。critical path 分解 (`experiments/exp_critical_path.py`) と
liz_parallel 試作の失敗を踏まえた、multi-LIZ コンパイラ再設計の具体案。
v2: swap 退避による T 競合 (指摘 1)、sync の決定時点 (指摘 2)、epoch 順序の
明確化 (指摘 3, 4)、placement の quota モデル (指摘 b)、所有権 window (指摘 c)
を反映。

前提 (既合意事項):

- **Phase 1**: ownership protocol を実行時に動かすのではなく、compile 時に
  競合が構成上起きないように計画し、併合検証で競合ゼロを保証する。
- **compiler は実行時間を見ない**。時間評価は estimator の責務。
- global TimeStep を LIZ 間に強制しない。
- 評価軸 = per-core タイムライン長 + sync 待ち時間。

---

## 0. Core モデルと handoff cell

`chained_mlm` の zone 列 `T-M-T-(L-T-M-T)×N`:

```text
zone:   0T  1M  2T  3L  4T  5M  6T  7L  8T  9M  10T
core 3:     [------1M..5M------]
core 7:                     [------5M..9M------]
```

- **core** = LIZ とその左右の M、間の T を含む連続 zone 区間。core_id = LIZ zone id。
- **T は core 排他ではない** (2026-07-06 指摘): `_expand_swap_inline`
  (qccd_compiler.py:1262) は中間ペアの swap で**左右両方の T** に退避する。
  つまり core A が shared M 内で並べ替えをすると、反対側 core の領分だった T
  まで使う。「反対側で同時に並べ替えが起きない限り実害はない」は事実だが、
  偶然に頼らず構造で保証する:
- **handoff cell** := {T_left, shared M, T_right} を**所有の単位**にする。
  根拠: 各 T の用途は (a) 隣接 L↔M 間の輸送、(b) 隣接 M の swap 退避、の
  2 つしかなく、shared M に隣接する T の用途は**すべて shared M アクセスに
  付随する**。cell ごと所有を渡せば T の競合は構成上起きない。
  破れがあっても race-freedom check (§6) が機械的に検出する。
- 実装: `QCCDArchitecture.handoff_cell(shared_m_id) -> List[int]` と
  `zone_to_cores(zone_id)` を追加。

## 1. IR: CoreStream + SyncOp

```python
@dataclass
class SyncOp(Operation):        # ast_def.py に追加
    sync_id: int                # release/acquire ペアで一致
    role: str                   # "release" | "acquire"
    cell_id: int                # 受け渡す handoff cell (shared M の zone id で代表)
    # 意味論: release(k) → acquire(k) の happens-before 辺 (一方向)。
    # rendezvous barrier ではない: release 側は待たない。

@dataclass
class CoreStream:
    core_id: int
    ops: List[Operation]        # flat ops。使用 zone ⊆ core zones、+ SyncOp
    timesteps: List[TimeStep]   # core 内 Scheduler の出力

@dataclass
class MultiStreamResult:
    streams: Dict[int, CoreStream]
    syncs: List[SyncPoint]      # (sync_id, cell_id, from_core, to_core) の索引
    # merged: canonical 併合ビュー (§5) から従来 CompileResult を生成し
    # downstream (noise, recompile, Stim) を当面無改修で通す
```

sync は stream 内マーカー。per-LIZ ファイル分割 (format 案 B) にそのまま
直列化でき、stream を切っても一緒に動く。索引 (SyncPoint) は派生情報。

## 2. sync はいつ決まるか (指摘 2 への回答)

**sync は「ロックされていたら待つ」ものではない。compile 時に epoch driver が
計画的に配置する。**

- 「どの core がいつ cell を所有するか」は plan 開始前に epoch schedule
  (§3) として決まっている。SyncOp はその境界に**計画的に**発行される。
- 検証 (§6) は sync を**挿入しない**。race-freedom を**確認するだけ**で、
  違反 = 設計バグとして compile error にする (Phase 1 は fail fast)。
- compile 中に「ロック待ち」という状況は存在しない: driver は epoch を
  大域順で**逐次** plan する (§4) ので、plan 時点で cell の状態は常に確定
  している。「待ち」は estimator の時間領域にのみ現れ、per-sync 待ち時間
  として観測される。
- 最適化不足の懸念は正しい: epoch が粗いほど sync は減るが待ちが増えうる。
  これは §8 の window 政策として扱い、estimator の待ち観測で粒度を詰める。

## 3. Epoch schedule: cell 需要による block 分割と交互所有 (指摘 3 への回答)

用語を既存に揃える (「pure 帯 / boundary 帯」は撤回):

- **local stab / boundary stab**: 既定義 (2026-05-28)。コード上の
  `pure_idx / shared_idx` (spatial_partitioner.py:206) に対応。
- **block**: 1 つの core が epoch 中に連続実行する stab の集まり。分類基準は
  local/boundary そのものではなく **cell 需要**: その stab の実行に
  handoff cell が必要か (boundary stab は常に必要。shared M に**住んでいる**
  local データを使う stab も必要 — §5 の quota 配置で生じる)。
  cell-free block / cell-using block と呼ぶ。

**所有順序は global な LIZ1 > LIZ2 > LIZ3 ではない。cell ごとに独立した
交互スケジュール**を組む。chain の core 列は 2-彩色できるので:

```text
round r,  phase 1: 奇数 core が自分の全 cell を所有 (cell-using block を実行)
                   偶数 core は cell-free block を実行
          phase 2: 所有を反転
```

- これは `phase_alternation` (偶 idx = pure→shared、奇 idx = shared→pure) の
  一般化・明示化そのもの。2-LIZ では現行挙動と一致する。
- sync 数 = O(rounds × shared M 数) で極小。
- **N≥3 の中央 core** (排他 M を持たない、§5): 左右 2 cell を持つ。phase 1 で
  両方所有し、左右の cell-using block を続けて実行する。phase 2 (非所有) に
  cell-free の仕事が無ければ idle になる — これは正しさの問題ではなく
  負荷分配の問題で、estimator の per-sync 待ちで観測し banding / block 構成で
  詰める (§8)。
- **release の位置**: epoch の「終わり」ではなく、block 中で cell に触る
  **最後の op の直後**に機械的に前倒しして置く (emit 後に stream を後方走査
  して決められる)。計算 (MS) はイオンが LIZ に出た後なので所有不要。

## 4. Per-core planning: 逐次スナップショット方式 (指摘 4 への回答)

「LIZ1 の core から順に stream を丸ごと出力する」の**ではない**。
**epoch 単位**で、sync 依存と整合する大域順に**逐次** plan し、各 core の
stream に追記していく。2-LIZ の 1 round の具体例:

```text
plan 順序 (compiler 内部の逐次処理):
 1. core7 (奇): cell-using block を plan   … M5-cell を所有。ops は stream7 へ
 2. core3 (偶): cell-free block を plan    … M1/T2/L3 のみ。ops は stream3 へ
    (1 と 2 の間に依存は無い。driver が決定的な順で処理するだけ)
 3. core7 の release(k) を発行、M5-cell の zone_contents (並び含む) を snapshot
 4. core3: acquire(k) を発行し、snapshot を core3 の state に注入して
    cell-using block を plan
 5. core7: cell-free block を plan  (3 と並行依存なし)
 … round r+1 は向きを反転
```

- 出力される stream3 / stream7 は並列の成果物だが、**plan 処理自体は逐次**。
  これにより consumer は producer が実際に残した shared M の並びを初期状態
  として plan でき、「epoch 終端で canonical 並びに復元する」規約が不要になる。
- 実装は MovementPlanner 流用: **PlannerSession** (core ごとに QCCDState が
  永続、`run(nodes)` を epoch 単位で複数回呼ぶ) + **CoreView** (zone 集合を
  core に mask し、非所有 cell を経路・退避候補から除外)。
- 副産物: core 内の path は他 LIZ を通らないので pass-through guard は
  invariant check に降格。ancilla は per-LIZ 専属 (partition 修正済み) で
  core を跨がない。

## 5. Placement v2: quota モデル (指摘 b への回答)

実行順は構想どおり: **partition (stab の LIZ 割当) → placement → per-core
plan**。現行コード (`partition_spatial_row` → `compute_initial_placement_3tier`
→ compile) と同じ骨格。

placement は shared M を考慮**する**。ただし「shared M には cross-LIZ qubit
だけ」という規則は**採らない**:

- 構造的事実: 中央 core (N≥3) は排他 M を持たない (両隣とも shared)。
  その local データは shared M に住む**しかない**。
- 「cross のみ」規則は端 core でも収容率を落とす (指摘のとおり)。
- **quota モデル**: 各 shared M の容量を
  `[left core の local 枠 | cross-LIZ 枠 | right core の local 枠]`
  に割る。cross-LIZ qubit は従来どおりその LIZ ペアの shared M へ。
  local データは自分の LIZ に隣接する M のうち枠の空いている所へ
  (排他 M 優先、あふれたら shared M の自枠へ)。
- **帰結** (§3 と結合): shared M に住む local データを使う stab は
  cell-using block に入る。つまり placement が block 構成を決める。
  収容率 (quota の切り方) と epoch 中の並列性はトレードオフであり、
  estimator の per-sync 待ちを見て調整する変数になる。
- 2-LIZ では quota モデルは現行 3-tier に退化する (端 core の local は
  排他 M に収まる限りそちらへ)。無回帰。

## 6. 併合検証 (MultiStreamValidator)

2 層:

1. **Race-freedom check** (新規、核心): stream 内順序 + sync 辺から vector
   clock を計算し、**異なる stream の 2 op が zone または ion を共有するなら
   happens-before で順序付いていること**を要求。違反 = compile error。
   `ParallelValidator._zones_used/_ions_used` (validator.py) を流用。
   粒度は zone レベル (cell 単位ではなく zone 単位で見れば十分 — cell 所有が
   正しく機能していれば自動的に通る。T の swap 退避 (§0) の破れもここで
   機械検出される)。
2. **意味検証**: race-free なら linearization 非依存が保証されるので、
   canonical 併合ビュー 1 本 (HB と整合する決定的 linearization、tie-break は
   round → epoch → core_id) を既存 Validator + ParallelValidator に通す。
   **MEASURE 順 = Stim rec[] index はこの併合ビューが定める**。downstream
   (recompile_multi_round / inject_noise_approach_b) は併合ビューを従来
   CompileResult として食える (DETECTOR は identity match)。

## 7. Estimator v2 (event simulation)

- per-core timesteps を時間軸に並べ、acquire を含む timestep の開始を
  対応する release を含む timestep の終了以降に制約する event simulation。
- wall clock = max(core 終了時刻)。**per-sync 待ち時間** (どちらが何 ms
  待ったか) を出す — 負荷分配・window 政策チューニングの直接の観測量。
- estimate_per_liz の帰属 tie-break 問題は消滅 (op は stream 所属で確定)。
- `_per_round_dephasing_time_us`: `sum(crit)` → per-core 時間の max に修正。

## 8. 所有権 window 政策 (指摘 c への回答)

**release を「Shuttle 用 Split が済んだ瞬間」ごとに置く (最細粒度) のは
Phase 1 では採らない。** 懸念のとおり、現行の「移動後即計算」だと
fetch→compute→return→fetch… の度に所有権が往復し、両隣が cell を連続要求
する状況で sync 待ちが膨らむ (ping-pong)。

- **Phase 1 = block 単位 window**: cell-using block の間所有し続け、block 内
  最後の cell アクセスの直後に release (§3)。非所有側はその間 cell-free block
  を実行するので、待ちが仕事で埋まるのが定常状態。
- **Phase 1.5 = configurable window**: estimator の per-sync 待ちが大きければ
  細分化する。このとき **lazy-move が新しい存在意義を得る**: 「必要分を
  まとめて fetch → release → 連続計算 → 次の window で return + fetch」は
  所有 window を最小化する政策そのもの。単一 LIZ では負の結果だった
  (project_lazy_move_result) が、multi-LIZ では sync 待ち削減という別の
  目的関数で再評価する。IR は変えず sync が増えるだけなので、政策は
  driver 層の差し替えで済む。

## 9. Partition (指摘 a への回答)

**入力回路の stab 順を機械的に N 分割する方式は採らない。既に実装がある**:
`partition_spatial_row` (spatial_partitioner.py:123) が data qubit の平均 row
で N 帯に分ける row banding を行い、ancilla も LIZ 数に合わせて確保する
(ace2c50 で N-LIZ 一般化済み)。

- 入力順の機械分割は同じ row の stab を別 LIZ に割きやすく、cross-LIZ data
  qubit を増やす。最悪「各 cross-LIZ qubit は高々 2 LIZ 訪問」条件
  (2026-05-28 の分割条件) を壊す。row banding は境界を薄い 1 列に保つ。
- 制約 (既知): 2D 座標 (trap=row) が要る。1D snake layout (全 trap=0) では
  退化する。座標の無い回路は座標付与するか、fallback として順序分割を許す。
- 割り切れない場合は band 幅の丸めで自然に不均等になるだけ (対応済み)。
  band 幅そのものが負荷分配のレバー (§8 と同じ観測量で回す)。

## 10. 実装マイルストーン

| # | 内容 | 触る場所 |
| --- | --- | --- |
| M1 | core/cell モデル + IR (SyncOp/CoreStream/MultiStreamResult) | ast_def, qccd_compiler (小) |
| M2 | PlannerSession + CoreView + eviction の core 局所化 | qccd_compiler (大、本丸) |
| M3 | epoch driver + snapshot handoff + sync 発行 + placement quota | qccd_compiler / spatial_partitioner |
| M4 | race-freedom check + canonical 併合 → 既存 validator 接続 | validator (新クラス) |
| M5 | estimator v2 (event sim + per-sync 待ちレポート) | parallel_estimator |
| M6 | 評価: d5 2-LIZ / d7 2·3-LIZ、band 幅・window 政策チューニング | experiments |

受け入れ基準:

- validation 0 errors (既存 + race-freedom)、multi-round recompile の Stim 一致。
- d5 2-LIZ: wall clock が現状 141 ms から per_liz_ideal 96.6 ms 側へ有意に接近
  (critical_path 126.9 ms は**旧 stream の**下限であり、stream 自体が変わるので
  下回ってよい)。残 gap は per-sync 待ちとして定量分解できること。

## 未決 (実装しながら決めてよい細部)

- 端の余り T (zone 0, 10) の帰属 — 端 core に含めるだけで実害なし。
- epoch 境界での MEASURE バッチの切り方 (block を跨ぐバッチは作らない、で開始)。
- quota の初期値 (排他 M 優先で詰めて、あふれ分だけ shared M、で開始)。
- 中央 core の左右 cell-using block の内部順序 (M3 で具体化)。
