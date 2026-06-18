# Code cycle time scaling vs distance

snake_reuse1 ファミリの 1 ラウンド実行時間 (parallel / sequential)
を符号距離 $d \in \{3, 5, 7, 9\}$ について plot した。

## 設定 (canonical)

- placement strategy: **greedy_alt** (回路の登場順に LIZ 近接 zone から左右交互配置)
- eviction strategy: balanced
- architecture: simple_mlm (LIZ × 1, capacity 4; memory zones, capacity 20)
- ancilla: 1 個 (reuse される)
- rounds: 1 (single round of syndrome measurement)

## Canonical (greedy_alt) numbers

| d | par_time (ms) | seq_time (ms) |
|---|---------------|---------------|
| 3 | 26.92 | 32.24 |
| 5 | 120.93 | 148.62 |
| 7 | 376.53 | 448.64 |
| 9 | 921.22 | 1066.53 |


## Archived (split_even placement) for comparison

| d | par_time (ms) | seq_time (ms) |
|---|---------------|---------------|
| 3 | 30.01 | 35.49 |
| 5 | 161.80 | 183.36 |
| 7 | 540.68 | 585.00 |
| 9 | 1406.25 | 1473.66 |

`greedy_alt` で `par_time` が大きい d ほど比例的に改善する傾向。詳細は
`examples/_archive/20260620_split_even/README.md` 参照。

## Notes

- これらの数字は 2026-04-07 のポスタープロット (`plots/code_cycle_scaling.png`)
  と一致するように、 placement strategy を当時と同じ `greedy_alt` に揃えた
  結果。 d=9 で par_time = 921 ms (poster: 919.2 ms) と誤差 0.2%。
- 過去 `split_even` で生成していた時代の数字も `examples/_archive/` から
  読めるので、 placement の効果を直接比較できる (右パネル)。
- 4/7 時点では `result.json` に `compile_config` が無かったため、 過去の
  数字との乖離の原因追跡が困難だった。 2026-06-21 以降の `result.json`
  には placement_strategy 等を `compile_config` セクションに必ず保存する。
