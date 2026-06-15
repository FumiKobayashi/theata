# Archive: 20260620 split_even placement

このディレクトリは 2026-06-20 時点までの `examples/` 配下のスナップショット。
当時の `exp_build_examples.py` は **split_even placement strategy** を使っていた:

> "前半 (=index < N/2) の data qubit を zone 1 へ、後半を zone 5 へ。
>  ancilla qubit はすべて zone 5 へ。"

これは「placement 戦略を真面目に選んでいないこと」の典型例で、 ポスター時代 (2026-04 頃)
に使われていた `greedy_alt` (回路の登場順に左右交互、 LIZ 近接 zone から配置) より
劣化した結果になっていた。具体的な比較は `examples/_archive/README.md` を参照。

## このスナップショットの内容

| dir | 説明 |
|------|------|
| `single_liz/` | d=3, 5, 7 の 1 round compile + noisy.stim (PAULI_CHANNEL_2 strict) |
| `multi_round/` | d=3, 5 の 3 round MWPM 復号結果 |
| `distance_scan/` | d=3, 5 の論理エラー率 (50k shots) |
| `phonon_scaling/` | d=3, 5, 7, 9 の最大フォノン数 vs 距離 |
| `rep_code/` | d=3 repetition code (placement 影響は受けるが、 1 ancilla しか動かさないので僅か) |

## なぜ捨てずに残すか

placement strategy が QEC simulation の論理エラー率に与える影響を可視化する反例として
価値がある。具体的には:

- `multi_round/d3_reuse1_3rounds/mwpm_result.json`: split_even での p_L = 2.11e-2
  - 同じ d=3 で greedy_alt 配置時の p_L (再生成後の `examples/multi_round/.../mwpm_result.json`) と比較すれば、 placement 改善で transport noise が減り p_L がどう改善するかが定量的に分かる
- distance_scan plot との比較で同様の議論ができる

## 注意事項

- このアーカイブ生成時、 `result.json` には `compile_config` フィールドがなかった。
  どの placement / eviction strategy で生成されたかは **このディレクトリ名でしか判別不可能**。
  この件を踏まえて、 2026-06-21 以降の `result.json` には `compile_config` を含めるように
  `src/formatter.py` を更新した。
