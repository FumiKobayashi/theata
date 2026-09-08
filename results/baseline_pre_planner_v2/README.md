# Baseline: planner v2 (per-memory state machine) 着手前の正準値

取得日: 2026-09-08
目的: MovementPlanner を state machine 方式に置き換えると命令列そのものが変わり、
p_L / n̄ / 時間の全データが影響を受けるため、変更前の正準値をここに凍結する。

## コード状態

- commit: f85b281 (main)
- 未コミット: src/event_scheduler.py, experiments/exp_event_schedule.py,
  experiments/exp_verify2_general.py, experiments/exp_placement_bridge.py (M)
  ※イベント駆動スケジューラは opt-in であり、本ベースラインの p_L 経路
  (ロックステップ) には影響しない。

## 実行コマンド

    python experiments/exp_cooling_distance_scan.py --quiet-clip \
        --out-dir results/baseline_pre_planner_v2

(E-C6 1-LIZ distance scan: d=3/5/7, rounds=3, shots=20000,
 arms = no-cool / anc@2000us / merge@1.5+anc, seed=2026)

## 正準値 (cooling_distance_scan_r3_20260908_190524.json)

| arm | d | n̄ cap | n̄ avg | wall [ms] | p_L (3 rounds) |
|---|---|---|---|---|---|
| no-cool       | 3 |  3.70 |  1.55 |   84.1 | 0.01655 |
| no-cool       | 5 | 15.02 |  6.00 |  366.7 | 0.20410 |
| no-cool       | 7 | 51.68 | 19.92 | 1135.0 | 0.48950* |
| anc@2000us    | 3 |  2.61 |  1.01 |  129.8 | 0.01090 |
| anc@2000us    | 5 |  9.88 |  3.50 |  504.3 | 0.06970 |
| anc@2000us    | 7 | 32.83 | 11.34 | 1409.7 | 0.47090* |
| merge@1.5+anc | 3 |  2.27 |  0.93 |  178.1 | 0.00975 |
| merge@1.5+anc | 5 |  6.49 |  2.62 |  786.4 | 0.03280 |
| merge@1.5+anc | 7 | 20.02 |  8.01 | 2059.8 | 0.39015* |

\* = LUT 上限 (n̄=20) 超過ゲートありの下限値。全 arm で Λ<1 (閾値未達) は既知の 8/31 判定と同じ。

## セットで参照する搬送・時間側ベースライン (同時期の日付入り CSV)

- results/placement_bridge/boustro_variants_d{3,5,7,9}_r3_20260903 / 20260908.csv
  — surface 符号 d3-d9 の検証① (schedule/placement 別 transport・T_total・n̄)
- results/verify2_general/verify2_general_20260908_014431.csv
  — 一般符号 4 種の検証② (T_lock/T_event/hidden_us_r1 込み)

## planner v2 で予測される変化 (2026-09-08 の議論より)

- 厳密 JIT (prefetch_depth=0): greedy_alt 初回スタビライザの両側同時稼働が消え直列化
- d3: Measure 中 sort により測定後配置が LIZ 近傍に整列 → PMode1 ほぼ直列化。
  d3 は LIZ 律速 (wall/資源下界=1.21x) なので直列化のコストは小さい。
  成功指標は並列度でなく「wall が LIZ busy 下界 (18.7ms) にどれだけ漸近するか」。
- prefetch_depth は {0,1} をパラメータ化して比較する。
