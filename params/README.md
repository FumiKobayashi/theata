# Simulation parameters

命令の時間 / フォノン / cooling パラメータの外部設定ファイル置き場。

- 正本 (single source of truth) はコード側: `src/estimator.py` の `TimeParameters` dataclass デフォルト
- `default.json` は全キーをダンプした**参照用サンプル** (`sim_params.dump_time_parameters` で生成)。
  コード側のデフォルトを変えたら再生成すること:

  ```
  python -c "import sys; sys.path.insert(0,'src'); from sim_params import *; dump_time_parameters(load_time_parameters(), 'params/default.json')"
  ```

## 使い方

部分指定の JSON を作り (未指定キーはデフォルト)、`sim_params.load_time_parameters` で読む:

```json
{ "ms_gate": 100.0, "cool": 300.0, "phonon_cool_residual_axial": 0.1 }
```

```python
from sim_params import load_time_parameters
tp = load_time_parameters("params/my_experiment.json")

QCCDCompiler(arch, config, time_params=tp)         # フォノン追跡に反映
compile_per_liz(..., time_params=tp)               # per-LIZ stream 側
ParallelEstimator(tp) / Estimator(tp)              # 時間推定に反映
```

未知キーは `ValueError` (typo 検出)。YAML (.yaml/.yml) も pyyaml があれば可。

## Cooling 関連キー

| キー | 意味 |
|---|---|
| `cool` | Cool 命令 1 回の所要時間 μs (placeholder 値、要文献確認) |
| `phonon_cool_residual_axial` | 冷却後の残留 axial フォノン数 |
| `phonon_cool_residual_transversal` | 冷却後の残留 transversal フォノン数 |

## フォノンのモード構成 (2026-10-02 会議決定)

1 イオンあたり **axial 1 本 + transverse 2 本の計 3 モード**を追跡する。

- 無印 `phonon_*_transversal` = **第 1 transverse モード**。誤り参照モード
  (MS 誤り LUT・閾値冷却ポリシーが参照する側)。heating rate の大きい方
  (worse-case) をこちらに割り当てる規約。
- `phonon_*_transversal2` = **第 2 transverse モード**。`null` (未指定) なら
  第 1 モードと同じ per-op コストを使う (モード間でコストを区別しない既定)。
