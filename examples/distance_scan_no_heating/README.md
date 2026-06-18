# Distance scan: heating-free scenario

すべての MS ゲートの Pauli error として **n_bar=0.01 を強制** した仮想シナリオ
での距離スケーリング。 transversal phonon の蓄積 (= heating) が完全に抑えられた
場合の理想上限性能を示す。

`examples/distance_scan/` (実際の phonon を使う = heating-on) と比較対象として
保存。

## 設定

- placement strategy: **greedy_alt** (canonical)
- noise model: Approach B (PAULI_CHANNEL_2 lookup) で **override_n_bar=0.01** 強制
- rounds: 3
- decoder: MWPM (pymatching)
- 全 MS で同じ 15 Pauli 確率 (= n_bar=0.01 の lookup table 行) を使用
- transport-time dephasing (lumped) は通常通り適用 (heating off は phonon-induced
  MS error にのみ影響、 idle dephasing には影響しない)

## ファイル

| ファイル | shots/d | 用途 |
|----------|---------|------|
| `plot_50000shots.png` / `data_50000shots.json` | 50,000 | 初期確認、軽量 |
| `plot_100000shots.png` / `data_100000shots.json` | 100,000 | 精度重視、 d=3/5/7 で stderr 半減 |

## 結果

### 50k shots

| d | n_qubits | errors | $p_L$ (3 rounds)   | per-round |
|---|----------|--------|--------------------|-----------|
| 3 |   10     |  133   | 2.66e-3 ± 2.3e-4   | 8.9e-4    |
| 5 |   26     |   25   | 5.00e-4 ± 1.0e-4   | 1.7e-4    |
| 7 |   50     |    8   | 1.60e-4 ± 5.7e-5   | 5.3e-5    |
| 9 |   82     |    1   | 2.00e-5 ± 2.0e-5   | 6.7e-6    |

### 100k shots

| d | n_qubits | errors | $p_L$ (3 rounds)   | per-round |
|---|----------|--------|--------------------|-----------|
| 3 |   10     |  300   | 3.00e-3 ± 1.7e-4   | 1.0e-3    |
| 5 |   26     |   52   | 5.20e-4 ± 7.2e-5   | 1.7e-4    |
| 7 |   50     |   11   | 1.10e-4 ± 3.3e-5   | 3.7e-5    |
| 9 |   82     |    2   | 2.00e-5 ± 1.4e-5   | 6.7e-6    |

両 shot 数で **$p_L$ が d とともに単調減少** (= surface code の本来の指数的
距離保護)。 d=3 → d=9 で **100〜150 倍改善**。

d=9 は 100k shots でも誤り 2 個しか観測されておらず統計揺らぎが大きい
(エラーバーが log スケールで非常に大きく見える)。 もし精度を上げたければ
1M shots 以上で再走させると良いが、 傾向は十分把握できる。

## 比較: heating-on (実 phonon) との対比

`examples/distance_scan/` の現実モードは d=3,5 のみ (d=7,9 は MS gate の
transversal phonon が lookup table 上限 20 を超えてクリップされ意味のない数値
になるため除外):

| d | heating-on $p_L$ | heating-off $p_L$ (100k) | 改善倍率 |
|---|------------------|--------------------------|---------|
| 3 | 1.58e-2          | 3.00e-3                  |  5.3× |
| 5 | 2.02e-1          | 5.20e-4                  |  **388×** |

## 解釈

heating (transversal phonon の累積) が、 我々の単一 LIZ QCCD compiler の
distance scaling を完全に破壊している主因であることがこの比較で **定量的に
明確化** される:

- heating-on: d↑ で transport が増え phonon が増え MS error が指数的に増大
  → distance protection が打ち消され、 d=5 で per-round 7% という壊滅的な値
- heating-off: 物理的に意味のあるエラー (gate error, idle dephasing) しか
  残らず、 surface code 本来の挙動 (d↑ で指数的に improve) が見える

この差が **multi-LIZ で並列化して transport を減らす** ことの定量的な意義を
示している (multi-LIZ では実 phonon が下がり、 heating-on の曲線が heating-off
側に近づく方向に動くはず)。

## 再生成方法

```bash
# 50k shots
python experiments/exp_distance_scan.py --distances 3 5 7 9 \
    --shots 50000 --no-heating

# 100k shots
python experiments/exp_distance_scan.py --distances 3 5 7 9 \
    --shots 100000 --no-heating
```

出力先は `results/noise_injection/distance_scan_no_heating_<timestamp>.{png,json}`。
本ディレクトリへの copy は手動 (or build script への組み込みは TODO)。
