# Distance scan: heating-scaled scenarios

Transversal phonon の蓄積に **一律の倍率 α (`phonon_scale`)** をかけた仮想シナリオ
での距離スケーリング。 ハードウェア改良で heating が α 倍に抑えられた場合の
論理エラー率を予測する。

`examples/distance_scan/` (α=1.0, heating-on)、
`examples/distance_scan_no_heating/` (α=0, n_bar=0.01 強制) と比較する位置付け。

## 設定

- placement strategy: **greedy_alt** (canonical)
- noise model: Approach B (PAULI_CHANNEL_2 lookup)
- rounds: 3
- shots: 50,000 per d
- decoder: MWPM (pymatching)
- distances: d = 3, 5, 7 (d=9 は scale 0.5 で transversal が table 範囲超過するため除外)

## 等価性の根拠

phonon 蓄積は **加算的** なので、 全 op 一律 α 倍の rate scaling と 最終 phonon の
α 倍 (= post-hoc scaling) は **数学的に等価**。 後者を `phonon_to_pauli_lookup_rates`
の `phonon_scale=α` 引数で実装している。 compiler 再走不要のため軽量。

検証 trace は `examples/distance_scan_heating_scaled/` 配下では作っていないが、
`src/pauli_twirled_error_probabilities.csv` の対応行を引いて手で再現可能。

## 結果

### scale = 0.5 (heating × 1/2)

| d | errors | $p_L$ (3 rounds)   | per-round |
|---|--------|--------------------|-----------|
| 3 |   448  | 8.96e-3 ± 4.2e-4   | 3.0e-3    |
| 5 |  2053  | 4.11e-2 ± 8.9e-4   | 1.4e-2    |
| 7 | 21231  | 4.25e-1 ± 2.2e-3   | 1.7e-1    |

heating の半減でも、 d=7 で per-round 17% という壊滅的な値。 placement 改善
だけでは追いつかないレベルの phonon 由来エラー。

### scale = 0.1 (heating × 1/10)

| d | errors | $p_L$ (3 rounds)   | per-round |
|---|--------|--------------------|-----------|
| 3 |   225  | 4.50e-3 ± 3.0e-4   | 1.5e-3    |
| 5 |   108  | 2.16e-3 ± 2.1e-4   | 7.2e-4    |
| 7 |   392  | 7.84e-3 ± 3.9e-4   | 2.6e-3    |

d=3→5 で改善が見え始める。 ただし d=5→7 で再び悪化 (V 字)。 まだ heating が
distance の利得を打ち消す領域。

### scale = 0.01 (heating × 1/100)

| d | errors | $p_L$ (3 rounds)   | per-round |
|---|--------|--------------------|-----------|
| 3 |   192  | 3.84e-3 ± 2.8e-4   | 1.3e-3    |
| 5 |    40  | 8.00e-4 ± 1.3e-4   | 2.7e-4    |
| 7 |     4  | 8.00e-5 ± 4.0e-5   | 2.7e-5    |

ようやく単調減少が出る (= surface code の本来の指数的距離保護が機能)。
**heating-free に近い挙動**。

## 全シナリオ比較

| d | heating-on (×1) | ×0.5 | ×0.1 | ×0.01 | heating-free (n=0.01) |
|---|-----------------|------|------|-------|----------------------|
| 3 | 1.58e-2*        | 8.96e-3 | 4.50e-3 | 3.84e-3 | 2.66e-3 |
| 5 | 2.02e-1*        | 4.11e-2 | 2.16e-3 | 8.00e-4 | 5.00e-4 |
| 7 | (table 超過で N/A) | 4.25e-1 | 7.84e-3 | 8.00e-5 | 1.60e-4 |

\* heating-on (×1) の数字は `examples/distance_scan/data.json` から、 d=3,5 のみ。
d=7 以上は実フォノンが lookup table 上限 20 を多数超えて clip artifact 出力
だったため、 信頼できる値が無い。

## 解釈

- **scale ≥ 0.5**: heating が dominant、 distance を上げても無意味 (悪化さえする)
- **scale = 0.1**: 部分的に protection が機能するが d=5→7 でまだ V 字
- **scale ≤ 0.01**: ようやく surface code の本来挙動 (単調減少)
- **scale = 0 (heating-free)**: 上限値、 これより良くはなれない

つまり **現実的なハードウェア改良パス (= scale を小さくする) でも、 単一 LIZ で
distance protection を活かすには heating を 100 分の 1 以下** に抑える必要がある。
これは厳しい目標。

代替パス: **multi-LIZ で transport 回数自体を減らす** → 元の phonon が下がる
→ 効果的な scale が小さくなったのと等価。 こちらの方が現実的なルート。

## ファイル一覧

| ファイル | 内容 |
|---------|------|
| `plot_comparison.png` | 全 scale 重ね合わせの主プロット |
| `plot_scale0.5_50k.png`, `data_scale0.5_50k.json` | scale=0.5 個別 |
| `plot_scale0.1_50k.png`, `data_scale0.1_50k.json` | scale=0.1 個別 |
| `plot_scale0.01_50k.png`, `data_scale0.01_50k.json` | scale=0.01 個別 |

## 再生成方法

```bash
python experiments/exp_distance_scan.py --distances 3 5 7 --shots 50000 --phonon-scale 0.5
python experiments/exp_distance_scan.py --distances 3 5 7 --shots 50000 --phonon-scale 0.1
python experiments/exp_distance_scan.py --distances 3 5 7 --shots 50000 --phonon-scale 0.01
```

出力先は `results/noise_injection/distance_scan_scale<α>_<timestamp>.{png,json}`。
本ディレクトリへの copy + comparison plot は手動。
