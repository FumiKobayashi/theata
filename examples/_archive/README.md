# examples/_archive

過去の `examples/` スナップショットをここに保存する。各サブディレクトリは
**日付 + 主要な設定差** で命名する: `YYYYMMDD_<discriminator>/`。

`examples/` のルート直下が常に「現時点の canonical 設定」を表すのに対し、
`_archive/` 配下は「過去の特定設定での結果」を保持する。

## 現在のアーカイブ

- [`20260620_split_even/`](20260620_split_even/) — placement strategy が
  split_even だった時代の examples。greedy_alt との比較対象として保持。

## アーカイブのライフサイクル

新しい設定 / アルゴリズム変更で `examples/` を再生成する前に、
古い内容を `_archive/YYYYMMDD_<short_label>/` に退避する。
これにより:

- canonical との数値比較が容易
- 「いつ、何が、何故変わったか」のトレーサビリティ
- 論文・スライドで具体的な比較データを参照可能

サイズが大きくなりすぎたら `.gitignore` に追加して切り離す判断もあり得る。
当面は git で追跡。
