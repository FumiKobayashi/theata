## 概要

1次元のQCCD構成に対する時間シミュレータの開発プロジェクトです。

## 構成
QCCD native circuit: QCCD native gate setで表された量子回路
- estimator.py: 入力のQCCD native circuitをparseして時間見積もりを行うファイル
- parameters.py: 各種操作の時間パラメータファイル
- qccd_parser.py: QCCD native gate setの言語を定義するファイル

## 変更履歴

(12/15) qccdの言語定義を変更
