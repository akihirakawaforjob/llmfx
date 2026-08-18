# llmfx — 開発ガイド

ダウ理論の「転換」を売買根拠とする FX 自動売買システム。
このファイルは Claude Code が作業を引き継ぐための前提資料。

## このプロジェクトで守ること

### 1. 売買判断は LLM に渡さない

エントリー・損切り・利確の判定はすべてルールベース(`llmfx/domain/`)。
LLM の役割は 3 つだけに限定する:

- **拒否権** — ルールが出したシグナルを見送るかだけを決める。新規に作らせない
- **所感の記録** — エントリー時と決済時の言語化
- **改善提案** — 蓄積した記録からパラメータ変更案を出す

理由: 判断が非決定的になるとバックテストと実運用が乖離し、検証が成立しない。
この境界を越える変更を提案するときは、必ず理由を添えて確認を取ること。

### 2. 先読み(look-ahead bias)を持ち込まない

- スイングは左右 N 本のピボットで確定するため、確定は常に N 本遅れる。
  `confirmed_index <= 現在バー` のものだけを参照する
- シグナル判定は確定足の終値のみ。既定の約定は翌足の始値
- 同一足で損切りと利確の両方に触れたら、必ず損切りが先に約定した扱い
- 逆指値・成行にはスリッページを乗せ、指値には乗せない

`tests/test_backtest.py::test_no_lookahead_truncating_data_does_not_change_past_trades`
がこれを守っている。データを途中で打ち切っても過去のトレードが 1 件も
変わらないことを検証しているので、このテストは絶対に緩めない。

### 3. 約定ロジックを二重に書かない

`llmfx/execution/fills.py` をバックテストとペーパー取引の両方が使う。
どちらか一方だけを直すと、検証した挙動が実運用で再現しなくなる。

### 4. 利確目標を RR が通るまで選び直さない

`entry.target_strategies` は「水準を出せた最初の戦略」を採用し、
その結果の RR で合否を決める。RR 2.0 を超えるまで戦略を渡り歩く実装は、
利確位置の後付けであり、フィルタが何も選別しなくなる。

## 確定済みの仕様(利用者と合意済み)

| 項目 | 決定 | 設定キー |
| --- | --- | --- |
| エントリー | 下降トレンド中に確定済み直近スイング高値を終値で上抜けた瞬間(売りは鏡像) | — |
| 損切り | **転換前の「深い谷」**。上抜け対象スイングから現在までの最安値 + ATRバッファ | `entry.stop_basis_mode: trend_extreme` |
| リスクリワード | `reward / risk >= 2.0` 未満は破棄 | `entry.min_rr: 2.0` |
| 利確目標 | 反転させたトレンドの起点 | `entry.target_strategies: [trend_origin, ...]` |
| 目標月利 | 1.4 倍。**達成必須のラインではなく、目指す方向** | `risk.monthly_target: 1.4` |

損切りを「直近の押し安値」にする案(`recent_swing`)も実装済みだが、
利用者は **深い谷(`trend_extreme`)で確定** と回答している。変更しないこと。

### 利確目標が trend_origin である理由

損切りを転換前の極値に置くと、リスク幅は必ず「直前の波の全長」になる。
そのため利確目標が波 2 本分より遠くないと RR 2.0 に届かない。実測では:

| 利確目標 | RR 中央値 | RR >= 2.0 の割合 |
| --- | ---: | ---: |
| `structure`(最も近い抵抗) | 0.55 | 1.2% |
| `measured_move`(波 1 本分) | 0.85 | 約 2% |
| `trend_origin`(トレンド起点) | 1.98 | 49.4% |

## 目標月利の扱い

月利 1.4 倍は非常に高い。**数字を良く見せない**こと。

`backtest` のレポート末尾と `llmfx target` は、目標達成に必要なリスク率を
対数成長の式から逆算し、モンテカルロで破産確率と最大ドローダウンを出す。
「到達不能」と出たらそう表示する。楽観的な見通しに書き換えない。

## コマンド

```bash
pip install -r requirements.txt
python -m pytest -q                     # テスト(97 件)

# 楽天 MT4 などからエクスポートした CSV を取り込む
python -m llmfx.cli data import-mt4 --in <MT4のCSV> --out data/usdjpy_m15.csv --server-tz-offset 2

python -m llmfx.cli backtest --config configs/default.yaml --data data/usdjpy_m15.csv --journal data/journal.sqlite
python -m llmfx.cli diagnose --config configs/default.yaml --data data/usdjpy_m15.csv
python -m llmfx.cli paper    --config configs/default.yaml --replay data/usdjpy_m15.csv
python -m llmfx.cli review   --journal data/journal.sqlite --llm --out out/review.md
python -m llmfx.cli target   --win-rate 0.35 --win-r 2.3 --loss-r 0.7 --trades-per-month 17
```

`ANTHROPIC_API_KEY` が無くても全機能が動く(LLM 層が自動的に無効化される)。

## 利用者の運用環境

- 普段の取引は **楽天FX**。データは楽天 MT4 のヒストリーセンターから CSV エクスポート
- **楽天FX には Python から直接発注できる公開 API が無い**。自動売買の正規ルートは MT4 の EA(MQL4)
- 実弾へ繋ぐときは、判断を Python 側に残したまま MT4 には発注だけさせるブリッジ構成にする。
  戦略ごと MQL4 へ移植すると LLM 層が使えなくなる

## 現在地と次にやること

完了:
- ダウ転換の検出、損切り・利確・RR フィルタ
- バックテスト(先読み防止つき)、成績評価、Markdown レポート
- ペーパー取引(CSV 再生 / OANDA デモ)
- LLM 層(拒否権・所感・改善レポート)
- MT4 / MT5 の CSV 取り込み

次:
1. 楽天 MT4 から実データを取得してバックテスト(合成データの成績には意味が無い)
2. 実データで `stop_basis_mode` と `target_strategies` を全比較し、数字で確認
3. 実測の勝率・平均 R から月利目標の必要リスク率と破産確率を再計算
4. OANDA 接続の実地検証(未テスト。資格情報が無い環境で実装したため)

未検証・既知の制約:
- **OANDA への実接続は未テスト**。初回はデモ口座かつ最小ロットで確認すること
- 本番口座は `OANDA_ENV=live` かつ `allow_live=True` の両方を明示しない限り動かない
- 同時保有は 1 銘柄・1 建玉。複数通貨ペアは未実装
- スワップポイント未計上。日をまたぐポジションでは実損益とずれる
