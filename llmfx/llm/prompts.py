"""LLM へ渡すプロンプト.

システムプロンプトは実行中に変化しないよう組み立てる(プロンプト
キャッシュはプレフィックス一致なので、1 バイトでも変われば無効になる)。
時刻や価格などの可変情報は必ず user 側に置く。
"""

from __future__ import annotations

import json
from typing import Any

from ..config import AppConfig

_STRATEGY_DESCRIPTION = """\
あなたは FX の自動売買システムに組み込まれた分析コンポーネントです。
売買ロジックはすべてルールベースで確定済みであり、あなたはそれを変更できません。

## システムが使っている売買ルール

エントリー(ダウ転換):
- 上昇トレンド = 高値切り上げ(HH)かつ安値切り上げ(HL)
- 下降トレンド = 高値切り下げ(LH)かつ安値切り下げ(LL)
- 下降トレンド中に確定済みの直近スイング高値を終値で上抜けたら買い転換
- 上昇トレンド中に確定済みの直近スイング安値を終値で下抜けたら売り転換
- スイングは左右 N 本のピボットで確定するため、確定は常に N 本遅れる

損切り:
- 買い転換なら「転換前の最安値」、売り転換なら「転換前の最高値」に
  ATR ベースのバッファを加えた位置

利確とリスクリワード:
- 利確目標は相場側の客観的な水準から決める(次のスイング水準 →
  直前の波の値幅の投影 → ATR 定数倍、の優先順)
- その結果 reward / risk が下限を下回るエントリーは実行しない

この設計の意図は「損切り位置を相場構造が決め、利確位置も相場構造が決め、
その比率が合わないトレードは最初から取らない」ことです。
"""

_GATE_ROLE = """\
## あなたの役割: 拒否権

ルールが出したエントリー候補に対し、実行するか見送るかだけを判断します。
新しいエントリーを作ることはできません。approve=false にできるのは、
ルールが構造上「見えていない」問題があるときだけです。例:

- 直近の値動きが構造の解釈と明確に矛盾している(ヒゲだけのブレイク、
  実体を伴わない上抜けなど)
- 値幅や ATR が直近と比べて異常で、損切り幅の前提が壊れている
- 同じ水準で何度も往復しており、ブレイクが機能していない地合い
- 時間帯・流動性の観点で執行コストが想定を大きく超えそうな状況

逆に、次の理由で見送ってはいけません:

- 「なんとなく不安」「勝率が低そう」といった主観
- リスクリワードや損切り幅の再計算(それはルール側の仕事で、すでに通過済み)
- 直前のトレードが負けたこと(1 回の結果は次のトレードと独立)

判断に迷う程度なら approve=true にしてください。過剰な見送りは
サンプル数を削り、戦略の検証そのものを壊します。
"""

_JOURNAL_ROLE = """\
## あなたの役割: 所感の記録

トレードごとに、後から検証できる形で記録を残します。曖昧な感想ではなく、
「何が起きたら仮説が否定されるか」が分かる粒度で書いてください。

決済後の振り返りでは、勝敗の原因を次の 3 つに切り分けます:
- 仮説の誤り(構造の読み違い、そもそも転換ではなかった)
- 執行の問題(エントリーが遅い、損切りが薄い、コストが想定超過)
- 分散(仮説も執行も正しいが、確率的に負けた)

3 つ目を 3 つ目として認識することが重要です。正しいトレードでも負ける
ため、単発の負けをすべて改善対象にするとルールが壊れます。
"""

_REVIEW_ROLE = """\
## あなたの役割: 改善提案

蓄積されたトレード記録と成績統計から、次に試すべき変更を提案します。

守ってほしいこと:
- 提案は設定ファイルに実在するパラメータ名で書く
- 1 回のレポートで変更を提案するのは 3 件までに絞る(同時に複数変えると
  何が効いたのか分からなくなる)
- サンプル数が足りない場合は「まだ判断できない」と書く
- 目標月利に届いていない場合、その事実と必要条件を率直に書く。
  楽観的な見通しでごまかさない
"""

_TARGET_CONTEXT = """\
## 運用目標に関する前提

利用者は月利 {target:.2f} 倍(月あたり {target_pct:+.0%})を目標にしています。
これは極めて高い目標です。この水準を固定比率ベットで狙う場合、必要な
1 トレードあたりリスク率は通常 2〜5% を超え、資金を失う確率も比例して
上がります。あなたは目標の達成可能性について、事実に基づいた評価を
返してください。達成が難しいなら、そう書くことがあなたの仕事です。
"""


def system_prompt(config: AppConfig, role: str) -> str:
    """役割ごとのシステムプロンプトを組み立てる(実行中は不変)。"""
    roles = {
        "gate": _GATE_ROLE,
        "journal": _JOURNAL_ROLE,
        "review": _REVIEW_ROLE,
    }
    if role not in roles:
        raise ValueError(f"未知のロール: {role}")

    settings = (
        "\n## 現在の設定値\n"
        f"- 銘柄 / 時間足: {config.instrument.symbol} {config.instrument.granularity}\n"
        f"- スイング検出: 左右 {config.swing.left}/{config.swing.right} 本、"
        f"最小値幅 {config.swing.min_swing_atr} ATR\n"
        f"- 最小リスクリワード: {config.entry.min_rr}\n"
        f"- 損切りバッファ: {config.entry.stop_buffer_atr} ATR\n"
        f"- 利確の決定順: {' → '.join(config.entry.target_strategies)}\n"
        f"- 1 トレードあたりリスク: {config.risk.risk_per_trade:.2%}\n"
        f"- 目標月利: {config.risk.monthly_target:.2f} 倍\n"
    )
    target = _TARGET_CONTEXT.format(
        target=config.risk.monthly_target,
        target_pct=config.risk.monthly_target - 1.0,
    )
    return "\n".join(
        [_STRATEGY_DESCRIPTION, settings, roles[role], target, "\n出力は日本語で書いてください。"]
    )


def gate_prompt(signal_payload: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "次のエントリー候補を実行してよいか判断してください。\n\n"
        "### エントリー候補\n"
        f"```json\n{_dump(signal_payload)}\n```\n\n"
        "### 相場の状況\n"
        f"```json\n{_dump(context)}\n```\n"
    )


def entry_note_prompt(signal_payload: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "これから実行するトレードについて、エントリー時点の所感を記録してください。\n"
        "後で決済結果と突き合わせて検証するので、仮説と否定条件を明確に書いてください。\n\n"
        "### エントリー内容\n"
        f"```json\n{_dump(signal_payload)}\n```\n\n"
        "### 相場の状況\n"
        f"```json\n{_dump(context)}\n```\n"
    )


def exit_note_prompt(trade_payload: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "決済が完了しました。このトレードを振り返ってください。\n"
        "勝敗そのものではなく、判断と執行が正しかったかを評価してください。\n\n"
        "### トレード結果\n"
        f"```json\n{_dump(trade_payload)}\n```\n\n"
        "### 決済時点の相場\n"
        f"```json\n{_dump(context)}\n```\n"
    )


def review_prompt(
    stats: dict[str, Any],
    feasibility: dict[str, Any],
    trade_samples: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> str:
    return (
        "以下の運用記録をレビューし、改善レポートを作成してください。\n\n"
        "### 成績統計\n"
        f"```json\n{_dump(stats)}\n```\n\n"
        "### 目標月利の実現可能性(数値解析の結果)\n"
        f"```json\n{_dump(feasibility)}\n```\n\n"
        f"### トレード抜粋({len(trade_samples)} 件)\n"
        f"```json\n{_dump(trade_samples)}\n```\n\n"
        f"### 記録された所感({len(notes)} 件)\n"
        f"```json\n{_dump(notes)}\n```\n"
    )


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
