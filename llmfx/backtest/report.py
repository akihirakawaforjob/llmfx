"""バックテスト結果の Markdown レポート生成.

成績だけでなく、「その成績で目標月利に届くのか」「そのとき必要な
リスク率と破産確率はいくつか」まで同じレポートに載せる。
成績表だけを見て資金を突っ込むのを防ぐため。
"""

from __future__ import annotations

import math
from pathlib import Path

from ..domain.risk import kelly_fraction, monte_carlo, required_risk_fraction
from .engine import BacktestResult
from .metrics import PerformanceStats, trades_frame


def feasibility_analysis(stats: PerformanceStats, ruin_threshold: float = 0.5) -> dict:
    """バックテストの勝率と平均 R から、目標月利の達成条件を逆算する。"""
    if stats.trades < 10:
        return {"status": "insufficient_trades", "trades": stats.trades}

    win_r = stats.avg_win_r or 0.0
    loss_r = stats.avg_loss_r or 1.0
    trades_per_month = stats.trades_per_month or 0.0

    kelly = kelly_fraction(stats.win_rate, win_r, loss_r)
    required = required_risk_fraction(
        monthly_target=stats.monthly_target,
        trades_per_month=trades_per_month,
        win_rate=stats.win_rate,
        win_r=win_r,
        loss_r=loss_r,
    )

    result: dict = {
        "status": "ok",
        "win_rate": stats.win_rate,
        "avg_win_r": win_r,
        "avg_loss_r": loss_r,
        "trades_per_month": trades_per_month,
        "expectancy_r": stats.expectancy_r,
        "kelly_fraction": kelly,
        "required_risk_fraction": required,
        "monthly_target": stats.monthly_target,
    }

    if required is None:
        result["status"] = "unreachable"
        return result

    result["monte_carlo"] = monte_carlo(
        risk_fraction=required,
        win_rate=stats.win_rate,
        win_r=win_r,
        loss_r=loss_r,
        trades_per_month=trades_per_month,
        months=12,
        monthly_target=stats.monthly_target,
        ruin_threshold=ruin_threshold,
    ).to_dict()
    return result


def _share(cost: float, gross: float) -> str:
    """コストがコスト差引前の損益の何割かを表す。損益がマイナスなら比率は無意味。"""
    if gross <= 0:
        return "—"
    return f"{cost / gross:.1%}"


def render_report(result: BacktestResult, stats: PerformanceStats) -> str:
    cfg = result.config
    lines: list[str] = []
    add = lines.append

    add("# バックテストレポート")
    add("")
    add(f"- 銘柄 / 時間足: **{cfg.instrument.symbol} {cfg.instrument.granularity}**")
    if result.start_time and result.end_time:
        add(
            f"- 期間: {result.start_time:%Y-%m-%d} 〜 {result.end_time:%Y-%m-%d} "
            f"({stats.months:.1f} ヶ月 / {result.bars:,} 本)"
        )
    add(f"- 1 トレードあたりリスク: {cfg.risk.risk_per_trade:.2%}")
    add(f"- 最小リスクリワード: {cfg.entry.min_rr:.2f}")
    spread_desc = f"{cfg.execution.spread_pips} pips"
    if cfg.execution.spread_bps:
        spread_desc += f" + {cfg.execution.spread_bps} bp"
    add(f"- 約定モデル: {cfg.execution.entry_mode} / スプレッド {spread_desc}")
    if cfg.entry.higher_timeframe:
        detail = f"上位足 {cfg.entry.higher_timeframe}"
        if cfg.entry.require_htf_alignment:
            detail += " / 方向一致"
        if cfg.entry.htf_proximity_atr:
            detail += f" / 極値から {cfg.entry.htf_proximity_atr} ATR 以内"
        add(f"- {detail}")
    if cfg.execution.commission_bps:
        add(f"- 取引手数料: 片道 {cfg.execution.commission_bps} bp(往復 {cfg.execution.commission_bps * 2} bp)")
    if cfg.execution.daily_holding_cost_bps:
        annual = cfg.execution.daily_holding_cost_bps * 365 / 100.0
        add(
            f"- 建玉管理料: {cfg.execution.daily_holding_cost_bps} bp/日 "
            f"(年率換算 約 {annual:.1f}%)"
        )
    if result.halt_reason:
        add(f"- ⚠️ **{result.halt_reason}**")
    add("")

    add("## 成績サマリー")
    add("")
    add("| 指標 | 値 |")
    add("| --- | ---: |")
    add(f"| 初期資金 | {stats.initial_equity:,.0f} |")
    add(f"| 最終資金 | {stats.final_equity:,.0f} |")
    add(f"| 総リターン | {stats.total_return:+.2%} |")
    add(f"| 幾何平均月利 | {stats.geometric_monthly_return:+.2%} |")
    add(f"| 目標月利 | {stats.monthly_target:.2f} 倍 ({stats.monthly_target - 1:+.0%}) |")
    add(f"| 目標達成月 | {stats.months_hitting_target} / {stats.months_total} ヶ月 |")
    add(f"| トレード数 | {stats.trades} ({stats.trades_per_month:.1f} 回/月) |")
    add(f"| 勝率 | {stats.win_rate:.1%} |")
    add(f"| 平均利益 | {stats.avg_win_r:.2f} R |")
    add(f"| 平均損失 | {stats.avg_loss_r:.2f} R |")
    add(f"| 期待値 | {stats.expectancy_r:+.3f} R / トレード |")
    add(f"| プロフィットファクター | {_fmt_pf(stats.profit_factor)} |")
    add(f"| 最大ドローダウン | {stats.max_drawdown:.2%} |")
    add(f"| 最大連敗 | {stats.max_consecutive_losses} |")
    add(f"| シャープレシオ(年率換算) | {stats.sharpe_daily:.2f} |")
    add(f"| 平均保有本数 | {stats.avg_bars_held:.1f} |")
    add("")

    commission = sum(t.commission_paid for t in result.trades)
    holding = sum(t.holding_cost_paid for t in result.trades)
    if commission or holding:
        net = stats.final_equity - stats.initial_equity
        gross = net + commission + holding
        add("## 取引コスト")
        add("")
        add("価格差だけで見た損益から、実際に引かれた費用を差し引いた内訳。")
        add("")
        add("| 項目 | 金額 | 純損益に対する比 |")
        add("| --- | ---: | ---: |")
        add(f"| コスト差引前の損益 | {gross:+,.0f} | — |")
        add(f"| 手数料 | {-commission:+,.0f} | {_share(commission, gross)} |")
        add(f"| 建玉管理料 | {-holding:+,.0f} | {_share(holding, gross)} |")
        add(f"| **手取り** | **{net:+,.0f}** | — |")
        add("")
        if stats.trades:
            add(
                f"1 トレードあたり平均 {(commission + holding) / stats.trades:,.0f} "
                f"(手数料 {commission / stats.trades:,.0f} / "
                f"建玉管理料 {holding / stats.trades:,.0f})"
            )
            add("")
        if gross > 0 and (commission + holding) / gross >= 0.3:
            add(
                "> ⚠️ コストがコスト差引前の損益の 3 割以上を食っています。"
                "保有期間の短縮か、より値幅の大きい足での検証を検討してください。"
            )
            add("")

    add("## シグナルの内訳")
    add("")
    add(f"- 生成されたシグナル: {result.signals_generated}")
    add(f"- 実際にエントリー: {result.signals_taken}")
    if result.gate_rejections:
        add(f"- LLM による見送り: {result.gate_rejections}")
    if result.rejections:
        add("- フィルタで却下されたダウ転換:")
        for reason, count in sorted(result.rejections.items(), key=lambda kv: -kv[1]):
            add(f"  - `{reason}`: {count}")
    add("")

    if stats.exit_reasons:
        add("## 決済理由")
        add("")
        add("| 理由 | 件数 |")
        add("| --- | ---: |")
        for reason, count in stats.exit_reasons.items():
            add(f"| {reason} | {count} |")
        add("")

    if stats.target_sources:
        add("## 利確目標の根拠")
        add("")
        add("| 根拠 | 件数 |")
        add("| --- | ---: |")
        for source, count in stats.target_sources.items():
            add(f"| {source} | {count} |")
        add("")

    if stats.monthly_returns:
        add("## 月次リターン")
        add("")
        add("| 月 | リターン | 目標達成 |")
        add("| --- | ---: | :---: |")
        for month, value in stats.monthly_returns.items():
            hit = "✅" if (1.0 + value) >= stats.monthly_target else "—"
            add(f"| {month} | {value:+.2%} | {hit} |")
        add("")

    add(_render_feasibility(stats))
    return "\n".join(lines)


def _render_feasibility(stats: PerformanceStats) -> str:
    analysis = feasibility_analysis(stats)
    lines: list[str] = ["## 目標月利の実現可能性", ""]

    if analysis["status"] == "insufficient_trades":
        lines.append(
            f"トレード数が {analysis['trades']} 件しかなく、統計的な評価に耐えません。"
            "期間を延ばすか、フィルタを緩めてサンプルを増やしてください。"
        )
        return "\n".join(lines)

    target = analysis["monthly_target"]
    lines.append(
        f"バックテストの勝率 {analysis['win_rate']:.1%}、平均利益 "
        f"{analysis['avg_win_r']:.2f}R、平均損失 {analysis['avg_loss_r']:.2f}R、"
        f"月間 {analysis['trades_per_month']:.1f} トレードを前提に、"
        f"目標 **月利 {target:.2f} 倍** の達成条件を逆算します。"
    )
    lines.append("")

    if analysis["expectancy_r"] <= 0:
        lines.append(
            "⛔ **この戦略設定は期待値がマイナスです。** リスク率をどう調整しても "
            "資産は減り続けます。目標月利の議論以前に、エントリー条件そのものを"
            "見直す必要があります。"
        )
        return "\n".join(lines)

    if analysis["status"] == "unreachable":
        lines.append(
            f"⛔ **この成績では月利 {target:.2f} 倍は数学的に到達できません。** "
            "ケリー基準(対数成長が最大になる点)でリスクを張っても、"
            "期待成長率が目標に届きません。到達するには勝率・平均 R・"
            "トレード頻度のいずれかを改善する必要があります。"
        )
        lines.append("")
        lines.append(f"- ケリー基準リスク率: {analysis['kelly_fraction']:.2%}")
        return "\n".join(lines)

    required = analysis["required_risk_fraction"]
    mc = analysis["monte_carlo"]
    lines.append(f"- 必要な 1 トレードあたりリスク率: **{required:.2%}**")
    lines.append(f"- ケリー基準リスク率: {analysis['kelly_fraction']:.2%}")
    lines.append(
        f"- ケリー比: {required / analysis['kelly_fraction']:.2f} 倍"
        if analysis["kelly_fraction"] > 0
        else "- ケリー基準: 算出不能"
    )
    lines.append("")
    lines.append(f"### そのリスク率で 12 ヶ月運用した場合(モンテカルロ {mc['paths']:,} 経路)")
    lines.append("")
    lines.append("| 指標 | 値 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 資金半減(破産)確率 | **{mc['prob_ruin']:.1%}** |")
    lines.append(f"| 各月が目標を達成する確率 | {mc['prob_hit_target_monthly']:.1%} |")
    lines.append(f"| 12 ヶ月平均で目標達成する確率 | {mc['prob_target_on_average']:.1%} |")
    lines.append(f"| 月次リターン中央値 | {mc['median_monthly_return']:+.2%} |")
    lines.append(f"| 12 ヶ月後の資産倍率(下位 5%) | {mc['p05_final_multiple']:.2f} 倍 |")
    lines.append(f"| 12 ヶ月後の資産倍率(中央値) | {mc['median_final_multiple']:.2f} 倍 |")
    lines.append(f"| 12 ヶ月後の資産倍率(上位 5%) | {mc['p95_final_multiple']:.2f} 倍 |")
    lines.append(f"| 最大ドローダウン中央値 | {mc['median_max_drawdown']:.1%} |")
    lines.append(f"| 最大ドローダウン最悪値 | {mc['worst_max_drawdown']:.1%} |")
    lines.append("")

    if mc["prob_ruin"] >= 0.10:
        lines.append(
            f"⚠️ **警告: 破産確率が {mc['prob_ruin']:.1%} あります。** "
            "目標月利を達成するために必要なリスク率は、資金を失う確率と"
            "表裏一体です。この数字を受け入れられない場合は、"
            "`risk.monthly_target` を下げるか、戦略の期待値を改善してください。"
        )
    if mc["median_max_drawdown"] >= 0.30:
        lines.append(
            f"⚠️ **警告: 最大ドローダウンの中央値が {mc['median_max_drawdown']:.1%} です"
            f"(最悪値 {mc['worst_max_drawdown']:.1%})。** 破産はしなくても、"
            "資産が半分近くまで削られる局面を平常運転として通過することになります。"
            "この振れ幅に耐えられるかを、資金を入れる前に判断してください。"
        )
    if required > 0.05:
        lines.append(
            f"⚠️ 必要リスク率 {required:.2%} は 1 トレードあたりの許容損失として"
            "かなり大きい水準です。`risk.max_risk_per_trade` の上限に阻まれて"
            "実際にはこのリスク率で運用できない場合、目標には届きません。"
        )
    if analysis["kelly_fraction"] > 0 and required > analysis["kelly_fraction"]:
        lines.append(
            "⚠️ 必要リスク率がケリー基準を超えています。この領域では"
            "リスクを上げるほど期待成長率が **下がり**、破産確率だけが上がります。"
        )
    return "\n".join(lines)


def _fmt_pf(value: float) -> str:
    if math.isinf(value):
        return "∞"
    return f"{value:.2f}"


def write_report(
    result: BacktestResult,
    stats: PerformanceStats,
    report_path: str | Path,
    trades_path: str | Path | None = None,
) -> None:
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(result, stats), encoding="utf-8")

    if trades_path is not None:
        frame = trades_frame(result.trades)
        if not frame.empty:
            trades_target = Path(trades_path)
            trades_target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(trades_target, index=False)
