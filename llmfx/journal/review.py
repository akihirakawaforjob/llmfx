"""改善レポートの生成.

蓄積したトレード記録 + LLM の所感 + 数値解析(目標達成に必要な条件)を
まとめて Claude に渡し、次に試すべき変更を提案させる。出力は Markdown で
保存され、そのまま人間のレビューに回せる。

LLM が使えない場合でも、数値部分だけのレポートは必ず生成される。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..config import AppConfig
from ..llm.client import LLMClient, LLMUnavailable
from ..llm.prompts import review_prompt, system_prompt
from ..llm.schemas import ReviewReport
from .store import JournalStore

logger = logging.getLogger(__name__)


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """記録から成績統計を組み立てる(バックテスト統計とは独立に計算)。"""
    if not trades:
        return {"trades": 0}

    r_values = [float(t["r_multiple"]) for t in trades]
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]
    gross_profit = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    gross_loss = -sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0)

    by_reason: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for trade in trades:
        by_reason[trade["exit_reason"]] = by_reason.get(trade["exit_reason"], 0) + 1
        by_source[trade["target_source"]] = by_source.get(trade["target_source"], 0) + 1

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win_r": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss_r": abs(sum(losses) / len(losses)) if losses else 0.0,
        "expectancy_r": sum(r_values) / len(r_values),
        "total_r": sum(r_values),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "exit_reasons": by_reason,
        "target_sources": by_source,
        "best_trade_r": max(r_values),
        "worst_trade_r": min(r_values),
    }


def build_review(
    store: JournalStore,
    config: AppConfig,
    client: LLMClient | None = None,
    since: datetime | None = None,
    limit: int = 300,
    sample_size: int = 40,
) -> tuple[str, dict[str, Any]]:
    """(Markdown, 構造化レポート) を返す。"""
    trades = store.recent_trades(limit=limit, since=since)
    stats = summarize_trades(trades)
    feasibility = _feasibility(stats, config, trades)

    header = _render_numeric_section(stats, feasibility, config, since)

    if not trades:
        return header + "\n\n記録されたトレードがありません。\n", {}

    if client is None or not client.available:
        reason = client.unavailable_reason if client else "LLM クライアント未指定"
        return (
            header
            + f"\n\n> LLM による改善提案は生成されませんでした({reason})。\n"
            + "> `ANTHROPIC_API_KEY` を設定したうえで `--llm` を付けて実行すると提案が付きます。\n",
            {},
        )

    samples = _select_samples(trades, sample_size)
    notes = store.notes(limit=limit, since=since)

    try:
        report = client.structured(
            system=system_prompt(config, "review"),
            user=review_prompt(stats, feasibility, samples, notes),
            schema=ReviewReport,
            use_cache=False,
            max_tokens=16000,
        )
    except LLMUnavailable as exc:
        logger.warning("改善レポートの生成に失敗しました: %s", exc)
        return header + f"\n\n> LLM による改善提案の生成に失敗しました: {exc}\n", {}

    return header + "\n\n" + _render_llm_section(report), report.model_dump()


# ----------------------------------------------------------------------
def _trades_per_month(trades: list[dict[str, Any]]) -> float:
    """記録された実際の期間から月間トレード数を求める。

    ここを概算で済ませると必要リスク率の逆算がずれ、レポートの結論
    そのものが変わってしまうため、実時刻から計算する。
    """
    if len(trades) < 2:
        return float(len(trades))
    times = sorted(datetime.fromisoformat(t["exit_time"]) for t in trades)
    span_days = (times[-1] - times[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return float(len(trades))
    return len(trades) / (span_days / 30.4375)


def _feasibility(
    stats: dict[str, Any], config: AppConfig, trades: list[dict[str, Any]]
) -> dict[str, Any]:
    from ..domain.risk import kelly_fraction, monte_carlo, required_risk_fraction

    if stats.get("trades", 0) < 10:
        return {"status": "insufficient_trades", "trades": stats.get("trades", 0)}

    win_rate = stats["win_rate"]
    win_r = stats["avg_win_r"] or 0.0
    loss_r = stats["avg_loss_r"] or 1.0
    trades_per_month = max(0.1, _trades_per_month(trades))

    required = required_risk_fraction(
        monthly_target=config.risk.monthly_target,
        trades_per_month=trades_per_month,
        win_rate=win_rate,
        win_r=win_r,
        loss_r=loss_r,
    )
    payload: dict[str, Any] = {
        "status": "ok" if required is not None else "unreachable",
        "monthly_target": config.risk.monthly_target,
        "trades_per_month": round(trades_per_month, 2),
        "kelly_fraction": kelly_fraction(win_rate, win_r, loss_r),
        "required_risk_fraction": required,
        "configured_risk_per_trade": config.risk.risk_per_trade,
    }
    if required is not None:
        payload["monte_carlo"] = monte_carlo(
            risk_fraction=required,
            win_rate=win_rate,
            win_r=win_r,
            loss_r=loss_r,
            trades_per_month=trades_per_month,
            months=12,
            monthly_target=config.risk.monthly_target,
        ).to_dict()
    return payload


def _select_samples(trades: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """大負け・大勝ち・直近を優先して抜粋する(全部渡すと文脈が溢れる)。"""
    by_r = sorted(trades, key=lambda t: float(t["r_multiple"]))
    worst = by_r[: size // 3]
    best = by_r[-(size // 3) :]
    recent = trades[: size - len(worst) - len(best)]

    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for trade in [*worst, *best, *recent]:
        if trade["id"] in seen:
            continue
        seen.add(trade["id"])
        selected.append(
            {
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "side": trade["side"],
                "rr_at_entry": trade["rr_at_entry"],
                "target_source": trade["target_source"],
                "r_multiple": trade["r_multiple"],
                "exit_reason": trade["exit_reason"],
                "bars_held": trade["bars_held"],
                "structure": trade.get("structure"),
                "entry_note": trade.get("entry_note"),
                "exit_note": trade.get("exit_note"),
            }
        )
    return selected


def _render_numeric_section(
    stats: dict[str, Any],
    feasibility: dict[str, Any],
    config: AppConfig,
    since: datetime | None,
) -> str:
    lines = ["# 運用レビュー", ""]
    scope = f"{since:%Y-%m-%d} 以降" if since else "全期間"
    lines.append(f"- 対象: {scope}")
    lines.append(f"- 銘柄: {config.instrument.symbol} {config.instrument.granularity}")
    lines.append("")

    if stats.get("trades", 0) == 0:
        return "\n".join(lines)

    lines.append("## 記録された成績")
    lines.append("")
    lines.append("| 指標 | 値 |")
    lines.append("| --- | ---: |")
    lines.append(f"| トレード数 | {stats['trades']} |")
    lines.append(f"| 勝率 | {stats['win_rate']:.1%} |")
    lines.append(f"| 平均利益 | {stats['avg_win_r']:.2f} R |")
    lines.append(f"| 平均損失 | {stats['avg_loss_r']:.2f} R |")
    lines.append(f"| 期待値 | {stats['expectancy_r']:+.3f} R |")
    lines.append(f"| 累計 | {stats['total_r']:+.1f} R |")
    if stats.get("profit_factor"):
        lines.append(f"| プロフィットファクター | {stats['profit_factor']:.2f} |")
    lines.append(f"| 最良 / 最悪 | {stats['best_trade_r']:+.2f}R / {stats['worst_trade_r']:+.2f}R |")
    lines.append("")

    lines.append("## 目標月利に対する現在地")
    lines.append("")
    if feasibility.get("status") == "insufficient_trades":
        lines.append(f"トレード数 {feasibility['trades']} 件では評価できません。")
    elif feasibility.get("status") == "unreachable":
        lines.append(
            f"⛔ 現在の期待値では月利 {feasibility['monthly_target']:.2f} 倍に"
            "到達できません(ケリー点でも成長率が足りません)。"
        )
    else:
        required = feasibility["required_risk_fraction"]
        mc = feasibility.get("monte_carlo", {})
        lines.append(f"- 目標: 月利 {feasibility['monthly_target']:.2f} 倍")
        lines.append(f"- 実測トレード頻度: {feasibility['trades_per_month']:.1f} 回/月")
        lines.append(f"- 必要リスク率: **{required:.2%}** / 設定値: {feasibility['configured_risk_per_trade']:.2%}")
        lines.append(f"- ケリー基準: {feasibility['kelly_fraction']:.2%}")
        if mc:
            lines.append(f"- そのリスク率での資金半減確率: **{mc['prob_ruin']:.1%}**")
            lines.append(f"- 12 ヶ月平均で目標達成する確率: {mc['prob_target_on_average']:.1%}")
    return "\n".join(lines)


def _render_llm_section(report: ReviewReport) -> str:
    lines = ["## LLM による改善提案", ""]
    lines.append(f"**総評**: {report.summary}")
    lines.append("")
    lines.append(f"**最大の問題**: {report.biggest_problem}")
    lines.append("")
    lines.append(f"**目標月利の評価**: {report.monthly_target_assessment}")
    lines.append("")

    for title, items in (
        ("うまくいっている点", report.strengths),
        ("うまくいっていない点", report.weaknesses),
        ("根本原因", report.root_causes),
    ):
        if items:
            lines.append(f"### {title}")
            lines.append("")
            lines.extend(f"- {item}" for item in items)
            lines.append("")

    if report.parameter_suggestions:
        lines.append("### パラメータ変更案")
        lines.append("")
        lines.append("| パラメータ | 現在 | 提案 | 根拠 | 期待効果 | 変更リスク |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for suggestion in report.parameter_suggestions:
            lines.append(
                f"| `{suggestion.parameter}` | {suggestion.current_value} | "
                f"{suggestion.suggested_value} | {suggestion.rationale} | "
                f"{suggestion.expected_effect} | {suggestion.risk_of_change} |"
            )
        lines.append("")

    if report.risk_warnings:
        lines.append("### リスク警告")
        lines.append("")
        lines.extend(f"- ⚠️ {item}" for item in report.risk_warnings)
        lines.append("")

    if report.next_experiments:
        lines.append("### 次に試すこと")
        lines.append("")
        lines.extend(f"{i}. {item}" for i, item in enumerate(report.next_experiments, 1))
        lines.append("")

    return "\n".join(lines)
