"""バックテスト結果の統計量.

R 倍数(1R = エントリー時に許容した損失額)を中心に据える。金額ベースの
数字は口座残高とロットサイズに依存して比較不能になるが、R 倍数は
戦略そのものの性能を表すため。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..domain.types import Trade
from .engine import BacktestResult


@dataclass
class PerformanceStats:
    initial_equity: float
    final_equity: float
    total_return: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    sharpe_daily: float
    months: float
    trades_per_month: float
    geometric_monthly_return: float
    monthly_target: float
    months_hitting_target: int
    months_total: int
    avg_bars_held: float
    largest_win_r: float
    largest_loss_r: float
    max_consecutive_losses: int
    exit_reasons: dict[str, int] = field(default_factory=dict)
    target_sources: dict[str, int] = field(default_factory=dict)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "expectancy_r": self.expectancy_r,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "sharpe_daily": self.sharpe_daily,
            "months": self.months,
            "trades_per_month": self.trades_per_month,
            "geometric_monthly_return": self.geometric_monthly_return,
            "monthly_target": self.monthly_target,
            "months_hitting_target": self.months_hitting_target,
            "months_total": self.months_total,
            "avg_bars_held": self.avg_bars_held,
            "largest_win_r": self.largest_win_r,
            "largest_loss_r": self.largest_loss_r,
            "max_consecutive_losses": self.max_consecutive_losses,
            "exit_reasons": self.exit_reasons,
            "target_sources": self.target_sources,
            "monthly_returns": self.monthly_returns,
        }


def equity_frame(result: BacktestResult) -> pd.DataFrame:
    if not result.equity_curve:
        return pd.DataFrame(columns=["equity", "realized_equity"])
    frame = pd.DataFrame(
        {
            "time": [p.time for p in result.equity_curve],
            "equity": [p.equity for p in result.equity_curve],
            "realized_equity": [p.realized_equity for p in result.equity_curve],
        }
    ).set_index("time")
    return frame


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side.value,
                "units": t.units,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "pnl": t.pnl,
                "r_multiple": t.r_multiple,
                "rr_at_entry": t.rr_at_entry,
                "target_source": t.target_source,
                "exit_reason": t.exit_reason.value,
                "bars_held": t.bars_held,
                "equity_after": t.equity_after,
            }
            for t in trades
        ]
    )


def compute_stats(result: BacktestResult) -> PerformanceStats:
    cfg = result.config
    initial = cfg.risk.initial_equity
    curve = equity_frame(result)
    final = float(curve["equity"].iloc[-1]) if not curve.empty else initial

    trades = result.trades
    r_values = np.array([t.r_multiple for t in trades], dtype=float)
    wins = r_values[r_values > 0]
    losses = r_values[r_values <= 0]

    gross_profit = float(sum(t.pnl for t in trades if t.pnl > 0))
    gross_loss = float(-sum(t.pnl for t in trades if t.pnl < 0))

    max_dd, dd_duration = _max_drawdown(curve["equity"] if not curve.empty else pd.Series(dtype=float))
    monthly = _monthly_returns(curve)
    months_total = len(monthly)
    target = cfg.risk.monthly_target
    months_hit = int(sum(1 for v in monthly.values() if (1.0 + v) >= target))

    span_days = _span_days(result)
    months = span_days / 30.4375 if span_days > 0 else 0.0
    geo_monthly = (
        (final / initial) ** (1.0 / months) - 1.0 if months > 0 and final > 0 and initial > 0 else 0.0
    )

    return PerformanceStats(
        initial_equity=initial,
        final_equity=final,
        total_return=(final / initial - 1.0) if initial > 0 else 0.0,
        trades=len(trades),
        wins=int(len(wins)),
        losses=int(len(losses)),
        win_rate=float(len(wins) / len(trades)) if trades else 0.0,
        avg_win_r=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_r=float(abs(losses.mean())) if len(losses) else 0.0,
        expectancy_r=float(r_values.mean()) if len(r_values) else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else math.inf if gross_profit > 0 else 0.0,
        max_drawdown=max_dd,
        max_drawdown_duration_bars=dd_duration,
        sharpe_daily=_sharpe(curve),
        months=months,
        trades_per_month=(len(trades) / months) if months > 0 else 0.0,
        geometric_monthly_return=geo_monthly,
        monthly_target=target,
        months_hitting_target=months_hit,
        months_total=months_total,
        avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        largest_win_r=float(wins.max()) if len(wins) else 0.0,
        largest_loss_r=float(losses.min()) if len(losses) else 0.0,
        max_consecutive_losses=_max_consecutive_losses(trades),
        exit_reasons=_count(t.exit_reason.value for t in trades),
        target_sources=_count(t.target_source for t in trades),
        monthly_returns=monthly,
    )


# ----------------------------------------------------------------------
def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _span_days(result: BacktestResult) -> float:
    if result.start_time is None or result.end_time is None:
        return 0.0
    return (result.end_time - result.start_time).total_seconds() / 86400.0


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    if equity.empty:
        return 0.0, 0
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    max_dd = float(drawdown.max())

    duration = 0
    longest = 0
    for value in drawdown.to_numpy():
        if value > 1e-12:
            duration += 1
            longest = max(longest, duration)
        else:
            duration = 0
    return max_dd, longest


def _monthly_returns(curve: pd.DataFrame) -> dict[str, float]:
    if curve.empty:
        return {}
    monthly = curve["equity"].resample("ME").last().dropna()
    if monthly.empty:
        return {}
    first_equity = float(curve["equity"].iloc[0])
    values = monthly.to_numpy()
    previous = np.concatenate([[first_equity], values[:-1]])
    returns = values / np.where(previous == 0, np.nan, previous) - 1.0
    return {
        index.strftime("%Y-%m"): float(value)
        for index, value in zip(monthly.index, returns)
        if not math.isnan(value)
    }


def _sharpe(curve: pd.DataFrame) -> float:
    if curve.empty:
        return 0.0
    daily = curve["equity"].resample("D").last().dropna()
    if len(daily) < 3:
        return 0.0
    returns = daily.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(252))


def _max_consecutive_losses(trades: list[Trade]) -> int:
    streak = 0
    longest = 0
    for trade in trades:
        if trade.pnl <= 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest
