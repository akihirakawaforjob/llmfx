"""ポジションサイジング、リスク制限、そして目標月利の実現可能性検証.

目標「月利 1.4 倍(+40%)」は非常に高い。この水準は勝率と RR が同時に
高くない限り、必要リスク率が現実的でない領域に飛ぶ。本モジュールは
その事実を数式とモンテカルロで可視化し、達成に必要な条件と破産確率を
そのまま提示する。数字を丸めたり隠したりはしない。

前提モデル(固定比率ベット):
    勝ち: equity *= (1 + f * R_win)
    負け: equity *= (1 - f * R_loss)
  f       = 1 トレードあたりのリスク率(口座残高比)
  R_win   = 平均利益(R 倍数)
  R_loss  = 平均損失(R 倍数, 通常 1.0)
  N       = 月間トレード数
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# ポジションサイジング
# ----------------------------------------------------------------------
def position_size(
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_price: float,
    quote_to_account_rate: float = 1.0,
    min_units: float = 1.0,
    max_units: float | None = None,
) -> tuple[float, float]:
    """建玉数量と、実際に晒すリスク額を返す。

    FX の 1 通貨単位あたり損益 = 価格差 x クオート通貨→口座通貨レート。
    戻り値: (units, risk_amount)
    """
    if equity <= 0:
        return 0.0, 0.0
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or quote_to_account_rate <= 0:
        return 0.0, 0.0

    risk_amount = equity * risk_fraction
    units = risk_amount / (stop_distance * quote_to_account_rate)
    units = math.floor(units)
    if max_units is not None:
        units = min(units, max_units)
    if units < min_units:
        return 0.0, 0.0
    actual_risk = units * stop_distance * quote_to_account_rate
    return float(units), float(actual_risk)


# ----------------------------------------------------------------------
# 目標月利の逆算
# ----------------------------------------------------------------------
def expected_log_growth(
    risk_fraction: float,
    win_rate: float,
    win_r: float,
    loss_r: float = 1.0,
) -> float:
    """1 トレードあたりの対数期待成長率。破産可能なら -inf。"""
    if risk_fraction <= 0:
        return 0.0
    if risk_fraction * loss_r >= 1.0:
        return float("-inf")  # 1 回の負けで口座が消える
    win_term = math.log1p(risk_fraction * win_r)
    loss_term = math.log1p(-risk_fraction * loss_r)
    return win_rate * win_term + (1.0 - win_rate) * loss_term


def kelly_fraction(win_rate: float, win_r: float, loss_r: float = 1.0) -> float:
    """ケリー基準による最適リスク率。負値は「期待値がマイナス」を意味する。"""
    if win_r <= 0 or loss_r <= 0:
        return 0.0
    b = win_r / loss_r
    f = (win_rate * b - (1.0 - win_rate)) / b
    return f / loss_r


def required_risk_fraction(
    monthly_target: float,
    trades_per_month: float,
    win_rate: float,
    win_r: float,
    loss_r: float = 1.0,
    max_fraction: float = 0.99,
) -> float | None:
    """目標月利に必要な 1 トレードあたりリスク率を二分法で解く。

    どれだけリスクを上げても届かない場合は None を返す(期待値が負、
    あるいは対数成長の上限が目標に満たないケース)。
    """
    if monthly_target <= 1.0:
        return 0.0
    if trades_per_month <= 0:
        return None

    required_growth = math.log(monthly_target) / trades_per_month
    if expected_log_growth(kelly_fraction(win_rate, win_r, loss_r), win_rate, win_r, loss_r) < required_growth:
        # ケリー点(対数成長の最大値)でも届かない = 数学的に到達不能。
        kelly = kelly_fraction(win_rate, win_r, loss_r)
        if kelly <= 0:
            return None
        if expected_log_growth(kelly, win_rate, win_r, loss_r) < required_growth:
            return None

    upper_bound = min(max_fraction, (1.0 / loss_r) * 0.999)
    lo, hi = 1e-6, upper_bound
    if expected_log_growth(hi, win_rate, win_r, loss_r) < required_growth and expected_log_growth(
        kelly_fraction(win_rate, win_r, loss_r), win_rate, win_r, loss_r
    ) < required_growth:
        return None

    # 対数成長はケリー点までは単調増加。探索範囲をそこまでに絞る。
    kelly = kelly_fraction(win_rate, win_r, loss_r)
    if kelly > 0:
        hi = min(hi, kelly)
    if expected_log_growth(hi, win_rate, win_r, loss_r) < required_growth:
        return None

    for _ in range(200):
        mid = (lo + hi) / 2
        if expected_log_growth(mid, win_rate, win_r, loss_r) < required_growth:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------
# モンテカルロ
# ----------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    risk_fraction: float
    win_rate: float
    win_r: float
    loss_r: float
    trades_per_month: float
    months: int
    paths: int
    prob_ruin: float
    """途中で口座が ruin_threshold を割った経路の割合。"""
    prob_hit_target_monthly: float
    """各月の成長率が目標以上になった割合(全経路・全月の平均)。"""
    prob_target_on_average: float
    """期間全体の幾何平均月利が目標以上になった経路の割合。"""
    median_monthly_return: float
    p05_final_multiple: float
    median_final_multiple: float
    p95_final_multiple: float
    median_max_drawdown: float
    worst_max_drawdown: float

    def to_dict(self) -> dict:
        return {
            "risk_fraction": self.risk_fraction,
            "win_rate": self.win_rate,
            "win_r": self.win_r,
            "loss_r": self.loss_r,
            "trades_per_month": self.trades_per_month,
            "months": self.months,
            "paths": self.paths,
            "prob_ruin": self.prob_ruin,
            "prob_hit_target_monthly": self.prob_hit_target_monthly,
            "prob_target_on_average": self.prob_target_on_average,
            "median_monthly_return": self.median_monthly_return,
            "p05_final_multiple": self.p05_final_multiple,
            "median_final_multiple": self.median_final_multiple,
            "p95_final_multiple": self.p95_final_multiple,
            "median_max_drawdown": self.median_max_drawdown,
            "worst_max_drawdown": self.worst_max_drawdown,
        }


def monte_carlo(
    risk_fraction: float,
    win_rate: float,
    win_r: float,
    loss_r: float = 1.0,
    trades_per_month: float = 20,
    months: int = 12,
    paths: int = 20_000,
    monthly_target: float = 1.4,
    ruin_threshold: float = 0.5,
    seed: int = 20260810,
) -> MonteCarloResult:
    """固定比率ベットの資産推移をブートストラップで評価する。

    ruin_threshold は初期資金に対する比率。0.5 なら「資金が半減したら破産」。
    """
    rng = np.random.default_rng(seed)
    trades = max(1, int(round(trades_per_month)))
    total_trades = trades * months

    wins = rng.random((paths, total_trades)) < win_rate
    multipliers = np.where(
        wins, 1.0 + risk_fraction * win_r, 1.0 - risk_fraction * loss_r
    )
    # 負のリスク率や過大なリスクで乗数が 0 以下になる場合は破産として扱う。
    multipliers = np.clip(multipliers, 1e-12, None)

    equity = np.cumprod(multipliers, axis=1)
    equity_with_start = np.concatenate([np.ones((paths, 1)), equity], axis=1)

    running_max = np.maximum.accumulate(equity_with_start, axis=1)
    drawdowns = 1.0 - equity_with_start / running_max
    max_drawdowns = drawdowns.max(axis=1)

    ruined = (equity_with_start <= ruin_threshold).any(axis=1)

    month_ends = equity_with_start[:, :: trades][:, : months + 1]
    if month_ends.shape[1] < months + 1:
        month_ends = equity_with_start[:, np.linspace(0, total_trades, months + 1).astype(int)]
    monthly_growth = month_ends[:, 1:] / np.maximum(month_ends[:, :-1], 1e-12)

    final = equity_with_start[:, -1]
    geo_monthly = np.power(np.maximum(final, 1e-12), 1.0 / months)

    return MonteCarloResult(
        risk_fraction=risk_fraction,
        win_rate=win_rate,
        win_r=win_r,
        loss_r=loss_r,
        trades_per_month=trades_per_month,
        months=months,
        paths=paths,
        prob_ruin=float(ruined.mean()),
        prob_hit_target_monthly=float((monthly_growth >= monthly_target).mean()),
        prob_target_on_average=float((geo_monthly >= monthly_target).mean()),
        median_monthly_return=float(np.median(monthly_growth) - 1.0),
        p05_final_multiple=float(np.percentile(final, 5)),
        median_final_multiple=float(np.median(final)),
        p95_final_multiple=float(np.percentile(final, 95)),
        median_max_drawdown=float(np.median(max_drawdowns)),
        worst_max_drawdown=float(max_drawdowns.max()),
    )


# ----------------------------------------------------------------------
# 実行時のリスク制限
# ----------------------------------------------------------------------
@dataclass
class RiskManager:
    """日次損失上限と最大ドローダウンのキルスイッチを管理する。"""

    initial_equity: float
    max_daily_loss: float = 0.06
    max_drawdown_stop: float = 0.25

    peak_equity: float = field(init=False)
    _day: object | None = field(default=None, init=False)
    _day_start_equity: float = field(init=False)
    halted: bool = field(default=False, init=False)
    halt_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.peak_equity = self.initial_equity
        self._day_start_equity = self.initial_equity

    def on_bar(self, equity: float, day: object) -> None:
        if self._day is None or day != self._day:
            self._day = day
            self._day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        drawdown = 1.0 - equity / self.peak_equity if self.peak_equity > 0 else 0.0
        if self.max_drawdown_stop > 0 and drawdown >= self.max_drawdown_stop:
            self.halted = True
            self.halt_reason = (
                f"最大ドローダウン {drawdown:.1%} が上限 "
                f"{self.max_drawdown_stop:.1%} に到達したため停止"
            )

    def can_open(self, equity: float) -> tuple[bool, str | None]:
        if self.halted:
            return False, self.halt_reason
        if self.max_daily_loss > 0 and self._day_start_equity > 0:
            daily_loss = 1.0 - equity / self._day_start_equity
            if daily_loss >= self.max_daily_loss:
                return False, (
                    f"当日損失 {daily_loss:.1%} が上限 {self.max_daily_loss:.1%} に到達"
                )
        return True, None
