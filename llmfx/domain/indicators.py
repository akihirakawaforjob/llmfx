"""逐次更新できるインジケータ.

バックテストで先読み(look-ahead bias)が混入しないよう、すべて
「1本ずつ食わせて、その時点までの値だけを返す」形にしてある。
"""

from __future__ import annotations

from .types import Candle


class ATR:
    """Wilder の ATR。`update()` は確定足を 1 本ずつ受け取る。"""

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("ATR period must be >= 1")
        self.period = period
        self._prev_close: float | None = None
        self._value: float | None = None
        self._seed: list[float] = []

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None

    def _true_range(self, candle: Candle) -> float:
        if self._prev_close is None:
            return candle.high - candle.low
        return max(
            candle.high - candle.low,
            abs(candle.high - self._prev_close),
            abs(candle.low - self._prev_close),
        )

    def update(self, candle: Candle) -> float | None:
        tr = self._true_range(candle)
        if self._value is None:
            self._seed.append(tr)
            if len(self._seed) >= self.period:
                self._value = sum(self._seed) / len(self._seed)
        else:
            self._value = (self._value * (self.period - 1) + tr) / self.period
        self._prev_close = candle.close
        return self._value
