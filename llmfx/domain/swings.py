"""スイング(ピボット)高値・安値の逐次検出.

ダウ理論のすべての判断はこのスイング列の上に乗る。ここで先読みが
混入するとバックテストが丸ごと嘘になるため、確定条件を厳密に扱う:

  - バー i が「左に left 本、右に right 本」より極値であるときピボット
  - よってピボット i が確定するのはバー i+right の時点
  - 戦略が参照してよいのは confirmed_index <= 現在バー のスイングのみ

さらに、ノイズを潰すために ATR ベースの最小値幅フィルタをかけ、
高値と安値が必ず交互に並ぶよう正規化する(ジグザグ化)。
"""

from __future__ import annotations

from .indicators import ATR
from .types import Candle, Swing, SwingLabel, SwingType


class SwingDetector:
    """確定足を 1 本ずつ受け取り、交互に並んだスイング列を維持する。"""

    def __init__(
        self,
        left: int = 3,
        right: int = 3,
        atr_period: int = 14,
        min_swing_atr: float = 0.6,
    ) -> None:
        if left < 1 or right < 1:
            raise ValueError("left/right must be >= 1")
        self.left = left
        self.right = right
        self.min_swing_atr = min_swing_atr

        self._atr = ATR(atr_period)
        self._candles: list[Candle] = []
        self._atr_series: list[float | None] = []
        self.swings: list[Swing] = []

    # ------------------------------------------------------------------
    # 参照系
    # ------------------------------------------------------------------
    @property
    def atr(self) -> float | None:
        return self._atr.value

    @property
    def candles(self) -> list[Candle]:
        return self._candles

    def last_swing(self, swing_type: SwingType) -> Swing | None:
        for swing in reversed(self.swings):
            if swing.type is swing_type:
                return swing
        return None

    def nth_last_swing(self, swing_type: SwingType, n: int = 1) -> Swing | None:
        """新しい方から n 番目(1 始まり)の該当スイングを返す。"""
        seen = 0
        for swing in reversed(self.swings):
            if swing.type is swing_type:
                seen += 1
                if seen == n:
                    return swing
        return None

    def swings_of(self, swing_type: SwingType) -> list[Swing]:
        return [s for s in self.swings if s.type is swing_type]

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------
    def update(self, candle: Candle) -> bool:
        """確定足を 1 本追加する。スイング列が変化したら True。"""
        self._candles.append(candle)
        self._atr.update(candle)
        self._atr_series.append(self._atr.value)

        current = len(self._candles) - 1
        pivot = current - self.right
        if pivot < self.left:
            return False

        changed = False
        for swing in self._detect_pivots(pivot, current):
            changed |= self._push(swing)
        return changed

    def _detect_pivots(self, pivot: int, current: int) -> list[Swing]:
        """バー `pivot` がピボット高値/安値かを判定する。

        左側は「厳密に極値」、右側は「同値を許容」とすることで、同じ
        平坦部から複数のピボットが二重に出るのを防ぐ。
        """
        target = self._candles[pivot]
        left_slice = self._candles[pivot - self.left : pivot]
        right_slice = self._candles[pivot + 1 : pivot + self.right + 1]

        is_high = all(c.high < target.high for c in left_slice) and all(
            c.high <= target.high for c in right_slice
        )
        is_low = all(c.low > target.low for c in left_slice) and all(
            c.low >= target.low for c in right_slice
        )

        found: list[Swing] = []
        if is_high:
            found.append(
                Swing(
                    index=pivot,
                    confirmed_index=current,
                    time=target.time,
                    price=target.high,
                    type=SwingType.HIGH,
                )
            )
        if is_low:
            found.append(
                Swing(
                    index=pivot,
                    confirmed_index=current,
                    time=target.time,
                    price=target.low,
                    type=SwingType.LOW,
                )
            )

        # 高値と安値が同時に立つ内包バーでは、直前スイングと交互になる方を優先。
        if len(found) == 2 and self.swings:
            last_type = self.swings[-1].type
            found.sort(key=lambda s: 0 if s.type is not last_type else 1)
        return found

    def _min_move(self, index: int) -> float:
        atr = self._atr_series[index] if index < len(self._atr_series) else None
        if atr is None or atr <= 0:
            return 0.0
        return atr * self.min_swing_atr

    def _push(self, swing: Swing) -> bool:
        """交互性と最小値幅を満たすようスイング列へ組み込む。"""
        if not self.swings:
            self.swings.append(self._label(swing))
            return True

        last = self.swings[-1]

        if swing.type is last.type:
            # 同じ向きが連続したら、より極端な方だけを残す(ジグザグ更新)。
            more_extreme = (
                swing.price > last.price
                if swing.type is SwingType.HIGH
                else swing.price < last.price
            )
            if not more_extreme:
                return False
            self.swings[-1] = self._label(swing, replace_index=len(self.swings) - 1)
            return True

        # 向きが変わる場合は、値幅がノイズ閾値を超えたときのみ採用する。
        if abs(swing.price - last.price) < self._min_move(swing.index):
            return False
        self.swings.append(self._label(swing))
        return True

    def _label(self, swing: Swing, replace_index: int | None = None) -> Swing:
        """直前の同種スイングと比較して HH/LH/HL/LL を付与する。"""
        history = self.swings if replace_index is None else self.swings[:replace_index]
        previous = None
        for candidate in reversed(history):
            if candidate.type is swing.type:
                previous = candidate
                break

        if previous is None:
            label = SwingLabel.UNKNOWN
        elif swing.type is SwingType.HIGH:
            label = SwingLabel.HH if swing.price > previous.price else SwingLabel.LH
        else:
            label = SwingLabel.HL if swing.price > previous.price else SwingLabel.LL

        return Swing(
            index=swing.index,
            confirmed_index=swing.confirmed_index,
            time=swing.time,
            price=swing.price,
            type=swing.type,
            label=label,
        )
