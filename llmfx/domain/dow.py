"""ダウ理論のトレンド判定と「ダウ転換」検出.

定義(本システムでの実装):

  上昇トレンド = 高値切り上げ(HH)かつ安値切り上げ(HL)
  下降トレンド = 高値切り下げ(LH)かつ安値切り下げ(LL)

  ダウ転換(強気) : 下降トレンド中に、確定済みの直近スイング高値を
                    終値で上抜けた瞬間
  ダウ転換(弱気) : 上昇トレンド中に、確定済みの直近スイング安値を
                    終値で下抜けた瞬間

すでに上昇トレンド中の高値更新は「継続」であって転換ではないため、
シグナルは出さない。1 つのスイング水準につき転換シグナルは 1 度だけ。
"""

from __future__ import annotations

from dataclasses import dataclass

from .swings import SwingDetector
from .types import (
    Candle,
    Side,
    StructureSnapshot,
    Swing,
    SwingLabel,
    SwingType,
    Trend,
)


@dataclass(frozen=True)
class ReversalEvent:
    """ダウ転換が成立したバーの情報。"""

    bar_index: int
    candle: Candle
    side: Side
    broken_level: float
    """上抜け/下抜けされたスイング価格。"""
    broken_swing: Swing
    stop_basis: float
    """転換前の最安値(強気時)/最高値(弱気時)。損切りラインの根拠。"""
    stop_basis_swing: Swing
    previous_trend: Trend
    extension: float
    """ブレイク水準から終値までの距離(飛び乗り度合いの判定に使う)。"""


class DowAnalyzer:
    """スイング検出 + トレンド状態機械 + 転換イベント生成。"""

    def __init__(
        self,
        detector: SwingDetector | None = None,
        require_prior_trend: bool = True,
        stop_basis_mode: str = "trend_extreme",
    ) -> None:
        self.detector = detector or SwingDetector()
        self.require_prior_trend = require_prior_trend
        self.stop_basis_mode = stop_basis_mode
        self.trend: Trend = Trend.RANGE

        self._bar_index = -1
        # 「同じ水準で何度もシグナルを出さない」ためのフラグ。
        # 新しいスイングが確定した時点でリセットされる。
        self._high_level_consumed = False
        self._low_level_consumed = False
        self._tracked_high: Swing | None = None
        self._tracked_low: Swing | None = None

    # ------------------------------------------------------------------
    @property
    def atr(self) -> float | None:
        return self.detector.atr

    @property
    def candles(self) -> list[Candle]:
        return self.detector.candles

    @property
    def swings(self) -> list[Swing]:
        return self.detector.swings

    def snapshot(self) -> StructureSnapshot:
        last_high = self.detector.last_swing(SwingType.HIGH)
        last_low = self.detector.last_swing(SwingType.LOW)
        prior_high = self.detector.nth_last_swing(SwingType.HIGH, 2)
        prior_low = self.detector.nth_last_swing(SwingType.LOW, 2)
        return StructureSnapshot(
            trend=self.trend,
            last_high=last_high.price if last_high else None,
            last_low=last_low.price if last_low else None,
            prior_high=prior_high.price if prior_high else None,
            prior_low=prior_low.price if prior_low else None,
            last_high_label=last_high.label if last_high else SwingLabel.UNKNOWN,
            last_low_label=last_low.label if last_low else SwingLabel.UNKNOWN,
            atr=self.atr or 0.0,
            swing_count=len(self.swings),
        )

    # ------------------------------------------------------------------
    def update(self, candle: Candle) -> ReversalEvent | None:
        """確定足を 1 本処理し、ダウ転換が成立していればイベントを返す。"""
        self._bar_index += 1
        self.detector.update(candle)

        last_high = self.detector.last_swing(SwingType.HIGH)
        last_low = self.detector.last_swing(SwingType.LOW)
        self._refresh_level_flags(last_high, last_low)

        # 構造だけからトレンドを推定できる場合は初期状態を埋めておく。
        if self.trend is Trend.RANGE:
            self.trend = self._infer_trend()

        event = self._detect_reversal(candle, last_high, last_low)
        if event is not None:
            self.trend = Trend.UP if event.side is Side.LONG else Trend.DOWN
        return event

    def _refresh_level_flags(self, last_high: Swing | None, last_low: Swing | None) -> None:
        """スイングが更新されたらブレイク済みフラグを解除する。"""
        if last_high is not None and (
            self._tracked_high is None
            or last_high.index != self._tracked_high.index
            or last_high.price != self._tracked_high.price
        ):
            self._tracked_high = last_high
            self._high_level_consumed = False

        if last_low is not None and (
            self._tracked_low is None
            or last_low.index != self._tracked_low.index
            or last_low.price != self._tracked_low.price
        ):
            self._tracked_low = last_low
            self._low_level_consumed = False

    def _infer_trend(self) -> Trend:
        highs = self.detector.swings_of(SwingType.HIGH)
        lows = self.detector.swings_of(SwingType.LOW)
        if len(highs) < 2 or len(lows) < 2:
            return Trend.RANGE
        if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
            return Trend.UP
        if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
            return Trend.DOWN
        return Trend.RANGE

    def _detect_reversal(
        self,
        candle: Candle,
        last_high: Swing | None,
        last_low: Swing | None,
    ) -> ReversalEvent | None:
        bullish = self._check_bullish(candle, last_high, last_low)
        if bullish is not None:
            return bullish
        return self._check_bearish(candle, last_high, last_low)

    def _check_bullish(
        self,
        candle: Candle,
        last_high: Swing | None,
        last_low: Swing | None,
    ) -> ReversalEvent | None:
        if last_high is None or last_low is None:
            return None
        if self.trend is Trend.UP or self._high_level_consumed:
            return None
        if self.require_prior_trend and self.trend is not Trend.DOWN:
            return None
        if candle.close <= last_high.price:
            return None
        self._high_level_consumed = True

        stop_basis = self._stop_basis(
            broken_index=last_high.index,
            recent_swing_index=last_low.index,
            swing_type=SwingType.LOW,
        )
        return ReversalEvent(
            bar_index=self._bar_index,
            candle=candle,
            side=Side.LONG,
            broken_level=last_high.price,
            broken_swing=last_high,
            stop_basis=stop_basis,
            stop_basis_swing=last_low,
            previous_trend=self.trend,
            extension=candle.close - last_high.price,
        )

    def _check_bearish(
        self,
        candle: Candle,
        last_high: Swing | None,
        last_low: Swing | None,
    ) -> ReversalEvent | None:
        if last_high is None or last_low is None:
            return None
        if self.trend is Trend.DOWN or self._low_level_consumed:
            return None
        if self.require_prior_trend and self.trend is not Trend.UP:
            return None
        if candle.close >= last_low.price:
            return None
        self._low_level_consumed = True

        stop_basis = self._stop_basis(
            broken_index=last_low.index,
            recent_swing_index=last_high.index,
            swing_type=SwingType.HIGH,
        )
        return ReversalEvent(
            bar_index=self._bar_index,
            candle=candle,
            side=Side.SHORT,
            broken_level=last_low.price,
            broken_swing=last_low,
            stop_basis=stop_basis,
            stop_basis_swing=last_high,
            previous_trend=self.trend,
            extension=last_low.price - candle.close,
        )

    def _stop_basis(
        self, broken_index: int, recent_swing_index: int, swing_type: SwingType
    ) -> float:
        """要件 2 の『ダウ転換前の最高値 / 最安値』を求める。

        起点の選び方が損切り幅を決め、ひいてはリスクリワードを決めるため、
        ここは設計上もっとも効く箇所:

        - trend_extreme(既定): ブレイクされたスイングのバーから現在までの
          極値。下降トレンドの最後の谷、つまり「転換前の最安値」そのもの。
          要件の文言に忠実だが、損切りは波 1 本分の幅になる。
        - recent_swing: 直近の押し安値/戻り高値からの極値。損切りは浅くなり
          RR は改善するが、転換前の谷を割る動きで先に狩られやすい。
        """
        start = broken_index if self.stop_basis_mode == "trend_extreme" else recent_swing_index
        start = max(0, min(start, self._bar_index))
        window = self.candles[start : self._bar_index + 1]
        if not window:
            candle = self.candles[self._bar_index]
            return candle.low if swing_type is SwingType.LOW else candle.high
        if swing_type is SwingType.LOW:
            return min(c.low for c in window)
        return max(c.high for c in window)
