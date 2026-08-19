"""上位足のダウ転換を下位足のフィルタとして使う.

狙い:
    下位足のダウ転換は単体では予測力が無い(実測 勝率 20.2% / 損益分岐 20.5%)。
    そこで下位足の転換を「候補」に格下げし、

      1. 上位足のダウ転換と同じ方向であること
      2. 上位足の転換後に付けた極値(上昇バイアスなら安値)の付近であること

    の 2 条件を満たすものだけを採る。負ける回数を減らすのが目的なので、
    1 回あたりの利幅が縮むことは織り込む。

先読み防止がこの実装の要:
    上位足のバーは、次の下位足が来た時点で初めて「確定した」と扱う。
    まだ形成中の上位足バーは絶対に参照しない。この 1 本分の余分な遅れは、
    実運用でも同じだけ発生するので、そのまま持たせておくのが正しい。

距離の物差し:
    「極値の付近」は **上位足の ATR** で測る。上位足スケールの押し目を
    下位足の ATR で測ると物差しが小さすぎ、ほぼ全部が「遠い」と判定されて
    標本が消える(実際に一度これで取引 0〜20 件になった)。
"""

from __future__ import annotations

from .dow import DowAnalyzer
from .swings import SwingDetector
from .types import Candle, Trend

# 上位足の指定に使える表記。下位足と同じ語彙を使う。
GRANULARITY_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D": 1440,
    "D1": 1440,
    "W": 10080,
    "W1": 10080,
}


class TimeframeError(ValueError):
    pass


def granularity_minutes(text: str) -> int:
    key = text.strip().upper()
    if key not in GRANULARITY_MINUTES:
        raise TimeframeError(
            f"未対応の時間足です: {text!r}。"
            f"指定できるのは {', '.join(GRANULARITY_MINUTES)} です。"
        )
    return GRANULARITY_MINUTES[key]


class HigherTimeframeFilter:
    """下位足の確定足を食べて上位足を組み立て、その転換方向と極値を持つ。"""

    def __init__(
        self,
        minutes: int,
        left: int = 3,
        right: int = 3,
        atr_period: int = 14,
        min_swing_atr: float = 0.6,
        require_prior_trend: bool = True,
        stop_basis_mode: str = "trend_extreme",
    ) -> None:
        if minutes <= 0:
            raise TimeframeError("上位足の分数は正の数である必要があります")
        self.seconds = minutes * 60
        self.analyzer = DowAnalyzer(
            detector=SwingDetector(
                left=left,
                right=right,
                atr_period=atr_period,
                min_swing_atr=min_swing_atr,
            ),
            require_prior_trend=require_prior_trend,
            stop_basis_mode=stop_basis_mode,
        )

        self.bias: Trend | None = None
        """直近の上位足ダウ転換の方向。まだ 1 度も出ていなければ None。"""
        self.extreme: float | None = None
        """バイアス成立後に付けた極値。上昇なら最安値、下降なら最高値。"""
        self.bars_since_reversal: int = 0
        """バイアス成立からの下位足の本数。設定の鮮度を見るのに使う。"""

        self._bucket: int | None = None
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._volume: float = 0.0
        self._time = None
        self._completed: int = 0

    @property
    def atr(self) -> float | None:
        """上位足の ATR。距離の物差しに使う。"""
        return self.analyzer.atr

    @property
    def completed_bars(self) -> int:
        """確定させた上位足の本数(ウォームアップ判定用)。"""
        return self._completed

    # ------------------------------------------------------------------
    def update(self, candle: Candle) -> None:
        """下位足を 1 本受け取る。上位足が 1 本閉じたらそこで判定を進める。"""
        bucket = int(candle.time.timestamp()) // self.seconds

        if self._bucket is None:
            self._start(bucket, candle)
        elif bucket != self._bucket:
            # ここで初めて前の上位足バーが確定する。閉じた足だけを流す。
            self._flush()
            self._start(bucket, candle)
        else:
            self._high = max(self._high, candle.high)
            self._low = min(self._low, candle.low)
            self._close = candle.close
            self._volume += candle.volume

        # 極値の追従は下位足の精度で行う。上位足の確定を待つ必要は無く、
        # 待つと「押し安値の付近」を取り逃がす。
        if self.bias is Trend.UP:
            self.extreme = candle.low if self.extreme is None else min(self.extreme, candle.low)
            self.bars_since_reversal += 1
        elif self.bias is Trend.DOWN:
            self.extreme = candle.high if self.extreme is None else max(self.extreme, candle.high)
            self.bars_since_reversal += 1

    # ------------------------------------------------------------------
    def _start(self, bucket: int, candle: Candle) -> None:
        self._bucket = bucket
        self._time = candle.time
        self._open = candle.open
        self._high = candle.high
        self._low = candle.low
        self._close = candle.close
        self._volume = candle.volume

    def _flush(self) -> None:
        assert self._time is not None
        bar = Candle(
            time=self._time,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
        )
        self._completed += 1
        event = self.analyzer.update(bar)
        if event is None:
            return
        # 新しい上位足の転換。バイアスを切り替え、極値の追従をやり直す。
        self.bias = Trend.UP if event.side.sign > 0 else Trend.DOWN
        self.extreme = None
        self.bars_since_reversal = 0
