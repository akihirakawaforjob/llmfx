"""ローソク足を上位足へまとめる.

Dukascopy は 1 分足でしか出ないため、M15 で検証するには集約が要る。
時間足の比較(M15 / H1 / H4 / D1)にも同じ処理を使う。

区切りは UNIX エポック起点。M15 なら 00:00, 00:15, ... と綺麗に並ぶ。
足が欠けている時間帯(FX の週末など)は、その区切り自体を作らない。
埋めてしまうと存在しない値動きを検証に混ぜることになる。
"""

from __future__ import annotations

from ..domain.types import Candle


def resample_candles(candles: list[Candle], minutes: int) -> list[Candle]:
    """`minutes` 分足へまとめる。入力は時系列昇順であること。"""
    if minutes <= 0:
        raise ValueError("minutes は正の数である必要があります")
    if not candles:
        return []

    seconds = minutes * 60
    out: list[Candle] = []
    bucket: int | None = None
    start: Candle | None = None
    high = low = close = 0.0
    volume = 0.0

    for candle in candles:
        key = int(candle.time.timestamp()) // seconds
        if bucket is None or key != bucket:
            if start is not None:
                out.append(
                    Candle(
                        time=start.time.fromtimestamp(bucket * seconds, tz=start.time.tzinfo),
                        open=start.open,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                    )
                )
            bucket = key
            start = candle
            high, low, close = candle.high, candle.low, candle.close
            volume = candle.volume
        else:
            high = max(high, candle.high)
            low = min(low, candle.low)
            close = candle.close
            volume += candle.volume

    if start is not None and bucket is not None:
        out.append(
            Candle(
                time=start.time.fromtimestamp(bucket * seconds, tz=start.time.tzinfo),
                open=start.open,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return out
