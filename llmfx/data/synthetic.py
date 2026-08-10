"""合成価格データの生成.

API キーもネットワークも無い状態で、パイプライン全体(検出→バックテスト
→レポート)を通しで検証するために使う。トレンドとレンジが交互に現れる
レジームスイッチ型の幾何ブラウン運動で、ダウ構造が実際に出るようにしている。

注意: これは動作確認用であり、ここで出た成績には何の意味も無い。
戦略の評価は必ず実データで行うこと。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from ..domain.types import Candle

GRANULARITY_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D": 1440,
}


def generate_synthetic_candles(
    count: int = 5000,
    start_price: float = 150.0,
    granularity: str = "M15",
    volatility: float = 0.0012,
    trend_strength: float = 0.35,
    regime_length: int = 180,
    seed: int = 20260810,
    start_time: datetime | None = None,
) -> list[Candle]:
    rng = np.random.default_rng(seed)
    minutes = GRANULARITY_MINUTES.get(granularity.upper(), 15)
    start = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)

    # レジーム: +1 上昇 / -1 下降 / 0 レンジ
    regimes: list[int] = []
    while len(regimes) < count:
        regime = int(rng.choice([1, -1, 0], p=[0.35, 0.35, 0.30]))
        length = max(20, int(rng.normal(regime_length, regime_length * 0.4)))
        regimes.extend([regime] * length)
    regimes = regimes[:count]

    drift = np.array(regimes, dtype=float) * volatility * trend_strength
    shocks = rng.normal(0.0, volatility, size=count)
    log_returns = drift + shocks
    closes = start_price * np.exp(np.cumsum(log_returns))

    opens = np.empty(count)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    # 実体の外側にヒゲを付ける。
    wick_scale = volatility * start_price * 0.8
    upper = np.abs(rng.normal(0.0, wick_scale, size=count))
    lower = np.abs(rng.normal(0.0, wick_scale, size=count))
    highs = np.maximum(opens, closes) + upper
    lows = np.minimum(opens, closes) - lower

    candles: list[Candle] = []
    for i in range(count):
        candles.append(
            Candle(
                time=start + timedelta(minutes=minutes * i),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(rng.integers(50, 500)),
            )
        )
    return candles
