"""明らかに壊れた足を落とす.

HistData の初期(2000〜2001 年ごろ)には、実在しない値が混ざっている。
実測で見つけたもの:

    EURUSD 2001-06-08  0.84850 → 1965.00010   2300 倍
    EURUSD 2001-09-11  0.91080 →   -0.00010   負の価格
    AUDUSD 2000-08-04  0.57950 →    0.10060   83% 下落

1 本混ざるだけでバックテストが壊れる。実際、EURUSD で 1 件の取引が
-277 R(損失 587 万円)を計上し、その銘柄の集計を無意味にしていた。

**実際に起きた急変は落とさないこと。**2015-01-15 のスイスフラン
ショックでは USD/CHF が数分で 18% 動いている。これは本物の値動きなので、
閾値をきつくして落としてしまうと、最も重要な場面を検証から外すことになる。

そこで判定は「あり得ない」水準だけに絞る:
  - 価格が 0 以下
  - 高安の整合が崩れている(low > high など)
  - 直近の中央値から 2 倍を超えて離れている(= 50% 未満か 200% 超)

判定は **四本値すべて** に掛ける。終値だけを見ていたときは、終値が正常で
高値だけ 39.82(実勢 0.79)という足が残り、1 件の取引が R=1018 を計上して
その銘柄の集計を壊した。

スイスフランショックの 18% は通る。2300 倍や 83% 下落は落ちる。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..domain.types import Candle

# 直近の中央値をとる本数。長すぎると急変に追従できず、短すぎると
# 異常値そのものに引きずられる。
REFERENCE_WINDOW = 200

# 中央値に対する許容比。0.5 〜 2.0 の外を「あり得ない」とみなす。
MIN_RATIO = 0.6
MAX_RATIO = 1.7

# 1 本の足の値幅(高値 - 安値)が終値に占める割合の上限。
# 「安値が終値のちょうど半分」という破損が実在し(USD/CHF 2004-05-17 は
# C=1.2826 に対し L=0.6413)、比率の判定だけでは境界のすぐ外側で生き残った。
# 本物の急変でも、スイスフランショックの 1 本あたりは 5% 程度。
MAX_BAR_RANGE = 0.25


@dataclass
class SanitizeReport:
    kept: int
    dropped: int
    reasons: dict[str, int]
    examples: list[str]

    @property
    def clean(self) -> bool:
        return self.dropped == 0

    def summary(self) -> str:
        if self.clean:
            return f"{self.kept:,} 本、異常なし"
        detail = " / ".join(f"{k} {v}" for k, v in sorted(self.reasons.items()))
        return f"{self.kept:,} 本 (除去 {self.dropped} 本: {detail})"


def _malformed(candle: Candle, max_range: float = MAX_BAR_RANGE) -> bool:
    values = (candle.open, candle.high, candle.low, candle.close)
    if any(v <= 0 for v in values):
        return True
    if not (candle.low <= min(candle.open, candle.close)
            and candle.high >= max(candle.open, candle.close)):
        return True
    # 1 本で終値の 25% を超える値幅は、為替ではあり得ない。
    return (candle.high - candle.low) / candle.close > max_range


def drop_bad_bars(
    candles: list[Candle],
    window: int = REFERENCE_WINDOW,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
) -> tuple[list[Candle], SanitizeReport]:
    """あり得ない足を落として、残りと報告を返す。

    基準は「直近の正常な足の中央値」。異常値そのものを基準に入れないよう、
    採用した足だけを窓に積む。
    """
    kept: list[Candle] = []
    reference: list[float] = []
    reasons: dict[str, int] = {}
    examples: list[str] = []

    def note(reason: str, candle: Candle) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        if len(examples) < 5:
            examples.append(f"{candle.time:%Y-%m-%d %H:%M} close={candle.close:g}")

    for candle in candles:
        if _malformed(candle):
            note("形が壊れている", candle)
            continue

        if reference:
            centre = median(reference)
            if centre <= 0:
                note("直近から乖離しすぎ", candle)
                continue
            # **四本値すべてを見る。**終値だけを見ていたときは、終値は
            # 正常なのに高値だけ 39.82(実勢 0.79)という足が残り、
            # 1 件の取引が R=1018 を計上して集計を壊した。
            ratios = [v / centre for v in (candle.open, candle.high, candle.low, candle.close)]
            if any(r < min_ratio or r > max_ratio for r in ratios):
                note("直近から乖離しすぎ", candle)
                continue

        kept.append(candle)
        reference.append(candle.close)
        if len(reference) > window:
            reference.pop(0)

    return kept, SanitizeReport(
        kept=len(kept),
        dropped=len(candles) - len(kept),
        reasons=reasons,
        examples=examples,
    )
