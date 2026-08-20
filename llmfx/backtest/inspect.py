"""トレードを図にするためのデータ作り.

集計値は「何が起きたか」を教えてくれるが、「なぜ起きたか」は教えてくれない。
勝ちと負けの実際の値動きを並べて見ることで、集計に埋もれた差を探す。

ここが受け持つのは切り出しと正規化だけで、描画そのものは持たない
(HTML でも Markdown でも使えるように)。

見るべきものを 2 つ用意する:

  1. 個別のトレード     エントリー前後の足と、判断に使った水準
  2. MAE / MFE の分布   建玉が最も逆行した幅と、最も順行した幅

2 は特に効く。負けが「一度も順行しないまま切られた」のか
「順行してから戻された」のかで、打つ手が真逆になるため。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.types import Candle, Trade


@dataclass
class TradeView:
    """1 トレード分の、図にできる形にした切り出し."""

    trade: Trade
    candles: list[Candle]
    """エントリー前後の足。前後の余白を含む。"""
    entry_offset: int
    """`candles` の何本目がエントリー足か。"""
    exit_offset: int

    @property
    def low(self) -> float:
        return min(min(c.low for c in self.candles), self.trade.stop_loss)

    @property
    def high(self) -> float:
        return max(max(c.high for c in self.candles), self.trade.stop_loss)

    @property
    def won(self) -> bool:
        return self.trade.pnl > 0


def extract_views(
    trades: list[Trade],
    candles: list[Candle],
    lead: int = 40,
    trail: int = 10,
    limit: int | None = None,
) -> list[TradeView]:
    """トレードごとに前後の足を切り出す。

    `lead` はエントリーより前、`trail` は決済より後に含める本数。
    前を厚めに取るのは、上位足の上抜けと押し目がその範囲に収まるため。
    """
    index_of = {c.time: i for i, c in enumerate(candles)}
    views: list[TradeView] = []
    for trade in trades:
        start_i = index_of.get(trade.entry_time)
        end_i = index_of.get(trade.exit_time)
        if start_i is None or end_i is None:
            continue
        lo = max(0, start_i - lead)
        hi = min(len(candles), end_i + trail + 1)
        views.append(
            TradeView(
                trade=trade,
                candles=candles[lo:hi],
                entry_offset=start_i - lo,
                exit_offset=end_i - lo,
            )
        )
        if limit is not None and len(views) >= limit:
            break
    return views


def pick_examples(
    views: list[TradeView], per_group: int = 6
) -> dict[str, list[TradeView]]:
    """見比べる価値のある組に分ける。

    負けは損切りできっちり -1R に揃うため、R の大小で分けても何も見えない。
    **順行したかどうか**で割ると、打つ手が真逆の 2 種類に分かれる:

      順行してから戻された  決済の置き方の問題。建値移動などが効く余地
      一度も順行しなかった  エントリーの質の問題。入るべきでなかった

    実測(4 銘柄 183 件)では、負け 148 件のうち 38.5% が +1.0R まで順行して
    から負けに転じており、一度も動かなかったのは 22.3% だけだった。
    """
    if not views:
        return {}
    by_r = sorted(views, key=lambda v: v.trade.r_multiple)
    losers = [v for v in by_r if not v.won]
    winners = [v for v in by_r if v.won]

    def mfe_r(view: TradeView) -> float:
        risk = view.trade.initial_risk_per_unit
        return view.trade.max_favorable_excursion / risk if risk > 0 else 0.0

    groups: dict[str, list[TradeView]] = {}
    if winners:
        groups["大きく勝った"] = winners[-per_group:][::-1]
    if losers:
        # 負けは損切りできっちり -1R に揃うので、R で分けても何も分からない。
        # 順行したかどうかで分けると、打つ手が変わる 2 種類に割れる。
        near = sorted((v for v in losers if mfe_r(v) >= 1.5), key=mfe_r, reverse=True)
        dead = sorted((v for v in losers if mfe_r(v) < 0.2), key=mfe_r)
        if near:
            groups["順行してから戻された負け"] = near[:per_group]
        if dead:
            groups["一度も順行しなかった負け"] = dead[:per_group]
    return groups


@dataclass
class ExcursionStats:
    """建玉がどこまで逆行し、どこまで順行したか(R 倍数)."""

    label: str
    count: int
    mae: list[float] = field(default_factory=list)
    """最大逆行幅。損切りに近いほど 1.0 へ近づく。"""
    mfe: list[float] = field(default_factory=list)
    """最大順行幅。"""

    @property
    def median_mae(self) -> float:
        return _median(self.mae)

    @property
    def median_mfe(self) -> float:
        return _median(self.mfe)

    def share_reaching(self, r: float) -> float:
        """順行が R 倍に届いた割合。"""
        if not self.mfe:
            return 0.0
        return sum(1 for v in self.mfe if v >= r) / len(self.mfe)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def excursion_stats(trades: list[Trade]) -> dict[str, ExcursionStats]:
    """勝ちと負けで、逆行・順行の分布を分けて集める。

    負けが「一度も順行しないまま切られた」のか「順行してから戻された」のかで
    打つ手が真逆になる。前者ならエントリーの質、後者は決済の置き方の問題。
    """
    groups = {
        "勝ち": ExcursionStats("勝ち", 0),
        "負け": ExcursionStats("負け", 0),
    }
    for trade in trades:
        risk = trade.initial_risk_per_unit
        if risk <= 0:
            continue
        key = "勝ち" if trade.pnl > 0 else "負け"
        stats = groups[key]
        stats.count += 1
        stats.mae.append(trade.max_adverse_excursion / risk)
        stats.mfe.append(trade.max_favorable_excursion / risk)
    return groups
