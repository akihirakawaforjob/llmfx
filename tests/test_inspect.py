"""トレードの切り出しと逆行/順行の集計のテスト.

負けは損切りできっちり -1R に揃うため、R の大小で分けても何も見えない。
順行したかどうかで割ると、打つ手が真逆の 2 種類に分かれる。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from llmfx.backtest.inspect import excursion_stats, extract_views, pick_examples
from llmfx.domain.types import Candle, ExitReason, Side, Trade

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def candles(n: int) -> list[Candle]:
    return [
        Candle(time=T0 + timedelta(hours=i), open=100.0 + i, high=101.0 + i,
               low=99.0 + i, close=100.5 + i, volume=1.0)
        for i in range(n)
    ]


def trade(entry_i: int, exit_i: int, r: float, mfe: float = 0.0,
          mae: float = 0.0, risk: float = 1.0) -> Trade:
    return Trade(
        side=Side.LONG, units=1000.0,
        entry_time=T0 + timedelta(hours=entry_i), entry_price=100.0,
        exit_time=T0 + timedelta(hours=exit_i), exit_price=100.0 + r,
        stop_loss=99.0, take_profit=110.0,
        initial_risk_per_unit=risk, risk_amount=risk * 1000,
        pnl=r * risk * 1000, r_multiple=r,
        exit_reason=ExitReason.STOP_LOSS if r < 0 else ExitReason.TAKE_PROFIT,
        bars_held=exit_i - entry_i, equity_after=1_000_000.0,
        rr_at_entry=2.0, target_source="trail_only",
        max_favorable_excursion=mfe * risk, max_adverse_excursion=mae * risk,
    )


# --- 切り出し -------------------------------------------------------------


def test_view_includes_context_before_and_after():
    cs = candles(100)
    views = extract_views([trade(50, 60, 2.0)], cs, lead=10, trail=5)
    assert len(views) == 1
    v = views[0]
    assert len(v.candles) == 10 + (60 - 50) + 5 + 1
    assert v.entry_offset == 10
    assert v.exit_offset == 20


def test_view_is_clipped_at_the_start_of_the_series():
    cs = candles(100)
    v = extract_views([trade(3, 8, 1.0)], cs, lead=40, trail=5)[0]
    assert v.entry_offset == 3, "先頭で切り詰めたときのずれ"
    assert v.candles[v.entry_offset].time == T0 + timedelta(hours=3)


def test_trades_outside_the_series_are_skipped():
    cs = candles(20)
    assert extract_views([trade(500, 510, 1.0)], cs) == []


def test_range_includes_the_stop_even_if_price_never_reached_it():
    cs = candles(30)
    v = extract_views([trade(10, 15, 1.0)], cs, lead=5, trail=2)[0]
    assert v.low <= 99.0


# --- 分け方 ---------------------------------------------------------------


def test_losers_are_split_by_whether_they_ever_moved_our_way():
    cs = candles(200)
    trades = (
        [trade(20 + i, 25 + i, 3.0, mfe=3.0) for i in range(3)]        # 勝ち
        + [trade(60 + i, 65 + i, -1.0, mfe=2.0) for i in range(4)]     # 順行してから負け
        + [trade(100 + i, 105 + i, -1.0, mfe=0.05) for i in range(4)]  # 即負け
    )
    groups = pick_examples(extract_views(trades, cs), per_group=6)
    assert set(groups) == {"大きく勝った", "順行してから戻された負け", "一度も順行しなかった負け"}
    assert len(groups["順行してから戻された負け"]) == 4
    assert len(groups["一度も順行しなかった負け"]) == 4


def test_middling_losers_belong_to_neither_group():
    """0.2R 〜 1.5R の負けはどちらとも言えないので入れない。"""
    cs = candles(200)
    trades = [trade(60 + i, 65 + i, -1.0, mfe=0.8) for i in range(4)]
    groups = pick_examples(extract_views(trades, cs))
    assert "順行してから戻された負け" not in groups
    assert "一度も順行しなかった負け" not in groups


def test_no_trades_gives_no_groups():
    assert pick_examples([]) == {}


# --- 逆行 / 順行 ----------------------------------------------------------


def test_excursions_are_expressed_in_r_multiples():
    """リスク幅が銘柄ごとに違うので、R に直さないと比べられない。

    ここだけは値幅(価格)を直に入れて、R への割り算が効いているかを見る。
    ヘルパの `mfe=` は R 倍数を受けるので、それでは往復して元に戻ってしまう。
    """
    raw = replace(
        trade(0, 5, -1.0, risk=0.05),
        max_favorable_excursion=1.0,   # 価格で 1.0 動いた = リスク 0.05 の 20 倍
        max_adverse_excursion=2.0,
    )
    stats = excursion_stats([raw])
    assert stats["負け"].mfe == [20.0]
    assert stats["負け"].mae == [40.0]


def test_winners_and_losers_are_counted_separately():
    stats = excursion_stats([
        trade(0, 5, 3.0, mfe=3.0), trade(6, 9, -1.0, mfe=0.5),
        trade(10, 14, -1.0, mfe=1.2),
    ])
    assert stats["勝ち"].count == 1
    assert stats["負け"].count == 2


def test_share_reaching_counts_the_tail():
    stats = excursion_stats([
        trade(0, 5, -1.0, mfe=m) for m in (0.1, 0.6, 1.2, 2.5)
    ])
    losers = stats["負け"]
    assert losers.share_reaching(0.5) == 0.75
    assert losers.share_reaching(1.0) == 0.5
    assert losers.share_reaching(2.0) == 0.25


def test_zero_risk_trades_are_ignored():
    assert excursion_stats([trade(0, 5, 0.0, risk=0.0)])["勝ち"].count == 0
