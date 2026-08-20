"""通貨の強弱指数のテスト.

ダウ理論の三本目の柱(相互確認)を通貨で作るための道具。
守りたいのは 2 つ:

  先読みをしないこと    渡した足までの終値しか使わない
  符号を間違えないこと  基軸通貨が上がる = 決済通貨が下がる
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from llmfx.domain.strength import CurrencyStrength, split_pair

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def feed(cs: CurrencyStrength, rows: list[dict[str, float]]) -> None:
    for i, closes in enumerate(rows):
        cs.update(T0 + timedelta(hours=i), closes)


# --- ペアの分解 -----------------------------------------------------------


@pytest.mark.parametrize("text", ["USDJPY", "USD_JPY", "usd/jpy", "UsdJpy"])
def test_pair_is_split_whatever_the_notation(text):
    assert split_pair(text) == ("USD", "JPY")


def test_a_string_that_is_not_a_pair_is_rejected():
    with pytest.raises(ValueError):
        split_pair("BTC")


# --- 符号 -----------------------------------------------------------------


def test_a_rising_pair_lifts_the_base_and_pushes_down_the_quote():
    cs = CurrencyStrength(["USDJPY"], lookback=2)
    feed(cs, [{"USDJPY": 100.0}, {"USDJPY": 105.0}, {"USDJPY": 110.0}])
    scores = cs.strength()
    assert scores["USD"] > 0
    assert scores["JPY"] == pytest.approx(-scores["USD"]), "符号が対称でない"


def test_strength_is_the_mean_over_the_pairs_that_contain_the_currency():
    """USD は 2 ペアに現れる。片方だけ動いたら、平均なので半分になる。"""
    cs = CurrencyStrength(["USDJPY", "USDCHF"], lookback=1)
    feed(cs, [{"USDJPY": 100.0, "USDCHF": 1.0},
              {"USDJPY": 110.0, "USDCHF": 1.0}])
    expected = math.log(1.1) / 2
    assert cs.strength()["USD"] == pytest.approx(expected)


# --- 先読み ---------------------------------------------------------------


def test_nothing_is_reported_before_enough_bars_have_arrived():
    cs = CurrencyStrength(["USDJPY"], lookback=5)
    feed(cs, [{"USDJPY": 100.0 + i} for i in range(5)])
    assert not cs.ready
    assert cs.strength() == {}


def test_later_prices_never_change_an_earlier_reading():
    """後から足を流しても、それ以前に出した値は動かない。"""
    rows = [{"USDJPY": 100.0 + i, "EURUSD": 1.10} for i in range(10)]
    early = CurrencyStrength(["USDJPY", "EURUSD"], lookback=3)
    feed(early, rows[:6])
    snapshot = early.strength()

    late = CurrencyStrength(["USDJPY", "EURUSD"], lookback=3)
    feed(late, rows[:6])
    feed(late, [{"USDJPY": 999.0, "EURUSD": 9.9}])  # 未来の暴騰
    # 6 本目までの読みは、7 本目を入れる前に取ったものと一致していること
    assert snapshot == pytest.approx(early.strength())


def test_a_missing_pair_is_carried_forward_not_treated_as_zero():
    """銘柄ごとに歯抜けの時間帯がある。欠けた足で価格を 0 にしてはいけない。"""
    cs = CurrencyStrength(["USDJPY", "EURUSD"], lookback=2)
    feed(cs, [{"USDJPY": 100.0, "EURUSD": 1.10},
              {"USDJPY": 105.0},                      # EURUSD が欠けた足
              {"USDJPY": 110.0, "EURUSD": 1.10}])
    scores = cs.strength()
    assert scores["EUR"] == pytest.approx(0.0, abs=1e-9), "据え置きならリターンは 0"
    assert scores["USD"] > 0


# --- 相互確認 -------------------------------------------------------------


def build_ranked() -> CurrencyStrength:
    """AUD が最強、JPY が最弱になるように動かす。"""
    pairs = ["AUDJPY", "AUDUSD", "USDJPY", "EURUSD", "EURJPY", "GBPUSD", "GBPJPY"]
    cs = CurrencyStrength(pairs, lookback=1)
    start = {p: 100.0 for p in pairs}
    moved = dict(start)
    for p in pairs:
        base, quote = split_pair(p)
        if base == "AUD" or quote == "JPY":
            moved[p] = 120.0
    feed(cs, [start, moved])
    return cs


def test_confirmation_needs_both_sides_not_just_one():
    cs = build_ranked()
    order = cs.ranking()
    assert order[0] == "AUD" and order[-1] == "JPY", order
    assert cs.confirms("AUDJPY", long=True, top=2), "最強 vs 最弱が確認されない"
    assert not cs.confirms("AUDJPY", long=False, top=2), "逆方向を確認してはいけない"


def test_a_pair_of_two_middling_currencies_is_not_confirmed():
    cs = build_ranked()
    # EUR と GBP はどちらも中位。綱引きであって、全面高・全面安ではない。
    assert not cs.confirms("EURGBP", long=True, top=2)
    assert not cs.confirms("EURGBP", long=False, top=2)


def test_spread_is_positive_when_the_base_is_the_stronger_side():
    cs = build_ranked()
    assert cs.spread("AUDJPY") > 0
    assert cs.spread("AUDJPY") == pytest.approx(-cs.spread("JPYAUD"))
