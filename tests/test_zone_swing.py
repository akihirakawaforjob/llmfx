"""第4版 — 抵抗帯で入り、ダウ転換で手仕舞ってドテンする形のテスト.

いちばん守りたいのは 3 つ。

1. **先読みを持ち込まない。**構造を上位足で読むぶん、確定の遅れを
   取り違えやすい。データを打ち切っても過去が動かないことを見る
2. **損切りが必ず効く。**出口が「転換」と「損切り」の 2 つしかないので、
   損切りが漏れると片方の裾が無限に伸びる
3. **入る前から在った構造で、いきなり損切りを詰めない。**
   「エントリー時の損切り位置は問題なし」という合意と食い違う
"""

from __future__ import annotations

from collections import Counter

import pytest

from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.research.zone_swing import collect_swing_trades


def legs(count: int = 20_000, seed: int = 5, **kwargs):
    candles = generate_synthetic_candles(count=count, seed=seed)
    return candles, collect_swing_trades(candles, **kwargs)


# --- 先読み ---------------------------------------------------------------


def test_truncating_the_data_does_not_change_the_past():
    """打ち切っても、それ以前に決済が済んだ脚は 1 つも変わらない。

    終端で成行決済される脚(`why == "end"`)だけは、打ち切った側にしか
    無いので比較から外す。
    """
    base = dict(max_adds=2, max_flips=1)
    candles, full = legs(**base)
    assert full
    for frac in (0.5, 0.7, 0.9):
        cut = int(len(candles) * frac)
        part = collect_swing_trades(candles[:cut], **base)
        by = {(t.position_id, t.kind, t.entry_index): t
              for t in part if t.why != "end"}
        done = [t for t in full if t.exit_index < cut and t.why != "end"]
        assert done, frac
        for t in done:
            u = by.get((t.position_id, t.kind, t.entry_index))
            assert u is not None, (frac, t.position_id, t.kind, t.entry_index)
            assert abs(u.r_multiple - t.r_multiple) < 1e-9, (frac, t.kind)
            assert u.exit_index == t.exit_index
            assert abs(u.exit - t.exit) < 1e-9


# --- 損切り ---------------------------------------------------------------


def test_no_leg_loses_more_than_its_risk():
    """損切りが効いていれば、負けはちょうど -1 R で揃う。"""
    _, ts = legs(max_adds=2, max_flips=2)
    assert ts
    worst = min(t.r_multiple for t in ts)
    assert worst >= -1.000001, worst


def test_a_wider_spread_makes_the_long_limit_harder_to_fill():
    """買いは Ask で約定するので、Bid が余計に下がらないと届かない。

    この形は決済がすべて **水準** なので、スプレッドは決済価格ではなく
    「どこで発動するか」と「そもそも約定するか」に効く。売りの損切りも
    Ask で発動するぶん手前で刺さるが、転換ラインも同じだけ手前へ動くので、
    出口の内訳はほとんど変わらない。**入り口のほうに出る。**

    件数で比べても揺れる。**枠が空く順番が変わると拾える機会も変わる**
    ので(時間帯フィルタでも踏んだ挙動)、約定した足そのものを見る。
    """
    sp = 0.05
    candles, ts = legs(spread=sp)
    buys = [t for t in ts if t.kind == "zone" and t.long_side]
    sells = [t for t in ts if t.kind == "zone" and not t.long_side]
    assert buys and sells
    for t in buys:
        # 買いは Ask で約定する。Bid はスプレッドぶん余計に下がっている。
        assert candles[t.entry_index].low <= t.entry - sp + 1e-12, t.entry_index
    for t in sells:
        # 売りは Bid で売るので、そのまま届いていればよい。
        assert candles[t.entry_index].high >= t.entry - 1e-12, t.entry_index


def test_a_wider_spread_never_improves_the_result():
    """コストを広げて成績が良くなったら、どこかで二重に足している。"""
    _, tight = legs(spread=0.0, slippage=0.0)
    _, wide = legs(spread=0.05, slippage=0.01)
    ta = sum(t.r_multiple for t in tight) / len(tight)
    tb = sum(t.r_multiple for t in wide) / len(wide)
    assert tb <= ta + 1e-9, (ta, tb)


def test_the_stop_does_not_tighten_before_a_new_swing_forms():
    """入った時点で在った折り返しでは損切りを動かさない。

    動かすと、帯の外 1.5 ATR で置いたはずの損切りが、入った直後に
    帯の内側まで来てしまう。
    """
    _, ts = legs(stop_buffer_atr=1.5, swing_stop_buffer_atr=0.0)
    first = [t for t in ts if t.kind == "zone"]
    assert first
    for t in first:
        gap = abs(t.stop_at_entry - t.entry) / t.atr
        assert gap == pytest.approx(1.5, abs=1e-6), gap


# --- ドテンと買い増し -----------------------------------------------------


def test_a_reversal_closes_every_leg_of_the_position_at_once():
    """手仕舞いは建玉ごと。買い増しした脚も同じ足・同じ値段で閉じる。"""
    _, ts = legs(max_adds=3, max_flips=1)
    by: dict[int, list] = {}
    for t in ts:
        by.setdefault(t.position_id, []).append(t)
    multi = [g for g in by.values() if len(g) > 1]
    assert multi, "買い増しした建玉が無いと確かめられない"
    for g in multi:
        assert len({t.exit_index for t in g}) == 1
        assert len({round(t.exit, 9) for t in g}) == 1
        assert len({t.why for t in g}) == 1


def test_the_flip_goes_the_other_way_from_the_same_price():
    """ドテンは、手仕舞ったのと同じ値段で逆へ入る。"""
    _, ts = legs(max_adds=0, max_flips=1)
    flips = [t for t in ts if t.kind == "flip"]
    assert flips
    closed = {(t.exit_index, round(t.exit, 9), t.long_side)
              for t in ts if t.why == "reversal"}
    for f in flips:
        assert (f.entry_index, round(f.entry, 9), not f.long_side) in closed


def test_adds_are_capped_and_never_land_on_the_same_swing():
    """買い増しは上限まで。同じ折り返しで二度足さない。"""
    for cap in (0, 1, 3):
        _, ts = legs(max_adds=cap, max_flips=1)
        by: dict[int, list] = {}
        for t in ts:
            by.setdefault(t.position_id, []).append(t)
        for g in by.values():
            adds = [t for t in g if t.kind == "add"]
            assert len(adds) <= cap, (cap, len(adds))
            stamps = {(t.entry_index, round(t.entry, 9)) for t in adds}
            assert len(stamps) == len(adds)


def test_flips_are_capped():
    """ドテンの回数は上限で止まる。"""
    for cap in (0, 1, 2):
        _, ts = legs(max_flips=cap)
        assert all(t.flips <= cap for t in ts), cap
        if cap == 0:
            assert not [t for t in ts if t.kind == "flip"]


def test_never_more_positions_open_than_the_cap():
    """同時保有は max_open を超えない。"""
    for cap in (1, 2, 4):
        candles, ts = legs(max_open=cap, max_adds=1, max_flips=1)
        spans: dict[int, tuple[int, int]] = {}
        for t in ts:
            lo, hi = spans.get(t.position_id, (t.entry_index, t.exit_index))
            spans[t.position_id] = (min(lo, t.entry_index), max(hi, t.exit_index))
        # 決済と新規が同じ足で起きるのは普通(ドテンがまさにそれ)。
        # 建てる前に閉じているので、先に減らしてから増やす。
        events = []
        for lo, hi in spans.values():
            events.append((lo, 1))
            events.append((hi, -1))
        events.sort(key=lambda e: (e[0], e[1]))
        live = peak = 0
        for _, d in events:
            live += d
            peak = max(peak, live)
        assert peak <= cap, (cap, peak)


def test_more_room_lets_more_trades_through():
    """枠を増やせば取引は増える。時間切れを外した分だけ枠が長く埋まる。"""
    counts = [len(legs(max_open=c)[1]) for c in (1, 4)]
    assert counts[1] > counts[0], counts


# --- 機構ごとに割れること -------------------------------------------------


def test_every_leg_carries_the_mechanism_that_made_it():
    """合算の 1 行だけでは、どれが効いたか分からない(利用者の指摘)。"""
    _, ts = legs(max_adds=2, max_flips=1)
    kinds = Counter(t.kind for t in ts)
    assert set(kinds) == {"zone", "flip", "add"}, kinds
    assert all(t.why in ("stop", "reversal", "end") for t in ts)
    assert all(t.risk > 0 for t in ts)


def test_there_is_no_time_exit():
    """時間では閉じない。出口は転換と損切りだけ(終端を除く)。"""
    _, ts = legs(max_adds=1, max_flips=1)
    assert all(t.why != "time" for t in ts)
    assert max(t.bars_held for t in ts) > 240, "240 本で切られていないこと"


def test_the_limit_sits_exactly_where_it_was_placed():
    """指値は帯の最値から指定した ATR ぶんだけ外側。約定値はその値ちょうど。

    **件数で確かめてはいけない。**注文は置きっぱなしで、約定したら
    そこから離れるまで次を張らないだけなので、奥へ置いても件数は
    ほとんど動かない(帯そのものが更新されていくため)。
    """
    for off in (0.0, 0.5, 1.5):
        _, ts = legs(entry_beyond_atr=off)
        zone = [t for t in ts if t.kind == "zone"]
        assert zone, off
        for t in zone:
            want = (t.zone_price - off * t.atr if t.long_side
                    else t.zone_price + off * t.atr)
            assert t.entry == pytest.approx(want, abs=1e-9), (off, t.entry_index)


def test_a_reversal_only_fires_when_price_crosses_the_line():
    """転換ラインは逆指値。**抜けたときだけ**発動する。

    売りを帯の上端で建てた瞬間、直近の確定高値は既に価格より下にある。
    そこで「届いた」ことにすると、**建てた足でいきなり利益確定**する。
    実データでこれをやると期待値 +0.326 R・t=+31.8 という嘘が出た。
    """
    candles, ts = legs(max_adds=2, max_flips=1)
    rev = [t for t in ts if t.why == "reversal"]
    assert rev
    for t in rev:
        prior = candles[t.exit_index - 1].close
        if t.long_side:
            assert prior > t.exit, (t.exit_index, prior, t.exit)
        else:
            assert prior < t.exit, (t.exit_index, prior, t.exit)


def test_an_add_only_fires_when_price_crosses_the_line():
    """買い増しも同じ。既に抜けている水準では足さない。"""
    candles, ts = legs(max_adds=3, max_flips=1)
    adds = [t for t in ts if t.kind == "add"]
    assert adds
    for t in adds:
        prior = candles[t.entry_index - 1].close
        if t.long_side:
            assert prior < t.entry, (t.entry_index, prior, t.entry)
        else:
            assert prior > t.entry, (t.entry_index, prior, t.entry)


def test_the_rearm_distance_is_measured_from_the_order_not_the_zone():
    """待ち伏せの解除は、注文を置いてある値段からの距離で測る。

    帯からの距離で測ると、指値を奥へ置くほど「大きくヒゲを出して帯の
    近くへ戻った足」だけが約定するようになる。**狙って作った選別では
    ないので、そこで成績が上がっても機構の手柄ではない。**実測では
    奥 2.0 ATR で +0.946 R(t=+14.2)という数字が出た。
    """
    for off in (0.0, 1.0, 2.0):
        candles, ts = legs(entry_beyond_atr=off, rearm_atr=1.0)
        zone = [t for t in ts if t.kind == "zone"]
        assert zone, off
        for t in zone:
            c = candles[t.entry_index]
            # 約定した足の終値は、注文からの再武装の距離の内側にあるはず。
            assert abs(c.close - t.entry) <= 1.0 * t.atr + 1e-9, (off, t.entry_index)
