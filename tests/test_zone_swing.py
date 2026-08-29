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


def test_the_rearm_distance_only_spaces_out_re_entries():
    """待ち伏せの解除は **連射を止めるためだけ**。約定の窓ではない。

    「注文の近くで引けた足でしか約定しない」にすると、大きな足で届いた
    のに見送る挙動が混ざり、距離を広げるほど選別が効いたように見える
    (実測で 0.5 / 1.0 / 2.0 ATR が +0.036 / +0.132 / +0.400 になった)。

    正しくは、**一度約定したらそこから離れるまで次を張らない**だけ。
    """
    import statistics

    def spacing(rearm):
        candles, ts = legs(rearm_atr=rearm)
        gaps = []
        for key in (True, False):
            same = sorted([t for t in ts if t.kind == "zone" and t.long_side == key],
                          key=lambda t: t.entry_index)
            for a, b in zip(same, same[1:]):
                gaps.append(max(abs(c.close - a.entry) / a.atr
                                for c in candles[a.entry_index:b.entry_index + 1]))
        return statistics.median(gaps)

    near, far = spacing(0.5), spacing(2.0)
    # 距離を広げれば、次を張るまでに price はより大きく離れている。
    assert far > near * 1.5, (near, far)
    assert far >= 2.0 - 1e-9, far


def test_the_nearer_level_is_reached_first():
    """損切りと転換ラインは同じ側にある。**近いほうが先に着く。**

    売りなら両方が上、買いなら両方が下。価格は片側から来るので順序に
    曖昧さが無い。ここで一律に損切りを先に見ると、**実際には先に届いて
    いた転換ラインでの手仕舞いを、毎回 -1 R の損切りへ振り替える**
    ことになる。負け側にだけ寄る誤り。
    """
    candles, ts = legs(max_adds=1, max_flips=1)
    for t in ts:
        if t.why != "stop":
            continue
        c = candles[t.exit_index]
        # 損切りで切れた足では、損切りのほうが手前にあったはず。
        if t.long_side:
            assert c.low <= t.exit + 1e-9, t.exit_index
        else:
            assert c.high >= t.exit - 1e-9, t.exit_index


def test_the_mirror_control_takes_the_same_fills_the_other_way():
    """対照: 同じ合図で逆に張る。約定する足と値段は変えない。"""
    _, base = legs(max_flips=0, max_adds=0)
    _, mirror = legs(max_flips=0, max_adds=0, reverse_entry=True)
    a = {(t.entry_index, round(t.entry, 9)): t
         for t in base if t.kind == "zone"}
    b = {(t.entry_index, round(t.entry, 9)): t
         for t in mirror if t.kind == "zone"}
    shared = set(a) & set(b)
    assert len(shared) > len(a) * 0.5, (len(a), len(b), len(shared))
    for k in shared:
        assert a[k].long_side is not b[k].long_side, k


def test_the_order_stays_live_even_when_price_is_far_away():
    """離れているあいだも注文は置いてある。**届けば約定する。**

    「近いときだけ約定を見る」にすると、大きな足で届いたのに見送る
    挙動が混ざり、再武装の距離を広げるほど選別が効いたように見える。
    実測で 0.5 / 1.0 / 2.0 ATR が +0.036 / +0.132 / +0.400 になった。

    再武装の距離を変えても、**約定した足はどれも注文へ届いている**。
    """
    for rearm in (0.5, 1.0, 2.0):
        candles, ts = legs(rearm_atr=rearm)
        zone = [t for t in ts if t.kind == "zone"]
        assert zone, rearm
        for t in zone:
            c = candles[t.entry_index]
            # **その足が実際に付けた値段でしか約定しない。**
            assert c.low - 1e-9 <= t.entry <= c.high + 1e-9, (rearm, t.entry_index)
            # 指値は市場のこちら側に置く。前の足の終値は向こう側にない。
            prior = candles[t.entry_index - 1].close
            if t.long_side:
                assert prior > t.entry - 1e-9, (rearm, t.entry_index)
            else:
                assert prior < t.entry + 1e-9, (rearm, t.entry_index)


def test_a_tight_stop_can_be_hit_on_the_fill_bar():
    """約定した足の残りで損切りへ届いたら、その足で切れる。

    建玉を翌足からしか見ないと、**損切りが狭いほど「同じ足で切られた
    はずの負け」を見逃す。**実測で 1.5 → 0.3 ATR と詰めると期待値が
    +0.067 → +0.801、平均勝ちが +4.50 → +22.49 R になった。分母が
    縮んだのではなく、負けが消えていた。
    """
    _, tight = legs(stop_buffer_atr=0.3, max_flips=0, max_adds=0)
    _, wide = legs(stop_buffer_atr=1.5, max_flips=0, max_adds=0)
    assert tight and wide
    same = sum(1 for t in tight if t.bars_held == 0 and t.why == "stop")
    assert same > 0, "狭い損切りなら約定足で切れる建玉があるはず"
    # 狭くするほど、約定足で切れる割合は増える
    a = same / len(tight)
    b = sum(1 for t in wide if t.bars_held == 0 and t.why == "stop") / len(wide)
    assert a > b, (a, b)


def test_the_fill_bar_check_ignores_what_happened_before_the_fill():
    """約定より前の値動きで切ってはいけない。

    順序は他所と同じ推し量り方(陽線 始値→安値→高値→終値)で解く。
    買いの逆指値は上げの途中で約定するので、その足の安値は普通
    **約定より前**に付いている。それで切ると、持ってもいない建玉を
    損切りしたことになる。
    """
    candles, ts = legs(stop_buffer_atr=0.3, max_flips=0, max_adds=0)
    checked = 0
    for t in ts:
        if t.bars_held or t.why != "stop":
            continue
        c = candles[t.entry_index]
        # **どちらから来たかは帯の側で決まる。**上端へは下から登って
        # 触れ、下端へは上から下りて触れる。
        from_below = t.zone_key == "top"
        if from_below and c.close >= c.open:
            # 陽線(始値→安値→高値→終値)を下から抜けて約定したなら、
            # 約定より後に残るのは 高値 と 終値 だけ。**安値は使えない。**
            checked += 1
            lo = min(c.high, c.close)
            hurt = lo if t.long_side else max(c.high, c.close)
            ok = (hurt <= t.stop_at_entry + 1e-9 if t.long_side
                  else hurt >= t.stop_at_entry - 1e-9)
            assert ok, (t.entry_index, t.zone_key, c.close, t.stop_at_entry)
        if not from_below and c.close < c.open:
            # 陰線(始値→高値→安値→終値)を上から下りて約定した場合も同じ。
            checked += 1
            hurt = min(c.low, c.close) if t.long_side else max(c.low, c.close)
            ok = (hurt <= t.stop_at_entry + 1e-9 if t.long_side
                  else hurt >= t.stop_at_entry - 1e-9)
            assert ok, (t.entry_index, t.zone_key, c.close, t.stop_at_entry)
    assert checked, "確かめる建玉が無い"


def test_the_flip_keeps_its_stop_behind_its_own_reversal_line():
    """ドテンした建玉も、損切りは **1 つ前** の折り返しへ。

    最新に置くと、乗り換えた側にとってのダウ転換も同じ水準を割ること
    なので、**損切りと利確が同じ値段**になる。利確側の出口が消え、
    遅れて動く損切りだけが収入源になる。実測でもドテンだけが向きに
    関係なくマイナスだった(順 -0.037 / 逆 -0.052)。
    """
    _, ts = legs(max_flips=2, max_adds=0)
    flips = [t for t in ts if t.kind == "flip"]
    assert flips
    for t in flips:
        assert t.line_at_entry, t.entry_index
        if t.long_side:
            assert t.stop_at_entry < t.line_at_entry, (t.entry_index,
                                                       t.stop_at_entry,
                                                       t.line_at_entry)
        else:
            assert t.stop_at_entry > t.line_at_entry, (t.entry_index,
                                                       t.stop_at_entry,
                                                       t.line_at_entry)


def test_a_flip_can_actually_take_profit_at_a_reversal():
    """利確側の出口が生きていること。

    損切りと同じ値段に置いていたときは、ドテンした建玉が転換で
    利益を出して終わることが原理的に起きなかった。
    """
    _, ts = legs(max_flips=2, max_adds=0)
    flips = [t for t in ts if t.kind == "flip"]
    won = [t for t in flips if t.why == "reversal" and t.r_multiple > 0]
    assert won, "転換で利益を出したドテンが 1 件も無い"
    share = sum(t.why == "reversal" for t in flips) / len(flips)
    assert share > 0.2, share


def test_the_fill_bar_assumption_brackets_a_tight_stop():
    """損切りを詰めるほど、結論は約定足の道順の仮定だけで決まる。

    損切りが足 1 本の値幅の内側に入るため。**掃引で端が最良に見えたら、
    まずここで挟む。**

    抜けた方向へ張る側でだけ差が出る。帯で跳ね返りを取る指値は、
    水準の向こう側へ進んで約定するので **不利側の端を必ず通る**
    (順序を問う余地がない)。
    """
    def stop_share(buf, mode):
        _, ts = legs(stop_buffer_atr=buf, fill_bar=mode, reverse_entry=True,
                     max_flips=0, max_adds=0)
        return sum(t.bars_held == 0 and t.why == "stop" for t in ts) / len(ts)
    tight = (stop_share(0.3, "path"), stop_share(0.3, "adverse"))
    wide = (stop_share(2.5, "path"), stop_share(2.5, "adverse"))
    assert tight[1] > tight[0], tight
    # 損切りが広ければ、どちらの仮定でもほとんど変わらない
    assert (tight[1] - tight[0]) > (wide[1] - wide[0]), (tight, wide)


# --- 足を三段にする -------------------------------------------------------


def test_the_three_timeframes_are_independent():
    """帯・構造・執行を別々の足で回せる。"""
    _, wide = legs(zone_minutes=240, structure_minutes=60)
    _, same = legs(zone_minutes=60, structure_minutes=60)
    assert wide and same
    assert len(wide) != len(same), (len(wide), len(same))


def test_higher_minutes_still_gives_the_old_two_rung_form():
    """旧来の呼び方は、帯と構造を同じ足にした形に落ちる。"""
    _, a = legs(higher_minutes=60, entry_signal="structure")
    _, b = legs(zone_minutes=60, structure_minutes=60, entry_signal="structure")
    assert len(a) == len(b)
    assert all(abs(x.r_multiple - y.r_multiple) < 1e-12 for x, y in zip(a, b))


def test_entering_on_the_execution_turn_fires_earlier():
    """執行の足の折り返しは、構造の水準より **手前** にあるので先に着く。

    構造の水準まで待つと、そこへ届くまでの値幅を丸ごと捨てる
    (利用者の指摘)。執行の足で引くと、転換での決済が増え、
    建玉を持っている時間が短くなる。

    **副作用がひとつある。**損切りは構造の足に預けたままなので、
    早く入るほど損切りが遠い。合成足では risk/ATR の中央が
    8.7 → 14.2 になった。R で測れば分母が伸びるだけだが、
    「負けても今より少なく済む」とは限らない。**そこは実測で見る。**
    """
    import statistics

    def look(mode):
        _, ts = legs(entry_signal=mode, max_flips=2, max_adds=0)
        z = [t for t in ts if t.kind == "zone"]
        assert z, mode
        return (sum(t.why == "reversal" for t in ts),
                statistics.median([t.bars_held for t in z]))

    rev_s, hold_s = look("structure")
    rev_x, hold_x = look("exec")
    assert rev_x > rev_s, (rev_s, rev_x)
    assert hold_x <= hold_s, (hold_s, hold_x)


def test_next_open_fills_at_the_next_bar_open():
    """水準で約定させない形は、次の足の始値ちょうどで入る。"""
    candles, ts = legs(entry_fill="next_open", max_flips=2, max_adds=2)
    later = [t for t in ts if t.kind in ("flip", "add")]
    assert later
    for t in later:
        assert t.entry == pytest.approx(candles[t.entry_index].open, abs=1e-9), \
            t.entry_index


def test_skipping_the_fallback_takes_fewer_flips():
    """執行の足が折り返さないまま走った場合に見送ると、ドテンは減る。"""
    _, take = legs(entry_fallback="structure", max_flips=2)
    _, skip = legs(entry_fallback="skip", max_flips=2)
    a = len([t for t in take if t.kind == "flip"])
    b = len([t for t in skip if t.kind == "flip"])
    assert 0 < b < a, (a, b)


def test_the_zone_entry_can_wait_for_the_execution_turn():
    """帯の最値に置いた指値は、**抜けたときだけ約定する。**

    跳ね返りを取りたいのに、届かずに見送るか、抜けてから掴まされるか
    のどちらかになる(利用者の指摘)。帯へ触れてから執行の足が折り返す
    のを待って入れば、ドテンや買い増しと同じ基準になり一貫する。
    """
    candles, ts = legs(zone_entry="exec_turn", max_flips=0, max_adds=0)
    zone = [t for t in ts if t.kind == "zone"]
    assert zone
    for t in zone:
        c = candles[t.entry_index]
        # その足が付けた値段で約定している
        assert c.low - 1e-9 <= t.entry <= c.high + 1e-9, t.entry_index
    # **帯の最値ちょうどではない。**最値に置くと抜けたときだけ約定する。
    at_edge = sum(abs(t.entry - t.zone_price) < 1e-9 for t in zone)
    assert at_edge < len(zone) * 0.2, (at_edge, len(zone))
    # **帯の近くで入っている。**執行の足の折り返しは帯と無関係に出るので、
    # 上限を掛けないと 15 ATR 先で約定して損切りだけ帯に残る。
    for t in zone:
        assert abs(t.entry - t.zone_price) <= 2.0 * t.atr + 1e-9, \
            (t.entry_index, abs(t.entry - t.zone_price) / t.atr)


def test_the_zone_entry_still_anchors_its_stop_to_the_band():
    """入り口が手前へ来ても、損切りは帯の外のまま。"""
    _, ts = legs(zone_entry="exec_turn", stop_buffer_atr=1.5,
                 max_flips=0, max_adds=0)
    zone = [t for t in ts if t.kind == "zone"]
    assert zone
    for t in zone:
        want = (t.zone_price - 1.5 * t.atr if t.long_side
                else t.zone_price + 1.5 * t.atr)
        assert t.stop_at_entry == pytest.approx(want, abs=1e-9), t.entry_index


def test_waiting_longer_after_the_touch_lets_more_trades_through():
    """帯へ触れてから待てる本数を延ばせば、拾える折り返しは増える。

    触れた記録そのものは帯が動くと基準もずれるので、**件数の向き**で見る。
    """
    counts = [len([t for t in legs(zone_entry="exec_turn", zone_wait_bars=w,
                                   max_flips=0, max_adds=0)[1]
                   if t.kind == "zone"])
              for w in (4, 48)]
    assert counts[1] > counts[0], counts


# --- 損切りの置き方(共通機構) --------------------------------------------


def test_the_entry_basis_puts_the_stop_a_fixed_distance_from_the_fill():
    """`entry` は約定値からの距離で置く。入り口がどこへ動いても幅は同じ。"""
    _, ts = legs(zone_entry="exec_turn", stop_basis="entry",
                 stop_buffer_atr=1.5, max_flips=0, max_adds=0)
    zone = [t for t in ts if t.kind == "zone"]
    assert zone
    for t in zone:
        assert t.risk / t.atr == pytest.approx(1.5, abs=1e-9), t.entry_index


def test_the_wave_basis_scales_the_stop_with_the_approach_wave():
    """`wave` は **直近の折り返しから帯までの長さ** に比例させる(利用者のやり方)。

    倍率を上げれば損切りは広がる。
    """
    import statistics

    def width(mult):
        _, ts = legs(zone_entry="exec_turn", stop_basis="wave",
                     stop_wave_mult=mult, max_flips=0, max_adds=0)
        z = [t.risk / t.atr for t in ts if t.kind == "zone"]
        assert z, mult
        return statistics.mean(z)

    # 倍率を動かすと拾える建玉も変わるので、隣どうしは揺れる。端で見る。
    assert width(4.0) > width(0.25) * 2, (width(0.25), width(4.0))


def test_the_floor_lifts_every_mechanism():
    """下限は **帯もドテンも買い増しも** 通る。一箇所で効く。

    掛けないと、入り口が帯の内側へ来たときに損切りが潰れ、そこは足の中の
    順序の仮定だけで符号が変わる領域に入る(実測で risk<1.1 ATR の
    4 分の 1 だけが +1.368 と -0.081 に割れた)。
    """
    for kw in ({"zone_entry": "exec_turn"}, {"zone_entry": "extreme"}):
        _, tight = legs(**kw, max_flips=1, max_adds=1, min_stop_atr=1.5)
        assert tight, kw
        assert not [t for t in tight if t.risk / t.atr < 1.5 - 1e-9], kw
    # 帯へ触れてから入る形では、掛けないと薄い脚ができる
    _, loose = legs(zone_entry="exec_turn", max_flips=1, max_adds=1)
    assert [t for t in loose if t.risk / t.atr < 1.5 - 1e-9]


def test_a_collapsed_stop_is_never_taken():
    """幅がほぼゼロの損切りで建玉にしない。**R が発散する。**

    波の 0.5 倍で -6,164,632 という値が出た。向きだけを見ていると
    1e-9 の幅でも通ってしまう。
    """
    for kw in ({"stop_basis": "wave", "stop_wave_mult": 0.1},
               {"stop_basis": "band", "stop_buffer_atr": 0.01},
               {"stop_basis": "entry", "stop_buffer_atr": 0.01}):
        _, ts = legs(zone_entry="exec_turn", max_flips=1, max_adds=1, **kw)
        for t in ts:
            assert t.risk / t.atr >= 0.02 - 1e-9, (kw, t.entry_index,
                                                   t.risk / t.atr)
            assert abs(t.r_multiple) < 200, (kw, t.r_multiple)


def test_the_wave_can_be_measured_from_either_turn():
    """「直近安値」を、いちばん新しい折り返しにも 1 つ前にも取れる。

    1 つ前から測ると波は長くなるので、損切りは広がる。
    """
    import statistics

    def width(ref):
        _, ts = legs(zone_entry="exec_turn", stop_basis="wave",
                     stop_wave_mult=1.0, wave_ref=ref, max_flips=0, max_adds=0)
        z = [t.risk / t.atr for t in ts if t.kind == "zone"]
        assert z, ref
        return statistics.mean(z)

    assert width("prev") > width("last"), (width("last"), width("prev"))


def test_add_never_records_a_stop_worse_than_an_earlier_leg() -> None:
    """後から入った脚の損切りが、前の脚より **不利な位置** にあってはならない。

    建玉の損切りは有利方向へしか動かない(買いなら上へ)。買い増しは
    その時点の `pos.stop` を位置ごと共有するので、脚を約定順に並べた
    とき損切りは買いで単調非減少、売りで単調非増加になるはず。

    以前はここで `min_stop_atr` の下限を **分母にだけ** 掛けていた。
    決済は必ず `pos.stop` で起きるのに、R をより遠い線で割るので
    負けが浅く見える。下限に届かない位置では見送るのが正しい。
    """
    # `min_stop_atr` を広めに取ると下限が効く場面が増える。旧コードは
    # この設定で 23 件中 3 件、後の脚に前より不利な損切りを記録した。
    msa = 4.0
    _, ts = legs(max_adds=3, max_flips=0, min_stop_atr=msa,
                 stop_basis="band", stop_buffer_atr=1.5)
    adds = [t for t in ts if t.kind == "add"]
    assert adds, "買い増しが 1 件も出ていない"

    by_pos: dict[int, list] = {}
    for t in ts:
        by_pos.setdefault(t.position_id, []).append(t)
    checked = 0
    for group in by_pos.values():
        group.sort(key=lambda t: (t.entry_index, t.kind != "zone"))
        for before, after in zip(group, group[1:]):
            if after.kind != "add":
                continue          # ドテンは別の建玉なので対象外
            if after.long_side:
                assert after.stop_at_entry >= before.stop_at_entry - 1e-9, (
                    f"買い増しの損切りが前の脚より下: "
                    f"{after.stop_at_entry} < {before.stop_at_entry}")
            else:
                assert after.stop_at_entry <= before.stop_at_entry + 1e-9, (
                    f"売り増しの損切りが前の脚より上: "
                    f"{after.stop_at_entry} > {before.stop_at_entry}")
            checked += 1
    assert checked > 0, "検査対象の買い増しが無い"

    # 下限に届かない位置の買い増しは、そもそも存在してはいけない。
    for t in adds:
        assert abs(t.entry - t.stop_at_entry) >= msa * t.atr - 1e-9, (
            f"下限に届かない買い増しが建っている: "
            f"{abs(t.entry - t.stop_at_entry) / t.atr:.3f} ATR")
