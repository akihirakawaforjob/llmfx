"""抵抗帯へ指値を置いて待つ取引のテスト.

利用者が実際にやっていた形。抵抗帯の少し奥に予め指値を置き、届けば
約定、届かなければ何も起きない。損切りは帯の外、利確は明示しない。

いちばん守りたいのは **逆選択を消さないこと**。指値は届いたときだけ
約定するので、価格が少し触れて反転する場面(いちばん美味しい形)では
約定しない。ここを「触れたら約定」にすると、成績が実際より良く出る。
"""

from __future__ import annotations

import pytest

from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.research.zone_fade import collect_fade_trades


def trades(**kwargs):
    candles = generate_synthetic_candles(count=8000, seed=5)
    return collect_fade_trades(candles, **kwargs)


# --- 指値の約定 -----------------------------------------------------------


def test_the_limit_only_fills_where_price_actually_traded():
    """届いていない指値で約定してはいけない(逆選択が消える)。"""
    for t in trades(horizon=24)[:200]:
        if t.from_below:
            assert t.entry <= t.stop, "売りなら損切りは上"
        else:
            assert t.entry >= t.stop, "買いなら損切りは下"


def test_a_limit_placed_far_inside_the_zone_rarely_fills():
    """帯の奥へ置くほど届きにくい。ここが逆選択の量を決める。

    件数は単調にならない。同時保有 1 建玉なので、ある機会を見送ると
    別の機会を拾える枠が空くため(過去に時間帯フィルタでも踏んだ挙動)。
    そこで、極端に奥へ置いた場合と比べる。
    """
    shallow = len(trades(horizon=24, entry_offset_atr=0.0))
    deep = len(trades(horizon=24, entry_offset_atr=3.0))
    assert deep < shallow * 0.7, (shallow, deep)


def test_waiting_longer_matters_only_when_the_limit_is_deep():
    """浅い指値は触れた足でほぼ約定するので、待ち時間は効かない。

    奥に置くと届くまでに時間がかかり、待ち時間が効きはじめる。
    (実測: 奥 1.0 ATR で 1 本待ち 30 件 -> 30 本待ち 465 件)
    """
    shallow = [len(trades(horizon=24, entry_offset_atr=0.1, max_wait_bars=w))
               for w in (1, 30)]
    deep = [len(trades(horizon=24, entry_offset_atr=1.0, max_wait_bars=w))
            for w in (1, 30)]
    assert shallow[1] < shallow[0] * 1.2, shallow
    assert deep[1] > deep[0] * 5, deep


# --- 損切り ---------------------------------------------------------------


def test_the_stop_sits_outside_the_zone():
    for t in trades(horizon=24, stop_buffer_atr=0.2)[:200]:
        assert t.risk_atr > 0
        # 帯の幅ぶんは必ず離れている
        assert t.risk_atr >= t.zone_width_atr * 0.3


def test_a_stopped_trade_is_exactly_minus_one_r():
    stopped = [t for t in trades(horizon=24) if t.hit_stop]
    assert stopped
    assert all(t.r_multiple == -1.0 for t in stopped)


def test_a_wider_stop_buffer_widens_the_risk():
    tight = trades(horizon=24, stop_buffer_atr=0.1)
    wide = trades(horizon=24, stop_buffer_atr=0.8)
    import statistics as st
    assert (st.median(t.risk_atr for t in wide)
            > st.median(t.risk_atr for t in tight))


# --- 建玉の重なり ---------------------------------------------------------


def test_positions_never_overlap():
    """同時に 1 建玉。重ねると同じ値動きを何度も数えることになる。"""
    got = trades(horizon=24)
    for a, b in zip(got, got[1:]):
        assert b.bar_index > a.bar_index


# --- 先読み ---------------------------------------------------------------


def test_truncating_the_series_does_not_change_earlier_trades():
    candles = generate_synthetic_candles(count=8000, seed=5)
    full = collect_fade_trades(candles, horizon=24)
    cut = collect_fade_trades(candles[:5000], horizon=24)
    assert cut
    for a, b in zip(cut, full):
        assert a.bar_index == b.bar_index
        assert a.entry == pytest.approx(b.entry)
        assert a.r_multiple == pytest.approx(b.r_multiple)


def test_narrow_zone_filter_keeps_only_narrow_zones():
    """件数は減るとは限らない。太い帯を見送ると、同時保有 1 建玉の枠が
    空いて別の機会を拾えるため。守るべきは幅の条件そのもの。"""
    strict = trades(horizon=24, max_zone_width_atr=1.5)
    assert strict
    for t in strict:
        assert t.zone_width_atr <= 1.5


def test_the_fill_bar_itself_can_stop_the_trade_out():
    """約定した足の残りで損切りまで走ることは普通にある。

    翌足から数えると、いちばん不利な場面だけを見逃して成績が良く出る。
    実測では、ここを直すだけで差引が +0.118 R から動いた。
    """
    got = trades(horizon=24)
    assert any(t.bars_held == 0 for t in got), "約定足での損切りが 1 件も無い"


# --- ブレイクリスクで見送る -----------------------------------------------


def swing(index, price, kind, lag=3):
    from datetime import datetime, timedelta, timezone

    from llmfx.domain.types import Swing

    return Swing(index=index, confirmed_index=index + lag,
                 time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index),
                 price=price, type=kind)


def test_rising_lows_into_a_resistance_zone_mean_the_defenders_are_losing():
    """安値切り上げ = 買い手が押し上げてきている。抵抗帯は破られやすい。"""
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    rising = [swing(10, 99.0, SwingType.LOW), swing(20, 99.6, SwingType.LOW)]
    falling = [swing(10, 99.6, SwingType.LOW), swing(20, 99.0, SwingType.LOW)]
    assert defenders_weakening(rising, from_below=True, bar_index=100)
    assert not defenders_weakening(falling, from_below=True, bar_index=100)


def test_the_support_side_is_the_mirror_image():
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    falling = [swing(10, 101.0, SwingType.HIGH), swing(20, 100.4, SwingType.HIGH)]
    rising = [swing(10, 100.4, SwingType.HIGH), swing(20, 101.0, SwingType.HIGH)]
    assert defenders_weakening(falling, from_below=False, bar_index=100)
    assert not defenders_weakening(rising, from_below=False, bar_index=100)


def test_unconfirmed_swings_are_not_used():
    """未確定のスイングで判定すると先読みになる。"""
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    later = [swing(10, 99.0, SwingType.LOW), swing(20, 99.6, SwingType.LOW)]
    assert not defenders_weakening(later, from_below=True, bar_index=22)
    assert defenders_weakening(later, from_below=True, bar_index=23)


def test_too_few_swings_means_no_opinion():
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    assert not defenders_weakening([], from_below=True, bar_index=100)
    assert not defenders_weakening(
        [swing(10, 99.0, SwingType.LOW)], from_below=True, bar_index=100
    )


def test_the_break_risk_filter_changes_which_setups_are_taken():
    loose = {(t.bar_index, t.entry) for t in trades(horizon=24)}
    strict = {(t.bar_index, t.entry)
              for t in trades(horizon=24, skip_break_risk=True)}
    assert strict and strict != loose
    # 件数は減るとは限らない。見送ると同時保有 1 建玉の枠が空くため。


def test_break_risk_uses_only_confirmed_swings():
    """未確定のスイングで判定すると先読みになる。"""
    from llmfx.data.synthetic import generate_synthetic_candles
    from llmfx.research.zone_fade import collect_fade_trades

    candles = generate_synthetic_candles(count=8000, seed=5)
    full = collect_fade_trades(candles, horizon=24, skip_break_risk=True)
    cut = collect_fade_trades(candles[:5000], horizon=24, skip_break_risk=True)
    assert cut
    for a, b in zip(cut, full):
        assert a.bar_index == b.bar_index
        assert a.r_multiple == pytest.approx(b.r_multiple)


def test_a_range_based_stop_is_never_tighter_than_the_zone_edge():
    """はみ出しで刈られないよう、帯の縁より外側にしか動かさない。"""
    import statistics as st

    edge = trades(horizon=24)
    wide = trades(horizon=24, stop_from_range_bars=60)
    assert wide
    assert (st.median(t.risk_atr for t in wide)
            >= st.median(t.risk_atr for t in edge))


def test_a_longer_window_never_tightens_the_stop():
    import statistics as st

    short = st.median(t.risk_atr for t in trades(horizon=24, stop_from_range_bars=20))
    long = st.median(t.risk_atr for t in trades(horizon=24, stop_from_range_bars=120))
    assert long >= short


# --- 指値を最値へ置く -----------------------------------------------------


def test_the_entry_limit_can_sit_at_the_range_extreme():
    """帯の縁の内側ではなく、直近 N 本の最値で待つ(利用者の本来の指摘)。

    縁の内側で待つと、上へ突き抜ける動きの途中で約定してしまい、
    そのまま損切りまで持っていかれる。
    """
    inside = trades(horizon=24, stop_buffer_atr=0.5)
    extreme = trades(horizon=24, stop_buffer_atr=0.5, entry_from_range_bars=20)
    assert extreme
    import statistics as st
    # 最値で待つほうが、指値は必ず帯の外側寄りになる
    assert (st.median(t.entry for t in extreme if t.from_below)
            >= st.median(t.entry for t in inside if t.from_below))


def test_the_extreme_excludes_the_bar_that_would_fill_it():
    """その足の高値で指値を決めると「更新したから約定した」の循環になる。

    循環していれば、触れた足で必ず約定するので待ち時間が効かなくなる。
    """
    brief = len(trades(horizon=24, entry_from_range_bars=60, max_wait_bars=1))
    patient = len(trades(horizon=24, entry_from_range_bars=60, max_wait_bars=30))
    assert brief < patient * 0.9, (brief, patient)


def test_the_stop_stays_outside_the_entry_extreme():
    for t in trades(horizon=24, entry_from_range_bars=20, stop_buffer_atr=0.5)[:200]:
        if t.from_below:
            assert t.stop > t.entry
        else:
            assert t.stop < t.entry


# --- 上位足の帯 / 上下 2 本のレンジ ---------------------------------------


def test_higher_timeframe_zones_use_only_closed_bars():
    """上位足は閉じてからしか使わない。先読みになる。"""
    from llmfx.research.zone_fade import collect_fade_trades

    candles = generate_synthetic_candles(count=12000, seed=5)
    full = collect_fade_trades(candles, horizon=24, higher_minutes=60)
    cut = collect_fade_trades(candles[:8000], horizon=24, higher_minutes=60)
    assert cut
    for a, b in zip(cut, full):
        assert a.bar_index == b.bar_index
        assert a.entry == pytest.approx(b.entry)
        assert a.r_multiple == pytest.approx(b.r_multiple)


def test_higher_timeframe_zones_are_fewer_and_wider():
    """上位足の帯は数が減り、間隔が空く。"""
    same = trades(horizon=24)
    higher = trades(horizon=24, higher_minutes=60)
    assert higher
    assert len(higher) < len(same)


def test_requiring_a_range_only_takes_setups_with_zones_on_both_sides():
    """片側だけで張ると、そこを抜けられたときに一方的に負ける。"""
    loose = trades(horizon=24)
    strict = trades(horizon=24, require_range=True)
    assert strict
    assert len(strict) < len(loose)


def test_a_range_wider_than_the_cap_is_skipped():
    wide = trades(horizon=24, require_range=True)
    narrow = trades(horizon=24, require_range=True, max_range_atr=3.0)
    assert narrow
    assert len(narrow) <= len(wide)


def test_the_two_modes_stay_switchable():
    """どちらが優れているかは測ってから決める。両方動くこと。"""
    for kwargs in (
        {},
        {"higher_minutes": 60},
        {"require_range": True},
        {"higher_minutes": 60, "require_range": True},
    ):
        got = trades(horizon=24, **kwargs)
        assert got, kwargs


# --- 反対側の帯で決済する(往復を刈る)------------------------------------


def test_exiting_at_the_opposite_zone_frees_the_slot_for_the_return_leg():
    """反対側で切れば建玉が早く空き、そこから逆向きに入れる。

    利用者の指摘: 区画に帯が二つあれば自ずとレンジになるので、
    両端から入れる仕組みが要る。
    """
    plain = trades(horizon=24)
    both = trades(horizon=24, exit_at_opposite_zone=True)
    assert both
    assert len(both) > len(plain), (len(plain), len(both))


def test_exiting_at_the_opposite_zone_raises_the_win_rate():
    plain = [t.r_multiple for t in trades(horizon=24)]
    both = [t.r_multiple for t in trades(horizon=24, exit_at_opposite_zone=True)]
    assert sum(1 for r in both if r > 0) / len(both) > \
        sum(1 for r in plain if r > 0) / len(plain)


def test_a_trade_closed_at_the_opposite_zone_is_not_a_full_stop_loss():
    """反対側での決済は損切りではない。-1.0R に潰してはいけない。"""
    both = trades(horizon=24, exit_at_opposite_zone=True)
    closed = [t for t in both if not t.hit_stop and t.bars_held < 24]
    assert closed, "反対側で決済した取引が 1 件も無い"
    assert any(t.r_multiple > 0 for t in closed)


def test_every_mechanism_can_be_switched_independently():
    """組み合わせを捨てていくために、機構は独立に入り切りできること。"""
    for kwargs in (
        {},
        {"higher_minutes": 60},
        {"require_range": True},
        {"exit_at_opposite_zone": True},
        {"skip_break_risk": True},
        {"higher_minutes": 60, "exit_at_opposite_zone": True},
        {"higher_minutes": 60, "require_range": True, "exit_at_opposite_zone": True},
    ):
        got = trades(horizon=24, **kwargs)
        assert got, kwargs


# --- 薄い時間帯を避ける ---------------------------------------------------


def test_blocked_hours_produce_no_fills_in_those_hours():
    """スプレッドが数倍に開く時間帯では建玉を持たない。"""
    got = trades(horizon=24, exit_at_opposite_zone=True,
                 blocked_hours_utc=frozenset({21, 22}))
    assert got
    assert not [t for t in got if t.entry_hour in (21, 22)]


def test_every_trade_records_the_hour_it_filled():
    """時間帯ごとのコストを後から評価するのに要る。"""
    got = trades(horizon=24)
    assert got
    assert all(0 <= t.entry_hour <= 23 for t in got)
    assert len({t.entry_hour for t in got}) > 1, "1 つの時刻に偏っている"


# --- 損切りが床であること(反対側の帯で決済するときの取り違え)-------------


def test_no_trade_loses_more_than_one_r():
    """**損切りより悪い決済が出たら、それは決済先の選び方のバグ。**

    一度これが起きていた。反対側の帯を「平均値が指値の向こう側にある帯」で
    選び、決済は **手前の縁** で行っていたため、幅の広い帯だと縁が指値の
    こちら側へ回り込み、損切りを飛び越えた地点で「利確」していた。
    実データ(USD/JPY 開発用)では負けの半分がこれで、平均 -3.78 R、
    最悪 -35.35 R。損切りが機能していない取引が半分あった。

    ただし**この床だけを見ていても気づけない**。合成データでは縁の
    回り込みが浅く、バグを戻しても床は割れなかった。原因を掴むのは
    次の 2 つ(利益方向にあるか / 縁を建玉時に確定しているか)のほう。
    """
    for kwargs in ({}, {"exit_at_opposite_zone": True},
                   {"higher_minutes": 60, "exit_at_opposite_zone": True}):
        got = trades(horizon=24, **kwargs)
        assert got, kwargs
        worst = min(got, key=lambda t: t.r_multiple)
        assert worst.r_multiple >= -1.0 - 1e-9, (kwargs, worst)


def test_the_opposite_zone_is_always_on_the_profit_side():
    """反対側の帯で決済したなら、必ず利益になっていること。

    建玉を持った時点で利益方向にある帯だけを選ぶので、そこへ届いた
    ということは利が乗っている。届かなければ損切りか時間切れになる。
    """
    got = trades(horizon=24, exit_at_opposite_zone=True)
    closed = [t for t in got if t.why == "opp"]
    assert closed, "反対側で決済した取引が 1 件も無い"
    for t in closed:
        assert t.r_multiple > 0, t
        if t.from_below:
            assert t.exit_price < t.entry, t   # 売りなので下で決済
        else:
            assert t.exit_price > t.entry, t


def test_the_opposite_edge_is_fixed_when_the_position_is_opened():
    """帯は接触が足されるたびに広がる。**参照のまま持つと縁が動く。**

    決済の判定時に縁が指値の向こう側へ回り込んでいた。決済価格が
    損切りより悪くなるのはこれが原因だった。建玉を持った時点の値で
    確定させること。
    """
    got = trades(horizon=24, exit_at_opposite_zone=True)
    for t in got:
        if t.why != "opp":
            continue
        gain = (t.entry - t.exit_price) if t.from_below else (t.exit_price - t.entry)
        assert gain > 0, t
        # R に直しても損切り幅より手前で切れていない
        assert abs(t.r_multiple) < 1e6


def test_every_trade_records_where_it_exited():
    got = trades(horizon=24, exit_at_opposite_zone=True)
    assert {t.why for t in got} <= {"stop", "opp", "time"}
    assert {"stop", "opp"} <= {t.why for t in got}
    for t in got:
        assert (t.why == "stop") == t.hit_stop, t


# --- 1 本の足の中の道順 ---------------------------------------------------


def test_the_fill_bar_can_be_excluded_from_taking_profit():
    """**四本値では 1 本の足の中の道順が分からない。**

    約定した足で反対側の帯へ届いていても、実際には安値を先に付けてから
    高値を付けた(= 指値に届く前に利確地点を通過していた)かもしれない。
    既定はこれをこちらに有利な側に解釈している。損切りは逆に常に不利な側
    (同じ足で両方に触れたら損切りが先)なので、利確だけが甘い。

    実測(USD/JPY 開発用)では利確の 87.6% が約定足そのもので起きていて、
    ここを不利側に倒すと差引 +0.114 R が -0.185 R になる。**符号が変わる。**
    決着には M1 のような細かい足で道順を解く必要がある。
    """
    same = trades(horizon=24, exit_at_opposite_zone=True, intrabar="stop_first")
    strict = trades(horizon=24, exit_at_opposite_zone=True,
                    intrabar="no_same_bar_profit")
    assert same and strict
    assert not [t for t in strict if t.why == "opp" and t.bars_held == 0]
    assert [t for t in same if t.why == "opp" and t.bars_held == 0]


def test_stops_are_still_taken_on_the_fill_bar_either_way():
    """利確だけを厳しくする。損切りは常に約定足から見る。"""
    strict = trades(horizon=24, exit_at_opposite_zone=True,
                    intrabar="no_same_bar_profit")
    assert [t for t in strict if t.why == "stop" and t.bars_held == 0]


def test_the_intrabar_path_can_be_resolved_from_finer_candles():
    """細かい足があるなら、四本値からの推測ではなく順序そのものを使う。

    合成データを M1 として作り、M15 へ集約したものを本体に渡す。
    `path` は `ohlc` の推測よりも実際の順序に近いので、両者は一致しない。
    """
    from llmfx.data.resample import resample_candles

    fine = generate_synthetic_candles(count=60_000, seed=9, granularity="M1")
    coarse = resample_candles(fine, 15)
    got = {}
    for mode in ("stop_first", "ohlc", "path"):
        got[mode] = collect_fade_trades(
            coarse, horizon=24, exit_at_opposite_zone=True,
            intrabar=mode, path_candles=fine if mode == "path" else None,
        )
        assert got[mode], mode
    mean = {k: sum(t.r_multiple for t in v) / len(v) for k, v in got.items()}
    # 約定足で利確できる分、高安だけで見るほうが必ず甘い側に出る。
    assert mean["stop_first"] >= mean["path"] - 1e-9, mean
    assert got["path"] != got["ohlc"], "細かい足を渡しても推測と同じでは意味がない"


def test_path_mode_requires_the_finer_candles():
    with pytest.raises(ValueError):
        trades(horizon=24, exit_at_opposite_zone=True, intrabar="path")


def test_an_unknown_intrabar_mode_is_refused():
    with pytest.raises(ValueError):
        trades(horizon=24, intrabar="whatever")


# --- 帯を引いた足の物差しで測る -------------------------------------------


def test_scaling_to_the_zone_timeframe_widens_the_stop():
    """上位足の帯を下位足の ATR で測ると、損切りが小さすぎて別物になる。

    利用者の指摘: 抵抗帯の最値は、参照している時間軸の抵抗帯と一緒で
    あるべき。物差しも最値も、帯を引いた足のものを使う。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(entry_from_range_bars=20, stop_buffer_atr=0.75,
                  max_zone_width_atr=1.5, max_wait_bars=12, horizon=24,
                  exit_at_opposite_zone=True, higher_minutes=60)
    low = collect_fade_trades(candles, **common)
    high = collect_fade_trades(candles, **common, scale_to_zone_timeframe=True)
    assert low and high
    med = lambda ts: sorted(abs(t.stop - t.entry) for t in ts)[len(ts) // 2]
    assert med(high) > med(low) * 1.5, (med(low), med(high))
    # R の分母は ATR で割り戻すので、どちらも 0.75 のまま
    assert abs(sorted(t.risk_atr for t in high)[len(high) // 2] - 0.75) < 0.01


def test_scaling_has_no_effect_without_a_higher_timeframe():
    """帯を下位足で引いているなら、合わせる先が無いので何も変わらない。"""
    plain = trades(horizon=24, entry_from_range_bars=20, exit_at_opposite_zone=True)
    same = trades(horizon=24, entry_from_range_bars=20, exit_at_opposite_zone=True,
                  scale_to_zone_timeframe=True)
    assert [t.entry for t in plain] == [t.entry for t in same]


def test_the_higher_timeframe_extreme_only_uses_closed_bars():
    """まだ閉じていない上位足の最値を使うと先読みになる。

    データを途中で打ち切っても、それ以前に決済が終わった取引は 1 件も
    変わらないこと。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(entry_from_range_bars=20, stop_buffer_atr=0.75,
                  max_zone_width_atr=1.5, max_wait_bars=12, horizon=24,
                  exit_at_opposite_zone=True, higher_minutes=60,
                  scale_to_zone_timeframe=True)
    full = collect_fade_trades(candles, **common)
    cut = collect_fade_trades(candles[:12_000], **common)
    assert cut
    done = [t for t in full if t.bar_index + t.bars_held < 12_000 - 24 - 12]
    keep = {t.bar_index: t for t in cut}
    checked = 0
    for t in done:
        if t.bar_index in keep:
            got = keep[t.bar_index]
            assert (got.entry, got.stop, got.r_multiple) == \
                (t.entry, t.stop, t.r_multiple), t
            checked += 1
    assert checked > 20, checked


def test_the_limit_can_sit_on_the_zone_extreme_itself():
    """利用者の手法: 上位足で帯を見つけ、**その最値**に指値を置く。

    直近 N 本の最値を使うと指値が帯から離れる。実データでは帯から
    2.5 ATR も外へ出て、守るべき帯とは無関係な場所で建玉を持っていた。
    """
    got = trades(horizon=24, higher_minutes=60, exit_at_opposite_zone=True,
                 entry_at_zone_extreme=True)
    assert got
    for t in got:
        edge = t.zone_high if t.from_below else t.zone_low
        assert abs(t.entry - edge) < 1e-9, t


def test_the_zone_extreme_wins_over_the_rolling_window():
    """両方渡されたら帯の極値を採る(窓は使わない)。"""
    got = trades(horizon=24, higher_minutes=60, exit_at_opposite_zone=True,
                 entry_at_zone_extreme=True, entry_from_range_bars=20)
    assert got
    for t in got:
        edge = t.zone_high if t.from_below else t.zone_low
        assert abs(t.entry - edge) < 1e-9, t


def test_the_stop_stays_on_the_lower_timeframe_scale():
    """帯は上位足でも、損切りの物差しは下位足のまま。

    利用者の理由: 損切ラインが遠くならない為と、利確ラインへの伸びが
    大きく期待出来るから。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=0.75, max_zone_width_atr=1.5, max_wait_bars=12,
                  horizon=24, exit_at_opposite_zone=True, higher_minutes=60,
                  entry_at_zone_extreme=True)
    low = collect_fade_trades(candles, **common)
    high = collect_fade_trades(candles, **common, scale_to_zone_timeframe=True)
    assert low and high
    med = lambda ts: sorted(abs(t.stop - t.entry) for t in ts)[len(ts) // 2]
    assert med(low) < med(high), (med(low), med(high))


# --- 映っている範囲の端で張る(利用者が線を引く場所)-----------------------


def test_range_edges_are_the_window_high_and_low():
    """帯 = 直近 N 本の最高値と最安値。スイングの塊ではない。

    利用者が見せてくれた 5 分足・15 分足・1 時間足の 3 枚とも、線は
    その足で表示されている窓の上端と下端に引かれていた。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    got = collect_fade_trades(
        candles, stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
        exit_at_opposite_zone=True, zone_source="range", range_bars=100,
        range_needs_turn=False, entry_at_zone_extreme=True)
    assert got
    for t in got[:200]:
        window = candles[max(0, t.bar_index - 100):t.bar_index]
        if t.from_below:
            assert abs(t.entry - max(c.high for c in window)) < 1e-9, t
        else:
            assert abs(t.entry - min(c.low for c in window)) < 1e-9, t


def test_the_range_edge_never_uses_the_current_bar():
    """その足自身の最値を使うと「更新したから約定した」という循環になる。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    got = collect_fade_trades(
        candles, stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
        exit_at_opposite_zone=True, zone_source="range", range_bars=100,
        range_needs_turn=False, entry_at_zone_extreme=True)
    for t in got[:200]:
        c = candles[t.bar_index]
        if t.from_below:
            assert t.entry <= max(x.high for x in candles[:t.bar_index]), t
        else:
            assert t.entry >= min(x.low for x in candles[:t.bar_index]), t


def test_the_limit_can_be_pushed_beyond_the_edge():
    """利用者の言う「抵抗帯の少し奥(スプレッド対策)に指値を置く」。

    外へ置くほど約定しなくなる。**約定しなければそもそも負けない。**
    """
    common = dict(stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
                  exit_at_opposite_zone=True, zone_source="range",
                  range_bars=100, entry_at_zone_extreme=True)
    candles = generate_synthetic_candles(count=20_000, seed=5)
    at = collect_fade_trades(candles, **common)
    out = collect_fade_trades(candles, **common, entry_beyond_atr=0.5)
    assert at and out
    assert len(out) < len(at), (len(at), len(out))


def test_range_edges_and_pivot_zones_are_different_places():
    """両者は別物。片方で測った結論をもう片方へ持ち込まない。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
                  exit_at_opposite_zone=True, entry_at_zone_extreme=True)
    a = collect_fade_trades(candles, **common, zone_source="pivots")
    b = collect_fade_trades(candles, **common, zone_source="range", range_bars=100)
    assert a and b
    assert {t.entry for t in a} != {t.entry for t in b}


def test_an_unknown_zone_source_is_refused():
    with pytest.raises(ValueError):
        trades(horizon=24, zone_source="whatever")


def test_the_edge_must_be_a_place_price_turned():
    """**いまの動きの端には線を引かない。**折り返した最値だけを使う。

    利用者の説明: 5 分足の下を書かなかったのは、はっきりと直近底値の
    折り返しかわからなかったから。上昇の途中の起点は帯ではない。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
                  exit_at_opposite_zone=True, zone_source="range",
                  range_bars=100, entry_at_zone_extreme=True)
    turn = collect_fade_trades(candles, **common, range_needs_turn=True)
    raw = collect_fade_trades(candles, **common, range_needs_turn=False)
    assert turn and raw
    assert {t.entry for t in turn} != {t.entry for t in raw}
    # 折り返し済みの最値は、窓の最値と同じかその内側に来る
    for t in turn[:150]:
        window = candles[max(0, t.bar_index - 100):t.bar_index]
        if t.from_below:
            assert t.entry <= max(c.high for c in window) + 1e-9, t
        else:
            assert t.entry >= min(c.low for c in window) - 1e-9, t


def test_the_turning_edge_is_a_confirmed_swing():
    """線を引く場所は確定したスイング。未確定を使うと先読みになる。"""
    from llmfx.domain.swings import SwingDetector

    candles = generate_synthetic_candles(count=20_000, seed=5)
    got = collect_fade_trades(
        candles, stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
        exit_at_opposite_zone=True, zone_source="range", range_bars=100,
        range_needs_turn=True, entry_at_zone_extreme=True)
    # **その時点で存在したスイングを集める。**検出器は同じ向きが続くと
    # 末尾を置き換えるので、最後まで回した列には残っていないものがある。
    det = SwingDetector(left=3, right=3, atr_period=14, min_swing_atr=0.6)
    prices, seen = set(), 0
    for c in candles:
        det.update(c)
        # 1 回の更新で 2 つ確定することがあり、末尾だけでは取りこぼす。
        for sw in det.swings[max(0, seen - 1):]:
            prices.add(round(sw.price, 8))
        seen = len(det.swings)
    for t in got[:150]:
        assert round(t.entry, 8) in prices, t


def test_the_window_only_keeps_recent_turns():
    """**何年も前の抵抗帯に意味はない。**窓から外れた折り返しは使わない。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
                  exit_at_opposite_zone=True, zone_source="range",
                  range_needs_turn=True, entry_at_zone_extreme=True)
    short = collect_fade_trades(candles, **common, range_bars=60)
    long_ = collect_fade_trades(candles, **common, range_bars=600)
    assert short and long_
    span = lambda ts: sum(abs(t.opposite_price - t.entry) / t.atr
                          for t in ts if t.opposite_price) / len(ts)
    assert span(long_) > span(short), (span(short), span(long_))


# --- 帯が示した方に乗る ---------------------------------------------------


def test_the_break_side_takes_the_opposite_direction():
    """利用者の説明: 弾かれたら跳ね返りに乗り、抜けたら抜けた側に乗る。

    上の端へ **下から** 来た場合、跳ね返りに乗るなら売り、抜けた側に
    乗るなら買い。`from_below` だけでは向きが決まらない。
    """
    common = dict(stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
                  zone_source="range", range_bars=100, entry_at_zone_extreme=True)
    candles = generate_synthetic_candles(count=20_000, seed=5)
    fade = collect_fade_trades(candles, **common, edge_mode="fade")
    brk = collect_fade_trades(candles, **common, edge_mode="break")
    assert fade and brk
    for t in fade:
        assert t.long_side is not t.from_below, t
    for t in brk:
        assert t.long_side is t.from_below, t


def test_the_break_side_puts_the_stop_back_inside_the_range():
    """抜けた側に乗るなら、損切りは帯の内側へ戻る。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    brk = collect_fade_trades(
        candles, stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
        zone_source="range", range_bars=100, entry_at_zone_extreme=True,
        edge_mode="break")
    assert brk
    for t in brk:
        if t.long_side:
            assert t.stop < t.entry, t
        else:
            assert t.stop > t.entry, t


def test_the_break_side_never_targets_the_opposite_edge():
    """反対側の端は損失方向。決済先にしてはいけない。"""
    brk = collect_fade_trades(
        generate_synthetic_candles(count=20_000, seed=5),
        stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
        zone_source="range", range_bars=100, entry_at_zone_extreme=True,
        edge_mode="break", exit_at_opposite_zone=True)
    assert brk
    assert all(t.opposite_price == 0.0 for t in brk)
    assert not [t for t in brk if t.why == "opp"]


def test_auto_picks_the_break_side_only_when_defenders_are_losing():
    """守り手が押し負けていれば抜けた側、そうでなければ跳ね返り側。

    利用者の指摘: 安値切り上げは見送る材料ではなく、
    **むしろエントリーすべきサイン**。
    """
    common = dict(stop_buffer_atr=0.75, max_wait_bars=12, horizon=24,
                  zone_source="range", range_bars=100, entry_at_zone_extreme=True)
    candles = generate_synthetic_candles(count=20_000, seed=5)
    auto = collect_fade_trades(candles, **common, edge_mode="auto")
    assert auto
    kinds = {t.long_side is t.from_below for t in auto}
    assert kinds == {True, False}, "自動なのに片側しか出ていない"


def test_an_unknown_edge_mode_is_refused():
    with pytest.raises(ValueError):
        trades(horizon=24, edge_mode="whatever")


def test_the_break_can_wait_for_a_close_outside_the_edge():
    """**端に触っただけでは抜けたことにならない。**利用者の指摘:

        指値の位置がブレイクを見てからではなく、折り返し刈り取り用と
        同じ場所をエントリーポイントに選んでしまっている。

    終値が外に出るのを待ってから、次の足の始値で乗る。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  zone_source="range", range_bars=100,
                  entry_at_zone_extreme=True, edge_mode="break")
    touch = collect_fade_trades(candles, **common, break_confirm="touch")
    close = collect_fade_trades(candles, **common, break_confirm="close")
    assert touch and close
    assert len(close) < len(touch), (len(touch), len(close))
    for t in close:
        # 約定は「確認できた足の次の足の始値」
        assert t.entry == candles[t.fill_index].open, t
        # 確認した足は必ず端の外で引けている
        prev = candles[t.fill_index - 1]
        edge = t.zone_high if t.from_below else t.zone_low
        assert (prev.close >= edge) if t.from_below else (prev.close <= edge), t


def test_a_wider_confirmation_asks_for_more_before_entering():
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  zone_source="range", range_bars=100,
                  entry_at_zone_extreme=True, edge_mode="break",
                  break_confirm="close")
    near = collect_fade_trades(candles, **common, break_confirm_atr=0.0)
    far = collect_fade_trades(candles, **common, break_confirm_atr=0.6)
    assert near and far
    assert len(far) < len(near), (len(near), len(far))


def test_confirming_the_break_removes_same_bar_decisions():
    """確認を待つと、約定した足の中で決着する取引がほぼ消える。

    道順の仮定に成績が乗らなくなるので、測定としても素直になる。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    close = collect_fade_trades(
        candles, stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
        zone_source="range", range_bars=100, entry_at_zone_extreme=True,
        edge_mode="break", break_confirm="close")
    same = [t for t in close if t.bars_held == 0]
    assert len(same) / len(close) < 0.05, len(same) / len(close)


def test_the_fade_side_ignores_the_break_confirmation():
    """確認は抜けた側だけの話。跳ね返り側の挙動を変えない。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  zone_source="range", range_bars=100,
                  entry_at_zone_extreme=True, edge_mode="fade")
    a = collect_fade_trades(candles, **common, break_confirm="touch")
    b = collect_fade_trades(candles, **common, break_confirm="close")
    assert a and [t.entry for t in a] == [t.entry for t in b]


def test_an_unknown_break_confirm_is_refused():
    with pytest.raises(ValueError):
        trades(horizon=24, break_confirm="whatever")


# --- 押し負けが起きた瞬間に乗る -------------------------------------------


def test_the_weakening_entry_does_not_wait_for_the_edge():
    """利用者の指摘: **抵抗帯の押し負けが発生した時点で乗る。**

    帯へ届くのを待たないので、建玉を持つ位置は帯から離れている。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  higher_minutes=60, zone_source="range", range_bars=120,
                  entry_at_zone_extreme=True)
    got = collect_fade_trades(candles, **common, edge_mode="weakening")
    assert got
    for t in got:
        edge = t.zone_high if t.long_side else t.zone_low
        assert abs(t.entry - edge) > 1e-9, "帯の上で建玉を持っている"
        assert t.defenders_weak, t


def test_the_weakening_entry_needs_a_zone_being_pressured():
    """押される相手が無ければ乗らない。ただの高値切り上げ買いではない。"""
    got = collect_fade_trades(
        generate_synthetic_candles(count=20_000, seed=5),
        stop_buffer_atr=1.5, max_wait_bars=12, horizon=240, higher_minutes=60,
        zone_source="range", range_bars=120, entry_at_zone_extreme=True,
        edge_mode="weakening")
    for t in got:
        if t.long_side:
            assert t.zone_price > t.entry, t
        else:
            assert t.zone_price < t.entry, t


def test_the_weakening_stop_sits_beyond_the_structure():
    """損切りは切り上がった安値の下(買い)/ 切り下がった高値の上(売り)。"""
    got = collect_fade_trades(
        generate_synthetic_candles(count=20_000, seed=5),
        stop_buffer_atr=1.5, max_wait_bars=12, horizon=240, higher_minutes=60,
        zone_source="range", range_bars=120, entry_at_zone_extreme=True,
        edge_mode="weakening")
    assert got
    for t in got:
        if t.long_side:
            assert t.stop < t.entry, t
        else:
            assert t.stop > t.entry, t


def test_more_open_positions_take_more_of_the_chances():
    """利用者の指摘:

        レンジの往復を取る時に建玉があると指値を入れられない
        = 機会損失となりうる。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  higher_minutes=60, zone_source="range", range_bars=120,
                  entry_at_zone_extreme=True, edge_mode="break")
    one = collect_fade_trades(candles, **common, max_open=1)
    many = collect_fade_trades(candles, **common, max_open=4)
    assert len(many) > len(one) * 1.5, (len(one), len(many))
    # 1 建玉のときの取引は、多建玉のときにも必ず含まれる
    keys = {(t.fill_index, round(t.entry, 8)) for t in many}
    assert all((t.fill_index, round(t.entry, 8)) in keys for t in one)


def test_max_open_must_be_at_least_one():
    with pytest.raises(ValueError):
        trades(horizon=24, max_open=0)


def test_widening_the_watch_band_finds_setups_that_never_reached_the_edge():
    """**触れた足しか見ないと、手前に置いた指値は意味を持たない。**

    利用者の指摘: 強固な帯なら、届く前に折り返して約定しないことがある。
    その場面はそもそも候補に入らないので、指値の位置を変えても件数が
    動かない(実測で 1,531 → 1,526)。見る範囲も一緒に広げる。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    # **建玉の枠で頭打ちにならないようにする。**枠が 1 だと、保有中に
    # 起きた場面は数えられないので、見る範囲を広げても件数が動かない。
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=24,
                  higher_minutes=60, zone_source="range", range_bars=120,
                  entry_at_zone_extreme=True, edge_mode="fade",
                  exit_at_opposite_zone=True, entry_beyond_atr=-0.5, max_open=8)
    tight = collect_fade_trades(candles, **common, arm_within_atr=0.0)
    wide = collect_fade_trades(candles, **common, arm_within_atr=0.5)
    assert tight and wide
    assert len(wide) > len(tight) * 1.1, (len(tight), len(wide))


def test_the_watch_band_does_not_change_where_the_limit_sits():
    """広げるのは「見る範囲」だけ。指値の位置は `entry_beyond_atr` が決める。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    got = collect_fade_trades(
        candles, stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
        higher_minutes=60, zone_source="range", range_bars=120,
        entry_at_zone_extreme=True, edge_mode="fade", exit_at_opposite_zone=True,
        entry_beyond_atr=-0.5, arm_within_atr=0.5)
    assert got
    for t in got[:100]:
        edge = t.zone_high if t.from_below else t.zone_low
        gap = (edge - t.entry) if t.from_below else (t.entry - edge)
        assert gap > 0, t          # 端より手前にある
        assert abs(gap - 0.5 * t.atr) < 1e-6, t


def test_a_broken_edge_is_dropped_along_with_older_turns():
    """利用者の説明:

        線が途中で消えているものは、抵抗帯を抜けてしまっているので、
        流石に戻ってこなそうだなと感じたもの → エントリー中止。

    抜けられた端も、それ以前に付けた折り返しも捨てる。切らないと、
    **一度抜けた水準へ価格が戻ってきたときに古い指値が生き残る。**
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  higher_minutes=60, zone_source="range", range_bars=120,
                  entry_at_zone_extreme=True, edge_mode="fade",
                  exit_at_opposite_zone=True, max_open=4)
    keep = collect_fade_trades(candles, **common, drop_broken_edges=False)
    drop = collect_fade_trades(candles, **common, drop_broken_edges=True)
    assert keep and drop
    assert len(drop) < len(keep), (len(keep), len(drop))


def test_dropping_broken_edges_never_uses_a_level_price_closed_beyond():
    """捨てた端では二度と建玉を持たない。"""
    from datetime import timedelta

    from llmfx.data.resample import resample_candles

    candles = generate_synthetic_candles(count=20_000, seed=5)
    got = collect_fade_trades(
        candles, stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
        higher_minutes=60, zone_source="range", range_bars=120,
        entry_at_zone_extreme=True, edge_mode="fade", exit_at_opposite_zone=True,
        max_open=4, drop_broken_edges=True)
    assert got
    h1 = resample_candles(candles, 60)
    checked = 0
    for t in got[:150]:
        # **その折り返しが付いた後**に終値で越えていたら、捨てられている
        # はず。付く前の値動きは関係ない(そこにはまだ線が無い)。
        if not t.zone_touch_bars:
            continue
        born = t.zone_touch_bars[0]
        level = t.entry
        # **閉じた上位足だけを見る。**08:00 の H1 は 09:00 に閉じるので、
        # 08:15 に約定した時点ではまだ確定していない。
        end = candles[t.fill_index].time
        window = [c for c in h1[born + 1:] if c.time + timedelta(hours=1) <= end]
        broken = [c for c in window
                  if (c.close > level if t.from_below else c.close < level)]
        assert not broken, (t.entry, len(broken), broken[-1].time)
        checked += 1
    assert checked > 30, checked


def test_the_break_is_judged_on_the_bar_the_zone_was_drawn_on():
    """**下位足の終値で見ると、はみ出しただけで捨ててしまう。**

    帯は上位足で引いているので、抜けたかどうかも上位足の終値で見る。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  zone_source="range", range_bars=120, entry_at_zone_extreme=True,
                  edge_mode="fade", exit_at_opposite_zone=True, max_open=4,
                  drop_broken_edges=True)
    # 上位足を使う場合と使わない場合で、捨てる頻度が変わる
    with_h = collect_fade_trades(candles, **common, higher_minutes=60)
    without = collect_fade_trades(candles, **common)
    assert with_h and without


def test_the_limit_can_be_tuned_to_the_most_recent_fold():
    """利用者の指定:

        大枠は 1 週間で良いが、その後のエントリーは必ず直近の折り目で
        微調整していかないとエントリー数も稼げないし勿体ない。

    1 週間の最値は遠いことが多く、そこまで戻ってこないと約定しない。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  higher_minutes=60, zone_source="range", range_bars=120,
                  entry_at_zone_extreme=True, edge_mode="fade",
                  exit_at_opposite_zone=True, max_open=4)
    window = collect_fade_trades(candles, **common, entry_from_recent_turn=False)
    fold = collect_fade_trades(candles, **common, entry_from_recent_turn=True)
    assert window and fold
    assert len(fold) > len(window) * 1.4, (len(window), len(fold))


def test_the_fold_never_sits_outside_the_weekly_frame():
    """折り目へ寄せるのは **内側** へだけ。大枠より外には出ない。"""
    candles = generate_synthetic_candles(count=20_000, seed=5)
    fold = collect_fade_trades(
        candles, stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
        higher_minutes=60, zone_source="range", range_bars=120,
        entry_at_zone_extreme=True, edge_mode="fade", exit_at_opposite_zone=True,
        max_open=4, entry_from_recent_turn=True)
    assert fold
    for t in fold:
        if not t.opposite_price:
            continue
        # 反対側の相手は大枠なので、指値より必ず遠い
        gap = ((t.entry - t.opposite_price) if t.from_below
               else (t.opposite_price - t.entry))
        assert gap > 0, t


def test_the_round_trip_target_stays_on_the_weekly_frame():
    """指値だけ折り目へ寄せ、往復の相手は大枠のまま。

    利確まで折り目にすると、取りに行く幅が消えてしまう。
    """
    candles = generate_synthetic_candles(count=20_000, seed=5)
    common = dict(stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
                  higher_minutes=60, zone_source="range", range_bars=120,
                  entry_at_zone_extreme=True, edge_mode="fade",
                  exit_at_opposite_zone=True, max_open=4)
    window = collect_fade_trades(candles, **common, entry_from_recent_turn=False)
    fold = collect_fade_trades(candles, **common, entry_from_recent_turn=True)
    span = lambda ts: sorted(abs(t.opposite_price - t.entry) / t.atr
                             for t in ts if t.opposite_price)
    a, b = span(window), span(fold)
    assert a and b
    # 指値が内側へ寄るぶん、**反対側の大枠までは近くなる**。
    # 取りに行く幅は減るが、相手は大枠のままなので折り目より遠い。
    assert b[len(b) // 2] < a[len(a) // 2], (a[len(a) // 2], b[len(b) // 2])
