"""第4版 — 抵抗帯で入り、ダウ転換で手仕舞ってドテンする.

第3版(`zone_fade.py`)との違いは 4 つ。

| | 第3版 | 第4版 |
| --- | --- | --- |
| 構造を読む足 | 読んでいない | **上位足(既定 H1)** |
| 出口 | 損切り / 反対側の帯 / 時間切れ | **ダウ転換 / 損切り** だけ |
| 転換したら | 何もしない | **手仕舞って、転換ラインの注文でドテン** |
| 損切りの移動 | 順行 0.05 R で建値へ | **折り返すたび「1 つ前」の高値・安値へ** |

**成行を使わない。**線は先に分かっているので、そこへ注文を置いておく。
届かなければ何も起きない。利用者の言葉:

    指値でダウ転換ラインを指せるならその方が良い。成り行きだと
    スプレッドが急に離れた時に困る。

**足が 2 つに分かれる。**帯もダウ転換も損切りの位置も上位足で決め、
下位足はそこへ置いた注文を約定させるだけ:

    ダウ転換自体は同じ足で見る。しかし、エントリーポイントは
    2 つ下の足で見ないと意味がなくなる。

過去に、構造を下位足で追ったら平均勝ちが 4.95 R から 1.02 R へ潰れた。
**この手法がわざわざ座って待っている調整波を、自分で「転換」と読む**ため。

**建玉ごとに、どの機構で生まれたかを記録する**(`SwingLeg.kind`)。
合算の 1 行だけを出すと、どれが効いてどれが足を引っ張っているのか
分からなくなる(利用者の指摘)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.swings import SwingDetector
from ..domain.types import Candle, SwingType
from .zone_fade import _atr_series


@dataclass
class SwingLeg:
    """1 回の約定から決済までの記録。**建玉ではなく約定単位。**

    ドテンも買い増しも「別の約定」なので別の脚になる。こうしないと
    機構ごとに損益を割れない。
    """

    kind: str
    """どの機構で生まれたか。`zone`(帯の指値) / `flip`(ドテン) / `add`(買い増し)。"""

    long_side: bool
    entry_index: int
    entry: float
    stop_at_entry: float
    risk: float
    """約定した時点の |損切り - 約定値|。**後で損切りが動いても分母は変えない。**"""

    atr: float
    size: float
    exit_index: int
    exit: float
    r_multiple: float
    why: str
    """`stop` / `reversal` / `end`(データの終端)。"""

    bars_held: int
    max_favourable_r: float
    max_adverse_r: float
    position_id: int
    zone_price: float
    flips: int
    """その建玉が何回ドテンした後か。0 が最初。"""

    adds: int
    """その建玉が何回買い増しした後か。0 が最初。"""


@dataclass
class _Leg:
    kind: str
    entry_index: int
    entry: float
    stop_at_entry: float
    risk: float
    size: float
    best: float = 0.0
    worst: float = 0.0


@dataclass
class _Position:
    pid: int
    long_side: bool
    stop: float
    atr: float
    zone_price: float
    zone_key: str
    flips: int = 0
    adds: int = 0
    legs: list[_Leg] = field(default_factory=list)
    add_swing: int = -1
    """買い増しに使った折り返しの足番号。同じ折り返しで二度足さない。"""

    anchor: int = -1
    """約定した時点の、守る側の折り返しの足番号。

    **これより新しい折り返しが確定するまで損切りを動かさない。**
    動かしてしまうと、入る前から在った構造でいきなり詰めることになり、
    「エントリー時の損切り位置は問題なし」という合意と食い違う。
    """


def collect_swing_trades(
    candles: list[Candle],
    *,
    higher_minutes: int = 60,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.6,
    range_bars: int = 120,
    entry_beyond_atr: float = 0.0,
    stop_buffer_atr: float = 1.5,
    swing_stop_buffer_atr: float = 0.0,
    reversal_signal: str = "both",
    max_flips: int = 1,
    max_adds: int = 0,
    add_size: float = 1.0,
    max_open: int = 4,
    rearm_atr: float = 1.0,
    blocked_hours_utc: frozenset[int] | None = None,
    spread: float = 0.0,
    slippage: float = 0.0,
    warmup: int = 200,
) -> list[SwingLeg]:
    """帯へ指値を置き、ダウ転換で手仕舞ってドテンする。

    `candles` は執行する足(既定は M15)。`higher_minutes` はそれを
    まとめる足で、**帯・ダウ転換・損切りの位置はすべてこちらで決める**。

    `range_bars` は帯を引く窓。上位足での本数で数える(120 本 = 1 週間)。
    使うのは **その窓の中で最も高い「折り返した」高値** と、最も低い
    「折り返した」安値。いま進行中の動きの端は使わない。

    `entry_beyond_atr` は指値を帯の最値から **外側へ** どれだけずらすか。
    利用者の指定「抵抗帯の少し奥(スプレッド対策)」なので + が本命だが、
    奥へ置くほど約定しないので掃引する軸。0 なら線ちょうど。

    `stop_buffer_atr` は最初の損切りを帯の外側へどれだけ離すか。
    `swing_stop_buffer_atr` は **折り返しへ預けた後** の余裕。

    `reversal_signal` は転換とみなす条件:

    | 値 | 売りを閉じる条件 |
    | --- | --- |
    | `high_only` | 直近の確定した高値を上抜けた |
    | `both` | 上に加えて、安値が既に切り上がっている |

    どちらも **上抜ける水準は同じ**(直近の確定高値)なので、注文は
    どちらでも置いておける。違うのは「置くかどうか」だけ。

    `max_flips` はドテンできる回数。0 なら転換で閉じるだけ。
    `max_adds` は買い増しの回数、`add_size` は 1 回あたりの量
    (最初を 1.0 とした比)。

    `max_open` は同時に持てる建玉。**時間切れを外したので枠が長く
    埋まる。**利用者の指摘どおり掃引する軸。

    `spread` / `slippage` は価格で渡す。**コストは差し引きではなく
    発動位置のずれとして入る**(`zone_fade` と同じ扱い)。返る
    `r_multiple` は既に差引後なので、集計側で二重に引かないこと。

    足の中の道順は **損切りを先に見る**。順序が分からない以上、
    同じ足で転換ラインにも届いていたら損切りを優先する。ここは
    不利側なので、成績が良く出る方向へは倒れない。
    """
    if reversal_signal not in ("both", "high_only"):
        raise ValueError(f"reversal_signal が不正: {reversal_signal!r}")
    if max_open < 1:
        raise ValueError("max_open は 1 以上")
    if not candles or higher_minutes is None or higher_minutes <= 0:
        return []

    from ..data.resample import resample_candles

    higher = resample_candles(candles, higher_minutes)
    if len(higher) < max(range_bars, warmup // 4, left + right + 2):
        return []
    atr_low = _atr_series(candles, atr_period)

    det = SwingDetector(left=left, right=right, atr_period=atr_period,
                        min_swing_atr=min_swing_atr)
    from datetime import timedelta

    span = timedelta(minutes=higher_minutes)
    hi_i = 0
    hi_bar = -1

    out: list[SwingLeg] = []
    positions: list[_Position] = []
    armed: dict[str, bool] = {"top": True, "bottom": True}
    pid_seq = 0
    zone_hi: tuple[int, float] | None = None
    zone_lo: tuple[int, float] | None = None

    def structure() -> dict:
        """確定した高値・安値を新しい順に 2 本ずつ。"""
        return {
            "last_high": det.nth_last_swing(SwingType.HIGH, 1),
            "prev_high": det.nth_last_swing(SwingType.HIGH, 2),
            "last_low": det.nth_last_swing(SwingType.LOW, 1),
            "prev_low": det.nth_last_swing(SwingType.LOW, 2),
        }

    def window_edges(cut: int) -> tuple[tuple[int, float] | None,
                                        tuple[int, float] | None]:
        """窓の中で最も高い折り返しの高値と、最も低い折り返しの安値。

        **確定したスイングだけを見る。**新しい方から窓を出るまで辿るので、
        走るのは窓の中の本数ぶんだけ。列を差分で読むと、同じ向きが続いた
        ときの **置き換え** を取りこぼすので、毎回辿り直す。
        """
        hi = lo = None
        for sw in reversed(det.swings):
            if sw.index < cut:
                break
            if sw.type is SwingType.HIGH:
                if hi is None or sw.price > hi[1]:
                    hi = (sw.index, sw.price)
            elif lo is None or sw.price < lo[1]:
                lo = (sw.index, sw.price)
        return hi, lo

    def close_position(pos: _Position, i: int, price: float, why: str,
                       slip: float) -> None:
        sign = 1.0 if pos.long_side else -1.0
        for leg in pos.legs:
            r = ((price - leg.entry) * sign - slip) / leg.risk
            out.append(SwingLeg(
                kind=leg.kind, long_side=pos.long_side,
                entry_index=leg.entry_index, entry=leg.entry,
                stop_at_entry=leg.stop_at_entry, risk=leg.risk, atr=pos.atr,
                size=leg.size, exit_index=i, exit=price, r_multiple=r, why=why,
                bars_held=i - leg.entry_index,
                max_favourable_r=leg.best / leg.risk,
                max_adverse_r=leg.worst / leg.risk,
                position_id=pos.pid, zone_price=pos.zone_price,
                flips=pos.flips, adds=pos.adds))
        pos.legs = []

    for i, candle in enumerate(candles):
        # --- 閉じた上位足だけを取り込む ---------------------------------
        moved = False
        while hi_i < len(higher) and higher[hi_i].time + span <= candle.time:
            det.update(higher[hi_i])
            hi_bar = hi_i
            hi_i += 1
            moved = True
        if hi_bar < 0 or i < warmup:
            continue
        if moved:
            zone_hi, zone_lo = window_edges(hi_bar - range_bars)

        a = atr_low[i]
        if a <= 0:
            continue
        st = structure()

        # --- 損切りを「1 つ前」の折り返しへ引き上げる --------------------
        # **最新ではなく 1 つ前。**最新に置くと、トレンドが壊れていない
        # 普通のより戻しで刺さる(利用者の指摘)。動くのは有利な向きだけ。
        if moved:
            for pos in positions:
                side = st["last_low"] if pos.long_side else st["last_high"]
                prev = st["prev_low"] if pos.long_side else st["prev_high"]
                # **入ってから新しい折り返しが出るまでは動かさない。**
                if side is None or prev is None or side.index == pos.anchor:
                    continue
                if pos.long_side:
                    lvl = prev.price - swing_stop_buffer_atr * a
                    # 逆側へは置けない。いまの値段の向こう側は損切りにならない。
                    if lvl < candle.close:
                        pos.stop = max(pos.stop, lvl)
                else:
                    lvl = prev.price + swing_stop_buffer_atr * a
                    if lvl > candle.close:
                        pos.stop = min(pos.stop, lvl)

        # --- いま持っている建玉を回す -----------------------------------
        for pos in list(positions):
            sign = 1.0 if pos.long_side else -1.0
            shift = 0.0 if pos.long_side else spread
            for leg in pos.legs:
                fav = max((candle.high - leg.entry) * sign,
                          (candle.low - leg.entry) * sign) - shift
                adv = -min((candle.high - leg.entry) * sign,
                           (candle.low - leg.entry) * sign) + shift
                leg.best = max(leg.best, fav)
                leg.worst = max(leg.worst, adv)

            # 1. 損切り。**同じ足で転換ラインにも届いていたらこちらが先。**
            hit_stop = ((candle.low <= pos.stop) if pos.long_side
                        else (candle.high >= pos.stop - spread))
            if hit_stop:
                close_position(pos, i, pos.stop, "stop", slippage)
                positions.remove(pos)
                continue

            # 2. ダウ転換。上抜ける(下抜ける)水準は直近の確定した折り返し。
            rev = st["last_low"] if pos.long_side else st["last_high"]
            ok = rev is not None
            if ok and reversal_signal == "both":
                # 高値も切り下がって(切り上がって)いること。
                if pos.long_side:
                    ok = (st["last_high"] is not None and st["prev_high"] is not None
                          and st["last_high"].price < st["prev_high"].price)
                else:
                    ok = (st["last_low"] is not None and st["prev_low"] is not None
                          and st["last_low"].price > st["prev_low"].price)
            if ok:
                line = rev.price
                # **抜けたときだけ。**転換ラインは逆指値なので、price が
                # 既に向こう側にあるなら発動しない。ここを見ないと、帯で
                # 建てた瞬間に「直近の高値へ届いている」ことになり、
                # 建てた足でいきなり利益確定する。
                prior = candles[i - 1].close if i else candle.open
                crossed = (prior > line) if pos.long_side else (prior < line)
                reached = crossed and ((candle.low <= line) if pos.long_side
                                       else (candle.high >= line - spread))
                if reached:
                    close_position(pos, i, line, "reversal", 0.0)
                    positions.remove(pos)
                    if pos.flips < max_flips:
                        # ドテン。損切りは **その 1 つ前の折り返し** の外側。
                        prot = st["last_high"] if pos.long_side else st["last_low"]
                        if prot is None:
                            continue
                        nl = not pos.long_side
                        stop = (prot.price - swing_stop_buffer_atr * a if nl
                                else prot.price + swing_stop_buffer_atr * a)
                        risk = abs(line - stop)
                        if risk <= 0 or ((line <= stop) if nl else (line >= stop)):
                            continue
                        pid_seq += 1
                        positions.append(_Position(
                            pid=pid_seq, long_side=nl, stop=stop, atr=a,
                            zone_price=pos.zone_price, zone_key=pos.zone_key,
                            flips=pos.flips + 1, anchor=prot.index,
                            # **乗り換えに使った折り返しでは買い増ししない。**
                            # 同じ水準・同じ足で 2 枚持つことになる。
                            add_swing=rev.index,
                            legs=[_Leg("flip", i, line, stop, risk, 1.0)]))
                    continue

            # 3. 買い増し。**流れが続く側の折り返しを抜けたら足す。**
            if pos.adds < max_adds:
                go = st["last_high"] if pos.long_side else st["last_low"]
                if go is not None and go.index != pos.add_swing:
                    line = go.price
                    # 買い増しも同じ。抜けたときだけ足す。
                    prior = candles[i - 1].close if i else candle.open
                    crossed = (prior < line) if pos.long_side else (prior > line)
                    reached = crossed and ((candle.high >= line - spread)
                                           if pos.long_side
                                           else (candle.low <= line))
                    if reached:
                        risk = abs(line - pos.stop)
                        if risk > 0:
                            pos.legs.append(_Leg("add", i, line, pos.stop, risk,
                                                 add_size))
                            pos.adds += 1
                            pos.add_swing = go.index

        # --- 新しく帯へ置いた指値 ---------------------------------------
        if blocked_hours_utc and candle.time.hour in blocked_hours_utc:
            continue
        for key, edge in (("top", zone_hi), ("bottom", zone_lo)):
            if edge is None:
                continue
            level = edge[1]
            long_side = key == "bottom"
            limit = (level - entry_beyond_atr * a if long_side
                     else level + entry_beyond_atr * a)
            # 帯から離れたら次の待ち伏せを許す(同じ水準で連射しない)。
            if abs(candle.close - level) > rearm_atr * a:
                armed[key] = True
                continue
            if not armed[key] or len(positions) >= max_open:
                continue
            probe = limit - (spread if long_side else 0.0)
            reached = ((candle.low <= probe) if long_side
                       else (candle.high >= probe))
            if not reached:
                continue
            stop = (limit - stop_buffer_atr * a if long_side
                    else limit + stop_buffer_atr * a)
            risk = abs(stop - limit)
            if risk <= 0:
                continue
            armed[key] = False
            pid_seq += 1
            base = st["last_low"] if long_side else st["last_high"]
            positions.append(_Position(
                pid=pid_seq, long_side=long_side, stop=stop, atr=a,
                zone_price=level, zone_key=key,
                anchor=base.index if base is not None else -1,
                legs=[_Leg("zone", i, limit, stop, risk, 1.0)]))

    # --- 終端。**残った建玉は成行で閉じ、印を付けて分けて数える。** ------
    last = len(candles) - 1
    for pos in positions:
        shift = 0.0 if pos.long_side else spread
        close_position(pos, last, candles[last].close - (shift + slippage)
                       * (1.0 if pos.long_side else -1.0), "end", 0.0)
    return out
