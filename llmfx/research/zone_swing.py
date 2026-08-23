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
from .zone_fade import _after_fill, _atr_series, _bar_path


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
    zone_key: str
    """`top`(抵抗帯)か `bottom`(支持帯)か。帯のどちら側から来たかで
    足の中の道順が変わるので、後から確かめるのに要る。"""

    line_at_entry: float
    """約定した時点の転換ライン(利確側)。0 なら まだ無い。

    **損切りより手前になければ、利確側の出口が存在しない。**
    ドテンした建玉でここが揃っていなかった。
    """

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
    line_at_entry: float = 0.0
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
    add_swing: tuple = ()
    """買い増しに使った折り返し(どの足の、何本目か)。二度足さない。"""

    anchor: int = -1
    """約定した時点の、守る側の折り返しの足番号。

    **これより新しい折り返しが確定するまで損切りを動かさない。**
    動かしてしまうと、入る前から在った構造でいきなり詰めることになり、
    「エントリー時の損切り位置は問題なし」という合意と食い違う。
    """


def collect_swing_trades(
    candles: list[Candle],
    *,
    zone_minutes: int = 240,
    structure_minutes: int = 60,
    higher_minutes: int | None = None,
    entry_signal: str = "exec",
    zone_entry: str = "extreme",
    zone_wait_bars: int = 24,
    zone_entry_max_atr: float = 2.0,
    entry_fill: str = "level",
    entry_fallback: str = "structure",
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
    reverse_entry: bool = False,
    fill_bar: str = "path",
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

    `fill_bar` は **約定した足の中で損切りへ届いたか** をどう見るか。
    `path` は他所と同じ推し量り(陽線 始値→安値→高値→終値)で約定より
    後だけを見る。`adverse` は不利側の端を必ず数える(不利側の端)。
    **損切りを詰めるほど、結論はこの仮定だけで決まる。**損切りが足 1 本の
    値幅の内側に入るため。掃引で端が最良に見えたら、まずここで挟む。

    足の中の道順は **損切りを先に見る**。順序が分からない以上、
    同じ足で転換ラインにも届いていたら損切りを優先する。ここは
    不利側なので、成績が良く出る方向へは倒れない。
    """
    if fill_bar not in ("path", "adverse"):
        raise ValueError(f"fill_bar が不正: {fill_bar!r}")
    if reversal_signal not in ("both", "high_only"):
        raise ValueError(f"reversal_signal が不正: {reversal_signal!r}")
    if zone_entry not in ("extreme", "exec_turn"):
        raise ValueError(f"zone_entry が不正: {zone_entry!r}")
    if entry_signal not in ("exec", "structure"):
        raise ValueError(f"entry_signal が不正: {entry_signal!r}")
    if entry_fill not in ("level", "next_open"):
        raise ValueError(f"entry_fill が不正: {entry_fill!r}")
    if entry_fallback not in ("structure", "skip"):
        raise ValueError(f"entry_fallback が不正: {entry_fallback!r}")
    if max_open < 1:
        raise ValueError("max_open は 1 以上")
    if higher_minutes:                       # 旧来の 2 段。両方を同じ足にする
        zone_minutes = structure_minutes = higher_minutes
    if not candles or not zone_minutes or not structure_minutes:
        return []

    from datetime import timedelta

    from ..data.resample import resample_candles

    zone_tf = resample_candles(candles, zone_minutes)
    struct_tf = (zone_tf if structure_minutes == zone_minutes
                 else resample_candles(candles, structure_minutes))
    if len(zone_tf) < max(range_bars, left + right + 2):
        return []
    if len(struct_tf) < left + right + 2:
        return []
    atr_low = _atr_series(candles, atr_period)

    def _det() -> SwingDetector:
        return SwingDetector(left=left, right=right, atr_period=atr_period,
                             min_swing_atr=min_swing_atr)

    zdet = _det()                            # 帯を引く足の折り返し
    sdet = zdet if struct_tf is zone_tf else _det()   # 構造の足
    xdet = _det()                            # 執行の足(入り口の合図)

    zone_span = timedelta(minutes=zone_minutes)
    struct_span = timedelta(minutes=structure_minutes)
    z_i, z_bar = 0, -1
    s_i, s_bar = 0, -1

    out: list[SwingLeg] = []
    positions: list[_Position] = []
    armed: dict[str, bool] = {"top": True, "bottom": True}
    touched: dict[str, int | None] = {"top": None, "bottom": None}
    pid_seq = 0
    zone_hi: tuple[int, float] | None = None
    zone_lo: tuple[int, float] | None = None

    def structure(d: SwingDetector) -> dict:
        """確定した高値・安値を新しい順に 2 本ずつ。"""
        return {
            "last_high": d.nth_last_swing(SwingType.HIGH, 1),
            "prev_high": d.nth_last_swing(SwingType.HIGH, 2),
            "last_low": d.nth_last_swing(SwingType.LOW, 1),
            "prev_low": d.nth_last_swing(SwingType.LOW, 2),
        }

    def window_edges(cut: int) -> tuple[tuple[int, float] | None,
                                        tuple[int, float] | None]:
        """窓の中で最も高い折り返しの高値と、最も低い折り返しの安値。

        **確定したスイングだけを見る。**新しい方から窓を出るまで辿るので、
        走るのは窓の中の本数ぶんだけ。列を差分で読むと、同じ向きが続いた
        ときの **置き換え** を取りこぼすので、毎回辿り直す。
        """
        hi = lo = None
        for sw in reversed(zdet.swings):
            if sw.index < cut:
                break
            if sw.type is SwingType.HIGH:
                if hi is None or sw.price > hi[1]:
                    hi = (sw.index, sw.price)
            elif lo is None or sw.price < lo[1]:
                lo = (sw.index, sw.price)
        return hi, lo

    def stopped_on_fill_bar(pos: _Position, i: int, fill: float,
                            from_below: bool) -> bool:
        """約定した足の **残り** で損切りへ届いていないか。

        建玉を翌足からしか見ないと、**損切りが狭いほど「同じ足で切られた
        はずの負け」を見逃す。**実測で損切りを 1.5 → 0.3 ATR と詰めると
        期待値が +0.067 → +0.801、平均勝ちが +4.50 → +22.49 R になった。
        分母が縮んだのではなく、負けが消えていた。

        足の中の道順は四本値からは分からないので、他所と同じ推し量り方
        (陽線 始値→安値→高値→終値)で解く。**約定より前の値動きは
        使わない。**
        """
        c = candles[i]
        if fill_bar == "adverse":
            # **不利側の端を必ず数える。**約定より前に付いていたかも
            # しれないが、順序が分からない以上こちらが不利側の端。
            hit = (c.low <= pos.stop if pos.long_side
                   else c.high >= pos.stop - spread)
        else:
            pts = _after_fill(_bar_path(c), fill, from_below)
            if not pts:
                return False
            hit = (min(pts) <= pos.stop if pos.long_side
                   else max(pts) >= pos.stop - spread)
        if not hit:
            return False
        close_position(pos, i, pos.stop, "stop", slippage)
        return True

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
                zone_key=pos.zone_key, line_at_entry=leg.line_at_entry,
                flips=pos.flips, adds=pos.adds))
        pos.legs = []

    for i, candle in enumerate(candles):
        # --- 閉じた足だけを取り込む。**足ごとに独立に進める。** ----------
        z_moved = s_moved = False
        while z_i < len(zone_tf) and zone_tf[z_i].time + zone_span <= candle.time:
            zdet.update(zone_tf[z_i])
            z_bar = z_i
            z_i += 1
            z_moved = True
        if sdet is not zdet:
            while (s_i < len(struct_tf)
                   and struct_tf[s_i].time + struct_span <= candle.time):
                sdet.update(struct_tf[s_i])
                s_bar = s_i
                s_i += 1
                s_moved = True
        else:
            s_bar, s_moved = z_bar, z_moved
        # 執行の足は 1 本前まで(その足自身の最値で判定すると循環する)。
        if i:
            xdet.update(candles[i - 1])
        if z_bar < 0 or s_bar < 0 or i < warmup:
            continue
        if z_moved:
            zone_hi, zone_lo = window_edges(z_bar - range_bars)

        a = atr_low[i]
        if a <= 0:
            continue
        st = structure(sdet)
        ex = structure(xdet)
        moved = s_moved

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

            # --- 損切りと転換ラインは **どちらも同じ側** にある ------------
            # 売りなら両方が上、買いなら両方が下。価格は片側から来るので
            # **近いほうが先に着く。**順序が分からないのは反対側どうしの
            # 水準だけで、ここには曖昧さが無い。
            #
            # ここで一律に損切りを先に見ていた。実際には先に届いていた
            # 転換ラインでの手仕舞いを、毎回 -1 R の損切りへ振り替える
            # ことになる。**負け側にだけ寄る誤り。**
            stop_hit = ((candle.low <= pos.stop) if pos.long_side
                        else (candle.high >= pos.stop - spread))

            def crossed_now(level: float, against: bool) -> bool:
                """前の足の終値が向こう側にないこと + この足で届いたこと。

                `against` は建玉に不利な向き(転換side)かどうか。
                """
                prior = candles[i - 1].close if i else candle.open
                down = pos.long_side if against else not pos.long_side
                if down:
                    return prior > level and candle.low <= level
                return prior < level and candle.high >= level - spread

            # --- 転換 -----------------------------------------------------
            # **上位足(構造)が向きを変えたかで門を開け、執行の足の
            # 折り返しで引き金を引く。**構造の水準まで待つと、そこへ
            # 届くまでの値幅を丸ごと捨てる(利用者の指摘)。
            gate = True
            if reversal_signal == "both":
                if pos.long_side:
                    gate = (st["last_high"] is not None
                            and st["prev_high"] is not None
                            and st["last_high"].price < st["prev_high"].price)
                else:
                    gate = (st["last_low"] is not None
                            and st["prev_low"] is not None
                            and st["last_low"].price > st["prev_low"].price)
            picks: list[tuple[float, str, object]] = []
            if gate:
                if entry_signal == "exec":
                    xr = ex["last_low"] if pos.long_side else ex["last_high"]
                    if xr is not None and crossed_now(xr.price, True):
                        picks.append((xr.price, "exec", xr))
                sr = st["last_low"] if pos.long_side else st["last_high"]
                if sr is not None and crossed_now(sr.price, True):
                    picks.append((sr.price, "structure", sr))
            rev_hit = bool(picks)
            line, source, rev = 0.0, "", None
            if rev_hit:
                # 手前にあるほうが先に着く。買いは高いほう、売りは低いほう。
                line, source, rev = (max(picks, key=lambda z: z[0]) if pos.long_side
                                     else min(picks, key=lambda z: z[0]))
            if stop_hit and rev_hit:
                rev_first = ((line > pos.stop) if pos.long_side
                             else (line < pos.stop))
                stop_hit = not rev_first
                rev_hit = rev_first

            if stop_hit:
                close_position(pos, i, pos.stop, "stop", slippage)
                positions.remove(pos)
                continue

            if rev_hit:
                at, px = i, line
                if entry_fill == "next_open" and i + 1 < len(candles):
                    # **水準では約定させない。**その足は丸ごと約定より後に
                    # なるので、足の中の順序を問う余地が消える。価格は悪くなる。
                    at, px = i + 1, candles[i + 1].open
                close_position(pos, at, px, "reversal", 0.0)
                positions.remove(pos)
                # 執行の足が折り返さないまま構造の水準まで走った場合、
                # そこで乗るか見送るか。どちらが良いかは測って決める。
                if source == "structure" and entry_fallback == "skip":
                    continue
                if pos.flips < max_flips:
                    # ドテン。損切りは **1 つ前** の折り返しの外側。
                    #
                    # **ここを「最新」に置いていた。**乗り換えた側にとっての
                    # ダウ転換も同じ水準を割ることなので、損切りと利確が
                    # 同じ値段になり、**利確側の出口が消えていた。**
                    nl = not pos.long_side
                    latest = st["last_low"] if nl else st["last_high"]
                    prot = st["prev_low"] if nl else st["prev_high"]
                    if prot is None or latest is None:
                        continue
                    stop = (prot.price - swing_stop_buffer_atr * a if nl
                            else prot.price + swing_stop_buffer_atr * a)
                    risk = abs(px - stop)
                    if risk <= 0 or ((px <= stop) if nl else (px >= stop)):
                        continue
                    pid_seq += 1
                    positions.append(_Position(
                        pid=pid_seq, long_side=nl, stop=stop, atr=a,
                        zone_price=pos.zone_price, zone_key=pos.zone_key,
                        flips=pos.flips + 1, anchor=latest.index,
                        # **乗り換えに使った折り返しでは買い増ししない。**
                        add_swing=(source, rev.index),
                        legs=[_Leg("flip", at, px, stop, risk, 1.0,
                                   latest.price)]))
                    if stopped_on_fill_bar(positions[-1], at, px, nl):
                        positions.pop()
                continue

            # --- 買い増し。**流れが続く側の折り返しを抜けたら足す。** ------
            if pos.adds < max_adds:
                picks = []
                if entry_signal == "exec":
                    xg = ex["last_high"] if pos.long_side else ex["last_low"]
                    if xg is not None and crossed_now(xg.price, False):
                        picks.append((xg.price, "exec", xg))
                sg = st["last_high"] if pos.long_side else st["last_low"]
                if sg is not None and crossed_now(sg.price, False):
                    picks.append((sg.price, "structure", sg))
                picks = [z for z in picks if (z[1], z[2].index) != pos.add_swing]
                if picks:
                    line, source, go = (min(picks, key=lambda z: z[0])
                                        if pos.long_side
                                        else max(picks, key=lambda z: z[0]))
                    if not (source == "structure" and entry_fallback == "skip"):
                        at, px = i, line
                        if entry_fill == "next_open" and i + 1 < len(candles):
                            at, px = i + 1, candles[i + 1].open
                        risk = abs(px - pos.stop)
                        if risk > 0 and ((px > pos.stop) if pos.long_side
                                         else (px < pos.stop)):
                            rev0 = (st["last_low"] if pos.long_side
                                    else st["last_high"])
                            pos.legs.append(_Leg("add", at, px, pos.stop, risk,
                                                 add_size,
                                                 rev0.price if rev0 else 0.0))
                            pos.adds += 1
                            pos.add_swing = (source, go.index)
                            if stopped_on_fill_bar(pos, at, px, pos.long_side):
                                positions.remove(pos)

        # --- 新しく帯へ置いた指値 ---------------------------------------
        if blocked_hours_utc and candle.time.hour in blocked_hours_utc:
            continue
        for key, edge in (("top", zone_hi), ("bottom", zone_lo)):
            if edge is None:
                continue
            level = edge[1]
            # `reverse_entry` は **同じ合図で逆に張る** ための対照。
            # 約定する足も値段も変えず、向きだけを裏返す。素の期待値が
            # 両向きともマイナスなら、方向ではなく **仕掛けそのものが
            # 値を削っている**(= どこかにバグがある)ことになる。
            long_side = (key == "bottom") != reverse_entry
            # 「少し奥」は **帯の外側**。向きを裏返しても外側は外側。
            limit = (level + entry_beyond_atr * a if key == "top"
                     else level - entry_beyond_atr * a)
            # 離れたら次の待ち伏せを許す(同じ水準で連射しない)。
            # **基準は帯ではなく、注文を置いてある値段。**帯を基準にすると、
            # 奥へ置くほど「大きくヒゲを出して帯の近くへ戻った足」だけが
            # 選別される。狙って作った選別ではないので、そこで成績が
            # 上がっても機構の手柄ではない。
            if abs(candle.close - limit) > rearm_atr * a:
                armed[key] = True
            # **離れているあいだも注文は置いてある。**ここで「近いときだけ
            # 見る」にすると、大きな足で届いたのに見送る挙動が混ざり、
            # 再武装の距離を広げるほど選別が効いたように見える(実測で
            # 0.5/1.0/2.0 ATR が +0.036/+0.132/+0.400 になった)。
            # 待ち伏せの解除は **連射を止めるためだけ** に使う。
            if not armed[key] or len(positions) >= max_open:
                continue

            if zone_entry == "exec_turn":
                # **帯の最値に置いた指値は、抜けたときだけ約定する。**
                # 跳ね返りを取りたいのに、届かずに見送るか、抜けてから
                # 掴まされるかのどちらかになる(利用者の指摘)。
                #
                # 帯へ触れたら、そこから **執行の足が折り返すのを待って**
                # 入る。ドテンや買い増しと同じ基準になり、一貫する。
                if ((candle.high >= level) if key == "top"
                        else (candle.low <= level)):
                    touched[key] = i
                if touched[key] is None or i - touched[key] > zone_wait_bars:
                    continue
                xg = ex["last_high"] if long_side else ex["last_low"]
                if xg is None:
                    continue
                # **帯から離れすぎたら、もう帯で入る話ではない。**
                # 執行の足の折り返しは帯と無関係に出るので、放っておくと
                # 15 ATR 先の折り返しで約定して、損切りだけ帯に残る。
                if (zone_entry_max_atr
                        and abs(xg.price - level) > zone_entry_max_atr * a):
                    continue
                limit = xg.price
                stop = (level - stop_buffer_atr * a if long_side
                        else level + stop_buffer_atr * a)
                probe = limit - (spread if long_side else 0.0)
                prior = candles[i - 1].close if i else candle.open
                reached = ((prior < probe and candle.high >= probe) if long_side
                           else (prior > probe and candle.low <= probe))
                if not reached:
                    continue
                at, px = i, limit
                if entry_fill == "next_open" and i + 1 < len(candles):
                    at, px = i + 1, candles[i + 1].open
                risk = abs(stop - px)
                if risk <= 0 or ((px <= stop) if long_side else (px >= stop)):
                    continue
                armed[key] = False
                touched[key] = None
                pid_seq += 1
                base = st["last_low"] if long_side else st["last_high"]
                positions.append(_Position(
                    pid=pid_seq, long_side=long_side, stop=stop, atr=a,
                    zone_price=level, zone_key=key,
                    anchor=base.index if base is not None else -1,
                    legs=[_Leg("zone", at, px, stop, risk, 1.0,
                               base.price if base is not None else 0.0)]))
                if stopped_on_fill_bar(positions[-1], at, px, long_side):
                    positions.pop()
                continue

            probe = limit - (spread if long_side else 0.0)
            # **触れ方は帯の側で決まる。**上端は高値で、下端は安値で触れる。
            # 向きを裏返しても、約定する足と値段は変えない。
            #
            # **そこへ来たときだけ約定する。**指値は市場のこちら側に置く
            # ものなので、前の足の終値が既に向こう側にあるなら、その注文は
            # とっくに約定しているか、そもそも置けない。ここを見ないと
            # **その足が一度も付けていない値段で約定する**(実測で
            # 123.4 の足が 131.2 で約定し、-16 R を計上していた)。
            prior = candles[i - 1].close if i else candle.open
            if key == "bottom":
                reached = prior > probe and candle.low <= probe
            else:
                reached = prior < probe and candle.high >= probe
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
                legs=[_Leg("zone", i, limit, stop, risk, 1.0,
                       base.price if base is not None else 0.0)]))
            if stopped_on_fill_bar(positions[-1], i, limit, key == "top"):
                positions.pop()

    # --- 終端。**残った建玉は成行で閉じ、印を付けて分けて数える。** ------
    last = len(candles) - 1
    for pos in positions:
        shift = 0.0 if pos.long_side else spread
        close_position(pos, last, candles[last].close - (shift + slippage)
                       * (1.0 if pos.long_side else -1.0), "end", 0.0)
    return out
