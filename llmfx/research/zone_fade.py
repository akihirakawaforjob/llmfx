"""抵抗帯へ指値を置いて待つ取引を、逆選択まで含めて測る.

利用者が実際にやっていた形:

    抵抗帯の少し奥(スプレッド対策)に予め指値を置いておく
    価格がそこへ届けば約定、届かなければ何も起きない
    損切りは帯の外側
    利確は明示しない(相場に従う)

**指値は「届いたときだけ約定する」ため、都合の良い場面を取りこぼす。**
価格が帯へ少し触れて反転する場面(いちばん美味しい形)では約定せず、
勢いよく突き抜ける場面(いちばん不利な形)では必ず約定する。この偏りを
逆選択という。成行を前提にした集計はこれを見落とし、コストの節約分が
そのまま残ると誤解させる。

ここでは指値の約定を素直に模擬する。届いた足でだけ建玉を持ち、
届かなければ事象そのものを作らない。よって残った標本には
最初から逆選択が織り込まれている。

利用者は約定を待つ間に指値を動かしていたが、それは再現しない。
「どう動かしたか」を後から決めると、値動きを見てから決めたことになる。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.swings import SwingDetector
from ..domain.types import Candle, SwingType
from ..domain.zones import ZoneTracker


def defenders_weakening(
    swings: list, from_below: bool, bar_index: int
) -> bool:
    """帯の守り手が押し負け始めているか(利用者の言う「ブレイクリスク」)。

        安値切り上げや高値切り下げが起き始めた抵抗帯はブレイクされやすい。
        = その帯の守り手が諦め始めている。

    抵抗帯(上から売る側)なら、直近の確定安値が切り上がっていれば
    買い手が押し上げてきているということ。支持帯ならその鏡像。

    確定済みのスイングだけを見る。未確定を混ぜると先読みになる。
    判定できるだけの本数が無ければ False(見送らない)。
    """
    kind = SwingType.LOW if from_below else SwingType.HIGH
    pivots = [s for s in swings if s.type is kind and s.confirmed_index <= bar_index]
    if len(pivots) < 2:
        return False
    rising = pivots[-1].price > pivots[-2].price
    return rising if from_below else not rising


def _near_nfp(moment, minutes: int) -> bool:
    """米雇用統計の前後 `minutes` 分か。毎月第 1 金曜と決まっている。"""
    from datetime import timedelta

    from ..domain.sessions import _neighbour_months, nfp_time

    window = timedelta(minutes=minutes)
    for year, month in _neighbour_months(moment.year, moment.month):
        if abs(moment - nfp_time(year, month)) <= window:
            return True
    return False


def _atr_series(candles: list[Candle], period: int) -> list[float]:
    """各足までの ATR(Wilder)。上位足で帯を引くとき、下位足の値幅を
    測るのに要る。"""
    out: list[float] = []
    run = 0.0
    prev = candles[0].close if candles else 0.0
    for i, c in enumerate(candles):
        tr = c.high - c.low if i == 0 else max(
            c.high - c.low, abs(c.high - prev), abs(c.low - prev)
        )
        run = tr if i == 0 else (run * (period - 1) + tr) / period
        prev = c.close
        out.append(run)
    return out


def _rolling_extremes(
    candles: list[Candle], window: int | None
) -> tuple[list[float] | None, list[float] | None]:
    """直近 `window` 本の高値の最大・安値の最小を、各足について先に作る。

    単調デックで O(n)。窓を毎回舐めると O(n x window) になり、
    掃引が実用にならない(48 通りで 1 銘柄 1 時間を超えた)。
    """
    if not window:
        return None, None
    from collections import deque

    highs: list[float] = []
    lows: list[float] = []
    hi_q: deque[int] = deque()
    lo_q: deque[int] = deque()
    for i, c in enumerate(candles):
        while hi_q and candles[hi_q[-1]].high <= c.high:
            hi_q.pop()
        hi_q.append(i)
        while lo_q and candles[lo_q[-1]].low >= c.low:
            lo_q.pop()
        lo_q.append(i)
        if hi_q[0] <= i - window:
            hi_q.popleft()
        if lo_q[0] <= i - window:
            lo_q.popleft()
        highs.append(candles[hi_q[0]].high)
        lows.append(candles[lo_q[0]].low)
    return highs, lows


def _bar_path(c: Candle) -> tuple[float, float, float, float]:
    """1 本の足の中を、四本値から推し量った順に並べる。

    陽線は 始値 → 安値 → 高値 → 終値、陰線は 始値 → 高値 → 安値 → 終値。
    値動きは一度どちらかへ振ってから逆へ抜ける、という最も普通の読み。
    **推測であって事実ではない。**細かい足があるならそちらを使う。
    """
    if c.close >= c.open:
        return (c.open, c.low, c.high, c.close)
    return (c.open, c.high, c.low, c.close)


def _reach(points: list[float], level_up: float | None,
           level_down: float | None) -> str | None:
    """点列を順に辿り、上下どちらの水準へ先に届くかを返す。

    隣り合う 2 点の間は一方向にしか動かないので、片方の区間で上下
    両方に届くことはない。**順序が分かるのはここだけで、区間の中は
    やはり分からない。**細かくするほど残る曖昧さが減る。
    """
    if level_up is not None and points[0] >= level_up:
        return "up"
    if level_down is not None and points[0] <= level_down:
        return "down"
    for b in points[1:]:
        if level_up is not None and b >= level_up:
            return "up"
        if level_down is not None and b <= level_down:
            return "down"
    return None


def _after_fill(points: tuple[float, ...] | list[float], limit: float,
                from_below: bool) -> list[float] | None:
    """指値へ届いた瞬間から先の点列。届かなければ None。

    **届く前に通り過ぎた分は数えない。**ここを飛ばすと、約定して
    いない時間帯の値動きを取り分に数えてしまう。
    """
    if (points[0] >= limit) if from_below else (points[0] <= limit):
        return list(points)          # 始値がすでに越えている(窓開け)
    for k, b in enumerate(points[1:], start=1):
        if (b >= limit) if from_below else (b <= limit):
            return [limit, *points[k:]]
    return None


@dataclass
class RangeEdge:
    """**チャートに映っている範囲の端**。利用者が赤い線を引く場所。

    スイングの塊(`Zone`)ではなく、直近 N 本の最高値・最安値そのもの。
    利用者が見せてくれた 5 分足・15 分足・1 時間足の 3 枚とも、線は
    その足で表示されている窓の上端と下端に引かれていた。**足を変えれば
    線も変わる**のはそのため。二つ上位の足で 2 本探すという説明とも合う
    (上端と下端で 1 組のレンジになる)。

    `Zone` と同じ面を持たせて、後段の処理を共通にする。
    """

    low: float
    high: float
    edge_key: str
    count: int = 0
    touches: tuple = ()
    arena: float = 0.0
    """大枠(1 週間の最値)。指値は直近の折り目へ寄せても、往復を刈る
    ときの相手はこちらを見る。利用者の指定:

        大枠は 1 週間で良いが、その後のエントリーは必ず直近の折り目で
        微調整していかないとエントリー数も稼げないし勿体ない。
    """

    @property
    def price(self) -> float:
        return (self.low + self.high) / 2

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass
class FadeTrade:
    """帯へ置いた指値が約定してからの成果."""

    bar_index: int
    zone_price: float
    zone_width_atr: float
    touches: int
    atr: float
    from_below: bool
    """True なら価格は **下から** 帯へ来た(上の端)。売買の向きではない。"""
    entry: float
    stop: float
    risk_atr: float
    """損切りまでの幅(ATR 倍)。R の分母。"""
    r_multiple: float
    """観測期間の終わりまで持った場合の成果。"""
    max_favourable_r: float
    max_adverse_r: float
    hit_stop: bool
    bars_held: int
    long_side: bool = False
    """True なら買い。跳ね返りに乗るか抜けた側に乗るかで逆になるので、
    `from_below` だけでは向きが決まらない。"""
    defenders_weak: bool = False
    """建玉を持った時点で、帯の守り手が押し負けていたか。

    抵抗帯なら「安値が切り上がっている」。利用者はこれを
    **エントリーすべきサイン**だと言っている。選別に効いているかを
    後から分けて数えるために残す。"""
    entry_hour: int = 0
    """約定した足の UTC 時。スプレッドが開く時間帯を後から評価するのに使う。"""
    why: str = ""
    """どこで決済したか。"stop" / "opp"(反対側の帯) / "time"(時間切れ)。

    負けの内訳を出口ごとに割るために要る。**反対側の帯での決済が
    損になっていたら、それは選び方のバグ。**実際に一度そうなっていて、
    平均 -3.78 R・最悪 -35 R の「利確」が負けの半分を占めていた。
    """
    exit_price: float = 0.0
    """決済した価格。損切りが効いているかを外から確かめられるようにする。"""
    fill_index: int = -1
    """指値が約定した足。`bar_index`(帯に触れて指値を決めた足)とは別。"""
    opposite_price: float = 0.0
    """建玉を持った時点で見ていた反対側の帯の縁。0 なら反対側が無かった。"""
    zone_low: float = 0.0
    zone_high: float = 0.0
    zone_touch_bars: tuple[int, ...] = ()
    """帯を作ったスイングの足番号。**その帯が本当に何度も試された水準か**
    を図で確かめるために要る。帯を上位足で引いた場合は上位足の番号。"""


def collect_fade_trades(
    candles: list[Candle],
    *,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.6,
    tolerance_atr: float = 0.5,
    max_age_bars: int | None = 2000,
    min_touches: int = 2,
    entry_offset_atr: float = 0.1,
    stop_buffer_atr: float = 0.2,
    max_wait_bars: int = 12,
    horizon: int = 24,
    max_zone_width_atr: float | None = None,
    higher_minutes: int | None = None,
    require_range: bool = False,
    max_range_atr: float | None = None,
    exit_at_opposite_zone: bool = False,
    blocked_hours_utc: frozenset[int] | None = None,
    nfp_blackout_minutes: int = 0,
    skip_break_risk: bool = False,
    entry_from_range_bars: int | None = None,
    stop_from_range_bars: int | None = None,
    intrabar: str = "stop_first",
    path_candles: list[Candle] | None = None,
    scale_to_zone_timeframe: bool = False,
    entry_at_zone_extreme: bool = False,
    zone_source: str = "pivots",
    range_bars: int = 120,
    range_needs_turn: bool = True,
    entry_from_recent_turn: bool = False,
    edge_mode: str = "fade",
    break_confirm: str = "touch",
    break_confirm_atr: float = 0.0,
    max_open: int = 1,
    arm_within_atr: float = 0.0,
    drop_broken_edges: bool = False,
    min_rejection_atr: float = 0.0,
    skip_against_trend: bool = False,
    max_tries_per_zone: int = 0,
    breakeven_at_r: float = 0.0,
    entry_beyond_atr: float = 0.0,
    warmup: int = 200,
    refresh_every: int = 50,
) -> list[FadeTrade]:
    """帯へ指値を置き、約定した取引だけを集める。

    `entry_offset_atr` は帯の手前側の縁から **帯の内側へ** どれだけ
    入れて指値を置くか。利用者の言う「少し奥」。0 なら縁ちょうど。
    `stop_buffer_atr` は帯の向こう側の縁から外側へどれだけ離すか。

    `skip_break_risk` は利用者の言う「ブレイクリスク」で見送る。

        安値切り上げや高値切り下げが起き始めた抵抗帯はブレイクされやすい。
        = その帯の守り手が諦め始めている。

    抵抗帯(上から売る)なら、直近の確定安値が切り上がっていたら見送る。
    買い手が押し上げてきている = 守り手が押し負けつつある。
    支持帯なら鏡像で、直近の確定高値が切り下がっていたら見送る。

    利用者は「そもそもエントリーしない」と言っていた。指値を動かして
    避けるのではなく、**その帯を丸ごと touch しない** のが本来の形。

    `entry_from_range_bars` は **指値そのもの** を直近 N 本の最値に置く。
    利用者の本来の指摘はこちら:

        1 つ前の帯の最値ではなく、より広い範囲での最値にすることで、
        よりはみ出しに刈られにくくなる。

    帯の縁の少し内側で待つと、上へ突き抜ける動きの **途中** で約定して
    しまい、そのまま損切りまで持っていかれる。直近 N 本の最値まで
    引き上げると、**そこまで実際に届いたときにしか約定しない**。
    売るなら天井で売る、という形になる。

    最値は **1 本前まで** で作る。その足自身の高値を使うと、
    「その足で最値を更新したから、その値で約定した」という循環になる。

    `stop_from_range_bars` は損切りの基準を帯の縁ではなく直近 N 本の
    最値にする。指値を最値へ置く場合は、損切りはそこから
    `stop_buffer_atr` だけ外側になるので、リスク幅を直接決められる。

    `higher_minutes` を渡すと **帯を上位足で引く**。利用者の指定は
    「エントリーに使う時間軸の 2 つ上位」(M15 なら H1)。閉じた上位足
    しか使わないので先読みにならない。

    `zone_source` は帯をどう作るか。

    | 値 | 中身 |
    | --- | --- |
    | `pivots` | 確定スイングを ATR 0.5 でまとめた塊。2 回以上試された水準 |
    | `range` | **直近 `range_bars` 本の最高値と最安値**。上端と下端の 2 本 |

    `range_needs_turn`(既定 True)は、その最値が **折り返した場所である
    こと**を要求する。利用者の説明:

        5 分足の下を書かなかったのは、左側があまり良く見えず、はっきりと
        直近底値の折り返しかわからなかったからだ。

    上昇の途中の起点(まだ折り返していない、いまの動きの端)には線を
    引かない。実装は「窓の中で **確定したスイング** のうち最も高いもの /
    最も安いもの」。確定には左右 N 本が要るので、いまの動きの端は
    自動的に外れる。False なら折り返しを問わず窓の最値そのもの。

    `range` は利用者が実際に線を引く場所。5 分足・15 分足・1 時間足の
    3 枚を見せてもらったところ、線はどれも **その足で表示されている窓の
    上端と下端** にあった。足を変えれば線も変わる。「二つ上位の足で
    抵抗帯を 2 つ探し、その間の往復を刈る」という説明とも合う。

    `pivots` は「何度も試された水準」を探しに行くが、実データでは
    接触 2 回だけの薄い帯が大量に出て、利用者が見て帯と認めない場所で
    建玉を持っていた(`docs/zone-trades-h1.html`)。

    `edge_mode` は **帯に来たときにどちらへ乗るか**。利用者の説明:

        帯が何を示したかに従う。弾かれたら跳ね返りに乗り、抜けたら
        抜けた側に乗る。**帯に来た時点では方向を決めない。**

    | 値 | 中身 |
    | --- | --- |
    | `fade` | いつも跳ね返り側(帯で逆張り) |
    | `break` | いつも抜けた側(帯の外へ乗る) |
    | `auto` | **守り手が押し負けていれば抜けた側、そうでなければ跳ね返り側** |
    | `weakening` | **押し負けが起きた時点で乗る。**帯に届くのを待たない |

    `weakening` は利用者の言う「抵抗帯の押し負けが発生した時点で乗る」。
    帯へ向かう安値が切り上がった瞬間(確定スイングで判定)に、次の足の
    始値で建玉を持つ。**帯まで届く前に入るので、端で反転する形を待たない。**
    損切りはその **切り上がった安値の下**(構造)に置く。利確は置かず、
    時間切れまで持つ。

    `max_open` は同時に持てる建玉の数。利用者の指摘:

        レンジの往復を取る時に建玉があると指値を入れられない
        = 機会損失となりうる。

    1 建玉に縛ると、ある機会を取ったせいで別の機会を丸ごと落とす。
    **ただし R は 1 建玉あたりの値なので、同時に持つほど口座の変動は
    大きくなる。**資金管理は別に決める必要がある。

    `auto` の判定は `defenders_weakening`(抵抗帯へ向かう安値が切り上がって
    いるか)。利用者の指摘:

        抵抗帯への高値切り下げや安値切り上げが起きたらエントリーを
        見送ったと思うが、**あれはむしろエントリーすべきサイン**だと僕は思う。

    これは `skip_break_risk`(見送る)と同じ判定を、逆の使い方にしたもの。

    抜けた側に乗る場合は、
    **入りが逆指値になるのでスプレッドを往復ぶん払う**(跳ね返り側は
    指値なので入りは無料)。反対側の帯は損失方向になるので決済先にしない。

    `break_confirm` は **抜けた側に乗るときの入り方**。利用者の指摘:

        指値の位置がブレイクを見てからではなく、折り返し刈り取り用と
        同じ場所をエントリーポイントに選んでしまっている。これに乗るなら、
        抵抗帯の押し負けが発生した時点、もしくは **ブレイクを見てから
        乗る** が正しい。

    | 値 | 中身 |
    | --- | --- |
    | `touch` | 端へ届いた瞬間に乗る。**端に触っただけで抜けていない** |
    | `close` | **終値が端の外に出てから**、次の足の始値で乗る |

    `break_confirm_atr` は「外に出た」と認める幅(端から ATR 何倍か)。
    0 なら端を 1 ティックでも超えた終値で認める。

    `close` は確認を待つぶん値段が悪くなるが、**触っただけで反転する形
    (跳ね返り側が狙っているまさにその形)を拾わずに済む**。

    `entry_from_recent_turn` は **指値を直近の折り目へ寄せる**。大枠(1 週間の
    最値)はそのまま残り、往復を刈るときの相手として使う。利用者の指定:

        大枠は 1 週間で良いが、その後のエントリーは必ず直近の折り目で
        微調整していかないとエントリー数も稼げないし勿体ない。

    1 週間の最値は遠いことが多く、そこまで戻ってこないと約定しない。
    直近の折り目まで寄せると、届く回数が増える。

    `breakeven_at_r` は **そこまで伸びたら損切りを建値へ動かす**(0 で無効)。

    解剖したところ、**46% の建玉が 1.0 R 以上伸びてから -1 R で切られていた**
    (USD/JPY・2,704 件)。入り口ではなく持ち方で失っている。

    ただし過去にダウ理論で建値移動を入れたときは、勝率が 17.9% -> 65.4% に
    上がる代わりに平均勝ちが 4.79 R -> 0.47 R まで潰れて **中身が空**に
    なった。利益が右の裾からしか来ていない構成では、裾に触る操作はすべて
    損になる。**必ず約定ロジックを通して測ること。**概算は強く楽観に出る。

    `skip_against_trend` は **流れに逆らう側では張らない**。利用者が
    リプレイ画面で見つけた形:

        高値切り上げ、安値切り下げが始まっているのに、必死にエントリーして
        往復ビンタを食らっている。

    高値も安値も切り上がっているとき(上げの流れ)は上の帯で売らない。
    どちらも切り下がっているときは下の帯で買わない。判定は帯を引いた足の
    確定スイング 2 本ずつ。`skip_break_risk`(安値だけを見る)より厳しい。

    `max_tries_per_zone` は **同じ帯で何回まで試すか**(0 で無制限)。
    同じ水準で負け続ける形を止める。数え直すのは、その帯が一度死んで
    新しく折り返し直したとき。

    `min_rejection_atr` は **弾きの強さ** で帯を絞る。利用者の指摘:

        大口投資家はそんなに何回もポジションを変えないし、大枠で見て
        弾きの強かった線の方がデータとしてももう少し強かった。

    「何回試されたか」は 3 回測って 3 回とも否定された(要求する回数を
    増やすほど単調に悪化)。**回数ではなく 1 回あたりの弾きの大きさ**で
    見る。各接触のあと価格がどれだけ離れたか(次のスイングまでの値幅)を
    ATR 倍で測り、その中央値がこの値に満たない帯は触らない。

    最後の接触には「次」がまだ無いので数に入れない(先読みを避ける)。

    `drop_broken_edges` は **抜けられた端を捨てる**。利用者の説明:

        線が途中で消えているものは、抵抗帯を抜けてしまっているので、
        流石に戻ってこなそうだなと感じたもの → エントリー中止。

    終値がその端を越えたら、**その端も、それ以前に付けた折り返しも捨てる**。
    以後は破られた後に新しく付いた折り返しだけを使う。切らないと、
    一度抜けた水準へ価格が戻ってきたときに古い指値が生き残る。利用者の懸念:

        約定しなかったお残りがそのまま、勢いよく戻って来てブレイクした際に
        約定して壊されている可能性がある。

    `arm_within_atr` は **帯へどこまで近づいたら場面として見るか**。
    0 なら「帯に触れた足」だけ。利用者の指摘:

        もし本当に強固な抵抗帯に乗れているとしたら、たまたま跳ねた部分に
        乗っているだけで **約定しなかった** 可能性がある。
        少し手前にズラすのもあり。

    **触れた足しか見ないと、手前に指値を置いても意味が無い。**帯の直前で
    折り返した場面はそもそも候補に入らないので、指値の位置を変えても
    件数が動かない(実測でも 1,531 → 1,526 とほぼ同じだった)。
    `entry_beyond_atr` を負にして手前へ寄せるなら、ここも同じだけ広げる。

    `entry_beyond_atr` は指値を極値の **さらに外側** へ置く。負にすると
    **手前(レンジの内側)**。利用者の言う
    「抵抗帯の少し奥(スプレッド対策)に予め指値を設定しておく」。
    外へ置くほど約定しなくなるが、**約定しなければそもそも負けない**。

    `entry_at_zone_extreme` は **帯そのものの極値に指値を置く**。
    利用者の手法はこれ:

        15 分足でエントリーするなら、1 時間足を見て抵抗帯を見つけ、
        その最値で 15 分足の基準でエントリーする。
        ここで下位足の基準でエントリーするのは、損切ラインが遠くならない為と、
        利確ラインへの伸びが大きく期待出来るからだ。

    つまり **帯は上位足・損切りの物差しは下位足** という組み合わせ。
    `higher_minutes=60` と併せ、`scale_to_zone_timeframe` は掛けない。

    `entry_from_range_bars`(直近 N 本の最値)とは併用しない。最値の窓を
    使うと指値が帯から離れる。実測では **指値が帯から 2.5 ATR も外へ出て、
    守るべき帯とは無関係な場所で建玉を持っていた**(`docs/zone-trades.html`
    の 3 例目)。

    `scale_to_zone_timeframe` は、帯を上位足で引いたときに **物差しも
    その足に合わせる**。利用者の指摘:

        抵抗帯の最値は、参照している時間軸の抵抗帯と一緒であるべき。

    切り替わるのは 3 つ。どれも「どの足で見た値か」という同じ話なので
    ひとまとめにしてある:

    - ATR(帯の幅の上限、指値の差し込み、損切りの余裕、再武装の距離)
    - 指値を置く直近 N 本の最値
    - 損切りの基準にする直近 N 本の最値

    **これを外すと、上位足の帯を下位足の物差しで測ることになる。**
    帯の幅の上限も損切りの余裕も小さすぎて、別物の取引になる。
    過去に「上位足の距離を下位足の ATR で測る」で同じ罠を踏んでいる。

    `exit_at_opposite_zone` は、**反対側の帯へ届いたらそこで手仕舞う**。
    利用者の説明:

        もしその区画に抵抗帯が二つあれば、自ずとそれらはレンジになる。
        その為、両端からエントリーする必要がある。

    帯 1 本での逆張りは、反対側まで走っても時間切れまで持ち続ける。
    反対側で切れば、そこは同時に **反対向きのエントリー地点** でもあるので、
    往復を刈れるようになる(手仕舞い後は待機が解けるため、同じ足で
    反対側の帯に指値を置ける)。

    注意: 反対側で切ると、**そのまま抜けて走る場合の裾も切る**。
    利用者は「反対側を抜けてそのまま走ったら全部が取り分」と言っている
    ので、ここは掃引して確かめる軸であって、既定では入れない。

    `intrabar` は **1 本の足の中の道順** をどう扱うか。四本値からは
    順序が分からないのに、結論はここで決まる。実測(開発用 4 銘柄)で
    符号が変わった:

        stop_first          +0.096 〜 +0.159 R
        no_same_bar_profit  -0.145 〜 -0.201 R

    利確の 87.6% が約定した足そのもので起きているため。

    | 値 | 扱い |
    | --- | --- |
    | `stop_first` | 高安だけで見る。損切りを先に見て、次に反対側の帯。約定足での利確も認める。**損切りだけ不利側、利確は有利側**という食い違いがある |
    | `no_same_bar_profit` | 上から、約定足での利確だけを外す。不利側の端 |
    | `ohlc` | 陽線なら 始値→安値→高値→終値、陰線ならその逆、と推し量って順序を解く。推測 |
    | `path` | `path_candles`(M1 など細かい足)で順序を解く。**事実に最も近い** |

    `path_candles` は同じ期間を覆う細かい足。`intrabar="path"` のときだけ
    使う。足りない区間は `ohlc` に落ちる。

    `blocked_hours_utc` はその UTC 時に **建玉を持たない**。

    実勢のスプレッドは時間帯で数倍に開く。とくに NY 17 時のロールオーバー
    (UTC 21-22 時、日本の早朝)は薄く、平常時の数倍になる。固定 pips で
    測ると、この時間帯の取引だけコストを大幅に過小評価する。

    **どの時間が薄いかは板の仕組みから事前に分かる。**成績を見てから
    悪い時間を外すのは選択バイアスだが、ロールオーバーを外すのは
    先読みにならない。

    `nfp_blackout_minutes` は米雇用統計の前後この分数を避ける。毎月第 1
    金曜と決まっているので事前に分かる。**指標は年 100 回程度で、
    毎日あるロールオーバーとは頻度が桁で違う**ため、効きは小さいはず。

    `require_range` は **上下 2 本の帯が揃っているときだけ**建玉を持つ。
    こちらは能力ではなく絞り込み。既定では掛けない。
    利用者の説明:

        抵抗帯を 2 つ探し、そこをレンジとしてその間の往復を刈り取る。

    片側だけで張ると、そこを抜けられたときに一方的に負ける。両側を
    押さえていれば、抜けた側が次のエントリーになる。
    `max_range_atr` は上下の間隔の上限(離れすぎた 2 本を組にしない)。
    """
    from datetime import timedelta

    from ..data.resample import resample_candles

    detector = SwingDetector(
        left=left, right=right, atr_period=atr_period, min_swing_atr=min_swing_atr
    )
    tracker = ZoneTracker(tolerance_atr=tolerance_atr, max_age_bars=max_age_bars)

    # 帯を上位足で引く場合は、閉じた上位足だけを取り込む。
    higher = resample_candles(candles, higher_minutes) if higher_minutes else None
    span = timedelta(minutes=higher_minutes) if higher_minutes else None
    atr_high = _atr_series(higher, atr_period) if higher else None
    atr_low = _atr_series(candles, atr_period) if higher else None
    hi_i = 0
    hi_bar = -1

    # 直近 N 本の最値は、帯に触れるたびに窓を舐め直すと重い。
    # 実測では 48 通りの掃引が 1 銘柄 1 時間を超えた。O(n) で先に作る。
    roll_high, roll_low = _rolling_extremes(candles, stop_from_range_bars)
    entry_high, entry_low = _rolling_extremes(candles, entry_from_range_bars)
    if zone_source not in ("pivots", "range"):
        raise ValueError(f"zone_source が不正: {zone_source!r}")
    if edge_mode not in ("fade", "break", "auto", "weakening"):
        raise ValueError(f"edge_mode が不正: {edge_mode!r}")
    if max_open < 1:
        raise ValueError("max_open は 1 以上")
    if break_confirm not in ("touch", "close"):
        raise ValueError(f"break_confirm が不正: {break_confirm!r}")
    if zone_source == "range" and not range_needs_turn:
        edge_high, edge_low = _rolling_extremes(
            higher if higher is not None else candles, range_bars)
    else:
        edge_high = edge_low = None
    # 折り返し済みの最値。確定スイングを単調デックで持つ(窓を毎回
    # 舐めると足数 x スイング数になり、22 万足では実用にならない)。
    from collections import deque as _deque

    turn_hi: _deque = _deque()
    turn_lo: _deque = _deque()

    # 直近 2 本の確定スイング。押し負け(安値切り上げ / 高値切り下げ)の
    # 判定と、そこに置く損切りに要る。**列を毎回舐めると重い。**
    swing_state = {"prev_low": None, "last_low": None,
                   "prev_high": None, "last_high": None, "seen": 0}

    def note_swings() -> None:
        sw = detector.swings
        st = swing_state
        for x in sw[st["seen"]:]:
            if x.type is SwingType.LOW:
                st["prev_low"], st["last_low"] = st["last_low"], x.price
            else:
                st["prev_high"], st["last_high"] = st["last_high"], x.price
        st["seen"] = len(sw)
        if sw:
            # 末尾は「より極端な方」に置き換わることがある。値だけ更新する。
            t = sw[-1]
            if t.type is SwingType.LOW:
                st["last_low"] = t.price
            else:
                st["last_high"] = t.price

    def absorb_turn(sw, series=None, upto: int = -1) -> None:
        # **確定するまでの間に抜けられていたら、そもそも線を引かない。**
        # スイングは左右 N 本で確定するので、折り返した足と気づく足の間に
        # 数本ある。そこで終値が越えていたら、その水準はもう生きていない。
        if drop_broken_edges and series is not None and upto > sw.index:
            for c in series[sw.index + 1 : upto + 1]:
                if (c.close > sw.price if sw.type is SwingType.HIGH
                        else c.close < sw.price):
                    return
        if sw.type is SwingType.HIGH:
            while turn_hi and turn_hi[-1][1] <= sw.price:
                turn_hi.pop()
            turn_hi.append((sw.index, sw.price))
        else:
            while turn_lo and turn_lo[-1][1] >= sw.price:
                turn_lo.pop()
            turn_lo.append((sw.index, sw.price))
    if higher is not None and scale_to_zone_timeframe:
        roll_high_h, roll_low_h = _rolling_extremes(higher, stop_from_range_bars)
        entry_high_h, entry_low_h = _rolling_extremes(higher, entry_from_range_bars)
    else:
        roll_high_h = roll_low_h = entry_high_h = entry_low_h = None

    if intrabar not in ("stop_first", "no_same_bar_profit", "ohlc", "path"):
        raise ValueError(f"intrabar が不正: {intrabar!r}")
    # 細かい足を M15 の各足へ割り当てる。両方とも時刻順なので 1 度なめれば済む。
    fine_at: list[tuple[int, int]] | None = None
    if intrabar == "path":
        if not path_candles:
            raise ValueError("intrabar='path' には path_candles が要る")
        fine_at = []
        k = 0
        step = candles[1].time - candles[0].time if len(candles) > 1 else None
        for c in candles:
            end = c.time + step if step else c.time
            while k < len(path_candles) and path_candles[k].time < c.time:
                k += 1
            j = k
            while j < len(path_candles) and path_candles[j].time < end:
                j += 1
            fine_at.append((k, j))
            k = j

    def bar_points(idx: int) -> list[float]:
        """その足の中を辿る点列。細かい足があればそれを繋ぐ。"""
        if fine_at is not None:
            lo, hi = fine_at[idx]
            if hi > lo:
                out: list[float] = []
                for f in path_candles[lo:hi]:
                    out.extend(_bar_path(f))
                return out
        return list(_bar_path(candles[idx]))

    atr_at: list[float] = []
    seen_swings = 0
    trades: list[FadeTrade] = []
    armed: dict[int, bool] = {}
    open_until: list[int] = []   # 建玉が空くまでの足番号。max_open まで持てる
    dead_zones: dict[int, int] = {}   # 抜けられた帯 -> 抜けられた足番号
    tries: dict[int, int] = {}        # 折り返しの足番号 -> 試した回数
    # スイングの足番号 -> 次のスイングまでの値幅(ATR 倍)。弾きの強さ。
    # **次が確定してから入る**ので、先読みにならない。
    swing_move: dict[int, float] = {}
    prev_swing = [None]

    def note_move(sw, atr: float) -> None:
        p = prev_swing[0]
        if p is not None and atr > 0 and p.index != sw.index:
            swing_move[p.index] = abs(sw.price - p.price) / atr
        prev_swing[0] = sw

    def rejection(z) -> float:
        """その帯が過去に弾いた大きさの中央値(ATR 倍)。"""
        vals = [swing_move[sw.index] for sw in z.touches if sw.index in swing_move]
        if not vals:
            return 0.0
        vals.sort()
        return vals[len(vals) // 2]
    was_rising = was_falling = False
    cached: list = []
    cached_swings = -1
    cached_at = -10**9

    def _run_position(*, i, fill_at, limit, stop, risk, long_side, from_below,
                      opposite, zone, width, a, weakening):
        """約定してから決済までを回して 1 件にまとめる。

        **入り口が何であれ、ここを通す。**帯の端で待つ形と、押し負けで
        乗る形の 2 つがあるが、約定と決済の判定を 2 か所に書くと、
        片方だけ直したときに挙動がずれる(過去に踏んでいる)。
        """
        sign = 1.0 if long_side else -1.0
        # **約定した足そのものから見る。**その足の残りで損切りまで
        # 走ることは普通にある。翌足から数えると、いちばん不利な
        # 場面だけを見逃して成績が良く出る。
        forward = candles[fill_at : fill_at + 1 + horizon]
        if len(forward) < horizon + 1:
            return None

        best = worst = 0.0
        why, exit_price = "time", 0.0
        hit_stop = False
        held = horizon
        result = 0.0
        # 売り(下から来た)なら損切りが上・反対側の帯が下。買いは鏡像。
        up = opposite if long_side else stop
        down = stop if long_side else opposite
        be_done = False
        for step, c in enumerate(forward):
            fav = max((c.high - limit) * sign, (c.low - limit) * sign)
            adv = -min((c.high - limit) * sign, (c.low - limit) * sign)
            best = max(best, fav)
            worst = max(worst, adv)
            if intrabar in ("stop_first", "no_same_bar_profit"):
                # 損切りは高安で判定する。同じ足で有利にも動いていても、
                # 順序が分からない以上こちらを先に見る。
                touched = (c.low <= stop) if long_side else (c.high >= stop)
                first = "stop" if touched else None
                if first is None and opposite is not None and (
                    intrabar == "stop_first" or step > 0
                ):
                    reached = ((c.high >= opposite) if long_side
                               else (c.low <= opposite))
                    first = "opp" if reached else None
            else:
                points = bar_points(fill_at + step)
                if step == 0:
                    # **指値へ届いた瞬間から先だけを見る。**
                    after = _after_fill(points, limit, from_below)
                    if after is None:      # 細かい足に無い(隙間)
                        after = [limit]
                    points = after
                else:
                    points = [forward[step - 1].close, *points]
                side = _reach(points, up, down)
                first = None
                if side == ("down" if long_side else "up"):
                    first = "stop"
                elif side is not None:
                    first = "opp"
            if first == "stop":
                hit_stop = True
                held = step
                # **建値へ動かしていれば -1 R ではない。**動かした後の
                # 損切り位置から計算する。
                result = (stop - limit) * sign / risk
                why, exit_price = "stop", stop
                break
            if first == "opp":
                # 反対側の帯の **手前の縁**(建玉を持った時点の値)で手仕舞う。
                held = step
                result = (opposite - limit) * sign / risk
                why, exit_price = "opp", opposite
                break
            # **損切りを動かすのは、その足を見終わってから。**同じ足の中で
            # 伸びてすぐ戻った場合に動かした損切りで切れたことにすると、
            # その足の道順を勝手に決めてしまう。
            if breakeven_at_r and not be_done and best >= breakeven_at_r * risk:
                stop = limit
                up = opposite if long_side else stop
                down = stop if long_side else opposite
                be_done = True
        if not hit_stop and why == "time":
            result = (forward[-1].close - limit) * sign / risk
            exit_price = forward[-1].close

        return FadeTrade(
                bar_index=i,
                zone_price=zone.price if zone is not None else limit,
                zone_width_atr=width,
                touches=zone.count if zone is not None else 0,
                atr=a,
                from_below=from_below,
                entry=limit,
                stop=stop,
                risk_atr=risk / a,
                r_multiple=result,
                max_favourable_r=best / risk,
                max_adverse_r=worst / risk,
                hit_stop=hit_stop,
                bars_held=held,
                long_side=long_side,
                defenders_weak=weakening,
                entry_hour=candles[fill_at].time.hour,
                why=why, exit_price=exit_price,
                fill_index=fill_at,
                opposite_price=opposite if opposite is not None else 0.0,
                zone_low=zone.low if zone is not None else 0.0,
                zone_high=zone.high if zone is not None else 0.0,
                # `Zone` は Swing の列、`RangeEdge` は足番号そのもの。
                zone_touch_bars=tuple(
                    sw if isinstance(sw, int) else sw.index
                    for sw in zone.touches) if zone is not None else (),
        )

    for i, candle in enumerate(candles):
        if higher is not None:
            while hi_i < len(higher) and higher[hi_i].time + span <= candle.time:
                detector.update(higher[hi_i])
                hi_bar = hi_i
                for swing in detector.swings[seen_swings:]:
                    tracker.update(swing, atr=atr_high[hi_i], bar_index=hi_bar)
                    absorb_turn(swing, higher, hi_bar)
                    note_move(swing, atr_high[hi_i])
                seen_swings = len(detector.swings)
                if detector.swings:
                    # **差分だけでは足りない。**同じ向きが続くと検出器は
                    # 末尾を「より極端な方」で **置き換える** ので、
                    # 列の長さが伸びない。折り返しの最値を追うには、
                    # 末尾を毎回入れ直す必要がある。
                    absorb_turn(detector.swings[-1], higher, hi_bar)
                note_swings()
                hi_i += 1
            # **帯を引いた足の物差しで測る。**下位足の ATR で上位足の帯を
            # 測ると、幅の上限も損切りの余裕も小さすぎて別物になる。
            a = (atr_high[hi_bar] if scale_to_zone_timeframe and hi_bar >= 0
                 else atr_low[i])
        else:
            detector.update(candle)
            a = detector.atr or 0.0
            for swing in detector.swings[seen_swings:]:
                tracker.update(swing, atr=a, bar_index=i)
                absorb_turn(swing, candles, i)
                note_move(swing, a)
            seen_swings = len(detector.swings)
            if detector.swings:
                absorb_turn(detector.swings[-1], candles, i)
            note_swings()
        atr_at.append(a)

        # **抜けられた端は捨てる。**その端も、それ以前に付けた折り返しも
        # 一緒に落とす(古い指値を生かさない)。判定は **帯を引いた足の
        # 終値**で行う。下位足の終値で見ると、はみ出しただけで捨ててしまう。
        # ここは建玉の空きや暖機に関係なく **毎足** 通す。飛ばすと、
        # 手が空いていない間の抜けを見落として古い水準が生き残る。
        if drop_broken_edges and zone_source == "range" and range_needs_turn:
            ref = higher[hi_bar].close if higher is not None and hi_bar >= 0 else candle.close
            # デックは前が最も外側、後ろへ行くほど内側。終値が越えた水準は
            # **後ろから** 落ちる。前まで届けば全部消える(帯が無くなる)。
            while turn_hi and turn_hi[-1][1] < ref:
                turn_hi.pop()
            while turn_lo and turn_lo[-1][1] > ref:
                turn_lo.pop()

        if i < warmup or a <= 0 or i + max_wait_bars + horizon >= len(candles):
            continue
        open_until = [b for b in open_until if b >= i]
        if len(open_until) >= max_open:
            continue

        # 有効な帯の一覧を毎足作り直すと、蓄積した帯の数に比例して重くなる
        # (600,000 足 x 数千の帯)。スイングが増えたときと、古い帯が落ちる
        # 頃合いだけ作り直す。**新しい帯の反映が遅れる方向にしかずれない**
        # ので、先読みにはならない。
        age_index = hi_bar if higher is not None else i
        if age_index < 0:
            continue
        if zone_source == "range":
            # 映っている範囲の端。**閉じた足まで**で作る(先読みを避ける)。
            k = hi_bar if higher is not None else i - 1
            if k < 0:
                continue
            if range_needs_turn:
                cut = k - range_bars
                while turn_hi and turn_hi[0][0] < cut:
                    turn_hi.popleft()
                while turn_lo and turn_lo[0][0] < cut:
                    turn_lo.popleft()
                if not turn_hi or not turn_lo:
                    continue
                (ti, top), (bi, bot) = turn_hi[0], turn_lo[0]
                if top <= bot:
                    continue
                hi_lv, lo_lv = top, bot
                if entry_from_recent_turn:
                    # 大枠は残したまま、指値だけ直近の折り目へ寄せる。
                    rh, rl = swing_state["last_high"], swing_state["last_low"]
                    if rh is None or rl is None or rh <= rl:
                        continue
                    hi_lv, lo_lv = min(top, rh), max(bot, rl)
                    if hi_lv <= lo_lv:
                        continue
                cached = [RangeEdge(hi_lv, hi_lv, "top", 1, (ti,), top),
                          RangeEdge(lo_lv, lo_lv, "bottom", 1, (bi,), bot)]
            else:
                if edge_high is None or edge_high[k] <= edge_low[k]:
                    continue
                cached = [RangeEdge(edge_high[k], edge_high[k], "top"),
                          RangeEdge(edge_low[k], edge_low[k], "bottom")]
        elif seen_swings != cached_swings or i - cached_at >= refresh_every:
            cached = tracker.zones(bar_index=age_index, min_touches=min_touches)
            cached_swings, cached_at = seen_swings, i

        if edge_mode == "weakening":
            # **押し負けが起きた瞬間に乗る。**帯に届くのを待たない。
            # 抵抗帯へ向かう安値が切り上がったら買い、支持帯へ向かう
            # 高値が切り下がったら売り。損切りはその構造の外側。
            st = swing_state
            rising = (st["prev_low"] is not None
                      and st["last_low"] > st["prev_low"])
            falling = (st["prev_high"] is not None
                       and st["last_high"] < st["prev_high"])
            fire_long = rising and not was_rising
            fire_short = falling and not was_falling
            was_rising, was_falling = rising, falling
            if not (fire_long or fire_short) or i + 1 >= len(candles):
                continue
            long_side = bool(fire_long)
            # **押されている帯が無ければ乗らない。**利用者の言う
            # 「抵抗帯の押し負け」なので、押される相手が要る。
            above = [z for z in cached if z.price > candle.close]
            below = [z for z in cached if z.price <= candle.close]
            if long_side and not above:
                continue
            if not long_side and not below:
                continue
            fill_at = i + 1                      # 次の足の始値で成行
            limit = candles[fill_at].open
            stop = (st["last_low"] - stop_buffer_atr * a if long_side
                    else st["last_high"] + stop_buffer_atr * a)
            risk = abs(stop - limit)
            if risk <= 0 or (limit <= stop) is long_side:
                continue                          # 既に損切りの向こう側
            target = (min(above, key=lambda z: z.price) if long_side
                      else max(below, key=lambda z: z.price))
            t = _run_position(
                i=i, fill_at=fill_at, limit=limit, stop=stop, risk=risk,
                long_side=long_side, from_below=long_side, opposite=None,
                zone=target, width=target.width / a, a=a, weakening=True,
            )
            if t is not None:
                trades.append(t)
                open_until.append(t.fill_index + t.bars_held)
            continue

        # 上下 2 本の帯が揃っているときだけ触る(利用者の言う「レンジ」)。
        # 片側だけで張ると、そこを抜けられたときに一方的に負ける。
        if drop_broken_edges and zone_source == "pivots":
            # **抜けられた帯は、そこでもう一度折り返すまで使わない。**
            # 消すのではなく休ませる。役割が入れ替わって効き直すことは
            # あるが、抜けられた直後に同じ指値を残すと、戻ってきた勢いで
            # 約定して壊される(利用者の言う「お残り」)。
            for z in cached:
                kz = id(z)
                if candle.close > z.high or candle.close < z.low:
                    dead_zones[kz] = age_index
            cached = [z for z in cached
                      if max((sw.index for sw in z.touches), default=-1)
                      > dead_zones.get(id(z), -1)]
            if not cached:
                continue

        if min_rejection_atr > 0:
            cached = [z for z in cached if rejection(z) >= min_rejection_atr]
            if not cached:
                continue

        usable = cached
        if require_range:
            price = candle.close
            above = [z for z in cached if z.price > price]
            below = [z for z in cached if z.price <= price]
            if not above or not below:
                continue
            top = min(above, key=lambda z: z.price)
            bottom = max(below, key=lambda z: z.price)
            if max_range_atr is not None and (top.price - bottom.price) > max_range_atr * a:
                continue
            usable = [top, bottom]

        for zone in usable:
            width = zone.width / a
            if max_zone_width_atr is not None and width > max_zone_width_atr:
                continue
            key = getattr(zone, "edge_key", None) or id(zone)

            # 帯から十分離れたら、次の待ち伏せを許す。
            # `arm_within_atr` のぶん帯を広げて見る(手前で折り返す形を拾う)。
            near = arm_within_atr * a
            if not (zone.low - near <= candle.high and zone.high + near >= candle.low):
                away = min(abs(candle.close - zone.low), abs(candle.close - zone.high))
                # **再武装の距離は広げない。**ここも広げると、見る範囲を
                # 広げたぶんだけ次の待ち伏せが遅れて、件数が逆に減る。
                if away > a:
                    armed[key] = True
                continue
            if not armed.get(key, True):
                continue

            from_below = candles[i - 1].close < zone.price

            # **流れに逆らう側では張らない。**高値も安値も切り上がって
            # いるのに上の帯で売ると、同じ水準で何度も刈られる。
            if skip_against_trend and edge_mode == "fade":
                st_ = swing_state
                up = (st_["prev_high"] is not None and st_["prev_low"] is not None
                      and st_["last_high"] > st_["prev_high"]
                      and st_["last_low"] > st_["prev_low"])
                down = (st_["prev_high"] is not None and st_["prev_low"] is not None
                        and st_["last_high"] < st_["prev_high"]
                        and st_["last_low"] < st_["prev_low"])
                if (up and from_below) or (down and not from_below):
                    continue

            # 守り手が押し負けているか。見送る材料にも、抜けた側へ乗る
            # 材料にもなる(利用者は後者だと言っている)。
            weakening = defenders_weakening(detector.swings, from_below, age_index)
            take_break = edge_mode == "break" or (edge_mode == "auto" and weakening)
            # 上の端を上抜けるなら買い、下の端を下抜けるなら売り。
            # 跳ね返りに乗るならその逆。
            long_side = from_below if take_break else not from_below

            # ブレイクリスク: 守り手が押し負け始めている帯には近づかない。
            # **足番号は帯を引いた足のもの。**上位足で帯を引いているのに
            # 下位足の番号で確定を判定すると、比較する物差しが揃わない
            # (数が 4 倍あるので、どのスイングも「確定済み」になる)。
            if skip_break_risk and weakening:
                continue

            # 帯を引いた足の最値を使うなら、最後に **閉じた** 上位足まで。
            # 下位足なら 1 本前まで(その足自身の最値で約定判定をすると
            # 「更新したからその値で約定した」という循環になる)。
            if entry_high_h is not None:
                e_hi, e_lo, r_hi, r_lo, k = (
                    entry_high_h, entry_low_h, roll_high_h, roll_low_h, hi_bar)
                k_stop = hi_bar
            else:
                e_hi, e_lo, r_hi, r_lo, k = (
                    entry_high, entry_low, roll_high, roll_low, i - 1)
                k_stop = i

            if from_below:
                if entry_at_zone_extreme:
                    # 帯の極値そのもの = 折り返しの天井で売る。
                    limit = zone.high + entry_beyond_atr * a
                elif e_hi is not None:
                    limit = max(zone.high, e_hi[k])
                else:
                    limit = zone.low + entry_offset_atr * a
                edge = max(zone.high, limit)
                if r_hi is not None:
                    # 帯そのものの縁ではなく、もっと広い範囲の最値を使う。
                    # 帯を作ったスイングの縁だけだと、少しのはみ出しで
                    # 刈られる。利用者の指摘。
                    edge = max(edge, r_hi[k_stop])
                stop = edge + stop_buffer_atr * a
            else:
                if entry_at_zone_extreme:
                    limit = zone.low - entry_beyond_atr * a
                elif e_lo is not None:
                    limit = min(zone.low, e_lo[k])
                else:
                    limit = zone.high - entry_offset_atr * a
                edge = min(zone.low, limit)
                if r_lo is not None:
                    edge = min(edge, r_lo[k_stop])
                stop = edge - stop_buffer_atr * a

            # 指値が約定するのは、価格がそこへ **届いた** ときだけ。
            # 抜けた側を終値で確認する場合は、**確認できた次の足の始値**。
            wait_close = take_break and break_confirm == "close"
            level = limit + (break_confirm_atr * a if from_below
                             else -break_confirm_atr * a)
            fill_at = None
            for j in range(i, min(i + max_wait_bars, len(candles))):
                c = candles[j]
                if blocked_hours_utc and c.time.hour in blocked_hours_utc:
                    continue
                if nfp_blackout_minutes and _near_nfp(c.time, nfp_blackout_minutes):
                    continue
                if wait_close:
                    if ((c.close >= level) if from_below else (c.close <= level)) \
                            and j + 1 < len(candles):
                        fill_at = j + 1
                        limit = candles[j + 1].open
                        break
                elif (c.high >= limit) if from_below else (c.low <= limit):
                    fill_at = j
                    break
            if fill_at is None:
                armed[key] = False
                continue

            armed[key] = False

            if max_tries_per_zone:
                # **同じ水準で負け続ける形を止める。**数えるのは水準の値段
                # ではなく、**その水準を作った折り返し**。帯は少しずつ動くので
                # 値段で数えると同じものと見なせない。
                t0 = zone.touches[0] if zone.touches else None
                tkey = (t0 if isinstance(t0, int)
                        else (id(zone) if t0 is None else t0.index))
                n = tries.get(tkey, 0)
                if n >= max_tries_per_zone:
                    continue
                tries[tkey] = n + 1

            if take_break:
                # 抜けた側に乗るなら、損切りは **帯の内側** へ戻る。
                # 上の端を上抜けて買うなら、損切りは端の下。
                # 終値で確認した場合は約定値(次の足の始値)から引く。
                stop = (limit - stop_buffer_atr * a if long_side
                        else limit + stop_buffer_atr * a)
            risk = abs(stop - limit)
            if risk <= 0:
                continue

            # 反対側(利益方向)の帯。往復を刈るときの手仕舞い先。
            # **抜けた側に乗る場合は損失方向なので使わない。**
            opposite = None
            if exit_at_opposite_zone and not take_break:
                # **利益方向にあり、かつ指値より手前で決済になる帯だけ**を
                # 反対側とみなす。ここを緩めると、決済価格が損切りより
                # 悪い帯を掴む。実測では負けの半分がそれで、平均 -3.78 R、
                # 最悪 -35 R まで行っていた(損切りが機能していない)。
                other = []
                for z in cached:
                    if z is zone:
                        continue
                    # 往復の相手は **大枠** の端(あれば)。指値だけ折り目へ
                    # 寄せているので、利確まで折り目にすると幅が消える。
                    arena = getattr(z, "arena", 0.0)
                    edge = (z.high if from_below else z.low) if not arena else arena
                    gain = (limit - edge) if from_below else (edge - limit)
                    if gain > 0:
                        other.append(z)
                if other:
                    lvl = lambda z: getattr(z, "arena", 0.0) or (
                        z.high if from_below else z.low)
                    picked = (max(other, key=lvl) if from_below
                              else min(other, key=lvl))
                    # **縁の値をここで確定させる。**Zone は接触が足される
                    # たびに low/high が広がるので、参照のまま持つと決済
                    # 判定の時点で縁が動いている。実測ではそれで縁が指値の
                    # 向こう側へ回り込み、-35 R の「利確」が発生していた。
                    opposite = (getattr(picked, "arena", 0.0)
                                or (picked.high if from_below else picked.low))

            t = _run_position(
                i=i, fill_at=fill_at, limit=limit, stop=stop, risk=risk,
                long_side=long_side, from_below=from_below,
                opposite=opposite, zone=zone, width=width, a=a,
                weakening=weakening,
            )
            if t is None:
                continue
            trades.append(t)
            open_until.append(t.fill_index + t.bars_held)
            if len(open_until) >= max_open:
                break

    return trades
