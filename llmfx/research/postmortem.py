"""建玉を後から解剖する — **勝ちが何を共有していたか**を数える。

利用者の提案:

    チャートを逆再生して勝ちに繋がるパターンを見つける。普通に再生すると
    答えが先に見えないので探すのが難しいが、逆再生なら予め答えがわかった
    上でその結末に至る要因を模索できる。

考え方は正しいが、**目でやるといちばん罠にはまる**。答えを知った状態で
見れば、人はどんな並びにも理由を見つけてしまう。そこで機械にやらせる。

守ること:

1. **因果性** — 特徴はその瞬間までのデータだけで作る。先の値動きは使わない
2. **単調性** — 5 分位に切って期待値が単調に動くものだけを残す。
   1 区分だけ跳ねるのはノイズ(作法 1)
3. **確認は別の銘柄で** — 探索に使った銘柄で確かめても意味が無い(作法 4)

もう一つ、利用者の切り分け:

    買いで入り最高値ではプラスだったのに最終的にマイナスなら、決済の位置に
    誤りがある。逆に直ぐに損切られているなら、損切り位置かエントリー位置が
    おかしい。

`diagnose()` がこれを 3 群に割る。**打つ手が群ごとに真逆になる。**
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..domain.types import Candle
from .zone_fade import FadeTrade


@dataclass
class Row:
    """1 建玉ぶんの特徴と結果."""

    at: int
    """約定した足。突き合わせと、図への戻し先に要る。"""
    r: float
    """コストを引いた成果(R)。"""
    mfe: float
    """決済までに最も利益方向へ動いた幅(R)。"""
    mae: float
    """決済までに最も損失方向へ動いた幅(R)。"""
    bars: int
    group: str
    """`diagnose()` が付ける切り分け。"""
    f: dict[str, float]
    """その瞬間までで計算できる特徴。**先の値動きは入っていない。**"""


def diagnose(r: float, mfe: float, mae: float) -> str:
    """利用者の切り分け。**群ごとに打つ手が真逆になる。**

    - 伸びたのに取れなかった → 決済の位置の問題
    - 一度も伸びなかった → 入り口か損切りの位置の問題
    - 取れた → そのまま
    """
    if r > 0:
        return "取れた"
    if mfe >= 1.0:
        return "伸びたのに取れなかった"
    if mfe < 0.5:
        return "一度も伸びなかった"
    return "どちらとも言えない"


def _efficiency(closes: list[float]) -> float:
    """効率比 — 端から端までの動き ÷ 1 本ずつの動きの合計。

    1 に近いほど一直線、0 に近いほど行ったり来たり。
    """
    if len(closes) < 3:
        return 0.0
    path = sum(abs(b - a) for a, b in zip(closes, closes[1:]))
    return abs(closes[-1] - closes[0]) / path if path else 0.0


def features(
    candles: list[Candle],
    trades: list[FadeTrade],
    costs: list[float] | None = None,
) -> list[Row]:
    """建玉ごとに、**約定した瞬間までで作れる特徴**を並べる。

    `costs` は 1 建玉ぶんのコスト(R)。渡さないと素の成果になる。
    """
    closes = [c.close for c in candles]
    rows: list[Row] = []
    for k, t in enumerate(trades):
        i = t.fill_index
        if i < 200 or t.atr <= 0:
            continue
        a = t.atr
        sign = 1.0 if t.long_side else -1.0
        near = closes[i - 96 : i + 1]
        far = closes[i - 200 : i + 1]
        bar = candles[i]
        # 帯を作った折り返しから何本経ったか
        age = i - max(t.zone_touch_bars) if t.zone_touch_bars else -1
        f = {
            "値幅の水準(ATR/価格 bp)": a / bar.close * 10_000,
            "値幅の伸び(いま/96本前)": a / max(
                1e-9, (max(x.high for x in candles[i - 110 : i - 96])
                       - min(x.low for x in candles[i - 110 : i - 96])) / 14),
            "直線性(96本)": _efficiency(near),
            "直線性(200本)": _efficiency(far),
            "直近の動き(24本 ATR・順方向+)": (closes[i] - closes[i - 24]) / a * sign,
            "直近の動き(96本 ATR・順方向+)": (closes[i] - closes[i - 96]) / a * sign,
            "約定足の値幅(ATR)": (bar.high - bar.low) / a,
            "損切りまで(ATR)": t.risk_atr,
            "利確まで(R)": (abs(t.opposite_price - t.entry) / (t.risk_atr * a)
                          if t.opposite_price else 0.0),
            "帯の接触回数": float(t.touches),
            "帯の幅(ATR)": t.zone_width_atr,
            "折り返しからの経過(本)": float(age),
            "UTC 時": float(t.entry_hour),
            "買いか": 1.0 if t.long_side else 0.0,
            "押し負けていたか": 1.0 if t.defenders_weak else 0.0,
        }
        r = t.r_multiple - (costs[k] if costs else 0.0)
        rows.append(Row(at=i, r=r, mfe=t.max_favourable_r, mae=t.max_adverse_r,
                        bars=t.bars_held,
                        group=diagnose(r, t.max_favourable_r, t.max_adverse_r), f=f))
    return rows


def group_table(rows: list[Row]) -> list[tuple[str, int, float, float, float, float]]:
    """切り分けごとの件数と、逆行 / 順行の中央値."""
    out = []
    for g in ("取れた", "伸びたのに取れなかった", "一度も伸びなかった",
              "どちらとも言えない"):
        sub = [x for x in rows if x.group == g]
        if not sub:
            continue
        out.append((g, len(sub), len(sub) / len(rows),
                    median(x.r for x in sub), median(x.mae for x in sub),
                    median(x.mfe for x in sub)))
    return out


def quantiles(rows: list[Row], key: str, n: int = 5):
    """特徴を n 分位に切って、区分ごとの期待値を出す。

    **単調に動くかどうかだけを見る。**1 区分だけ跳ねているものは
    ノイズとして捨てる(作法 1)。
    """
    vals = sorted((x.f[key], x.r) for x in rows if key in x.f)
    if len(vals) < n * 20:
        return []
    # **値がほとんど動かない特徴は分位に切れない。**切ると並び順で
    # 割ることになり、時間の順序を特徴と取り違える(単調に見えてしまう)。
    if len({round(v, 9) for v, _ in vals}) < n:
        return []
    step = len(vals) / n
    out = []
    for k in range(n):
        chunk = vals[int(k * step) : int((k + 1) * step)]
        if not chunk:
            continue
        rs = [r for _, r in chunk]
        out.append((chunk[0][0], chunk[-1][0], len(rs), sum(rs) / len(rs)))
    return out


def split_binary(rows: list[Row], key: str):
    """0/1 の特徴は 2 群で比べる。分位に切っても意味が無い。"""
    out = []
    for v in (0.0, 1.0):
        sub = [x.r for x in rows if x.f.get(key) == v]
        if len(sub) >= 20:
            out.append((v, len(sub), sum(sub) / len(sub)))
    return out if len(out) == 2 else []


def monotonic(table) -> float:
    """区分の期待値が単調に動いているか。1 に近いほど素直、0 ならばらばら。

    隣り合う区分の差の符号が揃っている割合で測る。
    """
    if len(table) < 3:
        return 0.0
    diffs = [b[3] - a[3] for a, b in zip(table, table[1:])]
    up = sum(1 for d in diffs if d > 0)
    return max(up, len(diffs) - up) / len(diffs)
