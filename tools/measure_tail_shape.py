"""裾の中身を左右そろえて見る。

利用者の問い:「裾とは、たまに値段がすっぽ抜けて大きく動くことか?」
1 本で飛んだだけなら、30 本のうち有利方向へ動く足は 50% 付近になるはず。
積み上がりなら 50% を明確に超える。左の裾も同じ厳しさで測る。
"""
from __future__ import annotations
import sys
import numpy as np
from llmfx.data import load_candles_csv
from llmfx.data.resample import resample_candles

PAIRS = ["gbpusd", "usdcad", "gbpjpy", "nzdusd", "usdjpy", "eurusd"]
FWD = 30          # H4 30 本 = 5 日
LOOK = 30         # 直近 30 本(=5 日)の最値を抜けたら「ブレイク」
ATR_N = 14
DEV = "2019-12-31"   # 開発用のみ


def atr(c, n=ATR_N):
    tr = np.empty(len(c))
    tr[0] = c[0].high - c[0].low
    for i in range(1, len(c)):
        pc = c[i - 1].close
        tr[i] = max(c[i].high - c[i].low, abs(c[i].high - pc), abs(c[i].low - pc))
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n] = tr[1:n + 1].mean()
        for i in range(n + 1, len(c)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def events(pair):
    c = [x for x in load_candles_csv(f"/home/user/llmfx/data/{pair}_m15.csv")
         if str(x.time)[:10] <= DEV]
    h4 = resample_candles(c, 240)
    a = atr(h4)
    hi = np.array([x.high for x in h4])
    lo = np.array([x.low for x in h4])
    cl = np.array([x.close for x in h4])
    out = []
    for i in range(max(LOOK, ATR_N) + 1, len(h4) - FWD):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up = cl[i] > hi[i - LOOK:i].max()
        dn = cl[i] < lo[i - LOOK:i].min()
        if up == dn:
            continue
        s = 1.0 if up else -1.0
        steps = (np.diff(cl[i:i + FWD + 1]) * s) / a[i]   # 1 本ごとの前進(ATR)
        out.append((steps.sum(), steps))
    return out


def tail_body(rows, label):
    print(f"\n{label}")
    print(f"{'銘柄':<10}{'件数':>7}{'いちばん大きい1本':>18}{'が占める割合':>14}"
          f"{'進んだ足の割合':>16}{'5日の合計':>12}")
    for pair, rs in rows_by_pair.items():
        tot = np.array([t for t, _ in rs])
        k = max(1, int(len(rs) * 0.1))
        idx = np.argsort(tot)
        pick = idx[-k:] if label.startswith("右") else idx[:k]
        biggest, share, fwd, tsum = [], [], [], []
        for j in pick:
            t, st = rs[j]
            sgn = 1.0 if label.startswith("右") else -1.0
            b = (st * sgn).max()
            biggest.append(b)
            share.append(b / abs(t) if t else np.nan)
            fwd.append(((st * sgn) > 0).mean())
            tsum.append(abs(t))
        print(f"{pair:<10}{k:>7}{np.mean(biggest):>15.2f} ATR{np.mean(share):>13.0%}"
              f"{np.mean(fwd):>15.0%}{np.mean(tsum):>9.2f} ATR")


rows_by_pair = {p: events(p) for p in PAIRS}
tail_body(None, "右の裾(上位1割)= 大きく勝った週の中身")
tail_body(None, "左の裾(下位1割)= 大きく負けた週の中身")
print("\n※ 「進んだ足の割合」= 30 本のうち何本がその方向へ動いたか。")
print("   1 本ですっぽ抜けただけなら 50% 付近になるはず。")
