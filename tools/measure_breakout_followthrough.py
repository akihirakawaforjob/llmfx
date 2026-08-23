"""ブレイクした後、走るのか打ち消されるのか。銘柄ごとに素で測る。

利用者の理解:「負けている銘柄は一方向に進み続けているのでは」
これが合っているかを、取引成績を一切見ずに値動きだけで確かめる。
"""
from __future__ import annotations
import numpy as np
from llmfx.data import load_candles_csv
from llmfx.data.resample import resample_candles

PAIRS = ["gbpusd", "usdcad", "gbpjpy", "nzdusd", "usdjpy",
         "eurusd", "audjpy", "audusd", "eurjpy", "usdchf"]
STRAT = {"gbpjpy": +0.301, "gbpusd": +0.202, "usdjpy": +0.117,
         "audjpy": +0.107, "eurusd": +0.104, "eurjpy": +0.042,
         "audusd": +0.029, "usdchf": -0.012, "usdcad": -0.132,
         "nzdusd": -0.141}
FWD, LOOK, ATR_N, DEV = 30, 30, 14, "2019-12-31"


def atr(c, n=ATR_N):
    tr = np.empty(len(c)); tr[0] = c[0].high - c[0].low
    for i in range(1, len(c)):
        pc = c[i - 1].close
        tr[i] = max(c[i].high - c[i].low, abs(c[i].high - pc), abs(c[i].low - pc))
    out = np.full(len(c), np.nan)
    out[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


rows = []
for pair in PAIRS:
    c = [x for x in load_candles_csv(f"/home/user/llmfx/data/{pair}_m15.csv")
         if str(x.time)[:10] <= DEV]
    h4 = resample_candles(c, 240)
    a = atr(h4)
    hi = np.array([x.high for x in h4]); lo = np.array([x.low for x in h4])
    cl = np.array([x.close for x in h4])
    move, eff = [], []
    for i in range(max(LOOK, ATR_N) + 1, len(h4) - FWD):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up = cl[i] > hi[i - LOOK:i].max()
        dn = cl[i] < lo[i - LOOK:i].min()
        if up == dn:
            continue
        s = 1.0 if up else -1.0
        seg = cl[i:i + FWD + 1]
        move.append((seg[-1] - seg[0]) * s / a[i])
        wig = np.abs(np.diff(seg)).sum()
        eff.append(abs(seg[-1] - seg[0]) / wig if wig else np.nan)
    m = np.array(move); e = np.array(eff)
    go, back = m[m > 0], m[m < 0]
    rows.append((pair, len(m), (m > 0).mean(), go.mean(), -back.mean(),
                 go.mean() + back.mean(), np.nanmean(e), STRAT[pair]))

print("### ブレイクした後 5 日でどうなったか(H4・開発用 2000-2019)")
print(f"{'銘柄':<9}{'件数':>7}{'走った割合':>11}{'走った幅':>10}{'戻された幅':>11}"
      f"{'差':>8}{'直線性':>8}{'手法R':>9}")
for r in rows:
    print(f"{r[0]:<9}{r[1]:>7,}{r[2]:>10.1%}{r[3]:>9.2f}{r[4]:>10.2f}"
          f"{r[5]:>+8.2f}{r[6]:>8.3f}{r[7]:>+9.3f}")

st = np.array([r[7] for r in rows])
for k, name in ((2, "走った割合"), (5, "差(走った-戻された)"), (6, "直線性")):
    v = np.array([r[k] for r in rows])
    rv, rs = v.argsort().argsort(), st.argsort().argsort()
    print(f"{name:<22} と手法の順位相関: "
          f"{np.corrcoef(rv, rs)[0, 1]:+.2f}")
