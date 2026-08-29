"""非対称は「持続しない」のか、「測れていない」のか。

年ごとの推定が雑音だらけなら、去年→今年の相関 r=+0.042 は
「持続しない」証拠にならない。3 つを分けて見る:

  A 折半法      同じ期間を 2 つに割って揃うか。推定の安定性そのもの
  B 窓の長さ    四半期 / 半年 / 1 年 / 2 年。**粗くするほど安定するはず**
  C 束ねる      10 銘柄をまたいで 1 つの「相場全体の状態」にする。
                2008 年に 9/10 銘柄が揃った以上、共通成分があるはず

**細かくすると悪化する。**月へ下ろすのは逆方向。

    PYTHONPATH=. python tools/measure_asymmetry_reliability.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.data.resample import resample_candles

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf",
         "audusd", "eurusd", "gbpjpy", "usdjpy"]
FWD = LOOK = 30
ATR_N = 14
OUT = "docs/handoff/asymmetry-events.json"


def atr_series(c, n=ATR_N):
    tr = np.empty(len(c))
    tr[0] = c[0].high - c[0].low
    for i in range(1, len(c)):
        pc = c[i - 1].close
        tr[i] = max(c[i].high - c[i].low, abs(c[i].high - pc), abs(c[i].low - pc))
    out = np.full(len(c), np.nan)
    out[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def load_events() -> dict:
    """[銘柄] -> [(年, 四半期通し番号, 追随)] を作る。保存して再利用する。"""
    if os.path.exists(OUT):
        return json.load(open(OUT, encoding="utf-8"))
    ev: dict = {}
    for p in PAIRS:
        cfg = AppConfig.load(f"configs/h1/{p}.yaml")
        cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                           cfg.backtest.holdout_start, "dev")
        h4 = resample_candles(cs, 240)
        a = atr_series(h4)
        hi = np.array([x.high for x in h4]); lo = np.array([x.low for x in h4])
        cl = np.array([x.close for x in h4])
        rows = []
        for i in range(max(LOOK, ATR_N) + 1, len(h4) - FWD):
            if not np.isfinite(a[i]) or a[i] <= 0:
                continue
            up = cl[i] > hi[i - LOOK:i].max()
            dn = cl[i] < lo[i - LOOK:i].min()
            if up == dn:
                continue
            s = 1.0 if up else -1.0
            t = h4[i].time
            rows.append([t.year * 4 + (t.month - 1) // 3,
                         float((cl[i + FWD] - cl[i]) * s / a[i])])
        ev[p] = rows
        print(f"  {p} {len(rows):,} 事象", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(ev, open(OUT, "w", encoding="utf-8"))
    return ev


def stat(v):
    m = np.asarray(v)
    return float(m.mean())


def buckets(ev, quarters: int, pooled: bool):
    """窓ごとに (キー -> 追随の配列)。`pooled` なら銘柄をまたいで束ねる。"""
    g = defaultdict(list)
    for p, rows in ev.items():
        for q, r in rows:
            key = q // quarters if pooled else (p, q // quarters)
            g[key].append(r)
    return g


def reliability(g, need=40):
    """折半法。奇数番目と偶数番目で揃うか。Spearman-Brown で補正。"""
    xs, ys = [], []
    for v in g.values():
        if len(v) < need:
            continue
        xs.append(stat(v[0::2])); ys.append(stat(v[1::2]))
    if len(xs) < 8:
        return None, len(xs)
    r = float(np.corrcoef(xs, ys)[0, 1])
    return 2 * r / (1 + r), len(xs)          # 全長に直す


def autocorr(g, pooled, need=40):
    """1 つ前の窓と今の窓。"""
    xs, ys = [], []
    keys = sorted(g, key=lambda k: k if pooled else k[1])
    for k in keys:
        prev = (k - 1) if pooled else (k[0], k[1] - 1)
        if prev in g and len(g[k]) >= need and len(g[prev]) >= need:
            xs.append(stat(g[prev])); ys.append(stat(g[k]))
    if len(xs) < 8:
        return None, len(xs)
    r = float(np.corrcoef(xs, ys)[0, 1])
    t = r * np.sqrt((len(xs) - 2) / max(1e-12, 1 - r * r))
    return (r, float(t)), len(xs)


ev = load_events()
print(f"\n{'=' * 86}")
print("## 推定は安定しているか(折半法)と、1 つ前の窓は当たるか")
print("=" * 86)
print(f"{'束ね方':<12}{'窓':<10}{'組の数':>8}{'折半の信頼性':>14}"
      f"{'1つ前との相関':>14}{'t':>8}{'雑音を補正':>19}")
for pooled, pname in ((False, "銘柄ごと"), (True, "10 銘柄を束ねる")):
    for q, qn in ((1, "四半期"), (2, "半年"), (4, "1 年"), (8, "2 年")):
        g = buckets(ev, q, pooled)
        rel, nrel = reliability(g)
        ac, nac = autocorr(g, pooled)
        if rel is None or ac is None:
            print(f"{pname:<12}{qn:<10}{len(g):>8}   件数不足"); continue
        r, t = ac
        adj = r / rel if rel > 0.05 else float("nan")
        print(f"{pname:<12}{qn:<10}{nac:>8}{rel:>+14.3f}"
              f"{r:>+12.3f}{t:>+8.2f}{adj:>+19.3f}")

print(f"\n{'=' * 86}")
print("## 相場全体の状態(10 銘柄を束ねた年ごとの追随)")
print("=" * 86)
g = buckets(ev, 4, True)
for k in sorted(g):
    v = np.array(g[k])
    print(f"  {k}年  {len(v):>5,} 事象  追随 {v.mean():>+6.3f}  "
          f"{'#' * max(0, int((v.mean() + 0.45) * 30))}")
print("\ndone", flush=True)
