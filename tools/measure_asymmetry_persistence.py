"""ブレイクの非対称は、続くのか。それとも当たらないのか。

期間による乖離を分解したら、**相場の素の性質は変わっていなかった**
(6 銘柄合算のブレイク追随は -0.06 / +0.01 / -0.04 で横ばい)。
変わっていたのは GBP/USD だけで、2000-2006 に +0.70 あった非対称が
その後 -0.03 / -0.20 へ消えた。

だとすれば問いは 1 つになる:
**去年の非対称は、来年の非対称を当てるか。**

当たるなら、直近の窓で測って銘柄と時期を選べる(取引せずに測れる)。
当たらないなら、ブレイク系はそもそも時期を選べない。

    PYTHONPATH=. python tools/measure_asymmetry_persistence.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.data.resample import resample_candles
from llmfx.research.zone_swing import collect_swing_trades

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf",
         "audusd", "eurusd", "gbpjpy", "usdjpy"]
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="exec_turn", zone_wait_bars=24, zone_entry_max_atr=2.0,
            entry_signal="exec", entry_fill="level", entry_fallback="structure",
            stop_basis="band", stop_buffer_atr=1.5, min_stop_atr=2.0,
            reversal_signal="both", max_flips=0, max_adds=0, max_open=4,
            reverse_entry=True, blocked_hours_utc=frozenset({21, 22}),
            fill_bar="path")
FWD = LOOK = 30
ATR_N = 14


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


asym: dict = defaultdict(dict)     # [銘柄][年] -> 非対称(走った − 戻された)
perf: dict = defaultdict(dict)     # [銘柄][年] -> 手法の期待値
for p in PAIRS:
    cfg = AppConfig.load(f"configs/h1/{p}.yaml")
    cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                       cfg.backtest.holdout_start, "dev")
    pv = cfg.instrument.pip_size
    h4 = resample_candles(cs, 240)
    a = atr_series(h4)
    hi = np.array([x.high for x in h4]); lo = np.array([x.low for x in h4])
    cl = np.array([x.close for x in h4])
    by_year = defaultdict(list)
    for i in range(max(LOOK, ATR_N) + 1, len(h4) - FWD):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up = cl[i] > hi[i - LOOK:i].max()
        dn = cl[i] < lo[i - LOOK:i].min()
        if up == dn:
            continue
        s = 1.0 if up else -1.0
        by_year[h4[i].time.year].append((cl[i + FWD] - cl[i]) * s / a[i])
    for y, v in by_year.items():
        if len(v) < 40:
            continue
        m = np.array(v)
        go, back = m[m > 0], m[m < 0]
        if len(go) and len(back):
            asym[p][y] = float(go.mean() + back.mean())
    pr = defaultdict(list)
    for t in collect_swing_trades(cs, **BASE,
                                  spread=cfg.execution.spread_pips * pv,
                                  slippage=cfg.execution.slippage_pips * pv):
        if t.kind == "zone":
            pr[cs[t.entry_index].time.year].append(t.r_multiple * t.size)
    for y, v in pr.items():
        if len(v) >= 30:
            perf[p][y] = float(np.mean(v))
    print(f"  {p} 済み", flush=True)


def corr(xs, ys, label):
    x, y = np.array(xs), np.array(ys)
    if len(x) < 10:
        print(f"  {label:<34} 件数不足 ({len(x)})"); return
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt((len(x) - 2) / max(1e-12, 1 - r * r))
    rk = float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
    print(f"  {label:<34} n={len(x):>4}  r={r:+.3f}  t={t:+.2f}  順位相関={rk:+.3f}")


print(f"\n{'=' * 78}\n## 去年の非対称は、来年を当てるか\n{'=' * 78}")
xa, ya, xp, yp, xs, ys = [], [], [], [], [], []
for p in PAIRS:
    for y, v in asym[p].items():
        if y + 1 in asym[p]:
            xa.append(v); ya.append(asym[p][y + 1])
        if y + 1 in perf[p]:
            xp.append(v); yp.append(perf[p][y + 1])
        if y in perf[p]:
            xs.append(v); ys.append(perf[p][y])
corr(xa, ya, "去年の非対称 → 今年の非対称")
corr(xs, ys, "同じ年の非対称 → 同じ年の成績")
corr(xp, yp, "去年の非対称 → 今年の成績")

print(f"\n{'=' * 78}\n## 銘柄ごとの非対称(年)\n{'=' * 78}")
years = sorted({y for p in PAIRS for y in asym[p]})
print(f"{'銘柄':<9}" + "".join(f"{y % 100:>5}" for y in years))
for p in PAIRS:
    print(f"{p:<9}" + "".join(
        f"{asym[p][y]:>+5.1f}" if y in asym[p] else "    ." for y in years))
print("\ndone", flush=True)
