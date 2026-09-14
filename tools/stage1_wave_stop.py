"""段 1(直した後)— コストを抜いて、全銘柄を 1 つに.

直したもの:
  1 跳ね返りの注文は **帯の手前** にしか置かない(買い・売り両方)
  2 損切りは **指値の位置から** 直近の推進波 1 本分

掃引は 2 軸だけ。軸を増やすほど偶然プラスのセルが出る。
  stop_wave_mult   0.8 / 1.0 / 1.2
  wave_spread_max  2.0 / 3.0 / 無制限

1 銘柄ごとに集計を保存して push する(コンテナが回収されるため)。

    PYTHONPATH=. python tools/stage1_wave_stop.py
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict

import numpy as np

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf",
         "audusd", "eurusd", "gbpjpy", "usdjpy"]
MULTS = (0.8, 1.0, 1.2)
SPREADS = (2.0, 3.0, 0.0)          # 0.0 = 無制限
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="method", zone_entry_max_atr=2.0, entry_signal="exec",
            entry_fill="level", entry_fallback="structure",
            stop_basis="recent_waves", wave_lookback=3,
            wave_source="structure", min_stop_atr=0.0,
            reversal_signal="both", max_flips=0, max_adds=0, max_open=4,
            blocked_hours_utc=frozenset({21, 22}), fill_bar="path",
            spread=0.0, slippage=0.0)
OUT = "docs/handoff/stage1-wave.json"


def run() -> dict:
    got = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for p in PAIRS:
        if p in got:
            continue
        cfg = AppConfig.load(f"configs/h1/{p}.yaml")
        cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                           cfg.backtest.holdout_start, "dev")
        cell = {}
        for m in MULTS:
            for sp in SPREADS:
                agg: dict = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
                for t in collect_swing_trades(cs, **BASE, stop_wave_mult=m,
                                              wave_spread_max=sp):
                    if t.kind != "zone":
                        continue
                    ym = cs[t.entry_index].time.strftime("%Y-%m")
                    mech = ("跳ね返り" if (t.zone_key == "bottom") == t.long_side
                            else "ブレイク")
                    r = float(t.r_multiple * t.size)
                    c = agg[f"{ym}|{mech}"]
                    c[0] += 1; c[1] += r; c[2] += r * r
                    if r > 0:
                        c[3] += 1; c[4] += r
                cell[f"{m}|{sp}"] = dict(agg)
                n = sum(v[0] for v in agg.values())
                print(f"  {p} 倍率 {m} ばらつき {sp or '無制限'} → {n:,} 件",
                      flush=True)
        got[p] = cell
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(got, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        for cmd in (["git", "add", "-f", OUT],
                    ["git", "commit", "-q", "-m", f"wip: 段 1(波)— {p}"],
                    ["git", "push", "-q", "origin", "HEAD"]):
            subprocess.run(cmd, check=False, capture_output=True)
        print(f"  {p} 保存", flush=True)
    return got


def pool(cells, kind=None):
    g = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
    for cell in cells:
        for key, v in cell.items():
            ym, k = key.split("|")
            if kind is not None and k != kind:
                continue
            c = g[ym]
            for j in range(5):
                c[j] += v[j]
    return g


def summarize(g):
    n = sum(v[0] for v in g.values())
    tot = sum(v[1] for v in g.values())
    sq = sum(v[2] for v in g.values())
    nw = sum(v[3] for v in g.values())
    sw = sum(v[4] for v in g.values())
    if n < 2:
        return None
    mean = tot / n
    var = max(0.0, (sq - n * mean * mean) / (n - 1))
    t = mean / np.sqrt(var / n) if var > 0 else 0.0
    return n, nw / n, (sw / nw if nw else 0.0), mean, float(t)


def boot(g, draws=10_000):
    keys = list(g)
    if not keys:
        return np.zeros(draws), 0
    sums = np.array([g[k][1] for k in keys])
    ns = np.array([g[k][0] for k in keys], dtype=float)
    rng = np.random.default_rng(20260914)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    tot, cnt = sums[idx].sum(axis=1), ns[idx].sum(axis=1)
    return np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0), len(keys)


got = run()
for kind, title in ((None, "合算"), ("跳ね返り", "跳ね返り"), ("ブレイク", "ブレイク")):
    print(f"\n{'=' * 88}\n## {title} — コスト無し・全銘柄を 1 つに\n{'=' * 88}")
    print(f"{'倍率':>6}{'ばらつき':>10}{'件数':>9}{'勝率':>7}{'平均勝':>8}"
          f"{'期待値R':>10}{'t':>7}{'月クラスタ 95%CI':>28}")
    for m in MULTS:
        for sp in SPREADS:
            g = pool([got[p][f"{m}|{sp}"] for p in PAIRS], kind)
            r = summarize(g)
            if r is None or r[0] < 200:
                print(f"{m:>6.1f}{sp or '無制限':>10}  件数不足"); continue
            n, wr, aw, mean, t = r
            bs, _ = boot(g)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            gate = "またぐ" if lo <= 0 <= hi else "**またがない**"
            print(f"{m:>6.1f}{(sp or '無制限'):>10}{n:>9,}{wr:>7.1%}{aw:>+8.2f}"
                  f"{mean:>+10.4f}{t:>7.2f}   {lo:+.4f} 〜 {hi:+.4f} {gate}")
print("\ndone", flush=True)
