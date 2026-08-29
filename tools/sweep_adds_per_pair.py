"""損切りの共有を直した後の買い増し。銘柄は混ぜない。R/月 を必ず出す。"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from llmfx.config import AppConfig
from llmfx.backtest.split import split_candles
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

USED = ["audusd", "eurusd", "gbpjpy", "usdjpy"]
UNUSED = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf"]
PAIRS = USED + UNUSED
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="exec_turn", zone_wait_bars=24, zone_entry_max_atr=2.0,
            entry_signal="exec", entry_fill="level", entry_fallback="structure",
            stop_basis="band", stop_buffer_atr=1.5, min_stop_atr=2.0,
            reversal_signal="both", max_flips=0, max_open=4,
            reverse_entry=True, blocked_hours_utc=frozenset({21, 22}))
D = {}
for p in PAIRS:
    cfg = AppConfig.load(f"/home/user/llmfx/configs/h1/{p}.yaml")
    cs = split_candles(load_candles_csv(f"/home/user/llmfx/data/{p}_m15.csv"),
                       cfg.backtest.holdout_start, "dev")
    pv = cfg.instrument.pip_size
    D[p] = (cs, cfg.execution.spread_pips*pv, cfg.execution.slippage_pips*pv,
            (cs[-1].time - cs[0].time).days / 30.44)
    print(p, flush=True)

HDR = (f"{'区分':<13}{'件数':>7}{'買い':>7}{'売り':>7}{'勝率':>7}"
       f"{'平均勝':>8}{'平均負':>8}{'期待値R':>9}{'t':>7}{'R/月':>8}")

def line(tag, sel, months):
    if len(sel) < 30:
        print(f"{tag:<13}{len(sel):>7}   件数不足"); return None
    v = np.array([t.r_multiple * t.size for t in sel])
    w, l = v[v > 0], v[v <= 0]
    t = float(v.mean()/(v.std(ddof=1)/np.sqrt(len(v))))
    nb = sum(1 for x in sel if x.long_side)
    print(f"{tag:<13}{len(v):>7,}{nb:>7,}{len(v)-nb:>7,}{(v>0).mean():>7.1%}"
          f"{w.mean():>+8.2f}{l.mean():>+8.2f}{v.mean():>+9.3f}{t:>7.2f}"
          f"{len(v)/months*v.mean():>+8.2f}", flush=True)
    return v.mean(), len(v)/months*v.mean()

for fb in ("path", "adverse"):
    print(f"\n{'='*92}\n## 測り方 {fb} — 損切りの共有を直した後\n{'='*92}", flush=True)
    tot = {}
    for grp, gn in ((USED, "条件探しに使った 4"), (UNUSED, "使っていない 6")):
        for p in grp:
            cs, sp, sl, months = D[p]
            tr = collect_swing_trades(cs, **BASE, max_adds=3, fill_bar=fb,
                                      spread=sp, slippage=sl)
            byp = defaultdict(list)
            for t in tr:
                if t.kind == "add":
                    byp[t.position_id].append(t)
            sq = {}
            for pid, lg in byp.items():
                for n, t in enumerate(sorted(lg, key=lambda x: x.entry_index), 1):
                    sq[id(t)] = n
            print(f"\n### {p}   ({gn})\n" + HDR, flush=True)
            tot[p] = {}
            tot[p]["zone"] = line("zone(最初)",
                                  [t for t in tr if t.kind == "zone"], months)
            for n in (1, 2, 3):
                tot[p][f"add{n}"] = line(
                    f"add {n} 本目",
                    [t for t in tr if t.kind == "add" and sq.get(id(t)) == n],
                    months)
            tot[p]["addall"] = line("add 合計",
                                    [t for t in tr if t.kind == "add"], months)
print("\ndone", flush=True)
