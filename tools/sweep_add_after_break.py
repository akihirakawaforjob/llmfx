"""帯を抜けた後だけ買い増した場合の実測。絞る前と並べる。

利用者の指摘:「ブレイク前の帯に向かって安値切り上げが起きた場合にも
買い増ししている。ダマシや大口の防衛が成功すれば損切りが増える」。

飛ばした買い増しの枠は後ろへずれるので、内訳の引き算では出ない。
必ず回して確かめること。銘柄は混ぜない。R/月 を必ず出す。

    PYTHONPATH=. python tools/sweep_add_after_break.py
"""
from __future__ import annotations
import numpy as np
from llmfx.config import AppConfig
from llmfx.backtest.split import split_candles
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

USED = ["audusd", "eurusd", "gbpjpy", "usdjpy"]
UNUSED = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf"]
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="exec_turn", zone_wait_bars=24, zone_entry_max_atr=2.0,
            entry_signal="exec", entry_fill="level", entry_fallback="structure",
            stop_basis="band", stop_buffer_atr=1.5, min_stop_atr=2.0,
            reversal_signal="both", max_flips=0, max_adds=3, max_open=4,
            reverse_entry=True, blocked_hours_utc=frozenset({21, 22}),
            fill_bar="path")

print(f"{'銘柄':<9}{'群':<8}{'区分':<16}{'件数':>7}{'勝率':>7}{'平均勝':>8}"
      f"{'期待値R':>9}{'t':>7}{'R/月':>8}")
SUM = {}
for grp, gn in ((USED, "使った4"), (UNUSED, "未使用6")):
    for p in grp:
        cfg = AppConfig.load(f"configs/h1/{p}.yaml")
        cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                           cfg.backtest.holdout_start, "dev")
        pv = cfg.instrument.pip_size
        months = (cs[-1].time - cs[0].time).days / 30.44
        row = {}
        for after in (False, True):
            tr = collect_swing_trades(
                cs, **BASE, add_after_break=after,
                spread=cfg.execution.spread_pips * pv,
                slippage=cfg.execution.slippage_pips * pv)
            for kind in ("zone", "add"):
                if kind == "zone" and after:
                    continue
                sel = [t for t in tr if t.kind == kind]
                if len(sel) < 30:
                    continue
                v = np.array([t.r_multiple * t.size for t in sel])
                w = v[v > 0]
                tv = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
                rpm = len(v) / months * v.mean()
                tag = ("絞る前 " if not after else "抜けた後だけ ") + kind
                print(f"{p:<9}{gn:<8}{tag:<16}{len(v):>7,}{(v > 0).mean():>7.1%}"
                      f"{w.mean():>+8.2f}{v.mean():>+9.3f}{tv:>7.2f}{rpm:>+8.2f}",
                      flush=True)
                row[tag] = (float(v.mean()), rpm)
        SUM[p] = (gn, row)

print("\n" + "=" * 78)
print("## 合計 R/月(銘柄ごとの R/月 を足したもの)")
print("=" * 78)
print(f"{'群':<10}{'最初の建玉だけ':>16}{'+ 絞る前':>14}{'+ 抜けた後だけ':>18}")
for gn in ("使った4", "未使用6"):
    rows = [r for _, (g, r) in SUM.items() if g == gn]
    z = sum(r["絞る前 zone"][1] for r in rows if "絞る前 zone" in r)
    a0 = sum(r["絞る前 add"][1] for r in rows if "絞る前 add" in r)
    a1 = sum(r["抜けた後だけ add"][1] for r in rows if "抜けた後だけ add" in r)
    print(f"{gn:<10}{z:>+16.2f}{z + a0:>+14.2f}{z + a1:>+18.2f}")
print("\ndone", flush=True)
