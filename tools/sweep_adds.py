"""買い増しは勝ちを増やすか。回数を掃引して、機構ごとに割って出す。

利用者の問い:「トレンドが継続される限り買い増すだけなので、
むしろ勝ちが増えそうだが」。合算 1 行では答えられないので
zone(最初の指値)と add(買い増し)を分けて出す。
"""
from __future__ import annotations
import sys
import numpy as np
from llmfx.config import AppConfig
from llmfx.backtest.split import split_candles
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

USED = ["audusd", "eurusd", "gbpjpy", "usdjpy"]
UNUSED = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf"]
PAIRS = USED + UNUSED
# docs/frozen-v5.md の条件そのまま。買い増しの回数だけを動かす。
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
    D[p] = (cs, cfg.execution.spread_pips * pv, cfg.execution.slippage_pips * pv,
            (cs[-1].time - cs[0].time).days / 30.44)


def run(**kw):
    rows = []
    for p in PAIRS:
        cs, sp, sl, _ = D[p]
        rows += [(p, t) for t in collect_swing_trades(
            cs, **{**BASE, **kw}, spread=sp, slippage=sl)]
    return rows


def show(tag, rows, pairs, kind=None):
    sel = [r for r in rows if r[0] in pairs and (kind is None or r[1].kind == kind)]
    if len(sel) < 20:
        print(f"  {tag:<20}{len(sel):>7}  件数不足", flush=True); return
    v = np.array([r[1].r_multiple * r[1].size for r in sel]); w = v[v > 0]
    t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
    per = {p: np.mean([r[1].r_multiple * r[1].size for r in sel if r[0] == p])
           for p in pairs if any(r[0] == p for r in sel)}
    rpm = sum(len([r for r in sel if r[0] == p]) / D[p][3] * m
              for p, m in per.items())
    pos = sum(1 for m in per.values() if m > 0)
    print(f"  {tag:<20}{len(v):>7,}{(v>0).mean():>7.1%}{w.mean() if len(w) else 0:>+8.2f}"
          f"{v.mean():>+9.3f}{t:>7.2f}{rpm:>+8.2f}{f'{pos}/{len(pairs)}':>8}", flush=True)


HDR = (f"  {'機構':<20}{'件数':>7}{'勝率':>7}{'平均勝':>8}"
       f"{'期待値R':>9}{'t':>7}{'R/月':>8}{'プラス':>8}")
for FB in ("adverse", "path"):
  print(f"\n{'='*78}\n## 測り方: {FB}\n{'='*78}", flush=True)
  for adds in (0, 1, 2, 3):
      rows = run(max_adds=adds, fill_bar=FB)
      print(f"\n### 買い増し 最大 {adds} 回\n" + HDR, flush=True)
      for label, pairs in (("使った4", USED), ("使っていない6", UNUSED),
                           ("10 銘柄", PAIRS)):
          for k in ("zone", "add", None):
              show(f"{label} / {k or '合算'}", rows, pairs, k)
      print(flush=True)
print("done", flush=True)
