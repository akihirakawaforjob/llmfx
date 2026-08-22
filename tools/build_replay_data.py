"""UI に載せる足を切り出す。**時刻は分単位の整数にして差分で持つ。**

そのまま JSON にすると 1 銘柄 2 年で 3MB を超える。時刻を差分、価格を
最初の値からの差分(整数のティック)にすると 3 分の 1 になる。
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from llmfx.data.csv_source import load_candles_csv

OUT = {}
for name, lo, hi in (("usdjpy", "2016-01-01", "2018-12-31"),
                     ("gbpjpy", "2016-01-01", "2018-12-31")):
    a = datetime.fromisoformat(lo).replace(tzinfo=timezone.utc)
    b = datetime.fromisoformat(hi).replace(tzinfo=timezone.utc)
    cs = [c for c in load_candles_csv(f"data/{name}_m15.csv") if a <= c.time <= b]
    if not cs:
        print(f"{name}: 期間に足が無い"); continue
    tick = 0.001
    t0 = int(cs[0].time.timestamp() // 60)
    p0 = round(cs[0].open / tick)
    ts, ohlc = [], []
    pt, pp = t0, p0
    for c in cs:
        t = int(c.time.timestamp() // 60)
        ts.append(t - pt); pt = t
        for v in (c.open, c.high, c.low, c.close):
            q = round(v / tick)
            ohlc.append(q - pp); pp = q
    OUT[name] = {"t0": t0, "p0": p0, "tick": tick, "dt": ts, "dp": ohlc,
                 "n": len(cs)}
    print(f"{name}: {len(cs):,} 本  {cs[0].time:%Y-%m-%d} 〜 {cs[-1].time:%Y-%m-%d}",
          flush=True)
path = os.path.join(os.path.dirname(__file__), "uidata.json")
json.dump(OUT, open(path, "w"), separators=(",", ":"))
print(f"{os.path.getsize(path)/1e6:.2f} MB")
