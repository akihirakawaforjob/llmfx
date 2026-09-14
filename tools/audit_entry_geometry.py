"""約定が帯のどちら側にあるか、損切り幅がどれだけかを数える.

利用者の指摘(図を見て):
  「帯に対して向かう様に売りに入っている。抵抗帯の抵抗帯力を信じて
    買いに入るべき」
  「損切り幅がやたらと広く見える。僕の想定では見ている足の推進波
    一つ分と同じくらいのイメージ」

跳ね返りは **帯の手前** で入るのが仕様。帯を越えた側で入ったら、
その帯はもう抵抗ではなく支持で、向かって張っていることになる。

    PYTHONPATH=. python tools/audit_entry_geometry.py
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf",
         "audusd", "eurusd", "gbpjpy", "usdjpy"]
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="method", zone_entry_max_atr=2.0, entry_signal="exec",
            entry_fill="level", entry_fallback="structure", stop_basis="band",
            stop_buffer_atr=1.5, min_stop_atr=2.0, reversal_signal="both",
            max_flips=0, max_adds=0, max_open=4,
            blocked_hours_utc=frozenset({21, 22}), fill_bar="path",
            spread=0.0, slippage=0.0)
OUT = "docs/handoff/entry-geometry.json"


def main() -> None:
    got = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for p in PAIRS:
        if p in got:
            continue
        cfg = AppConfig.load(f"configs/h1/{p}.yaml")
        cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                           cfg.backtest.holdout_start, "dev")
        rows = []
        for t in collect_swing_trades(cs, **BASE):
            if t.kind != "zone" or t.atr <= 0:
                continue
            mech = "跳ね返り" if (t.zone_key == "bottom") == t.long_side else "ブレイク"
            # 建玉の向きに、帯を **越えた** 側にいるか
            past = ((t.entry < t.zone_price) if t.long_side
                    else (t.entry > t.zone_price))
            rows.append([mech, bool(past),
                         float(abs(t.entry - t.zone_price) / t.atr),
                         float(abs(t.entry - t.stop_at_entry) / t.atr),
                         float(t.r_multiple * t.size)])
        got[p] = rows
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(got, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        for cmd in (["git", "add", "-f", OUT],
                    ["git", "commit", "-q", "-m", f"wip: 幾何の監査 — {p}"],
                    ["git", "push", "-q", "origin", "HEAD"]):
            subprocess.run(cmd, check=False, capture_output=True)
        print(f"  {p} {len(rows):,} 件", flush=True)

    allr = [r for v in got.values() for r in v]
    print(f"\n{'=' * 82}\n## 約定は帯のどちら側か\n{'=' * 82}")
    print(f"{'機構':<9}{'位置':<12}{'件数':>8}{'割合':>8}{'帯からの距離':>13}"
          f"{'損切り幅/ATR':>14}{'期待値R':>10}")
    for mech in ("跳ね返り", "ブレイク"):
        sub = [r for r in allr if r[0] == mech]
        for past, label in ((False, "帯の手前"), (True, "帯を越えた側")):
            sel = [r for r in sub if r[1] is past]
            if not sel:
                continue
            v = np.array([r[4] for r in sel])
            print(f"{mech:<9}{label:<12}{len(sel):>8,}{len(sel)/len(sub):>8.1%}"
                  f"{np.median([r[2] for r in sel]):>13.2f}"
                  f"{np.median([r[3] for r in sel]):>14.2f}{v.mean():>+10.4f}")

    print(f"\n{'=' * 82}\n## 損切り幅の分布(ATR 倍)\n{'=' * 82}")
    print(f"{'機構':<9}{'件数':>8}{'下位1割':>9}{'中央':>8}{'上位1割':>9}"
          f"{'最大':>9}{'4 ATR 超':>10}")
    for mech in ("跳ね返り", "ブレイク"):
        w = np.array([r[3] for r in allr if r[0] == mech])
        print(f"{mech:<9}{len(w):>8,}{np.percentile(w,10):>9.2f}"
              f"{np.median(w):>8.2f}{np.percentile(w,90):>9.2f}"
              f"{w.max():>9.2f}{(w > 4).mean():>10.1%}")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
