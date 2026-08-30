"""負けパターンを数え、上位 3 つの実例を切り出す.

利用者の求め:「それぞれの負けパターンの中で、上位 3 つの多いパターンを
図で見たい。またその負けパターンで見ていた時間足も知りたい」。

分け方は `llmfx/backtest/inspect.py` と同じ考え方:
  **負けを R の大小で割っても何も見えない**(損切りが効くので全部 -1.0R)。
  順行したかどうかで割る。打つ手が真逆になるため。

    順行せず   MFE < 0.25R  → 入る場所の問題
    少し順行   0.25〜1.0R   → 決済の置き方
    大きく順行 1.0R 以上    → 持ち方の問題

これに出口(損切り / 転換 / 終端)を掛ける。

    PYTHONPATH=. python tools/collect_loss_patterns.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf",
         "audusd", "eurusd", "gbpjpy", "usdjpy"]
LADDERS = (("H4-H1-M15", 240, 60), ("D1-H4-H1", 1440, 240))
BASE = dict(range_bars=120, zone_entry="method", zone_entry_max_atr=2.0,
            entry_signal="exec", entry_fill="level",
            entry_fallback="structure", stop_basis="band",
            stop_buffer_atr=1.5, min_stop_atr=2.0, reversal_signal="both",
            max_flips=0, max_adds=0, max_open=4,
            blocked_hours_utc=frozenset({21, 22}), fill_bar="path",
            spread=0.0, slippage=0.0)
OUT = "docs/handoff/loss-patterns.json"
WHY = {"stop": "損切り", "reversal": "転換で手仕舞い", "end": "終端"}


def bucket(mfe: float) -> str:
    if mfe < 0.25:
        return "順行せず"
    if mfe < 1.0:
        return "少し順行"
    return "大きく順行"


def main() -> None:
    counts: dict[str, Counter] = {}
    picks: dict[str, dict] = {}
    for name, zm, sm in LADDERS:
        c: Counter = Counter()
        best: dict = {}
        for p in PAIRS:
            cfg = AppConfig.load(f"configs/h1/{p}.yaml")
            cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                               cfg.backtest.holdout_start, "dev")
            for t in collect_swing_trades(cs, **BASE, zone_minutes=zm,
                                          structure_minutes=sm):
                if t.kind != "zone" or t.r_multiple > 0:
                    continue
                mech = ("跳ね返り" if (t.zone_key == "bottom") == t.long_side
                        else "ブレイク")
                key = f"{mech}|{bucket(t.max_favourable_r)}|{WHY.get(t.why, t.why)}"
                c[key] += 1
                # **代表例は「その区分の真ん中」を選ぶ。**いちばん派手な
                # ものを選ぶと、その区分の典型ではなくなる。
                score = abs(t.max_favourable_r - {"順行せず": 0.1,
                                                  "少し順行": 0.6,
                                                  "大きく順行": 1.6}[
                    bucket(t.max_favourable_r)])
                cur = best.get(key)
                if cur is None or score < cur["score"]:
                    lo = max(0, t.entry_index - 45)
                    hi = min(len(cs) - 1, t.exit_index + 12)
                    best[key] = dict(
                        score=score, pair=p, ladder=name,
                        when=cs[t.entry_index].time.strftime("%Y-%m-%d %H:%M"),
                        long_side=t.long_side, zone_key=t.zone_key,
                        band=t.zone_price, entry=t.entry, stop=t.stop_at_entry,
                        exit=t.exit, r=t.r_multiple, mfe=t.max_favourable_r,
                        mae=t.max_adverse_r, bars=t.bars_held, why=t.why,
                        atr=t.atr,
                        entry_at=t.entry_index - lo, exit_at=t.exit_index - lo,
                        candles=[dict(o=round(x.open, 5), h=round(x.high, 5),
                                      l=round(x.low, 5), c=round(x.close, 5),
                                      t="") for x in cs[lo:hi + 1]])
            print(f"  {name} {p} 済み", flush=True)
        counts[name] = c
        picks[name] = best
    out = {"counts": {k: dict(v) for k, v in counts.items()},
           "picks": {k: {kk: vv for kk, vv in v.items()} for k, v in picks.items()}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

    for name, c in counts.items():
        tot = sum(c.values())
        print(f"\n{'=' * 78}\n## {name}   負け {tot:,} 件\n{'=' * 78}")
        for mech in ("跳ね返り", "ブレイク"):
            sub = {k: v for k, v in c.items() if k.startswith(mech)}
            st = sum(sub.values())
            print(f"\n  {mech}  {st:,} 件")
            for k, v in sorted(sub.items(), key=lambda z: -z[1])[:5]:
                _, b, w = k.split("|")
                print(f"    {b:<10}{w:<16}{v:>7,}  {v/st:>6.1%}")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
