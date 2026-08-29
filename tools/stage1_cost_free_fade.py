"""段 1 — コストを抜いて、抵抗帯の逆張りに普遍的な予測力があるか。

利用者の指摘:
  「スプレッドの広さで如何様にも結果が変わるものに普遍的な手法を
    求めるのは筋が悪い。まずコストを抜いて計測し、手法の普遍性に
    問題がなければコスト有りで計測する」

**コストを抜けば条件は銘柄間で同じ**(R で正規化してある)ので、
ここでだけ全銘柄を 1 つの数字にしてよい。コストを入れた段 2 は
銘柄ごとに分けて出す。

上位足も一緒に見る。利用者の指摘「コストが問題になるなら上位足を
参照して 1 回あたりのコスト比率を下げられる」。ただし過去に
ダウ転換の逆張りでは M15 +0.033 → H1 -0.003 → H4 -0.053 と
**エッジ自体が上位足で消えた**。抵抗帯の形では未測定なので、
コストと無関係に決まる話としてここで確かめる。

1 銘柄ごとに JSON へ落とす(コンテナが 10〜20 分で巻き戻るため)。

    PYTHONPATH=. python tools/stage1_cost_free_fade.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf",
         "audusd", "eurusd", "gbpjpy", "usdjpy"]
# 帯 / 構造 / 執行。**2 つずらし**は利用者の指定どおり。
LADDERS = (("H4-H1-M15", 240, 60, 15),
           ("D1-H4-H1", 1440, 240, 60))
STOPS = (1.0, 2.0, 3.5)
BASE = dict(range_bars=120, zone_entry="exec_turn", zone_wait_bars=24,
            zone_entry_max_atr=2.0, entry_signal="exec", entry_fill="level",
            entry_fallback="structure", stop_basis="band",
            stop_buffer_atr=1.5, reversal_signal="both",
            max_flips=0, max_adds=0, max_open=4,
            reverse_entry=False,          # **帯で跳ね返りに乗る = 逆張り**
            blocked_hours_utc=frozenset({21, 22}), fill_bar="path",
            spread=0.0, slippage=0.0)     # **コストは抜く**
OUT = "docs/handoff/stage1-fade.json"


def run() -> dict:
    got = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for p in PAIRS:
        if p in got:
            continue
        cfg = AppConfig.load(f"configs/h1/{p}.yaml")
        cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                           cfg.backtest.holdout_start, "dev")
        cell = {}
        for name, zm, sm, xm in LADDERS:
            for ms in STOPS:
                tr = collect_swing_trades(
                    cs, **BASE, zone_minutes=zm, structure_minutes=sm,
                    left=3, right=3, min_stop_atr=ms)
                rows = [[cs[t.entry_index].time.strftime("%Y-%m"),
                         float(t.r_multiple * t.size)]
                        for t in tr if t.kind == "zone"]
                cell[f"{name}|{ms}"] = rows
                print(f"  {p} {name} 損切り下限 {ms} → {len(rows):,} 件",
                      flush=True)
        got[p] = cell
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(got, open(OUT, "w", encoding="utf-8"))
        print(f"  {p} 保存", flush=True)
    return got


def boot(rows, draws=10_000):
    """月クラスタ。銘柄をまたぐ相関も含めて保守的に見る。"""
    g = defaultdict(list)
    for ym, r in rows:
        g[ym].append(r)
    keys = list(g)
    sums = np.array([sum(g[k]) for k in keys])
    ns = np.array([len(g[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(20260829)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    return sums[idx].sum(axis=1) / ns[idx].sum(axis=1), len(keys)


got = run()
print(f"\n{'=' * 86}")
print("## 段 1 — コスト無し・全銘柄を 1 つに(2000-2019・帯で逆張り)")
print("=" * 86)
print(f"{'足の組':<14}{'損切り下限':>11}{'件数':>9}{'勝率':>7}{'平均勝':>8}"
      f"{'期待値R':>9}{'t':>7}{'月クラスタ 95%CI':>26}")
for name, *_ in LADDERS:
    for ms in STOPS:
        rows = [r for p in PAIRS for r in got[p][f"{name}|{ms}"]]
        if len(rows) < 200:
            print(f"{name:<14}{ms:>11}{len(rows):>9}  件数不足"); continue
        v = np.array([r for _, r in rows])
        w = v[v > 0]
        t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        bs, nk = boot(rows)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        gate = "0 をまたぐ" if lo <= 0 <= hi else "**0 をまたがない**"
        print(f"{name:<14}{ms:>11.1f}{len(v):>9,}{(v>0).mean():>7.1%}"
              f"{w.mean():>+8.2f}{v.mean():>+9.4f}{t:>7.2f}"
              f"   {lo:+.4f} 〜 {hi:+.4f} {gate}")
print("\ndone", flush=True)
