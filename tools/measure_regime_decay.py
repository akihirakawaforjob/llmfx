"""なぜ 2000-2007 にしかエッジが無いのか。

利用者の問い:「値動きを見てコードを合わせたわけではないのに、
期間でここまで差が出るのは何故か」。

手法のせいか、相場のせいかを分けるために 3 種類を並べる:

  A 手法の成績        コストあり / コスト無し / コストが R に占める比率
  B 相場の素の性質    ブレイク後 5 日の追随(手法と無関係に測れる)
  C 値幅              ATR の水準。コスト比率の分母

**コスト無しでも同じ形に減衰するなら、相場が変わったということ。**
コストを外すと平らになるなら、値幅が縮んでスプレッド比率が
上がっただけということ。銘柄は混ぜない。

    PYTHONPATH=. python tools/measure_regime_decay.py
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
from llmfx.research.zone_swing import collect_swing_trades

PAIRS = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf"]
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="exec_turn", zone_wait_bars=24, zone_entry_max_atr=2.0,
            entry_signal="exec", entry_fill="level", entry_fallback="structure",
            stop_basis="band", stop_buffer_atr=1.5, min_stop_atr=2.0,
            reversal_signal="both", max_flips=0, max_adds=0, max_open=4,
            reverse_entry=True, blocked_hours_utc=frozenset({21, 22}),
            fill_bar="path")
ERAS = (("2000-2006", "2000", "2006"), ("2007-2013", "2007", "2013"),
        ("2014-2019", "2014", "2019"))
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


def era_of(year: str) -> str | None:
    for name, lo, hi in ERAS:
        if lo <= year <= hi:
            return name
    return None


# **1 銘柄ごとに書き出す。**このコンテナは 10 分前後で巻き戻ることが
# あり、その都度、走らせていた処理ごと消える(実際に 12 回消えた)。
# 途中結果をリポジトリへ落として、再実行で続きから拾えるようにする。
OUT = "docs/handoff/regime-decay.json"

print("なぜ 2000-2007 にしかエッジが無いのか — 手法 / 相場 / 値幅")
print("開発用 2000-2019 のみ。**検証用データには触らない。**\n")

strat = defaultdict(lambda: defaultdict(list))   # [銘柄][(期, 種)] -> [(R, リスク pips)]
market = defaultdict(lambda: defaultdict(list))  # [銘柄][期] -> [追随 ATR 倍]
vol = defaultdict(lambda: defaultdict(list))     # [銘柄][期] -> [ATR pips]

done_pairs: dict = {}
if os.path.exists(OUT):
    done_pairs = json.load(open(OUT, encoding="utf-8"))
    print(f"  済み: {', '.join(done_pairs)}", flush=True)

for p in PAIRS:
    if p in done_pairs:
        continue
    cfg = AppConfig.load(f"configs/h1/{p}.yaml")
    cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                       cfg.backtest.holdout_start, "dev")
    pv = cfg.instrument.pip_size
    sp, sl = cfg.execution.spread_pips * pv, cfg.execution.slippage_pips * pv
    for tag, spread, slip in (("cost", sp, sl), ("free", 0.0, 0.0)):
        for t in collect_swing_trades(cs, **BASE, spread=spread, slippage=slip):
            if t.kind != "zone":
                continue
            era = era_of(cs[t.entry_index].time.strftime("%Y"))
            if era:
                strat[p][(era, tag)].append((t.r_multiple, t.risk / pv))
    # --- 相場の素の性質(手法と無関係) ---
    h4 = resample_candles(cs, 240)
    a = atr_series(h4)
    hi = np.array([x.high for x in h4]); lo = np.array([x.low for x in h4])
    cl = np.array([x.close for x in h4])
    for i in range(max(LOOK, ATR_N) + 1, len(h4) - FWD):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        era = era_of(h4[i].time.strftime("%Y"))
        if era is None:
            continue
        up = cl[i] > hi[i - LOOK:i].max()
        dn = cl[i] < lo[i - LOOK:i].min()
        vol[p][era].append(a[i] / pv)
        if up == dn:
            continue
        s = 1.0 if up else -1.0
        market[p][era].append((cl[i + FWD] - cl[i]) * s / a[i])
    done_pairs[p] = {
        "strat": {f"{e}|{t}": [[float(r), float(k)] for r, k in v]
                  for (e, t), v in strat[p].items()},
        "market": {e: [float(x) for x in v] for e, v in market[p].items()},
        "vol": {e: float(np.median(v)) for e, v in vol[p].items() if v},
        "spread_pips": float(cfg.execution.spread_pips),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(done_pairs, open(OUT, "w", encoding="utf-8"))
    print(f"  {p} 済み → {OUT} へ保存", flush=True)

# 保存済みを読み戻す(再実行で続きから拾った分を含む)
for p, d in done_pairs.items():
    for k, v in d["strat"].items():
        e, t = k.split("|")
        strat[p][(e, t)] = [(r, kk) for r, kk in v]
    for e, v in d["market"].items():
        market[p][e] = v
    for e, v in d["vol"].items():
        vol[p][e] = [v]

print(f"\n{'=' * 96}")
print("## 銘柄ごと(混ぜない)")
print("=" * 96)
print(f"{'銘柄':<9}{'期':<11}{'件数':>7}{'期待値':>9}{'t':>7}"
      f"{'コスト無し':>11}{'コスト/R':>10}{'ATR(pips)':>11}"
      f"{'走った幅':>10}{'戻された幅':>11}{'差':>8}")
for p in PAIRS:
    for era, _, _ in ERAS:
        c = strat[p][(era, "cost")]
        f = strat[p][(era, "free")]
        if len(c) < 30:
            continue
        vc = np.array([r for r, _ in c]); vf = np.array([r for r, _ in f])
        risk = np.median([k for _, k in c])
        cost_r = done_pairs[p]["spread_pips"] * 2 / risk
        t = float(vc.mean() / (vc.std(ddof=1) / np.sqrt(len(vc))))
        m = np.array(market[p][era])
        go, back = m[m > 0], m[m < 0]
        print(f"{p:<9}{era:<11}{len(vc):>7,}{vc.mean():>+9.3f}{t:>7.2f}"
              f"{vf.mean():>+11.3f}{cost_r:>10.3f}"
              f"{np.median(vol[p][era]):>11.1f}"
              f"{go.mean():>10.2f}{-back.mean():>11.2f}"
              f"{go.mean() + back.mean():>+8.2f}")

print(f"\n{'=' * 96}")
print("## 6 銘柄まとめ(期ごと)")
print("=" * 96)
print(f"{'期':<11}{'件数':>7}{'期待値':>9}{'t':>7}{'コスト無し':>11}"
      f"{'上位1割の寄与':>15}{'走った幅':>10}{'戻された幅':>11}{'差':>8}")
for era, _, _ in ERAS:
    vc = np.concatenate([[r for r, _ in strat[p][(era, "cost")]] for p in PAIRS])
    vf = np.concatenate([[r for r, _ in strat[p][(era, "free")]] for p in PAIRS])
    m = np.concatenate([market[p][era] for p in PAIRS])
    k = max(1, int(len(vc) * 0.1))
    top = np.sort(vc)[-k:].sum()
    t = float(vc.mean() / (vc.std(ddof=1) / np.sqrt(len(vc))))
    go, back = m[m > 0], m[m < 0]
    print(f"{era:<11}{len(vc):>7,}{vc.mean():>+9.3f}{t:>7.2f}{vf.mean():>+11.3f}"
          f"{top / vc.sum() if vc.sum() else float('nan'):>15.0%}"
          f"{go.mean():>10.2f}{-back.mean():>11.2f}"
          f"{go.mean() + back.mean():>+8.2f}")
print("\ndone", flush=True)
