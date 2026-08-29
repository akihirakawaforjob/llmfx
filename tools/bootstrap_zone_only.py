"""作法 3 — 帯の指値だけで、信頼区間が 0 をまたがないかを見る.

条件探しに使っていない 6 銘柄。開発用(2000-2019)のみ。
**検証用データには触らない。**

クラスタは 2 通りで出す:
  銘柄×月 … 同じ銘柄の同じ月の建玉が相関することだけを見る
  月      … 銘柄をまたぐ相関も見る。**こちらのほうが保守的。**
CLAUDE.md の注意どおり、通貨ペアは独立ではない(USD や JPY を共有
する)。銘柄×月だけで出すと、独立を仮定している分だけ楽観に出る。

    PYTHONPATH=. python tools/bootstrap_zone_only.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from llmfx.backtest.split import split_candles
from llmfx.config import AppConfig
from llmfx.data.csv_source import load_candles_csv
from llmfx.research.zone_swing import collect_swing_trades

UNUSED = ["gbpusd", "usdcad", "nzdusd", "audjpy", "eurjpy", "usdchf"]
BASE = dict(zone_minutes=240, structure_minutes=60, range_bars=120,
            zone_entry="exec_turn", zone_wait_bars=24, zone_entry_max_atr=2.0,
            entry_signal="exec", entry_fill="level", entry_fallback="structure",
            stop_basis="band", stop_buffer_atr=1.5, min_stop_atr=2.0,
            reversal_signal="both", max_flips=0, max_adds=0, max_open=4,
            reverse_entry=True, blocked_hours_utc=frozenset({21, 22}))
DRAWS = 10_000
RNG = np.random.default_rng(20260829)


def collect(fill_bar: str):
    """(銘柄, 年月, R) の列と、開発用の月数を返す。"""
    rows, months = [], 0.0
    for p in UNUSED:
        cfg = AppConfig.load(f"configs/h1/{p}.yaml")
        cs = split_candles(load_candles_csv(f"data/{p}_m15.csv"),
                           cfg.backtest.holdout_start, "dev")
        pv = cfg.instrument.pip_size
        months = max(months, (cs[-1].time - cs[0].time).days / 30.44)
        for t in collect_swing_trades(
                cs, **BASE, fill_bar=fill_bar,
                spread=cfg.execution.spread_pips * pv,
                slippage=cfg.execution.slippage_pips * pv):
            if t.kind != "zone":
                continue
            ym = cs[t.entry_index].time.strftime("%Y-%m")
            rows.append((p, ym, t.r_multiple * t.size))
        print(f"  {p} 読み込み済み", flush=True)
    return rows, months


def boot(rows, months, key):
    """クラスタ単位で取り直す。期待値と R/月 の両方を出す。"""
    groups = defaultdict(list)
    for pair, ym, r in rows:
        groups[key(pair, ym)].append(r)
    keys = list(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    ns = np.array([len(groups[k]) for k in keys], dtype=float)
    c = len(keys)
    idx = RNG.integers(0, c, size=(DRAWS, c))
    tot, cnt = sums[idx].sum(axis=1), ns[idx].sum(axis=1)
    return c, tot / cnt, tot / months


def report(name, rows, months):
    v = np.array([r for _, _, r in rows])
    t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
    print(f"\n### {name}   {len(v):,} 件")
    print(f"  期待値 {v.mean():+.4f} R   t={t:+.2f}   "
          f"R/月 {v.sum()/months:+.2f}   勝率 {(v>0).mean():.1%}")
    for label, key in (("銘柄×月", lambda p, m: (p, m)),
                       ("月(銘柄をまたぐ)", lambda p, m: m)):
        c, exp, rpm = boot(rows, months, key)
        lo, hi = np.percentile(exp, [2.5, 97.5])
        rlo, rhi = np.percentile(rpm, [2.5, 97.5])
        gate = "0 をまたぐ" if lo <= 0 <= hi else "**0 をまたがない**"
        print(f"  {label:<18} クラスタ {c:>5,}  "
              f"期待値 95%CI {lo:+.4f} 〜 {hi:+.4f}  {gate}")
        print(f"  {'':<18} {'':>11}  R/月   95%CI {rlo:+.2f} 〜 {rhi:+.2f}  "
              f"0 超の確率 {(rpm > 0).mean():.1%}")


def thirds(rows, months):
    """作法 2 — 期間を 3 分割しても符号が反転しないか。"""
    order = sorted(rows, key=lambda r: r[1])
    n = len(order) // 3
    print("\n  期間 3 分割")
    for i, part in enumerate(( order[:n], order[n:2*n], order[2*n:])):
        v = np.array([r for _, _, r in part])
        t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        print(f"    {part[0][1]} 〜 {part[-1][1]}  {len(v):>5,} 件  "
              f"{v.mean():+.4f} R  t={t:+.2f}")


def per_pair(rows, months):
    print("\n  銘柄ごと(混ぜない)")
    for p in UNUSED:
        v = np.array([r for q, _, r in rows if q == p])
        t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        print(f"    {p:<9}{len(v):>6,} 件  {v.mean():+.4f} R  t={t:+.2f}  "
              f"R/月 {v.sum()/months:+.2f}")


if __name__ == "__main__":
    print("作法 3 — 帯の指値だけ / 条件探しに使っていない 6 銘柄")
    print("開発用 2000-2019 のみ。**検証用データには触らない。**")
    for fb in ("path", "adverse"):
        print(f"\n{'=' * 78}\n## 測り方 {fb}\n{'=' * 78}", flush=True)
        rows, months = collect(fb)
        report(f"帯の指値だけ({fb})", rows, months)
        per_pair(rows, months)
        thirds(rows, months)
    print("\ndone", flush=True)
