"""段 1 — コストを抜いて、抵抗帯の手法に普遍的な予測力があるか。

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
import subprocess
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
# **利用者の手順どおりの形**(zone_entry="method")。跳ね返りと
# ブレイクの両方を、同じ帯で取る。reverse_entry は使わない。
BASE = dict(range_bars=120, zone_entry="method",
            zone_entry_max_atr=2.0, entry_signal="exec", entry_fill="level",
            entry_fallback="structure", stop_basis="band",
            stop_buffer_atr=1.5, reversal_signal="both",
            max_flips=0, max_adds=0, max_open=4,
            blocked_hours_utc=frozenset({21, 22}), fill_bar="path",
            spread=0.0, slippage=0.0)     # **コストは抜く**
OUT = "docs/handoff/stage1-method.json"


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
                # **生の建玉を持たない。**10 銘柄ぶんで 7 MB を超え、
                # リポジトリへ置くには重い。月クラスタの取り直しに要る
                # のは クラスタごとの 件数 と 合計 だけ。t 値のために
                # 二乗和も持つ。
                agg: dict = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
                for t in tr:
                    if t.kind != "zone":
                        continue
                    ym = cs[t.entry_index].time.strftime("%Y-%m")
                    # 帯の側と向きが揃っていれば跳ね返り、違えばブレイク
                    kind = ("跳ね返り" if (t.zone_key == "bottom") == t.long_side
                            else "ブレイク")
                    r = float(t.r_multiple * t.size)
                    c = agg[f"{ym}|{kind}"]
                    c[0] += 1; c[1] += r; c[2] += r * r
                    if r > 0:
                        c[3] += 1; c[4] += r
                rows = dict(agg)
                cell[f"{name}|{ms}"] = rows
                print(f"  {p} {name} 損切り下限 {ms} → "
                      f"{sum(v[0] for v in rows.values()):,} 件", flush=True)
        got[p] = cell
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(got, open(OUT, "w", encoding="utf-8"))
        print(f"  {p} 保存", flush=True)
    return got


def pool(cells, kind=None):
    """銘柄をまたいで、月ごとに 件数 / 合計 / 二乗和 / 勝ち を足す。"""
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
    """集計だけから 件数 / 勝率 / 平均勝 / 期待値 / t を出す。"""
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
    """月クラスタ。銘柄をまたぐ相関も含めて保守的に見る。"""
    keys = list(g)
    if not keys:
        return np.zeros(draws), 0
    sums = np.array([g[k][1] for k in keys])
    ns = np.array([g[k][0] for k in keys], dtype=float)
    rng = np.random.default_rng(20260829)
    idx = rng.integers(0, len(keys), size=(draws, len(keys)))
    tot, cnt = sums[idx].sum(axis=1), ns[idx].sum(axis=1)
    return np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0), len(keys)


got = run()
print(f"\n{'=' * 86}")
print("## 段 1 — コスト無し・全銘柄を 1 つに(2000-2019・利用者の手順)")
print("=" * 86)
print(f"{'足の組':<14}{'損切り下限':>11}{'件数':>9}{'勝率':>7}{'平均勝':>8}"
      f"{'期待値R':>9}{'t':>7}{'月クラスタ 95%CI':>26}")
for name, *_ in LADDERS:
    for ms in STOPS:
        g = pool([got[p][f"{name}|{ms}"] for p in PAIRS])
        r = summarize(g)
        if r is None or r[0] < 200:
            print(f"{name:<14}{ms:>11}  件数不足"); continue
        n, wr, aw, mean, t = r
        bs, nk = boot(g)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        gate = "0 をまたぐ" if lo <= 0 <= hi else "**0 をまたがない**"
        print(f"{name:<14}{ms:>11.1f}{n:>9,}{wr:>7.1%}"
              f"{aw:>+8.2f}{mean:>+9.4f}{t:>7.2f}"
              f"   {lo:+.4f} 〜 {hi:+.4f} {gate}")
print(f"\n{'=' * 86}")
print("## 機構ごと(跳ね返り / ブレイク)— 混ぜずに出す")
print("=" * 86)
print(f"{'足の組':<14}{'損切り下限':>11}{'機構':<8}{'件数':>9}{'勝率':>7}"
      f"{'平均勝':>8}{'期待値R':>9}{'t':>7}{'月クラスタ 95%CI':>26}")
for name, *_ in LADDERS:
    for ms in STOPS:
        cells = [got[p][f"{name}|{ms}"] for p in PAIRS]
        for kind in ("跳ね返り", "ブレイク"):
            g = pool(cells, kind)
            r = summarize(g)
            if r is None or r[0] < 200:
                print(f"{name:<14}{ms:>11.1f}{kind:<8}  件数不足"); continue
            n, wr, aw, mean, t = r
            bs, _ = boot(g)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            gate = "またぐ" if lo <= 0 <= hi else "**またがない**"
            print(f"{name:<14}{ms:>11.1f}{kind:<8}{n:>9,}"
                  f"{wr:>7.1%}{aw:>+8.2f}{mean:>+9.4f}{t:>7.2f}"
                  f"   {lo:+.4f} 〜 {hi:+.4f} {gate}")
print("\ndone", flush=True)
