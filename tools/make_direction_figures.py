"""「帯に来た時点では方向を決めない」の図で使うローソク足を作る.

利用者の仕様(CLAUDE.md):
  3. 帯が何を示したかに従う — 弾かれたら跳ね返りに乗り、抜けたら
     抜けた側に乗る。**帯に来た時点では方向を決めない**

いまの実装は `reverse_entry` で帯に触れる前に向きを固定している。
その差を図にする。ATR = 0.10。

    python tools/make_direction_figures.py docs/handoff/direction-figures.json
"""

from __future__ import annotations

import json
import sys

from make_spec_figures import series

ATR = 0.10


def build() -> dict:
    fig: dict = {}

    # --- 弾かれた場合。帯へ触れて、執行の足が下へ折り返す ---------------
    fig["reject"] = dict(
        candles=series([(0, 99.55), (5, 99.82), (9, 99.74), (13, 100.02),
                        (17, 99.88), (20, 99.96), (26, 99.62), (32, 99.40)],
                       wiggle=0.15),
        band=100.00, atr=ATR,
        swings=[(13, 100.02, "執行 高値1", "high"),
                (17, 99.88, "執行 安値1", "low"),
                (20, 99.96, "執行 高値2", "high")],
        touch_at=13,
        sell_line=99.88,      # 安値1 を下抜けたら売り
        sell_at=22,
        buy_line=100.02,      # 高値1 を上抜けたら買い(ここでは来ない)
        stop=100.17,          # 帯から 1.5 ATR 外
    )

    # --- 抜けた場合。帯へ触れて、執行の足が上へ折り返す -----------------
    fig["breakout"] = dict(
        candles=series([(0, 99.55), (5, 99.82), (9, 99.74), (13, 100.02),
                        (17, 99.90), (21, 100.06), (27, 100.42), (32, 100.66)],
                       wiggle=0.15),
        band=100.00, atr=ATR,
        swings=[(13, 100.02, "執行 高値1", "high"),
                (17, 99.90, "執行 安値1", "low")],
        touch_at=13,
        buy_line=100.02,      # 高値1 を上抜けたら買い
        buy_at=20,
        sell_line=99.90,      # 安値1 を下抜けたら売り(ここでは来ない)
        stop=99.75,           # 帯から 1.5 ATR 外(下側)
    )
    return fig


if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else "docs/handoff/direction-figures.json"
    fig = build()
    json.dump(fig, open(dst, "w"), ensure_ascii=False)
    for k, v in fig.items():
        cs = v["candles"]
        print(f"{k:<10}{len(cs):>4} 本  {min(c['l'] for c in cs):.2f}"
              f" 〜 {max(c['h'] for c in cs):.2f}")
