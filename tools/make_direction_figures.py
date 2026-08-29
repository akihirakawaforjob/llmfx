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


def build_v2() -> dict:
    """利用者の説明(2026-08-29)そのままの形。

    跳ね返り側 — 帯へ触れる直前に、**現在値の少し下**へ注文を置く。
    下に来ないと発動しないので、そのまま抜けたら発動しない。取り消す。

    ブレイク側 — 抜けた後、高値切り上げ・安値切り上げが見えてから、
    **次の**切り上げの押し目で、2 つ下の足のダウ転換で入る。
    """
    fig: dict = {}

    # 跳ね返った場合。注文は帯の少し下。触れて折り返して発動する。
    fig["v2_bounce"] = dict(
        candles=series([(0, 99.58), (6, 99.80), (10, 99.72), (15, 100.02),
                        (19, 99.80), (24, 99.90), (31, 99.44), (36, 99.22)],
                       wiggle=0.14),
        band=100.00, atr=ATR,
        arm_at=13,             # 帯へ届く直前。ここで注文を置く
        order=99.88,           # **帯の少し下**(0.12 = 1.2 ATR ではなく 0.12 価格)
        fill_at=18,
        stop=100.15,           # 帯の外 1.5 ATR
    )

    # 抜けた場合。同じ注文が **発動しない**。取り消す。
    fig["v2_break"] = dict(
        candles=series([(0, 99.58), (6, 99.80), (10, 99.72), (15, 100.02),
                        (20, 100.24), (26, 100.16), (32, 100.52)], wiggle=0.14),
        band=100.00, atr=ATR,
        arm_at=13,
        order=99.88,
        cancel_at=20,
    )

    # ブレイク側の入り方。抜けた後、構造ができてから押し目で入る。
    fig["v2_pullback"] = dict(
        candles=series([(0, 99.86), (5, 100.16), (9, 100.04), (14, 100.38),
                        (19, 100.22), (24, 100.60), (29, 100.44),
                        (36, 101.00)], wiggle=0.14),
        band=100.00, atr=ATR,
        swings=[(5, 100.16, "高値1", "high"), (9, 100.04, "安値1", "low"),
                (14, 100.38, "高値2", "high"), (19, 100.22, "安値2", "low"),
                (24, 100.60, "高値3", "high"), (29, 100.44, "安値3", "low")],
        confirmed_at=19,       # 高値切り上げ + 安値切り上げ が見えた
        entry=100.52,          # **次の**押し目で、2 つ下の足のダウ転換
        entry_at=31,
        stop=100.22,           # いつもの = 1 つ前の安値(安値2)
    )
    return fig


if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else "docs/handoff/direction-figures.json"
    fig = build()
    fig.update(build_v2())
    json.dump(fig, open(dst, "w"), ensure_ascii=False)
    for k, v in fig.items():
        cs = v["candles"]
        print(f"{k:<10}{len(cs):>4} 本  {min(c['l'] for c in cs):.2f}"
              f" 〜 {max(c['h'] for c in cs):.2f}")
