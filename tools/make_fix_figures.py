"""帯の手前でしか入らない / 推進波の起点はどこか、の模式図を作る.

利用者の指摘(2026-09-14、負けの図を見て):
  「帯に対して向かう様に売りに入っている。抵抗帯の抵抗帯力を信じて
    買いに入るべき」「買い側でも発動しない様に」
  「機関投資家の守りたいポイントは、大きいローソク足の付け根部分な
    はずだし、そこに指値を置いた方が良い」

    python tools/make_fix_figures.py docs/handoff/fix-figures.json
"""

from __future__ import annotations

import json
import sys

from make_spec_figures import series

ATR = 0.10


def build() -> dict:
    fig: dict = {}

    # --- 直し方(売り)。帯を上抜けても、注文は帯の手前に留める ---------
    fig["near_sell"] = dict(
        candles=series([(0, 99.60), (6, 99.86), (10, 99.78), (15, 100.18),
                        (19, 100.06), (24, 100.22), (30, 99.72), (36, 99.44)],
                       wiggle=0.14),
        band=100.00, atr=ATR,
        keep=99.90,        # 帯の手前に留めた注文(これが正)
        drift=100.06,      # いまの実装が置き直してしまう水準(誤)
        arm_at=12, fill_at=28,
        stop=100.15,
    )

    # --- 直し方(買い)。帯を下抜けても、注文は帯の手前に留める ---------
    fig["near_buy"] = dict(
        candles=series([(0, 100.40), (6, 100.14), (10, 100.22), (15, 99.82),
                        (19, 99.94), (24, 99.78), (30, 100.28), (36, 100.56)],
                       wiggle=0.14),
        band=100.00, atr=ATR,
        keep=100.10,       # 帯の手前に留めた注文(これが正)
        drift=99.94,       # いまの実装が置き直してしまう水準(誤)
        arm_at=12, fill_at=28,
        stop=99.85,
    )

    # --- 起点の候補。抜けた後の推進波を 1 本作り、3 通りの線を引く -------
    fig["origin"] = dict(
        candles=series([(0, 99.86), (4, 100.04), (8, 99.96), (12, 100.12),
                        (16, 100.60), (20, 100.52), (25, 100.88), (30, 100.70),
                        (38, 101.20)], wiggle=0.12),
        band=100.00, atr=ATR,
        big_at=14,           # 大きく動いた足
        a_swing=100.52,      # 候補 A 直近の確定した押し安値(構造足)
        b_base=100.12,       # 候補 B 大きな足の付け根
        c_band=100.00,       # 候補 C 抜けた帯そのもの(リテスト)
        now=101.20,
        swings=[(8, 99.96, "安値0", "low"), (12, 100.12, "付け根", "low"),
                (16, 100.60, "高値1", "high"), (20, 100.52, "安値1", "low"),
                (25, 100.88, "高値2", "high"), (30, 100.70, "安値2", "low")],
    )
    return fig


if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else "docs/handoff/fix-figures.json"
    fig = build()
    json.dump(fig, open(dst, "w"), ensure_ascii=False)
    for k, v in fig.items():
        cs = v["candles"]
        print(f"{k:<10}{len(cs):>4} 本  {min(c['l'] for c in cs):.2f}"
              f" 〜 {max(c['h'] for c in cs):.2f}")
