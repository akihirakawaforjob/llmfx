"""損切り位置の遷移の図で使うローソク足を作る.

利用者の問い:「買い増しと帯でのエントリーの損切り位置の遷移を図で
確認したい」。ATR = 0.10 の買い建玉で、帯の指値 → 1 つ前への引き上げ
→ 買い増し(同じ線を共有)→ 下限に届かない場合、の 4 場面を作る。

スクラッチパッドへ置くとコンテナが巻き戻るたびに消える(実際に 9 回
消えた)。**リポジトリに入れておくこと。**

    python tools/make_stop_figures.py docs/handoff/stop-figures.json
"""

from __future__ import annotations

import json
import sys

from make_spec_figures import series

ATR = 0.10


def build() -> dict:
    fig: dict = {}

    # --- 1. 帯で入った直後。損切りは帯の外 1.5 ATR、ただし約定値から
    #        2.0 ATR は必ず空ける(下限のほうが遠いので下限が採られる)。
    fig["place"] = dict(
        candles=series([(0, 99.55), (4, 99.72), (7, 99.63), (11, 100.00),
                        (15, 100.14), (19, 100.06), (24, 100.30)], wiggle=0.16),
        band=100.00, atr=ATR,
        entry=100.00, entry_at=11,
        band_stop=99.85,        # 帯から 1.5 ATR
        floor_stop=99.80,       # 約定値から 2.0 ATR(こちらが遠いので採用)
    )

    # --- 2. 1 つ前の安値へ引き上げる。最新ではない。
    fig["trail"] = dict(
        candles=series([(0, 100.00), (5, 100.30), (9, 100.12), (14, 100.52),
                        (19, 100.34), (25, 100.78), (30, 100.60),
                        (36, 101.10)], wiggle=0.15),
        band=100.00, atr=ATR, entry=100.00, entry_at=0,
        first_stop=99.80,
        swings=[(5, 100.30, "高値1", "high"), (9, 100.12, "安値1", "low"),
                (14, 100.52, "高値2", "high"), (19, 100.34, "安値2", "low"),
                (25, 100.78, "高値3", "high"), (30, 100.60, "安値3", "low")],
        stop_after_low2=100.12,   # 安値2 が出たら **安値1** へ
        stop_after_low3=100.34,   # 安値3 が出たら **安値2** へ
        move_at=19, move2_at=30,
    )

    # --- 3. 買い増しは同じ線を共有する。距離だけが伸びる。
    fig["share"] = dict(
        candles=series([(0, 100.00), (5, 100.30), (9, 100.12), (14, 100.52),
                        (19, 100.34), (23, 100.52), (30, 101.05)], wiggle=0.15),
        atr=ATR,
        zone_entry=100.00, zone_at=0,
        add_entry=100.52, add_at=23,
        stop=100.12,             # 引き上げ後。**両方ともこの線**
        risk_zone=0.20,          # 約定時点の幅(帯で入ったときの 2.0 ATR)
        risk_add=0.40,           # 100.52 - 100.12 = 4.0 ATR
    )

    # --- 4. 下限に届かない場合。旧は存在しない線で R を割っていた。
    fig["floorbug"] = dict(
        candles=series([(0, 100.20), (4, 100.46), (8, 100.36), (12, 100.52),
                        (16, 100.44), (21, 100.58), (27, 100.40)], wiggle=0.13),
        atr=ATR,
        add_entry=100.52, add_at=12,
        real_stop=100.40,        # 建玉の損切り。**決済は必ずここで起きる**
        ghost_stop=100.32,       # 旧コードが R の分母に使っていた線
        exit_at=27,
    )
    return fig


if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else "docs/handoff/stop-figures.json"
    fig = build()
    json.dump(fig, open(dst, "w"), ensure_ascii=False)
    for k, v in fig.items():
        cs = v["candles"]
        print(f"{k:<10}{len(cs):>4} 本  {min(c['l'] for c in cs):.2f}"
              f" 〜 {max(c['h'] for c in cs):.2f}")
