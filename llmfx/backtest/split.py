"""開発用と検証用のデータ分割.

なぜ必要か:
    ここまで約 90 通りの設定を試している。この規模で探せば、偶然プラスに
    なるセルは必ず出る。実際 BTC で 3 つ引いて 3 つとも崩れた。

    探索を続ける限り、見つけたものが本物か偶然かは判別できない。
    唯一の解決は、**採用を決めるまで一度も見ていないデータ**で確かめること。

守り方:
    口約束では守れないので仕組みで縛る。`--split` の既定を dev にして、
    検証用を見るには明示的に holdout / all と打たせる。
    「うっかり全期間を見てしまった」を起こしにくくするのが狙い。

使い方:
    1. 開発用(dev)だけで好きなだけ探索する
    2. 「これで行く」と決めた **1 つ** を holdout にかける
    3. 落ちたら諦める。holdout を見た後に条件を弄り直したら、
       それはもう holdout ではない
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.types import Candle

SPLITS = ("dev", "holdout", "all")


class SplitError(ValueError):
    pass


def parse_boundary(text: str) -> datetime:
    """"2025-01-01" のような境界日を UTC の datetime へ。"""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SplitError(
            f"境界日は YYYY-MM-DD 形式で指定してください: {text!r}"
        ) from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def split_candles(
    candles: list[Candle], boundary: str | None, which: str
) -> list[Candle]:
    """`which` に応じて期間を切る。

    boundary が None なら分割そのものが設定されていないので全期間を返す。
    """
    if which not in SPLITS:
        raise SplitError(f"--split は {' / '.join(SPLITS)} のいずれかです: {which!r}")
    if boundary is None or which == "all":
        return candles

    cut = parse_boundary(boundary)
    if which == "dev":
        return [c for c in candles if c.time < cut]
    return [c for c in candles if c.time >= cut]


def describe(candles: list[Candle], boundary: str | None, which: str) -> str:
    """レポートやログに出す一行。どの期間を見ているかを常に明示する。"""
    if boundary is None:
        return f"分割なし(全期間) / {len(candles):,} 本"
    label = {"dev": "開発用", "holdout": "検証用", "all": "全期間"}[which]
    span = (
        f"{candles[0].time:%Y-%m-%d} 〜 {candles[-1].time:%Y-%m-%d}"
        if candles
        else "該当なし"
    )
    return f"{label}(境界 {boundary}) / {len(candles):,} 本 / {span}"
