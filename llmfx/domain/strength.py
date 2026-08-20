"""通貨の強弱指数 — ダウ理論の「相互確認」を通貨で作る.

ダウは工業株平均と鉄道株平均の **両方** が同じ方向を示すことを条件に
していた。片方だけのシグナルは「未確認」として捨てる。この確認の柱を、
ここまでの実装は一度も持っていなかった。

通貨には自然な形がある。USD/JPY が下降転換したとき、それは

  ドルが全面的に弱い    -> 他のドルストレートでも同じことが起きている
  円が全面的に強い      -> 他の円クロスでも同じことが起きている
  どちらでもない        -> この 2 通貨だけの綱引き(= 未確認)

の 3 通りがある。複数のペアから通貨ごとの強弱を出せば、これを分離できる。

強弱の定義は素直に:

    通貨 C の強さ = C を含む全ペアの、直近 N 本の対数リターンの平均
                    (C が基軸通貨なら +、決済通貨なら符号を反転)

先読みを避けるため、参照するのは確定した終値だけ。`update()` に渡した
足までの情報しか使わない。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


def split_pair(pair: str) -> tuple[str, str]:
    """USDJPY / USD_JPY / usdjpy を ("USD", "JPY") にする。"""
    cleaned = pair.replace("_", "").replace("/", "").upper()
    if len(cleaned) != 6:
        raise ValueError(f"通貨ペアとして解釈できません: {pair}")
    return cleaned[:3], cleaned[3:]


@dataclass
class CurrencyStrength:
    """複数ペアの終値から、通貨ごとの強弱を逐次計算する。

    `lookback` 本前と比べた対数リターンを使う。本数が足りない間は
    `ready` が False のままで、`strength()` は空を返す。
    """

    pairs: list[str]
    lookback: int = 24

    _history: dict[str, deque] = field(default_factory=dict, init=False)
    _latest: dict[str, float] = field(default_factory=dict, init=False)
    _time: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback は 1 以上である必要があります")
        self._members: dict[str, list[tuple[str, int]]] = {}
        for pair in self.pairs:
            base, quote = split_pair(pair)
            self._history[pair] = deque(maxlen=self.lookback + 1)
            self._members.setdefault(base, []).append((pair, +1))
            self._members.setdefault(quote, []).append((pair, -1))

    @property
    def currencies(self) -> list[str]:
        return sorted(self._members)

    @property
    def time(self) -> datetime | None:
        return self._time

    def update(self, time: datetime, closes: dict[str, float]) -> None:
        """同じ時刻の終値をまとめて渡す。欠けているペアは据え置く。

        銘柄ごとに歯抜けの時間帯があるため、全ペアが毎回そろうとは限らない。
        そろわないペアは前回の値のまま持ち越し、リターンを 0 とみなす。
        """
        self._time = time
        for pair in self.pairs:
            price = closes.get(pair)
            if price is None or price <= 0:
                price = self._latest.get(pair)
                if price is None:
                    continue
            self._latest[pair] = price
            self._history[pair].append(price)

    @property
    def ready(self) -> bool:
        return any(len(h) > self.lookback for h in self._history.values())

    def _pair_return(self, pair: str) -> float | None:
        history = self._history[pair]
        if len(history) <= self.lookback:
            return None
        now, past = history[-1], history[-1 - self.lookback]
        if now <= 0 or past <= 0:
            return None
        return math.log(now / past)

    def strength(self) -> dict[str, float]:
        """通貨ごとの強さ。値が大きいほど強い。"""
        returns = {p: self._pair_return(p) for p in self.pairs}
        out: dict[str, float] = {}
        for currency, members in self._members.items():
            values = [sign * returns[p] for p, sign in members if returns[p] is not None]
            if values:
                out[currency] = sum(values) / len(values)
        return out

    def ranking(self) -> list[str]:
        """強い順に並べた通貨。"""
        scores = self.strength()
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def confirms(self, pair: str, long: bool, top: int = 3) -> bool:
        """そのペアのその方向を、通貨の強弱が確認しているか。

        買いなら「基軸通貨が上位 `top` 位以内、かつ決済通貨が下位 `top` 位以内」。
        ダウの言う相互確認にあたる。片方だけ強い(弱い)場合は未確認とみなす。
        """
        order = self.ranking()
        if len(order) < 2 * top:
            return False
        base, quote = split_pair(pair)
        if base not in order or quote not in order:
            return False
        strong, weak = set(order[:top]), set(order[-top:])
        if long:
            return base in strong and quote in weak
        return base in weak and quote in strong

    def spread(self, pair: str) -> float | None:
        """基軸通貨と決済通貨の強さの差。正なら基軸のほうが強い。"""
        scores = self.strength()
        base, quote = split_pair(pair)
        if base not in scores or quote not in scores:
            return None
        return scores[base] - scores[quote]
