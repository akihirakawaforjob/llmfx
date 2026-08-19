"""押し目待ちの状態機械.

利用者と図で合意した手順:

  1. 上位足が抵抗線を終値で上抜ける      → 待機状態に入る(まだ入らない)
  2. 押し目を待つ                        → 深さは問わない
  3. 下位足でダウ転換したら買う          → 調整波は下位足では下降トレンドに
                                            なっている。その直近スイング高値を
                                            上抜けた瞬間が推進波への復帰
  4. 損切りは押し安値の少し下

「上位足で方向を決め、下位足でタイミングを取る」という二段構え。
上抜けた足で飛び乗るより、押し安値が近いぶん損切りまでの幅が小さい。

打ち切りの条件は 2 つ:
  - 待ちすぎ        `max_bars` 本待って転換が来なければ見送る(追いかけない)
  - 深すぎる押し目  上抜け前の水準を割ったらダマシとみなして解除

このモジュールは状態の管理だけを持ち、シグナルの組み立ては行わない。
先読みを避けるため、参照するのは「その時点までに確定した足」のみ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Candle, Side


@dataclass
class PendingSetup:
    """上位足が上抜けてから、下位足の転換を待っている状態."""

    side: Side
    """狙う向き。上位足の転換方向。"""
    break_level: float
    """上位足が上抜けた(下抜けた)水準。無効化の判定に使う。"""
    armed_index: int
    """待機状態に入った下位足の本数目。待ちすぎの判定に使う。"""
    extreme: float
    """待機してから付けた極値。買い狙いなら最安値。これが損切りの基準。"""
    bars_waited: int = 0

    def observe(self, candle: Candle) -> None:
        """下位足を 1 本見て、押し安値(戻り高値)を更新する。"""
        self.bars_waited += 1
        if self.side is Side.LONG:
            self.extreme = min(self.extreme, candle.low)
        else:
            self.extreme = max(self.extreme, candle.high)

    def expired(self, max_bars: int) -> bool:
        return max_bars > 0 and self.bars_waited > max_bars

    def invalidated(self, candle: Candle, tolerance: float) -> bool:
        """上抜け前の水準を割ったか。

        上抜けた直後に水準へ戻る動き(リテスト)は正常なので、`tolerance` の
        余裕を持たせる。ここを 0 にすると、ほぼすべての押し目が無効になる。
        """
        if self.side is Side.LONG:
            return candle.close < self.break_level - tolerance
        return candle.close > self.break_level + tolerance


@dataclass
class PullbackTracker:
    """待機状態をひとつだけ持つ。上位足の新しい転換で上書きする。"""

    pending: PendingSetup | None = field(default=None)

    def arm(self, side: Side, break_level: float, candle: Candle, index: int) -> None:
        """上位足が上抜けた。ここから押し目を待つ。"""
        self.pending = PendingSetup(
            side=side,
            break_level=break_level,
            armed_index=index,
            extreme=candle.low if side is Side.LONG else candle.high,
        )

    def cancel(self) -> None:
        self.pending = None

    def observe(
        self, candle: Candle, max_bars: int, tolerance: float
    ) -> str | None:
        """1 本進める。打ち切った場合はその理由を返す。"""
        if self.pending is None:
            return None
        self.pending.observe(candle)
        if self.pending.invalidated(candle, tolerance):
            self.cancel()
            return "pullback_invalidated"
        if self.pending.expired(max_bars):
            self.cancel()
            return "pullback_timed_out"
        return None

    def matches(self, side: Side) -> bool:
        """下位足の転換が、待っている向きと一致するか。"""
        return self.pending is not None and self.pending.side is side
