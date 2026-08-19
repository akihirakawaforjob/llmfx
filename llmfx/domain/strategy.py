"""ダウ転換ストラテジー本体.

要件との対応:
  1. ダウ転換でエントリー          -> DowAnalyzer が転換イベントを生成
  2. 損切りは転換前の最高値/最安値  -> ReversalEvent.stop_basis + ATR バッファ
  3. RR が 1/2 を上回るときのみ    -> reward / risk >= entry.min_rr (既定 2.0)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from .dow import DowAnalyzer, ReversalEvent
from .mtf import HigherTimeframeFilter, granularity_minutes
from .swings import SwingDetector
from .targets import resolve_target
from .types import Candle, Side, Signal, Trend


@dataclass(frozen=True)
class RejectedSignal:
    """フィルタで落ちた転換。統計を取ると閾値調整の材料になる。"""

    bar_index: int
    side: Side
    reason: str
    rr: float | None = None


class DowReversalStrategy:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        detector = SwingDetector(
            left=config.swing.left,
            right=config.swing.right,
            atr_period=config.swing.atr_period,
            min_swing_atr=config.swing.min_swing_atr,
        )
        self.analyzer = DowAnalyzer(
            detector=detector,
            require_prior_trend=config.entry.require_prior_trend,
            stop_basis_mode=config.entry.stop_basis_mode,
        )
        self.htf: HigherTimeframeFilter | None = None
        if config.entry.higher_timeframe is not None:
            self.htf = HigherTimeframeFilter(
                minutes=granularity_minutes(config.entry.higher_timeframe),
                left=config.swing.left,
                right=config.swing.right,
                atr_period=config.swing.atr_period,
                min_swing_atr=config.swing.min_swing_atr,
                require_prior_trend=config.entry.require_prior_trend,
                stop_basis_mode=config.entry.stop_basis_mode,
            )
        self.rejections: list[RejectedSignal] = []
        self.last_event: ReversalEvent | None = None
        """直近バーで検出されたダウ転換。RR フィルタで落ちた場合も残る。"""

    # ------------------------------------------------------------------
    @property
    def atr(self) -> float | None:
        return self.analyzer.atr

    @property
    def trend(self):
        return self.analyzer.trend

    def update(self, candle: Candle) -> Signal | None:
        """確定足を 1 本処理し、全フィルタを通過したシグナルだけを返す。"""
        # 上位足を先に進める。確定済みの上位足バーだけが反映されるため、
        # この足の判断に使っても先読みにはならない。
        if self.htf is not None:
            self.htf.update(candle)
        event = self.analyzer.update(candle)
        self.last_event = event
        if event is None:
            return None
        return self._build_signal(event)

    # ------------------------------------------------------------------
    def _reject(self, event: ReversalEvent, reason: str, rr: float | None = None) -> None:
        self.rejections.append(
            RejectedSignal(
                bar_index=event.bar_index, side=event.side, reason=reason, rr=rr
            )
        )

    def _htf_allows(self, event: ReversalEvent, atr: float) -> bool:
        """上位足の状況がこの転換を許すか。落とす場合は理由を記録する。"""
        cfg = self.config.entry
        htf = self.htf
        assert htf is not None

        if cfg.require_htf_alignment:
            if cfg.htf_alignment_source == "trend":
                # 上位足が「いまトレンド中」であること。レンジ中は見送る。
                direction = htf.trend
                if direction is Trend.RANGE:
                    self._reject(event, "htf_no_trend")
                    return False
            else:
                direction = htf.bias
                if direction is None:
                    self._reject(event, "htf_no_bias")
                    return False
            if (event.side is Side.LONG) != (direction is Trend.UP):
                self._reject(event, "htf_not_aligned")
                return False

        if cfg.htf_max_bars is not None and htf.bars_since_reversal > cfg.htf_max_bars:
            self._reject(event, "htf_setup_stale")
            return False

        if cfg.htf_proximity_atr is not None:
            if htf.extreme is None:
                self._reject(event, "htf_no_extreme")
                return False
            # 距離は「上位足の ATR」で測る。上位足スケールの押し目を下位足の
            # ATR で測ると物差しが小さすぎ、ほぼ全部が「遠い」判定になる。
            scale = htf.atr if htf.atr and htf.atr > 0 else atr
            distance = abs(event.candle.close - htf.extreme)
            if distance > scale * cfg.htf_proximity_atr:
                self._reject(event, "htf_too_far_from_extreme")
                return False

        return True

    def _build_signal(self, event: ReversalEvent) -> Signal | None:
        cfg = self.config.entry

        # 取れない方向は最初に落とす(現物の暗号資産は売り建てができない)。
        if event.side is Side.LONG and not cfg.allow_long:
            self._reject(event, "long_not_allowed")
            return None
        if event.side is Side.SHORT and not cfg.allow_short:
            self._reject(event, "short_not_allowed")
            return None

        atr = self.analyzer.atr
        if atr is None or atr <= 0:
            self._reject(event, "atr_not_ready")
            return None

        # 上位足フィルタ: 下位足の転換を「候補」に格下げし、上位足の転換方向と
        # その後の極値(押し安値 / 戻り高値)の付近だけを採る。
        if self.htf is not None and not self._htf_allows(event, atr):
            return None

        # 飛び乗り防止: ブレイク水準から離れすぎている転換は見送る。
        if cfg.max_break_extension_atr > 0 and event.extension > atr * cfg.max_break_extension_atr:
            self._reject(event, "break_extension_too_large")
            return None

        entry = event.candle.close
        buffer = atr * cfg.stop_buffer_atr

        if event.side is Side.LONG:
            stop = event.stop_basis - buffer
            risk = entry - stop
        else:
            stop = event.stop_basis + buffer
            risk = stop - entry

        if risk <= 0:
            self._reject(event, "non_positive_risk")
            return None
        if risk < atr * cfg.min_stop_distance_atr:
            self._reject(event, "stop_too_tight")
            return None
        if risk > atr * cfg.max_stop_distance_atr:
            self._reject(event, "stop_too_wide")
            return None

        target = resolve_target(
            side=event.side,
            entry=entry,
            risk_per_unit=risk,
            swings=self.analyzer.swings,
            atr=atr,
            config=cfg,
        )
        if target is None:
            self._reject(event, "no_target_level")
            return None

        reward = abs(target.price - entry)
        rr = reward / risk

        # 要件 3: リスクリワードが 1/2 を上回る(= reward >= 2 x risk)場合のみ。
        if rr < cfg.min_rr:
            self._reject(event, "rr_below_minimum", rr=rr)
            return None

        structure = self.analyzer.snapshot()
        direction = "上抜け" if event.side is Side.LONG else "下抜け"
        reason = (
            f"{event.previous_trend.value} トレンド中に直近スイング "
            f"{event.broken_level:.5f} を終値 {entry:.5f} で{direction}(ダウ転換)。"
            f"損切りは転換前の極値 {event.stop_basis:.5f}、"
            f"利確根拠は {target.source}、RR={rr:.2f}"
        )

        return Signal(
            time=event.candle.time,
            bar_index=event.bar_index,
            side=event.side,
            reference_price=entry,
            stop_loss=stop,
            take_profit=target.price,
            risk_per_unit=risk,
            reward_per_unit=reward,
            rr=rr,
            broken_level=event.broken_level,
            stop_basis=event.stop_basis,
            target_source=target.source,
            structure=structure,
            reason=reason,
        )

    # ------------------------------------------------------------------
    def rejection_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for rejection in self.rejections:
            summary[rejection.reason] = summary.get(rejection.reason, 0) + 1
        return summary
