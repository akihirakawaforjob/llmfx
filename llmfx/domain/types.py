"""ドメインの基本型.

外部ライブラリに依存しない dataclass のみで構成する。バックテストと
ライブ実行の両方が同じ型を共有することで、ロジックの二重実装を防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """ロング=+1 / ショート=-1。価格差を損益方向へ変換するのに使う。"""
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class Trend(str, Enum):
    UP = "up"
    DOWN = "down"
    RANGE = "range"


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class SwingLabel(str, Enum):
    """ダウ理論における高値・安値の切り上げ/切り下げ分類."""

    HH = "higher_high"
    LH = "lower_high"
    HL = "higher_low"
    LL = "lower_low"
    UNKNOWN = "unknown"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_STOP = "time_stop"
    TRAILING_STOP = "trailing_stop"
    STRUCTURE_FLIP = "structure_flip"
    END_OF_DATA = "end_of_data"
    RISK_KILL_SWITCH = "risk_kill_switch"
    MANUAL = "manual"


@dataclass(frozen=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Swing:
    """確定したスイング(ピボット)高値・安値.

    `index` は元のローソク足列における位置、`confirmed_index` は
    そのスイングが確定した(=戦略が参照してよくなった)バー位置。
    右側 N 本の確定待ちがあるため両者は必ずずれる。
    """

    index: int
    confirmed_index: int
    time: datetime
    price: float
    type: SwingType
    label: SwingLabel = SwingLabel.UNKNOWN


@dataclass(frozen=True)
class StructureSnapshot:
    """シグナル発生時点での相場構造。LLM への説明材料にもなる。"""

    trend: Trend
    last_high: float | None
    last_low: float | None
    prior_high: float | None
    prior_low: float | None
    last_high_label: SwingLabel
    last_low_label: SwingLabel
    atr: float
    swing_count: int


@dataclass(frozen=True)
class Signal:
    """エントリー候補。RR フィルタを通過したものだけが生成される。"""

    time: datetime
    bar_index: int
    side: Side
    reference_price: float
    """シグナル検出バーの終値。実際の約定価格はこれとは別(執行モデル依存)。"""
    stop_loss: float
    take_profit: float
    risk_per_unit: float
    reward_per_unit: float
    rr: float
    broken_level: float
    """ブレイクされた直近スイング(ダウ転換の起点)。"""
    stop_basis: float
    """転換前の最安値/最高値そのもの(バッファ適用前)。"""
    target_source: str
    structure: StructureSnapshot
    reason: str


@dataclass
class Position:
    signal: Signal
    side: Side
    units: float
    entry_price: float
    entry_time: datetime
    entry_index: int
    stop_loss: float
    take_profit: float
    initial_risk_per_unit: float
    risk_amount: float
    """エントリー時に許容した損失額(口座通貨)。R 倍数の基準。"""
    moved_to_break_even: bool = False
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    scaled_out: bool = False
    """一部利確を済ませたか。1 建玉につき 1 回だけ。"""
    realized_pnl: float = 0.0
    """一部利確で確定済みの損益(口座通貨)。残玉の決済時に足す。"""
    scaled_units: float = 0.0
    """一部利確で落とした数量。手数料の計算に要る。"""
    entry_note: dict | None = None
    """LLM が書いたエントリー時の所感(あれば)。"""


@dataclass
class Trade:
    """決済まで完了した取引の記録."""

    side: Side
    units: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop_loss: float
    take_profit: float
    initial_risk_per_unit: float
    risk_amount: float
    pnl: float
    r_multiple: float
    exit_reason: ExitReason
    bars_held: int
    equity_after: float
    rr_at_entry: float
    target_source: str
    structure: StructureSnapshot | None = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    scaled_out: bool = False
    """途中で一部利確したか。"""
    commission_paid: float = 0.0
    holding_cost_paid: float = 0.0
    """建玉管理料(暗号資産FX のレバレッジ手数料など)。日跨ぎで発生する。"""
    entry_note: dict | None = None
    exit_note: dict | None = None
    gate_decision: dict | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0
