"""設定の読み込みと検証.

YAML → dataclass。未知のキーは黙って捨てずにエラーにする(タイポで
リスク設定が効かないまま動く事故を防ぐ)。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class InstrumentConfig:
    symbol: str = "USD_JPY"
    """OANDA 形式の通貨ペア名(例: USD_JPY, EUR_USD)。"""
    granularity: str = "M15"
    pip_size: float = 0.01
    """1 pip の価格幅。USD_JPY 系は 0.01、それ以外は概ね 0.0001。"""
    quote_to_account_rate: float = 1.0
    """クオート通貨→口座通貨の換算レート。1 単位あたり損益の計算に使う。"""


@dataclass
class SwingConfig:
    left: int = 3
    right: int = 3
    atr_period: int = 14
    min_swing_atr: float = 0.6
    """スイングとして認めるための最小値幅(ATR 倍)。小さいほど高頻度・高ノイズ。"""


@dataclass
class EntryConfig:
    require_prior_trend: bool = True
    """True の場合、直前が明確な逆方向トレンドであることを転換の条件にする。"""
    stop_basis_mode: str = "trend_extreme"
    """損切り根拠の起点。

    trend_extreme: 転換前の波全体の極値(要件の文言どおり。損切りは深い)
    recent_swing : 直近の押し安値/戻り高値(損切りは浅く RR は改善するが狩られやすい)
    """
    max_break_extension_atr: float = 1.0
    """ブレイク水準から終値がこれ以上離れていたら「飛び乗り」として見送る。"""
    stop_buffer_atr: float = 0.15
    """損切りを転換前の極値からさらに離す量(ATR 倍)。"""
    min_rr: float = 2.0
    """リスクリワードの下限。要件『1/2 を上回る場合のみ』= reward >= 2 x risk。"""
    min_stop_distance_atr: float = 0.25
    """損切り幅がこれ未満なら、スプレッドに対して薄すぎるので見送る。"""
    max_stop_distance_atr: float = 4.0
    """損切りが遠すぎる(=転換前の値幅が異常)場合も見送る。"""
    target_strategies: list[str] = field(
        default_factory=lambda: ["trend_origin", "measured_move", "atr"]
    )
    """利確目標の決定順。先に『水準を出せた』ものを採用し、その後 RR 判定する。

    既定が trend_origin なのは、損切りを転換前の極値に置く=リスク幅が波 1 本分に
    なるため。最も近い壁(structure)を目標にすると RR が構造的に 1 前後へ張り付き、
    RR>=2 のフィルタをほぼ何も通過できなくなる。"""
    structure_lookback_swings: int = 20
    measured_move_mult: float = 1.0
    atr_target_mult: float = 3.0
    min_target_distance_atr: float = 0.5
    """目標が近すぎる場合は構造上の水準として採用しない。"""


@dataclass
class RiskConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.02
    """1 トレードあたりの許容損失(口座残高比)。"""
    max_risk_per_trade: float = 0.05
    """risk_per_trade の絶対上限。動的サイジングでもここを超えない。"""
    max_daily_loss: float = 0.06
    """1 日の累計損失がこれを超えたらその日は新規エントリー停止。"""
    max_drawdown_stop: float = 0.25
    """最大ドローダウンがこれを超えたら全停止(キルスイッチ)。"""
    max_concurrent_positions: int = 1
    monthly_target: float = 1.4
    """目標月利。1.4 = 月あたり資産 1.4 倍(+40%)。実現可能性は target コマンドで検証する。"""
    compounding: bool = True


@dataclass
class ExecutionConfig:
    entry_mode: str = "next_open"
    """next_open(翌足始値で約定, 現実的) / close(シグナル足終値で約定, 楽観的)。"""
    spread_pips: float = 1.0
    slippage_pips: float = 0.2
    commission_per_unit: float = 0.0
    break_even_at_r: float | None = 1.0
    """含み益がこの R 倍数に達したら損切りを建値へ。None で無効。"""
    trail_to_structure: bool = True
    """新しい押し安値/戻り高値が確定するたびに損切りを追従させる。"""
    max_bars_in_trade: int = 400
    exit_on_structure_flip: bool = True
    """保有中に逆方向のダウ転換が出たら手仕舞う。"""


@dataclass
class LLMConfig:
    enabled: bool = False
    """バックテストでは既定 off(決定性とコストのため)。ペーパー取引では on 推奨。"""
    model: str = "claude-opus-5"
    effort: str = "medium"
    """low / medium / high / xhigh / max。"""
    max_tokens: int = 8000
    gate_enabled: bool = True
    """LLM に拒否権(見送り判断)を与えるか。"""
    journal_enabled: bool = True
    """エントリー時所感と決済時の敗因分析を書かせるか。"""
    fail_open: bool = True
    """API 障害時にエントリーを通すか(True)、見送るか(False)。"""
    cache_path: str = "data/llm_cache.sqlite"
    timeout_seconds: float = 120.0


@dataclass
class BacktestConfig:
    warmup_bars: int = 100
    """この本数までは統計を安定させるためエントリーしない。"""


@dataclass
class AppConfig:
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AppConfig":
        raw = raw or {}
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ConfigError(f"未知の設定セクション: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            section = raw.get(f.name)
            kwargs[f.name] = _build_section(f.type, f.name, section)  # type: ignore[arg-type]
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        if path is None:
            return cls.from_dict({})
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.entry.min_rr <= 0:
            raise ConfigError("entry.min_rr は正の数である必要があります")
        if not 0 < self.risk.risk_per_trade <= 1:
            raise ConfigError("risk.risk_per_trade は 0 < x <= 1 の範囲です")
        if self.risk.risk_per_trade > self.risk.max_risk_per_trade:
            raise ConfigError(
                "risk.risk_per_trade が max_risk_per_trade を超えています "
                f"({self.risk.risk_per_trade} > {self.risk.max_risk_per_trade})"
            )
        if self.risk.initial_equity <= 0:
            raise ConfigError("risk.initial_equity は正の数である必要があります")
        if self.entry.stop_basis_mode not in {"trend_extreme", "recent_swing"}:
            raise ConfigError(
                "entry.stop_basis_mode は trend_extreme か recent_swing です"
            )
        if self.execution.entry_mode not in {"next_open", "close"}:
            raise ConfigError("execution.entry_mode は next_open か close です")
        if self.instrument.pip_size <= 0:
            raise ConfigError("instrument.pip_size は正の数である必要があります")
        if self.llm.effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ConfigError("llm.effort は low/medium/high/xhigh/max のいずれかです")
        valid_targets = {"structure", "trend_origin", "measured_move", "atr", "fixed_r"}
        unknown = set(self.entry.target_strategies) - valid_targets
        if unknown:
            raise ConfigError(f"未知の利確戦略: {sorted(unknown)}")
        if not self.entry.target_strategies:
            raise ConfigError("entry.target_strategies が空です")


_SECTION_TYPES = {
    "instrument": InstrumentConfig,
    "swing": SwingConfig,
    "entry": EntryConfig,
    "risk": RiskConfig,
    "execution": ExecutionConfig,
    "llm": LLMConfig,
    "backtest": BacktestConfig,
}


def _build_section(_type: Any, name: str, raw: Any) -> Any:
    section_cls = _SECTION_TYPES[name]
    if raw is None:
        return section_cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"設定セクション '{name}' はマッピングである必要があります")
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"'{name}' に未知のキー: {sorted(unknown)}")
    return section_cls(**raw)
