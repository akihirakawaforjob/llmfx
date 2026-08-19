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
    min_order_size: float = 1.0
    """発注できる最小数量。FX は通貨単位なので 1、BTC は 0.001 など。"""
    size_step: float = 1.0
    """数量の刻み。これの倍数に切り下げて発注する。

    ここを 1 のまま暗号資産に使うと、1 BTC(= 数百万円)未満の建玉がすべて
    0 に丸められ、資金に見合った建玉がひとつも建たない。
    """


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
    mode: str = "breakout"
    """エントリーの向き。

    breakout — ダウ転換の方向へ順張り(当初の要件どおり)
    fade     — ダウ転換を **ダマシとみなして逆に張る**

    FX 8 銘柄 x 20 年 22,264 件で、順張りは -0.114 R (t=-7.68)、コストを
    外しても -0.071 R (t=-5.33) だった。値動きそのものがブレイク方向と逆に
    偏っている。その偏りを取りにいくのが fade。

    単純な符号反転ではない。損切りと利確を専用に組み直す(下記の
    fade_stop_buffer_atr / fade_target_r)。
    """
    fade_stop_buffer_atr: float = 0.3
    """fade の損切りを、ブレイクで付けた極値からどれだけ離すか(ATR 倍)。

    ダマシでなく本物のブレイクだった場合にすぐ切るための水準。
    ここが遠いと「損小」にならず、逆張りの利点が消える。
    """
    fade_target_r: float = 1.0
    """fade の利確を、リスクの何倍に置くか。

    遠い目標を狙うと勝率が落ちて元の木阿弥になることは、順張り側の
    RR 掃引で繰り返し確認済み(min_rr を上げるほど悪化した)。
    ここは固定の R 倍数で持ち、min_rr のフィルタは fade では使わない。
    """
    stop_basis_mode: str = "trend_extreme"
    allow_long: bool = True
    allow_short: bool = True
    """売買方向の許可。現物の暗号資産は売り建てができないため allow_short: false にする。

    ダウ転換は上下対称に出るので、片方を切ると単純にシグナルが半減する。
    """
    higher_timeframe: str | None = None
    """上位足(H1 / H4 / D1 など)。None で上位足フィルタを使わない。

    下位足のダウ転換は単体では予測力が無い(実測 勝率 20.2% / 損益分岐 20.5%)。
    上位足の転換と組み合わせて「候補の抽出」として使う。
    """
    require_htf_alignment: bool = True
    """上位足の方向と一致するシグナルだけを採る。"""
    htf_alignment_source: str = "reversal"
    """上位足の「方向」をどう取るか。

    reversal — 直近の上位足ダウ転換の向き。転換した瞬間の情報。
               その後トレンドが崩れても保持される。
    trend    — 上位足の *いまの* トレンド状態(HH/HL なら上昇)。
               「上位足でトレンドが起きている間ずっと」という持続的な条件。
               レンジ中は成立しないので、シグナル数は状況次第で増減する。

    どちらが良いかは実測で決めること。この 2 つは別物であり、
    「上位足のトレンドに従う」という言葉はふつう trend のほうを指す。
    """
    htf_proximity_atr: float | None = None
    """上位足の転換後に付けた極値から、この ATR 倍以内でのみエントリーする。

    上昇バイアスなら「転換後の最安値の付近」、下降なら「最高値の付近」。
    None で距離制限なし(方向の一致だけを見る)。

    倍率の基準は **上位足の ATR**。上位足スケールの押し目を下位足の ATR で
    測ると物差しが小さすぎ、ほぼ全部が「遠い」と判定されてしまう。
    """
    htf_max_bars: int | None = None
    """上位足の転換からこの本数(下位足)を超えたら設定を無効にする。None で無期限。"""
    allowed_hours_utc: list[list[int]] = field(default_factory=list)
    """エントリーを許す時間帯(UTC)。空なら全時間帯。例: [[7, 16]]

    ダウ転換はブレイクを捉える手法なので、ブレイクが機能する時間帯と
    ダマシになる時間帯があるはず。カレンダーから事前に分かる情報なので
    先読みにならない。
    """
    blocked_weekdays: list[int] = field(default_factory=list)
    """見送る曜日(月=0 … 日=6)。FX の月曜早朝や金曜終盤を外す用。"""
    nfp_blackout_minutes: int = 0
    """米雇用統計の前後この分数を遮断する。0 で無効。

    毎月第 1 金曜と決まっているので事前に分かる。「荒れた日を後から除く」の
    ような先読みとは別物。
    """
    max_atr_percentile: float | None = None
    """直近の ATR が過去分布のこの分位より上ならエントリーを見送る(0〜1)。

    介入や指標で価格構造の外側から動かされている場面を弾くための条件。
    「164 円で買わない」のような絶対水準の決め打ちは、その水準を知り得たのが
    介入の後である以上、先読みになる。過去のバーだけから計算できる相対値で
    表すこと。
    """
    atr_percentile_lookback: int = 500
    """max_atr_percentile の判定に使う過去の本数。"""
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
    use_take_profit: bool = True
    """固定の利確を置くか。

    False にすると利確を置かず、`execution.trail_to_structure` による
    損切りの引き上げだけで決済する。ダウ理論の読みでは、新しい押し安値が
    確定するたびに損切りを上げていけば、調整波で刈られずに伸ばせる。

    実測(BTC 3,026 件)では、決済管理を外すと平均勝ちが 1.53 R から 3.72 R へ
    2.4 倍に伸びた。勝率は落ちるが利幅は明確に伸びる。ただしこれまでの
    検証は「固定の利確を残したままトレーリングを足す」形で、目標が
    先に効いてトレーリングの意味が薄れていた。両者を分けて測るための設定。
    """
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
    max_leverage: float | None = None
    """建玉評価額 / 資産 の上限。None で無制限(実質的に検証用)。

    国内の暗号資産は法令でレバレッジ 2 倍が上限。これを設定しないと、
    損切りが近い場面で建玉が資産の何倍にも膨らみ、実際には建てられない
    サイズを前提にした成績が出る。
    """
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
    spread_bps: float = 0.0
    """価格に比例するスプレッド(1 bp = 0.01%)。spread_pips とは加算される。

    暗号資産のように価格水準が数倍動く銘柄では固定 pips では表せない。
    BTC が 300 万円の時期と 1,000 万円の時期で、同じ 1,000 円のスプレッドは
    3.3bp と 1.0bp で意味が変わってしまう。
    """
    commission_bps: float = 0.0
    """約定代金にかかる手数料(片道、1 bp = 0.01%)。往復では 2 倍かかる。

    GMOコインの取引所現物 Taker 0.05% なら 5。暗号資産FX は 0。
    """
    daily_holding_cost_bps: float = 0.0
    """建玉を 1 日持ち越すごとに建玉評価額へかかる費用(1 bp = 0.01%)。

    GMOコインの暗号資産FX のレバレッジ手数料 0.04%/日 なら 4。年率約 14.6%。

    FX のスワップポイントは売買方向で符号が変わる(受け取りにもなる)ため、
    この一律コストでは表せない。FX で使う場合は 0 のままにすること。
    """
    holding_cost_rollover_hour_utc: int = 21
    """持ち越し判定の時刻(UTC)。GMOコインは日本時間 6:00 = UTC 21:00。"""
    break_even_at_r: float | None = 1.0
    """含み益がこの R 倍数に達したら損切りを建値へ。None で無効。"""
    trail_to_structure: bool = True
    """新しい押し安値/戻り高値が確定するたびに損切りを追従させる。"""
    trail_timeframe: str | None = None
    """追従に使うスイングの時間足。None で取引足のスイングを使う。

    取引足のスイングは頻繁に更新されるため、伸びる途中の調整波で刈られる。
    実測(FX 3 銘柄 20 年)では、取引足で追従すると平均勝ちが 1.79 R まで縮み、
    追従なしの 7.96 R を大きく下回った。**太い右の裾を細かい勝ちに
    変換してしまう。**

    ここに上位足(H4 など)を指定すると、スイングの更新がずっと遅くなるので
    刈られる回数が減る。「調整波では動かず、トレンドが変わったときだけ動く」
    追従になる。
    """
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
    holdout_start: str | None = None
    """検証用データの開始日(例: "2025-01-01")。None で分割なし。

    ここより前が開発用、以降が検証用。約 90 通りを試している以上、
    探索に使ったデータで良し悪しを決めることはできない。採用を決めるまで
    一度も見ていない期間で確かめるために分ける。

    `--split` の既定は dev なので、検証用を見るには明示が要る。
    """


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
        if self.instrument.size_step <= 0:
            raise ConfigError("instrument.size_step は正の数である必要があります")
        if self.instrument.min_order_size <= 0:
            raise ConfigError("instrument.min_order_size は正の数である必要があります")
        if self.risk.max_leverage is not None and self.risk.max_leverage <= 0:
            raise ConfigError("risk.max_leverage は正の数か None である必要があります")
        if not self.entry.allow_long and not self.entry.allow_short:
            raise ConfigError(
                "entry.allow_long と entry.allow_short の両方が false ではエントリーできません"
            )
        for span in self.entry.allowed_hours_utc:
            if len(span) != 2 or not all(0 <= h <= 24 for h in span):
                raise ConfigError(
                    f"entry.allowed_hours_utc の要素は [開始時, 終了時] (0〜24) です: {span}"
                )
        if any(d < 0 or d > 6 for d in self.entry.blocked_weekdays):
            raise ConfigError("entry.blocked_weekdays は 0(月)〜6(日) です")
        if self.entry.max_atr_percentile is not None and not (
            0.0 < self.entry.max_atr_percentile <= 1.0
        ):
            raise ConfigError("entry.max_atr_percentile は 0 より大きく 1 以下です")
        if self.entry.mode not in {"breakout", "fade"}:
            raise ConfigError("entry.mode は breakout か fade です")
        if self.entry.fade_target_r <= 0:
            raise ConfigError("entry.fade_target_r は正の数である必要があります")
        if self.entry.fade_stop_buffer_atr <= 0:
            raise ConfigError("entry.fade_stop_buffer_atr は正の数である必要があります")
        if self.entry.htf_alignment_source not in {"reversal", "trend"}:
            raise ConfigError(
                "entry.htf_alignment_source は reversal か trend です"
            )
        if self.entry.htf_proximity_atr is not None and self.entry.htf_proximity_atr <= 0:
            raise ConfigError("entry.htf_proximity_atr は正の数か None です")
        if self.entry.higher_timeframe is not None:
            from .domain.mtf import granularity_minutes

            if granularity_minutes(self.entry.higher_timeframe) <= granularity_minutes(
                self.instrument.granularity
            ):
                raise ConfigError(
                    f"entry.higher_timeframe({self.entry.higher_timeframe})は "
                    f"instrument.granularity({self.instrument.granularity})より "
                    "上位の足である必要があります"
                )
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
