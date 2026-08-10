"""コマンドラインインターフェース.

    llmfx data synth      合成データを生成(API 不要の動作確認用)
    llmfx data fetch      OANDA から確定足を取得して CSV へ
    llmfx backtest        バックテスト実行 + Markdown レポート出力
    llmfx diagnose        シグナルの却下理由と RR 分布を確認(パラメータ調整用)
    llmfx target          目標月利に必要なリスク率と破産確率を計算
    llmfx paper           ペーパー取引(CSV 再生 / OANDA デモ)
    llmfx review          蓄積した記録から改善レポートを生成
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .backtest.engine import BacktestEngine
from .backtest.metrics import compute_stats
from .backtest.report import render_report, write_report
from .config import AppConfig, ConfigError
from .data.csv_source import load_candles_csv, save_candles_csv
from .data.synthetic import generate_synthetic_candles
from .domain.risk import kelly_fraction, monte_carlo, required_risk_fraction
from .domain.strategy import DowReversalStrategy
from .execution.broker import PaperBroker
from .execution.runner import ReplayFeed, TradingRunner
from .journal.review import build_review
from .journal.store import JournalStore
from .llm.client import LLMClient
from .llm.gate import EntryGate
from .llm.journalist import Journalist

logger = logging.getLogger("llmfx")


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except ConfigError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"ファイルが見つかりません: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断しました", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfx",
        description="ダウ転換ベースの自動売買システム(LLM レビュー層つき)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="デバッグログを出す")
    sub = parser.add_subparsers(dest="command")

    # -- data ----------------------------------------------------------
    data = sub.add_parser("data", help="価格データの取得・生成").add_subparsers(dest="data_command")

    synth = data.add_parser("synth", help="合成データを生成(API 不要)")
    synth.add_argument("--out", default="data/synthetic.csv")
    synth.add_argument("--count", type=int, default=20000)
    synth.add_argument("--granularity", default="M15")
    synth.add_argument("--start-price", type=float, default=150.0)
    synth.add_argument("--seed", type=int, default=20260810)
    synth.set_defaults(handler=_cmd_data_synth)

    fetch = data.add_parser("fetch", help="OANDA から確定足を取得")
    fetch.add_argument("--instrument", default="USD_JPY")
    fetch.add_argument("--granularity", default="M15")
    fetch.add_argument("--count", type=int, default=5000)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--price", default="M", choices=["M", "B", "A"])
    fetch.set_defaults(handler=_cmd_data_fetch)

    # -- backtest ------------------------------------------------------
    backtest = sub.add_parser("backtest", help="バックテストを実行")
    backtest.add_argument("--config", help="設定 YAML のパス")
    backtest.add_argument("--data", required=True, help="ローソク足 CSV")
    backtest.add_argument("--report", default="out/report.md")
    backtest.add_argument("--trades-csv", default="out/trades.csv")
    backtest.add_argument("--journal", help="結果を保存する journal DB のパス")
    backtest.add_argument("--llm", action="store_true", help="LLM 層を有効化(要 API キー)")
    backtest.set_defaults(handler=_cmd_backtest)

    # -- diagnose ------------------------------------------------------
    diagnose = sub.add_parser("diagnose", help="シグナルの却下理由と RR 分布を確認")
    diagnose.add_argument("--config")
    diagnose.add_argument("--data", required=True)
    diagnose.set_defaults(handler=_cmd_diagnose)

    # -- target --------------------------------------------------------
    target = sub.add_parser("target", help="目標月利に必要なリスク率と破産確率を計算")
    target.add_argument("--config")
    target.add_argument("--win-rate", type=float, required=True, help="勝率 (0-1)")
    target.add_argument("--win-r", type=float, required=True, help="平均利益 (R 倍数)")
    target.add_argument("--loss-r", type=float, default=1.0, help="平均損失 (R 倍数)")
    target.add_argument("--trades-per-month", type=float, required=True)
    target.add_argument("--monthly-target", type=float, help="目標月利(既定は設定値)")
    target.add_argument("--months", type=int, default=12)
    target.set_defaults(handler=_cmd_target)

    # -- paper ---------------------------------------------------------
    paper = sub.add_parser("paper", help="ペーパー取引を実行")
    paper.add_argument("--config")
    paper.add_argument("--replay", help="CSV を再生してオフライン実行する")
    paper.add_argument("--live-feed", action="store_true", help="OANDA から足を取得する")
    paper.add_argument("--broker", default="paper", choices=["paper", "oanda"])
    paper.add_argument("--journal", default="data/journal.sqlite")
    paper.add_argument("--max-bars", type=int, help="処理する足数の上限")
    paper.add_argument("--llm", action="store_true", help="LLM 層を有効化")
    paper.set_defaults(handler=_cmd_paper)

    # -- review --------------------------------------------------------
    review = sub.add_parser("review", help="記録から改善レポートを生成")
    review.add_argument("--config")
    review.add_argument("--journal", default="data/journal.sqlite")
    review.add_argument("--days", type=int, help="対象期間(日数)。省略で全期間")
    review.add_argument("--out", default="out/review.md")
    review.add_argument("--llm", action="store_true", help="LLM の改善提案を付ける")
    review.set_defaults(handler=_cmd_review)

    return parser


# ----------------------------------------------------------------------
def _load_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.load(getattr(args, "config", None))
    if getattr(args, "llm", False):
        config.llm.enabled = True
    return config


def _build_llm_layer(config: AppConfig):
    """(gate, journalist, client) を返す。利用不能なら (None, None, client)。"""
    client = LLMClient(config.llm)
    if not client.available:
        if config.llm.enabled:
            print(f"⚠️  LLM 層は無効です: {client.unavailable_reason}")
        return None, None, client
    gate = EntryGate(client, config) if config.llm.gate_enabled else None
    journalist = Journalist(client, config) if config.llm.journal_enabled else None
    return gate, journalist, client


# ----------------------------------------------------------------------
def _cmd_data_synth(args: argparse.Namespace) -> int:
    candles = generate_synthetic_candles(
        count=args.count,
        start_price=args.start_price,
        granularity=args.granularity,
        seed=args.seed,
    )
    written = save_candles_csv(candles, args.out)
    print(f"合成データを {written:,} 本書き出しました -> {args.out}")
    print("⚠️  これは動作確認用の人工データです。ここで出た成績に意味はありません。")
    return 0


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    from .data.oanda import OandaClient, OandaError

    try:
        with OandaClient() as client:
            candles = client.fetch_candles(
                instrument=args.instrument,
                granularity=args.granularity,
                count=args.count,
                price=args.price,
            )
    except OandaError as exc:
        print(f"OANDA からの取得に失敗しました: {exc}", file=sys.stderr)
        return 2

    if not candles:
        print("確定足が取得できませんでした", file=sys.stderr)
        return 2
    written = save_candles_csv(candles, args.out)
    print(
        f"{args.instrument} {args.granularity} を {written:,} 本取得しました "
        f"({candles[0].time:%Y-%m-%d} 〜 {candles[-1].time:%Y-%m-%d}) -> {args.out}"
    )
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    config = _load_config(args)
    candles = load_candles_csv(args.data)
    if len(candles) < 200:
        print(f"データが {len(candles)} 本しかありません。最低 200 本必要です。", file=sys.stderr)
        return 2

    gate, journalist, client = (None, None, None)
    if config.llm.enabled:
        gate, journalist, client = _build_llm_layer(config)

    print(f"バックテスト実行中: {len(candles):,} 本 ...")
    result = BacktestEngine(config, gate=gate, journalist=journalist).run(candles)
    stats = compute_stats(result)
    write_report(result, stats, args.report, args.trades_csv)

    print(render_report(result, stats))
    print(f"\nレポート -> {args.report}")
    if result.trades:
        print(f"トレード明細 -> {args.trades_csv}")

    if args.journal:
        with JournalStore(args.journal) as store:
            run = store.start_run(
                mode="backtest",
                instrument=config.instrument.symbol,
                granularity=config.instrument.granularity,
                config=config.to_dict(),
            )
            store.record_trades(result.trades, config.instrument.symbol, run.run_id)
        print(f"記録 -> {args.journal}")

    if client is not None:
        client.close()
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    config = _load_config(args)
    candles = load_candles_csv(args.data)

    # RR フィルタを外して、転換の RR 分布そのものを観測する。
    probe_config = AppConfig.from_dict(config.to_dict())
    probe_config.entry.min_rr = 1e-6
    strategy = DowReversalStrategy(probe_config)
    signals = [s for c in candles if (s := strategy.update(c))]

    print(f"データ: {len(candles):,} 本 ({candles[0].time:%Y-%m-%d} 〜 {candles[-1].time:%Y-%m-%d})")
    print(f"検出したダウ転換: {len(signals) + len(strategy.rejections)} 件")
    print()
    print("却下理由(RR フィルタを外した状態):")
    for reason, count in sorted(strategy.rejection_summary().items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s} {count:5d}")
    print()

    if not signals:
        print("RR を計算できたシグナルがありません。swing の設定を緩めてください。")
        return 0

    rr = np.array([s.rr for s in signals])
    risk_atr = np.array([s.risk_per_unit / s.structure.atr for s in signals if s.structure.atr > 0])
    print("リスクリワード分布:")
    for pct in (10, 25, 50, 75, 90):
        print(f"  p{pct:<3d} {np.percentile(rr, pct):6.2f}")
    print(f"  最大 {rr.max():6.2f}")
    print()
    print("設定した min_rr ごとの通過数:")
    for threshold in (1.0, 1.5, 2.0, 2.5, 3.0):
        passed = int((rr >= threshold).sum())
        mark = " <- 現在の設定" if abs(threshold - config.entry.min_rr) < 1e-9 else ""
        print(f"  RR >= {threshold:.1f}: {passed:5d} 件 ({passed / len(rr):5.1%}){mark}")
    print()
    if len(risk_atr):
        print(f"損切り幅の中央値: {np.median(risk_atr):.2f} ATR")
        print("(損切りを転換前の極値に置く仕様上、リスク幅は直前の波の全長になります)")
    return 0


def _cmd_target(args: argparse.Namespace) -> int:
    config = _load_config(args)
    monthly_target = args.monthly_target or config.risk.monthly_target

    kelly = kelly_fraction(args.win_rate, args.win_r, args.loss_r)
    required = required_risk_fraction(
        monthly_target=monthly_target,
        trades_per_month=args.trades_per_month,
        win_rate=args.win_rate,
        win_r=args.win_r,
        loss_r=args.loss_r,
    )
    expectancy = args.win_rate * args.win_r - (1 - args.win_rate) * args.loss_r

    print(f"前提: 勝率 {args.win_rate:.1%} / 平均利益 {args.win_r:.2f}R / "
          f"平均損失 {args.loss_r:.2f}R / 月間 {args.trades_per_month:.0f} トレード")
    print(f"1 トレードあたり期待値: {expectancy:+.3f} R")
    print(f"ケリー基準リスク率    : {kelly:.2%}")
    print(f"目標                  : 月利 {monthly_target:.2f} 倍 ({monthly_target - 1:+.0%})")
    print()

    if expectancy <= 0:
        print("⛔ 期待値がマイナスです。リスク率をどう変えても資産は減ります。")
        return 1
    if required is None:
        max_growth = np.exp(
            args.trades_per_month
            * (
                args.win_rate * np.log1p(kelly * args.win_r)
                + (1 - args.win_rate) * np.log1p(-kelly * args.loss_r)
            )
        )
        print("⛔ この成績では目標月利に到達できません。")
        print(f"   ケリー基準で張っても月利は最大 {max_growth:.3f} 倍 ({max_growth - 1:+.1%}) です。")
        print("   到達には勝率・平均 R・トレード頻度のいずれかの改善が必要です。")
        return 1

    print(f"✅ 必要な 1 トレードあたりリスク率: {required:.2%}")
    if kelly > 0:
        print(f"   (ケリー基準の {required / kelly:.2f} 倍)")
    print()

    mc = monte_carlo(
        risk_fraction=required,
        win_rate=args.win_rate,
        win_r=args.win_r,
        loss_r=args.loss_r,
        trades_per_month=args.trades_per_month,
        months=args.months,
        monthly_target=monthly_target,
    )
    print(f"モンテカルロ({mc.paths:,} 経路 / {args.months} ヶ月):")
    print(f"  資金半減(破産)確率      : {mc.prob_ruin:.1%}")
    print(f"  各月が目標を達成する確率  : {mc.prob_hit_target_monthly:.1%}")
    print(f"  期間平均で目標達成する確率: {mc.prob_target_on_average:.1%}")
    print(f"  最大ドローダウン中央値    : {mc.median_max_drawdown:.1%}")
    print(f"  最大ドローダウン最悪値    : {mc.worst_max_drawdown:.1%}")
    print(f"  最終資産倍率 p5/中央/p95  : {mc.p05_final_multiple:.2f} / "
          f"{mc.median_final_multiple:.2f} / {mc.p95_final_multiple:.2f}")

    if mc.prob_ruin >= 0.10:
        print()
        print(f"⚠️  破産確率 {mc.prob_ruin:.1%} は高すぎます。目標月利の引き下げを検討してください。")
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:
    config = _load_config(args)

    if args.replay:
        candles = load_candles_csv(args.replay)
        feed = ReplayFeed(candles, warmup_bars=min(500, len(candles) // 2))
        print(f"再生モード: {len(candles):,} 本の CSV を時系列で流します")
    elif args.live_feed:
        from .data.oanda import OandaClient
        from .execution.runner import OandaFeed

        oanda = OandaClient()
        feed = OandaFeed(oanda, config.instrument.symbol, config.instrument.granularity)
        print(f"ライブフィード: OANDA {config.instrument.symbol} {config.instrument.granularity}")
    else:
        print("--replay か --live-feed のどちらかを指定してください", file=sys.stderr)
        return 2

    if args.broker == "oanda":
        from .data.oanda import OandaClient
        from .execution.broker import OandaBroker

        broker = OandaBroker(config, OandaClient())
        print("執行: OANDA デモ口座へ実際に発注します")
    else:
        broker = PaperBroker(config)
        print(f"執行: シミュレーション(初期資金 {config.risk.initial_equity:,.0f})")

    gate = journalist = client = None
    if config.llm.enabled:
        gate, journalist, client = _build_llm_layer(config)

    store = JournalStore(args.journal)
    run = store.start_run(
        mode=f"paper:{args.broker}",
        instrument=config.instrument.symbol,
        granularity=config.instrument.granularity,
        config=config.to_dict(),
    )

    runner = TradingRunner(
        config=config,
        feed=feed,
        broker=broker,
        gate=gate,
        journalist=journalist,
        on_trade=lambda trade: store.record_trade(trade, config.instrument.symbol, run.run_id),
    )
    runner.warmup()
    stats = runner.run(max_bars=args.max_bars)

    print()
    print(f"処理した足      : {stats.bars_processed:,}")
    print(f"シグナル        : {stats.signals}")
    print(f"エントリー      : {stats.entries}")
    print(f"決済            : {stats.exits}")
    if stats.gate_vetoes:
        print(f"LLM による見送り: {stats.gate_vetoes}")
    if stats.blocked_by_risk:
        print(f"リスク制限で見送: {stats.blocked_by_risk}")
    print(f"最終資金        : {broker.equity():,.2f}")
    if stats.trades:
        total_r = sum(t.r_multiple for t in stats.trades)
        wins = sum(1 for t in stats.trades if t.pnl > 0)
        print(f"成績            : {wins}/{len(stats.trades)} 勝ち, 累計 {total_r:+.2f}R")
    print(f"記録            : {args.journal}")

    store.close()
    if client is not None:
        client.close()
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    config = _load_config(args)
    since = (
        datetime.now(timezone.utc) - timedelta(days=args.days) if args.days else None
    )

    client = None
    if config.llm.enabled:
        client = LLMClient(config.llm)
        if not client.available:
            print(f"⚠️  LLM は利用できません: {client.unavailable_reason}")

    with JournalStore(args.journal) as store:
        if store.trade_count() == 0:
            print(f"{args.journal} に記録がありません。先に backtest か paper を実行してください。")
            return 1
        markdown, report = build_review(store, config, client=client, since=since)
        if report:
            store.save_review(
                scope=f"{args.days}d" if args.days else "all",
                stats={},
                report=report,
                markdown=markdown,
            )

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nレビュー -> {args.out}")

    if client is not None:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
