"""ペーパー取引(実行ループ)のテスト.

API に触れずに、実行ループ全体が回ることを確認する。ここが動けば
OANDA へ差し替えるのはフィードとブローカーの入れ替えだけになる。
"""

from __future__ import annotations

import pytest

from llmfx.config import AppConfig
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.types import Side
from llmfx.execution.broker import PaperBroker
from llmfx.execution.runner import ReplayFeed, TradingRunner
from llmfx.journal.store import JournalStore


def build_runner(config: AppConfig | None = None, count: int = 6000, **kwargs):
    config = config or AppConfig.from_dict({})
    candles = generate_synthetic_candles(count=count, seed=29)
    feed = ReplayFeed(candles, warmup_bars=300)
    broker = PaperBroker(config)
    runner = TradingRunner(config=config, feed=feed, broker=broker, **kwargs)
    return runner, broker, candles


def test_replay_run_produces_trades():
    runner, broker, _ = build_runner()
    runner.warmup()
    stats = runner.run()

    assert stats.bars_processed > 0
    assert stats.entries > 0, "再生モードでエントリーが 1 件も出ないのは異常"
    assert stats.exits <= stats.entries


def test_paper_broker_equity_matches_sum_of_trade_pnl():
    config = AppConfig.from_dict({})
    runner, broker, _ = build_runner(config)
    runner.warmup()
    runner.run()

    expected = config.risk.initial_equity + sum(t.pnl for t in broker.trades)
    assert broker.equity() == pytest.approx(expected)


def test_only_one_position_is_open_at_a_time():
    runner, broker, _ = build_runner()
    runner.warmup()

    open_counts = []
    while not runner.feed.finished():
        for candle in runner.feed.poll():
            runner.process_candle(candle)
            open_counts.append(1 if broker.position() is not None else 0)
    assert max(open_counts) <= 1


def test_every_paper_trade_respected_the_rr_filter():
    config = AppConfig.from_dict({"entry": {"min_rr": 2.0}})
    runner, broker, _ = build_runner(config)
    runner.warmup()
    runner.run()

    assert broker.trades
    for trade in broker.trades:
        assert trade.rr_at_entry >= 2.0
        if trade.side is Side.LONG:
            assert trade.take_profit > trade.entry_price
        else:
            assert trade.take_profit < trade.entry_price


def test_trades_are_persisted_to_the_journal(tmp_path):
    config = AppConfig.from_dict({})
    store = JournalStore(tmp_path / "journal.sqlite")
    run = store.start_run("test", config.instrument.symbol, config.instrument.granularity, {})

    runner, broker, _ = build_runner(
        config,
        on_trade=lambda t: store.record_trade(t, config.instrument.symbol, run.run_id),
    )
    runner.warmup()
    runner.run()

    assert store.trade_count() == len(broker.trades)
    recorded = store.recent_trades(limit=5)
    if recorded:
        assert recorded[0]["target_source"]
        assert recorded[0]["exit_reason"]
    store.close()


def test_max_bars_limit_is_respected():
    runner, _broker, _ = build_runner()
    runner.warmup()
    stats = runner.run(max_bars=50)
    assert stats.bars_processed == 50


def test_llm_gate_can_veto_entries():
    """拒否権が実際にエントリーを止めることを、偽ゲートで確認する。"""

    class RejectEverything:
        def evaluate(self, signal, context):
            return {"approve": False, "primary_reason": "テスト", "source": "test"}

    runner, broker, _ = build_runner(gate=RejectEverything())
    runner.warmup()
    stats = runner.run()

    assert stats.signals > 0, "比較できるだけのシグナルが出ていること"
    assert stats.entries == 0
    assert stats.gate_vetoes == stats.signals
    assert broker.trades == []


def test_risk_kill_switch_stops_the_runner():
    config = AppConfig.from_dict(
        {"risk": {"max_drawdown_stop": 0.0005, "risk_per_trade": 0.05}}
    )
    runner, _broker, _ = build_runner(config)
    runner.warmup()
    runner.run()
    # 損失が出ていればキルスイッチが作動しているはず
    if runner.stats.trades and any(t.pnl < 0 for t in runner.stats.trades):
        assert runner.risk_manager.halted
