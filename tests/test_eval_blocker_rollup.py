"""Last-100 evaluation blocker frequency rollup."""
from app.services.feed_diagnostics import (
    log_scan_metric,
    summarize_recent_blockers,
)


def test_blocker_rollup_counts():
    from app.services import feed_diagnostics as fd
    fd._EVAL_RING.clear()

    for _ in range(5):
        log_scan_metric(
            symbol="XAUUSD",
            timeframe="15m",
            candles_loaded=0,
            strategy_evaluated=True,
            setup_detected=False,
            signal_generated=False,
            signal_sent=False,
            strategy_id=1,
            block_reason="no_price_data",
        )
    for _ in range(3):
        log_scan_metric(
            symbol="EURUSD",
            timeframe="15m",
            candles_loaded=80,
            strategy_evaluated=True,
            setup_detected=False,
            signal_generated=False,
            signal_sent=False,
            strategy_id=2,
            block_reason="ta_conditions",
        )

    summary = summarize_recent_blockers(100)
    assert summary["evaluations"] == 8
    assert summary["top_blocker"] == "no_price_data"
    assert summary["blocker_counts"]["no_price_data"] == 5
    assert summary["blocker_counts"]["ta_conditions"] == 3
