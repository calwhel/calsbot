-- List stale OPEN gemini-gold executions blocking max_open_position cap.
-- Run against production Postgres (Railway → PostgreSQL → Query).
--
-- One-time cleanup (after reviewing rows): POST
--   /api/gemini-gold-trader/reconcile?uid=TH-ZKJO6YKX
-- Or dry-run preview:
--   /api/gemini-gold-trader/reconcile?uid=TH-ZKJO6YKX&dry_run=true

SELECT
    id AS execution_id,
    user_id,
    fired_at,
    direction,
    entry_price,
    ctrader_account_id,
    ctrader_order_id,
    ctrader_position_id,
    notes
FROM strategy_executions
WHERE outcome = 'OPEN'
  AND notes LIKE '%gemini_gold_trader%'
ORDER BY fired_at DESC;

-- After orphan reconcile, cancelled rows look like:
-- SELECT id, outcome, closed_at, notes FROM strategy_executions
-- WHERE notes LIKE '%gemini orphan reconcile%'
-- ORDER BY closed_at DESC;
