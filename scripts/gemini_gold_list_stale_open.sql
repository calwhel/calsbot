-- List stale OPEN gemini-gold executions blocking max_open_position cap.
-- Run against production Postgres (Railway → PostgreSQL → Query).

SELECT
    id AS execution_id,
    user_id,
    fired_at,
    direction,
    entry_price,
    ctrader_order_id,
    ctrader_position_id,
    notes
FROM strategy_executions
WHERE outcome = 'OPEN'
  AND notes LIKE '%gemini_gold_trader%'
ORDER BY fired_at DESC;

-- Count only:
-- SELECT COUNT(*) FROM strategy_executions
-- WHERE outcome = 'OPEN' AND notes LIKE '%gemini_gold_trader%';
