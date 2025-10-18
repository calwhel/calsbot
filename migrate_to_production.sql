-- Migration script to copy paper trading data from development to production database
-- Run this in your PRODUCTION database using the Database tool in Replit

-- Step 1: Insert your user if not exists (user_id = 1, telegram_id = 5603353066)
INSERT INTO users (id, telegram_id, is_admin, created_at)
VALUES (1, 5603353066, true, NOW())
ON CONFLICT (id) DO NOTHING;

-- Step 2: Insert user preferences with all your settings
INSERT INTO user_preferences (
    user_id, muted_symbols, default_pnl_period, dm_alerts, auto_trading_enabled,
    position_size_percent, max_positions, safety_paused, accepted_risk_levels,
    risk_based_sizing, use_trailing_stop, trailing_stop_percent, use_breakeven_stop,
    daily_loss_limit, max_drawdown_percent, min_balance, max_consecutive_losses,
    cooldown_after_loss, emergency_stop, news_signals_enabled, min_news_impact,
    min_news_confidence, tp1_percent, tp2_percent, tp3_percent, paper_trading_mode,
    paper_balance, user_leverage, trailing_activation_percent, trailing_step_percent,
    adaptive_sizing_enabled, win_streak_multiplier, loss_streak_divider,
    current_win_streak, trade_cooldown_minutes, max_trades_per_day,
    same_symbol_cooldown_minutes, trades_today, market_condition_adaptive,
    volatility_threshold_high, volatility_threshold_low, high_volatility_size_reduction,
    use_limit_orders, entry_slippage_percent, limit_order_timeout_seconds,
    rr_scaling_enabled, min_rr_for_full_size, rr_scaling_multiplier,
    preferred_exchange, correlation_filter_enabled, max_correlated_positions,
    funding_rate_alerts_enabled, funding_rate_threshold
) VALUES (
    1, 'BTC/USDT', 'today', true, true,
    5, 3, false, 'LOW,MEDIUM',
    false, true, 5, true,
    1000, 50, 0, 5,
    0, false, true, 9,
    80, 30, 30, 40, true,
    1046.35, 10, 2, 1,
    true, 1.2, 0.8,
    0, 15, 10,
    60, 0, true,
    5, 1.5, 0.6,
    false, 0.3, 30,
    true, 3, 0.3,
    'mexc', true, 1,
    false, 0.1
)
ON CONFLICT (user_id) DO UPDATE SET
    paper_balance = 1046.35,
    paper_trading_mode = true,
    preferred_exchange = 'mexc';

-- Step 3: Insert your 5 paper trades
-- Trade 1: ETH/USDT LONG (CLOSED) - Profit: $25.00
INSERT INTO paper_trades (
    id, user_id, signal_id, symbol, direction, entry_price, exit_price,
    stop_loss, take_profit, take_profit_1, take_profit_2, take_profit_3,
    position_size, remaining_size, tp1_hit, tp2_hit, tp3_hit, status,
    pnl, pnl_percent, opened_at, closed_at
) VALUES (
    1, 1, 46, 'ETH/USDT', 'LONG', 3797.54, 3892.48,
    3721.5892, 3949.4416, 3854.5031, 3892.4785, 3949.4416,
    150, 73.5, true, true, false, 'closed',
    25.000394992547808, 2.5000394992547808,
    '2025-10-17 17:16:03.093978', '2025-10-18 10:56:50.888559'
)
ON CONFLICT (id) DO NOTHING;

-- Trade 2: ETH/USDT LONG (OPEN)
INSERT INTO paper_trades (
    id, user_id, signal_id, symbol, direction, entry_price, exit_price,
    stop_loss, take_profit, position_size, remaining_size,
    tp1_hit, tp2_hit, tp3_hit, status, pnl, pnl_percent, opened_at
) VALUES (
    2, 1, 47, 'ETH/USDT', 'LONG', 3834.04, NULL,
    3680.6784, 3987.4016, 127.5, 127.5,
    false, false, false, 'open', 0, 0,
    '2025-10-17 18:22:57.685062'
)
ON CONFLICT (id) DO NOTHING;

-- Trade 3: TRX/USDT LONG (CLOSED) - Profit: $14.72
INSERT INTO paper_trades (
    id, user_id, signal_id, symbol, direction, entry_price, exit_price,
    stop_loss, take_profit, position_size, remaining_size,
    tp1_hit, tp2_hit, tp3_hit, status, pnl, pnl_percent, opened_at, closed_at
) VALUES (
    3, 1, 48, 'TRX/USDT', 'LONG', 0.3092, 0.3134,
    0.296832, 0.321568, 108.375, 0,
    false, false, false, 'closed',
    14.721054333764684, 135.83441138421853,
    '2025-10-17 18:35:00.409609', '2025-10-18 12:02:13.537423'
)
ON CONFLICT (id) DO NOTHING;

-- Trade 4: BTC/USDT SHORT (OPEN)
INSERT INTO paper_trades (
    id, user_id, signal_id, symbol, direction, entry_price, exit_price,
    stop_loss, take_profit, position_size, remaining_size,
    tp1_hit, tp2_hit, tp3_hit, status, pnl, pnl_percent, opened_at
) VALUES (
    4, 1, 52, 'BTC/USDT', 'SHORT', 106694.9, NULL,
    110962.696, 102427.104, 92.11875, 92.11875,
    false, false, false, 'open', 0, 0,
    '2025-10-17 19:06:55.381738'
)
ON CONFLICT (id) DO NOTHING;

-- Trade 5: UNI/USDT SHORT (OPEN)
INSERT INTO paper_trades (
    id, user_id, signal_id, symbol, direction, entry_price, exit_price,
    stop_loss, take_profit, position_size, remaining_size,
    tp1_hit, tp2_hit, tp3_hit, status, pnl, pnl_percent, opened_at
) VALUES (
    5, 1, 54, 'UNI/USDT', 'SHORT', 6.0575, NULL,
    6.2998, 5.8152, 162.92671018606782, 162.92671018606782,
    false, false, false, 'open', 0, 0,
    '2025-10-18 09:05:37.056494'
)
ON CONFLICT (id) DO NOTHING;

-- Verification queries
SELECT 'Migration completed!' as message;
SELECT COUNT(*) as total_paper_trades FROM paper_trades WHERE user_id = 1;
SELECT symbol, direction, status, pnl FROM paper_trades WHERE user_id = 1 ORDER BY id;
SELECT paper_balance, paper_trading_mode FROM user_preferences WHERE user_id = 1;
