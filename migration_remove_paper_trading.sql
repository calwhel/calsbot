-- Migration: Remove paper trading columns and table from production database
-- Run this in Railway's PostgreSQL console to fix the frozen bot

-- Step 1: Drop paper_trades table if exists
DROP TABLE IF EXISTS paper_trades CASCADE;

-- Step 2: Remove paper trading columns from user_preferences
ALTER TABLE user_preferences DROP COLUMN IF EXISTS paper_trading_mode CASCADE;
ALTER TABLE user_preferences DROP COLUMN IF EXISTS paper_balance CASCADE;

-- Step 3: Verify the changes
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'user_preferences' 
ORDER BY ordinal_position;
