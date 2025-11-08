-- Add referral payout tracking table
CREATE TABLE IF NOT EXISTS referral_payouts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    amount_usd FLOAT NOT NULL,
    wallet_address VARCHAR NOT NULL,
    payment_method VARCHAR DEFAULT 'USDT_TRC20',
    status VARCHAR DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    admin_id INTEGER REFERENCES users(id),
    admin_notes TEXT,
    transaction_hash VARCHAR
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_referral_payouts_user_id ON referral_payouts(user_id);
CREATE INDEX IF NOT EXISTS idx_referral_payouts_status ON referral_payouts(status);
CREATE INDEX IF NOT EXISTS idx_referral_payouts_created_at ON referral_payouts(created_at);

-- Add comment
COMMENT ON TABLE referral_payouts IS 'Tracks $50 cash payout requests for successful referrals';
