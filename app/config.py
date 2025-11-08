from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    BROADCAST_CHAT_ID: str
    
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: Optional[str] = None
    DATABASE_URL: Optional[str] = None
    
    TIMEZONE: str = "UTC"
    SYMBOLS: str = "BTC/USDT:USDT,ETH/USDT:USDT"
    EXCHANGE: str = "binance"
    TIMEFRAME: str = "4h"
    SCAN_INTERVAL: int = 900  # 15 minutes for 4h timeframe
    
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    EMA_TREND: int = 50
    TRAIL_PCT: float = 1.5
    
    SOL_MERCHANT: Optional[str] = None
    SPL_USDC_MINT: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    SUB_PRICE_USDC: float = 200.0
    
    HELIUS_WEBHOOK_SECRET: Optional[str] = None
    WHOP_CHECKOUT_URL: Optional[str] = None
    WHOP_WEBHOOK_SECRET: Optional[str] = None
    
    ENCRYPTION_KEY: Optional[str] = None
    CRYPTONEWS_API_KEY: Optional[str] = None
    NOWPAYMENTS_API_KEY: Optional[str] = None
    NOWPAYMENTS_IPN_SECRET: Optional[str] = None
    SUBSCRIPTION_PRICE_USD: float = 200.00  # Single tier pricing
    
    # Subscription Tiers (unified pricing)
    MANUAL_SIGNALS_PRICE: float = 200.00
    AUTO_TRADING_PRICE: float = 200.00
    
    # Referral Payout Settings
    REFERRAL_PAYOUT_AMOUNT: float = 50.00  # $50 cash payout per referral
    MINIMUM_PAYOUT_THRESHOLD: float = 50.00  # Minimum to request withdrawal
    
    PORT: int = 5000

    class Config:
        env_file = ".env"
        extra = "allow"

    def get_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if all([self.DB_HOST, self.DB_PORT, self.DB_USER, self.DB_PASSWORD, self.DB_NAME]):
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        raise ValueError("Database configuration incomplete")


settings = Settings()
