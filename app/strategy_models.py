"""
User Strategy Models — Build Your Own Strategy Portal
New tables only — does not modify any existing tables.
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class UserStrategy(Base):
    __tablename__ = "user_strategies"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name        = Column(String(120), nullable=False)
    description = Column(Text, nullable=True)
    config      = Column(JSON, nullable=False)          # full strategy JSON spec
    status      = Column(String(20), default="draft")   # draft | active | paused | archived
    is_public   = Column(Boolean, default=False)        # marketplace visibility
    clone_count = Column(Integer, default=0)
    webhook_token = Column(String(64), nullable=True, unique=True, index=True)
    asset_class = Column(String(16), nullable=False, default="crypto", server_default="crypto", index=True)
    ctrader_account_id = Column(String(40), nullable=True)  # per-strategy cTrader ctid; null → user default
    ctrader_account_lot = Column(Float, nullable=True)  # per-assignment lot override; null → strategy default size
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    executions  = relationship("StrategyExecution", back_populates="strategy", cascade="all, delete-orphan")
    performance = relationship("StrategyPerformance", back_populates="strategy", uselist=False, cascade="all, delete-orphan")
    account_assignments = relationship(
        "StrategyAccountAssignment",
        back_populates="strategy",
        cascade="all, delete-orphan",
    )


class StrategyAccountAssignment(Base):
    """Per-strategy × per-account execution target (enable + lot size)."""
    __tablename__ = "strategy_account_assignments"
    __table_args__ = (
        UniqueConstraint(
            "strategy_id", "ctrader_account_id", name="uq_strategy_account_acct",
        ),
    )

    id                  = Column(Integer, primary_key=True, index=True)
    strategy_id         = Column(Integer, ForeignKey("user_strategies.id"), nullable=False, index=True)
    ctrader_account_id  = Column(String(40), nullable=False)
    enabled     = Column(Boolean, default=False, nullable=False, server_default="false")
    lot_size    = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

    strategy = relationship("UserStrategy", back_populates="account_assignments")


class StrategyExecution(Base):
    __tablename__ = "strategy_executions"

    id              = Column(Integer, primary_key=True, index=True)
    strategy_id     = Column(Integer, ForeignKey("user_strategies.id"), nullable=False, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol          = Column(String(30), nullable=False)
    direction       = Column(String(10), nullable=False)   # LONG | SHORT
    entry_price     = Column(Float, nullable=True)
    exit_price      = Column(Float, nullable=True)
    tp_price        = Column(Float, nullable=True)
    sl_price        = Column(Float, nullable=True)
    tp2_price       = Column(Float, nullable=True)         # optional second TP
    leverage        = Column(Integer, default=10)
    position_size   = Column(Float, nullable=True)         # USD notional
    outcome         = Column(String(20), default="OPEN")   # OPEN | WIN | LOSS | BREAKEVEN | CANCELLED
    pnl_pct         = Column(Float, nullable=True)
    pips_pnl        = Column(Float, nullable=True)          # P&L in pips (forex/metals only)
    spread_pips_applied = Column(Float, nullable=True)      # spread deducted in paper eval
    pnl_usd         = Column(Float, nullable=True)
    conditions_met  = Column(JSON, nullable=True)          # which conditions triggered
    fired_at        = Column(DateTime, default=datetime.utcnow)
    closed_at       = Column(DateTime, nullable=True)
    bitunix_order_id  = Column(String(80), nullable=True)
    ctrader_order_id  = Column(String(80), nullable=True)  # cTrader live forex orders
    ctrader_position_id = Column(String(40), nullable=True)
    ctrader_account_id  = Column(String(40), nullable=True)
    signal_group_id     = Column(String(40), nullable=True, index=True)  # links mirror legs of one signal
    broker_volume_units = Column(Integer, nullable=True)
    breakeven_applied   = Column(Boolean, default=False, server_default="false")
    tp1_done            = Column(Boolean, default=False, server_default="false")
    tp1_closed_volume   = Column(Float, nullable=True)
    tp1_realized_pips   = Column(Float, nullable=True)
    current_sl          = Column(Float, nullable=True)
    remaining_volume    = Column(Float, nullable=True)
    mfe_pips          = Column(Float, nullable=True)   # peak favorable excursion (pips)
    mae_pips          = Column(Float, nullable=True)   # peak adverse excursion (pips)
    notified_close_at = Column(DateTime, nullable=True)  # close Telegram/push sent once
    notes           = Column(Text, nullable=True)
    is_paper        = Column(Boolean, default=False)       # paper trade — no real order placed
    asset_class     = Column(String(16), nullable=False, default="crypto", server_default="crypto", index=True)

    strategy = relationship("UserStrategy", back_populates="executions")


class StrategyPortalSettings(Base):
    """Per-user defaults for the Build Your Own Strategy portal."""
    __tablename__ = "strategy_portal_settings"

    id                      = Column(Integer, primary_key=True, index=True)
    user_id                 = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    # Defaults applied when creating a new strategy
    default_leverage        = Column(Integer, default=10)
    default_position_size   = Column(Float, default=5.0)    # %
    default_daily_loss_limit = Column(Float, default=5.0)   # %
    default_max_positions   = Column(Integer, default=3)
    default_direction       = Column(String(10), default="LONG")
    default_cooldown_minutes = Column(Integer, default=60)
    default_max_trades_day  = Column(Integer, default=3)
    # Behaviour
    paper_mode_default      = Column(Boolean, default=False)  # new strategies start in paper mode
    auto_activate           = Column(Boolean, default=False)  # skip draft → go straight to paper/active
    dm_paper_alerts         = Column(Boolean, default=True)   # Telegram DMs for paper trade signals
    dm_live_alerts          = Column(Boolean, default=True)   # Telegram DMs for live trade signals
    # Global risk override
    global_daily_loss_pct   = Column(Float, default=0.0)      # 0 = no global override
    global_max_positions    = Column(Integer, default=0)       # 0 = no global override
    updated_at              = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"

    id            = Column(Integer, primary_key=True, index=True)
    strategy_id   = Column(Integer, ForeignKey("user_strategies.id"), unique=True, nullable=False)
    total_trades  = Column(Integer, default=0)
    open_trades   = Column(Integer, default=0)
    wins          = Column(Integer, default=0)
    losses        = Column(Integer, default=0)
    breakevens    = Column(Integer, default=0)
    win_rate      = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    avg_win_pct   = Column(Float, default=0.0)
    avg_loss_pct  = Column(Float, default=0.0)
    avg_rr        = Column(Float, default=0.0)
    best_trade        = Column(Float, default=0.0)
    worst_trade       = Column(Float, default=0.0)
    total_pips_pnl    = Column(Float, nullable=True)        # cumulative pips (forex/metals)
    avg_pips_per_trade = Column(Float, nullable=True)       # avg pips per closed trade
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    strategy = relationship("UserStrategy", back_populates="performance")


class StrategyMarketplace(Base):
    __tablename__ = "strategy_marketplace"

    id              = Column(Integer, primary_key=True, index=True)
    strategy_id     = Column(Integer, ForeignKey("user_strategies.id"), unique=True, nullable=False)
    author_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    title           = Column(String(120), nullable=False)
    summary         = Column(Text, nullable=True)            # AI-generated description
    tags            = Column(JSON, default=list)             # ["scalp", "reversal", "SMC"]
    category        = Column(String(40), default="general")  # scalp|swing|reversal|smc|breakout|momentum
    # Pricing
    pricing_model   = Column(String(20), default="free")     # free | one_time | subscription
    price_usdt      = Column(Float, default=0.0)             # price in USD
    # Stats
    clone_count     = Column(Integer, default=0)
    subscriber_count = Column(Integer, default=0)
    avg_rating      = Column(Float, default=0.0)
    rating_count    = Column(Integer, default=0)
    # Verification
    is_verified     = Column(Boolean, default=False)         # 10+ live trades tracked
    verified_trades = Column(Integer, default=0)
    verified_win_rate = Column(Float, default=0.0)
    verified_pnl    = Column(Float, default=0.0)
    # Meta
    published_at    = Column(DateTime, default=datetime.utcnow)
    is_featured     = Column(Boolean, default=False)
    is_trending     = Column(Boolean, default=False)
    view_count      = Column(Integer, default=0)
    # AI Originals — strategies generated, backtested and published autonomously
    # by the AI Strategy Generator service (app/services/ai_strategy_generator.py).
    is_ai_generated = Column(Boolean, default=False, index=True)
    backtest_sharpe = Column(Float, default=0.0)        # snapshot at publish time
    backtest_pnl_pct = Column(Float, default=0.0)        # snapshot at publish time
    backtest_trades = Column(Integer, default=0)
    backtest_win_rate = Column(Float, default=0.0)
    backtest_max_dd = Column(Float, default=0.0)
    backtest_days   = Column(Integer, default=0)


class StrategyOffer(Base):
    """An offer sent by a user to get access to someone else's paper strategy."""
    __tablename__ = "strategy_offers"

    id              = Column(Integer, primary_key=True, index=True)
    strategy_id     = Column(Integer, ForeignKey("user_strategies.id"), nullable=False, index=True)
    author_id       = Column(Integer, ForeignKey("users.id"), nullable=False)   # strategy owner
    requester_id    = Column(Integer, ForeignKey("users.id"), nullable=False)   # person sending offer
    message         = Column(Text, nullable=True)
    status          = Column(String(20), default="pending")   # pending | accepted | declined
    created_at      = Column(DateTime, default=datetime.utcnow)


class PortalSubscription(Base):
    """Tracks free vs Pro tier for the strategy portal (separate from Telegram bot subscription)."""
    __tablename__ = "portal_subscriptions"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    tier                = Column(String(20), default="free")       # 'free' | 'pro'
    subscription_start  = Column(DateTime, nullable=True)
    subscription_end    = Column(DateTime, nullable=True)
    chat_calls_used     = Column(Integer, default=0)               # resets monthly
    chat_calls_reset_at = Column(DateTime, nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    updated_at          = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ScanSchedule(Base):
    """Scheduled gold/forex/index discovery scans with Telegram alerts."""
    __tablename__ = "scan_schedules"

    id                 = Column(Integer, primary_key=True, index=True)
    uid                = Column(String(40), nullable=False, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name               = Column(String(120), default="")
    scan_type          = Column(String(20), nullable=False, index=True)  # gold | forex | index
    symbol             = Column(String(20), nullable=True)
    categories_json    = Column(JSON, nullable=True)
    quality_cfg_json   = Column(JSON, nullable=True)
    scan_params_json   = Column(JSON, nullable=True)   # days, direction, etc.
    interval_minutes   = Column(Integer, default=60)
    min_grade_alert    = Column(String(1), default="B")
    enabled            = Column(Boolean, default=True)
    last_run_at        = Column(DateTime, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow)


class DiscoveryScanJob(Base):
    """Background gold/index/forex discovery scans — shared across gunicorn workers."""
    __tablename__ = "discovery_scan_jobs"

    id          = Column(Integer, primary_key=True, index=True)
    job_key     = Column(String(96), unique=True, nullable=False, index=True)  # e.g. gold:TH-ABC
    scan_type   = Column(String(20), nullable=False, index=True)
    uid         = Column(String(40), nullable=False, index=True)
    status      = Column(String(20), default="queued")   # queued | running | done | error
    message     = Column(Text, default="")
    result_json = Column(JSON, nullable=True)
    error       = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


class LiveFireFailure(Base):
    """Durable log of live cTrader order failures (survives Railway log rotation)."""
    __tablename__ = "live_fire_failures"

    id               = Column(Integer, primary_key=True, index=True)
    ts               = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    strategy_id      = Column(Integer, nullable=True, index=True)
    execution_id     = Column(Integer, nullable=True, index=True)
    signal_group_id  = Column(String(40), nullable=True, index=True)
    ctid             = Column(String(40), nullable=True)
    symbol           = Column(String(30), nullable=True)
    direction        = Column(String(10), nullable=True)
    lots             = Column(String(20), nullable=True)
    reason           = Column(Text, nullable=False)
    category         = Column(String(32), nullable=False)
    attempts         = Column(Integer, default=1, nullable=False)
    broker_reply     = Column(Text, nullable=True)
    sibling_summary  = Column(Text, nullable=True)


class PortalPayment(Base):
    """Tracks OxaPay invoices so the webhook can match payment → user."""
    __tablename__ = "portal_payments"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    track_id   = Column(String(128), unique=True, nullable=False, index=True)
    amount     = Column(Float, default=60.0)
    months     = Column(Integer, default=1)
    status     = Column(String(20), default="pending")   # pending | paid | expired
    created_at = Column(DateTime, default=datetime.utcnow)
    paid_at    = Column(DateTime, nullable=True)


def init_strategy_tables(engine):
    """Create strategy tables if they don't exist. Safe to call on every startup."""
    try:
        import app.models  # noqa — registers User and other tables on the shared Base
    except Exception:
        pass
    Base.metadata.create_all(bind=engine, tables=[
        UserStrategy.__table__,
        StrategyAccountAssignment.__table__,
        StrategyExecution.__table__,
        StrategyPerformance.__table__,
        StrategyMarketplace.__table__,
        StrategyPortalSettings.__table__,
        PortalSubscription.__table__,
        PortalPayment.__table__,
        StrategyOffer.__table__,
        DiscoveryScanJob.__table__,
        LiveFireFailure.__table__,
        ScanSchedule.__table__,
    ])
    # Add new columns only if genuinely missing — avoids table locks when multiple
    # portal instances (dev + production) share the same Neon database.
    from sqlalchemy import text
    with engine.connect() as conn:
        # Bound statement time and — critically — cap how long any DDL will WAIT
        # for a lock. Without lock_timeout a blocked ALTER/CREATE INDEX queues
        # behind other table access and then blocks every subsequent SELECT on
        # that table (lock-queue starvation), which hangs all portal API calls.
        # Session-level SET (not LOCAL) so it applies to every DDL below.
        try:
            conn.execute(text("SET lock_timeout = '2s'"))
            conn.execute(text("SET statement_timeout = '30000'"))
            conn.commit()
        except Exception:
            pass

        existing_cols = {
            (row[0], row[1])
            for row in conn.execute(text(
                "SELECT table_name, column_name "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name IN ('strategy_executions', 'strategy_marketplace', "
                "'strategy_performance', 'user_strategies')"
            ))
        }

        for col, typ in [
            ("ctrader_account_id",   "VARCHAR(40)"),
            ("ctrader_account_lot",  "FLOAT"),
        ]:
            if ("user_strategies", col) not in existing_cols:
                try:
                    conn.execute(text(
                        f"ALTER TABLE user_strategies ADD COLUMN IF NOT EXISTS {col} {typ}"
                    ))
                    conn.commit()
                except Exception:
                    pass

        for col, typ in [
            ("is_paper",             "BOOLEAN DEFAULT FALSE"),
            ("tp2_price",            "FLOAT"),
            ("pips_pnl",             "FLOAT"),
            ("spread_pips_applied",  "FLOAT"),
            ("ctrader_position_id",  "VARCHAR(40)"),
            ("ctrader_account_id",   "VARCHAR(40)"),
            ("signal_group_id",      "VARCHAR(40)"),
            ("broker_volume_units",  "INTEGER"),
            ("mfe_pips",             "NUMERIC"),
            ("mae_pips",             "NUMERIC"),
            ("notified_close_at",    "TIMESTAMP"),
        ]:
            if ("strategy_executions", col) not in existing_cols:
                try:
                    conn.execute(text(f"ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS {col} {typ}"))
                    conn.commit()
                except Exception:
                    pass

        for col, typ in [
            ("total_pips_pnl",     "FLOAT"),
            ("avg_pips_per_trade", "FLOAT"),
        ]:
            if ("strategy_performance", col) not in existing_cols:
                try:
                    conn.execute(text(f"ALTER TABLE strategy_performance ADD COLUMN IF NOT EXISTS {col} {typ}"))
                    conn.commit()
                except Exception:
                    pass

        for col, typ in [
            ("category",          "VARCHAR(50) DEFAULT 'general'"),
            ("tags",              "TEXT"),
            ("is_ai_generated",   "BOOLEAN DEFAULT FALSE"),
            ("backtest_sharpe",   "FLOAT DEFAULT 0"),
            ("backtest_pnl_pct",  "FLOAT DEFAULT 0"),
            ("backtest_trades",   "INTEGER DEFAULT 0"),
            ("backtest_win_rate", "FLOAT DEFAULT 0"),
            ("backtest_max_dd",   "FLOAT DEFAULT 0"),
            ("backtest_days",     "INTEGER DEFAULT 0"),
        ]:
            if ("strategy_marketplace", col) not in existing_cols:
                try:
                    conn.execute(text(f"ALTER TABLE strategy_marketplace ADD COLUMN {col} {typ}"))
                    conn.commit()
                    # Index hot filter columns to keep marketplace queries snappy
                    # as AI Curator publishes thousands of listings.
                    if col == "is_ai_generated":
                        try:
                            conn.execute(text(
                                "CREATE INDEX IF NOT EXISTS ix_strategy_marketplace_is_ai_generated "
                                "ON strategy_marketplace(is_ai_generated)"
                            ))
                            conn.commit()
                        except Exception:
                            pass
                except Exception:
                    pass

        # Composite and single-column indexes on hot filter columns.
        # Use lock_timeout so a waiting CREATE INDEX never causes lock-queue
        # starvation that blocks all subsequent SELECTs on the table.
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_se_outcome    ON strategy_executions(outcome)",
            "CREATE INDEX IF NOT EXISTS idx_se_is_paper   ON strategy_executions(is_paper)",
            "CREATE INDEX IF NOT EXISTS idx_se_closed_at  ON strategy_executions(closed_at)",
            "CREATE INDEX IF NOT EXISTS idx_se_fired_at   ON strategy_executions(fired_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_se_strat_outcome ON strategy_executions(strategy_id, outcome)",
            "CREATE INDEX IF NOT EXISTS idx_se_strat_paper   ON strategy_executions(strategy_id, is_paper)",
            "CREATE INDEX IF NOT EXISTS idx_se_user_paper    ON strategy_executions(user_id, is_paper)",
            "CREATE INDEX IF NOT EXISTS idx_us_status     ON user_strategies(status)",
            "CREATE INDEX IF NOT EXISTS idx_us_public     ON user_strategies(is_public)",
            "CREATE INDEX IF NOT EXISTS idx_sm_featured   ON strategy_marketplace(is_featured, avg_rating DESC)",
        ]
        for sql in indexes:
            try:
                conn.execute(text("SET LOCAL lock_timeout = '2s'"))
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass

        # Weekly AI Trade Coach reports — cached one-per-(user, week) to keep
        # Claude Haiku spend bounded. JSONB so we can add fields without ALTERs.
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS weekly_coach_reports (
                    id           SERIAL PRIMARY KEY,
                    user_id      INTEGER NOT NULL,
                    week_start   DATE    NOT NULL,
                    report_json  JSONB   NOT NULL,
                    generated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE (user_id, week_start)
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_wcr_user_week "
                "ON weekly_coach_reports(user_id, week_start DESC)"
            ))
            conn.commit()
        except Exception:
            pass

    try:
        from app.services.strategy_account_assignments import migrate_legacy_strategy_assignments
        migrate_legacy_strategy_assignments(engine)
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(
            "init_strategy_tables: legacy assignment migrate: %s", type(exc).__name__
        )
