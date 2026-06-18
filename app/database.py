import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Optional
import contextvars
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from app.advisory_lock_ids import APP_NAME_EXECUTOR, APP_NAME_WEB
from app.config import settings
from app.db_resilience import is_transient_db_error, run_with_db_retry
import logging
import os
import time

logger = logging.getLogger(__name__)

Base = declarative_base()

_db_url = settings.get_database_url()


def _neon_connect_args(
    statement_timeout_ms: int,
    *,
    application_name: str,
    lock_timeout_ms: int = 0,
    keepalives_idle_s: int = 30,
    keepalives_interval_s: int = 10,
    keepalives_count: int = 5,
) -> dict:
    opts = [
        f"-c tcp_keepalives_idle={int(keepalives_idle_s)}",
        f"-c statement_timeout={max(1000, int(statement_timeout_ms))}",
    ]
    if int(lock_timeout_ms) > 0:
        opts.append(f"-c lock_timeout={int(lock_timeout_ms)}")
    args: dict = {
        "connect_timeout": 30,
        "options": " ".join(opts),
    }
    if "neon" in _db_url or "neondb" in _db_url:
        # Neon closes idle SSL connections after ~5 minutes.
        # TCP keepalives + pool_recycle=180 keep sockets warm under Neon's cutoff.
        args["sslmode"] = "require"
        args["keepalives"] = 1
        args["keepalives_idle"] = int(keepalives_idle_s)
        args["keepalives_interval"] = int(keepalives_interval_s)
        args["keepalives_count"] = int(keepalives_count)
    args["application_name"] = application_name
    return args


_WEB_DB_STATEMENT_TIMEOUT_MS = int(os.getenv("WEB_DB_STATEMENT_TIMEOUT_MS", "60000"))
_WEB_DB_LOCK_TIMEOUT_MS = int(os.getenv("WEB_DB_LOCK_TIMEOUT_MS", "0"))
_WEB_DB_KEEPALIVES_IDLE_S = int(os.getenv("WEB_DB_KEEPALIVES_IDLE_S", "30"))
_WEB_DB_KEEPALIVES_INTERVAL_S = int(os.getenv("WEB_DB_KEEPALIVES_INTERVAL_S", "10"))
_WEB_DB_KEEPALIVES_COUNT = int(os.getenv("WEB_DB_KEEPALIVES_COUNT", "5"))

_BG_DB_STATEMENT_TIMEOUT_MS = int(os.getenv("BG_DB_STATEMENT_TIMEOUT_MS", "9000"))
_BG_DB_LOCK_TIMEOUT_MS = int(os.getenv("BG_DB_LOCK_TIMEOUT_MS", "3000"))
_BG_DB_KEEPALIVES_IDLE_S = int(os.getenv("BG_DB_KEEPALIVES_IDLE_S", "15"))
_BG_DB_KEEPALIVES_INTERVAL_S = int(os.getenv("BG_DB_KEEPALIVES_INTERVAL_S", "5"))
_BG_DB_KEEPALIVES_COUNT = int(os.getenv("BG_DB_KEEPALIVES_COUNT", "3"))


# ─── HTTP engine ─────────────────────────────────────────────────────────────
# Serves API + page requests. statement_timeout=60 s — production showed even
# trivial indexed PK lookups on `users.uid` getting QueryCancelled at 20s
# because Neon's serverless compute was throttled by concurrent executor +
# AI generator load. 60s is high but it ONLY matters for the worst-case
# cold-wake — typical queries finish in <50 ms. The user-row cache layer
# (`_USER_CACHE` in strategy_portal_server.py) means each user only hits this
# query once per 30 s, so the timeout ceiling is the failsafe, not the norm.
# Pool sizes also bumped (3→6 base, 4→8 overflow) so a slow query can't
# starve other requests of a connection.
# pool_timeout=30: give requests 30 s to acquire a connection under burst load
# instead of the previous 10 s which caused spurious pool-exhaustion errors.
engine = create_engine(
    _db_url,
    poolclass=QueuePool,
    pool_size=6,
    max_overflow=8,
    pool_timeout=30,
    pool_recycle=180,
    pool_pre_ping=True,
    connect_args=_neon_connect_args(
        _WEB_DB_STATEMENT_TIMEOUT_MS,
        application_name=APP_NAME_WEB,
        lock_timeout_ms=_WEB_DB_LOCK_TIMEOUT_MS,
        keepalives_idle_s=_WEB_DB_KEEPALIVES_IDLE_S,
        keepalives_interval_s=_WEB_DB_KEEPALIVES_INTERVAL_S,
        keepalives_count=_WEB_DB_KEEPALIVES_COUNT,
    ),
)


# ─── Background engine ───────────────────────────────────────────────────────
# Used by strategy_executor + ai_strategy_generator. Completely isolated pool
# so background scans CANNOT starve HTTP requests of connections — which was
# the root cause of /app, /api/strategies, /api/portfolio all timing out
# while the executor held every connection waiting on slow queries. Keeps
# the 60 s statement_timeout the executor needs for heavy analytical work.
_bg_pool_size = int(os.getenv("BG_POOL_SIZE", "8"))
_bg_pool_overflow = int(os.getenv("BG_POOL_OVERFLOW", "10"))
BG_DB_RESERVE = int(os.getenv("BG_DB_RESERVE", "5"))
bg_engine = create_engine(
    _db_url,
    # Isolated from HTTP pool. Peak load ≈ FOREX_MAX_CONCURRENT + MAX_CONCURRENT
    # (eval tasks hold a session across async kline/TA fetches) + live/paper
    # monitors + gate-stats flush. Defaults 8+10=18; tune via BG_POOL_* env.
    poolclass=QueuePool,
    pool_size=_bg_pool_size,
    max_overflow=_bg_pool_overflow,
    pool_timeout=int(os.getenv("BG_POOL_TIMEOUT", "30")),
    pool_recycle=180,
    pool_pre_ping=True,
    connect_args=_neon_connect_args(
        _BG_DB_STATEMENT_TIMEOUT_MS,
        application_name=APP_NAME_EXECUTOR,
        lock_timeout_ms=_BG_DB_LOCK_TIMEOUT_MS,
        keepalives_idle_s=_BG_DB_KEEPALIVES_IDLE_S,
        keepalives_interval_s=_BG_DB_KEEPALIVES_INTERVAL_S,
        keepalives_count=_BG_DB_KEEPALIVES_COUNT,
    ),
)


def _register_pool_events(_eng):
    @event.listens_for(_eng, "checkin")
    def _on_checkin(dbapi_connection, connection_record):
        try:
            dbapi_connection.rollback()
        except Exception:
            pass

    @event.listens_for(_eng, "checkout")
    def _on_checkout(dbapi_connection, connection_record, connection_proxy):
        try:
            dbapi_connection.rollback()
        except Exception:
            pass

    @event.listens_for(_eng, "handle_error")
    def _on_handle_error(exception_context):
        if is_transient_db_error(exception_context.original_exception):
            exception_context.is_disconnect = True


for _eng in (engine, bg_engine):
    _register_pool_events(_eng)


def bg_engine_runtime_profile() -> dict:
    """Runtime-safe diagnostics for executor DB engine health guards."""
    return {
        "statement_timeout_ms": int(_BG_DB_STATEMENT_TIMEOUT_MS),
        "lock_timeout_ms": int(_BG_DB_LOCK_TIMEOUT_MS),
        "keepalives_idle_s": int(_BG_DB_KEEPALIVES_IDLE_S),
        "keepalives_interval_s": int(_BG_DB_KEEPALIVES_INTERVAL_S),
        "keepalives_count": int(_BG_DB_KEEPALIVES_COUNT),
        "pool_pre_ping": bool(getattr(bg_engine.pool, "_pre_ping", False)),
        "pool_recycle_s": 180,
        "pool_size": int(_bg_pool_size),
        "pool_max_overflow": int(_bg_pool_overflow),
        "pool_hard_limit": int(bg_pool_hard_limit()),
        "bg_db_reserve": int(BG_DB_RESERVE),
        "bg_db_slot_limit": int(bg_db_slot_limit()),
    }


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
BgSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=bg_engine)

_bg_db_sem: Optional[asyncio.Semaphore] = None
_bg_slot_wait_ms: contextvars.ContextVar[float] = contextvars.ContextVar(
    "bg_slot_wait_ms", default=0.0,
)


def last_bg_db_slot_wait_ms() -> float:
    """Wait time (ms) for the most recent bg_db_slot acquire in this task."""
    return float(_bg_slot_wait_ms.get())


def bg_pool_hard_limit() -> int:
    return _bg_pool_size + _bg_pool_overflow


def bg_db_slot_limit() -> int:
    """Max concurrent bg_engine checkouts across executor + monitors."""
    explicit = os.getenv("BG_DB_MAX_CONCURRENT", "").strip()
    if explicit:
        return max(2, int(explicit))
    return max(4, bg_pool_hard_limit() - BG_DB_RESERVE)


def _get_bg_db_sem() -> asyncio.Semaphore:
    global _bg_db_sem
    if _bg_db_sem is None:
        _bg_db_sem = asyncio.Semaphore(bg_db_slot_limit())
    return _bg_db_sem


@asynccontextmanager
async def bg_db_slot():
    """Gate background DB usage so scans cannot starve monitors of pool slots."""
    sem = _get_bg_db_sem()
    t0 = time.monotonic()
    await sem.acquire()
    wait_ms = (time.monotonic() - t0) * 1000.0
    _bg_slot_wait_ms.set(wait_ms)
    if wait_ms > 500:
        logger.warning(
            "[cycle-db] bg_db_slot wait=%.0fms (pool contended, limit=%s)",
            wait_ms,
            bg_db_slot_limit(),
        )
    try:
        yield
    finally:
        sem.release()


@contextmanager
def get_safe_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Full init — schema migrations + UID backfill. Slow; should only be
    called by the strategy_portal process. Bot + tracker should use
    init_db_minimal() so they don't repeatedly hammer Neon with full table
    scans on every boot (which can take 60 s+ and saturate the DB)."""
    import os as _os
    if _os.environ.get("SKIP_HEAVY_MIGRATIONS", "").lower() in ("1", "true", "yes"):
        logger.info("init_db: SKIP_HEAVY_MIGRATIONS=1 — skipping ensure_columns + backfill_user_uids")
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            logger.warning(f"init_db (create_all only): {e}")
        return
    ensure_columns()
    Base.metadata.create_all(bind=engine)
    backfill_user_uids()


def init_db_minimal():
    """Minimal init for sidecar processes (bot, tracker) — just verifies the
    DB connection is alive. The portal owns all schema migrations."""
    from sqlalchemy import text as _text
    try:
        with engine.connect() as c:
            c.execute(_text("SELECT 1"))
        logger.info("init_db_minimal: DB reachable")
    except Exception as e:
        logger.warning(f"init_db_minimal: DB unreachable: {e}")


def ensure_columns():
    from sqlalchemy import text, inspect as sa_inspect
    db = SessionLocal()
    try:
        inspector = sa_inspect(engine)
        
        type_map = {
            'String': 'VARCHAR',
            'Text': 'TEXT',
            'Integer': 'INTEGER',
            'Float': 'FLOAT',
            'Boolean': 'BOOLEAN',
            'DateTime': 'TIMESTAMP',
        }
        
        for table_class in Base.__subclasses__():
            table_name = table_class.__tablename__
            
            if not inspector.has_table(table_name):
                continue
            
            existing_cols = {col['name'] for col in inspector.get_columns(table_name)}
            
            for attr_name in dir(table_class):
                attr = getattr(table_class, attr_name, None)
                if attr is None:
                    continue
                if not hasattr(attr, 'property'):
                    continue
                try:
                    from sqlalchemy.orm.properties import ColumnProperty
                    if not isinstance(attr.property, ColumnProperty):
                        continue
                    col = attr.property.columns[0]
                    col_name = col.name
                    if col_name in existing_cols:
                        continue
                    
                    col_type_name = type(col.type).__name__
                    sql_type = type_map.get(col_type_name, 'VARCHAR')
                    
                    default_clause = ""
                    if col.default is not None and col.default.arg is not None and not callable(col.default.arg):
                        default_val = col.default.arg
                        if isinstance(default_val, bool):
                            default_clause = f" DEFAULT {'TRUE' if default_val else 'FALSE'}"
                        elif isinstance(default_val, (int, float)):
                            default_clause = f" DEFAULT {default_val}"
                        elif isinstance(default_val, str):
                            default_clause = f" DEFAULT '{default_val}'"
                    
                    unique_clause = " UNIQUE" if col.unique else ""
                    
                    alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {sql_type}{default_clause}{unique_clause}"
                    try:
                        db.execute(text(alter_sql))
                        db.commit()
                        logger.info(f"Migration: Added {table_name}.{col_name} ({sql_type})")
                    except Exception as e:
                        db.rollback()
                        logger.warning(f"Migration failed for {table_name}.{col_name}: {e}")
                except Exception:
                    continue
    except Exception as e:
        db.rollback()
        logger.error(f"ensure_columns error: {e}")
    finally:
        db.close()


def backfill_user_uids():
    from sqlalchemy import text
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT id FROM users WHERE uid IS NULL")).fetchall()
        if not result:
            return
        
        import random, string
        for (user_id,) in result:
            while True:
                chars = string.ascii_uppercase + string.digits
                uid = "TH-" + "".join(random.choices(chars, k=8))
                existing = db.execute(text("SELECT 1 FROM users WHERE uid = :uid"), {"uid": uid}).fetchone()
                if not existing:
                    break
            db.execute(text("UPDATE users SET uid = :uid WHERE id = :id"), {"uid": uid, "id": user_id})
        db.commit()
        logger.info(f"Backfilled UIDs for {len(result)} users")
    except Exception as e:
        db.rollback()
        logger.error(f"UID backfill error: {e}")
    finally:
        db.close()
