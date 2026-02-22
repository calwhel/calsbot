from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

engine = create_engine(
    settings.get_database_url(),
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=60,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 30,
        "options": "-c statement_timeout=60000"
    }
)


@event.listens_for(engine, "checkin")
def _on_checkin(dbapi_connection, connection_record):
    try:
        dbapi_connection.rollback()
    except Exception:
        pass


@event.listens_for(engine, "checkout")
def _on_checkout(dbapi_connection, connection_record, connection_proxy):
    try:
        dbapi_connection.rollback()
    except Exception:
        pass


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    ensure_columns()
    Base.metadata.create_all(bind=engine)
    backfill_user_uids()


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
