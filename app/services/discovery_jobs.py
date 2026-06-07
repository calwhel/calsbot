"""
Background jobs for tradfi discovery scans (gold / index / forex).

POST returns immediately; progress is polled via GET …/progress.
Job state is stored in Postgres so all gunicorn workers share the same status.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_JOB_TTL_SECS = 3600
_TASKS: Dict[str, asyncio.Task] = {}
_DISCOVERY_SEM: Optional[asyncio.Semaphore] = None


def _discovery_sem() -> asyncio.Semaphore:
    global _DISCOVERY_SEM
    if _DISCOVERY_SEM is None:
        _DISCOVERY_SEM = asyncio.Semaphore(int(os.environ.get("DISCOVERY_MAX_PARALLEL", "2")))
    return _DISCOVERY_SEM


def _job_key(scan_type: str, uid: str) -> str:
    return f"{scan_type}:{uid.strip()}"


def _db_session():
    from app.database import SessionLocal
    return SessionLocal()


def _row_to_progress(row) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": row.status or "idle",
        "message": row.message or "",
        "scan_type": row.scan_type,
    }
    if row.result_json is not None:
        if row.status == "done":
            out["result"] = row.result_json
        elif row.status == "running" and isinstance(row.result_json, dict):
            out.update(row.result_json)
    if row.status == "error" and row.error:
        out["error"] = row.error
    return out


def write_job_progress(scan_type: str, uid: str, *, message: str = None, partial: dict = None) -> None:
    """Persist in-progress scan state (streaming) for multi-worker polling."""
    key = _job_key(scan_type, uid)
    if not get_job(scan_type, uid):
        from app.strategy_models import DiscoveryScanJob
        db = _db_session()
        try:
            db.add(DiscoveryScanJob(
                job_key=key,
                scan_type=scan_type,
                uid=uid.strip(),
                status="running",
                message=message or "Starting…",
            ))
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()
    _write_job(key, status="running", message=message, result=partial)


def _prune_old_jobs() -> None:
    cutoff = datetime.utcnow() - timedelta(seconds=_JOB_TTL_SECS)
    db = _db_session()
    try:
        from app.strategy_models import DiscoveryScanJob
        db.query(DiscoveryScanJob).filter(
            DiscoveryScanJob.finished_at.isnot(None),
            DiscoveryScanJob.finished_at < cutoff,
        ).delete(synchronize_session=False)
        db.commit()
    except Exception as e:
        logger.debug(f"[discovery-job] prune failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


def get_job(scan_type: str, uid: str):
    from app.strategy_models import DiscoveryScanJob
    db = _db_session()
    try:
        return db.query(DiscoveryScanJob).filter(
            DiscoveryScanJob.job_key == _job_key(scan_type, uid)
        ).first()
    finally:
        db.close()


def job_progress(scan_type: str, uid: str) -> Dict[str, Any]:
    _prune_old_jobs()
    row = get_job(scan_type, uid)
    if not row:
        return {"status": "idle", "message": ""}
    return _row_to_progress(row)


def _write_job(key: str, *, status: str = None, message: str = None,
               result=None, error: str = None, finished: bool = False) -> None:
    from app.strategy_models import DiscoveryScanJob
    db = _db_session()
    try:
        row = db.query(DiscoveryScanJob).filter(DiscoveryScanJob.job_key == key).first()
        if not row:
            return
        if status is not None:
            row.status = status
        if message is not None:
            row.message = message
        if result is not None:
            row.result_json = result
        if error is not None:
            row.error = error
        if finished:
            row.finished_at = datetime.utcnow()
        row.updated_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        logger.warning(f"[discovery-job] write {key} failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


def start_discovery_job(
    scan_type: str,
    uid: str,
    runner: Callable[[Callable[[str], None]], Any],
) -> Dict[str, Any]:
    """
    Queue a discovery scan. Returns immediately; runner executes in background.
    """
    _prune_old_jobs()
    key = _job_key(scan_type, uid)
    existing = get_job(scan_type, uid)
    if existing and existing.status in ("queued", "running"):
        prog = _row_to_progress(existing)
        return {"ok": True, "started": False, "status": existing.status, **prog}

    from app.strategy_models import DiscoveryScanJob
    db = _db_session()
    try:
        row = db.query(DiscoveryScanJob).filter(DiscoveryScanJob.job_key == key).first()
        if row:
            row.status = "queued"
            row.message = "Queued…"
            row.result_json = None
            row.error = None
            row.finished_at = None
            row.updated_at = datetime.utcnow()
        else:
            row = DiscoveryScanJob(
                job_key=key,
                scan_type=scan_type,
                uid=uid.strip(),
                status="queued",
                message="Queued…",
            )
            db.add(row)
        db.commit()
    except Exception as e:
        logger.exception(f"[discovery-job] create {key} failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return {"ok": False, "error": f"Could not start scan: {type(e).__name__}"}
    finally:
        db.close()

    async def _wrapped() -> None:
        _write_job(key, status="running", message="Starting scan…")

        def _progress(msg: str) -> None:
            _write_job(key, message=msg)

        try:
            async with _discovery_sem():
                result = await runner(_progress)
            if isinstance(result, dict) and result.get("ok"):
                _write_job(key, status="done", message="Scan complete",
                           result=result, finished=True)
            else:
                err = (
                    (result or {}).get("error")
                    or (result or {}).get("message")
                    or "Scan failed"
                )
                _write_job(key, status="error", message=err, error=err,
                           result=result if isinstance(result, dict) else None,
                           finished=True)
        except Exception as e:
            logger.exception(f"[discovery-job] {scan_type} failed for {uid}")
            err = f"{type(e).__name__}: {e}"
            _write_job(key, status="error", message=err, error=err, finished=True)
        finally:
            _TASKS.pop(key, None)

    if key in _TASKS and not _TASKS[key].done():
        return {"ok": True, "started": True, "status": "running", "message": "Queued…"}

    try:
        task = asyncio.get_running_loop().create_task(_wrapped())
    except RuntimeError:
        task = asyncio.get_event_loop().create_task(_wrapped())
    _TASKS[key] = task
    return {"ok": True, "started": True, "status": "running", "message": "Queued…"}
