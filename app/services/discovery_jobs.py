"""
In-process background jobs for tradfi discovery scans (gold / index / forex).

POST /api/backtest/*-discovery returns immediately; the client polls
/api/backtest/*-discovery/progress until status is done|error.

This avoids gunicorn HTTP timeouts on 2–5 minute Claude + backtest runs.
Jobs are keyed by scan_type + uid (one active scan per type per user).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_JOB_TTL_SECS = 3600  # keep finished results for 1h so slow clients can fetch


@dataclass
class DiscoveryJob:
    scan_type: str
    uid: str
    status: str = "queued"          # queued | running | done | error
    message: str = ""
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_progress(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": self.status,
            "message": self.message or "",
            "scan_type": self.scan_type,
        }
        if self.status == "done" and self.result is not None:
            out["result"] = self.result
        if self.status == "error" and self.error:
            out["error"] = self.error
        return out


_JOBS: Dict[str, DiscoveryJob] = {}
_TASKS: Dict[str, asyncio.Task] = {}


def _job_key(scan_type: str, uid: str) -> str:
    return f"{scan_type}:{uid.strip()}"


def _prune_old_jobs() -> None:
    now = time.time()
    stale = [
        k for k, j in _JOBS.items()
        if j.finished_at and (now - j.finished_at) > _JOB_TTL_SECS
    ]
    for k in stale:
        _JOBS.pop(k, None)
        _TASKS.pop(k, None)


def get_job(scan_type: str, uid: str) -> Optional[DiscoveryJob]:
    _prune_old_jobs()
    return _JOBS.get(_job_key(scan_type, uid))


def job_progress(scan_type: str, uid: str) -> Dict[str, Any]:
    job = get_job(scan_type, uid)
    if not job:
        return {"status": "idle", "message": ""}
    return job.to_progress()


def start_discovery_job(
    scan_type: str,
    uid: str,
    runner: Callable[[Callable[[str], None]], Any],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Dict[str, Any]:
    """
    Start a discovery scan in the background if none is running for this uid+type.
    `runner` is an async callable: async def run(progress_cb) -> dict
    """
    _prune_old_jobs()
    key = _job_key(scan_type, uid)
    existing = _JOBS.get(key)
    if existing and existing.status in ("queued", "running"):
        return {"ok": True, "started": False, "status": existing.status, **existing.to_progress()}

    job = DiscoveryJob(scan_type=scan_type, uid=uid, status="queued", message="Queued…")
    _JOBS[key] = job

    async def _wrapped() -> None:
        job.status = "running"
        job.message = "Starting scan…"

        def _progress(msg: str) -> None:
            job.message = msg

        try:
            result = await runner(_progress)
            if isinstance(result, dict) and result.get("ok"):
                job.status = "done"
                job.result = result
                job.message = "Scan complete"
            else:
                job.status = "error"
                job.error = (
                    (result or {}).get("error")
                    or (result or {}).get("message")
                    or "Scan failed"
                )
                job.message = job.error
                job.result = result if isinstance(result, dict) else None
        except Exception as e:
            logger.exception(f"[discovery-job] {scan_type} failed for {uid}")
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.message = job.error
        finally:
            job.finished_at = time.time()
            _TASKS.pop(key, None)

    try:
        task = asyncio.get_running_loop().create_task(_wrapped())
    except RuntimeError:
        ev = loop or asyncio.get_event_loop()
        task = ev.create_task(_wrapped())
    _TASKS[key] = task
    return {"ok": True, "started": True, "status": "running", "message": job.message}
