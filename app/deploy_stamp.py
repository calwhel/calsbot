"""Deploy verification — log commit hash so production logs prove which build is running."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_deploy_commit() -> str:
    for key in (
        "RAILWAY_GIT_COMMIT_SHA",
        "RAILWAY_GIT_COMMIT",
        "GIT_COMMIT",
        "GITHUB_SHA",
    ):
        val = (os.getenv(key) or "").strip()
        if val:
            return val[:12]
    return "unknown"


def log_deploy_stamp(process: str) -> None:
    logger.info(
        "[deploy] process=%s commit=%s",
        process,
        get_deploy_commit(),
    )
