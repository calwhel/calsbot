"""Minimal entrypoint for start.sh — checks env before importing database."""
from __future__ import annotations

import os


def main() -> int:
    value = os.environ.get("SCHEMA_RESET_ONCE", "").strip().lower()
    if value not in ("1", "true", "yes", "on"):
        return 0
    from app.schema_bootstrap import run_schema_reset_once_if_env

    run_schema_reset_once_if_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
