---
name: Gunicorn flag names
description: Correct flag spellings for this gunicorn version — easy to get wrong.
---

## Rule
The keepalive flag is `--keep-alive INT` (hyphenated), NOT `--keepalive`.

**Why:** Gunicorn 26 uses the hyphenated form. `--keepalive` causes a hard startup failure ("unrecognized arguments") discovered when updating the Strategy Portal workflow.

**How to apply:** Always run `gunicorn --help | grep <flag>` before adding a new flag. The current working command is:
```
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --max-requests 1000 --max-requests-jitter 100 --keep-alive 5 --bind 0.0.0.0:5000 --timeout 120 --graceful-timeout 30 --log-level warning strategy_portal_server:app
```
