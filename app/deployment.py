"""Deployment environment helpers (Railway, Replit, local)."""
from __future__ import annotations

import os


def is_railway() -> bool:
  return bool(
    os.getenv("RAILWAY_ENVIRONMENT")
    or os.getenv("RAILWAY_GIT_COMMIT_SHA")
    or os.getenv("RAILWAY_SERVICE_ID")
  )


def is_replit() -> bool:
  return os.getenv("REPL_DEPLOYMENT") == "1" or bool(os.getenv("REPLIT_DEPLOYMENT"))


def is_production_deploy() -> bool:
  """True when this process should run production-only background work (executor, bots)."""
  if os.getenv("FORCE_EXECUTOR", "").lower() in ("1", "true", "yes"):
    return True
  if os.getenv("REPL_DEPLOYMENT") == "1":
    return True
  if is_railway():
    return True
  if os.getenv("REPLIT_DEPLOYMENT"):
    return True
  return False


def should_poll_telegram() -> bool:
  if os.getenv("DISABLE_TELEGRAM_POLL", "").lower() in ("1", "true", "yes"):
    return False
  if os.getenv("FORCE_BOT_POLL", "").lower() in ("1", "true", "yes"):
    return True
  if os.getenv("REPLIT_DEPLOYMENT"):
    return True
  if is_railway() and os.getenv("TELEGRAM_BOT_TOKEN"):
    return True
  return False


def portal_features_free() -> bool:
  """When True, every user gets Pro-tier portal features (no OxaPay gate)."""
  explicit = os.getenv("PORTAL_FEATURES_FREE", "").strip().lower()
  if explicit in ("1", "true", "yes"):
    return True
  if explicit in ("0", "false", "no"):
    return False
  if is_railway():
    return True
  try:
    from app.config import settings
    if not settings.OXAPAY_MERCHANT_API_KEY:
      return True
  except Exception:
    return True
  return False


def payments_enabled() -> bool:
  if portal_features_free():
    return False
  try:
    from app.config import settings
    return bool(settings.OXAPAY_MERCHANT_API_KEY)
  except Exception:
    return False


def google_auth_enabled() -> bool:
  return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


def request_is_https(request=None) -> bool:
  if is_railway() or os.getenv("REPL_DEPLOYMENT") == "1":
    return True
  if request is not None:
    proto = (request.headers.get("x-forwarded-proto") or "").lower()
    if proto == "https":
      return True
  return False


def _host_from_urlish(raw: str) -> str:
  raw = (raw or "").strip()
  if not raw:
    return ""
  if "://" in raw:
    from urllib.parse import urlparse
    return (urlparse(raw).netloc or "").split(":")[0].strip().lower()
  return raw.split(",")[0].strip().split(":")[0].strip().lower()


def _railway_hostname_from_values(*values: str) -> str:
  for raw in values:
    host = _host_from_urlish(raw)
    if host.endswith(".up.railway.app") or host.endswith(".railway.app"):
      return host
  return ""


_railway_hostname_cache: str | None = None


def railway_app_hostname() -> str:
  """*.up.railway.app host for this service — never a custom/marketing domain."""
  global _railway_hostname_cache
  if _railway_hostname_cache is not None:
    return _railway_hostname_cache

  found = _railway_hostname_from_values(
    os.getenv("CTRADER_RAILWAY_HOST") or "",
    os.getenv("CTRADER_REDIRECT_URI") or "",
    os.getenv("RAILWAY_STATIC_URL") or "",
    os.getenv("RAILWAY_PUBLIC_DOMAIN") or "",
    os.getenv("WEBHOOK_BASE_URL") or "",
  )
  if not found:
    for key, val in os.environ.items():
      if not val or ".railway.app" not in val:
        continue
      if any(tag in key.upper() for tag in ("RAILWAY", "CTRADER", "WEBHOOK", "PUBLIC")):
        found = _railway_hostname_from_values(val)
        if found:
          break
  if not found:
    for val in os.environ.values():
      if val and ".up.railway.app" in val:
        found = _railway_hostname_from_values(val)
        if found:
          break

  _railway_hostname_cache = found
  return found


def railway_service_base_url(request=None) -> str:
  """Railway deploy URL — ignores custom domains (tradehubmarkets.com) on request Host."""
  host = railway_app_hostname()
  if host:
    return f"https://{host}"
  if request is not None:
    req_host = (request.headers.get("host") or "").split(":")[0].strip().lower()
    if req_host.endswith(".up.railway.app") or req_host.endswith(".railway.app"):
      return f"https://{req_host}"
  return ""


def public_base_url(request=None) -> str:
  for key in ("PUBLIC_DOMAIN", "RAILWAY_PUBLIC_DOMAIN", "RAILWAY_STATIC_URL"):
    raw = (os.getenv(key) or "").strip()
    if not raw:
      continue
    if raw.startswith("http"):
      return raw.rstrip("/")
    return f"https://{raw.split(',')[0].strip()}"
  if request is not None:
    return str(request.base_url).rstrip("/")
  return "https://tradehubmarkets.com"


def deploy_commit() -> str:
  return (
    os.getenv("RAILWAY_GIT_COMMIT_SHA")
    or os.getenv("GIT_COMMIT")
    or "unknown"
  )
