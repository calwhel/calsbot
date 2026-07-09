"""cTrader account helpers for Gemini Gold Trader (demo + live pickers)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def parse_ctrader_accounts(raw) -> List[dict]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    try:
        parsed = json.loads(raw)
        return [a for a in parsed if isinstance(a, dict)] if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def is_confirmed_demo_account(acct: dict) -> bool:
    return acct.get("isLive") is False


def is_confirmed_live_account(acct: dict) -> bool:
    return acct.get("isLive") is True


def demo_account_label(acct: dict) -> str:
    ctid = acct.get("ctidTraderAccountId")
    login = acct.get("traderLogin")
    name = acct.get("accountName") or acct.get("brokerName") or acct.get("description")
    parts = [f"#{ctid}"]
    if login:
        parts.append(f"login {login}")
    elif name:
        parts.append(str(name))
    parts.append("Demo")
    return " · ".join(parts)


def live_account_label(acct: dict) -> str:
    ctid = acct.get("ctidTraderAccountId")
    login = acct.get("traderLogin")
    name = acct.get("accountName") or acct.get("brokerName") or acct.get("description")
    parts = [f"#{ctid}"]
    if login:
        parts.append(f"login {login}")
    elif name:
        parts.append(str(name))
    parts.append("Live")
    return " · ".join(parts)


def demo_accounts_from_prefs(prefs) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for acct in parse_ctrader_accounts(getattr(prefs, "ctrader_accounts", None) if prefs else None):
        if not is_confirmed_demo_account(acct):
            continue
        ctid = acct.get("ctidTraderAccountId")
        if ctid is None:
            continue
        out.append(
            {
                "ctid": str(ctid),
                "label": demo_account_label(acct),
                "trader_login": acct.get("traderLogin"),
                "balance": acct.get("balance"),
            }
        )
    return out


def live_accounts_from_prefs(prefs) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for acct in parse_ctrader_accounts(getattr(prefs, "ctrader_accounts", None) if prefs else None):
        if not is_confirmed_live_account(acct):
            continue
        ctid = acct.get("ctidTraderAccountId")
        if ctid is None:
            continue
        out.append(
            {
                "ctid": str(ctid),
                "label": live_account_label(acct),
                "trader_login": acct.get("traderLogin"),
                "balance": acct.get("balance"),
            }
        )
    return out


def accounts_for_user_id(db, user_id: Optional[int]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not user_id:
        return [], []
    from app.models import UserPreference

    prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    return demo_accounts_from_prefs(prefs), live_accounts_from_prefs(prefs)


def demo_accounts_for_user_id(db, user_id: Optional[int]) -> List[Dict[str, Any]]:
    demos, _ = accounts_for_user_id(db, user_id)
    return demos


def live_accounts_for_user_id(db, user_id: Optional[int]) -> List[Dict[str, Any]]:
    _, lives = accounts_for_user_id(db, user_id)
    return lives


def find_account(accounts: List[Dict[str, Any]], ctid: Optional[str]) -> Optional[Dict[str, Any]]:
    if not ctid:
        return None
    want = str(ctid).strip()
    for row in accounts:
        if str(row.get("ctid")) == want:
            return row
    return None


def find_demo_account(accounts: List[Dict[str, Any]], ctid: Optional[str]) -> Optional[Dict[str, Any]]:
    return find_account(accounts, ctid)


def find_live_account(accounts: List[Dict[str, Any]], ctid: Optional[str]) -> Optional[Dict[str, Any]]:
    return find_account(accounts, ctid)


def cached_balance_for_ctid(prefs, ctid: Optional[str]) -> Optional[float]:
    """Last-known balance from linked ctrader_accounts JSON (portal sync)."""
    if not ctid:
        return None
    want = str(ctid).strip()
    for acct in parse_ctrader_accounts(getattr(prefs, "ctrader_accounts", None) if prefs else None):
        if str(acct.get("ctidTraderAccountId") or "").strip() != want:
            continue
        raw = acct.get("balance")
        if raw is None:
            return None
        try:
            return round(float(raw), 2)
        except (TypeError, ValueError):
            return None
    return None


def validate_demo_ctid_allowed(accounts: List[Dict[str, Any]], ctid: str) -> None:
    if not find_demo_account(accounts, ctid):
        raise ValueError(f"ctid {ctid} is not a connected demo account (isLive=false)")


def validate_live_ctid_allowed(accounts: List[Dict[str, Any]], ctid: str) -> None:
    if not find_live_account(accounts, ctid):
        raise ValueError(f"ctid {ctid} is not a connected live account (isLive=true)")
