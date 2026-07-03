"""Gemini Gold account picker helpers."""
from app.gemini_gold_trader.accounts import (
    demo_accounts_from_prefs,
    find_demo_account,
    find_live_account,
    live_accounts_from_prefs,
    validate_demo_ctid_allowed,
    validate_live_ctid_allowed,
)


class _Prefs:
    ctrader_accounts = (
        '[{"ctidTraderAccountId": 111, "isLive": false}, '
        '{"ctidTraderAccountId": 222, "isLive": true}]'
    )


def test_demo_accounts_filters_live():
    accounts = demo_accounts_from_prefs(_Prefs())
    assert len(accounts) == 1
    assert accounts[0]["ctid"] == "111"


def test_live_accounts_filters_demo():
    accounts = live_accounts_from_prefs(_Prefs())
    assert len(accounts) == 1
    assert accounts[0]["ctid"] == "222"


def test_find_live_account():
    accounts = live_accounts_from_prefs(_Prefs())
    assert find_live_account(accounts, "222")["ctid"] == "222"
    assert find_live_account(accounts, "999") is None


def test_validate_live_ctid_allowed():
    accounts = live_accounts_from_prefs(_Prefs())
    validate_live_ctid_allowed(accounts, "222")
    try:
        validate_live_ctid_allowed(accounts, "111")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_validate_demo_ctid_allowed():
    accounts = demo_accounts_from_prefs(_Prefs())
    validate_demo_ctid_allowed(accounts, "111")
    try:
        validate_demo_ctid_allowed(accounts, "222")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_find_demo_account():
    accounts = demo_accounts_from_prefs(_Prefs())
    assert find_demo_account(accounts, "111")["ctid"] == "111"
    assert find_demo_account(accounts, "999") is None
