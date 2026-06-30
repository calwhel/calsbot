"""Gemini Gold account picker helpers."""
from app.gemini_gold_trader.accounts import (
    demo_accounts_from_prefs,
    find_demo_account,
    validate_demo_ctid_allowed,
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
