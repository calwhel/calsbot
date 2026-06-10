"""Tests for feed failover helpers and safe CoinGecko parsing."""
import pytest

from app.services.coingecko_safe import parse_coin_list, symbols_from_markets
from app.services.alpha_vantage_feed import _parse_time_series, _split_forex_pair


def test_parse_coin_list_none():
    assert parse_coin_list(None) == []


def test_parse_coin_list_error_dict():
    assert parse_coin_list({"status": {"error_code": 429}}) == []


def test_parse_coin_list_valid():
    data = [{"symbol": "btc"}, {"symbol": "eth"}, {"no_symbol": 1}]
    assert symbols_from_markets(data) == {"BTC", "ETH"}


def test_alpha_vantage_parse_time_series():
    body = {
        "Time Series FX (15min)": {
            "2024-01-01 12:00:00": {
                "1. open": "1.08",
                "2. high": "1.09",
                "3. low": "1.07",
                "4. close": "1.085",
            },
            "2024-01-01 12:15:00": {
                "1. open": "1.085",
                "2. high": "1.086",
                "3. low": "1.084",
                "4. close": "1.0855",
            },
        }
    }
    rows = _parse_time_series(body, 10)
    assert len(rows) == 2
    assert rows[-1][4] == pytest.approx(1.0855)


def test_split_forex_pair():
    assert _split_forex_pair("EURUSD") == ("EUR", "USD")
    assert _split_forex_pair("XAUUSD") == ("XAU", "USD")


def test_audit_ctrader_missing_env(monkeypatch):
    pytest.importorskip("ctrader_open_api")
    from app.services.ctrader_client import audit_ctrader_credentials
    monkeypatch.setattr("app.services.ctrader_client.CTRADER_CLIENT_ID", "")
    monkeypatch.setattr("app.services.ctrader_client.CTRADER_CLIENT_SECRET", "")
    out = audit_ctrader_credentials(1)
    assert out["ok"] is False
    assert "CLIENT_ID" in out["reason"]
