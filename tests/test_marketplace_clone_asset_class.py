"""Marketplace clone — preserve asset_class for forex/cTrader subscribers."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

from types import SimpleNamespace

from strategy_portal_server import (
    _locked_marketplace_clone_config,
    _scale_clone_risk,
    _strategy_asset_class,
)


def _strat(config=None, asset_class="crypto"):
    return SimpleNamespace(config=config or {}, asset_class=asset_class)


def _listing(title="Gold Scalp", strategy_id=1):
    return SimpleNamespace(title=title, strategy_id=strategy_id)


def test_scale_clone_risk_scales_pct_only():
    risk = _scale_clone_risk({"position_size_pct": 10, "leverage": 20}, 0.5)
    assert risk["position_size_pct"] == 5.0
    assert risk["leverage"] == 20


def test_locked_clone_config_preserves_forex_asset_class():
    original = _strat(
        config={
            "asset_class": "forex",
            "direction": "LONG",
            "risk": {"position_size_pct": 4, "leverage": 10},
            "exit": {"stop_loss_pct": 1},
            "filters": {},
            "universe": {"symbols": ["EURUSD"]},
        },
        asset_class="forex",
    )
    listing = _listing()
    locked, risk, ac = _locked_marketplace_clone_config(original, listing, 42, 1.0)
    assert ac == "forex"
    assert locked["asset_class"] == "forex"
    assert locked["_asset_class"] == "forex"
    assert locked["_locked"] is True
    assert locked["_listing_id"] == 42
    assert risk["position_size_pct"] == 4


def test_strategy_asset_class_preserves_metals_column():
    original = _strat(config={}, asset_class="metals")
    assert _strategy_asset_class(original) == "metals"


def test_strategy_asset_class_prefers_config_when_column_stale():
    original = _strat(config={"asset_class": "forex"}, asset_class="crypto")
    assert _strategy_asset_class(original) == "forex"
