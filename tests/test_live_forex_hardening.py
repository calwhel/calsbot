"""Live Forex tab must not HTTP 500 on partial/malformed data."""
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient
    import strategy_portal_server as sps
    _HAS_APP = True
except Exception:
    _HAS_APP = False


class Row(dict):
  def __init__(self, **kw):
    super().__init__(**kw)
    self.__dict__ = self

  @property
  def _mapping(self):
    return self


@unittest.skipUnless(_HAS_APP, "strategy_portal_server unavailable")
class TestLiveForexHardening(unittest.TestCase):
  def setUp(self):
    self.client = TestClient(sps.app, raise_server_exceptions=False)

  def _mock_db(self, *, accounts_raw="[]", live_rows=None, pos_fail=False):
    user = MagicMock()
    user.id = 1
    user.is_admin = False
    prefs = MagicMock()
    prefs.ctrader_access_token = "tok"
    prefs.ctrader_refresh_token = "rtok"
    prefs.ctrader_account_id = "12345"
    prefs.forex_approved = True
    prefs.ctrader_accounts = accounts_raw
    db = MagicMock()
    rows = live_rows or [
      Row(
        id=1, name="EUR Strat", status="active", asset_class="forex",
        config='{"risk":"bad"}',
        open_count=0, closed_count=3, win_count=2, total_pnl_pct=1.2,
        total_pips_pnl=8.5, last_fired_at=datetime.now(timezone.utc),
      ),
    ]

    def execute_side_effect(sql, params=None):
      from sqlalchemy.exc import ProgrammingError
      q = str(sql)
      result = MagicMock()
      if "ORDER BY e.fired_at DESC" in q:
        if pos_fail:
          raise ProgrammingError("x", {}, Exception("column current_sl does not exist"))
        result.fetchall.return_value = []
      elif "user_strategies" in q and "GROUP BY s.id" in q:
        result.fetchall.return_value = rows
      elif "DATE_TRUNC" in q:
        result.fetchall.return_value = []
      elif "week_pips" in q:
        result.fetchone.return_value = Row(week_pips=0)
      else:
        result.fetchall.return_value = []
        result.fetchone.return_value = None
      return result

    db.execute.side_effect = execute_side_effect
    db.query.return_value.filter.return_value.first.return_value = prefs
    return user, db

  def test_bad_ctrader_accounts_json_returns_200(self):
    user, db = self._mock_db(accounts_raw="not-json")
    with patch("strategy_portal_server._get_user_by_uid", return_value=user), \
         patch("app.database.SessionLocal", return_value=db), \
         patch("strategy_portal_server.get_cache", return_value=None), \
         patch("strategy_portal_server.set_cache"), \
         patch("app.services.ctrader_price_feed.feed_status", return_value={"live": True, "symbol_count": 1}), \
         patch("app.services.ctrader_client.get_account_balance_resilient", return_value=1000.0):
      r = self.client.get("/api/live-forex/account?uid=TEST&refresh=true")
    self.assertEqual(r.status_code, 200)
    body = r.json()
    self.assertTrue(body.get("connected"))
    self.assertEqual(body.get("accounts"), [])

  def test_missing_position_columns_falls_back(self):
    user, db = self._mock_db(pos_fail=True)
    with patch("strategy_portal_server._get_user_by_uid", return_value=user), \
         patch("app.database.SessionLocal", return_value=db), \
         patch("strategy_portal_server.get_cache", return_value=None), \
         patch("strategy_portal_server.set_cache"), \
         patch("app.services.ctrader_price_feed.feed_status", return_value={"live": True, "symbol_count": 1}), \
         patch("app.services.ctrader_client.get_account_balance_resilient", return_value=1000.0):
      r = self.client.get("/api/live-forex/account?uid=TEST&refresh=true")
    self.assertEqual(r.status_code, 200)

  def test_malformed_strategy_config_renders_minimal_card(self):
    user, db = self._mock_db(live_rows=[
      Row(
        id=1, name="EUR Strat", status="active", asset_class="forex",
        config='{"risk":{"position_size_pct":"nope"}}',
        open_count=0, closed_count=3, win_count=2, total_pnl_pct=1.2,
        total_pips_pnl=8.5, last_fired_at=datetime.now(timezone.utc),
      ),
    ])
    with patch("strategy_portal_server._get_user_by_uid", return_value=user), \
         patch("app.database.SessionLocal", return_value=db), \
         patch("strategy_portal_server.get_cache", return_value=None), \
         patch("strategy_portal_server.set_cache"), \
         patch("app.services.ctrader_price_feed.feed_status", return_value={"live": True, "symbol_count": 1}), \
         patch("app.services.ctrader_client.get_account_balance_resilient", return_value=1000.0):
      r = self.client.get("/api/live-forex/account?uid=TEST&refresh=true")
    self.assertEqual(r.status_code, 200)
    cards = r.json().get("live_strategies") or []
    self.assertEqual(len(cards), 1)
    self.assertEqual(cards[0].get("size_label"), "—")
    self.assertTrue(cards[0].get("_card_error"))

  def test_build_strategy_card_never_raises(self):
    card = sps._lf_build_strategy_card(
      {
        "id": 9, "name": "X", "status": "active", "asset_class": "forex",
        "config": '{"risk":{"position_size_pct":"nope"}}',
      },
      {},
    )
    self.assertEqual(card["size_label"], "—")
    self.assertTrue(card.get("_card_error"))

  def test_health_status_unavailable_on_failure(self):
    with patch("strategy_portal_server._get_user_by_uid", side_effect=RuntimeError("db down")):
      r = self.client.get("/api/ctrader/health?uid=TEST")
    self.assertEqual(r.status_code, 200)
    body = r.json()
    self.assertTrue(body.get("status_unavailable"))
    self.assertEqual(body.get("error"), "status unavailable")


if __name__ == "__main__":
  unittest.main()
