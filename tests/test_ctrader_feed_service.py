"""Dedicated cTrader feed service + remote consumer mode."""
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import ctrader_price_feed as feed


def _ctrader_client():
    try:
        from app.services import ctrader_client
        return ctrader_client
    except Exception:
        return None


class TestCtraderFeedService(unittest.TestCase):
    def setUp(self):
        feed._feed_live = False
        feed._stream_creds = None
        feed._spot_cache.clear()
        feed._auth_backoff_until = 0.0
        feed._last_auth_error = None

    def test_terminal_auth_detection(self):
        self.assertTrue(
            feed._is_terminal_auth_error("CH_ACCESS_TOKEN_INVALID: Invalid access token")
        )
        self.assertTrue(feed._is_terminal_auth_error("ACCESS_DENIED"))
        self.assertFalse(feed._is_terminal_auth_error("timeout waiting"))

    def test_terminal_auth_sets_backoff(self):
        feed._note_terminal_auth("CH_ACCESS_TOKEN_INVALID")
        self.assertGreater(feed._auth_backoff_until, time.monotonic())

    def test_is_live_remote_from_store(self):
        with patch.dict(os.environ, {"CTRADER_REMOTE_FEED": "1"}, clear=False):
            with patch.object(feed, "_shared_ctrader_ticks_fresh", return_value=True):
                self.assertTrue(feed.is_live())

    def test_broker_session_ready_false_on_remote_executor(self):
        """Executor must not open trendbar sockets when feed runs elsewhere."""
        with patch.dict(os.environ, {"CTRADER_REMOTE_FEED": "1"}, clear=False):
            with patch.object(feed, "_shared_ctrader_ticks_fresh", return_value=True):
                self.assertFalse(feed.broker_session_ready("XAUUSD"))

    def test_start_skipped_when_remote_feed(self):
        feed._feed_task = None
        with patch.dict(
            os.environ,
            {"CTRADER_REMOTE_FEED": "1", "CTRADER_CLIENT_ID": "x"},
            clear=False,
        ):
            with patch.object(feed, "_get_wake_event", return_value=MagicMock()):
                ok = feed.launch_ctrader_feed()
                self.assertFalse(ok)
                self.assertIsNone(feed._feed_task)

    def test_launch_schedules_task_when_no_linked_account(self):
        feed._feed_task = None
        mock_loop = MagicMock()
        with patch.dict(os.environ, {"CTRADER_CLIENT_ID": "x"}, clear=False):
            with patch.object(feed, "_PROTO_OK", True):
                with patch.object(feed, "probe_linked_accounts_sync", return_value=[]):
                    with patch.object(feed, "_get_wake_event", return_value=MagicMock()):
                        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
                            with patch("asyncio.get_event_loop", return_value=mock_loop):
                                ok = feed.launch_ctrader_feed()
        self.assertTrue(ok)
        mock_loop.create_task.assert_called_once()

    def test_feed_status_needs_relink(self):
        feed._last_auth_error = "CH_ACCESS_TOKEN_INVALID: Invalid access token"
        st = feed.feed_status()
        self.assertTrue(st["needs_relink"])
        self.assertIn("reconnect", (st.get("note") or "").lower())

    @unittest.skipUnless(_ctrader_client(), "ctrader_client unavailable")
    def test_refresh_terminal_codes(self):
        cclient = _ctrader_client()
        self.assertIn("CH_ACCESS_TOKEN_INVALID", cclient._REFRESH_TERMINAL_CODES)

    @unittest.skipUnless(_ctrader_client(), "ctrader_client unavailable")
    def test_is_refresh_denied(self):
        cclient = _ctrader_client()
        cclient._refresh_denied[1] = ("rtok", time.monotonic() + 60)
        self.assertTrue(cclient.is_refresh_denied(1))
        self.assertFalse(cclient.is_refresh_denied(2))


@unittest.skipUnless(_ctrader_client(), "ctrader_client unavailable")
class TestRefreshHttpRetry(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_retries_connect_timeout(self):
        import httpx
        cclient = _ctrader_client()

        calls = {"n": 0}

        async def _fake_post(*_a, **_k):
            calls["n"] += 1
            if calls["n"] < 2:
                raise httpx.ConnectTimeout("timeout")
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={"accessToken": "new"})
            return resp

        with patch("httpx.AsyncClient") as mock_client:
            inst = mock_client.return_value.__aenter__.return_value
            inst.post = _fake_post
            res = await cclient.refresh_access_token("rtok")
            self.assertEqual(res["accessToken"], "new")
            self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
