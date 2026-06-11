"""WIN/LOSS/breakeven Telegram builders and outbound queue."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import strategy_executor as se
from app.services import telegram_dm as tdm


class TestCloseTelegramSafe(unittest.TestCase):
    def test_build_close_safe_handles_null_entry(self):
        text = se._build_close_telegram_safe(
            execution_id=99,
            strategy_name="Gold",
            symbol="XAUUSD",
            direction="SHORT",
            entry=None,
            exit_price=None,
            outcome="WIN",
            pnl_pct=None,
            leverage=None,
        )
        self.assertIn("WIN", text)
        self.assertIn("XAUUSD", text)

    def test_fmt_close_card_entry_equals_exit(self):
        text = se._fmt_close_card(
            strategy_name="Gold",
            symbol="XAUUSD",
            direction="LONG",
            entry=2650.0,
            exit_price=2650.0,
            outcome="BREAKEVEN",
            pnl_pct=0.0,
            leverage=1,
        )
        self.assertIn("BREAKEVEN", text)
        self.assertIn("0 pips", text)

    def test_breakeven_safe_fallback_on_bad_html(self):
        with patch.object(se, "_fmt_breakeven_card", side_effect=ValueError("boom")):
            text = se._build_breakeven_telegram_safe(
                execution_id=7,
                strategy_name="X",
                symbol="XAUUSD",
                direction="LONG",
            )
        self.assertIn("Breakeven", text)
        self.assertIn("XAUUSD", text)


class TestTelegram429(unittest.IsolatedAsyncioTestCase):
    async def test_send_dm_honours_retry_after(self):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.json.return_value = {
            "ok": False,
            "parameters": {"retry_after": 1},
        }
        rate_limited.headers = {}
        rate_limited.text = '{"ok":false}'

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        ok_resp.text = '{"ok":true}'

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[rate_limited, ok_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(tdm, "bot_tokens", return_value=["tok"]):
            with patch("app.services.telegram_dm.httpx.AsyncClient", return_value=mock_client):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    ok = await tdm.send_dm(123, "hello")
        self.assertTrue(ok)
        mock_sleep.assert_awaited()
        self.assertEqual(mock_client.post.await_count, 2)


class TestTelegramQueue(unittest.TestCase):
    def test_enqueue_trade_telegram_puts_job(self):
        while not tdm._OUTBOUND.empty():
            tdm._OUTBOUND.get_nowait()
        with patch.object(tdm, "_ensure_outbound_drainer"):
            tdm.enqueue_trade_telegram(
                123,
                "✅ WIN XAUUSD",
                msg_type="win",
                symbol="XAUUSD",
                exec_id=42,
            )
        job = tdm._OUTBOUND.get_nowait()
        self.assertEqual(job.chat_id, 123)
        self.assertEqual(job.msg_type, "win")
        self.assertEqual(job.exec_id, 42)


if __name__ == "__main__":
    unittest.main()
