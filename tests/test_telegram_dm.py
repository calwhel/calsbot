import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import telegram_dm


class TelegramDmHelpersTest(unittest.TestCase):
    def test_owner_chat_id_prefers_settings(self):
        with patch.dict(os.environ, {"OWNER_TELEGRAM_ID": "111"}, clear=False):
            with patch("app.config.settings") as mock_settings:
                mock_settings.OWNER_TELEGRAM_ID = "999"
                self.assertEqual(telegram_dm.owner_chat_id(), "999")

    def test_owner_chat_id_env_fallback(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.OWNER_TELEGRAM_ID = None
            with patch.dict(os.environ, {"OWNER_TELEGRAM_ID": "5603353066"}, clear=False):
                self.assertEqual(telegram_dm.owner_chat_id(), "5603353066")


class TelegramDmSendTest(unittest.IsolatedAsyncioTestCase):
    async def test_send_dm_confirms_ok_true(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_resp.text = '{"ok":true}'

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(telegram_dm, "bot_tokens", return_value=["tok"]):
            with patch("app.services.telegram_dm.httpx.AsyncClient", return_value=mock_client):
                ok = await telegram_dm.send_dm(123, "hello")
        self.assertTrue(ok)

    async def test_send_dm_treats_200_ok_false_as_delivered(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": False, "description": "bad"}
        mock_resp.text = '{"ok":false}'

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(telegram_dm, "bot_tokens", return_value=["tok"]):
            with patch("app.services.telegram_dm.httpx.AsyncClient", return_value=mock_client):
                ok = await telegram_dm.send_dm(123, "hello")
        self.assertTrue(ok)
        self.assertEqual(mock_client.post.await_count, 1)

    async def test_send_dm_tries_second_token_on_403(self):
        bad = MagicMock(status_code=403, text='{"ok":false}', json=lambda: {"ok": False})
        good = MagicMock(status_code=200, text='{"ok":true}', json=lambda: {"ok": True})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[bad, good])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(telegram_dm, "bot_tokens", return_value=["a", "b"]):
            with patch("app.services.telegram_dm.httpx.AsyncClient", return_value=mock_client):
                ok = await telegram_dm.send_dm(123, "hello")
        self.assertTrue(ok)
        self.assertEqual(mock_client.post.await_count, 2)


if __name__ == "__main__":
    unittest.main()
