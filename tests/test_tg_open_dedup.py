"""Open-trade Telegram deduplication helpers."""
import unittest
from unittest.mock import MagicMock


class TgOpenDedupTest(unittest.TestCase):
    def test_claim_returns_true_when_row_updated(self):
        from app.services.strategy_executor import _claim_tg_open_notify

        db = MagicMock()
        db.execute.return_value.rowcount = 1
        self.assertTrue(_claim_tg_open_notify(db, 42))
        db.commit.assert_called_once()

    def test_claim_returns_false_when_already_sent(self):
        from app.services.strategy_executor import _claim_tg_open_notify

        db = MagicMock()
        db.execute.return_value.rowcount = 0
        self.assertFalse(_claim_tg_open_notify(db, 42))

    def test_queued_notice_is_not_full_open_card(self):
        from app.services.strategy_executor import _fmt_queued_open_notice

        text = _fmt_queued_open_notice("Gold Scalp", "XAUUSD", "LONG", leverage=1)
        self.assertIn("Placing live order", text)
        self.assertNotIn("LIVE TRADE OPENED", text)

    def test_be_claim_returns_false_when_already_sent(self):
        from app.services.strategy_executor import _claim_tg_be_notify

        db = MagicMock()
        db.execute.return_value.rowcount = 0
        self.assertFalse(_claim_tg_be_notify(db, 99))

    def test_release_open_notify_calls_update(self):
        from app.services.strategy_executor import _release_tg_open_notify

        db = MagicMock()
        _release_tg_open_notify(db, 7)
        db.execute.assert_called_once()
        db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
