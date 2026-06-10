import unittest
from datetime import datetime, timedelta

from app.services import economic_calendar as ec


class TestEconomicCalendar(unittest.TestCase):
    def setUp(self):
        self._saved = list(ec._EVENTS)
        ec._EVENTS = [{
            "time_utc": datetime(2026, 6, 10, 14, 0),
            "currency": "USD",
            "impact": "high",
            "event_name": "CPI",
        }]

    def tearDown(self):
        ec._EVENTS = self._saved

    def test_blocking_event_inside_window(self):
        cfg = {
            "news_filter_enabled": True,
            "news_buffer_before_min": 30,
            "news_buffer_after_min": 30,
            "news_impact": "high",
        }
        now = datetime(2026, 6, 10, 14, 15)
        self.assertEqual(ec.blocking_event("XAUUSD", cfg, now), "CPI")

    def test_blocking_event_fail_open_when_empty(self):
        ec._EVENTS = []
        cfg = {"news_filter_enabled": True}
        self.assertIsNone(ec.blocking_event("EURUSD", cfg, datetime.utcnow()))

    def test_blocking_event_respects_currency(self):
        cfg = {"news_filter_enabled": True, "news_impact": "high"}
        now = datetime(2026, 6, 10, 14, 0)
        self.assertIsNone(ec.blocking_event("EURGBP", cfg, now))

    def test_upcoming_high_within_15_min(self):
        ec._EVENTS[0]["time_utc"] = datetime.utcnow() + timedelta(minutes=10)
        evt = ec.upcoming_high_event("XAUUSD", datetime.utcnow(), within_min=15)
        self.assertEqual(evt, "CPI")


if __name__ == "__main__":
    unittest.main()
