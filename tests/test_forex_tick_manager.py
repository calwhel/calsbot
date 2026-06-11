"""Forex tick manager debounce."""
import time
import unittest

from app.services import forex_tick_manager as ftm


class TestForexTickDebounce(unittest.TestCase):
    def test_debounce_blocks_rapid_refire(self):
        ftm._last_manage_mono.clear()
        self.assertTrue(ftm._should_run("EURUSD", 42))
        self.assertFalse(ftm._should_run("EURUSD", 42))
        time.sleep(ftm._TICK_DEBOUNCE_S + 0.05)
        self.assertTrue(ftm._should_run("EURUSD", 42))


if __name__ == "__main__":
    unittest.main()
