"""Forex position id resolution for broker reconcile + SL management."""
import unittest

from app.services.strategy_executor import _ctrader_position_id_from_execution


class _FakeEx:
    def __init__(self, **kw):
        self.ctrader_position_id = kw.get("ctrader_position_id")
        self.notes = kw.get("notes", "")


class TestCtraderPositionIdFromExecution(unittest.TestCase):
    def test_reads_column_first(self):
        ex = _FakeEx(ctrader_position_id="47500123", notes="")
        self.assertEqual(_ctrader_position_id_from_execution(ex), 47500123)

    def test_falls_back_to_notes_token(self):
        ex = _FakeEx(ctrader_position_id=None, notes="live | pos=881122")
        self.assertEqual(_ctrader_position_id_from_execution(ex), 881122)

    def test_missing_returns_none(self):
        ex = _FakeEx(ctrader_position_id=None, notes="no broker id")
        self.assertIsNone(_ctrader_position_id_from_execution(ex))


if __name__ == "__main__":
    unittest.main()
