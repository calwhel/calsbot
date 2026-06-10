"""Regression: reload ORM row by PK after Neon commit churn."""
import unittest
from unittest import mock

from app.services.strategy_executor import _refresh_db_row


class _FakeInstance:
    def __init__(self, id=None):
        self.id = id


class TestDbSessionRebind(unittest.TestCase):
    def test_refresh_db_row_reloads_by_pk(self):
        inst = _FakeInstance(id=42)
        fresh = _FakeInstance(id=42)
        db = mock.MagicMock()
        db.get.return_value = fresh
        self.assertIs(_refresh_db_row(db, inst), fresh)
        db.get.assert_called_once_with(_FakeInstance, 42)


if __name__ == "__main__":
    unittest.main()
