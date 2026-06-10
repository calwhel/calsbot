import unittest


class TestTradeMgmtSchema(unittest.TestCase):
    def test_required_columns_list(self):
        from app.trade_mgmt_schema import TRADE_MGMT_REQUIRED_COLUMNS

        self.assertIn("breakeven_applied", TRADE_MGMT_REQUIRED_COLUMNS)
        self.assertIn("remaining_volume", TRADE_MGMT_REQUIRED_COLUMNS)
        self.assertEqual(len(TRADE_MGMT_REQUIRED_COLUMNS), 6)

    def test_schema_error_detection(self):
        from app.trade_mgmt_schema import is_trade_mgmt_schema_error

        exc = Exception(
            '(psycopg2.errors.UndefinedColumn) column strategy_executions.'
            'breakeven_applied does not exist'
        )
        self.assertTrue(is_trade_mgmt_schema_error(exc))
        self.assertFalse(is_trade_mgmt_schema_error(Exception("connection refused")))

    def test_migrations_use_if_not_exists(self):
        from app.trade_mgmt_schema import TRADE_MGMT_COLUMN_MIGRATIONS

        for _table, col, ddl in TRADE_MGMT_COLUMN_MIGRATIONS:
            self.assertIn("IF NOT EXISTS", ddl)
            self.assertIn(col, ddl)


if __name__ == "__main__":
    unittest.main()
