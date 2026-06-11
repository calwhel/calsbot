"""Lock connection must use direct Neon host (not pooler)."""
import unittest

from app.executor_lock import get_lock_database_url


class TestLockDatabaseUrl(unittest.TestCase):
    def test_strips_pooler_suffix(self):
        url = (
            "postgresql://u:p@ep-foo-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
        )
        with unittest.mock.patch(
            "app.config.settings.get_database_url", return_value=url
        ):
            direct = get_lock_database_url()
        self.assertIn("ep-foo.us-east-2.aws.neon.tech", direct)
        self.assertNotIn("-pooler", direct)


if __name__ == "__main__":
    unittest.main()
