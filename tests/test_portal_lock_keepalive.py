"""Portal advisory-lock paths must use build_lock_connection (TCP keepalive)."""
import ast
import unittest
from pathlib import Path


class TestPortalLockKeepalive(unittest.TestCase):
    def test_force_start_uses_build_lock_connection(self):
        src = Path("strategy_portal_server.py").read_text(encoding="utf-8")
        self.assertIn("build_lock_connection", src)
        self.assertNotIn("psycopg2.connect(settings.get_database_url())", src)

    def test_keepalive_thread_accepts_lock_path(self):
        tree = ast.parse(Path("strategy_portal_server.py").read_text(encoding="utf-8"))
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_advisory_lock_keepalive_thread"
        )
        arg_names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertIn("lock_path", arg_names)


if __name__ == "__main__":
    unittest.main()
