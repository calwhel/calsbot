import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PORTAL_PATH = ROOT / "strategy_portal_server.py"
PORTAL = PORTAL_PATH.read_text()
MAIN = (ROOT / "main.py").read_text()
EXECUTOR = (ROOT / "app" / "services" / "strategy_executor.py").read_text()


class _AsyncSleepVisitor(ast.NodeVisitor):
    def __init__(self):
        self.async_time_sleep_lines: list[int] = []
        self._async_depth = 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._async_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # A sync helper nested inside an async route is executed only when called;
        # if it is passed to asyncio.to_thread/run_in_executor, its blocking work
        # does not run on the event loop. Do not count its body as async context.
        if self._async_depth == 0:
            for stmt in node.body:
                self.visit(stmt)

    def visit_Call(self, node: ast.Call):
        if self._async_depth:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "sleep"
                and isinstance(func.value, ast.Name)
                and func.value.id in {"time", "_time"}
            ):
                self.async_time_sleep_lines.append(node.lineno)
        self.generic_visit(node)


class RailwayPerformanceSourceTests(unittest.TestCase):
    def test_neon_keepalive_uses_async_sleep_and_logs_success(self):
        self.assertIn("await asyncio.sleep(240)", PORTAL)
        self.assertIn("[neon-keepwarm] ping ok", PORTAL)
        self.assertIn("next ping in 240s", PORTAL)

    def test_target_endpoints_emit_cache_headers_with_expected_ttls(self):
        self.assertIn('headers = {"X-Cache": "HIT" if hit else "MISS"}', PORTAL)
        self.assertIn('headers["X-Cache-TTL"] = str(ttl_seconds)', PORTAL)
        for snippet in (
            "_cached_json(cached[0], True, 60)",
            "_cached_json(data, False, 60)",
            "_cached_json(payload, False, 60)",
            "_cached_json(_ldr_cached, True, 120)",
            "_cached_json(top, False, 120)",
            "_cached_json(_an_hit, True, 300)",
            "_cached_json(_an_payload, False, 300)",
            "_cached_json(cached_result, True, 600)",
            "_cached_json(result, False, 600)",
        ):
            self.assertIn(snippet, PORTAL)

    def test_no_time_sleep_directly_inside_async_functions(self):
        tree = ast.parse(PORTAL, filename=str(PORTAL_PATH))
        visitor = _AsyncSleepVisitor()
        visitor.visit(tree)
        self.assertEqual([], visitor.async_time_sleep_lines)

    def test_shared_pool_is_the_only_engine_factory(self):
        offenders = []
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts or "tests" in path.parts or path == ROOT / "app" / "database.py":
                continue
            if "create_engine(" in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual([], offenders)

    def test_executor_uses_background_shared_pool(self):
        self.assertIn("from app.database import BgSessionLocal", EXECUTOR)
        self.assertIn("from app.database import BgSessionLocal as SessionLocal", EXECUTOR)
        self.assertNotIn("create_engine(", EXECUTOR)


if __name__ == "__main__":
    unittest.main()
