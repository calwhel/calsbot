"""Strategy executor sharding — partition strategies without overlap."""
import unittest

from app.services.strategy_executor import (
    strategy_on_shard,
    strategy_shard_index,
    _executor_shard_label,
)


class TestExecutorSharding(unittest.TestCase):
    def test_partition_covers_all_ids(self):
        count = 3
        for sid in range(1, 200):
            matches = sum(
                1 for i in range(count) if strategy_on_shard(sid, i, count)
            )
            self.assertEqual(matches, 1, f"strategy {sid} must land on exactly one shard")

    def test_shard_index_modulo(self):
        self.assertEqual(strategy_shard_index(205, 3), 205 % 3)
        self.assertEqual(strategy_shard_index(78, 3), 0)

    def test_label_single_shard(self):
        self.assertEqual(_executor_shard_label("FX Executor", 0, 1), "FX Executor")

    def test_label_multi_shard(self):
        self.assertEqual(
            _executor_shard_label("FX Executor", 2, 3),
            "FX Executor S2/3",
        )


if __name__ == "__main__":
    unittest.main()
