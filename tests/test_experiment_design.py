"""检查跨深度对照是否只改变深度，避免把场景变化混入深度效果。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from extended_experiments import game_experiment


class ExperimentDesignTests(unittest.TestCase):
    def test_same_value_cases_are_paired_across_depths(self):
        # 使用确定性轻量结果，检验实验调度，无需在此重复计时算法。
        stats = SimpleNamespace(
            visited_nodes=1,
            evaluated_leaves=1,
            cache_hits=0,
            cache_entries=0,
            cutoffs=0,
            elapsed_ns=0,
        )
        outcome = SimpleNamespace(first_target="T1", utility=0, stats=stats)
        with patch("extended_experiments.compute_distances", return_value=object()), patch(
            "extended_experiments.solve_treasure_game", return_value=outcome
        ):
            rows, _ = game_experiment()
        self.assertEqual(len(rows), 3 * 40 * 4)
        cases_by_depth = {
            depth: {
                row["case"]: tuple(row[f"T{i}"] for i in range(1, 7))
                for row in rows
                if row["depth"] == depth
            }
            for depth in (2, 4, 6)
        }
        self.assertEqual(len(set(cases_by_depth[2].values())), 40)
        self.assertEqual(cases_by_depth[2], cases_by_depth[4])
        self.assertEqual(cases_by_depth[2], cases_by_depth[6])


if __name__ == "__main__":
    unittest.main()
