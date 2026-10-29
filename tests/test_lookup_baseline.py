"""直接查代价表基线的正确性与计时控制检查。"""

import unittest
from unittest.mock import patch

import advanced_experiments as advanced
from search_algorithms import reverse_cost_potential


class TestLookupBaseline(unittest.TestCase):
    def test_matches_search_for_zero_cost_goal_and_unreachable_start(self):
        grid = (("S", 3, "E"), (0, 0, 1), ("X", "X", "X"), (1, 1, 1))
        starts = [(0, 0), (0, 1), (0, 2), (1, 0), (3, 0)]
        goal = (0, 2)
        potential = reverse_cost_potential(grid, goal)
        # 上方起点可经底下的零代价格绕行，进入E仍为0。
        expected = [1, 0, 0, 1, None]
        self.assertEqual(advanced.direct_lookup_batch(starts, potential), expected)
        self.assertEqual(advanced.ordinary_batch(grid, starts, goal), expected)
        self.assertEqual(advanced.exact_batch(grid, starts, goal, potential), expected)

    def test_three_way_rotation_rebuilds_each_preprocessed_trial(self):
        grid, starts, goal = (("S", 1, "E"),), [(0, 0)], (0, 2)
        potential = reverse_cost_potential(grid, goal)
        order = []
        with (
            patch.object(advanced, "ordinary_batch", side_effect=lambda *_: order.append("ordinary")),
            patch.object(advanced, "exact_batch", side_effect=lambda *_: order.append("exact")),
            patch.object(advanced, "direct_lookup_batch", side_effect=lambda *_: order.append("lookup")),
            patch.object(advanced, "reverse_cost_potential", return_value=potential) as build,
        ):
            timings = advanced.measure_multiquery(grid, starts, goal, repeats=6)
        names = ["ordinary", "exact", "lookup"]
        expected = names.copy()  # 三方案各预热一次。
        for repetition in range(6):
            offset = repetition % 3
            expected.extend(names[offset:] + names[:offset])
        self.assertEqual(order, expected)
        self.assertEqual(build.call_count, 1 + 2 * 6)
        self.assertEqual(len(timings), 10)
        self.assertTrue(all(value >= 0 for value in timings.values()))


if __name__ == "__main__":
    unittest.main()
