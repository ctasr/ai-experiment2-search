"""题目样例、异常边界与独立参照实现。运行：python -m unittest discover -s tests -v"""

import heapq
import random
import unittest
from itertools import pairwise

from game_search import TreeNode, alpha_beta_best_score, sample_tree, tree_minimax
from search_algorithms import (
    maze_search,
    minimum_terrain_cost,
    path_cost_trace,
    reverse_cost_potential,
    shortest_maze_path,
    weighted_search,
    weighted_search_with_potential,
)
from treasure_game import (
    SAMPLE_MAP,
    SAMPLE_VALUES,
    compute_distances,
    initial_state,
    solve_treasure_game,
    take_treasure,
    treasure_game_answer,
)

MAZE = (
    ("S", ".", ".", "X", "."),
    (".", "X", ".", ".", "."),
    (".", "X", ".", "X", "."),
    (".", ".", ".", "X", "E"),
    ("X", "X", ".", ".", "."),
)
TERRAIN = (
    ("S", 1, 1, 8, 1),
    (1, 8, 3, 1, 1),
    (1, 8, 1, 8, 1),
    (1, 1, 1, 8, 1),
    (8, 8, 1, 1, "E"),
)


def oracle_dijkstra(grid, start, goal, cell_cost):
    """独立实现的零启发式最短路，避免只让 A* 自我比对。"""
    queue = [(0, start)]
    distances = {start: 0}
    while queue:
        g, (row, col) = heapq.heappop(queue)
        if g != distances[(row, col)]:
            continue
        if (row, col) == goal:
            return g
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            cell = grid[nr][nc]
            if str(cell) == "X":
                continue
            candidate = g + cell_cost(cell)
            if candidate < distances.get((nr, nc), float("inf")):
                distances[(nr, nc)] = candidate
                heapq.heappush(queue, (candidate, (nr, nc)))
    return None


def oracle_treasure(distances, values, horizon):
    """独立的四回合穷举，仅使用普通位置/剩余集合/分数变量。"""
    names = tuple(values)

    def recurse(a_place, b_place, remaining, score_a, score_b, rounds_left, a_turn):
        if rounds_left == 0 or not remaining:
            return score_a - score_b
        outcomes = []
        for target in remaining:
            rest = tuple(item for item in remaining if item != target)
            if a_turn:
                gain = values[target] - distances.costs[a_place][target]
                outcomes.append(
                    recurse(
                        target,
                        b_place,
                        rest,
                        score_a + gain,
                        score_b,
                        rounds_left - 1,
                        False,
                    )
                )
            else:
                gain = values[target] - distances.costs[b_place][target]
                outcomes.append(
                    recurse(
                        a_place,
                        target,
                        rest,
                        score_a,
                        score_b + gain,
                        rounds_left - 1,
                        True,
                    )
                )
        return max(outcomes) if a_turn else min(outcomes)

    root_scores = {}
    for target in names:
        gain = values[target] - distances.costs["A"][target]
        root_scores[target] = recurse(
            target,
            "B",
            tuple(item for item in names if item != target),
            gain,
            0,
            horizon - 1,
            False,
        )
    return root_scores


def assert_valid_search_path(test, grid, start, goal, result, cell_cost):
    """除最优代价外，逐步检查返回路径确实能走且代价相符。"""
    if result.cost is None:
        test.assertEqual(result.path, ())
        return
    test.assertEqual(result.path[0], start)
    test.assertEqual(result.path[-1], goal)
    test.assertTrue(all(str(grid[row][col]) != "X" for row, col in result.path))
    test.assertTrue(
        all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairwise(result.path))
    )
    test.assertEqual(
        sum(cell_cost(grid[row][col]) for row, col in result.path[1:]),
        result.cost,
    )


class TestMaze(unittest.TestCase):
    def test_given_sample(self):
        path = shortest_maze_path((0, 0), (3, 4), MAZE)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 4))
        self.assertEqual(len(path) - 1, 7)
        self.assertTrue(
            all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairwise(path))
        )
        self.assertEqual(maze_search(MAZE, (0, 0), (3, 4), "zero").cost, 7)

    def test_expansion_trace_explains_path(self):
        trace = []
        result = maze_search(MAZE, (0, 0), (3, 4), expansion_trace=trace)
        zero_trace = []
        zero_result = maze_search(
            MAZE, (0, 0), (3, 4), "zero", expansion_trace=zero_trace
        )
        self.assertEqual(len(trace), result.expanded)
        self.assertEqual(len(zero_trace), zero_result.expanded)
        self.assertEqual((len(zero_trace), len(trace)), (16, 8))
        self.assertEqual(zero_result.cost, result.cost)
        self.assertEqual(trace[0]["position"], (0, 0))
        self.assertEqual(trace[-1]["position"], (3, 4))
        self.assertEqual(trace[-1]["relaxed_neighbors"], [])
        for step in trace:
            self.assertEqual(step["f"], step["g"] + step["h"])
            for neighbor in step["relaxed_neighbors"]:
                self.assertEqual(neighbor["f"], neighbor["new_g"] + neighbor["h"])
        assert_valid_search_path(self, MAZE, (0, 0), (3, 4), result, lambda _cell: 1)
        assert_valid_search_path(
            self, MAZE, (0, 0), (3, 4), zero_result, lambda _cell: 1
        )

    def test_no_path_and_same_start_goal(self):
        grid = (("S", "X", "E"), ("X", "X", "X"))
        self.assertEqual(shortest_maze_path((0, 0), (0, 2), grid), [])
        self.assertEqual(shortest_maze_path((0, 0), (0, 0), grid), [(0, 0)])

    def test_multiple_equal_shortest_paths(self):
        grid = (("S", "."), (".", "E"))
        path = shortest_maze_path((0, 0), (1, 1), grid)
        self.assertIn(path, [[(0, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (1, 1)]])

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            shortest_maze_path((0, 0), (0, 1), (("S", "E"), (".",)))

    def test_random_grids_against_independent_dijkstra(self):
        rng = random.Random(42)
        for _ in range(80):
            grid = [
                ["X" if rng.random() < 0.28 else "." for _ in range(7)]
                for _ in range(7)
            ]
            grid[0][0], grid[6][6] = "S", "E"
            expected = oracle_dijkstra(grid, (0, 0), (6, 6), lambda _cell: 1)
            for order in ("deeper", "fifo", "shallower"):
                actual = maze_search(grid, (0, 0), (6, 6), tie_break=order)
                self.assertEqual(actual.cost, expected)
                assert_valid_search_path(
                    self, grid, (0, 0), (6, 6), actual, lambda _cell: 1
                )

    def test_tie_breaking_preserves_optimal_cost(self):
        for order in ("deeper", "fifo", "shallower"):
            result = maze_search(MAZE, (0, 0), (3, 4), tie_break=order)
            self.assertEqual(result.cost, 7)
            self.assertEqual(len(result.path), 8)


class TestTerrain(unittest.TestCase):
    def test_given_sample_and_trace(self):
        result = weighted_search(TERRAIN, (0, 0), (4, 4))
        self.assertEqual(result.cost, 7)
        self.assertEqual(minimum_terrain_cost((0, 0), (4, 4), TERRAIN), 7)
        trace = path_cost_trace(
            TERRAIN, result.path, lambda cell: 0 if cell == "E" else int(cell)
        )
        self.assertEqual(trace[-1]["enter_cost"], 0)
        self.assertEqual(trace[-1]["cumulative_cost"], 7)

    def test_string_numbers_and_no_path(self):
        strings = [[str(cell) for cell in row] for row in TERRAIN]
        self.assertEqual(minimum_terrain_cost((0, 0), (4, 4), strings), 7)
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 2), (("S", "X", "E"),)), -1)

    def test_zero_cost_goal_does_not_break_optimality(self):
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 1), (("S", "E"),)), 0)

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            minimum_terrain_cost((0, 0), (0, 2), (("S", -1, "E"),))

    def test_random_weights_against_independent_dijkstra(self):
        rng = random.Random(123)
        for _ in range(80):
            grid = [
                ["X" if rng.random() < 0.2 else rng.choice((1, 3, 8)) for _ in range(6)]
                for _ in range(6)
            ]
            grid[0][0], grid[5][5] = "S", "E"
            expected = oracle_dijkstra(
                grid,
                (0, 0),
                (5, 5),
                lambda cell: 0 if cell == "E" else (1 if cell == "S" else int(cell)),
            )
            for order in ("deeper", "fifo", "shallower"):
                actual = weighted_search(grid, (0, 0), (5, 5), tie_break=order)
                self.assertEqual(actual.cost, expected)
                assert_valid_search_path(
                    self,
                    grid,
                    (0, 0),
                    (5, 5),
                    actual,
                    lambda cell: (
                        0 if cell == "E" else (1 if cell == "S" else int(cell))
                    ),
                )

    def test_zero_cost_terrain_and_tie_breaking(self):
        grid = (("S", 0, 0, "E"), (5, 5, 5, 5))
        for order in ("deeper", "fifo", "shallower"):
            result = weighted_search(grid, (0, 0), (0, 3), tie_break=order)
            self.assertEqual(result.cost, 0)
            self.assertEqual(
                result.cost,
                oracle_dijkstra(
                    grid,
                    (0, 0),
                    (0, 3),
                    lambda c: 0 if c == "E" else (1 if c == "S" else int(c)),
                ),
            )

    def test_reverse_potential_matches_independent_shortest_costs(self):
        grid = (("S", 9, 1, "E"), (0, 0, 1, 9), (3, "X", 0, 1))
        goal = (0, 3)
        potential = reverse_cost_potential(grid, goal)
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == "X":
                    continue
                start = (row, col)
                expected = oracle_dijkstra(
                    grid,
                    start,
                    goal,
                    lambda c: 0 if c == "E" else (1 if c == "S" else int(c)),
                )
                self.assertEqual(potential.costs.get(start), expected)
                self.assertEqual(
                    weighted_search_with_potential(grid, start, goal, potential).cost,
                    expected,
                )
        with self.assertRaises(ValueError):
            weighted_search_with_potential(
                (("S", 1, 1, "E"), (0, 0, 1, 9), (3, "X", 0, 1)),
                (0, 0),
                goal,
                potential,
            )

    def test_potential_on_disconnected_grid(self):
        grid = (("S", "X", "E"), ("X", "X", "X"))
        potential = reverse_cost_potential(grid, (0, 2))
        self.assertNotIn((0, 0), potential.costs)
        self.assertIsNone(
            weighted_search_with_potential(grid, (0, 0), (0, 2), potential).cost
        )


class TestTree(unittest.TestCase):
    def test_given_shared_subtree(self):
        root = sample_tree()
        self.assertIs(root.children[0].children[1], root.children[1].children[0])
        self.assertEqual(alpha_beta_best_score(root), 5)
        self.assertEqual(tree_minimax(root, alpha_beta=False)[0], 5)
        _, stats = tree_minimax(root)
        self.assertEqual(len(stats.pruned_branches), stats.skipped_children)
        self.assertEqual(len(stats.cutoff_trace), stats.cutoffs)
        self.assertEqual(
            [path for cutoff in stats.cutoff_trace for path in cutoff["skipped_paths"]],
            stats.pruned_branches,
        )
        self.assertTrue(
            all(cutoff["alpha"] >= cutoff["beta"] for cutoff in stats.cutoff_trace)
        )
        self.assertEqual(alpha_beta_best_score(root), 5)

    def test_random_trees_equal_full_minimax(self):
        rng = random.Random(7)
        for _ in range(100):

            def make(level):
                if level == 0:
                    return TreeNode(value=rng.randint(-20, 20))
                return TreeNode(
                    children=[make(level - 1) for _ in range(rng.randint(2, 4))]
                )

            root = make(3)
            exact = tree_minimax(root, alpha_beta=False)[0]
            self.assertEqual(tree_minimax(root)[0], exact)
            self.assertEqual(tree_minimax(root, order="reverse")[0], exact)

    def test_cycle_rejected(self):
        root = TreeNode()
        root.children.append(root)
        with self.assertRaises(ValueError):
            alpha_beta_best_score(root)


class TestTreasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances = compute_distances()

    def test_locations_matrix_and_independent_dijkstra(self):
        d = self.distances
        self.assertEqual(d.sites, ("A", "B", "T1", "T2", "T3", "T4"))
        self.assertEqual(d.locations["A"], (0, 0))
        self.assertEqual(d.locations["B"], (6, 6))
        self.assertEqual(d.astar_runs, 30)
        for source in d.sites:
            for target in d.sites:
                self.assertEqual(d.costs[source][target], d.costs[target][source])
                expected = oracle_dijkstra(
                    SAMPLE_MAP,
                    d.locations[source],
                    d.locations[target],
                    lambda cell: 1 if str(cell) in d.sites else int(cell),
                )
                self.assertEqual(d.costs[source][target], expected)

    def test_official_h2_h4_and_all_orders(self):
        d = self.distances
        for depth, first, utility in ((2, "T2", -1), (4, "T3", 3)):
            for prune in (False, True):
                self.assertEqual(
                    treasure_game_answer(d, depth, alpha_beta=prune),
                    {"best_target": first, "best_value": utility},
                )
            root_scores = oracle_treasure(d, SAMPLE_VALUES, depth)
            self.assertEqual(
                root_scores,
                {"T1": -4, "T2": -1, "T3": -2, "T4": -4}
                if depth == 2
                else {"T1": -5, "T2": 1, "T3": 3, "T4": -7},
            )
            self.assertEqual(root_scores[first], utility)
            self.assertEqual(max(root_scores.values()), utility)
            for ordering in ("given", "reverse", "greedy", "lookahead"):
                for prune in (False, True):
                    actual = solve_treasure_game(
                        d, depth, alpha_beta=prune, order=ordering
                    )
                    self.assertEqual(
                        (actual.first_target, actual.utility), (first, utility)
                    )
                    self.assertEqual(
                        len(actual.stats.pruned_branches),
                        actual.stats.skipped_children,
                    )

    def test_state_and_negative_gain(self):
        state = initial_state()
        next_state = take_treasure(state, "T1", self.distances)
        self.assertEqual(next_state.score_a, -1)
        self.assertEqual(next_state.current_player, "B")
        self.assertNotIn("T1", next_state.remaining)
        self.assertEqual(next_state.b_position, "B")

    def test_perturbed_values_all_depths_match_full_minimax(self):
        rng = random.Random(5)
        for _ in range(30):
            values = {name: rng.randint(1, 15) for name in SAMPLE_VALUES}
            for depth in (1, 2, 3, 4):
                full = solve_treasure_game(
                    self.distances, depth, alpha_beta=False, values=values
                )
                pruned = solve_treasure_game(
                    self.distances, depth, alpha_beta=True, values=values
                )
                lookahead = solve_treasure_game(
                    self.distances,
                    depth,
                    alpha_beta=True,
                    order="lookahead",
                    values=values,
                )
                self.assertEqual(
                    (pruned.first_target, pruned.utility),
                    (full.first_target, full.utility),
                )
                oracle = oracle_treasure(self.distances, values, depth)
                self.assertEqual(full.utility, max(oracle.values()))
                self.assertEqual(oracle[full.first_target], full.utility)
                self.assertEqual(lookahead.utility, full.utility)

    def test_ordering_limitation_and_tied_optima(self):
        values = {"T1": 30, "T2": 12, "T3": 12, "T4": 29}
        given = solve_treasure_game(self.distances, 4, order="given", values=values)
        greedy = solve_treasure_game(self.distances, 4, order="greedy", values=values)
        reverse = solve_treasure_game(self.distances, 4, order="reverse", values=values)
        self.assertEqual((given.utility, greedy.utility, reverse.utility), (1, 1, 1))
        self.assertGreater(greedy.stats.visited_nodes, given.stats.visited_nodes)
        self.assertEqual({given.first_target, reverse.first_target}, {"T1", "T2"})

    def test_alternate_weighted_map_generalizes(self):
        grid = (
            ("A", "1", "X", "3", "T1"),
            ("1", "1", "1", "1", "1"),
            ("T2", "X", "8", "X", "T3"),
            ("1", "1", "1", "1", "1"),
            ("T4", "2", "X", "1", "B"),
        )
        values = {"T1": 8, "T2": 5, "T3": 10, "T4": 4}
        distances = compute_distances(grid, values)
        for source in distances.sites:
            for target in distances.sites:
                self.assertEqual(
                    distances.costs[source][target],
                    oracle_dijkstra(
                        grid,
                        distances.locations[source],
                        distances.locations[target],
                        lambda cell: 1 if str(cell) in distances.sites else int(cell),
                    ),
                )
        for depth in (2, 4):
            expected = oracle_treasure(distances, values, depth)
            for ordering in ("given", "greedy"):
                result = solve_treasure_game(
                    distances, depth, order=ordering, values=values
                )
                self.assertEqual(result.utility, max(expected.values()))
                self.assertEqual(expected[result.first_target], result.utility)

    def test_random_weighted_maps_against_independent_oracles(self):
        rng = random.Random(20261020)
        names = ("A", "B", "T1", "T2", "T3", "T4")
        checked = 0
        for _attempt in range(200):
            grid = [
                [
                    "X" if rng.random() < 0.12 else rng.choice((0, 1, 3, 8))
                    for _ in range(7)
                ]
                for _ in range(7)
            ]
            positions = rng.sample(
                [(row, col) for row in range(7) for col in range(7)], 6
            )
            markers = dict(zip(names, positions))
            for name, (row, col) in markers.items():
                grid[row][col] = name
            values = {f"T{i}": rng.randint(1, 20) for i in range(1, 5)}
            try:
                distances = compute_distances(grid, values)
            except ValueError as error:
                if "不可达" not in str(error):
                    raise
                continue
            for source in distances.sites:
                for target in distances.sites:
                    expected = oracle_dijkstra(
                        grid,
                        markers[source],
                        markers[target],
                        lambda cell: 1 if str(cell) in names else int(cell),
                    )
                    self.assertEqual(distances.costs[source][target], expected)
            for depth in (2, 4):
                root_scores = oracle_treasure(distances, values, depth)
                for order in ("given", "greedy", "lookahead"):
                    for alpha_beta in (False, True):
                        actual = solve_treasure_game(
                            distances,
                            depth,
                            alpha_beta=alpha_beta,
                            order=order,
                            values=values,
                        )
                        self.assertEqual(actual.utility, max(root_scores.values()))
                        self.assertEqual(
                            root_scores[actual.first_target], actual.utility
                        )
            checked += 1
            if checked == 30:
                break
        self.assertEqual(checked, 30)

    def test_six_treasures_and_memoization(self):
        grid = (
            ("A", "1", "T1", "1", "T2"),
            ("1", "1", "1", "1", "1"),
            ("T3", "1", "T4", "1", "T5"),
            ("1", "1", "1", "1", "1"),
            ("T6", "1", "1", "1", "B"),
        )
        values = {f"T{i}": i + 4 for i in range(1, 7)}
        distances = compute_distances(grid, values)
        for a in distances.sites:
            for b in distances.sites:
                self.assertEqual(
                    distances.costs[a][b],
                    oracle_dijkstra(
                        grid,
                        distances.locations[a],
                        distances.locations[b],
                        lambda cell: 1 if str(cell) in distances.sites else int(cell),
                    ),
                )
        for depth in (2, 4, 6):
            full = solve_treasure_game(
                distances, depth, alpha_beta=False, values=values
            )
            cached = solve_treasure_game(
                distances, depth, alpha_beta=False, memoize=True, values=values
            )
            pruned = solve_treasure_game(
                distances, depth, order="greedy", values=values
            )
            lookahead = solve_treasure_game(
                distances, depth, order="lookahead", values=values
            )
            self.assertEqual(
                (cached.first_target, cached.utility), (full.first_target, full.utility)
            )
            self.assertEqual(pruned.utility, full.utility)
            self.assertEqual(lookahead.utility, full.utility)
            self.assertGreater(lookahead.stats.ordering_evaluations, 0)
            self.assertEqual(
                full.utility, max(oracle_treasure(distances, values, depth).values())
            )
        self.assertGreater(cached.stats.cache_hits, 0)
        with self.assertRaises(ValueError):
            solve_treasure_game(distances, 6, memoize=True, values=values)

    def test_partial_state_and_depth_limit(self):
        state = take_treasure(initial_state(), "T1", self.distances)
        state = take_treasure(state, "T2", self.distances)
        for depth in (1, 2, 5):
            baseline = solve_treasure_game(
                self.distances, depth, alpha_beta=False, start=state
            )
            cached = solve_treasure_game(
                self.distances, depth, alpha_beta=False, memoize=True, start=state
            )
            pruned = solve_treasure_game(self.distances, depth, start=state)
            self.assertEqual(
                (cached.first_target, cached.utility),
                (baseline.first_target, baseline.utility),
            )
            self.assertEqual(pruned.utility, baseline.utility)

    def test_unreachable_treasure_rejected(self):
        grid = (("A", "X", "T1"), ("X", "X", "X"), ("T2", "1", "B"))
        with self.assertRaises(ValueError):
            compute_distances(grid, {"T1": 2, "T2": 3})

    def test_zero_cost_map_and_pass_through_treasure(self):
        grid = (("A", "T1", "T2", "B"), (0, 0, 0, 0))
        values = {"T1": 4, "T2": 6}
        distances = compute_distances(grid, values)
        for source in distances.sites:
            for target in distances.sites:
                self.assertEqual(
                    distances.costs[source][target],
                    oracle_dijkstra(
                        grid,
                        distances.locations[source],
                        distances.locations[target],
                        lambda cell: 1 if str(cell) in distances.sites else int(cell),
                    ),
                )
        corridor = compute_distances((("A", "T1", "T2", "B"),), values)
        after_t2 = take_treasure(initial_state(values), "T2", corridor, values)
        self.assertIn("T1", after_t2.remaining)
        self.assertEqual(corridor.costs["A"]["T2"], 2)
        full = solve_treasure_game(distances, 2, alpha_beta=False, values=values)
        cached = solve_treasure_game(
            distances, 2, alpha_beta=False, memoize=True, values=values
        )
        self.assertEqual(
            (full.first_target, full.utility), (cached.first_target, cached.utility)
        )


if __name__ == "__main__":
    unittest.main()
