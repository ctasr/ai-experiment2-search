"""实验二题四：A* 距离矩阵与轮流取宝的 Minimax/Alpha-beta。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from time import perf_counter_ns

from search_algorithms import Grid, Point, SearchResult, astar, manhattan

SAMPLE_MAP: tuple[tuple[str, ...], ...] = (
    ("A", "1", "1", "X", "2", "1", "T1"),
    ("1", "X", "6", "X", "1", "X", "1"),
    ("1", "1", "T2", "1", "1", "1", "1"),
    ("X", "1", "X", "8", "X", "1", "X"),
    ("1", "1", "1", "1", "T3", "1", "1"),
    ("1", "X", "9", "X", "1", "X", "1"),
    ("T4", "1", "1", "X", "1", "1", "B"),
)
SAMPLE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


@dataclass(frozen=True)
class DistanceTable:
    sites: tuple[str, ...]
    locations: dict[str, Point]
    costs: dict[str, dict[str, int]]
    astar_runs: int
    total_expanded: int
    total_elapsed_ns: int

    def as_matrix(self) -> list[list[int]]:
        return [[self.costs[a][b] for b in self.sites] for a in self.sites]


def _locations(grid: Grid, values: dict[str, int]) -> dict[str, Point]:
    names = ("A", "B", *values.keys())
    found: dict[str, Point] = {}
    for row, cells in enumerate(grid):
        for col, cell in enumerate(cells):
            if str(cell) in names:
                if str(cell) in found:
                    raise ValueError(f"地图中的 {cell} 出现多次")
                found[str(cell)] = (row, col)
    if set(found) != set(names):
        raise ValueError(f"地图缺少站点：{set(names) - set(found)}")
    return found


def compute_distances(
    grid: Grid = SAMPLE_MAP, values: dict[str, int] = SAMPLE_VALUES
) -> DistanceTable:
    """用 A* 计算所有站点之间的有向最短距离。"""
    sites = ("A", "B", *values.keys())
    places = _locations(grid, values)
    site_names = set(sites)

    def cell_cost(cell: str | int) -> int:
        return 1 if str(cell) in site_names else int(cell)

    positive_costs = [
        cell_cost(cell) for row in grid for cell in row if str(cell) != "X"
    ]
    minimum_cost = min(positive_costs)
    costs: dict[str, dict[str, int]] = {name: {} for name in sites}
    runs = expanded = elapsed = 0
    for source in sites:
        for destination in sites:
            if source == destination:
                costs[source][destination] = 0
                continue
            result: SearchResult = astar(
                grid,
                places[source],
                places[destination],
                cell_cost,
                lambda p, q: minimum_cost * manhattan(p, q),
            )
            if result.cost is None:
                raise ValueError(f"{source} 到 {destination} 不可达，不能继续博弈")
            costs[source][destination] = result.cost
            runs += 1
            expanded += result.expanded
            elapsed += result.elapsed_ns
    return DistanceTable(sites, places, costs, runs, expanded, elapsed)


@dataclass(frozen=True)
class GameState:
    a_position: str
    b_position: str
    remaining: tuple[str, ...]
    score_a: int
    score_b: int
    current_player: str

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


@dataclass
class GameStats:
    visited_nodes: int = 0
    evaluated_leaves: int = 0
    cutoffs: int = 0
    skipped_children: int = 0
    cache_hits: int = 0
    cache_entries: int = 0
    ordering_evaluations: int = 0
    elapsed_ns: int = 0
    pruned_branches: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class GameResult:
    first_target: str
    utility: int
    stats: GameStats


def initial_state(values: dict[str, int] = SAMPLE_VALUES) -> GameState:
    return GameState("A", "B", tuple(values), 0, 0, "A")


def take_treasure(
    state: GameState,
    target: str,
    distances: DistanceTable,
    values: dict[str, int] = SAMPLE_VALUES,
) -> GameState:
    if target not in state.remaining:
        raise ValueError(f"宝藏 {target} 已被取走或不存在")
    remaining = tuple(item for item in state.remaining if item != target)
    if state.current_player == "A":
        gain = values[target] - distances.costs[state.a_position][target]
        return GameState(
            target,
            state.b_position,
            remaining,
            state.score_a + gain,
            state.score_b,
            "B",
        )
    gain = values[target] - distances.costs[state.b_position][target]
    return GameState(
        state.a_position, target, remaining, state.score_a, state.score_b + gain, "A"
    )


def solve_treasure_game(
    distances: DistanceTable,
    depth: int,
    *,
    alpha_beta: bool = True,
    order: str = "given",
    values: dict[str, int] = SAMPLE_VALUES,
    start: GameState | None = None,
    memoize: bool = False,
) -> GameResult:
    """H 层搜索。正式基线顺序为 T1,T2,T3,T4；排序仅用于额外对照。"""
    if depth < 1:
        raise ValueError("搜索深度至少为 1")
    if order not in {"given", "reverse", "greedy", "lookahead"}:
        raise ValueError("order 必须是 given、reverse、greedy 或 lookahead")
    if memoize and alpha_beta:
        raise ValueError("剪枝节点可能只得到界，不能作为精确Minimax值缓存")
    state = start or initial_state(values)
    if not state.remaining:
        raise ValueError("根状态没有可选宝藏")
    stats = GameStats()
    began = perf_counter_ns()
    cache: dict[tuple[str, str, tuple[str, ...], str, int], int] = {}

    def ordered_targets(node: GameState) -> list[str]:
        targets = list(node.remaining)
        if order == "reverse":
            targets.reverse()
        elif order == "greedy":
            # 只影响访问顺序，不改变博弈树或效用。MAX 先试当前收益高的动作，
            # MIN 也先试自己当前收益高的动作，以尽早收紧 alpha/beta。
            origin = node.a_position if node.current_player == "A" else node.b_position
            targets.sort(
                key=lambda item: (
                    -(values[item] - distances.costs[origin][item]),
                    node.remaining.index(item),
                )
            )
        elif order == "lookahead":
            # 用下一回合的即时最优应对给当前动作排序；仅改变次序，不截断博弈。
            estimates: dict[str, int] = {}
            for target in targets:
                child = take_treasure(node, target, distances, values)
                stats.ordering_evaluations += 1
                if child.remaining:
                    replies = []
                    for reply in child.remaining:
                        replies.append(
                            take_treasure(child, reply, distances, values).utility
                        )
                        stats.ordering_evaluations += 1
                    estimates[target] = (
                        max(replies) if child.current_player == "A" else min(replies)
                    )
                else:
                    estimates[target] = child.utility
            targets.sort(
                key=lambda item: (
                    -estimates[item] if node.current_player == "A" else estimates[item],
                    node.remaining.index(item),
                )
            )
        return targets

    def visit(
        node: GameState,
        levels: int,
        alpha: float,
        beta: float,
        history: tuple[str, ...],
    ) -> int:
        stats.visited_nodes += 1
        if levels == 0 or not node.remaining:
            stats.evaluated_leaves += 1
            return node.utility
        key = (
            node.a_position,
            node.b_position,
            node.remaining,
            node.current_player,
            levels,
        )
        if memoize and key in cache:
            stats.cache_hits += 1
            return node.utility + cache[key]
        maximizing = node.current_player == "A"
        best = -inf if maximizing else inf
        targets = ordered_targets(node)
        for index, target in enumerate(targets):
            child = take_treasure(node, target, distances, values)
            score = visit(child, levels - 1, alpha, beta, history + (target,))
            if maximizing:
                best = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if alpha_beta and alpha >= beta and index + 1 < len(targets):
                stats.cutoffs += 1
                stats.skipped_children += len(targets) - index - 1
                stats.pruned_branches.extend(
                    [*history, skipped_target]
                    for skipped_target in targets[index + 1 :]
                )
                break
        if memoize:
            # 相同局面的未来收益差与此前累计分数无关，只缓存未来增量。
            cache[key] = int(best) - node.utility
        return int(best)

    # 明确枚举根动作，便于返回首步并对同分动作使用固定顺序。
    maximizing_root = state.current_player == "A"
    best = -inf if maximizing_root else inf
    selected = ""
    alpha, beta = -inf, inf
    stats.visited_nodes += 1
    for target in ordered_targets(state):
        child = take_treasure(state, target, distances, values)
        score = visit(child, depth - 1, alpha, beta, (target,))
        if score > best if maximizing_root else score < best:
            best, selected = score, target
        if maximizing_root:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
    stats.cache_entries = len(cache)
    stats.elapsed_ns = perf_counter_ns() - began
    return GameResult(selected, int(best), stats)


def treasure_game_answer(
    distances: DistanceTable,
    depth: int,
    *,
    alpha_beta: bool = True,
    values: dict[str, int] = SAMPLE_VALUES,
) -> dict[str, str | int]:
    """按题目测试用例的字段名返回最佳首步和收益差。"""
    result = solve_treasure_game(distances, depth, alpha_beta=alpha_beta, values=values)
    return {"best_target": result.first_target, "best_value": result.utility}
