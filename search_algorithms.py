"""实验二题一、题二共用的 A* 核心。坐标统一为 (行, 列)。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter_ns

Cell = str | int
Point = tuple[int, int]
Grid = Sequence[Sequence[Cell]]


@dataclass(frozen=True)
class SearchResult:
    path: tuple[Point, ...]
    cost: int | None
    pushed: int
    popped: int
    expanded: int
    stale_pops: int
    elapsed_ns: int


@dataclass(frozen=True)
class PotentialResult:
    """固定终点的精确剩余代价；构建成本必须计入多查询对照。"""

    costs: dict[Point, int]
    goal: Point
    grid_signature: tuple[tuple[str, ...], ...]
    expanded: int
    elapsed_ns: int


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _validate_grid(grid: Grid, start: Point, goal: Point) -> None:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是非空的矩形二维数组")
    for name, point in (("起点", start), ("终点", goal)):
        if len(point) != 2 or not (
            0 <= point[0] < len(grid) and 0 <= point[1] < len(grid[0])
        ):
            raise ValueError(f"{name}超出地图范围：{point}")
        if str(grid[point[0]][point[1]]) == "X":
            raise ValueError(f"{name}不能是障碍：{point}")


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    enter_cost: Callable[[Cell], int],
    heuristic: Callable[[Point, Point], int],
    *,
    tie_break: str = "deeper",
    expansion_trace: list[dict[str, object]] | None = None,
) -> SearchResult:
    """求最小进入代价路径；同一格发现更小 g 时允许重新入队。"""
    _validate_grid(grid, start, goal)
    if tie_break not in {"deeper", "fifo", "shallower"}:
        raise ValueError("tie_break 必须是 deeper、fifo 或 shallower")

    def secondary(g: int) -> int:
        return -g if tie_break == "deeper" else (g if tie_break == "shallower" else 0)

    began = perf_counter_ns()
    serial = count()
    frontier: list[tuple[int, int, int, int, Point]] = []
    heappush(frontier, (heuristic(start, goal), 0, next(serial), 0, start))
    best_g = {start: 0}
    parent: dict[Point, Point] = {}
    pushed, popped, expanded, stale_pops = 1, 0, 0, 0

    while frontier:
        f, _, _, g, current = heappop(frontier)
        popped += 1
        if g != best_g[current]:
            stale_pops += 1
            continue
        expanded += 1
        if current == goal:
            if expansion_trace is not None:
                expansion_trace.append(
                    {
                        "position": current,
                        "parent": parent.get(current),
                        "g": g,
                        "h": f - g,
                        "f": f,
                        "relaxed_neighbors": [],
                    }
                )
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return SearchResult(
                tuple(path),
                g,
                pushed,
                popped,
                expanded,
                stale_pops,
                perf_counter_ns() - began,
            )

        row, col = current
        relaxed_neighbors: list[dict[str, object]] = []
        # 固定顺序使同代价路径和计数可复现：上、右、下、左。
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            cell = grid[nr][nc]
            if str(cell) == "X":
                continue
            step = enter_cost(cell)
            if step < 0:
                raise ValueError("A* 不支持负边权")
            nxt = (nr, nc)
            candidate = g + step
            if candidate < best_g.get(nxt, inf):
                previous_g = best_g.get(nxt)
                best_g[nxt] = candidate
                parent[nxt] = current
                # f 相同节点的次序是实验变量；默认保持原有的较大 g 优先。
                next_f = candidate + heuristic(nxt, goal)
                heappush(
                    frontier,
                    (
                        next_f,
                        secondary(candidate),
                        next(serial),
                        candidate,
                        nxt,
                    ),
                )
                pushed += 1
                if expansion_trace is not None:
                    relaxed_neighbors.append(
                        {
                            "position": nxt,
                            "step_cost": step,
                            "previous_g": previous_g,
                            "new_g": candidate,
                            "h": next_f - candidate,
                            "f": next_f,
                        }
                    )
        if expansion_trace is not None:
            expansion_trace.append(
                {
                    "position": current,
                    "parent": parent.get(current),
                    "g": g,
                    "h": f - g,
                    "f": f,
                    "relaxed_neighbors": relaxed_neighbors,
                }
            )
    return SearchResult(
        (), None, pushed, popped, expanded, stale_pops, perf_counter_ns() - began
    )


def maze_search(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic_name: str = "manhattan",
    *,
    tie_break: str = "deeper",
    expansion_trace: list[dict[str, object]] | None = None,
) -> SearchResult:
    """题一。普通通行格每步代价 1。"""
    if heuristic_name not in {"manhattan", "zero"}:
        raise ValueError("题一只支持 manhattan 或 zero 启发式")
    h = manhattan if heuristic_name == "manhattan" else lambda _p, _q: 0
    return astar(
        grid,
        start,
        goal,
        lambda _cell: 1,
        h,
        tie_break=tie_break,
        expansion_trace=expansion_trace,
    )


def shortest_maze_path(start: Point, goal: Point, grid: Grid) -> list[Point]:
    """题一要求的直接输出接口：最短路径，无路返回 []。"""
    return list(maze_search(grid, start, goal).path)


def _terrain_cost(cell: Cell) -> int:
    if str(cell) == "E":
        return 0  # 题二明确规定进入 E 的代价为 0。
    if str(cell) == "S":
        return 1  # 仅在重新进入 S 时使用；起始位置不收取代价。
    try:
        value = int(cell)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法加权格：{cell!r}") from exc
    if value < 0:
        raise ValueError("地形权重不能为负")
    return value


def weighted_search(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic_name: str = "admissible",
    *,
    tie_break: str = "deeper",
) -> SearchResult:
    """题二。终点 E 零代价，因此普通曼哈顿距离未必可采纳。"""
    _validate_grid(grid, start, goal)
    costs = [
        _terrain_cost(cell)
        for row in grid
        for cell in row
        if str(cell) not in {"X", "E"}
    ]
    min_other = min(costs, default=0)
    goal_cost = _terrain_cost(grid[goal[0]][goal[1]])
    if heuristic_name == "zero":
        h = lambda _p, _q: 0
    elif heuristic_name == "admissible":
        # 最短步数为 d。若 d>0，至少 d-1 步进入普通格，最后一步进入终点。
        h = lambda p, q: (
            max(0, manhattan(p, q) - 1) * min_other + (goal_cost if p != q else 0)
        )
    else:
        raise ValueError("题二只支持 admissible 或 zero 启发式")
    return astar(grid, start, goal, _terrain_cost, h, tie_break=tie_break)


def reverse_cost_potential(grid: Grid, goal: Point) -> PotentialResult:
    """反向Dijkstra。原边p->q的代价为进入q，因此反向松弛代价为进入当前格。"""
    _validate_grid(grid, goal, goal)
    began = perf_counter_ns()
    signature = tuple(tuple(str(cell) for cell in row) for row in grid)
    frontier: list[tuple[int, Point]] = [(0, goal)]
    costs: dict[Point, int] = {goal: 0}
    expanded = 0
    while frontier:
        distance, current = heappop(frontier)
        if distance != costs[current]:
            continue
        expanded += 1
        row, col = current
        reverse_step = _terrain_cost(grid[row][col])
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            previous = (row + dr, col + dc)
            if not (0 <= previous[0] < len(grid) and 0 <= previous[1] < len(grid[0])):
                continue
            if str(grid[previous[0]][previous[1]]) == "X":
                continue
            candidate = distance + reverse_step
            if candidate < costs.get(previous, inf):
                costs[previous] = candidate
                heappush(frontier, (candidate, previous))
    return PotentialResult(costs, goal, signature, expanded, perf_counter_ns() - began)


def weighted_search_with_potential(
    grid: Grid,
    start: Point,
    goal: Point,
    potential: PotentialResult,
) -> SearchResult:
    """仅用于相同地图和终点的重复查询；不替代题二的默认接口。"""
    _validate_grid(grid, start, goal)
    signature = tuple(tuple(str(cell) for cell in row) for row in grid)
    if potential.goal != goal or potential.grid_signature != signature:
        raise ValueError("代价表必须对应当前地图和终点")
    return astar(
        grid,
        start,
        goal,
        _terrain_cost,
        lambda point, _goal: potential.costs.get(point, 0),
    )


def minimum_terrain_cost(start: Point, goal: Point, grid: Grid) -> int:
    """题二要求的直接输出接口：最小总代价，无路返回 -1。"""
    result = weighted_search(grid, start, goal)
    return -1 if result.cost is None else result.cost


def path_cost_trace(
    grid: Grid, path: Sequence[Point], enter_cost: Callable[[Cell], int]
) -> list[dict[str, object]]:
    """按实际路径列出每一步进入代价和累计代价，供实验记录核算。"""
    total = 0
    trace = []
    for index, (row, col) in enumerate(path):
        step = 0 if index == 0 else enter_cost(grid[row][col])
        total += step
        trace.append(
            {
                "position": [row, col],
                "cell": str(grid[row][col]),
                "enter_cost": step,
                "cumulative_cost": total,
            }
        )
    return trace
