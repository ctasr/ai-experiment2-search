"""一次运行四题样例、受控对照与自设场景，并保存可复核的原始结果。"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter_ns

from game_search import TreeNode, sample_tree, tree_minimax
from plot_style import configure_chinese_font
from search_algorithms import manhattan, maze_search, path_cost_trace, weighted_search
from treasure_game import (
    SAMPLE_MAP,
    SAMPLE_VALUES,
    compute_distances,
    initial_state,
    solve_treasure_game,
    take_treasure,
    treasure_game_answer,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TEST_SEEDS = {
    "maze": 42,
    "terrain": 123,
    "tree": 7,
    "treasure": 5,
    "treasure_random_maps": 20261020,
}
SWEEP_SEED = 42
ASTAR_SWEEP_SEEDS = {"maze": 101, "terrain": 102}

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
ALTERNATE_TREASURE_MAP = (
    ("A", "1", "X", "3", "T1"),
    ("1", "1", "1", "1", "1"),
    ("T2", "X", "8", "X", "T3"),
    ("1", "1", "1", "1", "1"),
    ("T4", "2", "X", "1", "B"),
)
ALTERNATE_TREASURE_VALUES = {"T1": 8, "T2": 5, "T3": 10, "T4": 4}


def _record(
    rows: list[dict[str, object]],
    question: str,
    scene: str,
    algorithm: str,
    result: object,
    *,
    extra: dict[str, object] | None = None,
) -> None:
    stats = result.stats if hasattr(result, "stats") else result
    row: dict[str, object] = {
        "question": question,
        "scene": scene,
        "algorithm": algorithm,
    }
    for field in (
        "pushed",
        "popped",
        "expanded",
        "stale_pops",
        "visited_nodes",
        "evaluated_leaves",
        "cutoffs",
        "skipped_children",
    ):
        if hasattr(stats, field):
            row[field] = getattr(stats, field)
    row["elapsed_ns_once"] = stats.elapsed_ns
    if hasattr(result, "cost"):
        row["cost"] = result.cost
    if hasattr(result, "utility"):
        row["utility"] = result.utility
        row["first_target"] = result.first_target
    if extra:
        row.update(extra)
    rows.append(row)


def _measure_interleaved(
    rows: list[dict[str, object]],
    calls: dict[tuple[str, int | None, str], Callable[[], object]],
    repetitions: int = 101,
) -> None:
    """同一场景的算法预热后轮换先后，避免固定测量次序偏向一方。"""
    grouped: dict[tuple[str, int | None], list[tuple[str, int | None, str]]] = {}
    for key in calls:
        grouped.setdefault(key[:2], []).append(key)
    for keys in grouped.values():
        for key in keys:
            calls[key]()
        samples: dict[tuple[str, int | None, str], list[int]] = {
            key: [] for key in keys
        }
        for repeat in range(repetitions):
            rotated = keys[repeat % len(keys) :] + keys[: repeat % len(keys)]
            for key in rotated:
                began = perf_counter_ns()
                calls[key]()
                samples[key].append(perf_counter_ns() - began)
        for row in rows:
            key = (str(row["scene"]), row.get("depth"), str(row["algorithm"]))
            if key not in samples:
                continue
            values = samples[key]
            quartiles = statistics.quantiles(values, n=4)
            row["time_median_ns"] = int(statistics.median(values))
            row["time_iqr_ns"] = int(quartiles[2] - quartiles[0])


def _tree_value_trace(root: TreeNode) -> list[dict[str, object]]:
    """沿原始左到右次序记录每个结点的Minimax回传值。"""
    records: list[dict[str, object]] = []

    def visit(node: TreeNode, maximizing: bool, path: tuple[int, ...]) -> int:
        if not node.children:
            assert node.value is not None
            score = node.value
        else:
            child_values = [
                visit(child, not maximizing, path + (index,))
                for index, child in enumerate(node.children)
            ]
            score = max(child_values) if maximizing else min(child_values)
        records.append(
            {
                "path": list(path),
                "role": "LEAF"
                if not node.children
                else ("MAX" if maximizing else "MIN"),
                "value": score,
            }
        )
        return score

    visit(root, True, ())
    return records


def _treasure_decision_trace(distances, values: dict[str, int], depth: int):
    """枚举根动作的精确博弈值，并重建最优应对序列与逐回合收益。"""
    state = initial_state(values)

    def evaluate(node, levels: int) -> tuple[int, tuple[str, ...]]:
        if levels == 0 or not node.remaining:
            return node.utility, ()
        outcomes = []
        for target in node.remaining:
            child = take_treasure(node, target, distances, values)
            score, continuation = evaluate(child, levels - 1)
            outcomes.append((score, (target, *continuation)))
        return (
            max(outcomes, key=lambda item: item[0])
            if node.current_player == "A"
            else min(outcomes, key=lambda item: item[0])
        )

    option_values = {}
    best_score = None
    best_line: tuple[str, ...] = ()
    for target in state.remaining:
        child = take_treasure(state, target, distances, values)
        score, continuation = evaluate(child, depth - 1)
        option_values[target] = score
        if best_score is None or score > best_score:
            best_score, best_line = score, (target, *continuation)

    rounds = []
    node = state
    for target in best_line:
        player = node.current_player
        origin = node.a_position if player == "A" else node.b_position
        following = take_treasure(node, target, distances, values)
        cost = distances.costs[origin][target]
        rounds.append(
            {
                "player": player,
                "from": origin,
                "target": target,
                "distance": cost,
                "treasure_value": values[target],
                "gain": values[target] - cost,
                "score_a_after": following.score_a,
                "score_b_after": following.score_b,
                "utility_after": following.utility,
            }
        )
        node = following
    assert node.utility == best_score
    return {
        "root_action_values": option_values,
        "principal_variation": list(best_line),
        "rounds": rounds,
        "optimal_utility": best_score,
    }


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_maze_process_figure(plt, profile: dict[str, object]) -> None:
    """用相同地图同时标出两种启发式的出队次序与最终路径。"""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    grid = profile["grid"]
    start, goal = profile["start"], profile["goal"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    for ax, name, title in zip(
        axes,
        ("zero", "manhattan"),
        ("零启发式", "曼哈顿距离"),
    ):
        trace = profile[f"{name}_expansion_trace"]
        path = profile[f"{name}_path"]
        ax.imshow(
            [[int(str(cell) == "X") for cell in row] for row in grid],
            cmap=ListedColormap(["#FAFBFC", "#424953"]),
            vmin=0,
            vmax=1,
        )
        for index, step in enumerate(trace, start=1):
            row, col = step["position"]
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor="#CDE2F2",
                    edgecolor="none",
                    zorder=2,
                )
            )
            ax.text(
                col,
                row,
                str(index),
                ha="center",
                va="center",
                fontsize=9,
                color="#173A59",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                zorder=4,
            )
        ax.plot(
            [col for _, col in path],
            [row for row, _ in path],
            color="#D56532",
            linewidth=2.5,
            zorder=3,
        )
        for label, (row, col) in (("S", start), ("E", goal)):
            ax.text(
                col - 0.37,
                row - 0.36,
                label,
                color="#A3262A" if label == "E" else "#167347",
                fontsize=9,
                fontweight="bold",
                zorder=5,
            )
        ax.set_xticks(range(len(grid[0])))
        ax.set_yticks(range(len(grid)))
        ax.set_xticks([i - 0.5 for i in range(len(grid[0]) + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(len(grid) + 1)], minor=True)
        ax.grid(which="minor", color="#C5CDD3", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("列")
        ax.set_ylabel("行")
        ax.set_title(f"{title}：扩展{len(trace)}个节点")
    fig.suptitle("数字：有效扩展顺序；橙线：返回的最短路径", fontsize=11)
    fig.savefig(FIGURES / "maze_search_process.png", dpi=180)
    plt.close(fig)


def _save_tree_pruning_figure(plt, profile: dict[str, object]) -> None:
    """按节点出现位置绘制样例树，突出被剪叶与剪枝时的界。"""
    records = {tuple(item["path"]): item for item in profile["value_trace"]}
    leaves = sorted(path for path, item in records.items() if item["role"] == "LEAF")
    leaf_x = {path: index for index, path in enumerate(leaves)}

    def horizontal_position(path: tuple[int, ...]) -> float:
        if path in leaf_x:
            return float(leaf_x[path])
        children = sorted(
            child
            for child in records
            if len(child) == len(path) + 1 and child[:-1] == path
        )
        return (
            horizontal_position(children[0]) + horizontal_position(children[-1])
        ) / 2

    cutoffs = profile["cutoff_trace"]["left_to_right"]
    skipped = {tuple(path) for cutoff in cutoffs for path in cutoff["skipped_paths"]}
    bounds = {tuple(cutoff["node_path"]): cutoff for cutoff in cutoffs}
    positions = {path: (horizontal_position(path), len(path)) for path in records}
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for path, (x, y) in positions.items():
        if not path:
            continue
        parent_x, parent_y = positions[path[:-1]]
        ax.plot(
            (parent_x, x),
            (parent_y, y),
            color="#B0B3B8" if path in skipped else "#526A7D",
            linestyle="--" if path in skipped else "-",
            linewidth=1.7,
            zorder=1,
        )
    for path, item in records.items():
        x, y = positions[path]
        role = item["role"]
        color = (
            "#D6D8DB"
            if path in skipped
            else "#DCE9F5"
            if role == "MAX"
            else "#FCE8D2"
            if role == "MIN"
            else "#F7F8F9"
        )
        ax.scatter(
            x,
            y,
            s=1700 if role != "LEAF" else 1000,
            facecolor=color,
            edgecolor="#9A9DA0" if path in skipped else "#46515C",
            linewidth=1.2,
            zorder=2,
        )
        if path in bounds:
            bound = bounds[path]
            symbol = "≥" if role == "MAX" else "≤"
            label = f"{role}\n{symbol}{bound['best_so_far']}"
        elif path in skipped:
            label = f"{item['value']}\n未访问"
        elif role == "LEAF":
            label = str(item["value"])
        else:
            label = f"{role}\n{item['value']}"
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3)
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(3.55, -0.5)
    ax.axis("off")
    fig.suptitle("题三样例树：从左到右的Alpha-beta剪枝", fontsize=12)
    fig.text(
        0.5,
        0.06,
        "左侧MAX读到6时α=6≥β=5，因此右叶9未访问；两处6、9是共享子树的两次到达。",
        ha="center",
        fontsize=10,
    )
    fig.savefig(FIGURES / "tree_pruning_process.png", dpi=180)
    plt.close(fig)


def _save_terrain_process_figure(plt, profile: dict[str, object]) -> None:
    """把样例地图、最优路径和进入各格后的累计代价放在同一图中。"""
    from matplotlib.patches import Rectangle

    grid = profile["grid"]
    trace = profile["cost_trace"]
    path = [tuple(step["position"]) for step in trace]
    cumulative = [int(step["cumulative_cost"]) for step in trace]
    assert len(grid) == len(grid[0]) == 5
    assert cumulative[-1] == int(profile["min_cost"]) == 7

    fig, (map_ax, cost_ax) = plt.subplots(
        1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": (1, 1.25)},
        constrained_layout=True,
    )
    path_costs = {point: cost for point, cost in zip(path, cumulative)}
    colors = {1: "#EAF1F6", 3: "#F5E8C8", 8: "#E6D6D0"}
    for row, cells in enumerate(grid):
        for col, cell in enumerate(cells):
            face = "#E4F0E3" if cell in {"S", "E"} else colors[int(cell)]
            on_path = (row, col) in path_costs
            map_ax.add_patch(
                Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=face,
                          edgecolor="#D97B43" if on_path else "#BBC4C9",
                          linewidth=2 if on_path else 0.8, zorder=0)
            )
            label = str(cell)
            map_ax.text(col, row - 0.13, label, ha="center", va="center",
                        fontsize=10, color="#263747", weight="bold" if cell in {"S", "E"} else "normal", zorder=3)
            if (row, col) in path_costs:
                map_ax.text(col, row + 0.22, str(path_costs[(row, col)]),
                            ha="center", va="center", fontsize=9,
                            color="#A63C2F", weight="bold", zorder=3)
    map_ax.set_xlim(-0.55, 4.55)
    map_ax.set_ylim(4.55, -0.55)
    map_ax.set_xticks(range(5))
    map_ax.set_yticks(range(5))
    map_ax.set_xlabel("列")
    map_ax.set_ylabel("行")
    map_ax.set_aspect("equal")
    map_ax.set_title("橙框为最优路径，红字为累计代价", fontsize=10)

    steps = list(range(len(cumulative)))
    cost_ax.plot(steps, cumulative, color="#D97B43", marker="o", linewidth=2)
    cost_ax.set_xticks(steps)
    cost_ax.set_xlim(-0.3, 8.45)
    cost_ax.set_ylim(-0.4, 8.4)
    cost_ax.set_xlabel("移动次数")
    cost_ax.set_ylabel("累计代价")
    cost_ax.set_title("最后一步进入E：代价增加0", fontsize=10)
    cost_ax.grid(axis="y", color="#D9E0E5", linewidth=0.7)
    cost_ax.annotate("7 → 7", xy=(7.5, 7), xytext=(6.5, 8.0),
                     arrowprops={"arrowstyle": "->", "color": "#A63C2F"},
                     color="#A63C2F", fontsize=9)
    fig.savefig(FIGURES / "terrain_cost_process.png", dpi=180)
    plt.close(fig)


def _save_figures(
    rows: list[dict[str, object]],
    distances,
    ordering_sweep: list[dict[str, object]],
    details: dict[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_chinese_font()

    FIGURES.mkdir(exist_ok=True)
    _save_maze_process_figure(plt, details["scenes"]["maze_sample"])
    _save_terrain_process_figure(plt, details["scenes"]["terrain_sample"])
    _save_tree_pruning_figure(plt, details["scenes"]["tree_sample"])
    decision_profiles = details["scenes"]["treasure_sample"]["depths"]
    selected = [r for r in rows if r["scene"] in {"maze_sample", "terrain_sample"}]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for axis, question, title in zip(
        axes, ("Q1", "Q2"), ("题一：普通迷宫", "题二：加权地形")
    ):
        subset = [r for r in selected if r["question"] == question]
        labels = [
            {
                "zero": "零启发式",
                "manhattan": "曼哈顿距离",
                "admissible": "可采纳估计",
            }.get(str(r["algorithm"]), str(r["algorithm"]))
            for r in subset
        ]
        heights = [int(r["expanded"]) for r in subset]
        bars = axis.bar(labels, heights, color=("#6C8EBB", "#54A875"))
        axis.bar_label(bars)
        axis.set_ylabel("有效扩展节点数")
        axis.set_title(title)
    fig.savefig(FIGURES / "astar_expanded.png", dpi=180)
    plt.close(fig)

    subset = [
        r
        for r in rows
        if r["question"] == "Q4"
        and r["scene"] == "treasure_sample"
        and r.get("depth") == 4
    ]
    values = {str(r["algorithm"]): int(r["visited_nodes"]) for r in subset}
    assert len(values) == 6
    fig, ax = plt.subplots(figsize=(8, 3.1), constrained_layout=True)
    x = list(range(3))
    minimax = [values[f"minimax_{order}"] for order in ("given", "reverse", "greedy")]
    alphabeta = [values[f"alphabeta_{order}"] for order in ("given", "reverse", "greedy")]
    left = ax.bar([point - 0.18 for point in x], minimax, width=0.34,
                  label="Minimax", color="#6C8EBB")
    right = ax.bar([point + 0.18 for point in x], alphabeta, width=0.34,
                   label="Alpha-beta", color="#54A875")
    ax.bar_label(left, padding=2)
    ax.bar_label(right, padding=2)
    ax.set_xticks(x, ("给定顺序", "逆序", "即时收益排序"))
    ax.set_ylim(0, 85)
    ax.set_ylabel("访问博弈状态数")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#E1E5E8", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / "game_search_nodes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, scene, title in zip(
        axes,
        ("tree_sample", "tree_order_sensitive"),
        ("题三样例树", "访问顺序敏感的自设树"),
    ):
        subset = [r for r in rows if r["question"] == "Q3" and r["scene"] == scene]
        labels = [
            {
                "minimax_left_to_right": "Minimax原顺序",
                "minimax_reverse": "Minimax逆序",
                "alphabeta_left_to_right": "剪枝原顺序",
                "alphabeta_reverse": "剪枝逆序",
            }[str(r["algorithm"])]
            for r in subset
        ]
        bars = ax.bar(labels, [int(r["visited_nodes"]) for r in subset])
        ax.bar_label(bars)
        ax.set_title(title)
        ax.set_ylabel("访问博弈节点数")
        ax.tick_params(axis="x", labelrotation=20)
    fig.savefig(FIGURES / "tree_search_nodes.png", dpi=180)
    plt.close(fig)

    # 与 details.json 使用同一张 A* 距离矩阵。
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    matrix = distances.as_matrix()
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            ax.text(
                col,
                row,
                str(matrix[row][col]),
                ha="center",
                va="center",
                color="white" if matrix[row][col] >= 9 else "black",
            )
    ax.set_xticks(range(len(distances.sites)), distances.sites)
    ax.set_yticks(range(len(distances.sites)), distances.sites)
    ax.set_title("A*计算的六点最小移动代价")
    fig.colorbar(heatmap, ax=ax, label="最小代价")
    fig.savefig(FIGURES / "treasure_distance_matrix.png", dpi=180)
    plt.close(fig)

    differences = [
        int(row["given_nodes"]) - int(row["greedy_nodes"]) for row in ordering_sweep
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.hist(
        differences,
        bins=range(min(differences), max(differences) + 2),
        color="#6C8EBB",
        edgecolor="white",
    )
    ax.axvline(0, color="#A33A3A", linestyle="--", linewidth=1)
    ax.set_xlabel("即时收益排序节省的节点数（原顺序－排序）")
    ax.set_ylabel("宝藏价值组合数")
    fig.savefig(FIGURES / "ordering_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    labels = ("T1", "T2", "T3", "T4")
    x = list(range(len(labels)))
    for offset, depth, label, color in (
        (-0.18, "2", "H=2", "#6C8EBB"),
        (0.18, "4", "H=4", "#54A875"),
    ):
        option_values = decision_profiles[depth]["decision_trace"]["root_action_values"]
        scores = [int(option_values[target]) for target in labels]
        bars = ax.bar([point + offset for point in x], scores, width=0.34,
                      label=label, color=color)
        ax.bar_label(bars, padding=2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(-8.5, 4.5)
    ax.set_ylabel("A最终收益差")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#E1E5E8", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / "depth_decisions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    timed_calls: dict[tuple[str, int | None, str], Callable[[], object]] = {}
    source_pdf = ROOT.parent / "实验二要求_new(1).pdf"
    details: dict[str, object] = {
        "test_random_seeds": TEST_SEEDS,
        "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest()
        if source_pdf.exists()
        else None,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "search_algorithms.py",
                "game_search.py",
                "treasure_game.py",
                "run_experiment.py",
                "plot_style.py",
                "requirements.txt",
            )
        },
        "scenes": {},
    }

    for scene, grid, start, goal in (
        ("maze_sample", MAZE, (0, 0), (3, 4)),
        (
            "maze_open_20",
            tuple(
                tuple(
                    "S" if (r, c) == (0, 0) else "E" if (r, c) == (19, 19) else "."
                    for c in range(20)
                )
                for r in range(20)
            ),
            (0, 0),
            (19, 19),
        ),
    ):
        outcomes = {}
        sample_expansion_traces: dict[str, list[dict[str, object]]] = {
            "zero": [],
            "manhattan": [],
        }
        for h in ("zero", "manhattan"):
            result = maze_search(grid, start, goal, h)
            outcomes[h] = result
            _record(
                rows,
                "Q1",
                scene,
                h,
                result,
            )
            timed_calls[(scene, None, h)] = partial(maze_search, grid, start, goal, h)
        assert outcomes["zero"].cost == outcomes["manhattan"].cost
        if scene == "maze_sample":
            for name, trace in sample_expansion_traces.items():
                traced = maze_search(grid, start, goal, name, expansion_trace=trace)
                assert (traced.path, traced.cost, traced.expanded) == (
                    outcomes[name].path,
                    outcomes[name].cost,
                    outcomes[name].expanded,
                )
        details["scenes"][scene] = {
            "grid": grid,
            "start": start,
            "goal": goal,
            "zero_cost": outcomes["zero"].cost,
            "manhattan_cost": outcomes["manhattan"].cost,
            "manhattan_path": outcomes["manhattan"].path,
            "zero_path": outcomes["zero"].path,
            "zero_expansion_trace": sample_expansion_traces["zero"],
            "manhattan_expansion_trace": sample_expansion_traces["manhattan"],
            "path_g_h_f": [
                {
                    "position": position,
                    "g": index,
                    "h": manhattan(position, goal),
                    "f": index + manhattan(position, goal),
                }
                for index, position in enumerate(outcomes["manhattan"].path)
            ],
        }

    astar_sweep = []
    for task in ("maze", "terrain"):
        rng = random.Random(ASTAR_SWEEP_SEEDS[task])
        size = 20 if task == "maze" else 12
        for case in range(100):
            if task == "maze":
                grid = [
                    ["X" if rng.random() < 0.2 else "." for _ in range(size)]
                    for _ in range(size)
                ]
                grid[0][0], grid[-1][-1] = "S", "E"
                baseline = maze_search(grid, (0, 0), (size - 1, size - 1), "zero")
                improved = maze_search(grid, (0, 0), (size - 1, size - 1), "manhattan")
            else:
                grid = [
                    [
                        "X" if rng.random() < 0.15 else rng.choice((1, 3, 8))
                        for _ in range(size)
                    ]
                    for _ in range(size)
                ]
                grid[0][0], grid[-1][-1] = "S", "E"
                baseline = weighted_search(grid, (0, 0), (size - 1, size - 1), "zero")
                improved = weighted_search(
                    grid, (0, 0), (size - 1, size - 1), "admissible"
                )
            assert baseline.cost == improved.cost
            astar_sweep.append(
                {
                    "task": task,
                    "case": case,
                    "seed": ASTAR_SWEEP_SEEDS[task],
                    "size": size,
                    "optimal_cost": baseline.cost,
                    "baseline_expanded": baseline.expanded,
                    "heuristic_expanded": improved.expanded,
                    "nodes_saved": baseline.expanded - improved.expanded,
                }
            )
    _save_csv(RESULTS / "astar_sweep.csv", astar_sweep)
    details["astar_sweep"] = {}
    for task in ("maze", "terrain"):
        subset = [row for row in astar_sweep if row["task"] == task]
        saved = [row["nodes_saved"] for row in subset]
        details["astar_sweep"][task] = {
            "seed": ASTAR_SWEEP_SEEDS[task],
            "cases": len(subset),
            "reachable": sum(row["optimal_cost"] is not None for row in subset),
            "heuristic_better": sum(value > 0 for value in saved),
            "same_nodes": sum(value == 0 for value in saved),
            "heuristic_worse": sum(value < 0 for value in saved),
            "median_nodes_saved": statistics.median(saved),
            "min_nodes_saved": min(saved),
            "max_nodes_saved": max(saved),
        }

    expensive_detour = (
        ("S", 8, 8, 8, "E"),
        (1, 1, 1, 1, 1),
    )
    for scene, grid, start, goal in (
        ("terrain_sample", TERRAIN, (0, 0), (4, 4)),
        ("terrain_detour", expensive_detour, (0, 0), (0, 4)),
    ):
        outcomes = {}
        for h in ("zero", "admissible"):
            result = weighted_search(grid, start, goal, h)
            outcomes[h] = result
            _record(
                rows,
                "Q2",
                scene,
                h,
                result,
            )
            timed_calls[(scene, None, h)] = partial(
                weighted_search, grid, start, goal, h
            )
        assert outcomes["zero"].cost == outcomes["admissible"].cost
        chosen = outcomes["admissible"]
        trace = path_cost_trace(
            grid, chosen.path, lambda cell: 0 if cell == "E" else int(cell)
        )
        assert trace[-1]["cumulative_cost"] == chosen.cost
        details["scenes"][scene] = {
            "grid": grid,
            "start": start,
            "goal": goal,
            "min_cost": chosen.cost,
            "path": chosen.path,
            "cost_trace": trace,
        }

    shared = sample_tree()
    order_sensitive = TreeNode(
        children=[
            TreeNode(children=[TreeNode(8), TreeNode(9), TreeNode(10)]),
            TreeNode(children=[TreeNode(2), TreeNode(3), TreeNode(4)]),
            TreeNode(children=[TreeNode(5), TreeNode(6), TreeNode(7)]),
        ]
    )
    for scene, root in (
        ("tree_sample", shared),
        ("tree_order_sensitive", order_sensitive),
    ):
        scores = []
        pruning_records = {}
        cutoff_records = {}
        for pruning in (False, True):
            for order in ("left_to_right", "reverse"):
                score, stats = tree_minimax(root, alpha_beta=pruning, order=order)
                scores.append(score)
                algorithm = ("alphabeta" if pruning else "minimax") + "_" + order
                if pruning:
                    pruning_records[order] = stats.pruned_branches
                    cutoff_records[order] = stats.cutoff_trace
                _record(
                    rows,
                    "Q3",
                    scene,
                    algorithm,
                    stats,
                    extra={"score": score},
                )
                timed_calls[(scene, None, algorithm)] = partial(
                    tree_minimax, root, alpha_beta=pruning, order=order
                )
        assert len(set(scores)) == 1
        details["scenes"][scene] = {
            "best_score": scores[0],
            "pruned_branches": pruning_records,
            "cutoff_trace": cutoff_records,
            "value_trace": _tree_value_trace(root),
            "leaf_values": [3, 5, 6, 9, 6, 9, 1, 2]
            if scene == "tree_sample"
            else [8, 9, 10, 2, 3, 4, 5, 6, 7],
        }

    distances = compute_distances(SAMPLE_MAP, SAMPLE_VALUES)
    matrix = distances.as_matrix()
    assert all(matrix[i][j] == matrix[j][i] for i in range(6) for j in range(6))
    details["distance_matrix"] = {
        "sites": distances.sites,
        "matrix": matrix,
        "grid": SAMPLE_MAP,
        "astar_runs": distances.astar_runs,
        "astar_total_expanded": distances.total_expanded,
    }
    _save_csv(
        RESULTS / "distance_matrix.csv",
        [
            {"source": source, **dict(zip(distances.sites, costs))}
            for source, costs in zip(distances.sites, matrix)
        ],
    )

    value_scenes = (
        ("treasure_sample", SAMPLE_VALUES),
        ("treasure_high_value", {"T1": 30, "T2": 12, "T3": 12, "T4": 29}),
    )
    for scene, values in value_scenes:
        details["scenes"][scene] = {"values": values, "depths": {}}
        for depth in (2, 4):
            results = []
            pruning_records = {}
            for pruning in (False, True):
                for order in ("given", "reverse", "greedy"):
                    result = solve_treasure_game(
                        distances, depth, alpha_beta=pruning, order=order, values=values
                    )
                    results.append(result)
                    algorithm = ("alphabeta" if pruning else "minimax") + "_" + order
                    if pruning:
                        pruning_records[order] = result.stats.pruned_branches
                    _record(
                        rows,
                        "Q4",
                        scene,
                        algorithm,
                        result,
                        extra={"depth": depth},
                    )
                    timed_calls[(scene, depth, algorithm)] = partial(
                        solve_treasure_game,
                        distances,
                        depth,
                        alpha_beta=pruning,
                        order=order,
                        values=values,
                    )
            assert len({r.utility for r in results}) == 1
            # 同分时首步随合法访问顺序变化，不能误判为搜索结果不一致。
            details["scenes"][scene]["depths"][str(depth)] = {
                "first_target": results[0].first_target,
                "tied_first_targets": sorted({r.first_target for r in results}),
                "utility": results[0].utility,
                "pruned_branches": pruning_records,
                "decision_trace": _treasure_decision_trace(distances, values, depth),
            }
            assert (
                details["scenes"][scene]["depths"][str(depth)]["decision_trace"][
                    "optimal_utility"
                ]
                == results[0].utility
            )

    # 同一地图上系统改变四个宝藏价值，评估排序策略的适用范围与反例。
    rng = random.Random(SWEEP_SEED)
    ordering_sweep: list[dict[str, object]] = []
    for case in range(200):
        values = {name: rng.randint(1, 30) for name in SAMPLE_VALUES}
        full = solve_treasure_game(distances, 4, alpha_beta=False, values=values)
        given = solve_treasure_game(distances, 4, order="given", values=values)
        greedy = solve_treasure_game(distances, 4, order="greedy", values=values)
        assert full.utility == given.utility == greedy.utility
        ordering_sweep.append(
            {
                "case": case,
                **values,
                "utility": full.utility,
                "given_nodes": given.stats.visited_nodes,
                "greedy_nodes": greedy.stats.visited_nodes,
                "given_first_target": given.first_target,
                "greedy_first_target": greedy.first_target,
            }
        )
    _save_csv(RESULTS / "ordering_sweep.csv", ordering_sweep)
    differences = [
        int(row["given_nodes"]) - int(row["greedy_nodes"]) for row in ordering_sweep
    ]
    details["ordering_sweep"] = {
        "seed": SWEEP_SEED,
        "cases": len(ordering_sweep),
        "value_range_inclusive": [1, 30],
        "greedy_better": sum(value > 0 for value in differences),
        "same_nodes": sum(value == 0 for value in differences),
        "greedy_worse": sum(value < 0 for value in differences),
        "median_nodes_saved": statistics.median(differences),
        "min_nodes_saved": min(differences),
        "max_nodes_saved": max(differences),
    }
    for depth, first, score in (("2", "T2", -1), ("4", "T3", 3)):
        official = details["scenes"]["treasure_sample"]["depths"][depth]
        assert official["first_target"] == first and official["utility"] == score
    official_answers = {
        f"H={depth}": {
            "Minimax": treasure_game_answer(
                distances, depth, alpha_beta=False, values=SAMPLE_VALUES
            ),
            "Alpha-beta": treasure_game_answer(
                distances, depth, alpha_beta=True, values=SAMPLE_VALUES
            ),
        }
        for depth in (2, 4)
    }
    (RESULTS / "official_answers.json").write_text(
        json.dumps(official_answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    alternate_distances = compute_distances(
        ALTERNATE_TREASURE_MAP, ALTERNATE_TREASURE_VALUES
    )
    alternate_matrix = alternate_distances.as_matrix()
    _save_csv(
        RESULTS / "distance_matrix_alternate.csv",
        [
            {"source": source, **dict(zip(alternate_distances.sites, costs))}
            for source, costs in zip(alternate_distances.sites, alternate_matrix)
        ],
    )
    alternate_details = {
        "grid": ALTERNATE_TREASURE_MAP,
        "values": ALTERNATE_TREASURE_VALUES,
        "distance_matrix": alternate_matrix,
        "depths": {},
    }
    for depth in (2, 4):
        variants = {}
        for algorithm, pruning, ordering in (
            ("minimax_given", False, "given"),
            ("alphabeta_given", True, "given"),
            ("alphabeta_greedy", True, "greedy"),
        ):
            result = solve_treasure_game(
                alternate_distances,
                depth,
                alpha_beta=pruning,
                order=ordering,
                values=ALTERNATE_TREASURE_VALUES,
            )
            variants[algorithm] = result
            _record(
                rows,
                "Q4",
                "treasure_alternate_map",
                algorithm,
                result,
                extra={"depth": depth},
            )
            timed_calls[("treasure_alternate_map", depth, algorithm)] = partial(
                solve_treasure_game,
                alternate_distances,
                depth,
                alpha_beta=pruning,
                order=ordering,
                values=ALTERNATE_TREASURE_VALUES,
            )
        assert len({result.utility for result in variants.values()}) == 1
        decision_trace = _treasure_decision_trace(
            alternate_distances, ALTERNATE_TREASURE_VALUES, depth
        )
        assert decision_trace["optimal_utility"] == variants["minimax_given"].utility
        alternate_details["depths"][str(depth)] = {
            "first_target": variants["minimax_given"].first_target,
            "utility": variants["minimax_given"].utility,
            "decision_trace": decision_trace,
        }
    details["scenes"]["treasure_alternate_map"] = alternate_details

    details["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "implementation": platform.python_implementation(),
    }
    _measure_interleaved(rows, timed_calls)
    _save_csv(RESULTS / "comparison.csv", rows)
    deterministic_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"elapsed_ns_once", "time_median_ns", "time_iqr_ns"}
        }
        for row in rows
    ]
    (RESULTS / "deterministic_results.json").write_text(
        json.dumps(
            {
                "comparisons": deterministic_rows,
                "distance_matrix": matrix,
                "alternate_distance_matrix": alternate_matrix,
                "astar_sweep": astar_sweep,
                "ordering_sweep": ordering_sweep,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_figures(rows, distances, ordering_sweep, details)
    print("四题样例、对照、自设场景均已完成。")
    print("题一最短路径：", details["scenes"]["maze_sample"]["manhattan_path"])
    print("题二最小代价：", details["scenes"]["terrain_sample"]["min_cost"])
    print("题三最佳得分：", details["scenes"]["tree_sample"]["best_score"])
    print("距离矩阵顺序：", ", ".join(distances.sites))
    for row in matrix:
        print(" ".join(f"{value:2d}" for value in row))
    for depth in ("2", "4"):
        outcome = details["scenes"]["treasure_sample"]["depths"][depth]
        print(
            f"题四 H={depth}: 首步{outcome['first_target']}，收益差{outcome['utility']}"
        )
    print("结果位置：", RESULTS)


if __name__ == "__main__":
    main()
