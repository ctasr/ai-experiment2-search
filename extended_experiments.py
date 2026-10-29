"""补充实验：A*同分顺序、精确Minimax缓存及其适用边界。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter_ns

from game_search import TreeNode, tree_minimax
from plot_style import configure_chinese_font
from search_algorithms import maze_search, weighted_search
from treasure_game import compute_distances, solve_treasure_game

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = {"astar": 20261015, "game": 20261016, "tree": 20261017}
TIE_BREAKS = ("deeper", "fifo", "shallower")
SIX_TREASURE_MAP = (
    ("A", "1", "T1", "1", "T2"),
    ("1", "1", "1", "1", "1"),
    ("T3", "1", "T4", "1", "T5"),
    ("1", "1", "1", "1", "1"),
    ("T6", "1", "1", "1", "B"),
)


def choose_cell(rng: random.Random, task: str, density: float) -> str | int:
    if rng.random() < density:
        return "X"
    if task == "maze":
        return "."
    weights = (0, 1, 3, 8) if task == "terrain_zero" else (1, 3, 8)
    return rng.choice(weights)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def astar_experiment() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEEDS["astar"])
    rows = []
    summary = []
    for task in ("maze", "terrain_positive", "terrain_zero"):
        for density in (0.0, 0.15, 0.3, 0.45):
            for case in range(40):
                size = 16
                grid = [
                    [choose_cell(rng, task, density) for _ in range(size)]
                    for _ in range(size)
                ]
                grid[0][0], grid[-1][-1] = "S", "E"
                grid_sha256 = hashlib.sha256(
                    json.dumps(grid, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                results = {
                    order: (
                        maze_search(grid, (0, 0), (size - 1, size - 1), tie_break=order)
                        if task == "maze"
                        else weighted_search(
                            grid, (0, 0), (size - 1, size - 1), tie_break=order
                        )
                    )
                    for order in TIE_BREAKS
                }
                if len({result.cost for result in results.values()}) != 1:
                    raise AssertionError("同分顺序不能改变最优代价")
                for order, result in results.items():
                    rows.append(
                        {
                            "task": task,
                            "obstacle_density": density,
                            "case": case,
                            "seed": SEEDS["astar"],
                            "grid_sha256": grid_sha256,
                            "tie_break": order,
                            "optimal_cost": result.cost,
                            "expanded": result.expanded,
                            "pushed": result.pushed,
                            "elapsed_ns_once": result.elapsed_ns,
                        }
                    )
            for order in TIE_BREAKS:
                group = [
                    row
                    for row in rows
                    if row["task"] == task
                    and row["obstacle_density"] == density
                    and row["tie_break"] == order
                ]
                reachable = [row for row in group if row["optimal_cost"] is not None]
                summary.append(
                    {
                        "task": task,
                        "obstacle_density": density,
                        "tie_break": order,
                        "cases": len(group),
                        "distinct_maps": len({row["grid_sha256"] for row in group}),
                        "reachable": len(reachable),
                        "median_expanded_all": statistics.median(
                            row["expanded"] for row in group
                        ),
                        "median_expanded_reachable": (
                            statistics.median(row["expanded"] for row in reachable)
                            if reachable
                            else None
                        ),
                        "mean_expanded": round(
                            statistics.mean(row["expanded"] for row in group), 2
                        ),
                    }
                )
    return rows, summary


def game_experiment() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEEDS["game"])
    values0 = {f"T{i}": i + 4 for i in range(1, 7)}
    distances = compute_distances(SIX_TREASURE_MAP, values0)
    rows = []
    summary = []
    # 同一case在各深度复用相同价值，只改变搜索深度。
    value_cases = [
        {name: rng.randint(1, 20) for name in values0} for _ in range(40)
    ]
    for depth in (2, 4, 6):
        for case, values in enumerate(value_cases):
            variants = {
                "minimax": solve_treasure_game(
                    distances, depth, alpha_beta=False, values=values
                ),
                "minimax_cache": solve_treasure_game(
                    distances, depth, alpha_beta=False, memoize=True, values=values
                ),
                "alphabeta_given": solve_treasure_game(
                    distances, depth, order="given", values=values
                ),
                "alphabeta_greedy": solve_treasure_game(
                    distances, depth, order="greedy", values=values
                ),
            }
            if len({result.utility for result in variants.values()}) != 1:
                raise AssertionError("缓存或排序改变了精确博弈值")
            if (
                variants["minimax"].first_target
                != variants["minimax_cache"].first_target
            ):
                raise AssertionError("缓存改变了同顺序下的首步选择")
            calls = {
                "minimax": partial(
                    solve_treasure_game,
                    distances,
                    depth,
                    alpha_beta=False,
                    values=values,
                ),
                "minimax_cache": partial(
                    solve_treasure_game,
                    distances,
                    depth,
                    alpha_beta=False,
                    memoize=True,
                    values=values,
                ),
                "alphabeta_given": partial(
                    solve_treasure_game, distances, depth, order="given", values=values
                ),
                "alphabeta_greedy": partial(
                    solve_treasure_game, distances, depth, order="greedy", values=values
                ),
            }
            names = tuple(calls)
            for call in calls.values():
                call()
            samples = {name: [] for name in names}
            for repeat in range(11):
                rotated = names[repeat % len(names) :] + names[: repeat % len(names)]
                for name in rotated:
                    began = perf_counter_ns()
                    calls[name]()
                    samples[name].append(perf_counter_ns() - began)
            for name, result in variants.items():
                quartiles = statistics.quantiles(samples[name], n=4)
                rows.append(
                    {
                        "depth": depth,
                        "case": case,
                        "seed": SEEDS["game"],
                        **values,
                        "algorithm": name,
                        "first_target": result.first_target,
                        "utility": result.utility,
                        "visited_nodes": result.stats.visited_nodes,
                        "evaluated_leaves": result.stats.evaluated_leaves,
                        "cache_hits": result.stats.cache_hits,
                        "cache_entries": result.stats.cache_entries,
                        "cutoffs": result.stats.cutoffs,
                        "elapsed_ns_once": result.stats.elapsed_ns,
                        "time_median_ns": int(statistics.median(samples[name])),
                        "time_iqr_ns": int(quartiles[2] - quartiles[0]),
                    }
                )
        for name in ("minimax", "minimax_cache", "alphabeta_given", "alphabeta_greedy"):
            group = [
                row
                for row in rows
                if row["depth"] == depth and row["algorithm"] == name
            ]
            summary.append(
                {
                    "depth": depth,
                    "algorithm": name,
                    "cases": len(group),
                    "median_visited_nodes": statistics.median(
                        row["visited_nodes"] for row in group
                    ),
                    "median_cache_hits": statistics.median(
                        row["cache_hits"] for row in group
                    ),
                    "median_cache_entries": statistics.median(
                        row["cache_entries"] for row in group
                    ),
                    "median_time_median_ns": statistics.median(
                        row["time_median_ns"] for row in group
                    ),
                }
            )
    return rows, summary


def tree_experiment() -> tuple[list[dict[str, object]], dict[str, object]]:
    """在多棵自设树上验证访问顺序不是恒定收益的优化。"""
    rng = random.Random(SEEDS["tree"])
    rows = []

    def make(level: int) -> TreeNode:
        if level == 0:
            return TreeNode(rng.randint(-20, 20))
        return TreeNode(children=[make(level - 1) for _ in range(rng.randint(2, 4))])

    for case in range(100):
        root = make(4)
        exact, full_stats = tree_minimax(root, alpha_beta=False)
        variants = {
            order: tree_minimax(root, alpha_beta=True, order=order)
            for order in ("left_to_right", "reverse")
        }
        if any(value != exact for value, _ in variants.values()):
            raise AssertionError("访问顺序不能改变Minimax值")
        rows.append(
            {
                "case": case,
                "seed": SEEDS["tree"],
                "score": exact,
                "minimax_nodes": full_stats.visited_nodes,
                "left_nodes": variants["left_to_right"][1].visited_nodes,
                "reverse_nodes": variants["reverse"][1].visited_nodes,
                "left_cutoffs": variants["left_to_right"][1].cutoffs,
                "reverse_cutoffs": variants["reverse"][1].cutoffs,
            }
        )
    differences = [int(row["left_nodes"]) - int(row["reverse_nodes"]) for row in rows]
    summary = {
        "cases": len(rows),
        "reverse_saves_nodes": sum(value > 0 for value in differences),
        "equal_nodes": sum(value == 0 for value in differences),
        "left_to_right_saves_nodes": sum(value < 0 for value in differences),
        "median_left_nodes": statistics.median(row["left_nodes"] for row in rows),
        "median_reverse_nodes": statistics.median(row["reverse_nodes"] for row in rows),
        "min_node_difference": min(differences),
        "max_node_difference": max(differences),
    }
    return rows, summary


def make_figures(astar_summary, game_summary) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_chinese_font()
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, task, title in zip(
        axes,
        ("maze", "terrain_positive", "terrain_zero"),
        ("普通迷宫", "正代价加权地形", "含零代价加权地形"),
    ):
        for order, label in (
            ("deeper", "较大g优先"),
            ("fifo", "入队顺序"),
            ("shallower", "较小g优先"),
        ):
            group = [
                row
                for row in astar_summary
                if row["task"] == task and row["tie_break"] == order
            ]
            adequately_sampled = [row for row in group if row["reachable"] >= 10]
            axis.plot(
                [row["obstacle_density"] for row in adequately_sampled],
                [row["median_expanded_reachable"] for row in adequately_sampled],
                marker="o",
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("障碍比例")
        axis.set_ylabel("可达地图的扩展节点中位数")
        axis.legend()
    fig.suptitle("每档40次；无障碍迷宫为同图重复；可达样本不足10张时不绘制", fontsize=10)
    fig.savefig(FIGURES / "tie_break_regimes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    depths = (2, 4, 6)
    for name, label in (
        ("minimax", "Minimax"),
        ("minimax_cache", "Minimax+精确缓存"),
        ("alphabeta_given", "Alpha-beta原顺序"),
        ("alphabeta_greedy", "Alpha-beta收益排序"),
    ):
        group = [row for row in game_summary if row["algorithm"] == name]
        ax.plot(
            depths,
            [row["median_visited_nodes"] for row in group],
            marker="o",
            label=label,
        )
    ax.set_xticks(depths)
    ax.set_xlabel("搜索深度")
    ax.set_ylabel("访问节点中位数")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(FIGURES / "cache_and_ordering_depth.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    astar_rows, astar_summary = astar_experiment()
    game_rows, game_summary = game_experiment()
    tree_rows, tree_summary = tree_experiment()
    save_csv(RESULTS / "optimization_astar.csv", astar_rows)
    save_csv(RESULTS / "optimization_game.csv", game_rows)
    save_csv(RESULTS / "optimization_tree.csv", tree_rows)
    summary = {
        "seeds": SEEDS,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "search_algorithms.py",
                "game_search.py",
                "treasure_game.py",
                "extended_experiments.py",
                "plot_style.py",
                "requirements.txt",
            )
        },
        "astar_summary": astar_summary,
        "game_summary": game_summary,
        "tree_summary": tree_summary,
        "timing_note": "博弈方案每组预热并轮换测量11次，记录中位数与四分位距；短时计时仍受系统负载影响，以节点数为主要对照指标。",
        "cache_note": "仅缓存未剪枝Minimax的精确未来增量；不能直接缓存剪枝返回的界。",
        "game_design_note": "同一case的六个宝藏价值在H=2、4、6之间保持不变，各深度使用同一组40个案例。",
        "map_design_note": "distinct_maps按地图内容哈希去重；无障碍迷宫的40次记录来自同一张16×16地图。",
    }
    (RESULTS / "optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    deterministic = {
        "seeds": SEEDS,
        "astar": [
            {key: value for key, value in row.items() if key != "elapsed_ns_once"}
            for row in astar_rows
        ],
        "game": [
            {
                key: value
                for key, value in row.items()
                if key not in {"elapsed_ns_once", "time_median_ns", "time_iqr_ns"}
            }
            for row in game_rows
        ],
        "tree": tree_rows,
        "astar_summary": astar_summary,
        "game_summary": [
            {key: value for key, value in row.items() if key != "median_time_median_ns"}
            for row in game_summary
        ],
        "tree_summary": tree_summary,
    }
    (RESULTS / "optimization_deterministic.json").write_text(
        json.dumps(deterministic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(astar_summary, game_summary)
    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (RESULTS / "test_log.txt").write_text(
        test_run.stdout + test_run.stderr, encoding="utf-8"
    )
    if test_run.returncode:
        raise AssertionError("单元测试未通过，详见results/test_log.txt")
    print(
        "补充优化与拓展实验完成：",
        len(astar_rows),
        "条A*记录，",
        len(game_rows),
        "条博弈记录，",
        len(tree_rows),
        "棵自设树",
    )


if __name__ == "__main__":
    main()
