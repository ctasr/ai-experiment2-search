"""进一步检验可复用启发函数和双步排序的收益、代价与反例。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from functools import partial
from pathlib import Path
from time import perf_counter_ns

from extended_experiments import SIX_TREASURE_MAP
from plot_style import configure_chinese_font
from search_algorithms import (
    reverse_cost_potential,
    weighted_search,
    weighted_search_with_potential,
)
from treasure_game import (
    SAMPLE_MAP,
    SAMPLE_VALUES,
    compute_distances,
    solve_treasure_game,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = {"multiquery": 20261018, "ordering": 20261019}


def iqr_ns(samples: list[int]) -> int:
    quartiles = statistics.quantiles(samples, n=4)
    return int(quartiles[2] - quartiles[0])


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def make_map(rng: random.Random, zero_costs: bool, size: int = 18):
    choices = (0, 1, 3, 8) if zero_costs else (1, 3, 8)
    grid = [
        ["X" if rng.random() < 0.12 else rng.choice(choices) for _ in range(size)]
        for _ in range(size)
    ]
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def ordinary_batch(grid, starts, goal) -> list[int | None]:
    return [weighted_search(grid, start, goal).cost for start in starts]


def exact_batch(grid, starts, goal, potential) -> list[int | None]:
    return [
        weighted_search_with_potential(grid, start, goal, potential).cost
        for start in starts
    ]


def direct_lookup_batch(starts, potential) -> list[int | None]:
    """相同地图、终点且只问总代价时，反向最短路表已包含查询答案。"""
    return [potential.costs.get(start) for start in starts]


def measure_multiquery(grid, starts, goal, repeats: int = 11) -> dict[str, int]:
    """轮换测量三方案；两种预处理方案各自建表并计入同次总耗时。"""
    ordinary_batch(grid, starts, goal)
    warm_potential = reverse_cost_potential(grid, goal)
    exact_batch(grid, starts, goal, warm_potential)
    direct_lookup_batch(starts, warm_potential)
    ordinary_samples = []
    prep_samples = []
    exact_samples = []
    total_samples = []
    lookup_prep_samples = []
    lookup_samples = []
    lookup_total_samples = []

    def ordinary_trial() -> None:
        began = perf_counter_ns()
        ordinary_batch(grid, starts, goal)
        ordinary_samples.append(perf_counter_ns() - began)

    def potential_trial() -> None:
        began = perf_counter_ns()
        potential = reverse_cost_potential(grid, goal)
        prepared = perf_counter_ns()
        exact_batch(grid, starts, goal, potential)
        finished = perf_counter_ns()
        prep_samples.append(prepared - began)
        exact_samples.append(finished - prepared)
        total_samples.append(finished - began)

    def lookup_trial() -> None:
        began = perf_counter_ns()
        potential = reverse_cost_potential(grid, goal)
        prepared = perf_counter_ns()
        direct_lookup_batch(starts, potential)
        finished = perf_counter_ns()
        lookup_prep_samples.append(prepared - began)
        lookup_samples.append(finished - prepared)
        lookup_total_samples.append(finished - began)

    trials = (ordinary_trial, potential_trial, lookup_trial)
    for repetition in range(repeats):
        offset = repetition % len(trials)
        for trial in trials[offset:] + trials[:offset]:
            trial()
    return {
        "ordinary_batch_time_median_ns": int(statistics.median(ordinary_samples)),
        "ordinary_batch_time_iqr_ns": iqr_ns(ordinary_samples),
        "preprocess_time_median_ns": int(statistics.median(prep_samples)),
        "exact_batch_time_median_ns": int(statistics.median(exact_samples)),
        "exact_total_time_median_ns": int(statistics.median(total_samples)),
        "exact_total_time_iqr_ns": iqr_ns(total_samples),
        "lookup_preprocess_time_median_ns": int(statistics.median(lookup_prep_samples)),
        "lookup_batch_time_median_ns": int(statistics.median(lookup_samples)),
        "lookup_total_time_median_ns": int(statistics.median(lookup_total_samples)),
        "lookup_total_time_iqr_ns": iqr_ns(lookup_total_samples),
    }


def measure_orderings(
    calls: dict[str, object], repeats: int = 11
) -> dict[str, dict[str, int]]:
    """轮换运行先后，以免固定次序把热缓存或系统漂移只给一种策略。"""
    names = tuple(calls)
    for call in calls.values():
        call()
    samples: dict[str, list[int]] = {name: [] for name in names}
    for repetition in range(repeats):
        rotated = names[repetition % len(names) :] + names[: repetition % len(names)]
        for name in rotated:
            began = perf_counter_ns()
            calls[name]()
            samples[name].append(perf_counter_ns() - began)
    return {
        name: {
            "time_median_ns": int(statistics.median(values)),
            "time_iqr_ns": iqr_ns(values),
        }
        for name, values in samples.items()
    }


def multiquery_experiment() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEEDS["multiquery"])
    rows = []
    for regime in ("positive", "with_zero"):
        for case in range(20):
            for attempt in range(100):
                grid = make_map(rng, regime == "with_zero")
                goal = (len(grid) - 1, len(grid[0]) - 1)
                potential = reverse_cost_potential(grid, goal)
                available = sorted(point for point in potential.costs if point != goal)
                if len(available) >= 15:
                    break
            else:
                raise AssertionError("固定种子未找到足够的可达起点")
            sources = rng.sample(available, 15)

            for queries in (1, 2, 3, 5, 15):
                starts = sources[:queries]
                ordinary = [weighted_search(grid, p, goal) for p in starts]
                exact = [
                    weighted_search_with_potential(grid, p, goal, potential)
                    for p in starts
                ]
                oracle = [potential.costs[p] for p in starts]
                lookup = direct_lookup_batch(starts, potential)
                if [r.cost for r in ordinary] != oracle or [
                    r.cost for r in exact
                ] != oracle or lookup != oracle:
                    raise AssertionError("普通A*、精确估计A*与直接查表的代价不一致")

                timings = measure_multiquery(grid, starts, goal)
                rows.append(
                    {
                        "regime": regime,
                        "case": case,
                        "seed": SEEDS["multiquery"],
                        "map_attempt": attempt,
                        "queries": queries,
                        "reachable_cells": len(potential.costs),
                        "preprocess_expanded": potential.expanded,
                        "ordinary_search_expanded": sum(r.expanded for r in ordinary),
                        "exact_search_expanded": sum(r.expanded for r in exact),
                        "lookup_search_expanded": 0,
                        **timings,
                        "optimal_costs": json.dumps(oracle),
                        "lookup_costs": json.dumps(lookup),
                    }
                )
    summary = []
    for regime in ("positive", "with_zero"):
        for queries in (1, 2, 3, 5, 15):
            group = [
                row
                for row in rows
                if row["regime"] == regime and row["queries"] == queries
            ]
            paired_savings = [
                row["ordinary_batch_time_median_ns"] - row["exact_total_time_median_ns"]
                for row in group
            ]
            ordinary_times = [row["ordinary_batch_time_median_ns"] for row in group]
            exact_times = [row["exact_total_time_median_ns"] for row in group]
            lookup_times = [row["lookup_total_time_median_ns"] for row in group]
            ordinary_quartiles = statistics.quantiles(ordinary_times, n=4)
            exact_quartiles = statistics.quantiles(exact_times, n=4)
            lookup_quartiles = statistics.quantiles(lookup_times, n=4)
            lookup_paired_savings = [
                row["ordinary_batch_time_median_ns"] - row["lookup_total_time_median_ns"]
                for row in group
            ]
            summary.append(
                {
                    "regime": regime,
                    "queries": queries,
                    "maps": len(group),
                    "median_ordinary_expanded": statistics.median(
                        row["ordinary_search_expanded"] for row in group
                    ),
                    "median_exact_search_expanded": statistics.median(
                        row["exact_search_expanded"] for row in group
                    ),
                    "median_preprocess_expanded": statistics.median(
                        row["preprocess_expanded"] for row in group
                    ),
                    "median_ordinary_time_ns": statistics.median(ordinary_times),
                    "median_exact_total_time_ns": statistics.median(exact_times),
                    "median_lookup_total_time_ns": statistics.median(lookup_times),
                    "ordinary_time_q1_ns": ordinary_quartiles[0],
                    "ordinary_time_q3_ns": ordinary_quartiles[2],
                    "exact_total_time_q1_ns": exact_quartiles[0],
                    "exact_total_time_q3_ns": exact_quartiles[2],
                    "lookup_total_time_q1_ns": lookup_quartiles[0],
                    "lookup_total_time_q3_ns": lookup_quartiles[2],
                    "exact_faster_total_cases": sum(
                        row["exact_total_time_median_ns"]
                        < row["ordinary_batch_time_median_ns"]
                        for row in group
                    ),
                    "median_paired_time_saved_ns": statistics.median(paired_savings),
                    "lookup_faster_than_ordinary_cases": sum(
                        row["lookup_total_time_median_ns"]
                        < row["ordinary_batch_time_median_ns"]
                        for row in group
                    ),
                    "lookup_faster_than_exact_cases": sum(
                        row["lookup_total_time_median_ns"]
                        < row["exact_total_time_median_ns"]
                        for row in group
                    ),
                    "median_lookup_paired_time_saved_ns": statistics.median(
                        lookup_paired_savings
                    ),
                    "paired_time_saved_q1_ns": statistics.quantiles(
                        paired_savings, n=4
                    )[0],
                    "paired_time_saved_q3_ns": statistics.quantiles(
                        paired_savings, n=4
                    )[2],
                }
            )
    return rows, summary


def ordering_experiment() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEEDS["ordering"])
    rows = []
    scenes = (
        (
            "four",
            compute_distances(SAMPLE_MAP, SAMPLE_VALUES),
            tuple(SAMPLE_VALUES),
            4,
            120,
        ),
        (
            "six",
            compute_distances(SIX_TREASURE_MAP, {f"T{i}": i + 4 for i in range(1, 7)}),
            tuple(f"T{i}" for i in range(1, 7)),
            6,
            30,
        ),
    )
    for scene, distances, names, depth, count in scenes:
        for case in range(count):
            values = {name: rng.randint(1, 25) for name in names}
            exact = solve_treasure_game(
                distances, depth, alpha_beta=False, values=values
            )
            calls = {
                order: partial(
                    solve_treasure_game,
                    distances,
                    depth,
                    order=order,
                    values=values,
                )
                for order in ("given", "greedy", "lookahead")
            }
            timings = measure_orderings(calls)
            for order in ("given", "greedy", "lookahead"):
                result = calls[order]()
                if result.utility != exact.utility:
                    raise AssertionError("双步排序不能改变精确收益差")
                rows.append(
                    {
                        "scene": scene,
                        "case": case,
                        "seed": SEEDS["ordering"],
                        "depth": depth,
                        **values,
                        "order": order,
                        "first_target": result.first_target,
                        "utility": result.utility,
                        "visited_nodes": result.stats.visited_nodes,
                        "cutoffs": result.stats.cutoffs,
                        "ordering_evaluations": result.stats.ordering_evaluations,
                        **timings[order],
                    }
                )
    summary = []
    for scene in ("four", "six"):
        group = [row for row in rows if row["scene"] == scene]
        by_case = {
            case: {row["order"]: row for row in group if row["case"] == case}
            for case in sorted({row["case"] for row in group})
        }
        differences = [
            int(case["greedy"]["visited_nodes"])
            - int(case["lookahead"]["visited_nodes"])
            for case in by_case.values()
        ]
        time_differences = [
            int(case["greedy"]["time_median_ns"])
            - int(case["lookahead"]["time_median_ns"])
            for case in by_case.values()
        ]
        summary.append(
            {
                "scene": scene,
                "cases": len(by_case),
                "lookahead_fewer_nodes_than_greedy": sum(v > 0 for v in differences),
                "same_nodes": sum(v == 0 for v in differences),
                "lookahead_more_nodes_than_greedy": sum(v < 0 for v in differences),
                "median_nodes_saved_over_greedy": statistics.median(differences),
                "lookahead_faster_than_greedy": sum(v > 0 for v in time_differences),
                "median_time_saved_ns": statistics.median(time_differences),
                "median_lookahead_evaluations": statistics.median(
                    case["lookahead"]["ordering_evaluations"]
                    for case in by_case.values()
                ),
            }
        )
    return rows, summary


def make_figures(multiquery_summary, ordering_summary) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_chinese_font()
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, regime, title in zip(
        axes, ("positive", "with_zero"), ("正代价地图", "含零代价地图")
    ):
        group = [row for row in multiquery_summary if row["regime"] == regime]
        x = [row["queries"] for row in group]
        ax.plot(
            x,
            [row["median_ordinary_time_ns"] / 1e6 for row in group],
            marker="o",
            label="普通A*累计",
            color="#1f77b4",
        )
        ax.fill_between(
            x,
            [row["ordinary_time_q1_ns"] / 1e6 for row in group],
            [row["ordinary_time_q3_ns"] / 1e6 for row in group],
            alpha=0.13,
            color="#1f77b4",
        )
        ax.plot(
            x,
            [row["median_exact_total_time_ns"] / 1e6 for row in group],
            marker="o",
            label="预处理+精确估计A*",
            color="#ff7f0e",
        )
        ax.fill_between(
            x,
            [row["exact_total_time_q1_ns"] / 1e6 for row in group],
            [row["exact_total_time_q3_ns"] / 1e6 for row in group],
            alpha=0.15,
            color="#ff7f0e",
        )
        ax.plot(
            x,
            [row["median_lookup_total_time_ns"] / 1e6 for row in group],
            marker="s",
            label="预处理+直接查代价表",
            color="#2ca02c",
        )
        ax.fill_between(
            x,
            [row["lookup_total_time_q1_ns"] / 1e6 for row in group],
            [row["lookup_total_time_q3_ns"] / 1e6 for row in group],
            alpha=0.13,
            color="#2ca02c",
        )
        ax.set_xticks(x)
        ax.set_xlabel("相同终点的查询次数")
        ax.set_ylabel("20张地图的用时中位数（毫秒）")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("线：中位数；阴影：20张地图的四分位区间", fontsize=10)
    fig.savefig(FIGURES / "multiquery_preprocessing_tradeoff.png", dpi=180)
    plt.close(fig)

    fig, (ax, time_ax) = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    labels = ["四宝藏", "六宝藏"]
    x = list(range(len(labels)))
    bottom = [0.0, 0.0]
    for field, label, color in (
        ("lookahead_fewer_nodes_than_greedy", "访问更少", "#54A875"),
        ("same_nodes", "访问相同", "#6C8EBB"),
        ("lookahead_more_nodes_than_greedy", "访问更多", "#D77A64"),
    ):
        heights = [100 * row[field] / row["cases"] for row in ordering_summary]
        ax.bar(x, heights, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, heights)]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("相对即时收益排序的案例比例（%）")
    ax.legend()

    penalties = [-row["median_time_saved_ns"] / 1e6 for row in ordering_summary]
    bars = time_ax.bar(x, penalties, color="#D77A64")
    time_ax.bar_label(bars, fmt="%.2f")
    time_ax.set_xticks(x, labels)
    time_ax.set_ylabel("双步排序增加的用时中位数（毫秒）")
    fig.suptitle("双步排序的节点数与计时权衡", fontsize=10)
    fig.savefig(FIGURES / "lookahead_ordering_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    multiquery_rows, multiquery_summary = multiquery_experiment()
    ordering_rows, ordering_summary = ordering_experiment()
    save_csv(RESULTS / "advanced_multiquery.csv", multiquery_rows)
    save_csv(RESULTS / "advanced_ordering.csv", ordering_rows)
    summary = {
        "seeds": SEEDS,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "search_algorithms.py",
                "treasure_game.py",
                "extended_experiments.py",
                "advanced_experiments.py",
                "plot_style.py",
                "requirements.txt",
            )
        },
        "multiquery_summary": multiquery_summary,
        "ordering_summary": ordering_summary,
        "timing_note": "预热后轮换运行11次，取中位数并记录四分位距。多查询的三方案使用相同地图和起点，两种预处理方案各自重新建表，建表与查询计入同次总用时。直接查表适用于地图和终点不变且只需要总代价的查询。",
    }
    (RESULTS / "advanced_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    deterministic = {
        "seeds": SEEDS,
        "multiquery": [
            {
                key: value
                for key, value in row.items()
                if not key.endswith(("_time_median_ns", "_time_iqr_ns"))
            }
            for row in multiquery_rows
        ],
        "ordering": [
            {
                key: value
                for key, value in row.items()
                if key not in {"time_median_ns", "time_iqr_ns"}
            }
            for row in ordering_rows
        ],
    }
    (RESULTS / "advanced_deterministic.json").write_text(
        json.dumps(deterministic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(multiquery_summary, ordering_summary)
    print(
        "进一步拓展完成：",
        len(multiquery_rows),
        "组多查询，",
        len(ordering_rows),
        "组排序",
    )


if __name__ == "__main__":
    main()
