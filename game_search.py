"""实验二题三：纯函数式 Minimax 与 Alpha-beta，对共享子树不作修改。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import inf
from time import perf_counter_ns


class TreeNode:
    def __init__(
        self, value: int | None = None, children: Sequence[TreeNode] | None = None
    ):
        self.value = value
        self.children = list(children) if children is not None else []


@dataclass
class TreeStats:
    visited_nodes: int = 0
    evaluated_leaves: int = 0
    cutoffs: int = 0
    skipped_children: int = 0
    elapsed_ns: int = 0
    pruned_branches: list[list[int]] = field(default_factory=list)
    cutoff_trace: list[dict[str, object]] = field(default_factory=list)


def tree_minimax(
    root: TreeNode, *, alpha_beta: bool = True, order: str = "left_to_right"
) -> tuple[int, TreeStats]:
    """根为 MAX，逐层交替。正式题意采用默认的从左至右访问。"""
    if order not in {"left_to_right", "reverse"}:
        raise ValueError("order 必须是 left_to_right 或 reverse")
    stats = TreeStats()
    began = perf_counter_ns()

    def visit(
        node: TreeNode,
        maximizing: bool,
        alpha: float,
        beta: float,
        ancestors: frozenset[int],
        path: tuple[int, ...],
    ) -> int:
        if id(node) in ancestors:
            raise ValueError("博弈树不能含循环引用")
        stats.visited_nodes += 1
        if not node.children:
            if node.value is None:
                raise ValueError("叶子节点缺少 value")
            stats.evaluated_leaves += 1
            return int(node.value)

        children = list(enumerate(node.children))
        if order == "reverse":
            children.reverse()
        best = -inf if maximizing else inf
        next_ancestors = ancestors | {id(node)}
        for index, (original_index, child) in enumerate(children):
            value = visit(
                child,
                not maximizing,
                alpha,
                beta,
                next_ancestors,
                path + (original_index,),
            )
            if maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            if alpha_beta and alpha >= beta and index + 1 < len(children):
                stats.cutoffs += 1
                stats.skipped_children += len(children) - index - 1
                stats.cutoff_trace.append(
                    {
                        "node_path": list(path),
                        "role": "MAX" if maximizing else "MIN",
                        "last_child_path": [*path, original_index],
                        "last_child_value": value,
                        "best_so_far": int(best),
                        "alpha": int(alpha),
                        "beta": int(beta),
                        "skipped_paths": [
                            [*path, skipped_index]
                            for skipped_index, _ in children[index + 1 :]
                        ],
                    }
                )
                stats.pruned_branches.extend(
                    [*path, skipped_index] for skipped_index, _ in children[index + 1 :]
                )
                break
        return int(best)

    result = visit(root, True, -inf, inf, frozenset(), ())
    stats.elapsed_ns = perf_counter_ns() - began
    return result, stats


def alpha_beta_best_score(root: TreeNode) -> int:
    """题三要求的直接输出接口。"""
    return tree_minimax(root)[0]


def sample_tree() -> TreeNode:
    """严格按题目第12页构造，max2 在左右 MIN 子树中共享。"""
    leaf1, leaf2 = TreeNode(3), TreeNode(5)
    leaf3, leaf4 = TreeNode(6), TreeNode(9)
    leaf5, leaf6 = TreeNode(1), TreeNode(2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])
