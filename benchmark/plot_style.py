"""Canonical visual identities for benchmark plots.

Color identifies a shortcutting method across every figure. Algorithm curves
are solid; line style is reserved for controls or an explicitly plotted
experimental variable such as the collision-check budget.
"""
from __future__ import annotations


METHOD_COLORS = {
    "planner_only": "#000000",
    "full_configuration": "#4c4c4c",
    "single_robot_random": "#d95f02",
    "robot_subset_random": "#7570b3",
    "single_robot_mode": "#1b9e77",
    "robot_subset_mode": "#1f78b4",
    "wafr_slide_halton": "#17a2a4",
    "wafr_single_robot": "#a6761d",
    "huang_round_robin": "#984ea3",
    "huang_round_robin_no_prioritized": "#984ea3",
    "huang_thompson": "#e7298a",
    "tree_gain": "#e66101",
    "tree_prefix": "#e66101",
    "upstream_tree": "#e66101",
    "tree_group_gain": "#d73027",
    "gain_only": "#7570b3",
    "full_tree_gain_round_robin": "#008695",
    "scalable_tree_group": "#332288",
    "tree_then_compatible_group": "#cc6677",
    "random_compatible_group": "#1f78b4",
}

ALGORITHM_LINESTYLE = "-"
CONTROL_LINESTYLE = "--"

# Fewer checks are dotted, the default is solid, and more checks are dashed.
CHECK_BUDGET_LINESTYLES = {
    100: ":",
    250: "-",
    500: "--",
}


def method_color(method_id: str) -> str:
    return METHOD_COLORS[method_id]


def method_linestyle(method_id: str) -> str:
    if method_id == "planner_only":
        return CONTROL_LINESTYLE
    return ALGORITHM_LINESTYLE
