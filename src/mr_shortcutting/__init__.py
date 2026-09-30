"""Reusable multi-robot shortcutting policies and benchmark engine."""

from .api import (
    Proposal,
    ProposalSource,
    Shortcutter,
    available_shortcutters,
    get_shortcutter,
    load_shortcutter_plugins,
    register_shortcutter,
)
from .shortcutters import POLICIES, run, run_checks

__all__ = [
    "POLICIES",
    "Proposal",
    "ProposalSource",
    "Shortcutter",
    "available_shortcutters",
    "get_shortcutter",
    "load_shortcutter_plugins",
    "register_shortcutter",
    "run",
    "run_checks",
]
