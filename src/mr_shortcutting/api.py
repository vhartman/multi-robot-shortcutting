"""Public extension API for shortcut proposal policies."""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Proposal:
    """A mode-safe partial shortcut proposed to the shared execution engine."""

    first: int
    last: int
    robots: tuple[int, ...]
    source: str = "custom"
    arm: tuple[int, ...] | None = None
    tree_depth: int = 0
    shortcut_type: str = "path"
    shortcut_end: int | None = None
    motion_steps: int | None = None
    force_cost_reject: bool = False


@runtime_checkable
class ProposalSource(Protocol):
    """Callable proposal stream; feedback hooks are optional."""

    def __call__(self) -> Proposal | None:
        ...


class Shortcutter(ABC):
    """Factory for one stateful proposal source per shortcutting invocation.

    Implementations only choose candidate endpoints and robot subsets. The
    benchmark engine remains responsible for cost gating, interpolation,
    collision checking, accepted-path updates, timing, and result traces.
    """

    name: str

    @abstractmethod
    def make_proposer(
        self,
        env: Any,
        path: list[Any],
        rng: np.random.Generator,
        interpolation_resolution: float,
    ) -> ProposalSource:
        """Return a fresh proposal source bound to the given path."""


_REGISTRY: dict[str, Shortcutter] = {}


def register_shortcutter(shortcutter: Shortcutter, *, replace: bool = False) -> None:
    """Register a shortcutter under its stable, non-empty name."""

    name = getattr(shortcutter, "name", "")
    if not isinstance(name, str) or not name:
        raise ValueError("a shortcutter needs a non-empty string name")
    if name in _REGISTRY and not replace:
        raise ValueError(f"shortcutter {name!r} is already registered")
    _REGISTRY[name] = shortcutter


def get_shortcutter(name: str) -> Shortcutter:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        available = ", ".join(available_shortcutters())
        raise ValueError(
            f"unknown shortcutter {name!r}; available: {available}"
        ) from error


def available_shortcutters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def load_shortcutter_plugins(modules: list[str] | tuple[str, ...]) -> None:
    """Import modules whose import side effect calls register_shortcutter."""

    for module in modules:
        importlib.import_module(module)
