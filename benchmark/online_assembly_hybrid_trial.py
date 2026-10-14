"""Load the scalable tree/group policy, then invoke the online runner."""
from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCALING = HERE / "scaling_2d"
if str(SCALING) not in sys.path:
    sys.path.insert(0, str(SCALING))

import scalable_tree_group  # noqa: E402,F401
import online  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(online.main())
