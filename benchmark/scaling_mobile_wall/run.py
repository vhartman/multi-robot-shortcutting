"""Run the additive mobile-wall scaling protocol with the shared serial runner."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
os.environ["MR_SHORTCUT_PROTOCOL"] = str(HERE / "protocol.json")
os.environ["MR_SHORTCUT_OUTPUT_ROOT"] = str(HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(HERE.parent / "expanded"))

if __name__ == "__main__":
    runpy.run_path(str(HERE.parent / "expanded" / "run.py"), run_name="__main__")
