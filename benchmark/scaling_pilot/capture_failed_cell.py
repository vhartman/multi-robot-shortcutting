"""Run an unchanged cell worker, retaining its returned path only on failure.

Capture/serialization happens after the shortcutter returns, outside its timer.
Failed diagnostics are not accepted benchmark results.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
import traceback

import mr_shortcutting

worker, diagnostic, *arguments = sys.argv[1:]
original = mr_shortcutting.run
captured = {}


def capture(*args, **kwargs):
    path, result = original(*args, **kwargs)
    captured.update(path=path, result=result)
    return path, result


mr_shortcutting.run = capture
sys.argv = [worker, *arguments]
try:
    runpy.run_path(worker, run_name="__main__")
except BaseException as error:
    if not isinstance(error, SystemExit) or error.code not in (None, 0):
        if captured:
            destination = Path(diagnostic)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "diagnostic_only": True,
                "worker": worker,
                "arguments": arguments,
                "exception": traceback.format_exc(),
                "states": [state.to_dict() for state in captured["path"]],
                "result": captured["result"],
            }, indent=2) + "\n")
    raise
