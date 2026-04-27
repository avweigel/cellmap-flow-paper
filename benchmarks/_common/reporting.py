from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


def write_result(out_path: str | Path, payload: dict[str, Any]) -> None:
    """Write a benchmark result JSON. Adds a timestamp and creates parent dirs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        **payload,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_results(results_dir: str | Path) -> list[dict]:
    """Recursively load every results/*.json under benchmarks/."""
    results = []
    for path in Path(results_dir).rglob("*.json"):
        if "results" not in path.parts:
            continue
        with path.open() as f:
            data = json.load(f)
        data["_source"] = str(path)
        results.append(data)
    return results
