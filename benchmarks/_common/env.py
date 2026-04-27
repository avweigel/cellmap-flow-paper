from __future__ import annotations

import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version


def _git_sha(repo_path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "name": None, "count": 0}
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "count": 0}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "count": torch.cuda.device_count(),
    }


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def capture_env(repo_path: str = ".") -> dict:
    """Snapshot the environment the benchmark ran in. Goes alongside results so
    they can be reproduced or filtered by hardware."""
    pkgs = ["cellmap-flow", "torch", "numpy", "zarr", "tensorstore", "daisy", "neuroglancer"]
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(repo_path),
        "gpu": _gpu_info(),
        "packages": {p: _pkg_version(p) for p in pkgs},
    }
