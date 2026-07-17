from __future__ import annotations

import os
import sysconfig
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def source_root() -> Path | None:
    """Return the repository root when ProVisE is running from a checkout."""
    candidate = PACKAGE_ROOT.parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
        return candidate
    return None


def runtime_root() -> Path:
    """Return the writable root for local configuration, outputs, and model paths."""
    configured = str(os.getenv("PROVISE_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = source_root()
    return checkout if checkout is not None else Path.cwd().resolve()


def resource_root() -> Path:
    """Locate read-only configuration shipped by a checkout or installed wheel."""
    configured = str(os.getenv("PROVISE_RESOURCE_ROOT") or "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    checkout = source_root()
    if checkout is not None:
        candidates.append(checkout)
    candidates.append(Path(sysconfig.get_path("data")) / "share" / "provise")

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "configs" / "protocol_specs").is_dir():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"ProVisE runtime resources were not found; searched: {searched}")


def protocol_spec_dir() -> Path:
    return resource_root() / "configs" / "protocol_specs"


def benchmark_suite_path(name: str = "validated_spatial") -> Path:
    return resource_root() / "configs" / "benchmark_suites" / f"{name}.yaml"
