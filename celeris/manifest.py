"""Run manifest management and metadata recording for Celeris pipeline runs."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_environment_info() -> Dict[str, str]:
    """Capture environment and package versions."""
    packages = ["pandas", "pyarrow", "streamlit", "scikit-learn", "lightgbm"]
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


@dataclass
class RunManifest:
    """Run manifest capturing inputs, configuration, environment, and diagnostics."""
    run_id: str
    run_timestamp: str
    run_mode: str  # "fixture" or "real"
    is_fixture: bool
    data_dir: str
    out_dir: str
    status: str    # "SUCCESS", "BLOCKED_MISSING_HISTORY", "FAILED"
    blocker: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=get_environment_info)
    resolved_roles: Dict[str, List[str]] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    diagnostics_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> RunManifest:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
