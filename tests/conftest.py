"""Pytest configuration and shared fixtures for Celeris test suite."""

from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    """Root directory of the project."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def student_data_dir(project_root: Path) -> Path:
    """Path to student_data directory."""
    return project_root / "student_data"


@pytest.fixture
def fixture_data_dir(project_root: Path) -> Path:
    """Path to generated fixture data directory."""
    return project_root / "fixtures" / "data"
