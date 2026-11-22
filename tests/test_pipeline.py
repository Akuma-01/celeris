"""Tests for pipeline coordination, CLI runs, and manifest generation."""

import json
from pathlib import Path
from celeris.pipeline import PipelineRunner
from celeris.manifest import RunManifest


def test_pipeline_run_on_student_data_halts_with_missing_history(
    student_data_dir: Path, tmp_path: Path
):
    """Running pipeline on student_data halts before fitting/publishing and writes blocked manifest."""
    runner = PipelineRunner(
        data_dir=student_data_dir,
        out_dir=tmp_path / "run_student",
        is_fixture=False,
    )
    result = runner.run()

    assert result.success is False
    assert result.status == "BLOCKED_MISSING_HISTORY"
    assert "Historical ERP sales extract is MISSING" in result.message

    # Verify manifest written
    manifest_file = tmp_path / "run_student" / "manifest.json"
    assert manifest_file.exists()
    manifest = RunManifest.load(manifest_file)
    assert manifest.status == "BLOCKED_MISSING_HISTORY"
    assert manifest.is_fixture is False
    assert "history" in manifest.diagnostics_summary["missing_roles"]


def test_pipeline_run_on_fixture_succeeds(
    fixture_data_dir: Path, tmp_path: Path
):
    """Running pipeline on fixture data resolves all roles, writes manifest, and succeeds."""
    runner = PipelineRunner(
        data_dir=fixture_data_dir,
        out_dir=tmp_path / "run_fixture",
        is_fixture=True,
    )
    result = runner.run()

    assert result.success is True
    assert result.status == "SUCCESS"
    assert result.run_mode == "fixture"

    # Verify manifest written
    manifest_file = tmp_path / "run_fixture" / "manifest.json"
    assert manifest_file.exists()
    manifest = RunManifest.load(manifest_file)
    assert manifest.status == "SUCCESS"
    assert manifest.is_fixture is True
    assert len(manifest.diagnostics_summary["missing_roles"]) == 0

    # Verify pointer written
    pointer_file = tmp_path / "run_fixture" / "current_run.json"
    assert pointer_file.exists()
    with open(pointer_file, "r") as f:
        pointer = json.load(f)
    assert pointer["current_run_id"] == runner.run_id
