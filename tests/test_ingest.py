"""Tests for input ingestion, role resolution, and diagnostics."""

from pathlib import Path
from celeris.schemas import DatasetRole
from celeris.ingest import diagnose_input_directory


def test_diagnose_student_data(student_data_dir: Path):
    """Diagnose updated student_data directory and verify all 7 roles are resolved."""
    report = diagnose_input_directory(student_data_dir)

    # 1. Scanned 7 files
    assert len(report.files_scanned) == 7

    # 2. Check exact mapping of all 7 roles
    assert "train.csv" in report.role_mapping[DatasetRole.HISTORY.value]
    assert "test.csv" in report.role_mapping[DatasetRole.FORECAST_ROWS.value]
    assert "sku_master.csv" in report.role_mapping[DatasetRole.SKU_MASTER.value]
    assert "sample_submission.csv" in report.role_mapping[DatasetRole.SAMPLE_SUBMISSION.value]
    assert "macro_indicators.csv" in report.role_mapping[DatasetRole.MACRO_INDICATORS.value]
    assert "inventory_stockouts.csv" in report.role_mapping[DatasetRole.STOCKOUTS.value]
    assert "events_calendar.csv" in report.role_mapping[DatasetRole.EVENTS_CALENDAR.value]

    # 3. All roles present and ready
    assert len(report.missing_roles) == 0
    assert report.is_ready is True
    assert report.blocker_message is None

    # 4. Check test rows properties
    assert report.forecast_keys_profile is not None
    assert report.forecast_keys_profile.total_rows == 58219
    assert report.forecast_keys_profile.min_date == "2026-11"
    assert report.forecast_keys_profile.max_date == "2028-05"
    assert report.forecast_keys_profile.num_months == 19
    assert report.forecast_keys_profile.has_target_column is False


def test_diagnose_fixture_data(fixture_data_dir: Path):
    """Diagnose fixture directory and verify that all 7 roles resolve."""
    report = diagnose_input_directory(fixture_data_dir)

    assert len(report.files_scanned) == 7
    assert len(report.missing_roles) == 0
    assert report.is_ready is True
    assert report.blocker_message is None

    # Verify each role resolved
    for role in DatasetRole:
        assert len(report.role_mapping[role.value]) >= 1
