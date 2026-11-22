"""Tests for input ingestion, role resolution, and diagnostics."""

from pathlib import Path
from celeris.schemas import DatasetRole
from celeris.ingest import diagnose_input_directory


def test_diagnose_student_data(student_data_dir: Path):
    """Diagnose student_data directory and verify exact mislabeling resolution."""
    report = diagnose_input_directory(student_data_dir)

    # 1. Scanned 7 files
    assert len(report.files_scanned) == 7

    # 2. Check exact mapping
    assert "train.csv" in report.role_mapping[DatasetRole.FORECAST_ROWS.value]
    assert "test.csv" in report.role_mapping[DatasetRole.SKU_MASTER.value]
    assert "sku_master.csv" in report.role_mapping[DatasetRole.SAMPLE_SUBMISSION.value]
    assert "sample_submission.csv" in report.role_mapping[DatasetRole.MACRO_INDICATORS.value]
    assert "macro_indicators.csv" in report.role_mapping[DatasetRole.STOCKOUTS.value]

    # 3. Two files mapped to events_calendar
    calendar_files = report.role_mapping[DatasetRole.EVENTS_CALENDAR.value]
    assert set(calendar_files) == {"events_calendar.csv", "inventory_stockouts.csv"}
    assert report.calendar_comparison is not None
    assert report.calendar_comparison.are_byte_identical is True

    # 4. History role must be missing
    assert DatasetRole.HISTORY.value in report.missing_roles
    assert report.is_ready is False
    assert report.blocker_message is not None
    assert "Record_ID" in report.blocker_message

    # 5. Check test rows properties
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
