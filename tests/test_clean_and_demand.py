"""Tests for Milestone 2: Cleaning, demand reconstruction, and anomaly register."""

from pathlib import Path
import pandas as pd
import pytest

from celeris.pipeline import PipelineRunner


def test_milestone_2_fixture_pipeline_run(fixture_data_dir: Path, tmp_path: Path):
    """Run pipeline on fixture and assert all published artifacts exist."""
    runner = PipelineRunner(
        data_dir=fixture_data_dir,
        out_dir=tmp_path / "fixture_artifacts",
        is_fixture=True,
    )
    result = runner.run()

    assert result.success is True
    assert result.status == "SUCCESS"

    # Verify published parquet files
    history_file = tmp_path / "fixture_artifacts" / "history.parquet"
    quality_file = tmp_path / "fixture_artifacts" / "quality.parquet"
    anomalies_file = tmp_path / "fixture_artifacts" / "anomalies.parquet"

    assert history_file.exists()
    assert quality_file.exists()
    assert anomalies_file.exists()

    history_df = pd.read_parquet(history_file)
    quality_df = pd.read_parquet(quality_file)
    anomalies_df = pd.read_parquet(anomalies_file)

    assert not history_df.empty
    assert not quality_df.empty
    assert not anomalies_df.empty

    # Verify panel structure
    expected_cols = [
        "Date", "Year", "Month", "SKU_ID", "Region", "Product_Family",
        "Business_Unit", "is_observed", "Units_Sold", "reconstructed_demand",
        "demand_evidence", "Lifecycle_Status"
    ]
    for col in expected_cols:
        assert col in history_df.columns


def test_duplicate_reload_verification(fixture_data_dir: Path, tmp_path: Path):
    """Verify that repeated Record_ID is deduplicated by latest Load_Timestamp."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "dup_test", is_fixture=True)
    result = runner.run()
    assert result.success is True

    history_df = pd.read_parquet(tmp_path / "dup_test" / "history.parquet")
    quality_df = pd.read_parquet(tmp_path / "dup_test" / "quality.parquet")

    # In 2020-05, BSS-001 had duplicate load REC-BSS-202005 (earlier: 45, later: 50)
    bss_202005 = history_df[(history_df["SKU_ID"] == "BSS-001") & (history_df["Date"] == "2020-05")]
    assert len(bss_202005) == 1
    # Kept row must have latest load Units_Sold = 50
    assert bss_202005.iloc[0]["Units_Sold"] == 50

    # Quality log must record superseded earlier load (45 units)
    dup_log = quality_df[
        (quality_df["source_row_id"] == "REC-BSS-202005") &
        (quality_df["rule"] == "latest_load_dedup")
    ]
    assert len(dup_log) >= 1
    assert dup_log.iloc[0]["decision"] == "superseded_by_later_load"
    assert "Units=45" in dup_log.iloc[0]["original_value"]
    assert "Units=50" in dup_log.iloc[0]["cleaned_value"]


def test_real_zero_order_verification(fixture_data_dir: Path, tmp_path: Path):
    """Verify that Units_Sold = 0 is treated as a genuine observed zero order."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "zero_test", is_fixture=True)
    result = runner.run()
    assert result.success is True

    history_df = pd.read_parquet(tmp_path / "zero_test" / "history.parquet")
    anomalies_df = pd.read_parquet(tmp_path / "zero_test" / "anomalies.parquet")

    # In 2021-03, BSS-001 has a genuine zero order
    bss_202103 = history_df[(history_df["SKU_ID"] == "BSS-001") & (history_df["Date"] == "2021-03")]
    assert len(bss_202103) == 1
    row = bss_202103.iloc[0]
    assert row["is_observed"] is True or row["is_observed"] == 1
    assert row["Units_Sold"] == 0
    assert row["reconstructed_demand"] == 0
    assert row["demand_evidence"] == "observed_zero"

    # Anomaly register must record ZERO_ORDER retained
    zero_anom = anomalies_df[
        (anomalies_df["SKU_ID"] == "BSS-001") &
        (anomalies_df["Date"] == "2021-03") &
        (anomalies_df["anomaly_type"] == "ZERO_ORDER")
    ]
    assert len(zero_anom) == 1
    assert zero_anom.iloc[0]["action"] == "retained_true_zero"


def test_missing_month_verification(fixture_data_dir: Path, tmp_path: Path):
    """Verify that a missing month (gap) is distinguished from a zero order."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "missing_test", is_fixture=True)
    result = runner.run()
    assert result.success is True

    history_df = pd.read_parquet(tmp_path / "missing_test" / "history.parquet")
    anomalies_df = pd.read_parquet(tmp_path / "missing_test" / "anomalies.parquet")

    # In 2021-07, BSS-001 has NO row in raw ERP history
    bss_202107 = history_df[(history_df["SKU_ID"] == "BSS-001") & (history_df["Date"] == "2021-07")]
    assert len(bss_202107) == 1
    row = bss_202107.iloc[0]
    assert row["is_observed"] is False or row["is_observed"] == 0
    assert pd.isna(row["Units_Sold"])
    assert pd.isna(row["reconstructed_demand"])
    assert row["demand_evidence"] == "missing_record"

    # Anomaly register must record MISSING_MONTH
    missing_anom = anomalies_df[
        (anomalies_df["SKU_ID"] == "BSS-001") &
        (anomalies_df["Date"] == "2021-07") &
        (anomalies_df["anomaly_type"] == "MISSING_MONTH")
    ]
    assert len(missing_anom) == 1
    assert missing_anom.iloc[0]["action"] == "flagged_unobserved_gap"


def test_two_month_backlog_carry_verification(fixture_data_dir: Path, tmp_path: Path):
    """Verify stockout shortage reconstruction and subsequent backlog carry deduction."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "backlog_test", is_fixture=True)
    result = runner.run()
    assert result.success is True

    history_df = pd.read_parquet(tmp_path / "backlog_test" / "history.parquet")
    anomalies_df = pd.read_parquet(tmp_path / "backlog_test" / "anomalies.parquet")

    # Month 1: 2023-06 Stockout Month (70 shipped, 100 requested, 20 backordered)
    june_row = history_df[(history_df["SKU_ID"] == "BSS-001") & (history_df["Date"] == "2023-06")].iloc[0]
    assert june_row["Units_Sold"] == 70
    assert june_row["reconstructed_demand"] == 100  # Restored to true requested demand
    assert june_row["demand_evidence"] == "stockout_ledger_requested"

    june_anom = anomalies_df[
        (anomalies_df["SKU_ID"] == "BSS-001") &
        (anomalies_df["Date"] == "2023-06") &
        (anomalies_df["anomaly_type"] == "STOCKOUT_SHORTAGE")
    ]
    assert len(june_anom) == 1
    assert june_anom.iloc[0]["action"] == "reconstructed_to_requested"

    # Month 2: 2023-07 Rebound Month (110 shipped, includes 20 carry from June -> True demand = 90)
    july_row = history_df[(history_df["SKU_ID"] == "BSS-001") & (history_df["Date"] == "2023-07")].iloc[0]
    assert july_row["Units_Sold"] == 110
    assert july_row["reconstructed_demand"] == 90  # 110 - 20 = 90
    assert july_row["demand_evidence"] == "backlog_carry_rebound"

    july_anom = anomalies_df[
        (anomalies_df["SKU_ID"] == "BSS-001") &
        (anomalies_df["Date"] == "2023-07") &
        (anomalies_df["anomaly_type"] == "BACKLOG_CARRY_REBOUND")
    ]
    assert len(july_anom) == 1
    assert july_anom.iloc[0]["action"] == "deducted_prior_backorders"
