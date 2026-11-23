"""Pipeline coordination and execution for Celeris."""

from __future__ import annotations

import datetime
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from celeris.clean import clean_history_transactions, QualityRecord
from celeris.demand import reconstruct_demand_and_anomalies, AnomalyRecord
from celeris.backtest import run_full_backtest, verify_leakage_safety, DEFAULT_FOLDS
from celeris.publish import (
    generate_final_forecast,
    derive_submission,
    validate_submission,
    SubmissionValidationReport,
)
from celeris.ingest import (
    DiagnosticReport,
    diagnose_input_directory,
    format_diagnostic_report,
)
from celeris.manifest import RunManifest
from celeris.schemas import DatasetRole


@dataclass
class PipelineRunResult:
    """Outcome of a pipeline execution."""
    success: bool
    status: str
    run_id: str
    data_dir: str
    out_dir: str
    run_mode: str
    manifest_path: Optional[str]
    message: str
    report: DiagnosticReport
    published_artifacts: Dict[str, str]
    backtest_summary: Optional[Dict[str, Any]] = None
    submission_validation: Optional[SubmissionValidationReport] = None


class PipelineRunner:
    """Coordinates ingestion, role resolution, diagnostics, cleaning, backtesting, forecast, and publication."""

    def __init__(
        self,
        data_dir: Path | str,
        out_dir: Path | str,
        is_fixture: Optional[bool] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.run_id = run_id or f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        if is_fixture is None:
            self.is_fixture = "fixture" in str(self.data_dir).lower()
        else:
            self.is_fixture = is_fixture

    def diagnose(self, save_report: bool = True) -> DiagnosticReport:
        """Run input diagnostics without full execution."""
        report = diagnose_input_directory(self.data_dir)
        if save_report:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.out_dir / "diagnostics.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            text_path = self.out_dir / "diagnostics.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(format_diagnostic_report(report))
        return report

    def run(self) -> PipelineRunResult:
        """Execute pipeline with contract enforcement, cleaning, backtesting, forecast, and publication."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        report = self.diagnose(save_report=True)

        run_mode = "fixture" if self.is_fixture else "real"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        file_hashes = {p.file_name: p.sha256_hash for p in report.files_scanned}

        # Check for missing history
        if not report.is_ready:
            manifest = RunManifest(
                run_id=self.run_id,
                run_timestamp=timestamp,
                run_mode=run_mode,
                is_fixture=self.is_fixture,
                data_dir=str(self.data_dir.resolve()),
                out_dir=str(self.out_dir.resolve()),
                status="BLOCKED_MISSING_HISTORY",
                blocker=report.blocker_message,
                resolved_roles=report.role_mapping,
                file_hashes=file_hashes,
                diagnostics_summary={
                    "missing_roles": report.missing_roles,
                    "scanned_file_count": len(report.files_scanned),
                    "summary_notes": report.summary_notes,
                },
                metadata={
                    "stage": "milestone_4_forecast_and_submission",
                    "execution_halted_reason": "Missing historical ERP sales extract. Halting before training or publishing.",
                },
            )
            manifest_path = self.out_dir / "manifest.json"
            manifest.save(manifest_path)

            msg = (
                f"Run {self.run_id} BLOCKED: {report.blocker_message}\n"
                f"Diagnostic report written to {self.out_dir / 'diagnostics.txt'}\n"
                "To exercise the complete pipeline end-to-end, run with fixture data: --data fixtures/data"
            )
            return PipelineRunResult(
                success=False,
                status="BLOCKED_MISSING_HISTORY",
                run_id=self.run_id,
                data_dir=str(self.data_dir),
                out_dir=str(self.out_dir),
                run_mode=run_mode,
                manifest_path=str(manifest_path),
                message=msg,
                report=report,
                published_artifacts={},
            )

        # Ingestion succeeded: load resolved dataframes
        staging_dir = self.out_dir / "staging" / self.run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        history_files = report.role_mapping.get(DatasetRole.HISTORY.value, [])
        forecast_files = report.role_mapping.get(DatasetRole.FORECAST_ROWS.value, [])
        sku_master_files = report.role_mapping.get(DatasetRole.SKU_MASTER.value, [])
        stockouts_files = report.role_mapping.get(DatasetRole.STOCKOUTS.value, [])
        events_files = report.role_mapping.get(DatasetRole.EVENTS_CALENDAR.value, [])

        history_path = self.data_dir / history_files[0]
        forecast_path = self.data_dir / forecast_files[0]
        sku_master_path = self.data_dir / sku_master_files[0]
        stockouts_path = self.data_dir / stockouts_files[0] if stockouts_files else None
        events_path = self.data_dir / events_files[0] if events_files else None

        raw_history_df = pd.read_csv(history_path)
        forecast_keys_df = pd.read_csv(forecast_path)
        sku_master_df = pd.read_csv(sku_master_path) if sku_master_path.exists() else pd.DataFrame()
        stockouts_df = pd.read_csv(stockouts_path) if stockouts_path and stockouts_path.exists() else None
        events_df = pd.read_csv(events_path) if events_path and events_path.exists() else None

        # Step 1: Clean transactions and latest-load deduplication
        cleaned_history_df, quality_records = clean_history_transactions(
            history_df=raw_history_df,
            sku_master_df=sku_master_df,
            file_name=history_path.name,
        )

        # Step 2: Demand reconstruction, missing vs zero, backlog carry, anomalies
        history_panel_df, anomaly_records = reconstruct_demand_and_anomalies(
            cleaned_history=cleaned_history_df,
            sku_master_df=sku_master_df,
            stockouts_df=stockouts_df,
            events_df=events_df,
        )

        # Step 3: Run 19-month Backtest protocol across 3 folds and verify leakage safety
        backtest_df, backtest_summary = run_full_backtest(history_panel_df, DEFAULT_FOLDS)
        leakage_passed = verify_leakage_safety(history_panel_df, DEFAULT_FOLDS[0])
        winning_model = backtest_summary["selected_winner"]

        # Step 4: Final forecast generation and submission derivation
        forecast_df = generate_final_forecast(
            history_df=history_panel_df,
            forecast_keys_df=forecast_keys_df,
            sku_master_df=sku_master_df,
            winning_model_name=winning_model,
            events_df=events_df,
            cutoff_date="2026-10",
        )
        submission_df = derive_submission(forecast_df)

        # Validate submission strictly
        val_report = validate_submission(submission_df, forecast_keys_df, forecast_df)
        if not val_report.is_valid:
            raise ValueError(f"Submission validation failed: {val_report.notes}")

        # Step 5: Materialize dataframes
        quality_df = pd.DataFrame([q.to_dict() for q in quality_records])
        if quality_df.empty:
            quality_df = pd.DataFrame(columns=[
                "file_name", "source_row_id", "planning_key", "field", "rule",
                "original_value", "cleaned_value", "decision", "reason", "linked_evidence"
            ])

        anomalies_df = pd.DataFrame([a.to_dict() for a in anomaly_records])
        if anomalies_df.empty:
            anomalies_df = pd.DataFrame(columns=[
                "anomaly_id", "planning_key", "Date", "SKU_ID", "Region",
                "anomaly_type", "description", "original_value", "cleaned_value",
                "confidence", "action", "reason", "linked_evidence"
            ])

        # Step 6: Save published artifacts
        history_parquet = staging_dir / "history.parquet"
        quality_parquet = staging_dir / "quality.parquet"
        anomalies_parquet = staging_dir / "anomalies.parquet"
        backtest_parquet = staging_dir / "backtest.parquet"
        forecast_parquet = staging_dir / "forecast.parquet"
        submission_csv = staging_dir / "submission.csv"

        history_panel_df.to_parquet(history_parquet, index=False)
        quality_df.to_parquet(quality_parquet, index=False)
        anomalies_df.to_parquet(anomalies_parquet, index=False)
        backtest_df.to_parquet(backtest_parquet, index=False)
        forecast_df.to_parquet(forecast_parquet, index=False)
        submission_df.to_csv(submission_csv, index=False)

        # Copy to out_dir
        final_history = self.out_dir / "history.parquet"
        final_quality = self.out_dir / "quality.parquet"
        final_anomalies = self.out_dir / "anomalies.parquet"
        final_backtest = self.out_dir / "backtest.parquet"
        final_forecast = self.out_dir / "forecast.parquet"
        final_submission = self.out_dir / "submission.csv"

        shutil.copy2(history_parquet, final_history)
        shutil.copy2(quality_parquet, final_quality)
        shutil.copy2(anomalies_parquet, final_anomalies)
        shutil.copy2(backtest_parquet, final_backtest)
        shutil.copy2(forecast_parquet, final_forecast)
        shutil.copy2(submission_csv, final_submission)

        published_artifacts = {
            "history": str(final_history),
            "quality": str(final_quality),
            "anomalies": str(final_anomalies),
            "backtest": str(final_backtest),
            "forecast": str(final_forecast),
            "submission": str(final_submission),
        }

        # Step 7: Manifest recording
        manifest = RunManifest(
            run_id=self.run_id,
            run_timestamp=timestamp,
            run_mode=run_mode,
            is_fixture=self.is_fixture,
            data_dir=str(self.data_dir.resolve()),
            out_dir=str(self.out_dir.resolve()),
            status="SUCCESS",
            blocker=None,
            resolved_roles=report.role_mapping,
            file_hashes=file_hashes,
            diagnostics_summary={
                "missing_roles": [],
                "scanned_file_count": len(report.files_scanned),
                "summary_notes": report.summary_notes,
                "raw_history_rows": len(raw_history_df),
                "cleaned_history_rows": len(cleaned_history_df),
                "panel_history_rows": len(history_panel_df),
                "observed_rows": int(history_panel_df["is_observed"].sum()),
                "unobserved_missing_rows": int((~history_panel_df["is_observed"]).sum()),
                "quality_issues_logged": len(quality_records),
                "anomalies_logged": len(anomaly_records),
                "total_reconstructed_demand": float(history_panel_df["reconstructed_demand"].sum(skipna=True)),
                "backtest_rows": len(backtest_df),
                "leakage_check_passed": leakage_passed,
                "selected_winning_model": winning_model,
                "development_metrics": backtest_summary["development_metrics"],
                "frozen_holdout_metrics": backtest_summary["frozen_holdout_metrics"],
                "submission_rows": val_report.total_rows,
                "submission_total_units": val_report.total_units_forecast,
                "submission_projected_revenue": val_report.total_projected_revenue,
                "submission_validation_passed": val_report.is_valid,
            },
            metadata={
                "stage": "milestone_4_forecast_and_submission",
                "published_artifacts": published_artifacts,
                "backtest_summary": backtest_summary,
                "submission_validation": {
                    "is_valid": val_report.is_valid,
                    "total_rows": val_report.total_rows,
                    "columns": val_report.columns,
                    "notes": val_report.notes,
                },
            },
        )
        staging_manifest = staging_dir / "manifest.json"
        manifest.save(staging_manifest)

        manifest_path = self.out_dir / "manifest.json"
        manifest.save(manifest_path)

        current_pointer = self.out_dir / "current_run.json"
        with open(current_pointer, "w", encoding="utf-8") as f:
            json.dump({
                "current_run_id": self.run_id,
                "run_mode": run_mode,
                "timestamp": timestamp,
                "manifest": str(manifest_path),
                "published_artifacts": published_artifacts,
            }, f, indent=2)

        msg = (
            f"Run {self.run_id} SUCCESS ({run_mode.upper()} MODE):\n"
            f"Forecast published: {len(forecast_df)} rows in forecast.parquet\n"
            f"Submission derived: {len(submission_df)} rows in submission.csv\n"
            f"Validation: 100% verified (exact 4 columns, exact key sequence, no nulls, non-negative integers)\n"
            f"Total Forecast Units: {val_report.total_units_forecast:,} units | Projected Revenue: ${val_report.total_projected_revenue:,.2f}\n"
            f"Published artifacts: {list(published_artifacts.keys())}\n"
            f"Manifest saved to {manifest_path}"
        )
        return PipelineRunResult(
            success=True,
            status="SUCCESS",
            run_id=self.run_id,
            data_dir=str(self.data_dir),
            out_dir=str(self.out_dir),
            run_mode=run_mode,
            manifest_path=str(manifest_path),
            message=msg,
            report=report,
            published_artifacts=published_artifacts,
            backtest_summary=backtest_summary,
            submission_validation=val_report,
        )
