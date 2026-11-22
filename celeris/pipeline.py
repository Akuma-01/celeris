"""Pipeline coordination and execution for Celeris."""

from __future__ import annotations

import datetime
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from celeris.ingest import (
    DiagnosticReport,
    diagnose_input_directory,
    format_diagnostic_report,
)
from celeris.manifest import RunManifest


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


class PipelineRunner:
    """Coordinates ingestion, role resolution, diagnostics, and staged runs."""

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

        # Auto-detect fixture if not explicitly specified
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
        """Execute pipeline with contract enforcement and staging."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        report = self.diagnose(save_report=True)

        run_mode = "fixture" if self.is_fixture else "real"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Input hashes
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
                    "stage": "milestone_1_skeleton",
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
            )

        # Ingestion succeeded and all roles resolved!
        staging_dir = self.out_dir / "staging" / self.run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

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
                "test_rows": report.forecast_keys_profile.total_rows if report.forecast_keys_profile else 0,
            },
            metadata={
                "stage": "milestone_1_skeleton",
                "notes": "Input contracts verified; skeleton pipeline ready for cleaning stages.",
            },
        )
        staging_manifest = staging_dir / "manifest.json"
        manifest.save(staging_manifest)

        # Switch current pointer / copy to out_dir
        manifest_path = self.out_dir / "manifest.json"
        manifest.save(manifest_path)

        # Save pointer
        current_pointer = self.out_dir / "current_run.json"
        with open(current_pointer, "w", encoding="utf-8") as f:
            json.dump({
                "current_run_id": self.run_id,
                "run_mode": run_mode,
                "timestamp": timestamp,
                "manifest": str(manifest_path),
            }, f, indent=2)

        msg = (
            f"Run {self.run_id} SUCCESS ({run_mode.upper()} MODE):\n"
            f"All seven dataset roles resolved successfully.\n"
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
        )
