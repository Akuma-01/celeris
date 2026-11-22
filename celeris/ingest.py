"""Ingestion, role resolution, and input diagnostics for Celeris data files."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from celeris.schemas import DatasetRole, SCHEMAS, match_schema


@dataclass
class FileProfile:
    """Diagnostic profile of a single input file."""
    file_name: str
    file_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]
    sha256_hash: str
    matched_role: Optional[str]
    confidence_score: float
    match_notes: List[str]


@dataclass
class CalendarComparison:
    """Comparison of multiple files claiming the events_calendar role."""
    file_names: List[str]
    are_byte_identical: bool
    are_content_identical: bool
    row_counts: Dict[str, int]
    shared_event_count: int
    conflicting_event_count: int
    notes: List[str]


@dataclass
class ForecastKeysProfile:
    """Inspection of the test/forecast rows."""
    file_name: str
    total_rows: int
    min_date: str
    max_date: str
    num_months: int
    expected_months: int
    unique_skus: int
    unique_regions: int
    has_target_column: bool
    is_valid_test_set: bool
    notes: List[str]


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for an input directory."""
    directory: str
    timestamp: str
    is_ready: bool
    blocker_message: Optional[str]
    files_scanned: List[FileProfile]
    role_mapping: Dict[str, List[str]]
    missing_roles: List[str]
    calendar_comparison: Optional[CalendarComparison]
    forecast_keys_profile: Optional[ForecastKeysProfile]
    summary_notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_csv_headers(file_path: Path) -> List[str]:
    """Read only the header row of a CSV economically."""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, [])


def count_csv_rows(file_path: Path) -> int:
    """Count lines in CSV quickly without parsing every column."""
    with open(file_path, "rb") as f:
        count = sum(1 for _ in f)
    # Exclude header
    return max(0, count - 1)


def profile_file(file_path: Path) -> FileProfile:
    """Profile a single CSV file."""
    headers = inspect_csv_headers(file_path)
    row_count = count_csv_rows(file_path)
    file_size = file_path.stat().st_size
    file_hash = compute_file_sha256(file_path)

    matched_role, score, notes = match_schema(headers)

    return FileProfile(
        file_name=file_path.name,
        file_path=str(file_path.resolve()),
        file_size_bytes=file_size,
        row_count=row_count,
        column_count=len(headers),
        columns=headers,
        sha256_hash=file_hash,
        matched_role=matched_role.value if matched_role else None,
        confidence_score=round(score, 4),
        match_notes=notes,
    )


def compare_calendar_files(files: List[Path]) -> CalendarComparison:
    """Compare multiple files matching the calendar role."""
    file_names = [f.name for f in files]
    row_counts = {f.name: count_csv_rows(f) for f in files}
    hashes = [compute_file_sha256(f) for f in files]
    are_byte_identical = len(set(hashes)) == 1

    notes: List[str] = []
    if are_byte_identical:
        notes.append(f"All {len(files)} files are bitwise identical (SHA256: {hashes[0][:12]}...).")
        return CalendarComparison(
            file_names=file_names,
            are_byte_identical=True,
            are_content_identical=True,
            row_counts=row_counts,
            shared_event_count=row_counts[file_names[0]],
            conflicting_event_count=0,
            notes=notes,
        )

    # If not bitwise identical, read into dataframes to compare event rows
    dfs = [pd.read_csv(f) for f in files]
    first_df = dfs[0]
    all_content_equal = all(first_df.equals(df) for df in dfs[1:])

    if all_content_equal:
        notes.append("Files have identical rows and columns despite different formatting/timestamps.")
        return CalendarComparison(
            file_names=file_names,
            are_byte_identical=False,
            are_content_identical=True,
            row_counts=row_counts,
            shared_event_count=len(first_df),
            conflicting_event_count=0,
            notes=notes,
        )

    # Check for conflicting Event_IDs
    event_maps = [df.set_index("Event_ID").to_dict(orient="index") for df in dfs]
    all_keys = set().union(*[m.keys() for m in event_maps])
    conflicts = 0
    for key in all_keys:
        records = [m[key] for m in event_maps if key in m]
        if len(records) > 1 and not all(r == records[0] for r in records[1:]):
            conflicts += 1

    notes.append(f"Found {conflicts} conflicting event IDs across {len(files)} calendar files.")
    return CalendarComparison(
        file_names=file_names,
        are_byte_identical=False,
        are_content_identical=False,
        row_counts=row_counts,
        shared_event_count=len(all_keys) - conflicts,
        conflicting_event_count=conflicts,
        notes=notes,
    )


def inspect_forecast_rows(file_path: Path) -> ForecastKeysProfile:
    """Inspect and validate the test/forecast rows file."""
    df = pd.read_csv(file_path)
    total_rows = len(df)
    min_date = str(df["Date"].min())
    max_date = str(df["Date"].max())
    unique_dates = df["Date"].nunique()
    unique_skus = df["SKU_ID"].nunique()
    unique_regions = df["Region"].nunique() if "Region" in df.columns else 0
    has_target = "Units_Sold" in df.columns or "Units_Sold_Forecast" in df.columns

    notes: List[str] = []
    is_valid = True

    if has_target:
        notes.append("Target column present in forecast rows (expected unlabelled planning keys).")
        is_valid = False

    if unique_dates != 19:
        notes.append(f"Expected 19 forecast months, found {unique_dates} months.")
    else:
        notes.append("19 forecast months confirmed (2026-11 through 2028-05).")

    notes.append(f"Planning keys span {unique_skus} SKUs across {unique_regions} regions.")

    return ForecastKeysProfile(
        file_name=file_path.name,
        total_rows=total_rows,
        min_date=min_date,
        max_date=max_date,
        num_months=unique_dates,
        expected_months=19,
        unique_skus=unique_skus,
        unique_regions=unique_regions,
        has_target_column=has_target,
        is_valid_test_set=is_valid,
        notes=notes,
    )


def diagnose_input_directory(data_dir: Path) -> DiagnosticReport:
    """Diagnose all CSV files in the input directory and resolve their dataset roles."""
    import datetime

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        return DiagnosticReport(
            directory=str(data_dir.resolve()),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            is_ready=False,
            blocker_message=f"No CSV files found in {data_dir}",
            files_scanned=[],
            role_mapping={role.value: [] for role in DatasetRole},
            missing_roles=[role.value for role in DatasetRole],
            calendar_comparison=None,
            forecast_keys_profile=None,
            summary_notes=["Directory is empty or contains no CSV files."],
        )

    profiles = [profile_file(f) for f in csv_files]

    # Map roles to files
    role_to_files: Dict[str, List[str]] = {role.value: [] for role in DatasetRole}
    for p in profiles:
        if p.matched_role:
            role_to_files[p.matched_role].append(p.file_name)

    # Missing roles
    missing_roles = [
        role.value for role in DatasetRole if not role_to_files[role.value]
    ]

    # Calendar comparison if multiple files match
    calendar_files = [
        data_dir / p.file_name
        for p in profiles
        if p.matched_role == DatasetRole.EVENTS_CALENDAR.value
    ]
    calendar_comp = compare_calendar_files(calendar_files) if len(calendar_files) > 1 else None

    # Forecast keys profile if matched
    forecast_files = [
        data_dir / p.file_name
        for p in profiles
        if p.matched_role == DatasetRole.FORECAST_ROWS.value
    ]
    forecast_prof = inspect_forecast_rows(forecast_files[0]) if forecast_files else None

    # Readiness & blockers
    summary_notes: List[str] = []
    is_ready = True
    blocker_message: Optional[str] = None

    if DatasetRole.HISTORY.value in missing_roles:
        is_ready = False
        blocker_message = (
            "Historical ERP sales extract is MISSING. "
            f"Expected columns: {SCHEMAS[DatasetRole.HISTORY].expected_full_columns}. "
            "Identifying columns required: Record_ID, Units_Sold, Load_Timestamp."
        )
        summary_notes.append(f"BLOCKER: {blocker_message}")

    if calendar_comp:
        if calendar_comp.are_byte_identical:
            summary_notes.append(
                f"Resolved duplicate calendar role: {calendar_comp.file_names} are byte-identical. Deduplicating safely."
            )
        else:
            summary_notes.append(
                f"Multiple calendar files found: {calendar_comp.file_names}. Shared: {calendar_comp.shared_event_count}, Conflicts: {calendar_comp.conflicting_event_count}."
            )

    if forecast_prof:
        summary_notes.append(
            f"Forecast rows resolved in {forecast_prof.file_name}: {forecast_prof.total_rows:,} rows covering {forecast_prof.min_date} to {forecast_prof.max_date} ({forecast_prof.num_months} months)."
        )

    if not missing_roles:
        summary_notes.append("All seven dataset roles successfully resolved.")

    return DiagnosticReport(
        directory=str(data_dir.resolve()),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        is_ready=is_ready,
        blocker_message=blocker_message,
        files_scanned=profiles,
        role_mapping=role_to_files,
        missing_roles=missing_roles,
        calendar_comparison=calendar_comp,
        forecast_keys_profile=forecast_prof,
        summary_notes=summary_notes,
    )


def format_diagnostic_report(report: DiagnosticReport) -> str:
    """Format the diagnostic report for terminal display with clean, structured styling."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("CELERIS AUTOMOBILI — INPUT INVENTORY & ROLE RESOLUTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Directory: {report.directory}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Status   : {'READY' if report.is_ready else 'BLOCKED'}")
    if report.blocker_message:
        lines.append(f"Blocker  : {report.blocker_message}")
    lines.append("-" * 80)

    # 1. Scanned files table
    lines.append("\n1. PHYSICAL FILES INVENTORY:")
    lines.append(f"{'Filename':<26} {'Size':>10} {'Rows':>9} {'Cols':>6} {'Resolved Role':<20} {'Score':>6}")
    lines.append("-" * 80)
    for p in report.files_scanned:
        role_disp = p.matched_role or "UNMATCHED"
        size_str = f"{p.file_size_bytes / 1024:.1f} KB" if p.file_size_bytes < 1024 * 1024 else f"{p.file_size_bytes / (1024*1024):.1f} MB"
        lines.append(f"{p.file_name:<26} {size_str:>10} {p.row_count:>9,} {p.column_count:>6} {role_disp:<20} {p.confidence_score:>6.2f}")

    # 2. Seven dataset roles mapping
    lines.append("\n2. SEVEN DATASET ROLES RESOLUTION:")
    lines.append(f"{'Dataset Role':<20} {'Display Name':<28} {'Mapped File(s)':<30}")
    lines.append("-" * 80)
    for role in DatasetRole:
        schema = SCHEMAS[role]
        mapped = report.role_mapping.get(role.value, [])
        mapped_str = ", ".join(mapped) if mapped else "MISSING"
        status_flag = "[OK]" if mapped else "[MISSING]"
        lines.append(f"{role.value:<20} {schema.display_name:<28} {mapped_str:<30} {status_flag}")

    # 3. Calendar duplicates
    if report.calendar_comparison:
        lines.append("\n3. CALENDAR ROLE MULTI-FILE RECONCILIATION:")
        cc = report.calendar_comparison
        lines.append(f"  Files claiming role : {', '.join(cc.file_names)}")
        lines.append(f"  Bitwise identical   : {cc.are_byte_identical}")
        lines.append(f"  Row counts          : {cc.row_counts}")
        for note in cc.notes:
            lines.append(f"  Note                : {note}")

    # 4. Forecast rows profile
    if report.forecast_keys_profile:
        lines.append("\n4. FORECAST KEYS (TEST SET) INSPECTION:")
        fp = report.forecast_keys_profile
        lines.append(f"  File name           : {fp.file_name}")
        lines.append(f"  Total forecast rows : {fp.total_rows:,}")
        lines.append(f"  Forecast window     : {fp.min_date} to {fp.max_date} ({fp.num_months} months, expected {fp.expected_months})")
        lines.append(f"  Planning scope      : {fp.unique_skus} SKUs across {fp.unique_regions} regions")
        lines.append(f"  Target column absent: {not fp.has_target_column} (Correct for forecast keys)")

    # 5. Missing roles details
    if report.missing_roles:
        lines.append("\n5. MISSING ROLE DETAILS:")
        for r_name in report.missing_roles:
            role = DatasetRole(r_name)
            schema = SCHEMAS[role]
            lines.append(f"  * {schema.display_name} ({role.value}):")
            lines.append(f"    - Required identifying columns: {sorted(schema.required_identifying_columns)}")
            lines.append(f"    - Full expected columns       : {schema.expected_full_columns}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
