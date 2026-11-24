"""Command-line interface for Celeris Automobili demand forecasting pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from celeris.ingest import format_diagnostic_report
from celeris.pipeline import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="celeris",
        description="Celeris Automobili Industrial Demand Forecasting & Planning Pipeline",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="celeris 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run the demand forecasting pipeline")
    run_parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="student_data",
        help="Path to directory containing input CSV files (default: student_data)",
    )
    run_parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="artifacts",
        help="Path to output directory for artifacts (default: artifacts)",
    )
    run_parser.add_argument(
        "--fixture",
        action="store_true",
        help="Explicitly mark this run as a fixture validation run",
    )

    # Command: diagnose
    diag_parser = subparsers.add_parser("diagnose", help="Diagnose input CSV files and resolve roles")
    diag_parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="student_data",
        help="Path to directory containing input CSV files (default: student_data)",
    )
    diag_parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Optional path to output directory for saving diagnostic report",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "diagnose":
        out_dir = Path(args.out) if args.out else Path("artifacts/diagnostics")
        runner = PipelineRunner(data_dir=args.data, out_dir=out_dir)
        report = runner.diagnose(save_report=bool(args.out))
        print(format_diagnostic_report(report))
        if not report.is_ready:
            print("\n[!] ATTENTION: Missing historical sales extract. Real-data run cannot proceed to fitting.")
            print("    To run an end-to-end backtest and publication, use the labeled fixture:")
            print("    python -m celeris run --data fixtures/data --out artifacts/fixture_run\n")
            return 2
        return 0

    if args.command == "run":
        runner = PipelineRunner(
            data_dir=args.data,
            out_dir=args.out,
            is_fixture=True if args.fixture else None,
        )
        result = runner.run()
        print(format_diagnostic_report(result.report))
        print("\n" + result.message)

        if not result.success:
            return 2
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
