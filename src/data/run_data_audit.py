"""Run an initial data audit and save report artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.load_data import DEFAULT_RAW_DATA_DIR, configure_logging, load_dataset, resolve_data_path
from src.data.preprocess import build_data_quality_summary


REPORTS_DIR = Path("artifacts/reports")
MARKDOWN_REPORT_PATH = REPORTS_DIR / "data_audit_report.md"
JSON_SUMMARY_PATH = REPORTS_DIR / "data_audit_summary.json"


def _json_default(value: Any) -> Any:
    """Convert numpy and pandas scalar types to plain Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def format_markdown_report(dataset_path: Path, summary: dict[str, Any]) -> str:
    """Render the markdown report."""
    missing_lines = [
        f"| {column} | {count} |"
        for column, count in summary["missing_values"].items()
    ]
    candidate_columns = summary["candidate_columns"]

    lines = [
        "# Data Audit Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Columns: `{summary['column_count']}`",
        f"- Duplicate rows: `{summary['duplicate_rows']}`",
        "",
        "## Schema",
        "",
        "| Column | Dtype |",
        "| --- | --- |",
    ]
    lines.extend(f"| {column} | {dtype} |" for column, dtype in summary["dtypes"].items())

    lines.extend(
        [
            "",
            "## Missing Values",
            "",
            "| Column | Missing count |",
            "| --- | --- |",
            *missing_lines,
            "",
            "## Column Groups",
            "",
            f"- Numeric columns: {summary['numeric_columns']}",
            f"- Categorical columns: {summary['categorical_columns']}",
            "",
            "## Candidate Special Columns",
            "",
            f"- Target candidates: {candidate_columns['target']}",
            f"- Timestamp candidates: {candidate_columns['timestamp']}",
            f"- User ID candidates: {candidate_columns['user_id']}",
            f"- Transaction ID candidates: {candidate_columns['transaction_id']}",
        ]
    )
    return "\n".join(lines) + "\n"


def save_report_files(markdown_content: str, summary: dict[str, Any]) -> None:
    """Persist report artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT_PATH.write_text(markdown_content, encoding="utf-8")
    JSON_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a basic data quality audit for a raw dataset.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Dataset filename inside data/raw. If omitted, the first supported file is used.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    dataset_path = resolve_data_path(filename=args.file, data_dir=DEFAULT_RAW_DATA_DIR)
    dataframe = load_dataset(dataset_path)
    summary = build_data_quality_summary(dataframe)
    markdown_report = format_markdown_report(dataset_path, summary)
    save_report_files(markdown_report, summary)


if __name__ == "__main__":
    main()
