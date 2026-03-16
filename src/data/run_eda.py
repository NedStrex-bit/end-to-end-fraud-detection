"""Run reproducible EDA and save plots and a markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.eda import run_full_eda
from src.data.load_data import DEFAULT_RAW_DATA_DIR, configure_logging, load_dataset, resolve_data_path
from src.data.preprocess import get_first_candidate


REPORTS_DIR = Path("artifacts/reports")
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.md"


def build_eda_report(dataset_path: Path, eda_summary: dict) -> str:
    """Render an EDA markdown report with concrete findings."""
    dataset_summary = eda_summary["dataset_summary"]
    target_distribution = eda_summary["target_distribution"]
    amount_analysis = eda_summary["amount_analysis"]
    fraud_group_analysis = eda_summary["fraud_group_analysis"]
    target_amount_analysis = eda_summary["target_amount_analysis"]
    temporal_analysis = eda_summary["temporal_analysis"]
    numeric_analysis = eda_summary["numeric_analysis"]
    notes = eda_summary["notes"]

    missing_values = dataset_summary["missing_values"]
    missing_columns = {column: count for column, count in missing_values.items() if count > 0}
    high_missing_columns = [
        column for column, count in missing_values.items() if dataset_summary["row_count"] and count / dataset_summary["row_count"] > 0.2
    ]

    findings: list[str] = []
    if target_distribution["available"]:
        findings.append(
            f"Классовый баланс по `{dataset_summary['target_column']}`: fraud rate = {target_distribution['fraud_rate']:.2%}."
        )
    else:
        findings.append("Целевая переменная не была надёжно определена, поэтому баланс классов требует ручной проверки.")

    if amount_analysis["available"]:
        amount_column = amount_analysis["amount_column"]
        summary = amount_analysis["summary"]
        findings.append(
            f"Признак `{amount_column}` выглядит информативным: median={summary['median']:.2f}, p95={summary['p95']:.2f}, max={summary['max']:.2f}."
        )

    if fraud_group_analysis["available"]:
        findings.append(
            f"Категориальный признак `{fraud_group_analysis['best_group_column']}` показывает различия fraud rate между группами."
        )

    if temporal_analysis["available"]:
        findings.append(
            f"Во временном анализе использована колонка `{dataset_summary['timestamp_column']}`; паттерны по часу и дню недели сохранены в артефакты."
        )
    else:
        findings.append("Временной анализ недоступен: timestamp-колонка не найдена или не распарсилась.")

    if high_missing_columns:
        findings.append(f"Колонки с заметной долей пропусков (>20%): {high_missing_columns}.")
    elif missing_columns:
        findings.append(f"Пропуски есть, но локальны: {missing_columns}.")
    else:
        findings.append("Существенных пропусков на текущем датасете не обнаружено.")

    leakage_risks: list[str] = []
    if dataset_summary["user_id_column"]:
        leakage_risks.append(
            f"`{dataset_summary['user_id_column']}` может приводить к переобучению на конкретных клиентах и требует аккуратной агрегации."
        )
    if dataset_summary["timestamp_column"]:
        leakage_risks.append(
            f"`{dataset_summary['timestamp_column']}` нельзя использовать напрямую без проверки временного split и причинности признаков."
        )
    if dataset_summary["transaction_id_column"]:
        leakage_risks.append(
            f"`{dataset_summary['transaction_id_column']}` является идентификатором транзакции и не должен попадать в модель напрямую."
        )
    leakage_risks.append("Любые ID-поля и пост-фактум статусные признаки нужно исключать из прямой модельной матрицы.")

    feature_hypotheses: list[str] = []
    if amount_analysis["available"]:
        amount_column = amount_analysis["amount_column"]
        feature_hypotheses.append(f"Логарифм `{amount_column}`, биннинг суммы и robust percentiles по пользователю/мерчанту.")
    if dataset_summary["timestamp_column"]:
        feature_hypotheses.append("Час транзакции, день недели, признак ночной активности, циклические time-features.")
    if fraud_group_analysis["available"]:
        feature_hypotheses.append(
            f"Target-agnostic frequency features по `{fraud_group_analysis['best_group_column']}` и related count encodings."
        )
    if dataset_summary["user_id_column"]:
        feature_hypotheses.append("User-level velocity features: число транзакций, уникальные мерчанты и сумма за окна 1h/24h/7d.")
    feature_hypotheses.append("Missingness indicators для признаков с пропусками.")

    useful_features: list[str] = []
    if amount_analysis["available"]:
        useful_features.append(f"`{amount_analysis['amount_column']}`")
    useful_features.extend(
        f"`{column}`"
        for column in numeric_analysis["numeric_columns"][:5]
        if column != dataset_summary["target_column"]
    )
    if fraud_group_analysis["available"]:
        useful_features.append(f"`{fraud_group_analysis['best_group_column']}`")
    useful_features = list(dict.fromkeys(useful_features))

    lines = [
        "# EDA Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Rows: `{dataset_summary['row_count']}`",
        f"- Columns: `{dataset_summary['column_count']}`",
        f"- Target column: `{dataset_summary['target_column']}`",
        f"- Timestamp column: `{dataset_summary['timestamp_column']}`",
        f"- User ID column: `{dataset_summary['user_id_column']}`",
        f"- Transaction ID column: `{dataset_summary['transaction_id_column']}`",
        "",
        "## Overall Size And Quality",
        "",
        f"- Duplicate rows: `{dataset_summary['duplicate_rows']}`",
        f"- Missing values by column: `{missing_columns if missing_columns else 'no missing values detected'}`",
        "",
        "## Class Balance",
        "",
    ]

    if target_distribution["available"]:
        lines.extend(
            [
                f"- Fraud counts: `{target_distribution['counts']}`",
                f"- Fraud rates: `{target_distribution['rates']}`",
                f"- Plot: `artifacts/plots/target_distribution.png`",
            ]
        )
    else:
        lines.append("- Target distribution was skipped because the target column was unavailable or non-binary.")

    lines.extend(
        [
            "",
            "## Generated Plots",
            "",
            "- `artifacts/plots/target_distribution.png`",
            "- `artifacts/plots/amount_distribution.png` if an amount-like column exists",
            "- `artifacts/plots/fraud_rate_by_group.png` if a categorical grouping is available",
            "- `artifacts/plots/amount_percentiles_by_target.png` if target and amount are both available",
            "- `artifacts/plots/fraud_rate_by_hour.png` if timestamp is available",
            "- `artifacts/plots/fraud_rate_by_day_of_week.png` if timestamp is available",
            "- `artifacts/plots/numeric_correlation_heatmap.png` if at least two numeric columns exist",
            "",
            "## Key Findings",
            "",
        ]
    )
    lines.extend(f"- {finding}" for finding in findings)

    lines.extend(
        [
            "",
            "## Potentially Useful Features",
            "",
            f"- Candidate useful columns: {useful_features if useful_features else 'none identified automatically'}",
            "",
            "## Data Leakage Risks",
            "",
        ]
    )
    lines.extend(f"- {risk}" for risk in leakage_risks)

    lines.extend(
        [
            "",
            "## Feature Engineering Hypotheses",
            "",
        ]
    )
    lines.extend(f"- {hypothesis}" for hypothesis in feature_hypotheses)

    lines.extend(
        [
            "",
            "## Amount Feature Comparison",
            "",
        ]
    )
    if target_amount_analysis["available"]:
        lines.extend(
            [
                f"- Amount column: `{target_amount_analysis['amount_column']}`",
                f"- Percentiles by target: `{target_amount_analysis['percentiles']}`",
                f"- Plot: `artifacts/plots/amount_percentiles_by_target.png`",
            ]
        )
    else:
        lines.append("- Fraud vs non-fraud amount comparison was not available.")

    lines.extend(
        [
            "",
            "## Temporal Patterns",
            "",
        ]
    )
    if temporal_analysis["available"]:
        lines.extend(
            [
                "- Plot: `artifacts/plots/fraud_rate_by_hour.png`",
                "- Plot: `artifacts/plots/fraud_rate_by_day_of_week.png`",
                f"- Hourly fraud rates: `{temporal_analysis['hour_rates']}`",
                f"- Day-of-week fraud rates: `{temporal_analysis['day_of_week_rates']}`",
            ]
        )
    else:
        lines.append("- Timestamp analysis is unavailable for this dataset.")

    lines.extend(
        [
            "",
            "## Notes And Graceful Degradation",
            "",
        ]
    )
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- All expected EDA blocks ran successfully.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run reproducible EDA and save visual artifacts.")
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

    target_column = get_first_candidate(dataframe, "target")
    timestamp_column = get_first_candidate(dataframe, "timestamp")
    user_id_column = get_first_candidate(dataframe, "user_id")
    transaction_id_column = get_first_candidate(dataframe, "transaction_id")

    eda_summary = run_full_eda(
        dataframe=dataframe,
        target_column=target_column,
        timestamp_column=timestamp_column,
        user_id_column=user_id_column,
        transaction_id_column=transaction_id_column,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EDA_REPORT_PATH.write_text(build_eda_report(dataset_path, eda_summary), encoding="utf-8")


if __name__ == "__main__":
    main()
