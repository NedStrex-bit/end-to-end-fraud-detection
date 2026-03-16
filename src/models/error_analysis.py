"""Thin wrappers for error analysis reporting."""

from __future__ import annotations

from typing import Any


def build_explainability_report(
    top_features: list[dict[str, Any]],
    explainability_notes: list[str],
    importance_plot_path: str,
    threshold: float,
) -> str:
    """Render the explainability markdown report."""
    lines = [
        "# Explainability Report",
        "",
        f"- Decision threshold used for local interpretation: `{threshold:.2f}`",
        f"- Feature importance plot: `{importance_plot_path}`",
        "",
        "## Main Drivers",
        "",
    ]
    lines.extend(
        f"- `{row['feature']}`: importance=`{row['importance']:.6f}` ({row.get('importance_type', 'importance')})"
        for row in top_features
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Эти признаки отражают те же источники сигнала, которые уже появлялись в EDA и feature engineering: сумма транзакции, временные признаки и пользовательская динамика.",
            "- Наиболее бизнес-осмысленные признаки обычно связаны с объёмом транзакции, временем события и отклонением от типичного поведения пользователя.",
            "- Если модель действительно опирается на amount/time/velocity features, это согласуется с типичной природой fraud-паттернов в табличных транзакционных данных.",
            "",
            "## Method Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in explainability_notes)
    return "\n".join(lines) + "\n"


def build_error_analysis_report(
    findings: list[str],
    confusion_matrix: list[list[int]],
    summary_table_path: str,
    exported_case_paths: dict[str, str],
    plotted_features: list[str],
    threshold: float,
) -> str:
    """Render the error analysis markdown report."""
    lines = [
        "# Error Analysis Report",
        "",
        f"- Threshold used for decisions: `{threshold:.2f}`",
        f"- Confusion matrix: `{confusion_matrix}`",
        f"- Error summary table: `{summary_table_path}`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Saved Tables",
            "",
            f"- FP cases: `{exported_case_paths.get('FP')}`",
            f"- FN cases: `{exported_case_paths.get('FN')}`",
            f"- TP cases: `{exported_case_paths.get('TP')}`",
            f"- TN cases: `{exported_case_paths.get('TN')}`",
            "",
            "## Improvement Directions",
            "",
            "- Сложными обычно остаются fraud-кейсы, которые выглядят похоже на обычное пользовательское поведение по сумме и времени.",
            "- Для будущих версий проекта стоит усиливать user-level velocity features, merchant/network features и более длинные исторические окна.",
            "- Если модель делает много false positives, стоит отдельно оптимизировать threshold и calibrated alert policy под бизнес-стоимость ручной проверки.",
        ]
    )
    if plotted_features:
        lines.extend(
            [
                "",
                "## Plotted Feature Comparisons",
                "",
            ]
        )
        lines.extend(f"- `artifacts/plots/{feature}_by_error_type.png`" for feature in plotted_features)
    return "\n".join(lines) + "\n"
