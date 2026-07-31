
#!/usr/bin/env python3
"""
Run the litigation and stock-price analyses from the terminal.

Example
-------
python run_litigation_analysis.py \
    --cases cleaned_dataframe.csv \
    --stocks all_stock_data.parquet \
    --output-dir analysis_output

Supported input formats: CSV, Parquet, Pickle, Feather. Simple filenames work
when the data files are in the current working directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from litigation_stock_analysis import (
    ANALYSIS_VERSION,
    analyze_case_outcome_factors,
    analyze_stock_price_impact,
    plot_case_outcome_factors,
    plot_stock_price_impact,
)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(
        f"Unsupported file type {suffix!r}. Use CSV, Parquet, Pickle, or Feather."
    )


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save_goal1_tables(results: dict, output_dir: Path) -> None:
    results["metrics"].to_csv(output_dir / "goal1_model_metrics.csv")
    results["exclusion_summary"].to_csv(
        output_dir / "goal1_outcome_exclusions.csv", index=False
    )
    results["cv_predictions"].to_csv(
        output_dir / "goal1_cross_validated_predictions.csv", index=False
    )
    results["party_win_rates"].to_csv(
        output_dir / "goal1_party_win_rates.csv", index=False
    )
    results["district_win_rates"].to_csv(
        output_dir / "goal1_district_win_rates.csv", index=False
    )

    for model_name, table in results["feature_effects"].items():
        table.to_csv(
            output_dir / f"goal1_{model_name}_feature_effects.csv",
            index=False,
        )
    for model_name, table in results["permutation_importance"].items():
        table.to_csv(
            output_dir / f"goal1_{model_name}_permutation_importance.csv",
            index=False,
        )


def save_goal2_tables(results: dict, output_dir: Path) -> None:
    combined_tables = {
        "event_level": "goal2_event_level_cars.csv",
        "analysis_events": "goal2_primary_analysis_events.csv",
        "daily_abnormal_returns": "goal2_daily_abnormal_returns.csv",
        "group_summary": "goal2_group_summary.csv",
        "win_loss_tests": "goal2_win_loss_tests.csv",
        "regression_table": "goal2_car_regressions.csv",
        "regression_metadata": "goal2_car_regression_metadata.csv",
        "failed_events": "goal2_failed_events.csv",
    }
    for key, filename in combined_tables.items():
        table = results.get(key)
        if isinstance(table, pd.DataFrame):
            table.to_csv(output_dir / filename, index=False)

    for date_source, date_results in results.get("by_event_date", {}).items():
        prefix = f"goal2_{date_source}"
        date_tables = {
            "event_level": f"{prefix}_event_level_cars.csv",
            "analysis_events": f"{prefix}_primary_analysis_events.csv",
            "daily_abnormal_returns": f"{prefix}_daily_abnormal_returns.csv",
            "group_summary": f"{prefix}_group_summary.csv",
            "win_loss_tests": f"{prefix}_win_loss_tests.csv",
            "regression_table": f"{prefix}_car_regressions.csv",
            "regression_metadata": f"{prefix}_car_regression_metadata.csv",
            "failed_events": f"{prefix}_failed_events.csv",
        }
        for key, filename in date_tables.items():
            table = date_results.get(key)
            if isinstance(table, pd.DataFrame):
                table.to_csv(output_dir / filename, index=False)


def print_goal1_summary(results: dict) -> None:
    print("\n=== Goal 1: Factors associated with case outcome ===")
    print(results["metrics"].round(4).to_string())

    effects = results["feature_effects"]["full"].copy()
    stable = effects.loc[
        (effects["ci_low_log_odds"] > 0) | (effects["ci_high_log_odds"] < 0)
    ].head(10)
    if stable.empty:
        stable = effects.head(10)

    print("\nTop adjusted factors from the full model:")
    columns = [
        "feature",
        "odds_ratio",
        "ci_low_odds_ratio",
        "ci_high_odds_ratio",
        "direction",
    ]
    print(stable[columns].round(4).to_string(index=False))


def print_goal2_summary(results: dict) -> None:
    print("\n=== Goal 2: Stock-price impact of case outcome ===")
    tests = results["win_loss_tests"].copy()
    columns = [
        "event_date_source",
        "impact_horizon_days",
        "n_wins",
        "n_losses",
        "mean_win_minus_loss",
        "cluster_bootstrap_ci_low",
        "cluster_bootstrap_ci_high",
        "welch_p_value",
        "permutation_p_value",
    ]
    available = [col for col in columns if col in tests.columns]
    print(tests[available].round(6).to_string(index=False))

    regression = results.get("regression_table", pd.DataFrame())
    if not regression.empty and regression["term"].eq("entity_win").any():
        print("\nAdjusted win coefficients:")
        rows = regression.loc[regression["term"].eq("entity_win")].copy()
        columns = [
            "event_date_source",
            "impact_horizon_days",
            "coefficient",
            "ci_low",
            "ci_high",
            "p_value",
        ]
        print(rows[columns].round(6).to_string(index=False))

    metadata = results.get("regression_metadata", pd.DataFrame())
    if isinstance(metadata, pd.DataFrame) and "error" in metadata.columns:
        errors = metadata.loc[metadata["error"].notna()]
        if not errors.empty:
            print("\nRegression notes:")
            print(
                errors[
                    ["event_date_source", "impact_horizon_days", "error"]
                ].to_string(index=False)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pharmaceutical litigation outcomes and stock effects."
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="Filename or path to cleaned case data.",
    )
    parser.add_argument(
        "--stocks",
        required=True,
        help="Filename or path to all stock-price data.",
    )
    parser.add_argument(
        "--output-dir",
        default="litigation_analysis_output",
        help="Directory for tables and figures.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="Bootstrap repetitions for case-factor coefficient intervals.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Randomization repetitions for win/loss CAR tests.",
    )
    parser.add_argument(
        "--event-dates",
        nargs="+",
        choices=["date_filed", "date_closed"],
        default=["date_filed", "date_closed"],
        help=(
            "Legal dates used as event days. The default analyzes both filing "
            "and closure dates."
        ),
    )
    parser.add_argument(
        "--event-date",
        choices=["date_filed", "date_closed"],
        default=None,
        help=(
            "Backward-compatible single-date option. When supplied, it "
            "overrides --event-dates."
        ),
    )
    parser.add_argument(
        "--impact-horizons",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help=(
            "Post-event trading-day horizons. A horizon H sums abnormal "
            "returns from day 0 through day H-1."
        ),
    )
    parser.add_argument(
        "--include-overlapping",
        action="store_true",
        help="Include events with another same-ticker legal event within the buffer.",
    )
    parser.add_argument(
        "--overlap-buffer-days",
        type=int,
        default=20,
        help="Calendar-day radius used to flag nearby same-ticker legal events.",
    )
    parser.add_argument(
        "--min-estimation-obs",
        type=int,
        default=60,
        help="Minimum pre-event observations needed for the market model.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Litigation analysis module: {ANALYSIS_VERSION}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_dataframe(args.cases)
    stocks = load_dataframe(args.stocks)

    goal1 = analyze_case_outcome_factors(
        cases,
        n_bootstrap=args.bootstrap,
    )
    save_goal1_tables(goal1, output_dir)
    plot_case_outcome_factors(
        goal1,
        model_name="full",
        output_dir=output_dir,
        show=args.show,
    )
    print_goal1_summary(goal1)

    selected_event_dates = (
        [args.event_date] if args.event_date is not None else args.event_dates
    )
    goal2 = analyze_stock_price_impact(
        cases,
        stocks,
        event_date_cols=selected_event_dates,
        impact_horizons=args.impact_horizons,
        min_estimation_obs=args.min_estimation_obs,
        overlap_buffer_days=args.overlap_buffer_days,
        exclude_overlapping=not args.include_overlapping,
        n_permutations=args.permutations,
    )
    save_goal2_tables(goal2, output_dir)
    plot_stock_price_impact(
        goal2,
        output_dir=output_dir,
        show=args.show,
    )
    print_goal2_summary(goal2)

    metadata = {
        "goal1": goal1["metadata"],
        "goal2": goal2["metadata"],
        "goal2_regression": goal2["regression_metadata"].to_dict(orient="records"),
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(metadata), handle, indent=2)

    print(f"\nSaved all tables and figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
