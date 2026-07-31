
"""
litigation_stock_analysis.py

Reusable analysis tools for two related questions:

1. Which observable case factors are associated with whether the focal
   pharmaceutical entity wins its case?
2. Does winning or losing a case affect the focal entity's stock price?

The module uses:
- Regularized logistic regression, cross-validation, bootstrap coefficient
  intervals, and permutation importance for case outcomes.
- A leave-one-stock-out market-model event study for stock-price effects.
- Cluster-robust regression and randomization inference for win/loss CAR gaps.

Required packages:
    pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib

Important interpretation note
-----------------------------
These functions estimate associations, not guaranteed causal effects. In
particular, case documents and case duration are only known after a case has
progressed. The outcome analysis therefore reports both an "ex_ante" model
(with filing-time information) and a "full" model that adds post-filing
variables.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Shared preparation helpers
# ---------------------------------------------------------------------------

ANALYSIS_VERSION = "2026-07-22-dual-date-v2"


_REQUIRED_CASE_COLUMNS = {
    "case_row_id",
    "entity_name",
    "case_number",
    "party_type",
    "district_id",
    "date_filed",
    "date_closed",
    "plaintiffs_attorneys_num",
    "defendants_attorneys_num",
    "case_docs_num",
    "jury_demand",
    "ticker",
    "outcome",
}

_REQUIRED_STOCK_COLUMNS = {
    "Date",
    "Ticker",
    "Close",
}


def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalise_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )


def _derive_entity_result(case_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the legal-side outcome into the focal entity's perspective.

    Examples
    --------
    party_type == "Defendant" and outcome == "defendant" -> win
    party_type == "Defendant" and outcome == "plaintiff" -> loss
    outcome == "both" -> mixed
    missing/unknown/unrecognised combinations -> unknown
    """
    df = case_df.copy()
    party = _normalise_text(df["party_type"])
    outcome = _normalise_text(df["outcome"])

    is_plaintiff = party.str.contains("plaintiff", na=False) & ~party.str.contains(
        "defendant", na=False
    )
    is_defendant = party.str.contains("defendant", na=False) & ~party.str.contains(
        "plaintiff", na=False
    )

    entity_result = pd.Series("unknown", index=df.index, dtype="string")
    entity_result.loc[outcome.eq("both")] = "mixed"

    entity_result.loc[is_plaintiff & outcome.eq("plaintiff")] = "win"
    entity_result.loc[is_plaintiff & outcome.eq("defendant")] = "loss"
    entity_result.loc[is_defendant & outcome.eq("defendant")] = "win"
    entity_result.loc[is_defendant & outcome.eq("plaintiff")] = "loss"

    entity_win = pd.Series(np.nan, index=df.index, dtype="float64")
    entity_win.loc[entity_result.eq("win")] = 1.0
    entity_win.loc[entity_result.eq("loss")] = 0.0

    df["entity_result"] = entity_result
    df["entity_win"] = entity_win
    return df


def _prepare_case_features(
    cleaned_dataframe: pd.DataFrame,
    min_district_cases: int = 10,
) -> pd.DataFrame:
    _validate_columns(cleaned_dataframe, _REQUIRED_CASE_COLUMNS, "cleaned_dataframe")
    df = _derive_entity_result(cleaned_dataframe)

    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce")
    df["date_closed"] = pd.to_datetime(df["date_closed"], errors="coerce")
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    df["party_type"] = df["party_type"].astype("string").str.strip().str.title()
    df["district_id"] = df["district_id"].astype("string").str.strip().str.lower()

    numeric_source = [
        "plaintiffs_attorneys_num",
        "defendants_attorneys_num",
        "case_docs_num",
    ]
    for col in numeric_source:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["case_duration_days"] = (df["date_closed"] - df["date_filed"]).dt.days
    df.loc[df["case_duration_days"] < 0, "case_duration_days"] = np.nan
    df["filing_year"] = df["date_filed"].dt.year
    df["filing_month"] = df["date_filed"].dt.month.astype("Int64").astype("string")

    party_lower = _normalise_text(df["party_type"])
    plaintiff_side = party_lower.str.contains(
        "plaintiff", na=False
    ) & ~party_lower.str.contains("defendant", na=False)
    defendant_side = party_lower.str.contains(
        "defendant", na=False
    ) & ~party_lower.str.contains("plaintiff", na=False)

    df["entity_attorneys_num"] = np.where(
        plaintiff_side,
        df["plaintiffs_attorneys_num"],
        np.where(defendant_side, df["defendants_attorneys_num"], np.nan),
    )
    df["opponent_attorneys_num"] = np.where(
        plaintiff_side,
        df["defendants_attorneys_num"],
        np.where(defendant_side, df["plaintiffs_attorneys_num"], np.nan),
    )
    df["attorney_advantage"] = (
        df["entity_attorneys_num"] - df["opponent_attorneys_num"]
    )

    # Log transforms reduce the leverage of a few unusually large cases.
    for source, target in [
        ("plaintiffs_attorneys_num", "log1p_plaintiffs_attorneys"),
        ("defendants_attorneys_num", "log1p_defendants_attorneys"),
        ("entity_attorneys_num", "log1p_entity_attorneys"),
        ("opponent_attorneys_num", "log1p_opponent_attorneys"),
        ("case_docs_num", "log1p_case_docs"),
        ("case_duration_days", "log1p_case_duration"),
    ]:
        clipped = pd.to_numeric(df[source], errors="coerce").clip(lower=0)
        df[target] = np.log1p(clipped)

    jury = _normalise_text(df["jury_demand"])
    jury = jury.fillna("missing")
    # Keep the original information while making common forms consistent.
    jury = jury.replace(
        {
            "y": "yes",
            "n": "no",
            "true": "yes",
            "false": "no",
            "1": "yes",
            "0": "no",
        }
    )
    df["jury_demand_clean"] = jury.astype("string")

    district_counts = df["district_id"].value_counts(dropna=False)
    frequent_districts = set(
        district_counts[district_counts >= max(1, min_district_cases)].index
    )
    df["district_grouped"] = df["district_id"].where(
        df["district_id"].isin(frequent_districts), "other"
    )
    df["district_grouped"] = df["district_grouped"].fillna("missing")

    return df


def _make_one_hot_encoder() -> OneHotEncoder:
    # sparse_output replaced sparse in newer scikit-learn versions.
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=True,
        )
    except TypeError:  # pragma: no cover - for older sklearn
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=True,
        )


def _make_sklearn_safe_matrix(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Return a model matrix without pandas ``pd.NA`` scalar values.

    Some scikit-learn transformers compare object arrays with ``X != X`` to
    locate missing values. That operation is ambiguous for pandas' nullable
    ``pd.NA`` scalar. Numeric columns are therefore converted to ordinary
    float arrays with ``np.nan`` and categorical columns use an explicit
    ``"missing"`` level.
    """
    features = list(numeric_features) + list(categorical_features)
    X = df.loc[:, features].copy()

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

    for col in categorical_features:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("missing")
            .replace({"<NA>": "missing", "": "missing"})
            .astype(str)
        )

    return X


def _build_logistic_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _clean_feature_names(names: Sequence[str]) -> List[str]:
    cleaned = []
    for name in names:
        name = str(name)
        name = name.replace("num__", "").replace("cat__", "")
        name = name.replace("missingindicator_", "missing:")
        cleaned.append(name)
    return cleaned


def _bootstrap_logistic_effects(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Bootstrap regularized log-odds coefficients after fixing the fitted
    preprocessing map. Numeric coefficients therefore describe a one-standard-
    deviation change; categorical coefficients compare with the omitted level.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    base_model = fitted_pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X)

    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())
    base_coef = base_model.coef_.ravel()

    rng = np.random.default_rng(random_state)
    samples: List[np.ndarray] = []
    n = len(y)
    y_array = np.asarray(y, dtype=int)

    for _ in range(max(0, int(n_bootstrap))):
        idx = rng.integers(0, n, size=n)
        if np.unique(y_array[idx]).size < 2:
            continue
        model = clone(base_model)
        try:
            model.fit(X_transformed[idx], y_array[idx])
            samples.append(model.coef_.ravel())
        except Exception:
            continue

    if samples:
        boot = np.vstack(samples)
        lower = np.nanpercentile(boot, 2.5, axis=0)
        upper = np.nanpercentile(boot, 97.5, axis=0)
        probability_positive = (boot > 0).mean(axis=0)
    else:
        lower = np.full_like(base_coef, np.nan, dtype=float)
        upper = np.full_like(base_coef, np.nan, dtype=float)
        probability_positive = np.full_like(base_coef, np.nan, dtype=float)

    effects = pd.DataFrame(
        {
            "feature": feature_names,
            "log_odds_coefficient": base_coef,
            "odds_ratio": np.exp(np.clip(base_coef, -20, 20)),
            "ci_low_log_odds": lower,
            "ci_high_log_odds": upper,
            "ci_low_odds_ratio": np.exp(np.clip(lower, -20, 20)),
            "ci_high_odds_ratio": np.exp(np.clip(upper, -20, 20)),
            "bootstrap_probability_positive": probability_positive,
        }
    )
    effects["absolute_log_odds"] = effects["log_odds_coefficient"].abs()
    effects["direction"] = np.where(
        effects["log_odds_coefficient"] >= 0,
        "higher win odds",
        "lower win odds",
    )
    return effects.sort_values(
        "absolute_log_odds",
        ascending=False,
    ).reset_index(drop=True)


def _evaluate_logistic_model(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    n_splits: int,
    random_state: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    class_counts = y.value_counts()
    feasible_splits = int(min(n_splits, class_counts.min()))
    if feasible_splits < 2:
        raise ValueError(
            "At least two wins and two losses are required for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=feasible_splits,
        shuffle=True,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*during transform",
            category=UserWarning,
        )
        probabilities = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "n_observations": float(len(y)),
        "n_wins": float(y.sum()),
        "n_losses": float((1 - y).sum()),
        "cv_folds": float(feasible_splits),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }
    return metrics, probabilities


def _raw_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    random_state: int,
    n_repeats: int = 20,
) -> pd.DataFrame:
    if len(X) < 20 or y.value_counts().min() < 3:
        return pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std"]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )
    if y_test.nunique() < 2:
        return pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std"]
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*during transform",
            category=UserWarning,
        )
        held_out_model = clone(pipeline).fit(X_train, y_train)
        perm = permutation_importance(
            held_out_model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=None,
        )
    result = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    )
    return result.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Goal 1
# ---------------------------------------------------------------------------

def analyze_case_outcome_factors(
    cleaned_dataframe: pd.DataFrame,
    *,
    min_district_cases: int = 10,
    n_splits: int = 5,
    n_bootstrap: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Test which factors are associated with the focal entity winning a case.

    The outcome is converted from legal side to entity perspective using both
    `outcome` and `party_type`. Only clear wins and losses enter the binary
    models; "both", missing, and unknown outcomes are reported but excluded.

    Two complementary models are fitted:

    ex_ante
        Uses information available at or near filing. This is the safer model
        for prediction and avoids obvious post-outcome leakage.

    full
        Adds case document volume and duration. This is useful descriptively,
        but the extra variables should be interpreted as associations rather
        than pre-case causal drivers.

    Returns
    -------
    dict
        Keys include `models`, `metrics`, `feature_effects`,
        `permutation_importance`, `modeling_data`, `all_prepared_cases`,
        and `exclusion_summary`.
    """
    prepared = _prepare_case_features(
        cleaned_dataframe,
        min_district_cases=min_district_cases,
    )

    exclusion_summary = (
        prepared["entity_result"]
        .value_counts(dropna=False)
        .rename_axis("entity_result")
        .reset_index(name="rows")
    )

    model_df = prepared.loc[prepared["entity_win"].notna()].copy()
    model_df["entity_win"] = model_df["entity_win"].astype(int)

    if model_df["entity_win"].nunique() < 2:
        raise ValueError(
            "The cleaned case data must contain at least one win and one loss."
        )

    ex_ante_numeric = [
        "log1p_entity_attorneys",
        "log1p_opponent_attorneys",
        "attorney_advantage",
        "filing_year",
    ]
    ex_ante_categorical = [
        "party_type",
        "district_grouped",
        "jury_demand_clean",
        "filing_month",
    ]

    full_numeric = ex_ante_numeric + [
        "log1p_case_docs",
        "log1p_case_duration",
    ]
    full_categorical = ex_ante_categorical

    specifications = {
        "ex_ante": (ex_ante_numeric, ex_ante_categorical),
        "full": (full_numeric, full_categorical),
    }

    models: Dict[str, Pipeline] = {}
    metrics_rows: List[Dict[str, Any]] = []
    feature_effects: Dict[str, pd.DataFrame] = {}
    raw_importances: Dict[str, pd.DataFrame] = {}
    cv_predictions = model_df[
        [
            "case_row_id",
            "case_number",
            "entity_name",
            "ticker",
            "entity_result",
            "entity_win",
        ]
    ].copy()

    for model_name, (numeric_features, categorical_features) in specifications.items():
        X = _make_sklearn_safe_matrix(
            model_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        y = model_df["entity_win"].astype(int)

        pipeline = _build_logistic_pipeline(
            numeric_features,
            categorical_features,
            random_state=random_state,
        )
        model_metrics, probabilities = _evaluate_logistic_model(
            X,
            y,
            pipeline,
            n_splits=n_splits,
            random_state=random_state,
        )
        model_metrics["model"] = model_name
        metrics_rows.append(model_metrics)
        cv_predictions[f"{model_name}_cv_win_probability"] = probabilities

        fitted = pipeline.fit(X, y)
        models[model_name] = fitted
        feature_effects[model_name] = _bootstrap_logistic_effects(
            fitted,
            X,
            y,
            n_bootstrap=n_bootstrap,
            random_state=random_state + (0 if model_name == "ex_ante" else 1000),
        )
        raw_importances[model_name] = _raw_permutation_importance(
            X,
            y,
            pipeline,
            random_state=random_state + (0 if model_name == "ex_ante" else 1000),
        )

    metrics = (
        pd.DataFrame(metrics_rows)
        .set_index("model")
        .sort_index()
    )

    # Empirical descriptive tables are useful for sanity checking the model.
    party_win_rates = (
        model_df.groupby("party_type", dropna=False)
        .agg(cases=("entity_win", "size"), win_rate=("entity_win", "mean"))
        .sort_values("cases", ascending=False)
        .reset_index()
    )
    district_win_rates = (
        model_df.groupby("district_grouped", dropna=False)
        .agg(cases=("entity_win", "size"), win_rate=("entity_win", "mean"))
        .sort_values("cases", ascending=False)
        .reset_index()
    )

    return {
        "models": models,
        "metrics": metrics,
        "feature_effects": feature_effects,
        "permutation_importance": raw_importances,
        "cv_predictions": cv_predictions,
        "modeling_data": model_df,
        "all_prepared_cases": prepared,
        "exclusion_summary": exclusion_summary,
        "party_win_rates": party_win_rates,
        "district_win_rates": district_win_rates,
        "metadata": {
            "target_definition": (
                "entity_win=1 when outcome matches the entity's party_type; "
                "entity_win=0 when the opposing side won"
            ),
            "excluded_outcomes": ["mixed", "unknown"],
            "random_state": random_state,
            "n_bootstrap": n_bootstrap,
            "min_district_cases": min_district_cases,
        },
    }


# ---------------------------------------------------------------------------
# Goal 2
# ---------------------------------------------------------------------------

def _prepare_stock_returns(all_stock_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(all_stock_data, _REQUIRED_STOCK_COLUMNS, "all_stock_data")
    stock = all_stock_data.copy()
    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")
    stock["Ticker"] = stock["Ticker"].astype("string").str.strip().str.upper()
    stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")

    stock = stock.dropna(subset=["Date", "Ticker", "Close"])
    stock = stock.loc[stock["Close"] > 0]
    stock = (
        stock.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )

    stock["firm_return"] = stock.groupby("Ticker", observed=True)["Close"].transform(
        lambda s: np.log(s).diff()
    )

    daily = (
        stock.groupby("Date", observed=True)["firm_return"]
        .agg(return_sum="sum", return_count="count", market_return="mean")
        .reset_index()
    )
    stock = stock.merge(daily, on="Date", how="left")

    denominator = stock["return_count"] - 1
    stock["benchmark_return"] = np.where(
        denominator >= 2,
        (stock["return_sum"] - stock["firm_return"]) / denominator,
        stock["market_return"],
    )
    stock.replace([np.inf, -np.inf], np.nan, inplace=True)
    return stock


def _map_events_to_trading_dates(
    prepared_cases: pd.DataFrame,
    stock: pd.DataFrame,
    event_date_col: str,
) -> pd.DataFrame:
    events = prepared_cases.copy()
    if event_date_col not in events.columns:
        raise ValueError(f"Unknown event date column: {event_date_col}")

    events["raw_event_date"] = pd.to_datetime(events[event_date_col], errors="coerce")
    events = events.dropna(subset=["raw_event_date", "ticker"]).copy()
    events["ticker"] = events["ticker"].astype("string").str.strip().str.upper()

    trading_dates_by_ticker = {
        ticker: group["Date"].sort_values().drop_duplicates().to_numpy()
        for ticker, group in stock.groupby("Ticker", observed=True)
    }

    mapped_dates = []
    for row in events[["ticker", "raw_event_date"]].itertuples(index=False):
        ticker = row.ticker
        raw_date = np.datetime64(row.raw_event_date.to_datetime64())
        dates = trading_dates_by_ticker.get(ticker)
        if dates is None or len(dates) == 0:
            mapped_dates.append(pd.NaT)
            continue
        position = int(np.searchsorted(dates, raw_date, side="left"))
        mapped_dates.append(
            pd.Timestamp(dates[position]) if position < len(dates) else pd.NaT
        )

    events["event_date"] = mapped_dates
    events["event_date_source"] = event_date_col
    events = events.dropna(subset=["event_date"]).copy()
    return events


def _collapse_same_day_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple cases for one ticker on the same trading day so the same
    stock return is not counted repeatedly. Conflicting same-day outcomes are
    labelled mixed.
    """
    records: List[Dict[str, Any]] = []
    group_cols = ["ticker", "event_date"]

    for (ticker, event_date), group in events.groupby(group_cols, observed=True):
        binary = group["entity_win"].dropna().unique()
        if len(binary) == 1:
            entity_win = float(binary[0])
            entity_result = "win" if entity_win == 1 else "loss"
        elif len(binary) == 0:
            entity_win = np.nan
            known = set(group["entity_result"].dropna().astype(str))
            entity_result = "mixed" if "mixed" in known else "unknown"
        else:
            entity_win = np.nan
            entity_result = "mixed"

        records.append(
            {
                "ticker": ticker,
                "event_date": pd.Timestamp(event_date),
                "event_date_source": (
                    group["event_date_source"].iloc[0]
                    if "event_date_source" in group.columns
                    else "event_date"
                ),
                "raw_event_date_min": group["raw_event_date"].min(),
                "raw_event_date_max": group["raw_event_date"].max(),
                "entity_win": entity_win,
                "entity_result": entity_result,
                "case_count": int(len(group)),
                "case_numbers": " | ".join(
                    sorted(group["case_number"].dropna().astype(str).unique())
                ),
                "party_type": (
                    group["party_type"].dropna().astype(str).mode().iloc[0]
                    if not group["party_type"].dropna().empty
                    else pd.NA
                ),
                "district_id": (
                    group["district_id"].dropna().astype(str).mode().iloc[0]
                    if not group["district_id"].dropna().empty
                    else pd.NA
                ),
                "attorney_advantage": group["attorney_advantage"].mean(),
                "log1p_case_docs": group["log1p_case_docs"].mean(),
                "log1p_case_duration": group["log1p_case_duration"].mean(),
            }
        )
    return pd.DataFrame(records)


def _add_overlap_flags(
    event_table: pd.DataFrame,
    overlap_buffer_days: int,
) -> pd.DataFrame:
    events = event_table.sort_values(["ticker", "event_date"]).copy()
    overlap_counts = pd.Series(1, index=events.index, dtype=int)

    for _, idx in events.groupby("ticker", observed=True).groups.items():
        locs = list(idx)
        dates = events.loc[locs, "event_date"].to_numpy(dtype="datetime64[D]")
        for position, row_idx in enumerate(locs):
            distance = np.abs(
                (dates - dates[position]).astype("timedelta64[D]").astype(int)
            )
            overlap_counts.loc[row_idx] = int((distance <= overlap_buffer_days).sum())

    events["nearby_event_count"] = overlap_counts
    events["clean_event"] = events["nearby_event_count"].eq(1)
    return events


def _estimate_one_event(
    event: pd.Series,
    ticker_data: pd.DataFrame,
    estimation_window: Tuple[int, int],
    event_window: Tuple[int, int],
    min_estimation_obs: int,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)
    matches = np.flatnonzero(ticker_data["Date"].eq(event["event_date"]).to_numpy())
    if len(matches) == 0:
        return None, None
    event_position = int(matches[0])
    relative_day = np.arange(len(ticker_data)) - event_position
    ticker_data = ticker_data.assign(relative_day=relative_day)

    est_mask = (
        ticker_data["relative_day"].between(estimation_window[0], estimation_window[1])
        & ticker_data["firm_return"].notna()
        & ticker_data["benchmark_return"].notna()
    )
    estimation = ticker_data.loc[est_mask]
    if len(estimation) < min_estimation_obs:
        return None, None

    design = np.column_stack(
        [
            np.ones(len(estimation)),
            estimation["benchmark_return"].to_numpy(dtype=float),
        ]
    )
    response = estimation["firm_return"].to_numpy(dtype=float)
    alpha, beta = np.linalg.lstsq(design, response, rcond=None)[0]

    event_slice = ticker_data.loc[
        ticker_data["relative_day"].between(event_window[0], event_window[1])
    ].copy()
    if event_slice.empty:
        return None, None

    event_slice["expected_return"] = (
        alpha + beta * event_slice["benchmark_return"]
    )
    event_slice["abnormal_return"] = (
        event_slice["firm_return"] - event_slice["expected_return"]
    )
    event_slice["cumulative_abnormal_return"] = event_slice[
        "abnormal_return"
    ].fillna(0).cumsum()

    event_date_source = str(event.get("event_date_source", "event_date"))
    event_id = (
        f"{event_date_source}|{event['ticker']}|"
        f"{pd.Timestamp(event['event_date']).date()}"
    )
    event_slice["event_id"] = event_id
    for col in [
        "ticker",
        "event_date",
        "event_date_source",
        "entity_win",
        "entity_result",
        "case_count",
        "case_numbers",
        "party_type",
        "district_id",
        "attorney_advantage",
        "log1p_case_docs",
        "log1p_case_duration",
        "nearby_event_count",
        "clean_event",
    ]:
        event_slice[col] = event.get(col, np.nan)

    metadata = event.to_dict()
    metadata.update(
        {
            "event_id": event_id,
            "alpha": float(alpha),
            "beta": float(beta),
            "estimation_observations": int(len(estimation)),
        }
    )
    return event_slice, metadata


def _car_column_name(window: Tuple[int, int]) -> str:
    return f"CAR_{window[0]:+d}_{window[1]:+d}"


def _one_sample_summary(values: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    if len(values) == 1 or values.std(ddof=1) == 0:
        t_stat, p_value = np.nan, np.nan
    else:
        t_stat, p_value = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
    }


def _win_loss_tests(
    event_level: pd.DataFrame,
    car_columns: Sequence[str],
    n_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(random_state)

    binary = event_level.loc[event_level["entity_win"].isin([0.0, 1.0])].copy()
    for car_col in car_columns:
        wins = binary.loc[binary["entity_win"].eq(1), car_col].dropna().to_numpy()
        losses = binary.loc[binary["entity_win"].eq(0), car_col].dropna().to_numpy()

        observed = np.nan
        welch_t = np.nan
        welch_p = np.nan
        permutation_p = np.nan
        ci_low = np.nan
        ci_high = np.nan

        if len(wins) > 0 and len(losses) > 0:
            observed = float(np.mean(wins) - np.mean(losses))
            if len(wins) > 1 and len(losses) > 1:
                welch_t, welch_p = stats.ttest_ind(
                    wins,
                    losses,
                    equal_var=False,
                    nan_policy="omit",
                )
                welch_t = float(welch_t)
                welch_p = float(welch_p)

            combined = np.concatenate([wins, losses])
            n_wins = len(wins)
            perm_diffs = []
            for _ in range(max(0, int(n_permutations))):
                shuffled = rng.permutation(combined)
                perm_diffs.append(
                    shuffled[:n_wins].mean() - shuffled[n_wins:].mean()
                )
            if perm_diffs:
                perm_array = np.asarray(perm_diffs)
                permutation_p = float(
                    (np.sum(np.abs(perm_array) >= abs(observed)) + 1)
                    / (len(perm_array) + 1)
                )

            # Cluster bootstrap at ticker level preserves within-company
            # dependence across repeated legal events.
            tickers = binary["ticker"].dropna().unique()
            boot_diffs = []
            if len(tickers) >= 2:
                for _ in range(max(200, min(2000, int(n_permutations)))):
                    sampled = rng.choice(tickers, size=len(tickers), replace=True)
                    pieces = []
                    for draw_id, ticker in enumerate(sampled):
                        piece = binary.loc[binary["ticker"].eq(ticker)].copy()
                        piece["_draw_id"] = draw_id
                        pieces.append(piece)
                    boot = pd.concat(pieces, ignore_index=True)
                    bw = boot.loc[boot["entity_win"].eq(1), car_col].dropna()
                    bl = boot.loc[boot["entity_win"].eq(0), car_col].dropna()
                    if len(bw) and len(bl):
                        boot_diffs.append(float(bw.mean() - bl.mean()))
            if boot_diffs:
                ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

        rows.append(
            {
                "car_window": car_col,
                "n_wins": int(len(wins)),
                "n_losses": int(len(losses)),
                "mean_win_minus_loss": observed,
                "cluster_bootstrap_ci_low": (
                    float(ci_low) if np.isfinite(ci_low) else np.nan
                ),
                "cluster_bootstrap_ci_high": (
                    float(ci_high) if np.isfinite(ci_high) else np.nan
                ),
                "welch_t_stat": welch_t,
                "welch_p_value": welch_p,
                "permutation_p_value": permutation_p,
            }
        )
    return pd.DataFrame(rows)


def _fit_car_regression(
    event_level: pd.DataFrame,
    primary_car_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Regress CAR on entity win and case controls. Standard errors are clustered
    by ticker when possible; HC3 is used as a fallback.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        return (
            pd.DataFrame(),
            {"error": f"statsmodels is required for CAR regression: {exc}"},
        )

    columns = [
        primary_car_col,
        "entity_win",
        "attorney_advantage",
        "log1p_case_docs",
        "log1p_case_duration",
        "party_type",
        "ticker",
    ]
    regression_df = event_level[columns].copy()
    regression_df = regression_df.loc[
        regression_df["entity_win"].isin([0.0, 1.0])
    ]

    numeric_controls = [
        "entity_win",
        "attorney_advantage",
        "log1p_case_docs",
        "log1p_case_duration",
    ]
    for col in numeric_controls + [primary_car_col]:
        regression_df[col] = pd.to_numeric(regression_df[col], errors="coerce")

    # Median-impute controls, then standardize non-binary controls.
    for col in numeric_controls[1:]:
        median = regression_df[col].median()
        regression_df[col] = regression_df[col].fillna(median)
        std = regression_df[col].std(ddof=0)
        if np.isfinite(std) and std > 0:
            regression_df[col] = (regression_df[col] - regression_df[col].mean()) / std
        else:
            regression_df[col] = 0.0

    regression_df["party_type"] = regression_df["party_type"].fillna("Missing")
    design = pd.get_dummies(
        regression_df[numeric_controls + ["party_type"]],
        columns=["party_type"],
        drop_first=True,
        dtype=float,
    )
    design = sm.add_constant(design, has_constant="add")
    response = regression_df[primary_car_col]

    valid = response.notna() & design.notna().all(axis=1)
    design = design.loc[valid].astype(float)
    response = response.loc[valid].astype(float)
    groups = regression_df.loc[valid, "ticker"].astype(str)

    if len(response) <= design.shape[1] + 2:
        return (
            pd.DataFrame(),
            {
                "error": "Too few complete events for the requested CAR regression.",
                "n_observations": int(len(response)),
                "n_parameters": int(design.shape[1]),
            },
        )

    fitted = sm.OLS(response, design).fit()
    covariance = "HC3"
    if groups.nunique() >= 2:
        try:
            fitted = fitted.get_robustcov_results(
                cov_type="cluster",
                groups=groups,
            )
            covariance = "clustered by ticker"
        except Exception:
            fitted = fitted.get_robustcov_results(cov_type="HC3")
    else:
        fitted = fitted.get_robustcov_results(cov_type="HC3")

    names = list(design.columns)
    confidence = np.asarray(fitted.conf_int())
    table = pd.DataFrame(
        {
            "term": names,
            "coefficient": np.asarray(fitted.params),
            "std_error": np.asarray(fitted.bse),
            "t_stat": np.asarray(fitted.tvalues),
            "p_value": np.asarray(fitted.pvalues),
            "ci_low": confidence[:, 0],
            "ci_high": confidence[:, 1],
        }
    )

    metadata = {
        "dependent_variable": primary_car_col,
        "n_observations": int(fitted.nobs),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "covariance": covariance,
    }
    return table, metadata


def _post_event_car_column(horizon_days: int) -> str:
    return f"CAR_{int(horizon_days)}D"


def _analyze_single_event_date(
    prepared_cases: pd.DataFrame,
    stock: pd.DataFrame,
    *,
    event_date_col: str,
    estimation_window: Tuple[int, int],
    event_window: Tuple[int, int],
    impact_horizons: Sequence[int],
    min_estimation_obs: int,
    overlap_buffer_days: int,
    exclude_overlapping: bool,
    n_permutations: int,
    random_state: int,
) -> Dict[str, Any]:
    mapped = _map_events_to_trading_dates(
        prepared_cases,
        stock,
        event_date_col=event_date_col,
    )
    collapsed = _collapse_same_day_events(mapped)
    collapsed = _add_overlap_flags(collapsed, overlap_buffer_days)

    stock_groups = {
        ticker: group.copy()
        for ticker, group in stock.groupby("Ticker", observed=True)
    }

    daily_results: List[pd.DataFrame] = []
    event_metadata: List[Dict[str, Any]] = []
    failed_events: List[Dict[str, Any]] = []

    for _, event in collapsed.iterrows():
        ticker_data = stock_groups.get(event["ticker"])
        if ticker_data is None:
            failed_events.append(
                {
                    "event_date_source": event_date_col,
                    "ticker": event["ticker"],
                    "event_date": event["event_date"],
                    "reason": "ticker not found in stock data",
                }
            )
            continue

        daily, metadata = _estimate_one_event(
            event,
            ticker_data,
            estimation_window=estimation_window,
            event_window=event_window,
            min_estimation_obs=min_estimation_obs,
        )
        if daily is None or metadata is None:
            failed_events.append(
                {
                    "event_date_source": event_date_col,
                    "ticker": event["ticker"],
                    "event_date": event["event_date"],
                    "reason": "insufficient estimation or event-window data",
                }
            )
            continue
        daily_results.append(daily)
        event_metadata.append(metadata)

    if not daily_results:
        raise ValueError(
            f"No usable stock-price events were created for {event_date_col}. "
            "Check ticker coverage, event dates, and min_estimation_obs."
        )

    daily_abnormal_returns = pd.concat(daily_results, ignore_index=True)
    event_level = pd.DataFrame(event_metadata)
    event_level["event_date_source"] = event_date_col

    # A horizon of H means exactly H post-event trading returns: day 0 through
    # day H-1. Incomplete windows are deliberately reported as missing rather
    # than being summed over fewer observations.
    car_columns: List[str] = []
    for horizon in impact_horizons:
        car_col = _post_event_car_column(horizon)
        car_columns.append(car_col)
        selected = daily_abnormal_returns.loc[
            daily_abnormal_returns["relative_day"].between(0, horizon - 1)
        ]
        aggregation = selected.groupby("event_id", observed=True)[
            "abnormal_return"
        ].agg(["sum", "count"])
        complete_car = aggregation["sum"].where(aggregation["count"].eq(horizon))
        event_level[car_col] = event_level["event_id"].map(complete_car)
        event_level[f"{car_col}_observations"] = event_level["event_id"].map(
            aggregation["count"]
        )

    analysis_events = event_level.copy()
    if exclude_overlapping:
        analysis_events = analysis_events.loc[analysis_events["clean_event"]].copy()

    summary_rows: List[Dict[str, Any]] = []
    for result_name, group in analysis_events.groupby("entity_result", dropna=False):
        for horizon, car_col in zip(impact_horizons, car_columns):
            row = {
                "event_date_source": event_date_col,
                "entity_result": result_name,
                "impact_horizon_days": int(horizon),
                "car_window": car_col,
            }
            row.update(_one_sample_summary(group[car_col]))
            summary_rows.append(row)
    group_summary = pd.DataFrame(summary_rows)

    win_loss_tests = _win_loss_tests(
        analysis_events,
        car_columns,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    win_loss_tests.insert(0, "event_date_source", event_date_col)
    horizon_map = {
        _post_event_car_column(h): int(h) for h in impact_horizons
    }
    win_loss_tests.insert(
        1,
        "impact_horizon_days",
        win_loss_tests["car_window"].map(horizon_map),
    )

    regression_tables: List[pd.DataFrame] = []
    regression_metadata_rows: List[Dict[str, Any]] = []
    for horizon, car_col in zip(impact_horizons, car_columns):
        table, metadata = _fit_car_regression(
            analysis_events,
            primary_car_col=car_col,
        )
        if not table.empty:
            table.insert(0, "event_date_source", event_date_col)
            table.insert(1, "impact_horizon_days", int(horizon))
            table.insert(2, "car_window", car_col)
            regression_tables.append(table)
        regression_metadata_rows.append(
            {
                "event_date_source": event_date_col,
                "impact_horizon_days": int(horizon),
                "car_window": car_col,
                **metadata,
            }
        )

    regression_table = (
        pd.concat(regression_tables, ignore_index=True)
        if regression_tables
        else pd.DataFrame()
    )
    regression_metadata = pd.DataFrame(regression_metadata_rows)

    return {
        "event_level": event_level,
        "analysis_events": analysis_events,
        "daily_abnormal_returns": daily_abnormal_returns,
        "group_summary": group_summary,
        "win_loss_tests": win_loss_tests,
        "regression_table": regression_table,
        "regression_metadata": regression_metadata,
        "failed_events": pd.DataFrame(failed_events),
        "metadata": {
            "event_date_col": event_date_col,
            "event_date_rule": "first trading date on or after legal event date",
            "estimation_window": list(estimation_window),
            "event_window": list(event_window),
            "impact_horizons": [int(h) for h in impact_horizons],
            "car_columns": car_columns,
            "benchmark": "leave-one-stock-out equal-weight daily market return",
            "return_type": "log return based on Close",
            "overlap_buffer_days": overlap_buffer_days,
            "exclude_overlapping_from_primary_tests": exclude_overlapping,
            "usable_events_all": int(len(event_level)),
            "usable_events_primary": int(len(analysis_events)),
            "n_permutations": int(n_permutations),
            "random_state": int(random_state),
        },
    }


def analyze_stock_price_impact(
    cleaned_dataframe: pd.DataFrame,
    all_stock_data: pd.DataFrame,
    *,
    event_date_cols: Sequence[str] = ("date_filed", "date_closed"),
    impact_horizons: Sequence[int] = (1, 5, 10),
    event_date_col: Optional[str] = None,
    estimation_window: Tuple[int, int] = (-120, -21),
    event_window: Tuple[int, int] = (-10, 10),
    min_estimation_obs: int = 60,
    overlap_buffer_days: int = 20,
    exclude_overlapping: bool = True,
    min_district_cases: int = 10,
    n_permutations: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Analyze stock-price abnormalities around filing and closure dates.

    By default, this runs two separate event studies: one using ``date_filed``
    and one using ``date_closed``. For each legal date it computes complete
    post-event abnormal-return windows of 1, 5, and 10 trading days.

    ``event_date_col`` is retained for backward compatibility. When supplied,
    it overrides ``event_date_cols`` and runs a single event-date definition.
    """
    if event_date_col is not None:
        event_date_cols = (event_date_col,)

    date_cols = list(dict.fromkeys(str(col) for col in event_date_cols))
    allowed = {"date_filed", "date_closed"}
    invalid = sorted(set(date_cols) - allowed)
    if invalid:
        raise ValueError(f"event_date_cols contains unsupported values: {invalid}")
    if not date_cols:
        raise ValueError("At least one event date column is required.")

    horizons = sorted({int(h) for h in impact_horizons})
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("impact_horizons must contain positive integers.")

    required_event_end = max(horizons) - 1
    effective_event_window = (
        int(event_window[0]),
        max(int(event_window[1]), required_event_end),
    )

    prepared_cases = _prepare_case_features(
        cleaned_dataframe,
        min_district_cases=min_district_cases,
    )
    stock = _prepare_stock_returns(all_stock_data)

    by_event_date: Dict[str, Dict[str, Any]] = {}
    for offset, date_col in enumerate(date_cols):
        by_event_date[date_col] = _analyze_single_event_date(
            prepared_cases,
            stock,
            event_date_col=date_col,
            estimation_window=estimation_window,
            event_window=effective_event_window,
            impact_horizons=horizons,
            min_estimation_obs=min_estimation_obs,
            overlap_buffer_days=overlap_buffer_days,
            exclude_overlapping=exclude_overlapping,
            n_permutations=n_permutations,
            random_state=random_state + 10000 * offset,
        )

    def combine(key: str) -> pd.DataFrame:
        frames = [result[key] for result in by_event_date.values()]
        frames = [
            frame
            for frame in frames
            if isinstance(frame, pd.DataFrame) and not frame.empty
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    metadata = {
        "event_date_cols": date_cols,
        "event_date_rule": "first trading date on or after each legal date",
        "impact_horizons": horizons,
        "impact_window_definition": (
            "H-day impact equals cumulative abnormal return from trading day 0 "
            "through trading day H-1; only complete windows are retained"
        ),
        "estimation_window": list(estimation_window),
        "event_window": list(effective_event_window),
        "benchmark": "leave-one-stock-out equal-weight daily market return",
        "return_type": "log return based on Close",
        "overlap_buffer_days": overlap_buffer_days,
        "exclude_overlapping_from_primary_tests": exclude_overlapping,
        "n_permutations": int(n_permutations),
        "random_state": int(random_state),
        "per_date": {
            date_col: result["metadata"]
            for date_col, result in by_event_date.items()
        },
    }

    return {
        "by_event_date": by_event_date,
        "event_level": combine("event_level"),
        "analysis_events": combine("analysis_events"),
        "daily_abnormal_returns": combine("daily_abnormal_returns"),
        "group_summary": combine("group_summary"),
        "win_loss_tests": combine("win_loss_tests"),
        "regression_table": combine("regression_table"),
        "regression_metadata": combine("regression_metadata"),
        "failed_events": combine("failed_events"),
        "prepared_cases": prepared_cases,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Goal 1 visualisation
# ---------------------------------------------------------------------------

def plot_case_outcome_factors(
    outcome_results: Mapping[str, Any],
    *,
    model_name: str = "full",
    top_n: int = 15,
    output_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """
    Create multiple visual answers for the case-outcome question.

    Graphs
    ------
    coefficient_forest
        Largest adjusted odds ratios with bootstrap intervals.
    permutation_importance
        Held-out decrease in ROC AUC when each raw feature is shuffled.
    empirical_win_rates
        Observed win rates by party type and by the most common districts.
    """
    import matplotlib.pyplot as plt

    if model_name not in outcome_results["feature_effects"]:
        raise ValueError(
            f"model_name must be one of {list(outcome_results['feature_effects'])}"
        )

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures: Dict[str, Any] = {}

    effects = outcome_results["feature_effects"][model_name].copy()
    effects = effects.replace([np.inf, -np.inf], np.nan)
    effects = effects.dropna(subset=["log_odds_coefficient"]).head(top_n)
    effects = effects.sort_values("log_odds_coefficient")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(effects) + 1.5)))
    y = np.arange(len(effects))
    coef = effects["log_odds_coefficient"].to_numpy()
    low = effects["ci_low_log_odds"].to_numpy()
    high = effects["ci_high_log_odds"].to_numpy()
    # A regularized point estimate is not mathematically required to lie
    # inside a percentile bootstrap interval. Clip displayed widths at zero
    # so Matplotlib always receives valid non-negative errors.
    xerr = np.vstack([
        np.maximum(0.0, coef - low),
        np.maximum(0.0, high - coef),
    ])
    if not np.isfinite(xerr).all():
        xerr = None
    ax.errorbar(coef, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(effects["feature"])
    ax.set_xlabel("Adjusted log-odds effect on entity win")
    ax.set_title(f"Case outcome factors: {model_name} model")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    figures["coefficient_forest"] = fig
    if output_path is not None:
        fig.savefig(
            output_path / f"goal1_{model_name}_coefficient_forest.png",
            dpi=180,
            bbox_inches="tight",
        )

    importance = outcome_results["permutation_importance"][model_name].copy()
    importance = importance.dropna(subset=["importance_mean"]).head(top_n)
    importance = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.42 * len(importance) + 1.5)))
    ax.barh(
        importance["feature"],
        importance["importance_mean"],
        xerr=importance["importance_std"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Held-out ROC AUC decrease after shuffling")
    ax.set_title("Predictive importance of raw case factors")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    figures["permutation_importance"] = fig
    if output_path is not None:
        fig.savefig(
            output_path / f"goal1_{model_name}_permutation_importance.png",
            dpi=180,
            bbox_inches="tight",
        )

    party = outcome_results["party_win_rates"].copy()
    district = outcome_results["district_win_rates"].head(12).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(party["party_type"].astype(str), party["win_rate"])
    axes[0].axhline(
        outcome_results["modeling_data"]["entity_win"].mean(),
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Observed entity win rate")
    axes[0].set_title("Win rate by entity party type")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(district["district_grouped"].astype(str), district["win_rate"])
    axes[1].axhline(
        outcome_results["modeling_data"]["entity_win"].mean(),
        linestyle="--",
        linewidth=1,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Observed entity win rate")
    axes[1].set_title("Win rate in the most common districts")
    axes[1].tick_params(axis="x", rotation=60)

    fig.tight_layout()
    figures["empirical_win_rates"] = fig
    if output_path is not None:
        fig.savefig(
            output_path / "goal1_empirical_win_rates.png",
            dpi=180,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    return figures


# ---------------------------------------------------------------------------
# Goal 2 visualisation
# ---------------------------------------------------------------------------

def plot_stock_price_impact(
    stock_results: Mapping[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """Visualize filing-date and closure-date effects at 1, 5, and 10 days."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures: Dict[str, Any] = {}
    by_event_date = stock_results.get("by_event_date", {})

    # Direct comparison of win-minus-loss effects across dates and horizons.
    tests = stock_results["win_loss_tests"].copy()
    if not tests.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        date_labels = {
            "date_filed": "Filed",
            "date_closed": "Closed",
        }
        for date_source, group in tests.groupby("event_date_source", observed=True):
            group = group.sort_values("impact_horizon_days")
            estimate = group["mean_win_minus_loss"].to_numpy(dtype=float)
            low = group["cluster_bootstrap_ci_low"].to_numpy(dtype=float)
            high = group["cluster_bootstrap_ci_high"].to_numpy(dtype=float)
            errors = np.vstack([
                np.maximum(0.0, estimate - low),
                np.maximum(0.0, high - estimate),
            ])
            if not np.isfinite(errors).all():
                errors = None
            ax.errorbar(
                group["impact_horizon_days"],
                estimate,
                yerr=errors,
                marker="o",
                capsize=4,
                label=date_labels.get(str(date_source), str(date_source)),
            )
        ax.axhline(0, linewidth=1)
        ax.set_xticks(sorted(tests["impact_horizon_days"].dropna().unique()))
        ax.set_xlabel("Post-event trading-day horizon")
        ax.set_ylabel("Win minus loss cumulative abnormal return")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title("Win-loss stock effect by legal event date")
        ax.legend(title="Event date")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figures["win_loss_effect_by_event_date"] = fig
        if output_path is not None:
            fig.savefig(
                output_path / "goal2_win_loss_effect_by_event_date.png",
                dpi=180,
                bbox_inches="tight",
            )

    for date_source, result in by_event_date.items():
        readable_date = "case filing" if date_source == "date_filed" else "case closure"
        daily = result["daily_abnormal_returns"].copy()
        analysis_ids = set(result["analysis_events"]["event_id"])
        daily = daily.loc[daily["event_id"].isin(analysis_ids)].copy()
        daily = daily.loc[daily["entity_result"].isin(["win", "loss"])]

        # CAR path for each event date definition.
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for result_name, group in daily.groupby("entity_result", observed=True):
            pivot = group.pivot_table(
                index="relative_day",
                columns="event_id",
                values="cumulative_abnormal_return",
                aggfunc="first",
            )
            mean = pivot.mean(axis=1)
            sem = pivot.sem(axis=1)
            ax.plot(
                mean.index,
                mean.values,
                label=f"{result_name} (n={pivot.shape[1]})",
            )
            ax.fill_between(
                mean.index.to_numpy(),
                (mean - 1.96 * sem).to_numpy(),
                (mean + 1.96 * sem).to_numpy(),
                alpha=0.2,
            )
        ax.axvline(0, linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel(f"Trading days relative to {readable_date}")
        ax.set_ylabel("Mean cumulative abnormal return")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"Stock response around {readable_date}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        key = f"cumulative_abnormal_return_{date_source}"
        figures[key] = fig
        if output_path is not None:
            fig.savefig(
                output_path / f"goal2_{date_source}_cumulative_abnormal_return.png",
                dpi=180,
                bbox_inches="tight",
            )

        # Mean daily abnormal returns.
        fig, ax = plt.subplots(figsize=(10, 5.5))
        grouped = (
            daily.groupby(["relative_day", "entity_result"], observed=True)[
                "abnormal_return"
            ]
            .mean()
            .unstack("entity_result")
        )
        for result_name in grouped.columns:
            ax.plot(grouped.index, grouped[result_name], marker="o", label=result_name)
        ax.axvline(0, linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel(f"Trading days relative to {readable_date}")
        ax.set_ylabel("Mean abnormal return")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"Daily abnormal returns around {readable_date}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        key = f"daily_abnormal_return_{date_source}"
        figures[key] = fig
        if output_path is not None:
            fig.savefig(
                output_path / f"goal2_{date_source}_daily_abnormal_return.png",
                dpi=180,
                bbox_inches="tight",
            )

        # One distribution graph for every requested horizon.
        events = result["analysis_events"].copy()
        horizons = result["metadata"]["impact_horizons"]
        for horizon in horizons:
            car_col = _post_event_car_column(horizon)
            groups = [
                events.loc[events["entity_result"].eq(label), car_col]
                .dropna()
                .to_numpy()
                for label in ["win", "loss"]
            ]
            nonempty = [
                (label, values)
                for label, values in zip(["win", "loss"], groups)
                if len(values)
            ]
            fig, ax = plt.subplots(figsize=(7.5, 5.25))
            if nonempty:
                labels, values = zip(*nonempty)
                try:
                    ax.boxplot(values, tick_labels=labels, showmeans=True)
                except TypeError:  # pragma: no cover - older Matplotlib
                    ax.boxplot(values, labels=labels, showmeans=True)
            ax.axhline(0, linewidth=1)
            ax.set_ylabel(f"{horizon}-day cumulative abnormal return")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_title(
                f"{horizon}-day abnormal return after {readable_date}"
            )
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            key = f"car_distribution_{date_source}_{horizon}d"
            figures[key] = fig
            if output_path is not None:
                fig.savefig(
                    output_path
                    / f"goal2_{date_source}_{horizon}d_car_distribution.png",
                    dpi=180,
                    bbox_inches="tight",
                )

    if show:
        plt.show()
    return figures


__all__ = [
    "ANALYSIS_VERSION",
    "analyze_case_outcome_factors",
    "analyze_stock_price_impact",
    "plot_case_outcome_factors",
    "plot_stock_price_impact",
]
