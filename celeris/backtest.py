"""Backtesting engine, leakage verification, and model evaluation protocols for Celeris."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from celeris.forecast import (
    SeasonalNaiveModel,
    TrailingMeanModel,
    PooledDirectModel,
    IntermittentChallengerModel,
    EnsembleModel,
)


@dataclass(frozen=True)
class BacktestFold:
    """Specification of a backtest fold."""
    fold_name: str
    fold_type: str  # "development" or "frozen_holdout"
    origin_date: str
    target_start_date: str
    target_end_date: str
    horizon_months: int = 19


DEFAULT_FOLDS: List[BacktestFold] = [
    BacktestFold(
        fold_name="Fold_1",
        fold_type="development",
        origin_date="2022-01",
        target_start_date="2022-02",
        target_end_date="2023-08",
        horizon_months=19,
    ),
    BacktestFold(
        fold_name="Fold_2",
        fold_type="development",
        origin_date="2023-08",
        target_start_date="2023-09",
        target_end_date="2025-03",
        horizon_months=19,
    ),
    BacktestFold(
        fold_name="Fold_3_Frozen_Holdout",
        fold_type="frozen_holdout",
        origin_date="2025-03",
        target_start_date="2025-04",
        target_end_date="2026-10",
        horizon_months=19,
    ),
]


def compute_wape(actuals: np.ndarray, forecasts: np.ndarray) -> Optional[float]:
    """Compute Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|forecast - actual|) / sum(actual)
    Returns None if total actual is zero.
    """
    total_actual = float(np.sum(actuals))
    if total_actual <= 0:
        return None
    total_abs_err = float(np.sum(np.abs(forecasts - actuals)))
    return round(total_abs_err / total_actual, 4)


def compute_signed_bias(actuals: np.ndarray, forecasts: np.ndarray) -> Optional[float]:
    """Compute Signed Bias.

    Bias = sum(forecast - actual) / sum(actual)
    Positive bias indicates overforecasting.
    Returns None if total actual is zero.
    """
    total_actual = float(np.sum(actuals))
    if total_actual <= 0:
        return None
    total_bias_err = float(np.sum(forecasts - actuals))
    return round(total_bias_err / total_actual, 4)


def run_fold_backtest(
    history_df: pd.DataFrame,
    fold: BacktestFold,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Execute backtest for a single fold across candidate models."""
    origin_date = fold.origin_date
    as_of_history = history_df[history_df["Date"] <= origin_date].copy()

    # Fit all candidate models strictly on origin history
    snaive = SeasonalNaiveModel().fit(as_of_history, origin_date)
    tmean = TrailingMeanModel().fit(as_of_history, origin_date)
    pooled = PooledDirectModel().fit(as_of_history, origin_date)
    intermittent = IntermittentChallengerModel().fit(as_of_history, origin_date, pooled_model=pooled)
    ensemble = EnsembleModel().fit(as_of_history, origin_date, snaive=snaive, pooled=pooled, tmean=tmean)

    # Evaluation targets in the 19-month window
    target_mask = (history_df["Date"] >= fold.target_start_date) & (history_df["Date"] <= fold.target_end_date)
    eval_df = history_df[target_mask].copy()

    # Identify series to score (observed series)
    target_dates = pd.date_range(f"{fold.target_start_date}-01", f"{fold.target_end_date}-01", freq="MS").strftime("%Y-%m").tolist()
    series_keys = eval_df[["SKU_ID", "Region", "Product_Family", "Business_Unit"]].drop_duplicates().reset_index(drop=True)

    rows: List[dict] = []
    candidates = {
        "seasonal_naive": snaive,
        "trailing_mean": tmean,
        "pooled_direct": pooled,
        "intermittent_challenger": intermittent,
        "ensemble": ensemble,
    }

    # Actual target map: (Date, SKU, Region) -> (actual_demand, demand_evidence)
    actual_map = dict(zip(
        zip(eval_df["Date"], eval_df["SKU_ID"], eval_df["Region"]),
        zip(eval_df["reconstructed_demand"], eval_df["demand_evidence"])
    ))

    series_records = series_keys.to_dict("records")
    series_keys_list = [(s["SKU_ID"], s["Region"]) for s in series_records]

    for h_idx, t_date in enumerate(target_dates, start=1):
        # Predict all series simultaneously in batch
        f_snaive_all = snaive.predict_batch(series_keys_list, t_date, h_idx)
        f_tmean_all = tmean.predict_batch(series_keys_list, t_date, h_idx)
        f_pooled_all = pooled.predict_batch(series_keys_list, t_date, h_idx)
        f_intermittent_all = intermittent.predict_batch(series_keys_list, t_date, h_idx)
        f_ensemble_all = ensemble.predict_batch(series_keys_list, t_date, h_idx)

        for i, s in enumerate(series_records):
            sku = s["SKU_ID"]
            reg = s["Region"]
            fam = s["Product_Family"]
            bu = s["Business_Unit"]

            actual_val, ev = actual_map.get((t_date, sku, reg), (None, "unobserved"))

            preds = {
                "seasonal_naive": f_snaive_all[i],
                "trailing_mean": f_tmean_all[i],
                "pooled_direct": f_pooled_all[i],
                "intermittent_challenger": f_intermittent_all[i],
                "ensemble": f_ensemble_all[i],
            }

            for c_name, pred_val in preds.items():
                err = float(pred_val - actual_val) if actual_val is not None and pd.notna(actual_val) else None
                abs_err = abs(err) if err is not None else None

                rows.append({
                    "fold": fold.fold_name,
                    "fold_type": fold.fold_type,
                    "origin": origin_date,
                    "target_date": t_date,
                    "horizon": h_idx,
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": fam,
                    "Business_Unit": bu,
                    "candidate": c_name,
                    "forecast": pred_val,
                    "actual": actual_val,
                    "demand_evidence": ev,
                    "error": err,
                    "abs_error": abs_err,
                })

    results_df = pd.DataFrame(rows)

    # Compute fold-level metrics per candidate on eligible targets
    metrics: Dict[str, Dict[str, Any]] = {}
    scored = results_df.dropna(subset=["actual", "error"])

    for c_name in candidates.keys():
        c_df = scored[scored["candidate"] == c_name]
        acts = c_df["actual"].to_numpy(dtype=float)
        fcts = c_df["forecast"].to_numpy(dtype=float)

        wape = compute_wape(acts, fcts)
        bias = compute_signed_bias(acts, fcts)

        metrics[c_name] = {
            "fold": fold.fold_name,
            "fold_type": fold.fold_type,
            "origin": origin_date,
            "wape": wape,
            "signed_bias": bias,
            "total_actual": float(np.sum(acts)),
            "total_forecast": float(np.sum(fcts)),
            "scored_rows": len(acts),
        }

    return results_df, metrics


def run_full_backtest(
    history_df: pd.DataFrame,
    folds: Optional[List[BacktestFold]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete 3-fold backtesting protocol, determine winner on dev folds, and evaluate frozen holdout."""
    active_folds = folds or DEFAULT_FOLDS
    all_results: List[pd.DataFrame] = []
    fold_metrics: Dict[str, Dict[str, Any]] = {}

    for fold in active_folds:
        f_df, f_metrics = run_fold_backtest(history_df, fold)
        all_results.append(f_df)
        fold_metrics[fold.fold_name] = f_metrics

    combined_df = pd.concat(all_results, ignore_index=True)

    # Evaluate development folds (Fold 1 + Fold 2)
    dev_scored = combined_df[(combined_df["fold_type"] == "development") & combined_df["actual"].notna()]
    dev_metrics: Dict[str, Dict[str, Any]] = {}

    candidates = ["seasonal_naive", "trailing_mean", "pooled_direct", "intermittent_challenger"]
    best_candidate = None
    lowest_wape = float("inf")

    for c in candidates:
        sub = dev_scored[dev_scored["candidate"] == c]
        acts = sub["actual"].to_numpy(dtype=float)
        fcts = sub["forecast"].to_numpy(dtype=float)
        wape = compute_wape(acts, fcts)
        bias = compute_signed_bias(acts, fcts)

        dev_metrics[c] = {
            "wape": wape,
            "signed_bias": bias,
            "total_actual": float(np.sum(acts)),
            "total_forecast": float(np.sum(fcts)),
            "rows": len(acts),
        }

        if wape is not None and wape < lowest_wape:
            lowest_wape = wape
            best_candidate = c

    # Evaluate frozen holdout fold (Fold 3)
    holdout_scored = combined_df[(combined_df["fold_type"] == "frozen_holdout") & combined_df["actual"].notna()]
    holdout_metrics: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        sub = holdout_scored[holdout_scored["candidate"] == c]
        acts = sub["actual"].to_numpy(dtype=float)
        fcts = sub["forecast"].to_numpy(dtype=float)
        holdout_metrics[c] = {
            "wape": compute_wape(acts, fcts),
            "signed_bias": compute_signed_bias(acts, fcts),
            "total_actual": float(np.sum(acts)),
            "total_forecast": float(np.sum(fcts)),
            "rows": len(acts),
        }

    summary = {
        "development_metrics": dev_metrics,
        "selected_winner": best_candidate,
        "frozen_holdout_metrics": holdout_metrics,
        "per_fold_metrics": fold_metrics,
    }

    return combined_df, summary


def verify_leakage_safety(history_df: pd.DataFrame, fold: BacktestFold) -> bool:
    """Verify origin-safety: perturbing future outcomes after fold origin must yield identical forecasts."""
    # 1. Base run
    res_base, _ = run_fold_backtest(history_df, fold)

    # 2. Perturb future data (Date > origin_date)
    perturbed_df = history_df.copy()
    future_mask = perturbed_df["Date"] > fold.origin_date

    # Corrupt future reconstructed demand with wild numbers
    perturbed_df.loc[future_mask, "reconstructed_demand"] = (
        perturbed_df.loc[future_mask, "reconstructed_demand"].fillna(100.0) * 17.5 + 999.0
    )
    perturbed_df.loc[future_mask, "Units_Sold"] = 8888.0

    # 3. Rerun backtest on perturbed data
    res_perturbed, _ = run_fold_backtest(perturbed_df, fold)

    # 4. Compare forecasts generated by each candidate model
    for candidate in ["seasonal_naive", "trailing_mean", "pooled_direct", "intermittent_challenger"]:
        base_preds = res_base[res_base["candidate"] == candidate]["forecast"].tolist()
        pert_preds = res_perturbed[res_perturbed["candidate"] == candidate]["forecast"].tolist()

        if base_preds != pert_preds:
            return False

    return True
