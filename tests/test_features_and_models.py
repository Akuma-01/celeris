"""Tests for Milestone 3: Features, intermittency classes, models, leakage checks, and backtests."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from celeris.features import compute_intermittency_class, extract_origin_features
from celeris.forecast import SeasonalNaiveModel, TrailingMeanModel, PooledDirectModel
from celeris.backtest import (
    DEFAULT_FOLDS,
    compute_wape,
    compute_signed_bias,
    run_full_backtest,
    verify_leakage_safety,
)
from celeris.pipeline import PipelineRunner


def test_intermittency_classification():
    """Verify ADI and CV^2 calculation and 4-quadrant classification."""
    # 1. Smooth: constant monthly demand (ADI = 1.0 < 1.32, CV2 = 0.0 < 0.49)
    smooth_series = pd.Series([50, 50, 50, 50, 50, 50, 50, 50])
    c, adi, cv2, elig, pos = compute_intermittency_class(smooth_series)
    assert c == "smooth"
    assert adi == 1.0
    assert cv2 == 0.0

    # 2. Intermittent: demand every 2nd month, constant size (ADI = 2.0 >= 1.32, CV2 = 0.0 < 0.49)
    intermittent_series = pd.Series([50, 0, 50, 0, 50, 0, 50, 0])
    c, adi, cv2, elig, pos = compute_intermittency_class(intermittent_series)
    assert c == "intermittent"
    assert adi == 2.0
    assert cv2 == 0.0

    # 3. Erratic: demand every month, high variability (ADI = 1.0 < 1.32, CV2 >= 0.49)
    # mean = 105, variance = ((10-105)^2 + (200-105)^2)/2 = (9025 + 9025)/2 = 9025
    # CV2 = 9025 / 105^2 = 9025 / 11025 = 0.8186 >= 0.49
    erratic_series = pd.Series([10, 200, 10, 200, 10, 200])
    c, adi, cv2, elig, pos = compute_intermittency_class(erratic_series)
    assert c == "erratic"
    assert adi == 1.0
    assert cv2 >= 0.49

    # 4. Lumpy: both high ADI and high CV2
    lumpy_series = pd.Series([10, 0, 200, 0, 10, 0, 200, 0])
    c, adi, cv2, elig, pos = compute_intermittency_class(lumpy_series)
    assert c == "lumpy"
    assert adi >= 1.32
    assert cv2 >= 0.49

    # 5. Insufficient history (< 2 positive months)
    insuf_series = pd.Series([0, 0, 50, 0, 0])
    c, adi, cv2, elig, pos = compute_intermittency_class(insuf_series)
    assert c == "insufficient_history"
    assert adi is None
    assert cv2 is None


def test_metric_calculations():
    """Verify hand-calculated WAPE and Signed Bias examples."""
    actuals = np.array([100.0, 50.0, 80.0])
    forecasts = np.array([110.0, 40.0, 80.0])

    # Total actual = 230
    # Absolute errors: |110-100| = 10, |40-50| = 10, |80-80| = 0 -> Total abs err = 20
    # WAPE = 20 / 230 = 0.0870
    assert compute_wape(actuals, forecasts) == round(20.0 / 230.0, 4)

    # Signed errors: +10, -10, 0 -> Total signed err = 0
    # Bias = 0 / 230 = 0.0
    assert compute_signed_bias(actuals, forecasts) == 0.0

    # Overforecasting example
    over_fct = np.array([120.0, 60.0, 90.0])  # +20, +10, +10 -> Total bias = +40
    assert compute_signed_bias(actuals, over_fct) == round(40.0 / 230.0, 4)
    assert compute_signed_bias(actuals, over_fct) > 0  # positive bias = overforecasting


def test_leakage_safety_check(fixture_data_dir: Path, tmp_path: Path):
    """Verify that perturbing future outcomes after fold origin leaves predictions strictly unchanged."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "leak_test", is_fixture=True)
    res = runner.run()
    assert res.success is True

    history_df = pd.read_parquet(tmp_path / "leak_test" / "history.parquet")
    assert verify_leakage_safety(history_df, DEFAULT_FOLDS[0]) is True
    assert verify_leakage_safety(history_df, DEFAULT_FOLDS[1]) is True


def test_backtest_execution_and_selection(fixture_data_dir: Path, tmp_path: Path):
    """Verify 3-fold backtests, winner selection on development folds, and holdout reporting."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "bt_test", is_fixture=True)
    res = runner.run()
    assert res.success is True

    backtest_file = tmp_path / "bt_test" / "backtest.parquet"
    assert backtest_file.exists()
    bt_df = pd.read_parquet(backtest_file)

    # Assert 3 folds are present
    assert set(bt_df["fold"].unique()) == {"Fold_1", "Fold_2", "Fold_3_Frozen_Holdout"}
    assert {"seasonal_naive", "trailing_mean", "pooled_direct"}.issubset(set(bt_df["candidate"].unique()))

    # Assert all forecasts are non-negative integers
    assert (bt_df["forecast"] >= 0).all()
    assert (bt_df["forecast"] % 1 == 0).all()

    # Verify summary metrics
    summary = res.backtest_summary
    assert summary is not None
    assert summary["selected_winner"] in ["seasonal_naive", "trailing_mean", "pooled_direct", "intermittent_challenger", "ensemble"]
    assert "development_metrics" in summary
    assert "frozen_holdout_metrics" in summary

    # Winner must have the lowest WAPE on development folds
    dev_metrics = summary["development_metrics"]
    winner = summary["selected_winner"]
    winner_wape = dev_metrics[winner]["wape"]
    for c_name, m in dev_metrics.items():
        assert winner_wape <= m["wape"]

    # Frozen holdout must be evaluated separately
    holdout_metrics = summary["frozen_holdout_metrics"]
    assert winner in holdout_metrics
    assert holdout_metrics[winner]["wape"] is not None
