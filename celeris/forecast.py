"""Forecasting candidate models: Baselines (Seasonal Naïve, Trailing Mean) and Pooled Direct Model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from celeris.features import extract_origin_features


class SeasonalNaiveModel:
    """Seasonal Naïve Baseline.

    Uses the same delivery month in the most recent available seasonal cycle as of origin.
    Repeats that cycle across multi-year horizons.
    """

    def __init__(self) -> None:
        self.history_by_series: Dict[Tuple[str, str], pd.DataFrame] = {}

    def fit(self, history_df: pd.DataFrame, origin_date: str) -> SeasonalNaiveModel:
        """Store origin-frozen history."""
        as_of_df = history_df[history_df["Date"] <= origin_date].copy()
        for (sku, region), g in as_of_df.groupby(["SKU_ID", "Region"]):
            self.history_by_series[(sku, region)] = g.sort_values(by="Date").copy()
        return self

    def predict(self, sku_id: str, region: str, target_date: str) -> int:
        """Predict integer demand for a specific SKU-region at target_date."""
        target_month = int(target_date.split("-")[1])
        g = self.history_by_series.get((sku_id, region))
        if g is None or g.empty:
            return 0

        # Look for matching seasonal month in history, latest first
        matching = g[g["Month"] == target_month].dropna(subset=["reconstructed_demand"])
        if not matching.empty:
            val = matching.iloc[-1]["reconstructed_demand"]
            return int(round(max(0.0, float(val))))

        # Fallback to trailing mean of last 6 observed months if seasonal month not yet observed
        tail = g["reconstructed_demand"].dropna().tail(6)
        if not tail.empty:
            return int(round(max(0.0, float(tail.mean()))))

        return 0


class TrailingMeanModel:
    """Trailing Mean Baseline.

    Averages the last six calendar months of eligible demand, including observed zeros.
    Missing months contribute no observation to the denominator.
    """

    def __init__(self, window: int = 6) -> None:
        self.window = window
        self.averages: Dict[Tuple[str, str], float] = {}

    def fit(self, history_df: pd.DataFrame, origin_date: str) -> TrailingMeanModel:
        """Compute trailing mean per series as of origin."""
        as_of_df = history_df[history_df["Date"] <= origin_date].copy()
        for (sku, region), g in as_of_df.groupby(["SKU_ID", "Region"]):
            tail = g.sort_values(by="Date")["reconstructed_demand"].dropna().tail(self.window)
            if not tail.empty:
                self.averages[(sku, region)] = float(tail.mean())
            else:
                self.averages[(sku, region)] = 0.0
        return self

    def predict(self, sku_id: str, region: str, target_date: str) -> int:
        """Predict integer demand for SKU-region."""
        mean_val = self.averages.get((sku_id, region), 0.0)
        return int(round(max(0.0, mean_val)))


class PooledDirectModel:
    """Pooled Direct Model.

    Trains a gradient-boosted regressor on historical origin–target pairs with horizon 1–19 as a feature.
    Combines origin-frozen lag and rolling stats with target-month calendar information.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=random_state,
        )
        self.sku_encoding: Dict[str, int] = {}
        self.region_encoding: Dict[str, int] = {}
        self.origin_features: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.fallback_averages: Dict[Tuple[str, str], float] = {}

    def _prepare_training_data(
        self,
        history_df: pd.DataFrame,
        origin_date: str,
        max_origins: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct origin-target training pairs strictly prior to origin_date."""
        as_of_df = history_df[history_df["Date"] <= origin_date].copy()
        all_dates = sorted(as_of_df["Date"].unique().tolist())

        # Select candidate origin dates for training (needing at least 12 months history prior)
        train_origins = [d for d in all_dates if d >= "2020-01" and d < origin_date]
        if len(train_origins) > max_origins:
            # Subsample evenly
            step = max(1, len(train_origins) // max_origins)
            train_origins = train_origins[::step]

        X_rows: List[List[float]] = []
        y_vals: List[float] = []

        # Target lookup
        target_map = {
            (r["Date"], r["SKU_ID"], r["Region"]): float(r["reconstructed_demand"])
            for _, r in as_of_df.dropna(subset=["reconstructed_demand"]).iterrows()
        }

        for o_date in train_origins:
            o_feats = extract_origin_features(as_of_df, o_date)
            o_idx = all_dates.index(o_date)
            # Future months available within as_of_df
            future_dates = all_dates[o_idx + 1: min(len(all_dates), o_idx + 20)]

            for h_idx, t_date in enumerate(future_dates, start=1):
                t_month = int(t_date.split("-")[1])
                for (sku, reg), f in o_feats.items():
                    y = target_map.get((t_date, sku, reg))
                    if y is None:
                        continue

                    sku_code = self.sku_encoding.setdefault(sku, len(self.sku_encoding))
                    reg_code = self.region_encoding.setdefault(reg, len(self.region_encoding))

                    row = [
                        float(h_idx),
                        float(t_month),
                        float(sku_code),
                        float(reg_code),
                        float(f["lag_1"]),
                        float(f["lag_2"]),
                        float(f["lag_3"]),
                        float(f["lag_6"]),
                        float(f["lag_12"]),
                        float(f["trailing_mean_3"]),
                        float(f["trailing_mean_6"]),
                        float(f["trailing_std_6"]),
                        float(f["trailing_mean_12"]),
                        float(f["zero_share_12"]),
                        float(f["time_since_last_order"]),
                        float(f["adi"]),
                        float(f["cv2"]),
                    ]
                    X_rows.append(row)
                    y_vals.append(y)

        return np.array(X_rows, dtype=np.float32), np.array(y_vals, dtype=np.float32)

    def fit(self, history_df: pd.DataFrame, origin_date: str) -> PooledDirectModel:
        """Fit model on historical origin-target pairs up to origin_date."""
        as_of_df = history_df[history_df["Date"] <= origin_date].copy()

        # Build encodings for all known SKUs and regions
        for sku in as_of_df["SKU_ID"].unique():
            if sku not in self.sku_encoding:
                self.sku_encoding[sku] = len(self.sku_encoding)
        for reg in as_of_df["Region"].unique():
            if reg not in self.region_encoding:
                self.region_encoding[reg] = len(self.region_encoding)

        # Compute fallback averages per series
        for (sku, reg), g in as_of_df.groupby(["SKU_ID", "Region"]):
            tail = g.sort_values(by="Date")["reconstructed_demand"].dropna().tail(6)
            self.fallback_averages[(sku, reg)] = float(tail.mean()) if not tail.empty else 0.0

        # Extract features as of origin
        self.origin_features = extract_origin_features(as_of_df, origin_date)

        # Generate training dataset
        X, y = self._prepare_training_data(as_of_df, origin_date)

        if len(X) >= 10:
            self.model.fit(X, y)
        else:
            self.model = None  # Fallback to mean if insufficient training data

        return self

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int) -> int:
        """Predict non-negative integer demand."""
        target_month = int(target_date.split("-")[1])
        f = self.origin_features.get((sku_id, region))
        fallback = self.fallback_averages.get((sku_id, region), 0.0)

        if self.model is None or f is None:
            return int(round(max(0.0, fallback)))

        sku_code = self.sku_encoding.get(sku_id, 0)
        reg_code = self.region_encoding.get(region, 0)

        feature_vector = np.array([[
            float(horizon),
            float(target_month),
            float(sku_code),
            float(reg_code),
            float(f["lag_1"]),
            float(f["lag_2"]),
            float(f["lag_3"]),
            float(f["lag_6"]),
            float(f["lag_12"]),
            float(f["trailing_mean_3"]),
            float(f["trailing_mean_6"]),
            float(f["trailing_std_6"]),
            float(f["trailing_mean_12"]),
            float(f["zero_share_12"]),
            float(f["time_since_last_order"]),
            float(f["adi"]),
            float(f["cv2"]),
        ]], dtype=np.float32)

        pred = self.model.predict(feature_vector)[0]
        return int(round(max(0.0, float(pred))))
