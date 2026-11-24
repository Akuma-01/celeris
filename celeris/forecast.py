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
        self.seasonal_lookup: Dict[Tuple[str, str], Dict[int, int]] = {}
        self.fallback_lookup: Dict[Tuple[str, str], int] = {}

    def fit(self, history_df: pd.DataFrame, origin_date: str) -> SeasonalNaiveModel:
        """Store origin-frozen history and pre-compute seasonal month and fallback mappings."""
        as_of_df = history_df[history_df["Date"] <= origin_date].sort_values(by=["SKU_ID", "Region", "Date"])
        valid_df = as_of_df.dropna(subset=["reconstructed_demand"])

        self.seasonal_lookup = {}
        self.fallback_lookup = {}

        for (sku, reg), g in valid_df.groupby(["SKU_ID", "Region"]):
            m_dict: Dict[int, int] = {}
            for m, m_group in g.groupby("Month"):
                demands = m_group["reconstructed_demand"].tolist()
                if not demands:
                    continue
                if len(demands) >= 3:
                    smoothed = 0.60 * demands[-1] + 0.25 * demands[-2] + 0.15 * demands[-3]
                elif len(demands) == 2:
                    smoothed = 0.70 * demands[-1] + 0.30 * demands[-2]
                else:
                    smoothed = demands[-1]
                m_dict[int(m)] = int(round(max(0.0, float(smoothed))))

            self.seasonal_lookup[(sku, reg)] = m_dict
            tail = g["reconstructed_demand"].tail(6)
            self.fallback_lookup[(sku, reg)] = int(round(max(0.0, float(tail.mean())))) if not tail.empty else 0

        return self

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int = 1) -> int:
        """Predict integer demand for a specific SKU-region at target_date."""
        target_month = int(str(target_date).strip().split("-")[1])
        s_dict = self.seasonal_lookup.get((sku_id, region))
        if s_dict is not None and target_month in s_dict:
            return s_dict[target_month]
        return self.fallback_lookup.get((sku_id, region), 0)

    def predict_batch(self, series_keys: List[Tuple[str, str]], target_date: str, horizon: int = 1) -> List[int]:
        """Predict integer demand for a batch of series keys."""
        target_month = int(str(target_date).strip().split("-")[1])
        results: List[int] = []
        for k in series_keys:
            s_dict = self.seasonal_lookup.get(k)
            if s_dict is not None and target_month in s_dict:
                results.append(s_dict[target_month])
            else:
                results.append(self.fallback_lookup.get(k, 0))
        return results


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

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int = 1) -> int:
        """Predict integer demand for SKU-region."""
        mean_val = self.averages.get((sku_id, region), 0.0)
        return int(round(max(0.0, mean_val)))

    def predict_batch(self, series_keys: List[Tuple[str, str]], target_date: str, horizon: int = 1) -> List[int]:
        """Predict integer demand for a batch of series keys."""
        return [int(round(max(0.0, self.averages.get(k, 0.0)))) for k in series_keys]


class PooledDirectModel:
    """Pooled Direct Model.

    Trains a gradient-boosted regressor on historical origin–target pairs with horizon 1–19 as a feature.
    Combines origin-frozen lag and rolling stats with target-month calendar information.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=80,
            learning_rate=0.08,
            max_depth=5,
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
        max_origins: int = 4,
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
        valid_targets = as_of_df.dropna(subset=["reconstructed_demand"])
        target_map = dict(zip(
            zip(valid_targets["Date"], valid_targets["SKU_ID"], valid_targets["Region"]),
            valid_targets["reconstructed_demand"].astype(float)
        ))

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

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int = 1) -> int:
        """Predict non-negative integer demand."""
        target_month = int(str(target_date).strip().split("-")[1])
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

    def predict_batch(
        self,
        series_keys: List[Tuple[str, str]],
        target_date: str,
        horizon: int = 1,
    ) -> List[int]:
        """Predict non-negative integer demand for a batch of (SKU, Region) keys."""
        if not series_keys:
            return []

        target_month = int(str(target_date).strip().split("-")[1])
        if self.model is None or not self.origin_features:
            return [int(round(max(0.0, self.fallback_averages.get(k, 0.0)))) for k in series_keys]

        X_rows: List[List[float]] = []
        indices_to_predict: List[int] = []
        results: List[int] = [0] * len(series_keys)

        for i, (sku_id, region) in enumerate(series_keys):
            f = self.origin_features.get((sku_id, region))
            if f is None:
                results[i] = int(round(max(0.0, self.fallback_averages.get((sku_id, region), 0.0))))
                continue

            sku_code = self.sku_encoding.get(sku_id, 0)
            reg_code = self.region_encoding.get(region, 0)

            X_rows.append([
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
            ])
            indices_to_predict.append(i)

        if X_rows:
            X_mat = np.array(X_rows, dtype=np.float32)
            preds = self.model.predict(X_mat)
            for orig_idx, pred_val in zip(indices_to_predict, preds):
                results[orig_idx] = int(round(max(0.0, float(pred_val))))

        return results


class CrostonSBAModel:
    """Syntetos-Boylan Approximation (SBA) of Croston's Method for intermittent demand.

    Separates demand size and inter-demand arrival intervals with exponential smoothing,
    applying the (1 - alpha/2) debiasing correction factor.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.series_estimates: Dict[Tuple[str, str], float] = {}

    def fit(self, history_df: pd.DataFrame, origin_date: str) -> CrostonSBAModel:
        """Fit Croston/SBA model on series demand history strictly up to origin_date."""
        as_of_df = history_df[history_df["Date"] <= origin_date].copy()

        for (sku, region), group in as_of_df.groupby(["SKU_ID", "Region"]):
            demands = group.sort_values(by="Date")["reconstructed_demand"].dropna().tolist()
            if not demands:
                self.series_estimates[(sku, region)] = 0.0
                continue

            # Find first positive demand to initialize
            first_pos_idx = next((i for i, v in enumerate(demands) if v > 0), None)
            if first_pos_idx is None:
                self.series_estimates[(sku, region)] = 0.0
                continue

            z = float(demands[first_pos_idx])
            p = float(first_pos_idx + 1)
            q = 1

            for t in range(first_pos_idx + 1, len(demands)):
                y_t = float(demands[t])
                if y_t > 0:
                    z = self.alpha * y_t + (1.0 - self.alpha) * z
                    p = self.alpha * q + (1.0 - self.alpha) * p
                    q = 1
                else:
                    q += 1

            # SBA debiased rate: (1 - alpha / 2) * (z / p)
            sba_rate = (1.0 - self.alpha / 2.0) * (z / p) if p > 0 else 0.0
            self.series_estimates[(sku, region)] = float(max(0.0, sba_rate))

        return self

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int = 1) -> int:
        """Predict non-negative integer demand."""
        rate = self.series_estimates.get((sku_id, region), 0.0)
        return int(round(max(0.0, rate)))

    def predict_batch(self, series_keys: List[Tuple[str, str]], target_date: str, horizon: int = 1) -> List[int]:
        """Predict integer demand for a batch of series keys."""
        return [int(round(max(0.0, self.series_estimates.get(k, 0.0)))) for k in series_keys]


class IntermittentChallengerModel:
    """Intermittent Challenger Policy.

    Applies Croston SBA on series classified as sporadic (intermittent or lumpy)
    at origin, and dispatches to the pooled direct gradient boosted model elsewhere.
    """

    def __init__(self, random_state: int = 42, alpha: float = 0.1, pooled_model: Optional[PooledDirectModel] = None) -> None:
        self.croston = CrostonSBAModel(alpha=alpha)
        self.pooled = pooled_model or PooledDirectModel(random_state=random_state)
        self.sporadic_series: set[Tuple[str, str]] = set()

    def fit(self, history_df: pd.DataFrame, origin_date: str, pooled_model: Optional[PooledDirectModel] = None) -> IntermittentChallengerModel:
        """Fit both underlying models and record sporadic classifications."""
        self.croston.fit(history_df, origin_date)
        if pooled_model is not None:
            self.pooled = pooled_model
        else:
            self.pooled.fit(history_df, origin_date)

        # Identify sporadic series (intermittent or lumpy)
        as_of_df = history_df[history_df["Date"] <= origin_date].copy()
        origin_feats = extract_origin_features(as_of_df, origin_date)
        for (sku, reg), f in origin_feats.items():
            if f.get("intermittency_class") in ["intermittent", "lumpy"]:
                self.sporadic_series.add((sku, reg))

        return self

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int = 1) -> int:
        """Dispatch based on intermittency class."""
        if (sku_id, region) in self.sporadic_series:
            return self.croston.predict(sku_id, region, target_date, horizon)
        return self.pooled.predict(sku_id, region, target_date, horizon)

    def predict_batch(self, series_keys: List[Tuple[str, str]], target_date: str, horizon: int = 1) -> List[int]:
        """Batch dispatch based on intermittency class."""
        pooled_preds = self.pooled.predict_batch(series_keys, target_date, horizon)
        results: List[int] = []
        for i, key in enumerate(series_keys):
            if key in self.sporadic_series:
                results.append(self.croston.predict(key[0], key[1], target_date, horizon))
            else:
                results.append(pooled_preds[i])
        return results


class EnsembleModel:
    """Ensemble Blend Model.

    Combines Multi-Cycle Seasonal Naïve (70%), Pooled Direct (25%), and Trailing Mean (5%)
    to achieve optimal accuracy and zero-bias balance across both seasonal cycles and macro trends.
    """

    def __init__(
        self,
        snaive: Optional[SeasonalNaiveModel] = None,
        pooled: Optional[PooledDirectModel] = None,
        tmean: Optional[TrailingMeanModel] = None,
        w_snaive: float = 0.70,
        w_pooled: float = 0.25,
        w_tmean: float = 0.05,
    ) -> None:
        self.snaive = snaive or SeasonalNaiveModel()
        self.pooled = pooled or PooledDirectModel()
        self.tmean = tmean or TrailingMeanModel()
        self.w_snaive = w_snaive
        self.w_pooled = w_pooled
        self.w_tmean = w_tmean

    def fit(
        self,
        history_df: pd.DataFrame,
        origin_date: str,
        snaive: Optional[SeasonalNaiveModel] = None,
        pooled: Optional[PooledDirectModel] = None,
        tmean: Optional[TrailingMeanModel] = None,
    ) -> EnsembleModel:
        """Fit underlying constituent models, optionally reusing pre-fitted instances."""
        if snaive is not None:
            self.snaive = snaive
        else:
            self.snaive.fit(history_df, origin_date)

        if pooled is not None:
            self.pooled = pooled
        else:
            self.pooled.fit(history_df, origin_date)

        if tmean is not None:
            self.tmean = tmean
        else:
            self.tmean.fit(history_df, origin_date)
        return self

    def predict(self, sku_id: str, region: str, target_date: str, horizon: int = 1) -> int:
        """Predict blended demand."""
        p_sn = self.snaive.predict(sku_id, region, target_date, horizon)
        p_pd = self.pooled.predict(sku_id, region, target_date, horizon)
        p_tm = self.tmean.predict(sku_id, region, target_date, horizon)
        blend = self.w_snaive * p_sn + self.w_pooled * p_pd + self.w_tmean * p_tm
        return int(round(max(0.0, blend)))

    def predict_batch(self, series_keys: List[Tuple[str, str]], target_date: str, horizon: int = 1) -> List[int]:
        """Batch predict blended demand."""
        preds_sn = self.snaive.predict_batch(series_keys, target_date, horizon)
        preds_pd = self.pooled.predict_batch(series_keys, target_date, horizon)
        preds_tm = self.tmean.predict_batch(series_keys, target_date, horizon)
        results: List[int] = []
        for sn_val, pd_val, tm_val in zip(preds_sn, preds_pd, preds_tm):
            blend = self.w_snaive * sn_val + self.w_pooled * pd_val + self.w_tmean * tm_val
            results.append(int(round(max(0.0, blend))))
        return results
