"""Origin-safe feature engineering and intermittency classification for Celeris."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class IntermittencyProfile:
    """Intermittency profile of a time series as of an origin date."""
    sku_id: str
    region: str
    origin_date: str
    eligible_observed_months: int
    positive_demand_months: int
    average_demand_interval: Optional[float]  # ADI
    coefficient_of_variation_sq: Optional[float]  # CV^2
    intermittency_class: str  # smooth, intermittent, erratic, lumpy, insufficient_history


def compute_intermittency_class(
    series_demand: pd.Series,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> Tuple[str, Optional[float], Optional[float], int, int]:
    """Classify series intermittency using ADI and CV^2 according to Syntetos-Boylan / Core spec.

    ADI = eligible observed months / positive demand months
    CV^2 = population variance of positive order sizes / (mean of positive order sizes)^2

    Thresholds:
    - ADI threshold = 1.32
    - CV^2 threshold = 0.49
    Values at threshold are treated as high.
    Fewer than 2 positive months -> insufficient_history.
    """
    valid = series_demand.dropna()
    eligible_observed_months = len(valid)
    positives = valid[valid > 0]
    num_positives = len(positives)

    if num_positives < 2:
        return "insufficient_history", None, None, eligible_observed_months, num_positives

    adi = float(eligible_observed_months / num_positives)
    pos_mean = float(np.mean(positives))
    pos_var = float(np.var(positives, ddof=0))  # population variance
    cv2 = float(pos_var / (pos_mean ** 2)) if pos_mean > 0 else 0.0

    is_high_adi = adi >= adi_threshold
    is_high_cv2 = cv2 >= cv2_threshold

    if not is_high_adi and not is_high_cv2:
        intermittency_class = "smooth"
    elif is_high_adi and not is_high_cv2:
        intermittency_class = "intermittent"
    elif not is_high_adi and is_high_cv2:
        intermittency_class = "erratic"
    else:
        intermittency_class = "lumpy"

    return intermittency_class, round(adi, 4), round(cv2, 4), eligible_observed_months, num_positives


def extract_origin_features(
    history_df: pd.DataFrame,
    origin_date: str,
    sku_master_df: Optional[pd.DataFrame] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Extract origin-safe historical features for every (SKU_ID, Region) as of origin_date.

    Guarantees no data after origin_date is accessed.
    """
    # Strict cutoff: only history strictly on or before origin_date
    as_of_df = history_df[history_df["Date"] <= origin_date].copy()
    features: Dict[Tuple[str, str], Dict[str, Any]] = {}

    sku_meta: Dict[str, dict] = {}
    if sku_master_df is not None and not sku_master_df.empty:
        for _, r in sku_master_df.iterrows():
            sku_meta[str(r["SKU_ID"])] = r.to_dict()

    for (sku_id, region), group in as_of_df.groupby(["SKU_ID", "Region"], sort=False):
        group = group.sort_values(by="Date").reset_index(drop=True)
        demands = group["reconstructed_demand"]

        # Intermittency class
        iclass, adi, cv2, elig_obs, num_pos = compute_intermittency_class(demands)

        # Lags as of origin
        # The last row is origin_date (or the latest observed month <= origin_date)
        n = len(group)
        val_0 = float(demands.iloc[-1]) if n >= 1 and pd.notna(demands.iloc[-1]) else 0.0
        val_1 = float(demands.iloc[-2]) if n >= 2 and pd.notna(demands.iloc[-2]) else val_0
        val_2 = float(demands.iloc[-3]) if n >= 3 and pd.notna(demands.iloc[-3]) else val_1
        val_5 = float(demands.iloc[-6]) if n >= 6 and pd.notna(demands.iloc[-6]) else val_2
        val_11 = float(demands.iloc[-12]) if n >= 12 and pd.notna(demands.iloc[-12]) else val_5

        # Trailing windows (3, 6, 12)
        tail_3 = demands.iloc[-3:].dropna()
        tail_6 = demands.iloc[-6:].dropna()
        tail_12 = demands.iloc[-12:].dropna()

        mean_3 = float(tail_3.mean()) if len(tail_3) > 0 else 0.0
        mean_6 = float(tail_6.mean()) if len(tail_6) > 0 else 0.0
        std_6 = float(tail_6.std(ddof=0)) if len(tail_6) > 1 else 0.0
        mean_12 = float(tail_12.mean()) if len(tail_12) > 0 else 0.0

        # Zero share in trailing 12 months
        zero_share_12 = float((tail_12 == 0).sum() / len(tail_12)) if len(tail_12) > 0 else 0.0

        # Time since last positive order
        pos_indices = group.index[group["reconstructed_demand"] > 0].tolist()
        time_since_last_order = (n - 1 - pos_indices[-1]) if pos_indices else n

        # Dimensions
        first_row = group.iloc[0]
        meta = sku_meta.get(sku_id, {})

        features[(sku_id, region)] = {
            "SKU_ID": sku_id,
            "Region": region,
            "Product_Family": first_row.get("Product_Family", meta.get("Product_Family", "")),
            "Business_Unit": first_row.get("Business_Unit", meta.get("Business_Unit", "")),
            "origin_date": origin_date,
            "intermittency_class": iclass,
            "adi": adi if adi is not None else 1.0,
            "cv2": cv2 if cv2 is not None else 0.0,
            "eligible_months": elig_obs,
            "positive_months": num_pos,
            "lag_1": val_0,
            "lag_2": val_1,
            "lag_3": val_2,
            "lag_6": val_5,
            "lag_12": val_11,
            "trailing_mean_3": mean_3,
            "trailing_mean_6": mean_6,
            "trailing_std_6": std_6,
            "trailing_mean_12": mean_12,
            "zero_share_12": zero_share_12,
            "time_since_last_order": time_since_last_order,
            "last_price": float(first_row.get("Unit_Price_USD", 0.0)) if pd.notna(first_row.get("Unit_Price_USD")) else 0.0,
        }

    return features
