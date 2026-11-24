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
    as_of_df = history_df[history_df["Date"] <= origin_date].sort_values(by=["SKU_ID", "Region", "Date"]).reset_index(drop=True)
    if as_of_df.empty:
        return {}

    sku_meta: Dict[str, dict] = {}
    if sku_master_df is not None and not sku_master_df.empty:
        sku_meta = {str(r["SKU_ID"]): r for r in sku_master_df.to_dict("records")}

    sku_arr = as_of_df["SKU_ID"].astype(str).to_numpy()
    reg_arr = as_of_df["Region"].astype(str).to_numpy()
    fam_arr = as_of_df["Product_Family"].astype(str).to_numpy() if "Product_Family" in as_of_df.columns else None
    bu_arr = as_of_df["Business_Unit"].astype(str).to_numpy() if "Business_Unit" in as_of_df.columns else None
    demand_arr = as_of_df["reconstructed_demand"].to_numpy(dtype=np.float64)
    price_arr = as_of_df["Unit_Price_USD"].to_numpy(dtype=np.float64) if "Unit_Price_USD" in as_of_df.columns else None

    change_mask = (sku_arr[1:] != sku_arr[:-1]) | (reg_arr[1:] != reg_arr[:-1])
    split_indices = np.where(change_mask)[0] + 1
    start_indices = np.concatenate(([0], split_indices))
    end_indices = np.concatenate((split_indices, [len(sku_arr)]))

    features: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for start, end in zip(start_indices, end_indices):
        sku_id = sku_arr[start]
        region = reg_arr[start]
        meta = sku_meta.get(sku_id, {})
        fam = fam_arr[start] if (fam_arr is not None and fam_arr[start] != "" and fam_arr[start] != "nan") else meta.get("Product_Family", "")
        bu = bu_arr[start] if (bu_arr is not None and bu_arr[start] != "" and bu_arr[start] != "nan") else meta.get("Business_Unit", "")

        demands = demand_arr[start:end]
        valid_demands = demands[~np.isnan(demands)]
        elig_obs = len(valid_demands)
        positives = valid_demands[valid_demands > 0]
        num_pos = len(positives)

        if num_pos < 2:
            iclass, adi, cv2 = "insufficient_history", None, None
        else:
            adi = float(elig_obs / num_pos)
            pos_mean = float(np.mean(positives))
            pos_var = float(np.var(positives, ddof=0))
            cv2 = float(pos_var / (pos_mean ** 2)) if pos_mean > 0 else 0.0
            is_high_adi = adi >= 1.32
            is_high_cv2 = cv2 >= 0.49
            if not is_high_adi and not is_high_cv2:
                iclass = "smooth"
            elif is_high_adi and not is_high_cv2:
                iclass = "intermittent"
            elif not is_high_adi and is_high_cv2:
                iclass = "erratic"
            else:
                iclass = "lumpy"

        n = len(demands)
        val_0 = demands[-1] if n >= 1 and not np.isnan(demands[-1]) else 0.0
        val_1 = demands[-2] if n >= 2 and not np.isnan(demands[-2]) else val_0
        val_2 = demands[-3] if n >= 3 and not np.isnan(demands[-3]) else val_1
        val_5 = demands[-6] if n >= 6 and not np.isnan(demands[-6]) else val_2
        val_11 = demands[-12] if n >= 12 and not np.isnan(demands[-12]) else val_5

        d3 = demands[max(0, n - 3):]
        v3 = d3[~np.isnan(d3)]
        mean_3 = float(np.mean(v3)) if len(v3) > 0 else 0.0

        d6 = demands[max(0, n - 6):]
        v6 = d6[~np.isnan(d6)]
        mean_6 = float(np.mean(v6)) if len(v6) > 0 else 0.0
        std_6 = float(np.std(v6, ddof=0)) if len(v6) > 1 else 0.0

        d12 = demands[max(0, n - 12):]
        v12 = d12[~np.isnan(d12)]
        mean_12 = float(np.mean(v12)) if len(v12) > 0 else 0.0
        zero_share_12 = float(np.sum(v12 == 0) / len(v12)) if len(v12) > 0 else 0.0

        pos_rel = np.where(demands > 0)[0]
        time_since_last_order = int(n - 1 - pos_rel[-1]) if len(pos_rel) > 0 else n

        last_price = float(price_arr[start]) if price_arr is not None and not np.isnan(price_arr[start]) else 0.0

        features[(sku_id, region)] = {
            "SKU_ID": sku_id,
            "Region": region,
            "Product_Family": fam,
            "Business_Unit": bu,
            "origin_date": origin_date,
            "intermittency_class": iclass,
            "adi": round(adi, 4) if adi is not None else 1.0,
            "cv2": round(cv2, 4) if cv2 is not None else 0.0,
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
            "last_price": last_price,
        }

    return features
