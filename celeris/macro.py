"""Macroeconomic indicator lead/lag analysis and scoped event matching for Celeris."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class MacroLeadLagResult:
    """Best leading indicator and lag for a product family."""
    product_family: str
    best_indicator: str
    best_lag: int
    correlation: float
    all_correlations: Dict[str, Dict[int, float]]

    def to_dict(self) -> dict:
        return asdict(self)


INDICATOR_COLS = [
    "ISM_Manufacturing_PMI",
    "Auto_Production_YoY_Pct",
    "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index",
    "New_Vehicle_Orders_Index",
    "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index",
    "EV_Adoption_Index",
]


def analyze_macro_lead_lag(
    history_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    max_lag: int = 12,
) -> Dict[str, MacroLeadLagResult]:
    """Analyze which macroeconomic indicator leads which product family and at what lag (0-12 months)."""
    results: Dict[str, MacroLeadLagResult] = {}

    if macro_df.empty or history_df.empty:
        return results

    # Ensure Date format
    m_df = macro_df.copy()
    m_df["Date"] = m_df["Date"].astype(str).str.strip()
    m_df = m_df.sort_values(by="Date").reset_index(drop=True)

    available_indicators = [c for c in INDICATOR_COLS if c in m_df.columns]
    if not available_indicators:
        return results

    # Aggregate demand by month and product family
    family_monthly = (
        history_df.dropna(subset=["reconstructed_demand"])
        .groupby(["Date", "Product_Family"])["reconstructed_demand"]
        .sum()
        .reset_index()
    )

    for fam, f_group in family_monthly.groupby("Product_Family"):
        f_series = f_group.set_index("Date")["reconstructed_demand"]

        family_corrs: Dict[str, Dict[int, float]] = {}
        best_ind = available_indicators[0]
        best_lag = 0
        max_abs_corr = -1.0

        for ind in available_indicators:
            family_corrs[ind] = {}
            ind_series = m_df.set_index("Date")[ind].astype(float)

            for lag in range(max_lag + 1):
                # Shift indicator forward by lag months
                shifted_ind = ind_series.shift(lag)
                joined = pd.concat([f_series, shifted_ind], axis=1, join="inner").dropna()

                if len(joined) >= 6:
                    corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
                    if pd.isna(corr):
                        corr = 0.0
                else:
                    corr = 0.0

                family_corrs[ind][lag] = round(corr, 4)
                if abs(corr) > max_abs_corr:
                    max_abs_corr = abs(corr)
                    best_ind = ind
                    best_lag = lag

        results[fam] = MacroLeadLagResult(
            product_family=fam,
            best_indicator=best_ind,
            best_lag=best_lag,
            correlation=family_corrs[best_ind][best_lag],
            all_correlations=family_corrs,
        )

    return results


def match_scoped_events(
    events_df: Optional[pd.DataFrame],
    target_date: str,
    sku_id: str,
    region: str,
    product_family: str,
    sales_channel: str,
    customer_name: str,
) -> Dict[str, Any]:
    """Match calendar events to a series at target_date based on hierarchical scope."""
    if events_df is None or events_df.empty:
        return {
            "has_event": False,
            "event_count": 0,
            "is_price_change": False,
            "is_supply_constraint": False,
            "active_event_ids": [],
        }

    # Filter events active during target_date
    active_events = []
    is_price_change = False
    is_supply_constraint = False

    for _, ev in events_df.iterrows():
        s_date = str(ev.get("Start_Date", "")).strip()[:7]
        e_date = str(ev.get("End_Date", "")).strip()[:7] if pd.notna(ev.get("End_Date")) and str(ev.get("End_Date")).strip() else s_date

        if s_date <= target_date <= e_date:
            scope_type = str(ev.get("Scope_Type", "Global")).strip()
            scope_val = str(ev.get("Scope_Value", "All")).strip()
            ev_type = str(ev.get("Event_Type", "")).strip()

            # Scope matching
            matched = False
            if scope_type == "Global" or scope_val == "All":
                matched = True
            elif scope_type == "Region" and scope_val == region:
                matched = True
            elif scope_type == "Product_Family" and scope_val == product_family:
                matched = True
            elif scope_type == "Channel" and scope_val == sales_channel:
                matched = True
            elif scope_type == "Customer" and scope_val == customer_name:
                matched = True

            if matched:
                active_events.append(str(ev.get("Event_ID", "")))
                if "Price Change" in ev_type:
                    is_price_change = True
                if "Supply" in ev_type or "Disruption" in ev_type:
                    is_supply_constraint = True

    return {
        "has_event": len(active_events) > 0,
        "event_count": len(active_events),
        "is_price_change": is_price_change,
        "is_supply_constraint": is_supply_constraint,
        "active_event_ids": active_events,
    }
