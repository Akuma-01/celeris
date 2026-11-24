"""Demand reconstruction, missing month vs zero handling, stockout backlog carry, and anomaly register."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class AnomalyRecord:
    """Entry in the anomaly register."""
    anomaly_id: str
    planning_key: str
    Date: str
    SKU_ID: str
    Region: str
    anomaly_type: str
    description: str
    original_value: str
    cleaned_value: str
    confidence: float
    action: str
    reason: str
    linked_evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def reconstruct_demand_and_anomalies(
    cleaned_history: pd.DataFrame,
    sku_master_df: pd.DataFrame,
    stockouts_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    hist_min_date: str = "2019-01",
    hist_max_date: str = "2026-10",
) -> Tuple[pd.DataFrame, List[AnomalyRecord]]:
    """Build the complete monthly panel (2019-01 -> 2026-10), reconstruct true unconstrained

    demand by resolving stockout shortfalls and backlog rebounds, distinguish missing months
    from zero orders, and produce the comprehensive anomaly register.
    """
    anomalies: List[AnomalyRecord] = []
    cleaned_df = cleaned_history.copy()

    # 1. Distinct planning keys (SKU_ID, Region) from history and master
    # Capture dimension metadata from cleaned history or SKU master
    sku_metadata: Dict[str, Dict[str, Any]] = {}
    if not sku_master_df.empty:
        for _, row in sku_master_df.iterrows():
            sku_metadata[str(row["SKU_ID"])] = {
                "Product_Family": row.get("Product_Family", ""),
                "Business_Unit": row.get("Business_Unit", ""),
                "SKU_Description": row.get("SKU_Description", ""),
                "Launch_Date": str(row.get("Launch_Date", "")).strip() if pd.notna(row.get("Launch_Date")) else "",
                "EOL_Date": str(row.get("EOL_Date", "")).strip() if pd.notna(row.get("EOL_Date")) else "",
                "Lifecycle_Status": row.get("Lifecycle_Status", "Active"),
                "Replaces_SKU": row.get("Replaces_SKU", ""),
            }

    # Distinct series observed in history
    required_series_cols = ["SKU_ID", "Region", "Product_Family", "Business_Unit", "SKU_Description", "Customer_Name", "Sales_Channel"]
    for c in required_series_cols:
        if c not in cleaned_df.columns:
            cleaned_df[c] = ""

    series_keys = (
        cleaned_df[required_series_cols]
        .drop_duplicates(subset=["SKU_ID", "Region"])
        .reset_index(drop=True)
    )

    if series_keys.empty:
        # Return empty panel with full typed schema
        empty_cols = [
            "Date", "Year", "Month", "SKU_ID", "Region", "Product_Family", "Business_Unit",
            "SKU_Description", "Customer_Name", "Sales_Channel", "is_observed",
            "Units_Sold", "reconstructed_demand", "demand_evidence", "Unit_Price_USD",
            "Revenue_USD", "Lifecycle_Status"
        ]
        return pd.DataFrame(columns=empty_cols), []

    # 2. Complete date range
    dates = pd.date_range(f"{hist_min_date}-01", f"{hist_max_date}-01", freq="MS").strftime("%Y-%m").tolist()

    # 3. Build Cartesian product of (Date) x (Planning Series)
    panel_rows: List[dict] = []
    for d in dates:
        for _, s in series_keys.iterrows():
            panel_rows.append({
                "Date": d,
                "SKU_ID": s["SKU_ID"],
                "Region": s["Region"],
                "Product_Family": s.get("Product_Family", ""),
                "Business_Unit": s.get("Business_Unit", ""),
                "SKU_Description": s.get("SKU_Description", ""),
                "Customer_Name": s.get("Customer_Name", ""),
                "Sales_Channel": s.get("Sales_Channel", ""),
            })

    panel_df = pd.DataFrame(panel_rows)

    # Merge cleaned transactions (summing units/revenue if multiple records exist for the same month)
    agg_history = (
        cleaned_df.groupby(["Date", "SKU_ID", "Region"], as_index=False)
        .agg({
            "Units_Sold": "sum",
            "Revenue_USD": "sum",
            "Unit_Price_USD": "mean",
            "Record_ID": lambda x: "|".join(x.astype(str)),
        })
        .rename(columns={"Record_ID": "Source_Record_IDs"})
    )

    panel_df = panel_df.merge(agg_history, on=["Date", "SKU_ID", "Region"], how="left")

    # Determine is_observed
    panel_df["is_observed"] = panel_df["Units_Sold"].notna()

    # Prepare stockouts lookup
    stockout_map: Dict[Tuple[str, str, str], dict] = {}
    if stockouts_df is not None and not stockouts_df.empty:
        s_df = stockouts_df.copy()
        s_df["Date"] = s_df["Date"].astype(str).str.strip()
        s_df["SKU_ID"] = s_df["SKU_ID"].astype(str).str.strip()
        s_df["Region"] = s_df["Region"].astype(str).str.strip()
        for _, s_row in s_df.iterrows():
            key = (s_row["Date"], s_row["SKU_ID"], s_row["Region"])
            stockout_map[key] = s_row.to_dict()

    # Chronological traversal per series for demand reconstruction and carry tracking
    panel_df = panel_df.sort_values(by=["SKU_ID", "Region", "Date"]).reset_index(drop=True)

    reconstructed_demands: List[Optional[float]] = []
    demand_evidences: List[str] = []
    lifecycle_states: List[str] = []

    # Iterate by series to track pending backlog across months
    for (sku_id, region), series_data in panel_df.groupby(["SKU_ID", "Region"], sort=False):
        pending_carry = 0.0
        meta = sku_metadata.get(sku_id, {})
        launch_date = meta.get("Launch_Date", "")
        eol_date = meta.get("EOL_Date", "")

        for idx, row in series_data.iterrows():
            d = row["Date"]
            pkey = f"{d}|{sku_id}|{region}"
            is_obs = row["is_observed"]
            shipped = float(row["Units_Sold"]) if is_obs and pd.notna(row["Units_Sold"]) else None

            # Lifecycle status evaluation
            is_pre_launch = bool(launch_date and d < launch_date)
            is_post_eol = bool(eol_date and d > eol_date)

            if is_pre_launch:
                lifecycle_state = "Pre-Launch"
                if is_obs and shipped and shipped > 0:
                    anomalies.append(AnomalyRecord(
                        anomaly_id=f"ANOM-{uuid.uuid4().hex[:8]}",
                        planning_key=pkey,
                        Date=d,
                        SKU_ID=sku_id,
                        Region=region,
                        anomaly_type="PRE_LAUNCH_TRANSACTION",
                        description=f"Transaction observed prior to Launch_Date ({launch_date})",
                        original_value=f"Units_Sold={shipped}",
                        cleaned_value=str(shipped),
                        confidence=0.95,
                        action="retained_for_audit",
                        reason="Order booked before official product launch date",
                        linked_evidence=f"Launch_Date={launch_date}",
                    ))
            elif is_post_eol:
                lifecycle_state = "Post-EOL"
                if is_obs and shipped and shipped > 0:
                    anomalies.append(AnomalyRecord(
                        anomaly_id=f"ANOM-{uuid.uuid4().hex[:8]}",
                        planning_key=pkey,
                        Date=d,
                        SKU_ID=sku_id,
                        Region=region,
                        anomaly_type="POST_EOL_TRANSACTION",
                        description=f"Transaction observed after EOL_Date ({eol_date})",
                        original_value=f"Units_Sold={shipped}",
                        cleaned_value=str(shipped),
                        confidence=0.95,
                        action="flagged_lifecycle_conflict",
                        reason="Business rule 5: No sales permitted after an SKU's EOL_Date",
                        linked_evidence=f"EOL_Date={eol_date}",
                    ))
            else:
                lifecycle_state = "Active"

            lifecycle_states.append(lifecycle_state)

            # Demand reconstruction & anomaly classification
            stockout_entry = stockout_map.get((d, sku_id, region))

            # Case A: Stockout Ledger Record Present
            if stockout_entry is not None:
                req_units = float(stockout_entry.get("Requested_Units", 0))
                backordered = float(stockout_entry.get("Backordered_Units", 0))
                lost = float(stockout_entry.get("Lost_Units", 0))
                reason = str(stockout_entry.get("Reason", "Stockout"))

                reconstructed_demands.append(req_units)
                demand_evidences.append("stockout_ledger_requested")

                anomalies.append(AnomalyRecord(
                    anomaly_id=f"ANOM-{uuid.uuid4().hex[:8]}",
                    planning_key=pkey,
                    Date=d,
                    SKU_ID=sku_id,
                    Region=region,
                    anomaly_type="STOCKOUT_SHORTAGE",
                    description=f"Supply constrained month: Shipped {shipped or 0:.0f} of {req_units:.0f} requested units",
                    original_value=f"Shipped={shipped or 0:.0f}",
                    cleaned_value=f"Reconstructed_Demand={req_units:.0f}",
                    confidence=1.0,
                    action="reconstructed_to_requested",
                    reason=f"Corrected for supply limitation: {reason}. Backordered: {backordered:.0f}, Lost: {lost:.0f}",
                    linked_evidence=f"Stockout_Ledger: Reason={reason}, FillRate={stockout_entry.get('Fill_Rate_Pct')}%",
                ))

                # Track carry-over backorders to subsequent month
                pending_carry = backordered

            # Case B: Backlog carry rebound from previous stockout month
            elif pending_carry > 0:
                if is_obs and shipped is not None:
                    # Deduct prior backorder to isolate true new requested demand
                    net_demand = max(0.0, shipped - pending_carry)
                    reconstructed_demands.append(net_demand)
                    demand_evidences.append("backlog_carry_rebound")

                    anomalies.append(AnomalyRecord(
                        anomaly_id=f"ANOM-{uuid.uuid4().hex[:8]}",
                        planning_key=pkey,
                        Date=d,
                        SKU_ID=sku_id,
                        Region=region,
                        anomaly_type="BACKLOG_CARRY_REBOUND",
                        description=f"Shipment of {shipped:.0f} includes {pending_carry:.0f} backlog units carried from prior month shortage",
                        original_value=f"Shipped={shipped:.0f}",
                        cleaned_value=f"Reconstructed_Demand={net_demand:.0f}",
                        confidence=0.95,
                        action="deducted_prior_backorders",
                        reason=f"Subtracted {pending_carry:.0f} carry-in units to prevent training model on false demand rebound",
                        linked_evidence=f"Prior_Month_Backorder={pending_carry:.0f}",
                    ))
                    # Reset carry after attribution
                    pending_carry = 0.0
                else:
                    # Missing or unobserved month immediately after stockout
                    reconstructed_demands.append(None)
                    demand_evidences.append("missing_record")
                    pending_carry = 0.0

            # Case C: Genuine zero order
            elif is_obs and shipped == 0.0:
                reconstructed_demands.append(0.0)
                demand_evidences.append("observed_zero")
                anomalies.append(AnomalyRecord(
                    anomaly_id=f"ANOM-{uuid.uuid4().hex[:8]}",
                    planning_key=pkey,
                    Date=d,
                    SKU_ID=sku_id,
                    Region=region,
                    anomaly_type="ZERO_ORDER",
                    description="Genuine zero-order month (observed order of 0 units)",
                    original_value="0",
                    cleaned_value="0",
                    confidence=1.0,
                    action="retained_true_zero",
                    reason="Business rule 3: Units_Sold = 0 means no order. Retained as valid zero observation.",
                    linked_evidence=f"Source_IDs={row.get('Source_Record_IDs', '')}",
                ))

            # Case D: Missing month (absence != zero)
            elif not is_obs:
                reconstructed_demands.append(None)
                demand_evidences.append("missing_record")
                anomalies.append(AnomalyRecord(
                    anomaly_id=f"ANOM-{uuid.uuid4().hex[:8]}",
                    planning_key=pkey,
                    Date=d,
                    SKU_ID=sku_id,
                    Region=region,
                    anomaly_type="MISSING_MONTH",
                    description="Unobserved month with no ERP transaction record",
                    original_value="None",
                    cleaned_value="NaN",
                    confidence=1.0,
                    action="flagged_unobserved_gap",
                    reason="Business rule 3: A missing row is not a zero. Contributes no zero observation.",
                    linked_evidence="ERP_Transaction_Absent",
                ))

            # Case E: Ordinary observed month
            else:
                reconstructed_demands.append(shipped)
                demand_evidences.append("cleaned_shipments")

    panel_df["reconstructed_demand"] = pd.Series(reconstructed_demands, dtype="Float64")
    panel_df["demand_evidence"] = demand_evidences
    panel_df["Lifecycle_Status"] = lifecycle_states

    # Add Year and Month integer columns
    parsed_d = pd.to_datetime(panel_df["Date"], format="%Y-%m")
    panel_df["Year"] = parsed_d.dt.year.astype("Int64")
    panel_df["Month"] = parsed_d.dt.month.astype("Int64")

    # Format typed output table
    cols_order = [
        "Date", "Year", "Month", "SKU_ID", "Region", "Product_Family", "Business_Unit",
        "SKU_Description", "Customer_Name", "Sales_Channel", "is_observed",
        "Units_Sold", "reconstructed_demand", "demand_evidence", "Unit_Price_USD",
        "Revenue_USD", "Lifecycle_Status"
    ]
    # Keep any extra columns at the end
    existing_cols = [c for c in cols_order if c in panel_df.columns]
    panel_df = panel_df[existing_cols]

    return panel_df, anomalies
