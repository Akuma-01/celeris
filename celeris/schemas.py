"""Schema definitions and contracts for Celeris dataset roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Tuple, Optional


class DatasetRole(str, Enum):
    """The seven expected dataset roles."""
    HISTORY = "history"
    FORECAST_ROWS = "forecast_rows"
    SKU_MASTER = "sku_master"
    SAMPLE_SUBMISSION = "sample_submission"
    MACRO_INDICATORS = "macro_indicators"
    STOCKOUTS = "stockouts"
    EVENTS_CALENDAR = "events_calendar"


@dataclass(frozen=True)
class DatasetSchema:
    """Contract definition for a dataset role."""
    role: DatasetRole
    display_name: str
    description: str
    required_identifying_columns: Set[str]
    distinguishing_columns: Set[str]
    forbidden_columns: Set[str]
    expected_full_columns: List[str]
    min_distinguishing_match: int = 1


SCHEMAS: dict[DatasetRole, DatasetSchema] = {
    DatasetRole.HISTORY: DatasetSchema(
        role=DatasetRole.HISTORY,
        display_name="Historical ERP Sales",
        description="Raw ERP extract of monthly sales per SKU-region (2019-01 -> 2026-10)",
        required_identifying_columns={"Record_ID", "Units_Sold", "Load_Timestamp"},
        distinguishing_columns={"Record_ID", "Units_Sold", "Load_Timestamp", "Unit_Price_USD", "Revenue_USD"},
        forbidden_columns={"Units_Sold_Forecast"},
        expected_full_columns=[
            "Record_ID", "Date", "Year", "Month", "Product_Family", "Business_Unit",
            "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region",
            "Units_Sold", "Unit_Price_USD", "Revenue_USD", "Load_Timestamp"
        ],
        min_distinguishing_match=3,
    ),
    DatasetRole.FORECAST_ROWS: DatasetSchema(
        role=DatasetRole.FORECAST_ROWS,
        display_name="Rows to Forecast (Test Keys)",
        description="Planning keys to forecast (2026-11 -> 2028-05, 19 months)",
        required_identifying_columns={"Date", "SKU_ID", "Region"},
        distinguishing_columns={"Customer_Name", "Sales_Channel", "Product_Family", "Business_Unit"},
        forbidden_columns={"Units_Sold", "Record_ID", "Load_Timestamp", "Units_Sold_Forecast"},
        expected_full_columns=[
            "Date", "Year", "Month", "Product_Family", "Business_Unit",
            "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region"
        ],
        min_distinguishing_match=2,
    ),
    DatasetRole.SKU_MASTER: DatasetSchema(
        role=DatasetRole.SKU_MASTER,
        display_name="SKU Master",
        description="Product catalog, lifecycle status, launch/EOL dates and replacements",
        required_identifying_columns={"SKU_ID"},
        distinguishing_columns={
            "Size_Class", "Powertrain_Application", "Launch_Date",
            "EOL_Date", "Replaces_SKU", "Lifecycle_Status"
        },
        forbidden_columns={"Units_Sold", "Record_ID", "Units_Sold_Forecast", "Date"},
        expected_full_columns=[
            "SKU_ID", "Product_Family", "Business_Unit", "SKU_Description",
            "Size_Class", "Powertrain_Application", "Launch_Date", "EOL_Date",
            "Replaces_SKU", "Lifecycle_Status"
        ],
        min_distinguishing_match=2,
    ),
    DatasetRole.SAMPLE_SUBMISSION: DatasetSchema(
        role=DatasetRole.SAMPLE_SUBMISSION,
        display_name="Sample Submission",
        description="Target submission format with Date, SKU_ID, Region, Units_Sold_Forecast",
        required_identifying_columns={"Date", "SKU_ID", "Region", "Units_Sold_Forecast"},
        distinguishing_columns={"Units_Sold_Forecast"},
        forbidden_columns={"Record_ID", "Units_Sold", "Load_Timestamp", "Customer_Name"},
        expected_full_columns=["Date", "SKU_ID", "Region", "Units_Sold_Forecast"],
        min_distinguishing_match=1,
    ),
    DatasetRole.MACRO_INDICATORS: DatasetSchema(
        role=DatasetRole.MACRO_INDICATORS,
        display_name="Macro Indicators",
        description="Monthly economic and industry indicators (2019-01 -> 2028-05)",
        required_identifying_columns={"Date"},
        distinguishing_columns={
            "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
            "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
            "Fuel_Price_Index", "EV_Adoption_Index"
        },
        forbidden_columns={"SKU_ID", "Region", "Record_ID", "Units_Sold"},
        expected_full_columns=[
            "Date", "Year", "Month", "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct",
            "Vehicle_Inventory_Days", "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index",
            "Industrial_Util_Rate_Pct", "Fuel_Price_Index", "EV_Adoption_Index"
        ],
        min_distinguishing_match=2,
    ),
    DatasetRole.STOCKOUTS: DatasetSchema(
        role=DatasetRole.STOCKOUTS,
        display_name="Stockout Ledger",
        description="Supply-constrained months with requested, shipped, backordered, and lost units",
        required_identifying_columns={"Date", "SKU_ID", "Region"},
        distinguishing_columns={
            "Requested_Units", "Carried_In_Units", "Shipped_Units",
            "Backordered_Units", "Lost_Units", "Fill_Rate_Pct", "Stockout_Days", "Reason"
        },
        forbidden_columns={"Record_ID", "Load_Timestamp", "Units_Sold_Forecast"},
        expected_full_columns=[
            "Date", "SKU_ID", "Region", "Requested_Units", "Carried_In_Units",
            "Shipped_Units", "Backordered_Units", "Lost_Units", "Fill_Rate_Pct",
            "Stockout_Days", "Reason"
        ],
        min_distinguishing_match=3,
    ),
    DatasetRole.EVENTS_CALENDAR: DatasetSchema(
        role=DatasetRole.EVENTS_CALENDAR,
        display_name="Events Calendar",
        description="Known business events, price changes, disruptions and holidays",
        required_identifying_columns={"Event_ID", "Start_Date"},
        distinguishing_columns={"Event_ID", "Scope_Type", "Scope_Value", "Event_Type", "Description"},
        forbidden_columns={"SKU_ID", "Units_Sold", "Record_ID", "Units_Sold_Forecast"},
        expected_full_columns=[
            "Event_ID", "Start_Date", "End_Date", "Scope_Type", "Scope_Value", "Event_Type", "Description"
        ],
        min_distinguishing_match=2,
    ),
}


def match_schema(columns: list[str]) -> Tuple[Optional[DatasetRole], float, list[str]]:
    """Match a list of CSV columns against all registered role schemas.
    
    Returns:
        (best_role, score, notes)
    """
    col_set = set(columns)
    best_role: Optional[DatasetRole] = None
    best_score = 0.0
    best_notes: list[str] = []

    for role, schema in SCHEMAS.items():
        # Check forbidden columns first
        forbidden_found = col_set.intersection(schema.forbidden_columns)
        if forbidden_found:
            continue

        # Check required identifying columns
        missing_required = schema.required_identifying_columns - col_set
        if missing_required:
            continue

        # Check distinguishing columns
        matched_dist = col_set.intersection(schema.distinguishing_columns)
        if len(matched_dist) < schema.min_distinguishing_match:
            continue

        # Compute overlap score against expected full columns
        expected_set = set(schema.expected_full_columns)
        overlap = len(col_set.intersection(expected_set))
        score = (len(matched_dist) * 2.0 + overlap) / (len(schema.distinguishing_columns) * 2.0 + len(expected_set))

        notes = [
            f"Matched {len(matched_dist)} distinguishing columns: {sorted(matched_dist)}",
            f"Total column overlap: {overlap}/{len(expected_set)}",
        ]

        if score > best_score:
            best_score = score
            best_role = role
            best_notes = notes

    return best_role, best_score, best_notes
