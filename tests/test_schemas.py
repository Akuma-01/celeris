"""Tests for schema contracts and role matching."""

from celeris.schemas import DatasetRole, SCHEMAS, match_schema


def test_all_seven_roles_registered():
    """Verify that all 7 expected roles have schema contracts defined."""
    assert len(DatasetRole) == 7
    for role in DatasetRole:
        assert role in SCHEMAS
        schema = SCHEMAS[role]
        assert schema.display_name
        assert schema.required_identifying_columns
        assert schema.expected_full_columns


def test_match_history_schema():
    """Verify historical ERP sales matching and forbidden checks."""
    cols = [
        "Record_ID", "Date", "Year", "Month", "Product_Family", "Business_Unit",
        "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region",
        "Units_Sold", "Unit_Price_USD", "Revenue_USD", "Load_Timestamp"
    ]
    role, score, _ = match_schema(cols)
    assert role == DatasetRole.HISTORY
    assert score > 0.8

    # Adding Units_Sold_Forecast should disqualify it from HISTORY
    cols_forbidden = cols + ["Units_Sold_Forecast"]
    role_forbidden, _, _ = match_schema(cols_forbidden)
    assert role_forbidden != DatasetRole.HISTORY


def test_match_forecast_rows_schema():
    """Verify forecast rows schema matches test keys without target."""
    cols = [
        "Date", "Year", "Month", "Product_Family", "Business_Unit",
        "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region"
    ]
    role, score, _ = match_schema(cols)
    assert role == DatasetRole.FORECAST_ROWS

    # Having Units_Sold should disqualify it from FORECAST_ROWS
    cols_with_target = cols + ["Units_Sold"]
    role_with_target, _, _ = match_schema(cols_with_target)
    assert role_with_target != DatasetRole.FORECAST_ROWS


def test_match_sku_master_schema():
    """Verify SKU master schema matching."""
    cols = [
        "SKU_ID", "Product_Family", "Business_Unit", "SKU_Description",
        "Size_Class", "Powertrain_Application", "Launch_Date", "EOL_Date",
        "Replaces_SKU", "Lifecycle_Status"
    ]
    role, score, _ = match_schema(cols)
    assert role == DatasetRole.SKU_MASTER


def test_match_stockouts_schema():
    """Verify stockouts schema matching."""
    cols = [
        "Date", "SKU_ID", "Region", "Requested_Units", "Carried_In_Units",
        "Shipped_Units", "Backordered_Units", "Lost_Units", "Fill_Rate_Pct",
        "Stockout_Days", "Reason"
    ]
    role, score, _ = match_schema(cols)
    assert role == DatasetRole.STOCKOUTS
