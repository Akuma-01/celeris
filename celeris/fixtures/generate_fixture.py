"""Generate deterministic fixture data for testing Celeris pipeline end-to-end."""

from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd


def generate_fixture_data(out_dir: Path) -> dict[str, Path]:
    """Generate all 7 fixture CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    # 1. sku_master.csv
    sku_master_rows = [
        {
            "SKU_ID": "BSS-001",
            "Product_Family": "Brake & Safety Systems",
            "Business_Unit": "Safety & Compliance",
            "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
            "Size_Class": "Industrial 100",
            "Powertrain_Application": "Universal",
            "Launch_Date": "2018-05",
            "EOL_Date": "",
            "Replaces_SKU": "",
            "Lifecycle_Status": "Active",
        },
        {
            "SKU_ID": "HYD-101",
            "Product_Family": "Hydraulic Systems",
            "Business_Unit": "Fluid Power",
            "SKU_Description": "Electro-Hydraulic Actuator 40 kN",
            "Size_Class": "Hydraulic 40",
            "Powertrain_Application": "ICE",
            "Launch_Date": "2017-01",
            "EOL_Date": "2024-12",
            "Replaces_SKU": "",
            "Lifecycle_Status": "Discontinued",
        },
        {
            "SKU_ID": "HYD-102",
            "Product_Family": "Hydraulic Systems",
            "Business_Unit": "Fluid Power",
            "SKU_Description": "Electro-Hydraulic Actuator NextGen 40 kN",
            "Size_Class": "Hydraulic 40",
            "Powertrain_Application": "Universal",
            "Launch_Date": "2025-01",
            "EOL_Date": "",
            "Replaces_SKU": "HYD-101",
            "Lifecycle_Status": "Active",
        },
        {
            "SKU_ID": "PWR-201",
            "Product_Family": "Powertrain Components",
            "Business_Unit": "Propulsion",
            "SKU_Description": "Brushless Stator Core Compact 12",
            "Size_Class": "Compact 12",
            "Powertrain_Application": "EV",
            "Launch_Date": "2019-01",
            "EOL_Date": "",
            "Replaces_SKU": "",
            "Lifecycle_Status": "Active",
        },
    ]
    sku_master_path = out_dir / "sku_master.csv"
    pd.DataFrame(sku_master_rows).to_csv(sku_master_path, index=False)
    generated_files["sku_master"] = sku_master_path

    # Helper for dates
    all_hist_dates = pd.date_range("2019-01-01", "2026-10-01", freq="MS").strftime("%Y-%m").tolist()
    all_forecast_dates = pd.date_range("2026-11-01", "2028-05-01", freq="MS").strftime("%Y-%m").tolist()
    all_dates = all_hist_dates + all_forecast_dates

    # 2. macro_indicators.csv
    macro_rows = []
    base_pmi = 52.0
    for i, d in enumerate(all_dates):
        year, month = d.split("-")
        macro_rows.append({
            "Date": d,
            "Year": int(year),
            "Month": int(month),
            "ISM_Manufacturing_PMI": round(base_pmi + (1.5 if int(month) % 2 == 0 else -1.2) + (0.5 if int(year) > 2022 else -0.5), 1),
            "Auto_Production_YoY_Pct": round(0.02 + 0.01 * (int(month) % 3), 3),
            "Vehicle_Inventory_Days": round(65.0 - 0.5 * (int(month) % 5), 1),
            "Raw_Material_Cost_Index": round(95.0 + 0.8 * (i // 6), 1),
            "New_Vehicle_Orders_Index": round(100.0 + 1.2 * (int(month) % 4), 1),
            "Industrial_Util_Rate_Pct": round(78.0 + 0.3 * (int(month) % 6), 1),
            "Fuel_Price_Index": round(80.0 + 0.5 * (i // 4), 1),
            "EV_Adoption_Index": round(20.0 + 0.7 * i, 1),
        })
    macro_path = out_dir / "macro_indicators.csv"
    pd.DataFrame(macro_rows).to_csv(macro_path, index=False)
    generated_files["macro_indicators"] = macro_path

    # 3. events_calendar.csv
    event_rows = [
        {"Event_ID": "EV-001", "Start_Date": "2019-01", "End_Date": "2019-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2%"},
        {"Event_ID": "EV-002", "Start_Date": "2020-01", "End_Date": "2020-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2.5%"},
        {"Event_ID": "EV-003", "Start_Date": "2021-01", "End_Date": "2021-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 3%"},
        {"Event_ID": "EV-004", "Start_Date": "2022-01", "End_Date": "2022-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 4%"},
        {"Event_ID": "EV-005", "Start_Date": "2023-01", "End_Date": "2023-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 3%"},
        {"Event_ID": "EV-006", "Start_Date": "2023-06", "End_Date": "2023-06", "Scope_Type": "Region", "Scope_Value": "North America", "Event_Type": "Supply Constraint", "Description": "Logistics delay at port affecting brake shipments"},
        {"Event_ID": "EV-007", "Start_Date": "2024-01", "End_Date": "2024-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2.5%"},
        {"Event_ID": "EV-008", "Start_Date": "2025-01", "End_Date": "2025-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2.0%"},
        {"Event_ID": "EV-009", "Start_Date": "2026-01", "End_Date": "2026-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2.0%"},
        {"Event_ID": "EV-010", "Start_Date": "2027-01", "End_Date": "2027-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2.0%"},
        {"Event_ID": "EV-011", "Start_Date": "2028-01", "End_Date": "2028-01", "Scope_Type": "Global", "Scope_Value": "All", "Event_Type": "Price Change", "Description": "Annual list-price increase 2.0%"},
    ]
    events_path = out_dir / "events_calendar.csv"
    pd.DataFrame(event_rows).to_csv(events_path, index=False)
    generated_files["events_calendar"] = events_path

    # 4. inventory_stockouts.csv (The June 2023 stockout ledger row)
    stockouts_rows = [
        {
            "Date": "2023-06",
            "SKU_ID": "BSS-001",
            "Region": "North America",
            "Requested_Units": 100,
            "Carried_In_Units": 0,
            "Shipped_Units": 70,
            "Backordered_Units": 20,
            "Lost_Units": 10,
            "Fill_Rate_Pct": 70.0,
            "Stockout_Days": 12,
            "Reason": "Logistics delay at port",
        }
    ]
    stockouts_path = out_dir / "inventory_stockouts.csv"
    pd.DataFrame(stockouts_rows).to_csv(stockouts_path, index=False)
    generated_files["stockouts"] = stockouts_path

    # 5. history.csv (Historical ERP extract)
    history_rows = []
    rec_counter = 1000

    for d in all_hist_dates:
        year, month = d.split("-")
        y_int, m_int = int(year), int(month)

        # SKU 1: BSS-001 (North America, Solaris Automotive, Direct OEM)
        # Price: 250
        price_bss = 250.0 + (y_int - 2019) * 5.0

        if d == "2020-05":
            # DUPLICATE LOAD CASE:
            # First load: 45 units
            rec_counter += 1
            history_rows.append({
                "Record_ID": "REC-BSS-202005",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Brake & Safety Systems",
                "Business_Unit": "Safety & Compliance",
                "SKU_ID": "BSS-001",
                "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
                "Customer_Name": "Solaris Automotive",
                "Sales_Channel": "Direct OEM",
                "Region": "North America",
                "Units_Sold": 45,
                "Unit_Price_USD": price_bss,
                "Revenue_USD": round(45 * price_bss, 2),
                "Load_Timestamp": "2020-05-10T08:30:00",
            })
            # Second load: 50 units (LATEST load, should supersede!)
            history_rows.append({
                "Record_ID": "REC-BSS-202005",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Brake & Safety Systems",
                "Business_Unit": "Safety & Compliance",
                "SKU_ID": "BSS-001",
                "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
                "Customer_Name": "Solaris Automotive",
                "Sales_Channel": "Direct OEM",
                "Region": "North America",
                "Units_Sold": 50,
                "Unit_Price_USD": price_bss,
                "Revenue_USD": round(50 * price_bss, 2),
                "Load_Timestamp": "2020-05-15T18:00:00",
            })
        elif d == "2021-03":
            # GENUINE ZERO ORDER CASE
            rec_counter += 1
            history_rows.append({
                "Record_ID": f"REC-{rec_counter}",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Brake & Safety Systems",
                "Business_Unit": "Safety & Compliance",
                "SKU_ID": "BSS-001",
                "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
                "Customer_Name": "Solaris Automotive",
                "Sales_Channel": "Direct OEM",
                "Region": "North America",
                "Units_Sold": 0,
                "Unit_Price_USD": price_bss,
                "Revenue_USD": 0.0,
                "Load_Timestamp": f"{d}-28T12:00:00",
            })
        elif d == "2021-07":
            # MISSING MONTH CASE: Intentionally omit row!
            pass
        elif d == "2023-06":
            # STOCKOUT MONTH CASE: Shipped 70 units
            rec_counter += 1
            history_rows.append({
                "Record_ID": f"REC-{rec_counter}",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Brake & Safety Systems",
                "Business_Unit": "Safety & Compliance",
                "SKU_ID": "BSS-001",
                "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
                "Customer_Name": "Solaris Automotive",
                "Sales_Channel": "Direct OEM",
                "Region": "North America",
                "Units_Sold": 70,
                "Unit_Price_USD": price_bss,
                "Revenue_USD": round(70 * price_bss, 2),
                "Load_Timestamp": f"{d}-28T12:00:00",
            })
        elif d == "2023-07":
            # BACKLOG CARRY REBOUND MONTH CASE: Shipped 110 units (20 carry + 90 true)
            rec_counter += 1
            history_rows.append({
                "Record_ID": f"REC-{rec_counter}",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Brake & Safety Systems",
                "Business_Unit": "Safety & Compliance",
                "SKU_ID": "BSS-001",
                "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
                "Customer_Name": "Solaris Automotive",
                "Sales_Channel": "Direct OEM",
                "Region": "North America",
                "Units_Sold": 110,
                "Unit_Price_USD": price_bss,
                "Revenue_USD": round(110 * price_bss, 2),
                "Load_Timestamp": f"{d}-28T12:00:00",
            })
        else:
            # Normal steady demand with slight seasonality
            base_units = 60 + (m_int % 4) * 5 + (y_int - 2019) * 2
            rec_counter += 1
            history_rows.append({
                "Record_ID": f"REC-{rec_counter}",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Brake & Safety Systems",
                "Business_Unit": "Safety & Compliance",
                "SKU_ID": "BSS-001",
                "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
                "Customer_Name": "Solaris Automotive",
                "Sales_Channel": "Direct OEM",
                "Region": "North America",
                "Units_Sold": base_units,
                "Unit_Price_USD": price_bss,
                "Revenue_USD": round(base_units * price_bss, 2),
                "Load_Timestamp": f"{d}-28T12:00:00",
            })

        # SKU 2: HYD-101 (Europe, Apex Motion GmbH, Tier-1 Distributor)
        # EOL: 2024-12 -> Has sales only up to 2024-12!
        if d <= "2024-12":
            price_hyd1 = 420.0 + (y_int - 2019) * 8.0
            # Intermittent pattern: every 2nd or 3rd month
            if m_int % 2 == 1:
                units = 30 + (m_int * 3)
                rec_counter += 1
                history_rows.append({
                    "Record_ID": f"REC-{rec_counter}",
                    "Date": d,
                    "Year": y_int,
                    "Month": m_int,
                    "Product_Family": "Hydraulic Systems",
                    "Business_Unit": "Fluid Power",
                    "SKU_ID": "HYD-101",
                    "SKU_Description": "Electro-Hydraulic Actuator 40 kN",
                    "Customer_Name": "Apex Motion GmbH",
                    "Sales_Channel": "Tier-1 Distributor",
                    "Region": "Europe",
                    "Units_Sold": units,
                    "Unit_Price_USD": price_hyd1,
                    "Revenue_USD": round(units * price_hyd1, 2),
                    "Load_Timestamp": f"{d}-28T12:00:00",
                })
            else:
                # Intermittent 0
                rec_counter += 1
                history_rows.append({
                    "Record_ID": f"REC-{rec_counter}",
                    "Date": d,
                    "Year": y_int,
                    "Month": m_int,
                    "Product_Family": "Hydraulic Systems",
                    "Business_Unit": "Fluid Power",
                    "SKU_ID": "HYD-101",
                    "SKU_Description": "Electro-Hydraulic Actuator 40 kN",
                    "Customer_Name": "Apex Motion GmbH",
                    "Sales_Channel": "Tier-1 Distributor",
                    "Region": "Europe",
                    "Units_Sold": 0,
                    "Unit_Price_USD": price_hyd1,
                    "Revenue_USD": 0.0,
                    "Load_Timestamp": f"{d}-28T12:00:00",
                })

        # SKU 3: HYD-102 (Europe, Apex Motion GmbH, Tier-1 Distributor)
        # Launch: 2025-01, Replaces HYD-101
        if d >= "2025-01":
            price_hyd2 = 450.0 + (y_int - 2025) * 10.0
            units = 35 + (m_int * 2)
            rec_counter += 1
            history_rows.append({
                "Record_ID": f"REC-{rec_counter}",
                "Date": d,
                "Year": y_int,
                "Month": m_int,
                "Product_Family": "Hydraulic Systems",
                "Business_Unit": "Fluid Power",
                "SKU_ID": "HYD-102",
                "SKU_Description": "Electro-Hydraulic Actuator NextGen 40 kN",
                "Customer_Name": "Apex Motion GmbH",
                "Sales_Channel": "Tier-1 Distributor",
                "Region": "Europe",
                "Units_Sold": units,
                "Unit_Price_USD": price_hyd2,
                "Revenue_USD": round(units * price_hyd2, 2),
                "Load_Timestamp": f"{d}-28T12:00:00",
            })

        # SKU 4: PWR-201 (North America, Veloce Motors, Direct OEM)
        # Steady growing series
        price_pwr = 180.0 + (y_int - 2019) * 4.0
        units_pwr = 80 + (y_int - 2019) * 10 + (m_int % 6) * 3
        rec_counter += 1
        history_rows.append({
            "Record_ID": f"REC-{rec_counter}",
            "Date": d,
            "Year": y_int,
            "Month": m_int,
            "Product_Family": "Powertrain Components",
            "Business_Unit": "Propulsion",
            "SKU_ID": "PWR-201",
            "SKU_Description": "Brushless Stator Core Compact 12",
            "Customer_Name": "Veloce Motors",
            "Sales_Channel": "Direct OEM",
            "Region": "North America",
            "Units_Sold": units_pwr,
            "Unit_Price_USD": price_pwr,
            "Revenue_USD": round(units_pwr * price_pwr, 2),
            "Load_Timestamp": f"{d}-28T12:00:00",
        })

    hist_path = out_dir / "history.csv"
    pd.DataFrame(history_rows).to_csv(hist_path, index=False)
    generated_files["history"] = hist_path

    # 6. forecast_rows.csv (Test set planning keys: Nov 2026 -> May 2028, 19 months)
    forecast_rows = []
    for d in all_forecast_dates:
        year, month = d.split("-")
        y_int, m_int = int(year), int(month)

        # BSS-001
        forecast_rows.append({
            "Date": d,
            "Year": y_int,
            "Month": m_int,
            "Product_Family": "Brake & Safety Systems",
            "Business_Unit": "Safety & Compliance",
            "SKU_ID": "BSS-001",
            "SKU_Description": "Industrial 100 mm Disc Brake Caliper Assembly",
            "Customer_Name": "Solaris Automotive",
            "Sales_Channel": "Direct OEM",
            "Region": "North America",
        })

        # HYD-102 (Active successor)
        forecast_rows.append({
            "Date": d,
            "Year": y_int,
            "Month": m_int,
            "Product_Family": "Hydraulic Systems",
            "Business_Unit": "Fluid Power",
            "SKU_ID": "HYD-102",
            "SKU_Description": "Electro-Hydraulic Actuator NextGen 40 kN",
            "Customer_Name": "Apex Motion GmbH",
            "Sales_Channel": "Tier-1 Distributor",
            "Region": "Europe",
        })

        # PWR-201
        forecast_rows.append({
            "Date": d,
            "Year": y_int,
            "Month": m_int,
            "Product_Family": "Powertrain Components",
            "Business_Unit": "Propulsion",
            "SKU_ID": "PWR-201",
            "SKU_Description": "Brushless Stator Core Compact 12",
            "Customer_Name": "Veloce Motors",
            "Sales_Channel": "Direct OEM",
            "Region": "North America",
        })

        # HYD-101 (EOL SKU - included in test to verify lifecycle 0 enforcement)
        forecast_rows.append({
            "Date": d,
            "Year": y_int,
            "Month": m_int,
            "Product_Family": "Hydraulic Systems",
            "Business_Unit": "Fluid Power",
            "SKU_ID": "HYD-101",
            "SKU_Description": "Electro-Hydraulic Actuator 40 kN",
            "Customer_Name": "Apex Motion GmbH",
            "Sales_Channel": "Tier-1 Distributor",
            "Region": "Europe",
        })

    forecast_path = out_dir / "forecast_rows.csv"
    pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)
    generated_files["forecast_rows"] = forecast_path

    # 7. sample_submission.csv
    sample_sub_rows = [
        {"Date": r["Date"], "SKU_ID": r["SKU_ID"], "Region": r["Region"], "Units_Sold_Forecast": 50 if r["SKU_ID"] != "HYD-101" else 0}
        for r in forecast_rows
    ]
    sub_path = out_dir / "sample_submission.csv"
    pd.DataFrame(sample_sub_rows).to_csv(sub_path, index=False)
    generated_files["sample_submission"] = sub_path

    return generated_files


if __name__ == "__main__":
    out = Path("fixtures/data")
    files = generate_fixture_data(out)
    print(f"Generated {len(files)} fixture files in {out}:")
    for role, path in files.items():
        print(f"  {role:<20}: {path.name}")
