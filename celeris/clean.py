"""Data cleaning, label normalization, deduplication, and quality logging for Celeris."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import pandas as pd


@dataclass
class QualityRecord:
    """Audit record for a data quality rule execution, repair, or deduplication."""
    file_name: str
    source_row_id: str
    planning_key: str
    field: str
    rule: str
    original_value: str
    cleaned_value: str
    decision: str
    reason: str
    linked_evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_history_transactions(
    history_df: pd.DataFrame,
    sku_master_df: Optional[pd.DataFrame] = None,
    file_name: str = "history.csv",
) -> Tuple[pd.DataFrame, List[QualityRecord]]:
    """Clean the raw historical ERP transactions.

    Rules applied:
    1. Authoritative Date: derive Year and Month from Date.
    2. Normalize string labels (strip whitespace, ensure standard casing).
    3. Latest-load deduplication: keep the latest Load_Timestamp for repeated Record_ID.
    4. Price and revenue reconciliation: check Units_Sold * Unit_Price_USD == Revenue_USD.
    """
    df = history_df.copy()
    quality_records: List[QualityRecord] = []

    # 1. Clean column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Date Authority: Parse Date and verify Year/Month
    raw_date_str = df["Date"].astype(str).str.strip()
    cleaned_date_str = raw_date_str.str.slice(0, 7)

    # Normalize YYYY-MM if full timestamp or day was provided
    format_diff = (raw_date_str != cleaned_date_str) & df["Date"].notna()
    for idx in df[format_diff].index:
        rec_id = str(df.loc[idx, "Record_ID"]) if "Record_ID" in df.columns else str(idx)
        pkey = f"{cleaned_date_str.loc[idx]}|{df.loc[idx, 'SKU_ID'] if 'SKU_ID' in df.columns else ''}|{df.loc[idx, 'Region'] if 'Region' in df.columns else ''}"
        quality_records.append(QualityRecord(
            file_name=file_name,
            source_row_id=rec_id,
            planning_key=pkey,
            field="Date",
            rule="date_format_normalization",
            original_value=str(raw_date_str.loc[idx]),
            cleaned_value=str(cleaned_date_str.loc[idx]),
            decision="normalized_to_monthly_grain",
            reason=f"Date string '{raw_date_str.loc[idx]}' normalized to authoritative monthly grain YYYY-MM",
            linked_evidence=f"Original={raw_date_str.loc[idx]}",
        ))

    parsed_date = pd.to_datetime(cleaned_date_str, format="%Y-%m", errors="coerce")
    valid_date_mask = parsed_date.notna()

    # Derived Year and Month
    derived_year = parsed_date.dt.year
    derived_month = parsed_date.dt.month

    for idx in df[valid_date_mask].index:
        rec_id = str(df.loc[idx, "Record_ID"]) if "Record_ID" in df.columns else str(idx)
        normalized_d = cleaned_date_str.loc[idx]
        pkey = f"{normalized_d}|{df.loc[idx, 'SKU_ID'] if 'SKU_ID' in df.columns else ''}|{df.loc[idx, 'Region'] if 'Region' in df.columns else ''}"
        curr_year = df.loc[idx, "Year"] if "Year" in df.columns else None
        curr_month = df.loc[idx, "Month"] if "Month" in df.columns else None
        d_year = int(derived_year.loc[idx])
        d_month = int(derived_month.loc[idx])

        if pd.notna(curr_year) and int(curr_year) != d_year:
            quality_records.append(QualityRecord(
                file_name=file_name,
                source_row_id=rec_id,
                planning_key=pkey,
                field="Year",
                rule="authoritative_date",
                original_value=str(curr_year),
                cleaned_value=str(d_year),
                decision="repaired_from_date",
                reason="Year in record conflicted with authoritative Date",
                linked_evidence=f"Date={normalized_d}",
            ))

        if pd.notna(curr_month) and int(curr_month) != d_month:
            quality_records.append(QualityRecord(
                file_name=file_name,
                source_row_id=rec_id,
                planning_key=pkey,
                field="Month",
                rule="authoritative_date",
                original_value=str(curr_month),
                cleaned_value=str(d_month),
                decision="repaired_from_date",
                reason="Month in record conflicted with authoritative Date",
                linked_evidence=f"Date={normalized_d}",
            ))

    df["Date"] = cleaned_date_str
    df["Year"] = derived_year.astype("Int64")
    df["Month"] = derived_month.astype("Int64")

    # 3. Label normalization
    str_cols = [
        "Product_Family", "Business_Unit", "SKU_ID", "SKU_Description",
        "Customer_Name", "Sales_Channel", "Region"
    ]
    for col in str_cols:
        if col in df.columns:
            cleaned_col = df[col].astype(str).str.strip()
            diff_mask = (df[col].astype(str) != cleaned_col) & df[col].notna()
            for idx in df[diff_mask].index:
                rec_id = str(df.loc[idx, "Record_ID"])
                pkey = f"{df.loc[idx, 'Date']}|{df.loc[idx, 'SKU_ID']}|{df.loc[idx, 'Region']}"
                quality_records.append(QualityRecord(
                    file_name=file_name,
                    source_row_id=rec_id,
                    planning_key=pkey,
                    field=col,
                    rule="label_whitespace_normalization",
                    original_value=str(df.loc[idx, col]),
                    cleaned_value=str(cleaned_col.loc[idx]),
                    decision="whitespace_stripped",
                    reason=f"Leading/trailing whitespace stripped from {col}",
                    linked_evidence=f"Record_ID={rec_id}",
                ))
            df[col] = cleaned_col

    # 4. Latest-load Deduplication on Record_ID
    if "Record_ID" in df.columns and "Load_Timestamp" in df.columns:
        df["_load_dt"] = pd.to_datetime(df["Load_Timestamp"], errors="coerce")
        # Identify duplicate Record_IDs
        dup_mask = df.duplicated(subset=["Record_ID"], keep=False)
        if dup_mask.any():
            dup_df = df[dup_mask]
            for rec_id, group in dup_df.groupby("Record_ID"):
                # Sort descending by load timestamp
                sorted_group = group.sort_values(by="_load_dt", ascending=False)
                kept_row = sorted_group.iloc[0]
                superseded_rows = sorted_group.iloc[1:]

                pkey = f"{kept_row.get('Date')}|{kept_row.get('SKU_ID')}|{kept_row.get('Region')}"
                for _, sup_row in superseded_rows.iterrows():
                    quality_records.append(QualityRecord(
                        file_name=file_name,
                        source_row_id=str(rec_id),
                        planning_key=pkey,
                        field="Record_ID",
                        rule="latest_load_dedup",
                        original_value=f"Units={sup_row['Units_Sold']}, Load={sup_row['Load_Timestamp']}",
                        cleaned_value=f"Units={kept_row['Units_Sold']}, Load={kept_row['Load_Timestamp']}",
                        decision="superseded_by_later_load",
                        reason=f"Repeated Record_ID resolved by latest Load_Timestamp ({kept_row['Load_Timestamp']} > {sup_row['Load_Timestamp']})",
                        linked_evidence=f"Kept_Timestamp={kept_row['Load_Timestamp']}",
                    ))

            # Keep only latest load row per Record_ID
            df = df.sort_values(by="_load_dt", ascending=False).drop_duplicates(subset=["Record_ID"], keep="first")

        df = df.drop(columns=["_load_dt"])

    # 5. Price & Revenue Reconciliation
    if "Units_Sold" in df.columns and "Unit_Price_USD" in df.columns and "Revenue_USD" in df.columns:
        df["Units_Sold"] = pd.to_numeric(df["Units_Sold"], errors="coerce")
        df["Unit_Price_USD"] = pd.to_numeric(df["Unit_Price_USD"], errors="coerce")
        df["Revenue_USD"] = pd.to_numeric(df["Revenue_USD"], errors="coerce")

        expected_rev = (df["Units_Sold"] * df["Unit_Price_USD"]).round(2)
        diff = (df["Revenue_USD"] - expected_rev).abs()
        mismatch_mask = diff > 0.05

        for idx in df[mismatch_mask].index:
            rec_id = str(df.loc[idx, "Record_ID"])
            pkey = f"{df.loc[idx, 'Date']}|{df.loc[idx, 'SKU_ID']}|{df.loc[idx, 'Region']}"
            units = df.loc[idx, "Units_Sold"]
            price = df.loc[idx, "Unit_Price_USD"]
            actual_rev = df.loc[idx, "Revenue_USD"]
            calc_rev = expected_rev.loc[idx]

            quality_records.append(QualityRecord(
                file_name=file_name,
                source_row_id=rec_id,
                planning_key=pkey,
                field="Revenue_USD",
                rule="revenue_price_reconciliation",
                original_value=str(actual_rev),
                cleaned_value=str(calc_rev),
                decision="reconciled_revenue",
                reason=f"Revenue mismatch: Units ({units}) x Price (${price:.2f}) = ${calc_rev:.2f} != reported ${actual_rev:.2f}",
                linked_evidence=f"Units_Sold={units}, Unit_Price_USD={price}",
            ))
            # Reconcile revenue to computed product
            df.loc[idx, "Revenue_USD"] = calc_rev

    # Sort deterministically
    sort_cols = [c for c in ["Date", "SKU_ID", "Region", "Record_ID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df, quality_records
