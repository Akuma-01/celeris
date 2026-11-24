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

    # Vectorized conflict detection for Year and Month
    if "Year" in df.columns:
        year_num = pd.to_numeric(df["Year"], errors="coerce")
        year_diff_mask = valid_date_mask & year_num.notna() & (year_num != derived_year)
        for idx in df[year_diff_mask].index:
            rec_id = str(df.loc[idx, "Record_ID"]) if "Record_ID" in df.columns else str(idx)
            normalized_d = cleaned_date_str.loc[idx]
            pkey = f"{normalized_d}|{df.loc[idx, 'SKU_ID'] if 'SKU_ID' in df.columns else ''}|{df.loc[idx, 'Region'] if 'Region' in df.columns else ''}"
            curr_year = df.loc[idx, "Year"]
            d_year = int(derived_year.loc[idx])
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

    if "Month" in df.columns:
        month_num = pd.to_numeric(df["Month"], errors="coerce")
        month_diff_mask = valid_date_mask & month_num.notna() & (month_num != derived_month)
        for idx in df[month_diff_mask].index:
            rec_id = str(df.loc[idx, "Record_ID"]) if "Record_ID" in df.columns else str(idx)
            normalized_d = cleaned_date_str.loc[idx]
            pkey = f"{normalized_d}|{df.loc[idx, 'SKU_ID'] if 'SKU_ID' in df.columns else ''}|{df.loc[idx, 'Region'] if 'Region' in df.columns else ''}"
            curr_month = df.loc[idx, "Month"]
            d_month = int(derived_month.loc[idx])
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
    REGION_MAPPING = {
        "india": "India",
        "eu": "Europe",
        "europe": "Europe",
        "north america": "North America",
        "n. america": "North America",
        "asia pacific": "Asia Pacific",
        "asia-pacific": "Asia Pacific",
        "apac": "Asia Pacific",
        "latin america": "Latin America",
        "latin-america": "Latin America",
        "latam": "Latin America",
        "middle east & africa": "Middle East & Africa",
        "middle east and africa": "Middle East & Africa",
        "me&a": "Middle East & Africa",
        "mea": "Middle East & Africa",
    }

    CHANNEL_MAPPING = {
        "direct oem": "Direct OEM",
        "direct-oem": "Direct OEM",
        "oem direct": "Direct OEM",
        "tier-1 distributor": "Tier-1 Distributor",
        "tier 1 distributor": "Tier-1 Distributor",
        "t1 distributor": "Tier-1 Distributor",
        "aftermarket": "Aftermarket",
        "after-market": "Aftermarket",
        "export": "Export",
    }

    # Normalize Region variants
    if "Region" in df.columns:
        raw_reg = df["Region"].astype(str).str.strip()
        mapped_reg = raw_reg.str.lower().map(REGION_MAPPING)

        # Log region standardizations
        std_mask = mapped_reg.notna() & (df["Region"] != mapped_reg)
        for idx in df[std_mask].head(100).index:  # sample log to prevent millions of records
            rec_id = str(df.loc[idx, "Record_ID"]) if "Record_ID" in df.columns else str(idx)
            pkey = f"{df.loc[idx, 'Date']}|{df.loc[idx, 'SKU_ID'] if 'SKU_ID' in df.columns else ''}|{mapped_reg.loc[idx]}"
            quality_records.append(QualityRecord(
                file_name=file_name,
                source_row_id=rec_id,
                planning_key=pkey,
                field="Region",
                rule="region_label_standardization",
                original_value=str(df.loc[idx, "Region"]),
                cleaned_value=str(mapped_reg.loc[idx]),
                decision="standardized_to_canonical_region",
                reason=f"Standardized regional variant '{df.loc[idx, 'Region']}' to '{mapped_reg.loc[idx]}'",
                linked_evidence=f"Original={df.loc[idx, 'Region']}",
            ))

        # Quarantine unresolvable/NaN regions
        null_reg_mask = df["Region"].isna() | mapped_reg.isna()
        if null_reg_mask.any():
            for idx in df[null_reg_mask].head(50).index:
                rec_id = str(df.loc[idx, "Record_ID"]) if "Record_ID" in df.columns else str(idx)
                quality_records.append(QualityRecord(
                    file_name=file_name,
                    source_row_id=rec_id,
                    planning_key=f"{df.loc[idx, 'Date']}|{df.loc[idx, 'SKU_ID'] if 'SKU_ID' in df.columns else ''}|UNKNOWN",
                    field="Region",
                    rule="quarantine_unresolvable_region",
                    original_value=str(df.loc[idx, "Region"]),
                    cleaned_value="QUARANTINED",
                    decision="quarantined_from_planning_panel",
                    reason="Transaction missing identifiable sales region; quarantined per rule 1",
                    linked_evidence=f"Units_Sold={df.loc[idx, 'Units_Sold'] if 'Units_Sold' in df.columns else ''}",
                ))
            # Keep only rows with valid mapped regions
            df = df[~null_reg_mask].copy()
            mapped_reg = mapped_reg[~null_reg_mask]

        df["Region"] = mapped_reg

    # Normalize Sales_Channel variants
    if "Sales_Channel" in df.columns:
        raw_chan = df["Sales_Channel"].astype(str).str.strip()
        mapped_chan = raw_chan.str.lower().map(CHANNEL_MAPPING)
        valid_chan_mask = mapped_chan.notna()
        df.loc[valid_chan_mask, "Sales_Channel"] = mapped_chan[valid_chan_mask]

    # Impute missing Product_Family / Business_Unit from SKU Master
    if sku_master_df is not None and not sku_master_df.empty and "SKU_ID" in df.columns:
        if "Product_Family" in sku_master_df.columns:
            fam_map = dict(zip(sku_master_df["SKU_ID"].astype(str).str.strip(), sku_master_df["Product_Family"]))
            fam_na_mask = df["Product_Family"].isna() | (df["Product_Family"] == "") | (df["Product_Family"] == "nan")
            df.loc[fam_na_mask, "Product_Family"] = df.loc[fam_na_mask, "SKU_ID"].map(fam_map)
        if "Business_Unit" in sku_master_df.columns:
            bu_map = dict(zip(sku_master_df["SKU_ID"].astype(str).str.strip(), sku_master_df["Business_Unit"]))
            bu_na_mask = df["Business_Unit"].isna() | (df["Business_Unit"] == "") | (df["Business_Unit"] == "nan")
            df.loc[bu_na_mask, "Business_Unit"] = df.loc[bu_na_mask, "SKU_ID"].map(bu_map)

    # General whitespace stripping
    str_cols = [
        "Product_Family", "Business_Unit", "SKU_ID", "SKU_Description",
        "Customer_Name", "Sales_Channel", "Region"
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

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
