"""Data cleaning, label normalization, deduplication, and quality logging for Celeris."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd


REGION_MAPPING: Dict[str, str] = {
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

CHANNEL_MAPPING: Dict[str, str] = {
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

_SKU_PATTERN = re.compile(r"^([A-Z]+)[-_\s]?(\d+)$")


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


def _log_sample_records(
    quality_records: List[QualityRecord],
    df: pd.DataFrame,
    mask: pd.Series,
    file_name: str,
    field: str,
    rule: str,
    original_series: pd.Series,
    cleaned_series: pd.Series,
    decision: str,
    reason_fn,
    evidence_fn,
    max_samples: int = 200,
) -> None:
    """Append bounded audit QualityRecords for vectorized repair masks."""
    if not mask.any():
        return
    has_rec = "Record_ID" in df.columns
    has_sku = "SKU_ID" in df.columns
    has_reg = "Region" in df.columns
    has_date = "Date" in df.columns

    for idx in df[mask].head(max_samples).index:
        rec_id = str(df.loc[idx, "Record_ID"]) if has_rec else str(idx)
        d_val = df.loc[idx, "Date"] if has_date else ""
        s_val = df.loc[idx, "SKU_ID"] if has_sku else ""
        r_val = df.loc[idx, "Region"] if has_reg else ""
        pkey = f"{d_val}|{s_val}|{r_val}"
        orig = original_series.loc[idx]
        clean = cleaned_series.loc[idx]
        quality_records.append(
            QualityRecord(
                file_name=file_name,
                source_row_id=rec_id,
                planning_key=pkey,
                field=field,
                rule=rule,
                original_value=str(orig),
                cleaned_value=str(clean),
                decision=decision,
                reason=reason_fn(orig, clean),
                linked_evidence=evidence_fn(idx),
            )
        )


def normalize_sku_id(raw_sku: object, canonical_skus: Optional[set[str]] = None) -> str:
    """Normalize SKU identifiers like 'BSS-10', 'ELA062', or 'bss-001' to canonical 'PREFIX-NNN'."""
    s = str(raw_sku).strip().upper()
    if canonical_skus and s in canonical_skus:
        return s
    match = _SKU_PATTERN.match(s)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):03d}"
    return s


def _normalize_dates(
    df: pd.DataFrame,
    file_name: str,
    quality_records: List[QualityRecord],
) -> pd.DataFrame:
    """Parse multi-format Date strings to canonical YYYY-MM and reconcile Year/Month."""
    if df.empty or "Date" not in df.columns:
        return df

    raw_date_str = df["Date"].astype(str).str.strip()

    # Fast path for canonical YYYY-MM prefix when matching standard ISO pattern
    iso_mask = raw_date_str.str.match(r"^\d{4}-\d{2}$")
    parsed_date = pd.to_datetime(raw_date_str.where(iso_mask), format="%Y-%m", errors="coerce")

    # Fallback mixed parser for '2024/02', 'Apr 2024', '01-2022', '2020-05-15', etc.
    need_mixed = parsed_date.isna() & df["Date"].notna()
    if need_mixed.any():
        parsed_date.loc[need_mixed] = pd.to_datetime(
            raw_date_str.loc[need_mixed], format="mixed", errors="coerce"
        )

    # Guard year digit-transposition typos (e.g., '2062-MM' when Year == 2026)
    if "Year" in df.columns:
        yr_num = pd.to_numeric(df["Year"], errors="coerce")
        typo_mask = parsed_date.notna() & (parsed_date.dt.year == 2062) & (yr_num == 2026)
        if typo_mask.any():
            parsed_date.loc[typo_mask] = parsed_date.loc[typo_mask].apply(lambda dt: dt.replace(year=2026))

    valid_mask = parsed_date.notna()
    cleaned_date_str = raw_date_str.copy()
    cleaned_date_str.loc[valid_mask] = parsed_date.loc[valid_mask].dt.strftime("%Y-%m")

    derived_year = parsed_date.dt.year
    derived_month = parsed_date.dt.month

    # Log non-ISO date format repairs
    format_diff = valid_mask & (raw_date_str != cleaned_date_str)
    _log_sample_records(
        quality_records,
        df,
        format_diff,
        file_name,
        field="Date",
        rule="date_format_normalization",
        original_series=raw_date_str,
        cleaned_series=cleaned_date_str,
        decision="normalized_to_monthly_grain",
        reason_fn=lambda o, c: f"Date string '{o}' normalized to authoritative monthly grain YYYY-MM",
        evidence_fn=lambda idx: f"Original={raw_date_str.loc[idx]}",
        max_samples=2000,
    )

    if "Year" in df.columns:
        year_num = pd.to_numeric(df["Year"], errors="coerce")
        year_diff = valid_mask & year_num.notna() & (year_num != derived_year)
        _log_sample_records(
            quality_records,
            df,
            year_diff,
            file_name,
            field="Year",
            rule="authoritative_date",
            original_series=df["Year"],
            cleaned_series=derived_year.astype("Int64"),
            decision="repaired_from_date",
            reason_fn=lambda o, c: "Year in record conflicted with authoritative Date",
            evidence_fn=lambda idx: f"Date={cleaned_date_str.loc[idx]}",
        )

    if "Month" in df.columns:
        month_num = pd.to_numeric(df["Month"], errors="coerce")
        month_diff = valid_mask & month_num.notna() & (month_num != derived_month)
        _log_sample_records(
            quality_records,
            df,
            month_diff,
            file_name,
            field="Month",
            rule="authoritative_date",
            original_series=df["Month"],
            cleaned_series=derived_month.astype("Int64"),
            decision="repaired_from_date",
            reason_fn=lambda o, c: "Month in record conflicted with authoritative Date",
            evidence_fn=lambda idx: f"Date={cleaned_date_str.loc[idx]}",
        )

    df["Date"] = cleaned_date_str
    df["Year"] = derived_year.astype("Int64")
    df["Month"] = derived_month.astype("Int64")
    return df


def _normalize_skus_regions_channels(
    df: pd.DataFrame,
    sku_master_df: Optional[pd.DataFrame],
    file_name: str,
    quality_records: List[QualityRecord],
) -> pd.DataFrame:
    """Canonicalize SKU_IDs, Regions, Sales_Channels, and impute missing product hierarchy."""
    if df.empty:
        return df

    canonical_skus: Optional[set[str]] = None
    if sku_master_df is not None and not sku_master_df.empty and "SKU_ID" in sku_master_df.columns:
        canonical_skus = set(sku_master_df["SKU_ID"].astype(str).str.strip().str.upper())

    if "SKU_ID" in df.columns:
        raw_sku = df["SKU_ID"].astype(str).str.strip()
        norm_sku = raw_sku.map(lambda val: normalize_sku_id(val, canonical_skus))
        sku_diff = df["SKU_ID"].notna() & (raw_sku != norm_sku)
        _log_sample_records(
            quality_records,
            df,
            sku_diff,
            file_name,
            field="SKU_ID",
            rule="sku_id_normalization",
            original_series=raw_sku,
            cleaned_series=norm_sku,
            decision="standardized_to_canonical_sku",
            reason_fn=lambda o, c: f"SKU_ID variant '{o}' standardized to canonical '{c}'",
            evidence_fn=lambda idx: f"Original={raw_sku.loc[idx]}",
        )
        df["SKU_ID"] = norm_sku

    if "Region" in df.columns:
        raw_reg = df["Region"].astype(str).str.strip()
        mapped_reg = raw_reg.str.lower().map(REGION_MAPPING)

        std_mask = mapped_reg.notna() & (df["Region"] != mapped_reg)
        _log_sample_records(
            quality_records,
            df,
            std_mask,
            file_name,
            field="Region",
            rule="region_label_standardization",
            original_series=df["Region"],
            cleaned_series=mapped_reg,
            decision="standardized_to_canonical_region",
            reason_fn=lambda o, c: f"Standardized regional variant '{o}' to '{c}'",
            evidence_fn=lambda idx: f"Original={df.loc[idx, 'Region']}",
            max_samples=100,
        )

        null_reg_mask = df["Region"].isna() | mapped_reg.isna()
        if null_reg_mask.any():
            _log_sample_records(
                quality_records,
                df,
                null_reg_mask,
                file_name,
                field="Region",
                rule="quarantine_unresolvable_region",
                original_series=df["Region"],
                cleaned_series=pd.Series("QUARANTINED", index=df.index),
                decision="quarantined_from_planning_panel",
                reason_fn=lambda o, c: "Transaction missing identifiable sales region; quarantined per rule 1",
                evidence_fn=lambda idx: f"Units_Sold={df.loc[idx, 'Units_Sold'] if 'Units_Sold' in df.columns else ''}",
                max_samples=50,
            )
            df = df[~null_reg_mask].copy()
            mapped_reg = mapped_reg[~null_reg_mask]

        df["Region"] = mapped_reg

    if "Sales_Channel" in df.columns:
        raw_chan = df["Sales_Channel"].astype(str).str.strip()
        mapped_chan = raw_chan.str.lower().map(CHANNEL_MAPPING)
        valid_chan_mask = mapped_chan.notna()
        df.loc[valid_chan_mask, "Sales_Channel"] = mapped_chan[valid_chan_mask]

    if sku_master_df is not None and not sku_master_df.empty and "SKU_ID" in df.columns:
        if "Product_Family" in sku_master_df.columns and "Product_Family" in df.columns:
            fam_map = dict(zip(sku_master_df["SKU_ID"].astype(str).str.strip(), sku_master_df["Product_Family"]))
            fam_na = df["Product_Family"].isna() | (df["Product_Family"] == "") | (df["Product_Family"] == "nan")
            df.loc[fam_na, "Product_Family"] = df.loc[fam_na, "SKU_ID"].map(fam_map)
        if "Business_Unit" in sku_master_df.columns and "Business_Unit" in df.columns:
            bu_map = dict(zip(sku_master_df["SKU_ID"].astype(str).str.strip(), sku_master_df["Business_Unit"]))
            bu_na = df["Business_Unit"].isna() | (df["Business_Unit"] == "") | (df["Business_Unit"] == "nan")
            df.loc[bu_na, "Business_Unit"] = df.loc[bu_na, "SKU_ID"].map(bu_map)

    for col in ["Product_Family", "Business_Unit", "SKU_ID", "SKU_Description", "Customer_Name", "Sales_Channel", "Region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def _deduplicate_latest_load(
    df: pd.DataFrame,
    file_name: str,
    quality_records: List[QualityRecord],
) -> pd.DataFrame:
    """Keep strictly the row with latest Load_Timestamp for duplicate Record_IDs."""
    if df.empty or "Record_ID" not in df.columns or "Load_Timestamp" not in df.columns:
        return df

    df["_load_dt"] = pd.to_datetime(df["Load_Timestamp"], errors="coerce")
    dup_mask = df.duplicated(subset=["Record_ID"], keep=False)
    if dup_mask.any():
        dup_df = df[dup_mask]
        for rec_id, group in dup_df.groupby("Record_ID"):
            sorted_group = group.sort_values(by="_load_dt", ascending=False)
            kept_row = sorted_group.iloc[0]
            superseded_rows = sorted_group.iloc[1:]
            pkey = f"{kept_row.get('Date')}|{kept_row.get('SKU_ID')}|{kept_row.get('Region')}"
            for _, sup_row in superseded_rows.iterrows():
                quality_records.append(
                    QualityRecord(
                        file_name=file_name,
                        source_row_id=str(rec_id),
                        planning_key=pkey,
                        field="Record_ID",
                        rule="latest_load_dedup",
                        original_value=f"Units={sup_row.get('Units_Sold')}, Load={sup_row.get('Load_Timestamp')}",
                        cleaned_value=f"Units={kept_row.get('Units_Sold')}, Load={kept_row.get('Load_Timestamp')}",
                        decision="superseded_by_later_load",
                        reason=f"Repeated Record_ID resolved by latest Load_Timestamp ({kept_row.get('Load_Timestamp')} > {sup_row.get('Load_Timestamp')})",
                        linked_evidence=f"Kept_Timestamp={kept_row.get('Load_Timestamp')}",
                    )
                )

        df = df.sort_values(by="_load_dt", ascending=False).drop_duplicates(subset=["Record_ID"], keep="first")

    return df.drop(columns=["_load_dt"])


def _parse_numeric_column(series: pd.Series, allow_dash_zero: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Parse formatted numeric strings (commas, optional '-' as 0) into floats cleanly."""
    raw_str = series.astype(str).str.strip()
    dash_mask = (raw_str == "-") if allow_dash_zero else pd.Series(False, index=series.index)
    comma_mask = series.notna() & raw_str.str.contains(",", na=False)

    cleaned_str = raw_str.str.replace(",", "", regex=False)
    if allow_dash_zero:
        cleaned_str = cleaned_str.mask(dash_mask, "0")

    numeric = pd.to_numeric(cleaned_str.where(series.notna()), errors="coerce")
    return numeric, dash_mask, comma_mask


def _reconcile_quantities_and_prices(
    df: pd.DataFrame,
    file_name: str,
    quality_records: List[QualityRecord],
) -> pd.DataFrame:
    """Clean numeric formatting, recover missing/sign-flipped/scaled units and cents prices, and reconcile Revenue."""
    if df.empty or not {"Units_Sold", "Unit_Price_USD", "Revenue_USD"}.issubset(df.columns):
        return df

    raw_units = df["Units_Sold"].copy()
    units_num, dash_mask, comma_mask = _parse_numeric_column(df["Units_Sold"], allow_dash_zero=True)
    price_num, _, _ = _parse_numeric_column(df["Unit_Price_USD"], allow_dash_zero=False)
    rev_num, _, _ = _parse_numeric_column(df["Revenue_USD"], allow_dash_zero=False)

    _log_sample_records(
        quality_records, df, dash_mask, file_name,
        field="Units_Sold", rule="dash_zero_normalization",
        original_series=raw_units, cleaned_series=units_num,
        decision="parsed_dash_as_zero",
        reason_fn=lambda o, c: "Dash '-' placeholder in Units_Sold normalized to genuine zero",
        evidence_fn=lambda idx: f"Revenue_USD={rev_num.loc[idx]}",
    )
    _log_sample_records(
        quality_records, df, comma_mask & units_num.notna(), file_name,
        field="Units_Sold", rule="comma_numeric_normalization",
        original_series=raw_units, cleaned_series=units_num,
        decision="parsed_formatted_integer",
        reason_fn=lambda o, c: f"Comma-formatted Units_Sold '{o}' parsed to {c}",
        evidence_fn=lambda idx: f"Revenue_USD={rev_num.loc[idx]}",
    )

    # 1. Negative sign-flip repair (negative quantities matching negative revenue typos)
    neg_mask = units_num.notna() & (units_num < 0) & rev_num.notna() & (rev_num <= 0)
    if neg_mask.any():
        orig_neg = units_num.copy()
        units_num.loc[neg_mask] = units_num.loc[neg_mask].abs()
        rev_num.loc[neg_mask] = rev_num.loc[neg_mask].abs()
        _log_sample_records(
            quality_records, df, neg_mask, file_name,
            field="Units_Sold", rule="negative_sign_reversal_repair",
            original_series=orig_neg, cleaned_series=units_num,
            decision="flipped_negative_sign_typo",
            reason_fn=lambda o, c: f"Negative shipment sign typo ({o}) flipped to positive ({c})",
            evidence_fn=lambda idx: f"Unit_Price_USD={price_num.loc[idx]}",
        )

    # 2. Reference median price per (SKU_ID, Year) with SKU-level fallback
    ref_price = pd.Series(np.nan, index=df.index, dtype=float)
    pos_price_mask = price_num.notna() & (price_num > 0)
    if "SKU_ID" in df.columns and pos_price_mask.any():
        sku_med = price_num[pos_price_mask].groupby(df.loc[pos_price_mask, "SKU_ID"]).median()
        if "Year" in df.columns:
            sku_yr_med = price_num[pos_price_mask].groupby(
                [df.loc[pos_price_mask, "SKU_ID"], df.loc[pos_price_mask, "Year"]]
            ).median()
            yr_lookup = df.set_index(["SKU_ID", "Year"]).index.map(sku_yr_med)
            ref_price = pd.Series(yr_lookup, index=df.index, dtype=float)
        ref_price = ref_price.fillna(df["SKU_ID"].map(sku_med))

    # 3. Cents-to-dollars price error repair (Unit_Price_USD == 100x ref_price and matches Revenue)
    cents_mask = (
        units_num.notna() & (units_num > 0)
        & price_num.notna() & (price_num > 0)
        & rev_num.notna() & (rev_num > 0)
        & ref_price.notna() & (ref_price > 0)
        & ((price_num / ref_price - 100.0).abs() < 5.0)
        & ((units_num * (price_num / 100.0) - rev_num).abs() <= np.maximum(1.0, rev_num * 0.01))
    )
    if cents_mask.any():
        orig_p = price_num.copy()
        price_num.loc[cents_mask] = (price_num.loc[cents_mask] / 100.0).round(2)
        _log_sample_records(
            quality_records, df, cents_mask, file_name,
            field="Unit_Price_USD", rule="cents_to_dollars_price_repair",
            original_series=orig_p, cleaned_series=price_num,
            decision="divided_cents_price_by_100",
            reason_fn=lambda o, c: f"Price entered in cents ({o}) divided by 100 to match reference USD ({c})",
            evidence_fn=lambda idx: f"Units={units_num.loc[idx]}, Revenue={rev_num.loc[idx]}",
        )

    # 4. Recover missing Units_Sold from valid Revenue_USD / Unit_Price_USD
    recov_mask = units_num.isna() & price_num.notna() & (price_num > 0) & rev_num.notna() & (rev_num >= 0)
    if recov_mask.any():
        units_num.loc[recov_mask] = (rev_num.loc[recov_mask] / price_num.loc[recov_mask]).round(0)
        _log_sample_records(
            quality_records, df, recov_mask, file_name,
            field="Units_Sold", rule="recovered_units_from_revenue",
            original_series=raw_units, cleaned_series=units_num,
            decision="derived_from_revenue_and_price",
            reason_fn=lambda o, c: f"Missing Units_Sold recovered from Revenue_USD / Unit_Price_USD = {c}",
            evidence_fn=lambda idx: f"Revenue={rev_num.loc[idx]}, Price={price_num.loc[idx]}",
        )

    # 5. Repair 10x / 100x unit-scale typos when Unit_Price matches reference price and Revenue confirms
    price_agrees_ref = ref_price.isna() | (ref_price <= 0) | ((price_num - ref_price).abs() / ref_price < 0.25)
    expected_rev = units_num * price_num
    valid_calc = units_num.notna() & (units_num > 0) & price_num.notna() & (price_num > 0) & rev_num.notna() & (rev_num > 0)
    scale_ratio = pd.Series(np.nan, index=df.index, dtype=float)
    scale_ratio.loc[valid_calc] = expected_rev.loc[valid_calc] / rev_num.loc[valid_calc]

    scale_10_or_100 = (
        valid_calc
        & price_agrees_ref
        & (((scale_ratio - 10.0).abs() < 0.2) | ((scale_ratio - 100.0).abs() < 2.0))
    )
    if scale_10_or_100.any():
        orig_u = units_num.copy()
        units_num.loc[scale_10_or_100] = (rev_num.loc[scale_10_or_100] / price_num.loc[scale_10_or_100]).round(0)
        _log_sample_records(
            quality_records, df, scale_10_or_100, file_name,
            field="Units_Sold", rule="unit_scale_error_repair",
            original_series=orig_u, cleaned_series=units_num,
            decision="repaired_unit_scale_typo",
            reason_fn=lambda o, c: f"Units_Sold scale typo ({o}) corrected to {c} using verified Revenue / Price",
            evidence_fn=lambda idx: f"Revenue={rev_num.loc[idx]}, Price={price_num.loc[idx]}",
        )

    # 6. Final Revenue vs Units * Price reconciliation
    final_expected_rev = (units_num * price_num).round(2)
    diff = (rev_num - final_expected_rev).abs()
    mismatch_mask = units_num.notna() & price_num.notna() & diff.notna() & (diff > 0.05)
    if mismatch_mask.any():
        orig_rev = rev_num.copy()
        rev_num.loc[mismatch_mask] = final_expected_rev.loc[mismatch_mask]
        _log_sample_records(
            quality_records, df, mismatch_mask, file_name,
            field="Revenue_USD", rule="revenue_price_reconciliation",
            original_series=orig_rev, cleaned_series=rev_num,
            decision="reconciled_revenue",
            reason_fn=lambda o, c: f"Revenue mismatch reconciled to Units x Price (${c})",
            evidence_fn=lambda idx: f"Units_Sold={units_num.loc[idx]}, Unit_Price_USD={price_num.loc[idx]}",
            max_samples=2000,
        )

    df["Units_Sold"] = units_num
    df["Unit_Price_USD"] = price_num
    df["Revenue_USD"] = rev_num
    return df


def clean_history_transactions(
    history_df: pd.DataFrame,
    sku_master_df: Optional[pd.DataFrame] = None,
    file_name: str = "history.csv",
) -> Tuple[pd.DataFrame, List[QualityRecord]]:
    """Clean the raw historical ERP transactions.

    Rules applied:
    1. Authoritative Date: parse multi-format Date strings to YYYY-MM and derive Year/Month.
    2. Normalize SKU IDs, regions, channels, and impute missing hierarchy from SKU master.
    3. Latest-load deduplication: keep the latest Load_Timestamp for repeated Record_ID.
    4. Clean formatted numerics, repair negative/scale/cents typos, recover missing units, reconcile revenue.
    """
    df = history_df.copy()
    quality_records: List[QualityRecord] = []

    df.columns = [c.strip() for c in df.columns]
    df = _normalize_dates(df, file_name, quality_records)
    df = _normalize_skus_regions_channels(df, sku_master_df, file_name, quality_records)
    df = _deduplicate_latest_load(df, file_name, quality_records)
    df = _reconcile_quantities_and_prices(df, file_name, quality_records)

    sort_cols = [c for c in ["Date", "SKU_ID", "Region", "Record_ID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df, quality_records
