"""Celeris Automobili Industrial Demand Forecasting & Planning Control Tower."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and industrial aesthetic
st.set_page_config(
    page_title="Celeris Automobili | Demand Planning Control Tower",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for distinctive industrial aesthetic (guided by frontend-design skill)
st.markdown("""
<style>
    /* Clean industrial typography & accents */
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    .badge-fixture {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #FCD34D;
    }
    .badge-real {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #31C48D;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(artifact_dir: str = "artifacts/fixture_run") -> Dict[str, Any]:
    """Load published Parquet artifacts and manifest."""
    base_path = Path(artifact_dir)

    history_path = base_path / "history.parquet"
    forecast_path = base_path / "forecast.parquet"
    quality_path = base_path / "quality.parquet"
    anomalies_path = base_path / "anomalies.parquet"
    backtest_path = base_path / "backtest.parquet"
    stockouts_path = base_path / "stockouts.parquet"
    sku_master_path = base_path / "sku_master.parquet"
    macro_path = base_path / "macro_lead_lag.parquet"
    manifest_path = base_path / "manifest.json"

    data: Dict[str, Any] = {}
    if history_path.exists():
        data["history"] = pd.read_parquet(history_path)
    else:
        data["history"] = pd.DataFrame()

    if forecast_path.exists():
        data["forecast"] = pd.read_parquet(forecast_path)
    else:
        data["forecast"] = pd.DataFrame()

    if quality_path.exists():
        data["quality"] = pd.read_parquet(quality_path)
    else:
        data["quality"] = pd.DataFrame()

    if anomalies_path.exists():
        data["anomalies"] = pd.read_parquet(anomalies_path)
    else:
        data["anomalies"] = pd.DataFrame()

    if backtest_path.exists():
        data["backtest"] = pd.read_parquet(backtest_path)
    else:
        data["backtest"] = pd.DataFrame()

    if stockouts_path.exists():
        data["stockouts"] = pd.read_parquet(stockouts_path)
    else:
        data["stockouts"] = pd.DataFrame()

    if sku_master_path.exists():
        data["sku_master"] = pd.read_parquet(sku_master_path)
    else:
        data["sku_master"] = pd.DataFrame()

    if macro_path.exists():
        data["macro_lead_lag"] = pd.read_parquet(macro_path)
    else:
        data["macro_lead_lag"] = pd.DataFrame()

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data["manifest"] = json.load(f)
    else:
        data["manifest"] = {}

    return data


def filter_dataframe(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply shared sidebar filters to any dataframe."""
    if df.empty:
        return df

    filtered = df.copy()
    for col, selected_vals in filters.items():
        if selected_vals and "All" not in selected_vals and col in filtered.columns:
            filtered = filtered[filtered[col].isin(selected_vals)]
    return filtered


def main():
    # Sidebar: Data Source & Run Information
    st.sidebar.markdown("### ⚡ CELERIS AUTOMOBILI")
    st.sidebar.markdown("**Demand Planning Control Tower**")

    # Select Artifact Directory
    available_dirs = []
    if Path("artifacts/real_run").exists():
        available_dirs.append("artifacts/real_run")
    if Path("artifacts/fixture_run").exists():
        available_dirs.append("artifacts/fixture_run")
    if not available_dirs:
        available_dirs = ["artifacts"]

    selected_run = st.sidebar.selectbox("Active Artifact Run", options=available_dirs, index=0)
    artifact_dir = st.sidebar.text_input("Or Custom Directory", value=selected_run)

    data = load_data(artifact_dir)
    manifest = data.get("manifest", {})
    run_mode = manifest.get("run_mode", "fixture")
    is_fixture = manifest.get("is_fixture", True)

    if is_fixture:
        st.sidebar.markdown(
            '<span class="badge-fixture">⚠️ FIXTURE VALIDATION RUN</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="badge-real">✓ REAL DATA RUN</span>',
            unsafe_allow_html=True,
        )

    st.sidebar.caption(f"Run ID: `{manifest.get('run_id', 'unknown')}`")
    st.sidebar.markdown("---")

    # Shared Filters
    st.sidebar.subheader("Shared Planning Filters")

    history_df = data["history"]
    forecast_df = data["forecast"]

    # Gather available values across history & forecast
    families = sorted(list(set(history_df.get("Product_Family", pd.Series(dtype=str)).dropna().unique()) |
                           set(forecast_df.get("Product_Family", pd.Series(dtype=str)).dropna().unique())))
    sel_families = st.sidebar.multiselect("Product Family", options=["All"] + families, default=["All"])

    # Filter SKUs dynamically
    skus_series = history_df.get("SKU_ID", pd.Series(dtype=str))
    if sel_families and "All" not in sel_families:
        skus_series = history_df[history_df["Product_Family"].isin(sel_families)].get("SKU_ID", pd.Series(dtype=str))
    skus = sorted(list(set(skus_series.dropna().unique()) | set(forecast_df.get("SKU_ID", pd.Series(dtype=str)).dropna().unique())))
    sel_skus = st.sidebar.multiselect("SKU ID", options=["All"] + skus, default=["All"])

    regions = sorted(list(set(history_df.get("Region", pd.Series(dtype=str)).dropna().unique()) |
                          set(forecast_df.get("Region", pd.Series(dtype=str)).dropna().unique())))
    sel_regions = st.sidebar.multiselect("Region", options=["All"] + regions, default=["All"])

    channels = sorted(list(set(history_df.get("Sales_Channel", pd.Series(dtype=str)).dropna().unique()) |
                           set(forecast_df.get("Sales_Channel", pd.Series(dtype=str)).dropna().unique())))
    sel_channels = st.sidebar.multiselect("Sales Channel", options=["All"] + channels, default=["All"])

    customers = sorted(list(set(history_df.get("Customer_Name", pd.Series(dtype=str)).dropna().unique()) |
                            set(forecast_df.get("Customer_Name", pd.Series(dtype=str)).dropna().unique())))
    sel_customers = st.sidebar.multiselect("Customer", options=["All"] + customers, default=["All"])

    filters = {
        "Product_Family": sel_families,
        "SKU_ID": sel_skus,
        "Region": sel_regions,
        "Sales_Channel": sel_channels,
        "Customer_Name": sel_customers,
    }

    # Apply filters to core tables
    filt_history = filter_dataframe(history_df, filters)
    filt_forecast = filter_dataframe(forecast_df, filters)
    filt_quality = filter_dataframe(data["quality"], filters)
    filt_anomalies = filter_dataframe(data["anomalies"], filters)
    filt_backtest = filter_dataframe(data["backtest"], filters)
    filt_stockouts = filter_dataframe(data.get("stockouts", pd.DataFrame()), filters)
    filt_sku_master = filter_dataframe(data.get("sku_master", pd.DataFrame()), filters)
    filt_macro = filter_dataframe(data.get("macro_lead_lag", pd.DataFrame()), filters)

    st.sidebar.markdown("---")
    st.sidebar.caption("Navigation Views")

    # Main Navigation Tabs / Views
    view = st.sidebar.radio(
        "Select Control Tower View",
        [
            "1. Executive Overview",
            "2. Data Health",
            "3. Demand Explorer",
            "4. Anomaly Monitor",
            "5. Forecast & Accuracy",
            "6. Stockouts & Lost Sales",
            "7. Lifecycle Planning",
        ],
    )

    # VIEW 1: EXECUTIVE OVERVIEW
    if view == "1. Executive Overview":
        st.markdown('<h1 class="main-header">Executive Overview</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Continuous 113-month timeline spanning historical shipments and 19-month unconstrained demand forecast.</p>',
            unsafe_allow_html=True,
        )

        # Top Headline KPIs
        c1, c2, c3, c4 = st.columns(4)
        total_fct_units = int(filt_forecast["Units_Sold_Forecast"].sum()) if not filt_forecast.empty else 0
        total_fct_rev = float(filt_forecast["Projected_Revenue_USD"].sum()) if not filt_forecast.empty else 0.0
        hist_units = int(filt_history["Units_Sold"].sum(skipna=True)) if not filt_history.empty else 0
        hist_reconstructed = float(filt_history["reconstructed_demand"].sum(skipna=True)) if not filt_history.empty else 0.0

        with c1:
            st.metric("19-Month Forecast Units", f"{total_fct_units:,} units", delta=f"{len(filt_forecast)} test rows")
        with c2:
            st.metric("19-Month Projected Revenue", f"${total_fct_rev:,.2f}", delta="USD")
        with c3:
            st.metric("Cleaned Historical Shipments", f"{hist_units:,} units", delta=f"{int(filt_history['is_observed'].sum()) if not filt_history.empty else 0} observed")
        with c4:
            st.metric("Reconstructed True Demand", f"{hist_reconstructed:,.0f} units", delta=f"Delta: {hist_reconstructed - hist_units:+,.0f}")

        st.markdown("---")
        st.subheader("Continuous Planning Timeline (Jan 2019 → May 2028)")

        # Prepare continuous timeline dataframe
        hist_monthly = (
            filt_history.groupby("Date", as_index=False)
            .agg({
                "Units_Sold": "sum",
                "reconstructed_demand": "sum",
                "Revenue_USD": "sum",
            })
            .rename(columns={"Units_Sold": "Historical Shipments", "reconstructed_demand": "Reconstructed Demand"})
        )

        fct_monthly = (
            filt_forecast.groupby("Date", as_index=False)
            .agg({
                "Units_Sold_Forecast": "sum",
                "Projected_Revenue_USD": "sum",
            })
            .rename(columns={"Units_Sold_Forecast": "Point Forecast", "Projected_Revenue_USD": "Projected Revenue"})
        )

        # Melt for clean Altair visualization
        timeline_df = pd.merge(hist_monthly, fct_monthly, on="Date", how="outer").sort_values(by="Date").reset_index(drop=True)

        chart_data = []
        for _, r in timeline_df.iterrows():
            d = r["Date"]
            is_forecast = d >= "2026-11"
            if pd.notna(r.get("Historical Shipments")):
                chart_data.append({"Date": d, "Units": r["Historical Shipments"], "Series": "Cleaned Shipments", "Section": "History"})
            if pd.notna(r.get("Reconstructed Demand")):
                chart_data.append({"Date": d, "Units": r["Reconstructed Demand"], "Series": "Reconstructed True Demand", "Section": "History"})
            if pd.notna(r.get("Point Forecast")):
                chart_data.append({"Date": d, "Units": r["Point Forecast"], "Series": "Final Point Forecast", "Section": "Forecast"})

        cdf = pd.DataFrame(chart_data)
        if not cdf.empty:
            line_chart = (
                alt.Chart(cdf)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Date:N", title="Delivery Month (YYYY-MM)", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Units:Q", title="Monthly Units"),
                    color=alt.Color("Series:N", scale=alt.Scale(
                        domain=["Cleaned Shipments", "Reconstructed True Demand", "Final Point Forecast"],
                        range=["#64748B", "#0284C7", "#E11D48"],
                    )),
                    strokeDash=alt.condition(
                        alt.datum.Series == "Final Point Forecast",
                        alt.value([6, 4]),
                        alt.value([0]),
                    ),
                    tooltip=["Date", "Series", "Units", "Section"],
                )
                .properties(height=380)
            )

            # Cutoff rule
            rule = (
                alt.Chart(pd.DataFrame([{"Cutoff": "2026-10"}]))
                .mark_rule(color="#94A3B8", strokeDash=[4, 4], size=2)
                .encode(x="Cutoff:N")
            )

            st.altair_chart(line_chart + rule, use_container_width=True)
        else:
            st.info("No timeline observations available for the current filter criteria.")

        # Mix breakdowns
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Forecast Mix by Product Family")
            if not filt_forecast.empty:
                fam_mix = (
                    filt_forecast.groupby("Product_Family")[["Units_Sold_Forecast", "Projected_Revenue_USD"]]
                    .sum()
                    .reset_index()
                )
                st.dataframe(
                    fam_mix.style.format({"Units_Sold_Forecast": "{:,}", "Projected_Revenue_USD": "${:,.2f}"}),
                    use_container_width=True,
                )
        with col_m2:
            st.markdown("#### Forecast Mix by Region")
            if not filt_forecast.empty:
                reg_mix = (
                    filt_forecast.groupby("Region")[["Units_Sold_Forecast", "Projected_Revenue_USD"]]
                    .sum()
                    .reset_index()
                )
                st.dataframe(
                    reg_mix.style.format({"Units_Sold_Forecast": "{:,}", "Projected_Revenue_USD": "${:,.2f}"}),
                    use_container_width=True,
                )

    # VIEW 2: DATA HEALTH
    elif view == "2. Data Health":
        st.markdown('<h1 class="main-header">Data Health & Audit Trail</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Transparent log of data quality rules, latest-load deduplication, and price-revenue reconciliation.</p>',
            unsafe_allow_html=True,
        )

        quality_df = data["quality"]
        anomalies_df = data["anomalies"]

        q1, q2, q3 = st.columns(3)
        with q1:
            st.metric("Total Quality Issues Logged", f"{len(quality_df):,} events")
        with q2:
            dedup_count = len(quality_df[quality_df["rule"] == "latest_load_dedup"]) if not quality_df.empty else 0
            st.metric("Latest-Load Deduplications", f"{dedup_count:,} records")
        with q3:
            anom_count = len(anomalies_df) if not anomalies_df.empty else 0
            st.metric("Registered Anomalies", f"{anom_count:,} flags")

        st.markdown("---")
        st.subheader("Data Cleaning Audit Log (`quality.parquet`)")

        if not quality_df.empty:
            rule_filter = st.multiselect("Filter by Cleaning Rule", options=sorted(quality_df["rule"].unique()), default=[])
            display_q = quality_df[quality_df["rule"].isin(rule_filter)] if rule_filter else quality_df

            st.dataframe(
                display_q[[
                    "file_name", "source_row_id", "planning_key", "field",
                    "rule", "original_value", "cleaned_value", "decision", "reason"
                ]],
                use_container_width=True,
            )
        else:
            st.info("No data quality repairs logged.")

    # VIEW 3: DEMAND EXPLORER
    elif view == "3. Demand Explorer":
        st.markdown('<h1 class="main-header">Demand Explorer</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Drill into SKU-level order intervals, genuine zeros vs missing months, and stockout adjustments.</p>',
            unsafe_allow_html=True,
        )

        if not filt_history.empty:
            selected_sku = st.selectbox("Select SKU for Deep Dive", options=sorted(filt_history["SKU_ID"].unique()))
            sku_df = filt_history[filt_history["SKU_ID"] == selected_sku].sort_values(by="Date")

            # Series metadata cards
            first_r = sku_df.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Product Family", first_r.get("Product_Family", "N/A"))
            with m2:
                st.metric("Lifecycle Status", first_r.get("Lifecycle_Status", "Active"))
            with m3:
                observed_months = int(sku_df["is_observed"].sum())
                st.metric("Observed Months", f"{observed_months} / {len(sku_df)}")
            with m4:
                zeros = int((sku_df["Units_Sold"] == 0).sum())
                missing = int((~sku_df["is_observed"]).sum())
                st.metric("Zeros vs Missing", f"{zeros} zeros, {missing} missing")

            st.markdown("---")
            st.subheader(f"Historical Demand Series: {selected_sku}")

            sku_chart_data = []
            for _, r in sku_df.iterrows():
                d = r["Date"]
                if pd.notna(r["Units_Sold"]):
                    sku_chart_data.append({"Date": d, "Units": r["Units_Sold"], "Series": "Cleaned Shipments"})
                if pd.notna(r["reconstructed_demand"]):
                    sku_chart_data.append({"Date": d, "Units": r["reconstructed_demand"], "Series": "Reconstructed Demand"})

            scdf = pd.DataFrame(sku_chart_data)
            if not scdf.empty:
                chart = (
                    alt.Chart(scdf)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:N", title="Month (YYYY-MM)", axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y("Units:Q", title="Units"),
                        color=alt.Color("Series:N", scale=alt.Scale(
                            domain=["Cleaned Shipments", "Reconstructed Demand"],
                            range=["#64748B", "#0284C7"],
                        )),
                        tooltip=["Date", "Series", "Units"],
                    )
                    .properties(height=340)
                )
                st.altair_chart(chart, use_container_width=True)

            st.dataframe(
                sku_df[["Date", "Units_Sold", "reconstructed_demand", "demand_evidence", "is_observed", "Unit_Price_USD", "Revenue_USD"]],
                use_container_width=True,
            )

            # Macroeconomic Leading Drivers for this Product Family
            fam = first_r.get("Product_Family", "")
            macro_data = data.get("macro_lead_lag", pd.DataFrame())
            if not macro_data.empty and fam:
                fam_macro = macro_data[macro_data["Product_Family"] == fam]
                if not fam_macro.empty:
                    best_rec = fam_macro[fam_macro["Is_Best_For_Family"]].iloc[0] if fam_macro["Is_Best_For_Family"].any() else fam_macro.iloc[0]
                    st.markdown("---")
                    st.subheader(f"Leading Macro Drivers for {fam}")
                    st.info(
                        f"⚡ **Leading Macro Indicator:** `{best_rec['Best_Indicator']}` leads `{fam}` demand by "
                        f"**{int(best_rec['Best_Lag'])} months** (Peak Pearson Correlation: `r = {best_rec['Best_Correlation']:+.2f}`)."
                    )
        else:
            st.warning("No historical series match current filters.")

    # VIEW 4: ANOMALY MONITOR
    elif view == "4. Anomaly Monitor":
        st.markdown('<h1 class="main-header">Anomaly Monitor</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Automated detection of stockout shortages, backlog rebounds, zero orders, missing gaps, and lifecycle violations.</p>',
            unsafe_allow_html=True,
        )

        anom_df = filt_anomalies
        if not anom_df.empty:
            type_counts = anom_df["anomaly_type"].value_counts()
            c_cols = st.columns(min(4, len(type_counts)))
            for i, (atype, cnt) in enumerate(type_counts.items()):
                with c_cols[i % len(c_cols)]:
                    st.metric(atype.replace("_", " ").title(), f"{cnt:,} events")

            st.markdown("---")
            sel_types = st.multiselect("Filter Anomaly Types", options=sorted(anom_df["anomaly_type"].unique()), default=[])
            disp_anom = anom_df[anom_df["anomaly_type"].isin(sel_types)] if sel_types else anom_df

            st.dataframe(
                disp_anom[[
                    "Date", "SKU_ID", "Region", "anomaly_type", "description",
                    "original_value", "cleaned_value", "action", "reason", "linked_evidence"
                ]],
                use_container_width=True,
            )
        else:
            st.info("No anomalies recorded for the current filter selection.")

    # VIEW 5: FORECAST & ACCURACY
    elif view == "5. Forecast & Accuracy":
        st.markdown('<h1 class="main-header">Forecast & Accuracy Control</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Backtest accuracy metrics across candidate models, 19-month rolling origins, and frozen holdout validation.</p>',
            unsafe_allow_html=True,
        )

        backtest_df = filt_backtest
        if not backtest_df.empty:
            st.subheader("Candidate Model Leaderboard (Development Folds vs Frozen Holdout)")

            scored_bt = backtest_df.dropna(subset=["actual", "error"])

            summary_rows = []
            for fold_name in sorted(scored_bt["fold"].unique()):
                f_df = scored_bt[scored_bt["fold"] == fold_name]
                fold_type = f_df["fold_type"].iloc[0]
                for cand in sorted(f_df["candidate"].unique()):
                    c_df = f_df[f_df["candidate"] == cand]
                    acts = c_df["actual"].to_numpy()
                    fcts = c_df["forecast"].to_numpy()
                    total_act = np.sum(acts)
                    wape = np.sum(np.abs(fcts - acts)) / total_act if total_act > 0 else 0.0
                    bias = np.sum(fcts - acts) / total_act if total_act > 0 else 0.0

                    summary_rows.append({
                        "Fold": fold_name,
                        "Type": fold_type,
                        "Model Candidate": cand,
                        "WAPE": f"{wape:.2%}",
                        "Signed Bias": f"{bias:+.2%}",
                        "Total Actual": f"{total_act:,.0f}",
                        "Total Forecast": f"{np.sum(fcts):,.0f}",
                        "Scored Months": len(c_df),
                    })

            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.markdown("---")
                st.subheader("Backtest Origin Deep-Dive (Predicted vs Realized Demand)")
                available_folds = sorted(scored_bt["fold"].unique())
                if available_folds:
                    sel_fold = st.selectbox("Select Backtest Fold", options=available_folds)
                    fold_plot_df = scored_bt[scored_bt["fold"] == sel_fold]

                    fold_chart_data = []
                    for _, r in fold_plot_df.groupby(["target_date", "candidate"])[["forecast", "actual"]].sum().reset_index().iterrows():
                        fold_chart_data.append({"Target Month": r["target_date"], "Units": r["actual"], "Series": "Actual Demand"})
                        fold_chart_data.append({"Target Month": r["target_date"], "Units": r["forecast"], "Series": f"Forecast ({r['candidate']})"})

                    fcdf = pd.DataFrame(fold_chart_data).drop_duplicates()
                    if not fcdf.empty:
                        b_chart = (
                            alt.Chart(fcdf)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("Target Month:N", axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y("Units:Q", title="Monthly Units"),
                                color=alt.Color("Series:N"),
                                tooltip=["Target Month", "Series", "Units"],
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(b_chart, use_container_width=True)
            else:
                st.info("No backtest predictions match the selected filter criteria.")

        st.markdown("---")
        st.subheader("Published Final Forecast Table (`forecast.parquet`)")
        st.dataframe(
            filt_forecast[[
                "Date", "SKU_ID", "Region", "Product_Family", "horizon",
                "Units_Sold_Forecast", "Unit_Price_USD", "Projected_Revenue_USD",
                "Model_Or_Fallback", "Lifecycle_Status"
            ]],
            use_container_width=True,
        )

    # VIEW 6: STOCKOUTS & LOST SALES (Advanced View 1)
    elif view == "6. Stockouts & Lost Sales":
        st.markdown('<h1 class="main-header">Stockouts & Lost Sales Analysis</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Unconstrained requested demand restoration, backorders, lost sales, and fill rate timeline.</p>',
            unsafe_allow_html=True,
        )

        stockouts_df = filt_stockouts
        if not stockouts_df.empty:
            req_units = int(stockouts_df["Requested_Units"].sum()) if "Requested_Units" in stockouts_df.columns else 0
            shp_units = int(stockouts_df["Shipped_Units"].sum()) if "Shipped_Units" in stockouts_df.columns else 0
            bck_units = int(stockouts_df["Backordered_Units"].sum()) if "Backordered_Units" in stockouts_df.columns else 0
            car_units = int(stockouts_df["Carried_In_Units"].sum()) if "Carried_In_Units" in stockouts_df.columns else 0
            lst_units = int(stockouts_df["Lost_Units"].sum()) if "Lost_Units" in stockouts_df.columns else 0
            fill_pct = (shp_units / req_units * 100.0) if req_units > 0 else 100.0
            stk_days = int(stockouts_df["Stockout_Days"].sum()) if "Stockout_Days" in stockouts_df.columns else 0

            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                st.metric("Total Requested Units", f"{req_units:,} units")
            with s2:
                st.metric("Shipped Units", f"{shp_units:,} units", delta=f"{fill_pct:.1f}% fill rate")
            with s3:
                st.metric("Backorder / Carry-In", f"{(bck_units + car_units):,} units")
            with s4:
                st.metric("Permanently Lost Sales", f"{lst_units:,} units")
            with s5:
                st.metric("Stockout Duration", f"{stk_days:,} days", delta=f"{len(stockouts_df)} incident(s)")

            st.markdown("---")
            st.subheader("Requested vs Shipped vs Lost Units Timeline")

            monthly_stk = stockouts_df.groupby("Date")[["Requested_Units", "Shipped_Units", "Lost_Units"]].sum().reset_index()
            stk_chart_rows = []
            for _, r in monthly_stk.iterrows():
                stk_chart_rows.append({"Date": r["Date"], "Units": r["Requested_Units"], "Category": "Requested Units (Unconstrained)"})
                stk_chart_rows.append({"Date": r["Date"], "Units": r["Shipped_Units"], "Category": "Shipped Units"})
                stk_chart_rows.append({"Date": r["Date"], "Units": r["Lost_Units"], "Category": "Lost Units"})
            stk_chart_df = pd.DataFrame(stk_chart_rows)

            if not stk_chart_df.empty:
                c_stk = (
                    alt.Chart(stk_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Date:N", axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y("Units:Q"),
                        color=alt.Color("Category:N", scale=alt.Scale(range=["#3B82F6", "#10B981", "#EF4444"])),
                        xOffset="Category:N",
                        tooltip=["Date", "Category", "Units"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(c_stk, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Fill Rate % Timeline")
                if "Fill_Rate_Pct" in stockouts_df.columns:
                    fr_chart = (
                        alt.Chart(stockouts_df)
                        .mark_line(point=True, color="#F59E0B")
                        .encode(
                            x=alt.X("Date:N", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("Fill_Rate_Pct:Q", title="Fill Rate (%)", scale=alt.Scale(domain=[0, 100])),
                            tooltip=["Date", "SKU_ID", "Region", "Fill_Rate_Pct", "Stockout_Days"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(fr_chart, use_container_width=True)

            with col_b:
                st.subheader("Root-Cause Reason Distribution")
                if "Reason" in stockouts_df.columns:
                    reason_counts = stockouts_df["Reason"].value_counts().reset_index()
                    reason_counts.columns = ["Shortage Reason", "Incident Count"]
                    r_chart = (
                        alt.Chart(reason_counts)
                        .mark_bar(color="#6366F1")
                        .encode(
                            x=alt.X("Incident Count:Q"),
                            y=alt.Y("Shortage Reason:N", sort="-x"),
                            tooltip=["Shortage Reason", "Incident Count"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(r_chart, use_container_width=True)

            st.markdown("---")
            st.subheader("Stockout Incident Register (`stockouts.parquet`)")
            st.dataframe(stockouts_df, use_container_width=True)
        else:
            st.info("No stockout incidents match the current filters.")

    # VIEW 7: LIFECYCLE PLANNING (Advanced View 2)
    elif view == "7. Lifecycle Planning":
        st.markdown('<h1 class="main-header">Lifecycle & Transition Planning</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Catalog lifecycle phases, predecessor replacement links (`Replaces_SKU`), and phase-in/phase-out ramps.</p>',
            unsafe_allow_html=True,
        )

        sku_master = filt_sku_master if not filt_sku_master.empty else data.get("sku_master", pd.DataFrame())
        if not sku_master.empty:
            total_skus = len(sku_master)
            active_skus = len(sku_master[sku_master["Lifecycle_Status"] == "Active"]) if "Lifecycle_Status" in sku_master.columns else 0
            discont_skus = len(sku_master[sku_master["Lifecycle_Status"] == "Discontinued"]) if "Lifecycle_Status" in sku_master.columns else 0
            replaces_count = int(sku_master["Replaces_SKU"].notna().sum()) if "Replaces_SKU" in sku_master.columns else 0

            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.metric("Catalog SKUs", f"{total_skus:,} SKUs")
            with l2:
                st.metric("Active Lifecycle", f"{active_skus:,} SKUs", delta=f"{active_skus/total_skus:.0%}" if total_skus else "")
            with l3:
                st.metric("Discontinued / EOL", f"{discont_skus:,} SKUs", delta=f"-{discont_skus/total_skus:.0%}" if total_skus else "", delta_color="inverse")
            with l4:
                st.metric("Predecessor Links", f"{replaces_count:,} SKUs", delta="Cold-start mappings")

            st.markdown("---")
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.subheader("SKU Distribution by Lifecycle Status")
                if "Lifecycle_Status" in sku_master.columns and "Product_Family" in sku_master.columns:
                    status_summary = sku_master.groupby(["Product_Family", "Lifecycle_Status"]).size().reset_index(name="SKU Count")
                    ls_chart = (
                        alt.Chart(status_summary)
                        .mark_bar()
                        .encode(
                            x=alt.X("Product_Family:N", axis=alt.Axis(labelAngle=-30)),
                            y=alt.Y("SKU Count:Q"),
                            color=alt.Color("Lifecycle_Status:N", scale=alt.Scale(domain=["Active", "Discontinued"], range=["#10B981", "#EF4444"])),
                            tooltip=["Product_Family", "Lifecycle_Status", "SKU Count"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(ls_chart, use_container_width=True)

            with col_l2:
                st.subheader("Powertrain Application Mix")
                if "Powertrain_Application" in sku_master.columns:
                    pt_summary = sku_master["Powertrain_Application"].value_counts().reset_index()
                    pt_summary.columns = ["Powertrain", "SKU Count"]
                    pt_chart = (
                        alt.Chart(pt_summary)
                        .mark_arc(innerRadius=45)
                        .encode(
                            theta=alt.Theta("SKU Count:Q"),
                            color=alt.Color("Powertrain:N", scale=alt.Scale(range=["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"])),
                            tooltip=["Powertrain", "SKU Count"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(pt_chart, use_container_width=True)

            st.markdown("---")
            st.subheader("Predecessor-Successor Cold Start Transition Ramp")
            replaces_skus = sku_master[sku_master["Replaces_SKU"].notna()]["SKU_ID"].tolist() if "Replaces_SKU" in sku_master.columns else []
            if replaces_skus:
                sel_trans_sku = st.selectbox("Select New SKU with Predecessor", options=replaces_skus)
                pred_row = sku_master[sku_master["SKU_ID"] == sel_trans_sku].iloc[0]
                pred_sku = pred_row["Replaces_SKU"]
                launch_date = pred_row.get("Launch_Date", "N/A")

                st.info(f"⚡ **SKU Transition:** `{sel_trans_sku}` replaces predecessor `{pred_sku}` (Effective Launch: `{launch_date}`). Forecast cold-start is seeded from predecessor history.")

                pred_hist = filt_history[filt_history["SKU_ID"] == pred_sku].sort_values("Date")
                succ_fct = filt_forecast[filt_forecast["SKU_ID"] == sel_trans_sku].sort_values("Date")

                ramp_rows = []
                for _, r in pred_hist.iterrows():
                    ramp_rows.append({"Date": r["Date"], "Units": r.get("reconstructed_demand", r.get("Units_Sold", 0)), "Series": f"Predecessor ({pred_sku}) Demand"})
                for _, r in succ_fct.iterrows():
                    ramp_rows.append({"Date": r["Date"], "Units": r.get("Units_Sold_Forecast", 0), "Series": f"New SKU ({sel_trans_sku}) Forecast Ramp"})

                ramp_df = pd.DataFrame(ramp_rows)
                if not ramp_df.empty:
                    ramp_chart = (
                        alt.Chart(ramp_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Date:N", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("Units:Q", title="Monthly Units"),
                            color=alt.Color("Series:N", scale=alt.Scale(range=["#6B7280", "#2563EB"])),
                            tooltip=["Date", "Series", "Units"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(ramp_chart, use_container_width=True)
            else:
                st.caption("No replacement links found in current filtered catalog.")

            st.markdown("---")
            st.subheader("SKU Master Catalog (`sku_master.parquet`)")
            st.dataframe(sku_master, use_container_width=True)
        else:
            st.info("No SKU master data found.")


if __name__ == "__main__":
    main()
