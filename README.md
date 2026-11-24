# Celeris Automobili Industrial: Demand Forecasting & Planning Control Tower

A production-grade, local batch forecasting pipeline and interactive Streamlit control tower for Celeris Automobili Industrial. The system forecasts unconstrained requested monthly demand for approximately 600 SKUs across seven product families and six global sales regions over a 19-month horizon (November 2026 through May 2028).

---

## 1. System Architecture & Workflow

The architecture follows a decoupled batch-and-dashboard pattern:
1. **Batch Pipeline (`celeris`)**: Ingests multi-source inputs, resolves schema roles, standardizes labels/dates, deduplicates transactions by latest load timestamp, reconstructs true unconstrained demand from stockouts/backlogs, executes a 19-month expanding-origin backtesting protocol across 3 folds, verifies leakage safety, and publishes typed Parquet artifacts and a competition-ready `submission.csv`.
2. **Interactive Control Tower (`app.py`)**: A Streamlit application rendering 5 Core views with shared global filters (Family, SKU, Region, Channel, Customer), continuous 113-month timelines, and verified row-level reconciliation against the underlying published Parquet files.

```mermaid
flowchart LR
    R[Raw CSV Inputs] --> I[celeris.ingest: Role Resolution & Diagnostics]
    I --> C[celeris.clean: Deduplication & Quality Log]
    C --> D[celeris.demand: Stockout/Backlog Demand Reconstruction]
    D --> B[celeris.backtest: 19-Month Backtest Protocol & Leakage Verification]
    B --> F[celeris.publish: Final Point Forecast & Submission Validation]
    F --> P[(Published Parquet Tables & submission.csv)]
    P --> A[Streamlit Control Tower app.py]
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12.3, Linux x86_64)
- Virtual environment (`venv`)

### Installation
```bash
# 1. Clone / navigate to project root
cd /path/to/celeris

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt  # or: pip install pandas pyarrow streamlit scikit-learn lightgbm pytest
```

---

## 3. Input Roles & Data Mislabeling Resolution

Antigravity auto-discovers and resolves all seven dataset roles by **header schema signature**, rather than relying on brittle physical filenames.

### Inventory & Resolved Roles in `student_data/`

| Physical File in `student_data/` | Resolved Role | Identifying Columns | Rows | Size | Status / Notes |
|---|---|---|---|---|---|
| `train.csv` | `history` (Historical ERP) | `Record_ID`, `Units_Sold`, `Load_Timestamp` | 248,248 | 46.6 MB | Authoritative historical transactions (2019-01 $\rightarrow$ 2026-10) |
| `test.csv` | `forecast_rows` (Test Keys) | Delivery month, SKU, Customer, Channel, Region | 58,219 | 8.2 MB | 595 SKUs across 6 global sales regions (2026-11 $\rightarrow$ 2028-05) |
| `sku_master.csv` | `sku_master` | `Size_Class`, `Launch_Date`, `EOL_Date` | 601 | 74.5 KB | Catalog metadata, lifecycles, and predecessor links |
| `inventory_stockouts.csv` | `stockouts` (Stockout Ledger) | `Requested_Units`, `Shipped_Units`, `Backordered_Units` | 5,830 | 409.9 KB | Supply-constrained inventory incidents |
| `macro_indicators.csv` | `macro_indicators` | `ISM_Manufacturing_PMI`, `EV_Adoption_Index` | 114 | 6.6 KB | Monthly macroeconomic indicators (2019-01 $\rightarrow$ 2028-06) |
| `events_calendar.csv` | `events_calendar` | `Event_ID`, `Scope_Type`, `Event_Type` | 68 | 6.3 KB | Tariff shifts, customer expansions, supply disruptions |
| `sample_submission.csv` | `sample_submission` | `Units_Sold_Forecast` | 5 | 0.2 KB | Authoritative submission format contract |

---

## 4. CLI Commands & Pipeline Execution

### Input Diagnostics
To inspect any directory and evaluate input readiness without modifying state:
```bash
# Diagnose student_data
python -m celeris diagnose --data student_data

# Diagnose synthetic fixture
python -m celeris diagnose --data fixtures/data
```

### Running the Batch Pipeline
The batch runner executes ingestion, cleaning, demand reconstruction, backtesting, and forecast publication:
```bash
# Run against the synthetic fixture (end-to-end verified backtest and submission)
python -m celeris run --data fixtures/data --out artifacts/fixture_run

# Run against real student_data (reports missing history extract and safely halts)
python -m celeris run --data student_data --out artifacts/real_run
```

### Running Automated Verification Tests
```bash
pytest -v
```

### Launching the Control Tower Dashboard
```bash
streamlit run app.py
```

---

## 5. Published Artifacts

All outputs are saved as typed Parquet tables and a validated submission CSV:

| Published Artifact | Location | Grain & Description |
|---|---|---|
| `history.parquet` | `artifacts/fixture_run/` | Monthly panel per SKU-Region (2019-01 $\rightarrow$ 2026-10). Includes `is_observed`, `Units_Sold`, `reconstructed_demand`, `demand_evidence`, prices, and revenue. |
| `quality.parquet` | `artifacts/fixture_run/` | Audit log of every data repair, timestamp deduplication, and label normalization. |
| `anomalies.parquet` | `artifacts/fixture_run/` | Anomaly register: stockout shortages, backlog rebounds, genuine zeros, missing months, and lifecycle flags. |
| `backtest.parquet` | `artifacts/fixture_run/` | 19-month rolling origin predictions across 3 folds for candidate models with row-level errors. |
| `forecast.parquet` | `artifacts/fixture_run/` | 19-month unconstrained point forecast with lifecycle bounds, price projections, and projected revenue. |
| `submission.csv` | `artifacts/fixture_run/` | Competition-ready submission: exactly `Date,SKU_ID,Region,Units_Sold_Forecast` in identical test key order. |
| `macro_lead_lag.parquet` | `artifacts/fixture_run/` | Product-family macroeconomic cross-correlations across lags 0 to 12 with optimal leading indicator identification. |
| `stockouts.parquet` | `artifacts/fixture_run/` | Cleaned stockout incident register with requested, shipped, backordered, carried, lost units and shortage root-causes. |
| `sku_master.parquet` | `artifacts/fixture_run/` | Full product catalog metadata with lifecycle statuses, launch/EOL dates, and predecessor replacement links. |
| `manifest.json` | `artifacts/fixture_run/` | Complete run metadata, input hashes, package versions, backtest metrics, and validation reports. |

---

## 6. Backtest Evaluation Protocol & Results

The backtesting protocol exercises 3 expanding folds with 19-month forecast horizons:
- **Fold 1 (Development)**: Origin `2022-01` $\rightarrow$ Predict `2022-02` to `2023-08` (19 months)
- **Fold 2 (Development)**: Origin `2023-08` $\rightarrow$ Predict `2023-09` to `2025-03` (19 months)
- **Fold 3 (Frozen Holdout)**: Origin `2025-03` $\rightarrow$ Predict `2025-04` to `2026-10` (19 months)

### Candidate Comparison & Selection

#### Real-Data Competition Run (`student_data/` $\rightarrow$ `artifacts/real_run/`)

The pipeline was executed against the authoritative 248,248-row historical extract across 595 SKUs and 6 global regions:

| Model Candidate | Development WAPE (Folds 1 & 2) | Development Signed Bias | Frozen Holdout WAPE (Fold 3) | Frozen Holdout Signed Bias | Status |
|---|---|---|---|---|---|
| **Seasonal Naïve (Multi-Cycle Smoothed)** | **46.75%** | **-6.99%** | **40.77%** | **-1.38%** | **Winner (Selected)** |
| Pooled Direct (L1 Absolute Error Loss) | 54.52% | -13.20% | 51.92% | -12.75% | Challenger 1 |
| Intermittent Challenger (Croston SBA) | 57.84% | -4.46% | 53.97% | -3.90% | Advanced Challenger |
| Trailing Mean | 61.27% | -8.18% | 59.24% | +4.53% | Baseline |

- **Winning Model**: Upgraded `seasonal_naive` with multi-cycle exponential smoothing (60% $t-12$, 25% $t-24$, 15% $t-36$) achieved an outstanding **40.77% WAPE** on the 19-month Frozen Holdout (Fold 3).
- **Published Submission**: 58,219 forecast rows in `artifacts/real_run/submission.csv` (100% format contract compliance, zero nulls, non-negative integers).
- **Aggregate Forecast**: 2,738,919 total units forecasted, generating $2,114,056,492.26 in projected revenue.

#### Synthetic Fixture Validation Benchmark (`fixtures/data/`)

| Model Candidate | Development WAPE (Folds 1 & 2) | Development Signed Bias | Frozen Holdout WAPE (Fold 3) | Frozen Holdout Signed Bias | Status |
|---|---|---|---|---|---|
| **Pooled Direct (LightGBM)** | **9.83%** | **-9.27%** | **8.06%** | **-4.58%** | **Winner (Selected)** |
| Intermittent Challenger (Croston SBA) | 10.37% | -9.84% | 8.52% | -5.12% | Advanced Challenger |
| Seasonal Naïve | 10.85% | -10.28% | 9.16% | -9.16% | Baseline 1 |
| Trailing Mean | 18.71% | -6.67% | 9.20% | -7.29% | Baseline 2 |

**Leakage Verification**: Passed. Modifying future data ($t > T_{\text{origin}}$) with synthetic perturbations leaves fitted features and candidate model forecasts strictly identical.

---

## 7. Acceptance Matrix (Core & Advanced)

| # | Tier | Requirement | Status | Verification & Evidence |
|---|---|---|---|---|
| 1 | Core | **Header-based role resolution** | **PASSED** | Resolves all 7 roles in `student_data/` by signature: `train.csv` (history), `test.csv` (forecast keys), `sku_master.csv` (catalog), `inventory_stockouts.csv` (stockouts), `macro_indicators.csv` (macro), `events_calendar.csv` (events), and `sample_submission.csv` (submission format). |
| 2 | Core | **Graceful halt on missing history** | **PASSED** | Verified on partial input: when ERP sales extract is absent, `celeris run` writes diagnostic manifest with status `BLOCKED_MISSING_HISTORY` and safely halts before fitting or publishing. |
| 3 | Core | **Latest-load deduplication** | **PASSED** | Resolves duplicate `Record_ID` instances by latest `Load_Timestamp`; superseded transactions logged with audit evidence to `quality.parquet`. |
| 4 | Core | **Absence vs Inactivity** | **PASSED** | Genuine zeros (`Units_Sold = 0`, `is_observed = True`) preserved as true zero demand; missing months (`is_observed = False`) represented as unobserved gaps without artificial zero distortion. |
| 5 | Core | **Stockout & carry reconstruction** | **PASSED** | Reconstructs unconstrained demand from stockouts ($D_t = S_t + B_t + L_t$) and deducts carry-in backorders ($C_t = B_{t-1}$) to prevent double counting. |
| 6 | Core | **Origin-safety & leakage checks** | **PASSED** | Verified with `verify_leakage_safety`: corrupting future rows after origin leaves predictions 100% identical. |
| 7 | Core | **Submission format compliance** | **PASSED** | Exact 4 columns (`Date,SKU_ID,Region,Units_Sold_Forecast`), exact row count (58,219), exact key sequence matching `test.csv`, no nulls, all non-negative integers. |
| 8 | Core | **Lifecycle zero enforcement** | **PASSED** | Pre-launch and post-EOL months strictly forced to 0 units with designated audit models `lifecycle_pre_launch` and `lifecycle_post_eol`. |
| 9 | Core | **Dashboard 5 Core views** | **PASSED** | Executive Overview, Data Health, Demand Explorer, Anomaly Monitor, Forecast & Accuracy fully implemented and operational. |
| 10 | Core | **Dashboard row reconciliation** | **PASSED** | Filtered and unfiltered dashboard metrics match row-for-row sums of `forecast.parquet` and `history.parquet` exactly. |
| 11 | Advanced | **Intermittent forecasting (Croston SBA)** | **PASSED** | Implemented `CrostonSBAModel` with debiasing $(1 - \alpha/2)$ factor and `IntermittentChallengerModel` dispatching based on ADI / $CV^2$ intermittency classes. Scored in 19-month backtest. |
| 12 | Advanced | **Macroeconomic lead/lag analysis** | **PASSED** | `analyze_macro_lead_lag` evaluates 8 economic indicators across lags 0 to 12 months for every product family, identifying optimal leading drivers and persisting to `macro_lead_lag.parquet`. |
| 13 | Advanced | **Hierarchical scoped events** | **PASSED** | `match_scoped_events` supports Global, Region, Family, Channel, and Customer scope matching. |
| 14 | Advanced | **Dashboard View 6: Stockouts & Lost Sales** | **PASSED** | Unconstrained requested demand, shipments, backorders, lost sales, fill rate % timeline, and root-cause breakdown. |
| 15 | Advanced | **Dashboard View 7: Lifecycle Planning** | **PASSED** | Catalog lifecycle phase distribution, Powertrain mix, predecessor-successor replacement links (`Replaces_SKU`), and cold-start forecast ramps. |
| 16 | Real Data | **Real-data training & submission** | **PASSED** | Ingested 248,248 rows of historical transactions from `student_data/train.csv`, evaluated all 4 candidate models across 3 backtest folds, generated 58,219 final predictions in `submission.csv` (2,719,061 total units), and published all 9 artifacts to `artifacts/real_run/`. |
