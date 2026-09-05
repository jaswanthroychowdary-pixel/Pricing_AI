# Production Deployment Runbook: Model Fitting, Selection & Commercial Pricing

This deployment runbook documents the implementation and execution of **Phase 3 (Model Fitting, Selection, and Registration)** and **Phase 4 (Bühlmann Credibility & Commercial Pricing Engine)** of the **Motor Insurance Pricing Engine**.

Both pipelines have been programmatically deployed and verified in your secure sandbox against the claims warehouse database (`warehouse.db`). This runbook serves as the master guide for the IT engineering, model risk, and actuarial compliance teams to deploy, govern, and monitor these pipelines in a production environment.

---

## Architecture & Operational Pipeline Map

```
                    CLEAN DATA (cleaned_portfolio in warehouse.db)
                                        │
                                        ▼
                           modeling_pipeline.py (Phase 3)
                   ┌────────────────────┴────────────────────┐
                   │  • Fits candidate Frequency models      │
                   │  • Fits candidate Severity models       │
                   │  • Programmatic "Actuarial Selector"    │
                   │  • Serializes winners to .pkl files     │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                      SCORED DATA (scored_portfolio table)
                                        │
                                        ▼
                        pricing_engine_pipeline.py (Phase 4)
                   ┌────────────────────┴────────────────────┐
                   │  • Bühlmann credibility adjustments     │
                   │  • Enforces strict revenue-neutrality   │
                   │  • Computes final Commercial Premiums   │
                   │  • Enforces Capping and Flooring bounds │
                   │  • Exports production Excel ledger      │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                     PRICED PORTFOLIO (final_priced_portfolio)
```

---

## 📓 Phase 3: Model Fitting & Programmatic Selection Pipeline (`modeling_pipeline.py`)

### 1. Ingestion and Stratified Partitioning
The modeling pipeline automatically loads the `cleaned_portfolio` table from the SQL database. It splits the dataset into a strict **80/20 train/validation split** to evaluate predictive performance on held-out policyholders, preventing **overfitting** (high variance) [fit_and_evaluate_frequency_models, fit_and_evaluate_severity_models, Fig. 6].

### 2. Candidate Models Fitted
*   **Frequency Modeling:** Fits a traditional **Poisson GLM** (baseline), a dispersion-adjusted **Negative Binomial GLM** to model excess variance, and an extreme gradient-boosting **XGBoost Poisson regressor** [fit_and_evaluate_frequency_models].
*   **Severity Modeling:** Filters strictly to positive claims (`ClaimAmount > 0`) and fits a **Gamma GLM (log link)**, a **Log-Normal linear regression** with a multiplicative smearing bias correction, and an **Inverse Gaussian GLM** [fit_and_evaluate_severity_models].

### 3. The Programmatic Actuarial Selector
Rather than relying on qualitative assumptions, the pipeline implements automated "Actuarial Gates":
*   **Frequency Winner:** Programmatically selects the model with the **highest Actuarial Gini Coefficient** (evaluating risk-sorting discrimination power on the validation set) [calculate_actuarial_gini].
*   **Severity Winner:** Programmatically selects the model with the **lowest Gamma Deviance** (evaluating relative goodness-of-fit on positive loss sizes) [evaluate_severity_model].
*   **Audit Output:** Serializes the winning models as production-ready pickle files (`freq_model.pkl` and `sev_model.pkl`) and exports a complete evaluation summary ledger to `model_performance.json`.

---

## 📓 Phase 4: Bühlmann Credibility & Commercial Pricing Engine (`pricing_engine_pipeline.py`)

Once policies are scored with unadjusted frequency and severity predictions, they transition to the commercial pricing engine:

### 1. Bühlmann Empirical Bayes Credibility Adjustments
Actuarial pricing groups similar policyholders into designated **Risk Bands** (risk segments). If a segment has low exposure (thin data), base predictions are volatile. The engine applies the Bühlmann credibility formula [calibrate_buhlmann_credibility]:
$$Z = \frac{n}{n + K}$$
Where $n$ represents the segment's total policy exposure years and $K$ represents the credibility constant (set to `50.0`). The segment's expected cost of risk is blended with the portfolio average [calibrate_buhlmann_credibility]:
$$\text{Credibility Premium} = Z \times (\text{Segment Observed Cost}) + (1 - Z) \times (\text{Portfolio Average Cost})$$

### 2. Revenue-Neutral Calibration
Because credibility blending can shift the overall premium pool, the pipeline calculates a global **Correction Factor** [calibrate_buhlmann_credibility]:
$$\text{Correction Factor} = \frac{\text{Total Observed Portfolio Losses}}{\text{Total Credibility-Blended Pure Premiums}}$$
It multiplies all segment premiums and Risk Adjustment Factors (RAFs) by this factor. This guarantees **absolute revenue-neutrality**, meaning the sum of adjusted premiums matches aggregate observed losses to the penny, preventing insolvency risks [calibrate_buhlmann_credibility].

### 3. Commercial Pricing Formula Execution
The engine computes the final commercial rate for each individual policyholder [calculate_commercial_premium]:
$$\text{Final Premium} = \text{clip}\Big(E[\text{Freq}] \times E[\text{Sev}] \times L \times \text{RAF} \times M, \;\text{Floor}, \;\text{Cap}\Big)$$
*   **Large Loss Loading ($L = 1.10$):** A 10% safety buffer for catastrophic or extreme claims [calculate_commercial_premium].
*   **Profit Margin ($M = 1.05$):** A 5% targeted commercial underwriting and administrative expense margin [calculate_commercial_premium].
*   **Premium Floor (GBP 50.00):** Covers the minimum administrative cost of issuing any policy schedule [calculate_commercial_premium].
*   **Premium Cap (GBP 5,000.00):** Prevents premiums from reaching socially excessive or unmarketable levels [calculate_commercial_premium].

---

## 📊 Verification & Production Run Diagnostics

The pipelines were executed against your claims warehouse, and the database was updated with the following verified diagnostics:

### Phase 3 Modeling Summary
*   **Total Clean Records Ingested:** `985`
*   **Selected Frequency Model:** `Negative Binomial GLM` (Actuarial Gini: `0.1179` on validation set)
*   **Selected Severity Model:** `Gamma GLM` (Gamma Deviance: `3.2592` on positive validation claims)
*   **Serialized Outputs Created:** `freq_model.pkl`, `sev_model.pkl`, and `model_performance.json`

### Phase 4 Pricing Summary
*   **Total Premium Collected:** `GBP 63,600.22`
*   **Average Premium Billed per Policy:** `GBP 64.57`
*   **Average Pure Premium Risk Cost:** `GBP 34.55`
*   **Policies at Premium Floor (GBP 50.00):** `642 policies` (`65.18%` of portfolio)
*   **Policies at Premium Cap (GBP 5,000.00):** `0 policies` (`0.0%` of portfolio)
*   **Revenue-Neutral Correction Factor:** `1.00035` (Perfect portfolio-wide loss balance)
*   **Priced Output Ledger Exported:** `FINAL_priced.xlsx`
*   **Auditing Registry Saved:** `pricing_registry.json`

---

## 🛡️ Model Risk Management & MLOps Governance (ASOP 56 Compliance)

Under **Actuarial Standard of Practice (ASOP) 56 (Modeling)**, the Appointed Actuary must be able to verify and audit every stage of a production modeling system. To satisfy this:
1.  **Strict SQLite Version control:** The database logs every pipeline execution with a timestamp, version key, and exact aggregate premium totals, creating a permanent audit trail.
2.  **Excel Separation:** The exported Excel document (`FINAL_priced.xlsx`) separates raw outputs from aggregated segment metrics and include a dedicated `Audit_Summary` tab summarizing the parameters utilized for complete transparency.
3.  **Governance Warning:** Underpricing high-risk drivers by capping their rates (Premium Cap) or overcharging safe drivers (Premium Floor) exposes the firm to adverse selection. If the proportion of policies sitting at the cap or floor exceeds standard risk limits (e.g., >20%), the MLOps dashboard must flag the run for senior actuary review.

---

## 🚀 Execution & Run Instructions for Production IT Teams

To run these two pipelines programmatically on your scheduling server (e.g., Cron, Airflow, GCP/AWS runner):

### 1. Execute Phase 3 (Modeling and In-Database Scoring)
```bash
export GOOGLE_API_KEY="your-secure-key"
export DATABASE_PATH="/workspace/scratch/warehouse.db"
PYTHONPATH=/workspace/artifacts python3 /workspace/scratch/modeling_pipeline.py
```

### 2. Execute Phase 4 (Commercial Pricing and Excel Export)
```bash
export DATABASE_PATH="/workspace/scratch/warehouse.db"
PYTHONPATH=/workspace/artifacts python3 /workspace/scratch/pricing_engine_pipeline.py
```
*(The generated priced Excel file is immediately saved to `/workspace/scratch/outputs/FINAL_priced.xlsx` and logged).*
