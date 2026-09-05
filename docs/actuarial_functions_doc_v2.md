# Actuarial Reference Manual: Pipeline Engine Utilities (Version 2.0)

This reference manual documents the reusable, actuarial-grade Python functions developed for the **Motor Insurance Pricing Engine**, spanning all 7 notebooks of the pipeline.

Designed to support the deployment of an **AI-powered actuarial system**, this documentation ensures that when a user asks an AI model questions like *“What does this function do?”* or *“Why are we using this method?”*, the AI can query this document and respond with a clear, accurate, and non-technical explanation.

---

## Design Philosophy

To support modern, automated, and auditable pipelines, every function follows three core design principles:
1. **Technical Reusability:** Operating strictly on standardized NumPy arrays and Pandas DataFrames, making them easily pluggable across different insurance portfolios and lines of business.
2. **AI-Readability:** Standardized Google-style docstrings structured so that Large Language Models (LLMs) can parse the function signature, argument types, and descriptions to select and invoke the tool correctly (per *Chapter 10: Tool Use and Function Calling*).
3. **Structured Outputs:** Each function returns a typed dictionary containing the numerical result, essential metadata for auditability, and execution status codes (`status: "ok"`, `status: "error"`, etc.) to prevent silent, compounding failures.

---

## 1. Data Profiling & Exposure Synthesis (`profile_data_and_synthesize`) - Notebook 01

### A. What the Function Does
This function acts as a **data gatekeeper and profiler** for raw insurance datasets. It automatically scans the dataset to detect target columns (claim counts and payout sizes), verifies if "Exposure" (the duration of coverage, e.g., 1.0 for a full year) is present, synthesizes exposure to a default value if missing, and compiles a clean, standardized DataFrame alongside summary distribution metrics.

### B. Why It Is Used (Actuarial & Business Context)
Under **Actuarial Standard of Practice (ASOP) 23 (Data Quality)**, data validation and profiling are critical first steps. If policy exposure is ignored or missing, we cannot calculate claim rates accurately. For example, a driver with a 1-month policy who has 1 accident appears to have the same claim count as a driver with a 12-month policy who has 1 accident. Expressed as a rate, the first driver is 12 times riskier than the second! This function standardizes our data to an exposure-weighted basis, ensuring fair downstream pricing models.

### C. What Inputs It Requires
*   `df` *(Raw Portfolio)*: A Pandas DataFrame containing our historical policyholder and claims database.
*   `claim_nb_col` *(Claims count column)*: The name of the column storing the number of claims (default is `"ClaimNb"`).
*   `claim_amount_col` *(Payout column)*: The name of the column storing historical claim sizes (default is `"ClaimAmount"`).
*   `exposure_col` *(Exposure column)*: The name of the column representing policy duration (default is `"Exposure"`).

### D. Process or Calculation Performed
1.  **Column Detection:** It scans the column names for variations of "claim number" or "claim amount" to locate the targets.
2.  **Exposure Synthesis:** If the exposure column is missing, it creates it and sets it to `1.0` (one full year of coverage). If exposure is present but contains invalid values (like 0 or negative numbers), it fills them with conservative defaults.
3.  **Distribution Profiling:** It calculates total policies, total exposure years, aggregate claims, and overall portfolio claim frequency (claims per exposure year) and average claim severity.

### E. What Output It Produces
A structured dictionary containing:
*   `df`: The standardized Pandas DataFrame.
*   `meta`: A high-level metadata dictionary (reporting total records, total exposure years, portfolio claim frequency, total claim payout, and average claim severity).

### F. How to Interpret the Results
*   **Total Exposure Years:** The total duration of risk the company covered. For a healthy motor portfolio, this should align with budgeted risk units.
*   **Portfolio Claim Frequency:** The average rate of claims per year (e.g., a frequency of `0.05` means 5 out of 100 policyholders claim each year).
*   **Exposure Synthesized = True:** Signals that the raw file lacked exposure details and the function safely generated standard assumptions, which must be flagged for underwriting review.

---

## 2. Multi-Dimensional Anomaly Pipeline (`detect_anomalies_pipeline`) - Notebook 02

### A. What the Function Does
This function is our **ultimate data quality scanner**. It applies a rigorous three-stage filter to clean our datasets:
1.  **Logical Checks:** Spots simple data entries that violate physics (like negative claims, negative payouts, or payouts on zero claims).
2.  **Multivariate Outliers (Isolation Forest):** Uses an AI machine learning model to spot policies whose combined characteristics are highly abnormal.
3.  **Influence & Leverage Diagnostics:** Spots extreme outlier policies that will heavily distort and warp our regression models.

### B. Why It Is Used (Actuarial & Business Context)
Manual data cleaning cannot catch multi-dimensional outliers (e.g., a driver whose age, car value, and region risk are individually ordinary, but whose combined pattern is highly anomalous). Furthermore, standard regressions like GLMs are highly sensitive to extreme data points. A single typo (e.g., writing `$999,999` instead of `$99` for a claim) can warp our entire premium structure. Running this pipeline flags these records for manual review (`KEEP` or `REMOVE`) as required under **ASOP 23**.

### C. What Inputs It Requires
*   `df` : Standardized DataFrame.
*   `predictor_cols` : List of feature columns (e.g., Age, Vehicle Value, Region).
*   `claim_nb_col` & `claim_amount_col` : Claim targets.
*   `contamination_rate` : The percentage of rows to flag as multivariate outliers via the Isolation Forest (default is `1%`).
*   `leverage_multiplier` & `residual_threshold` : Sensitivity parameters for influence scoring.

### D. Process or Calculation Performed
1.  **Logical checks:** Evaluates a series of boolean masks checking for invalid values.
2.  **Isolation Forest:** Trains an unsupervised `IsolationForest` model on the predictor columns to score multivariate anomalies.
3.  **Leverage and Residual Scoring ($H_{ii}$ & $r_i^D$):** Computes hat matrix diagonals to evaluate feature extremity and Poisson deviance residuals to evaluate model fit. Rows that have high leverage AND high prediction error are flagged.
4.  **Global Anomaly Mask:** Combines all three triggers into a single True/False flag per row.

### E. What Output It Produces
A dictionary containing:
*   `df`: The DataFrame with new flagging columns (`dq_flag`, `iso_outlier_flag`, `influence_flag`, `global_anomaly_flag`).
*   `metrics`: A summary statistics dictionary (reporting counts of DQ failures, Isolation Forest anomalies, high-influence rows, and the global flagged percentage).

### F. How to Interpret the Results
*   **Global Flagged Pct:** Represents the overall noise level of our raw data. Usually, this is around `1% to 2%`. If it exceeds `5%`, it indicates severe systemic database issues.
*   **High Influence Anomaly:** These rows represent extreme cases that are pulling our pricing parameters off track. They must be manually audited before fitting models.

---

## 3. Frequency Model Fitting & Evaluation (`fit_and_evaluate_frequency_models`) - Notebook 03

### A. What the Function Does
This function trains and evaluates three competing pricing models to predict **how often** accidents occur: a standard **Poisson GLM**, a **Negative Binomial GLM (NB2)**, and a machine learning **XGBoost Poisson regressor**. It automatically splits the portfolio into training (80%) and testing (20%) datasets, fits the models, and reports comparative performance metrics.

### B. Why It Is Used (Actuarial & Business Context)
Actuaries must pick the most accurate and stable model to price risk. This function automates the model-comparison phase. It evaluates models using standard statistical measures (AIC and deviance), prediction errors (MAE and RMSE), and—critically—the **Actuarial Gini coefficient** (evaluating risk-differentiation power). This ensures we pick the model that best prevents adverse selection.

### C. What Inputs It Requires
*   `df` : Cleaned dataset (with anomalies removed).
*   `predictor_cols` : List of features.
*   `claim_nb_col` : Claim count target.
*   `exposure_col` : Coverage duration.

### D. Process or Calculation Performed
1.  **Split:** Splits data 80/20 into train/validation sets.
2.  **Fits Poisson GLM:** Fits statsmodels Poisson GLM with a log link, incorporating policy exposure as an offset.
3.  **Fits Negative Binomial GLM:** Fits Negative Binomial family to account for overdispersed claims count data.
4.  **Fits XGBoost:** Fits a gradient boosted trees model using a Poisson count objective, incorporating log-exposure as a base margin.
5.  **Gini Coefficient:** Sorts predictions descending, computes the Lorenz curve AUC, and calculates Gini: $\text{Gini} = 2 \times (\text{AUC} - 0.5)$.

### E. What Output It Produces
A dictionary containing:
*   `summary`: A summary table (DataFrame) reporting AIC, deviance, MAE, RMSE, and Actuarial Gini for all three models.
*   `predictions`: A DataFrame with validation targets and predictions from each candidate model.

### F. How to Interpret the Results
*   **Best Model Selection:** The model with the **lowest Deviance/AIC** and the **highest Actuarial Gini** on the test set is our winning frequency model.
*   **Actuarial Gini of 0.10 to 0.20:** Standard risk-sorting power for car insurance frequency. Accidents are highly random, so values in this range are mathematically strong.
*   **Gini > 0.40:** Indicates severe risk of data leakage (accidentally including future claim records during model training).

---

## 4. Severity Model Fitting & Evaluation (`fit_and_evaluate_severity_models`) - Notebook 04

### A. What the Function Does
This function trains and evaluates three competing models to predict the **cost of each accident** (claim size): **Gamma GLM (log link)**, **Log-Normal linear regression (with log-bias correction)**, and **Inverse Gaussian GLM**. It filters the dataset strictly to positive claims where payouts occurred ($ClaimAmount > 0$) and splits them 80/20 for testing.

### B. Why It Is Used (Actuarial & Business Context)
Claim sizes are positive-only and highly skewed (a few giant claims dominate overall payouts). Standard linear models are designed for symmetric bell-curves and fail here. Actuaries use Gamma and Inverse Gaussian GLMs because they naturally handle skewed distributions. This function compares these models using **Gamma Deviance** and the **Actual-to-Expected (A/E) ratio** to ensure overall premium adequacy and insurer solvency.

### C. What Inputs It Requires
*   `df` : Dataset containing claims.
*   `predictor_cols` : Feature columns.
*   `claim_amount_col` : Claim payout target.

### D. Process or Calculation Performed
1.  **Filter & Split:** Isolates positive claims and splits them 80/20.
2.  **Fits Gamma GLM:** Fits GLM with Gamma family and Log link.
3.  **Fits Log-Normal:** Fits linear regression on $\ln(ClaimAmount)$ and transforms predictions back using a log-bias smearing correction: $e^{\mu + \sigma^2/2}$ where $\sigma^2$ is the training residual variance.
4.  **Fits Inverse Gaussian GLM:** Fits GLM with Inverse Gaussian family and Log link.
5.  **Multiplicative Gamma Deviance:** Evaluates Unit Dispersion Gamma Deviance for each model's test predictions.

### E. What Output It Produces
A dictionary containing:
*   `summary`: A summary table reporting MAE, RMSE, global A/E ratio, Gamma deviance, and AIC equivalent for each candidate model.
*   `predictions`: A DataFrame of validation actuals and predicted severities.

### F. How to Interpret the Results
*   **Actual-to-Expected (A/E) Ratio:** Should be close to `1.0`. An A/E ratio of `1.02` means actual payouts were 2% higher than predicted (excellent global calibration). If A/E is `1.15`, the model is underestimating payouts by 15% and will cause underwriting losses.
*   **Lowest Gamma Deviance:** Indicates the model that best fits the skewed distribution shape of the claim sizes.

---

## 5. Bühlmann Credibility Calibration (`calibrate_buhlmann_credibility`) - Notebook 05

### A. What the Function Does
This function acts as a **mathematical scale balance** that blends a segment's own historical loss experience with the broader portfolio's stable baseline. It automatically calculates how much we should "trust" a segment's limited history based on its data volume (exposure).

### B. Why It Is Used (Actuarial & Business Context)
When pricing car insurance, we segment policyholders into groups (e.g., driver age bands or geographic territories). 
*   If we price a tiny segment (e.g., a rural village with only 10 drivers) *purely* on its own history, a single random claim will cause their premiums to jump 1000% next year (high variance).
*   If we ignore their local experience entirely, we ignore their unique risk profile (high bias).
**Bühlmann Credibility** mathematically solves this by calculating a trust factor ($Z$) between 0.0 and 1.0. Large segments get a high $Z$ (we price them on their own data); tiny segments get a low $Z$ (we blend their pricing with the safe, stable average of the entire portfolio).

### C. What Inputs It Requires
*   `df`: The historical policy dataset.
*   `segment_col` : The column representing risk segments (e.g., 'Risk_Band', 'Region').
*   `exposure_col` : The column representing years of policy coverage.
*   `observed_loss_col` : The actual total claim costs.
*   `predicted_loss_col` : The baseline predicted claim costs (prior Pure Premium).
*   `K` *(Credibility Parameter)*: The "credibility constant." It represents the volume of exposure a segment needs to reach exactly 50% credibility ($Z = 0.50$). It is set by the actuarial committee based on process variance and parameter variance.

### D. Process or Calculation Performed
1.  **Credibility Factor ($Z$):** For each segment, it calculates:
    $$Z = \frac{\text{Exposure}}{\text{Exposure} + K}$$
2.  **Blending:** It calculates a credibility-weighted Pure Premium:
    $$\text{Credibility Premium} = Z \times (\text{Segment Observed Cost}) + (1 - Z) \times (\text{Portfolio Average Cost})$$
3.  **Risk Adjustment Factor (RAF):** It computes the multiplier applied to our baseline predictions:
    $$\text{RAF} = Z \times \left(\frac{\text{Actual Losses}}{\text{Predicted Losses}}\right) + (1 - Z) \times 1.0$$
4.  **Revenue Neutrality:** Because blending shifts the total pricing pool, it calculates a global **Correction Factor**:
    $$\text{Correction Factor} = \frac{\text{Total Observed Losses}}{\text{Total Credibility-Blended Premiums}}$$
    It multiplies all segment premiums and RAFs by this factor so the final portfolio premium remains perfectly balanced.

### E. What Output It Produces
A dictionary containing:
*   `segment_metrics`: A summary table (DataFrame) reporting each segment's total exposure, observed loss, credibility score ($Z$), unadjusted RAF, and revenue-neutral adjusted RAF.
*   `correction_factor`: The portfolio-wide balancing multiplier.

### F. How to Interpret the Results
*   **Credibility ($Z$) near 1.0:** The segment has extensive data. We trust its local experience completely.
*   **Credibility ($Z$) near 0.0:** The segment has very thin data. We ignore its local volatility and price it using the portfolio average.
*   **Adjusted RAF > 1.0:** The segment has historically run worse than predicted, even after adjusting for credibility. Their premiums are adjusted upward.
*   **Adjusted RAF < 1.0:** The segment has historically run better than predicted, receiving a credibility-weighted premium discount.

---

## 6. Commercial Premium Formula (`calculate_commercial_premium`) - Notebook 06

### A. What the Function Does
This function is the **final commercial calculator** that converts the raw mathematical risk cost (Pure Premium) into the final commercial price printed on a customer's policy schedule.

### B. Why It Is Used (Actuarial & Business Context)
We cannot sell insurance strictly at "cost" (Pure Premium). If we did, the company would go bankrupt during a catastrophic storm, and we would have no money to cover corporate salaries, underwriter expenses, or profit requirements. \nThis function takes the raw risk cost and applies critical business, safety, and legal loadings:
1.  **Large Loss Loading ($L$):** Protects the insurer against extreme, unpredictable, catastrophic claims.
2.  **Profit Margin ($M$):** Covers general operating expenses and targeted shareholder return.
3.  **Risk Adjustment Factor ($\text{RAF}$):** Applies the credibility-weighted segment adjustment.
4.  **Premium Floor & Cap:** Floor ensures we cover the minimum administrative cost of issuing any policy; Cap prevents premiums from reaching extreme, socially unacceptable, or unmarketable levels.

### C. What Inputs It Requires
*   `predicted_freq`: Predicted claim frequency from your frequency model.
*   `predicted_sev`: Predicted claim severity from your severity model.
*   `risk_adjustment_factor`: Credibility-adjusted segment RAF (from Bühlmann calibration).
*   `large_loss_loading` *(Safety Loading)*: Multiplier for cat provisions (default is `1.10` or a 10% load).
*   `profit_margin` *(Underwriting Margin)*: Multiplier for expenses and profit (default is `1.05` or a 5% margin).
*   `premium_floor`: The minimum premium allowed (default is `50.00`).
*   `premium_cap`: The maximum premium allowed (default is `5,000.00`).

### D. Process or Calculation Performed
1.  **Pure Premium Calculation:**
    $$\text{Pure Premium} = \text{Predicted Frequency} \times \text{Predicted Severity}$$
2.  **Multiplicative Pricing Loadings:**
    $$\text{Gross Premium} = \text{Pure Premium} \times L \times \text{RAF} \times M$$
3.  **Clipping (Floor and Cap Enforcement):**
    $$\text{Final Premium} = \text{clip}(\text{Gross Premium}, \text{Min} = \text{Floor}, \text{Max} = \text{Cap})$$

### E. What Output It Produces
A structured dictionary containing:
*   `final_premium`: The final commercial premium billed to each customer.
*   `pure_premium`: The base expected cost of claims.
*   `gross_premium`: The commercial premium before any clipping is applied.
*   `portfolio_metrics`: A summary dictionary of portfolio-level pricing stats (total premium collected, average premium, average pure premium, count and percentage of policies limited by the floor or the cap).

### F. How to Interpret the Results
*   **Portfolio Metrics - Policies at Floor:** If a very high percentage of policies are sitting at the floor (e.g., >30%), our baseline risk pricing is too low, or our floor is too high. We are charging these policyholders more than their true risk.
*   **Portfolio Metrics - Policies at Cap:** If many policies hit the cap, we are clipping our revenue on extremely high-risk drivers. This exposes the insurer to **adverse selection**, as high-risk drivers will flock to us because we are under-charging them relative to their true risk.
*   **Total Premium Collected:** Must reconcile against the general ledger and the company's annual budget plan.

---

## 7. Autonomous Agentic Validation Simulation (`simulate_agent_validation_audit`) - Notebook 07

### A. What the Function Does
This function represents our **programmatic AI Auditor Panel**. It simulates the five specialized validation agents from Notebook 7 (Data Profiling, Frequency, Severity, Credibility, and Chief Actuary Auditor) and executes hard-coded programmatic sanity checks on our pricing outputs.

### B. Why It Is Used (Actuarial & Business Context)
Under strict solvency and insurance guidelines, no model can be deployed without documented validation and sign-off. This function implements a robust automated "first line of defense" in our MLOps and Model Risk Management pipeline. It automatically catches parameter failures, low risk-sorting scores, or calibration drifts, preventing corrupted model results from reaching production.

### C. What Inputs It Requires
It consumes the aggregate diagnostic dictionaries from all preceding steps:
*   `df_profile_meta` (from Step 1)
*   `anomaly_metrics` (from Step 2)
*   `frequency_metrics` (from Step 3)
*   `severity_metrics` (from Step 4)
*   `credibility_metrics` (from Step 5)
*   `commercial_metrics` (from Step 6)

### D. Process or Calculation Performed
1.  **Data Profiling Audit:** Verifies row counts are positive and that schemas were correctly validated.
2.  **Frequency Model Audit:** Verifies that our winning model's Gini score is at or above the standard baseline threshold ($\text{Gini} \ge 0.10$).
3.  **Severity Model Audit:** Verifies that our severity model is globally calibrated, checking if the Actual-to-Expected (A/E) ratio is within the standard tolerance interval of $[0.95, 1.05]$.
4.  **Credibility & Underwriting Audit:** Audits the Bühlmann blending phase, checking if the revenue-neutral correction factor is within tolerance $[0.98, 1.02]$.
5.  **Chief Actuary Sign-Off Assembly:** Aggregates checks. If any agent failed, the Chief Actuary issues a `"REJECTED FOR MANUAL AUDIT"` decision and logs descriptive warnings. If all checks passed, the portfolio is `"APPROVED"` and signed.

### E. What Output It Produces
A highly structured dictionary containing individual reports for each of the five validation agents, detailing their pass/fail status, descriptive comments, checklists, and the final Chief Actuary sign-off decision (with warnings, if any).

### F. How to Interpret the Results
*   **APPROVED:** The portfolio is completely healthy. Row counts reconcile, the risk-sorting power is strong, calibration is perfect, blending is revenue-neutral, and the rates are safe to file.
*   **REJECTED FOR MANUAL AUDIT:** A critical actuarial standard has been violated. Check the logged warnings to identify which step of the pipeline failed and adjust model assumptions accordingly.
