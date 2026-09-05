# 🛡️ Agentic Actuarial Pricing Engine

An end-to-end, actuarial-grade **Motor Insurance Pricing Engine** built with Python and designed to run cell-by-cell in Jupyter notebooks. 

The architecture follows modern casualty actuarial practices (CAS/SOA standards) combined with machine learning benchmarks and autonomous AI validation agents.

---

## 🏗️ Architecture & Pipeline Overview

```
                        RAW DATA (Pricing_Data.xlsx)
                                     │
                                     ▼
                      01_Data_Profiling.ipynb
                   (Schema, Exposures, Metadata)
                                     │
                                     ▼
                    02_Anomaly_Detection.ipynb
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │  • Data Quality & Tail Analysis (P99+ Percentiles)               │
   │  • Isolation Forest (Predictor Space)                            │
   │  • Leverage Scoring (Hat Matrix Diagonals)                        │
   │  • Poisson Deviance Residuals & Influence Analysis                │
   │  • Business Review Action (Genuine → KEEP / Error → REMOVE)       │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
                     03_Frequency_Models.ipynb
         (Poisson GLM  vs  NegBinomial GLM  vs  XGBoost)
             └─► Evaluation: AIC, Deviance, MAE, RMSE, Gini
             └─► 👤 USER PICKS WINNING MODEL (Cell 3.8)
                                     │
                                     ▼
                     04_Severity_Models.ipynb
        (Gamma GLM  vs  Log-Normal WLS  vs  Inverse Gaussian)
             └─► Evaluation: AIC, Deviance, MAE, RMSE, Overall A/E
             └─► 👤 USER PICKS WINNING MODEL (Cell 4.7)
                                     │
                                     ▼
                  05_Credibility_Calibration.ipynb
            (Bühlmann Empirical Bayes Credibility: Z = n / (n + K))
            (Risk Adjustment Factor & Revenue-Neutral Calibration)
                                     │
                                     ▼
                      06_Final_Premium.ipynb
     (Commercial Premium Formula, Floor/Cap, A/E Deciles, Excel Export)
                                     │
                                     ▼
                     07_Agent_Validation.ipynb
        (5 Autonomous Gemini AI Agents Audit & Sign-off on Pricing)
```

---

## 📓 Notebook Directory

The pipeline consists of **7 sequential notebooks**:

| # | Notebook | Key Methodologies & Objectives | Primary Outputs |
|:---:|---|---|---|
| **01** | [`01_Data_Profiling.ipynb`](notebooks/01_Data_Profiling.ipynb) | • Auto-detects target columns (`ClaimNb`, `ClaimAmount`)<br>• Synthesizes `Exposure = 1.0` if not present<br>• Profiles data distributions and missing values | `outputs/df_profiled.parquet`<br>`outputs/meta.json` |
| **02** | [`02_Anomaly_Detection.ipynb`](notebooks/02_Anomaly_Detection.ipynb) | • **Data Quality Check:** Nulls, negative values, logical sanity<br>• **Distribution Tail Analysis:** Multi-percentile breakdown (`P50` to `P99.9`)<br>• **Isolation Forest:** Multi-dimensional anomaly scoring on predictors<br>• **Leverage Scoring:** Hat matrix diagonal $H_{ii} = x_i (X^T X)^{-1} x_i^T$<br>• **Residual Diagnostics:** Poisson deviance residuals ($r_i^D$)<br>• **Influence Check:** High leverage × high deviance residual<br>• **Business Review:** Classifies records into KEEP or REMOVE | `outputs/df_clean.parquet`<br>`outputs/anomaly_report.csv`<br>`outputs/iso_preprocessor.pkl` |
| **03** | [`03_Frequency_Models.ipynb`](notebooks/03_Frequency_Models.ipynb) | • 80/20 stratified train/validation split<br>• Fits **Poisson GLM**, **Negative Binomial GLM (NB2)**, and **XGBoost (Poisson objective)**<br>• Evaluates AIC, Deviance, MAE, RMSE, and **Actuarial Gini (Lorenz AUC)**<br>• **User Choice:** Interactive cell to select the frequency model | `outputs/df_with_freq.parquet`<br>`outputs/freq_model.pkl`<br>`outputs/frequency_results.json` |
| **04** | [`04_Severity_Models.ipynb`](notebooks/04_Severity_Models.ipynb) | • Filters strictly to positive claims (`ClaimAmount > 0`)<br>• Fits **Gamma GLM (log link)**, **Log-Normal (WLS + bias correction)**, and **Inverse Gaussian GLM**<br>• Evaluates AIC, Deviance, MAE, RMSE, and Overall A/E<br>• **User Choice:** Interactive cell to select the severity model | `outputs/df_with_severity.parquet`<br>`outputs/sev_model.pkl`<br>`outputs/severity_results.json` |
| **05** | [`05_Credibility_Calibration.ipynb`](notebooks/05_Credibility_Calibration.ipynb) | • Computes Pure Premium: E[Pure Premium] = E[Freq] × E[Sev]<br>• Segments portfolio into risk bands<br>• Applies **Bühlmann Credibility**: $Z = n / (n + K)$<br>• Adjusts credibility factor for revenue neutrality | `outputs/df_with_adj.parquet`<br>`outputs/credibility_model.pkl`<br>`outputs/credibility_results.json` |
| **06** | [`06_Final_Premium.ipynb`](notebooks/06_Final_Premium.ipynb) | • Evaluates final commercial premium formula<br>• Enforces actuarial **Premium Floor** and **Cap**<br>• Generates premium distribution and A/E ratio decile chart<br>• Exports final priced portfolio to Excel and registers parameters | `outputs/df_final_premiums.parquet`<br>`outputs/FINAL_priced.xlsx`<br>`outputs/pricing_registry.json` |
| **07** | [`07_Agent_Validation.ipynb`](notebooks/07_Agent_Validation.ipynb) | • Multi-agent AI audit using Google Gemini API<br>• **5 Specialized Actuarial Agents:**<br>&nbsp;&nbsp;1. Data Profiling Agent<br>&nbsp;&nbsp;2. Frequency Modeling Agent<br>&nbsp;&nbsp;3. Severity Modeling Agent<br>&nbsp;&nbsp;4. Credibility & Underwriting Agent<br>&nbsp;&nbsp;5. Chief Actuary / Governance Auditor | `outputs/agent_validation_report.json` |

---

## 💰 Commercial Pricing Formula

Every policy in the portfolio is priced according to:

$$\text{Final Premium} = \text{clip}\Big(\underbrace{\mathbb{E}[\text{Freq}] \times \mathbb{E}[\text{Sev}]}_{\text{Pure Premium}} \times \underbrace{L}_{\text{Large Loss Loading}} \times \underbrace{\text{RAF}}_{\text{Risk Adj Factor}} \times \underbrace{M}_{\text{Profit Margin}}, \;\text{Floor}, \;\text{Cap}\Big)$$

### Default Tunable Parameters:
* **Large Loss Loading ($L$):** `1.10` (10% catastrophic loss provision)
* **Profit Margin ($M$):** `1.05` (5% target commercial underwriting margin)
* **Premium Floor:** `50.00`
* **Premium Cap:** `5,000.00`

---

## 📊 Data Directory Structure

```
Pricing_AI/
├── data/
│   └── raw/
│       └── Pricing_Data.xlsx          # Historical policy & claims portfolio (~111.6k rows)
├── docs/
│   └── actuarial_functions_doc.md     # Comprehensive plain-English Actuarial Reference Manual
├── notebooks/
│   ├── 01_Data_Profiling.ipynb
│   ├── 02_Anomaly_Detection.ipynb
│   ├── 03_Frequency_Models.ipynb
│   ├── 04_Severity_Models.ipynb
│   ├── 05_Credibility_Calibration.ipynb
│   ├── 06_Final_Premium.ipynb
│   └── 07_Agent_Validation.ipynb
├── outputs/                           # Generated Parquet, Excel, JSON, and PNG artifacts
├── src/
│   ├── pricing_functions.py           # Unified central facade re-exporting all 28 tools
│   └── tools/                         # Modular tool modules (<= 6 tools per file)
│       ├── profiling_tools.py         # 3 tools for Notebook 01
│       ├── anomaly_tools.py           # 5 tools for Notebook 02
│       ├── frequency_tools.py         # 5 tools for Notebook 03
│       ├── severity_tools.py          # 4 tools for Notebook 04
│       ├── credibility_tools.py       # 4 tools for Notebook 05
│       ├── premium_tools.py           # 4 tools for Notebook 06
│       └── validation_tools.py        # 3 tools for Notebook 07
├── tests/
│   └── test_all_tools.py              # Automated test suite for all 28 tools
├── requirements.txt                   # Environment dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup
Make sure you have Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
pip install pyarrow openpyxl scikit-learn xgboost statsmodels plotly
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/
```

### 3. Execution Order
Run the notebooks sequentially from **01** to **07**.
* In **Notebook 3 (Cell 3.8)**: Review the model comparison table and Lorenz curve, then specify your chosen frequency model (`"Poisson GLM"`, `"NegBinomial"`, or `"XGBoost"`).
* In **Notebook 4 (Cell 4.7)**: Review the severity metrics and residual plots, then specify your chosen severity model (`"Gamma GLM"`, `"Log-Normal"`, or `"Inv Gaussian"`).
* In **Notebook 6**: The completed commercial portfolio will be saved directly to `outputs/FINAL_priced.xlsx` for business review and underwriting deployment.
* In **Notebook 7**: Provide your Google Gemini API key to receive the automated 5-agent actuarial sign-off report.
