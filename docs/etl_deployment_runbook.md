# Deploying the Agentic Pricing Data Pipeline

This deployment guide establishes the operational protocols for taking the **Motor Insurance Pricing Engine** from Jupyter Notebook prototypes to an enterprise-grade production environment. It walks through the active implementation of **Phase 1 (Secure Secret Management)** and **Phase 2 (Idempotent ETL Pipeline with Mathematical Reconciliation)**.

---

## Architecture & Data Flow

```
                      RAW DATA (Pricing_Data.xlsx)
                                   │
                                   ▼
                        01_Data_Profiling.ipynb
                     (Schema, Exposures, Metadata)
                                   │
                                   ▼
                      02_Anomaly_Detection.ipynb
     ┌─────────────────────────────┴─────────────────────────────┐
     │  • Logical Data Quality check                             │
     │  • Isolation Forest Multi-Dimensional scoring             │
     │  • Hat Matrix Leverage Diagonals                          │
     │  • Poisson Deviance Residual diagnostics                  │
     └─────────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                   `etl_pipeline.py` Ingestion Step
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
      [RECONCILIATION GATE]                     [DEAD-LETTER QUEUE]
     `cleaned_portfolio` table               `quarantine_anomalies` table
   (Pristine rows for GLM training)         (Flagged with Failure Reasons)
```

---

## Phase 1: Secure Secret & Configuration Management (`config_setup.py`)

In production, secret credentials and configurations must be kept strictly separate from codebase execution. Hardcoding keys triggers critical security flags and violates corporate governance standards.

Our configuration manager (`config_setup.py`) establishes this barrier:
1.  **Environment Separation:** It automatically scans the server's runtime environment for the `GOOGLE_API_KEY`, database file paths, and default model parameters.
2.  **Strict Pricing Parameter Validation:** Under **ASOP 56 (Modeling)**, it runs automated assertions on initialization (e.g. verifying that the Large Loss Loading $L \ge 1.0$, the Profit Margin $M \ge 0.90$, and that the Premium Floor is logically lower than the Premium Cap). Any invalid configuration halts the application before any policy is mispriced.
3.  **Audit Masking:** Provides a `.get_summary_dictionary()` method that returns masked parameter logs, preventing secrets from leaking into raw text logs or downstream AI agent traces.

---

## Phase 2: Idempotent SQL Ingestion & Quarantine Pipeline (`etl_pipeline.py`)

A raw Excel sheet is highly vulnerable to data corruption. Our SQL-based ingestion pipeline translates your preprocessing notebooks into an automated, highly secure production workflow:

1.  **Database Separation (SQLite):** Initializes standard schemas separating the raw transactions (`raw_portfolio`) from the clean rows (`cleaned_portfolio`) and the quarantined anomalies (`quarantine_anomalies`).
2.  **Multi-Layer Quality Screening:** Merges your Jupyter Notebook diagnostics into a programmatic gatekeeper:
    *   **Logical DQ Checks:** Flags and drops negative claims, negative payouts, or payouts occurring without a recorded claim.
    *   **Tail Percentile Filters:** Caps payouts exceeding the extreme $P_{99.9}$ distribution threshold.
    *   **Isolation Forest:** Drops multi-dimensional predictor outliers.
    *   **Leverage & Influence:** Flags extreme anomalies that would warp regression coefficients.
3.  **The Dead-Letter Queue (DLQ):** Rather than silently deleting records, anomalous rows are written into `quarantine_anomalies` alongside a string explicitly detailing the failure reason (e.g. `Logical DQ Violation`, `Extreme Tail Outlier`, etc.), creating a secure queue for human-in-the-loop triage.
4.  **Strict Reconciliation (ASOP 23 Compliance):** At the end of every pipeline run, the script executes three strict mathematical audits:
    $$\text{Total Raw Rows} == \text{Total Clean Rows} + \text{Total Quarantined Rows}$$
    $$\sum \text{Raw Exposures} == \sum \text{Clean Exposures} + \sum \text{Quarantined Exposures}$$
    $$\sum \text{Raw Payouts} == \sum \text{Clean Payouts} + \sum \text{Quarantined Payouts}$$
    If any check deviates by even a fraction of a cent, the database transaction is **immediately rolled back**, a `FAILED_RECONCILIATION` flag is written to the ledger, and the pipeline halts to protect downstream models.

---

## Deployment & Verification Runbook

### Step 1: Set Up Your Local Environment
First, ensure that your local shell or `.env` configuration file has the correct environment variables injected:
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
export DATABASE_PATH="/workspace/scratch/warehouse.db"
export LARGE_LOSS_LOADING="1.10"
export PROFIT_MARGIN="1.05"
```

### Step 2: Initialize & Execute the Pipeline
Run the ETL script from your terminal. It will automatically initialize the database schemas, seed the synthetic portfolio, execute the multi-layer diagnostics, quarantine the failures, and verify the reconciliation gates:
```bash
python3 /workspace/scratch/etl_pipeline.py
```

### Step 3: Audit Execution Logs
Query the audit log database to verify the run status and examine quarantined records:
```bash
python3 /workspace/scratch/verify_db.py
```

This guarantees 100% compliance with data-quality and modeling standards, setting up your pricing engine for fully automated, autonomous AI verification.
