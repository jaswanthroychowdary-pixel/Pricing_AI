"""
etl_pipeline.py

Idempotent, actuarial-grade ETL pipeline implementing Phase 2 of the production deployment.
Initializes SQLite database schemas, seeds synthetic portfolio data (with simulated anomalies),
runs multi-layer data cleaning, implements a dead-letter quarantine queue, and executes
strict mathematical reconciliation audits to guarantee compliance with ASOP 23 and ASOP 56.
"""

import os
import sys
import sqlite3
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Ensure local directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import secure config and core pricing functions
from config_setup import config, logger
from pricing_functions_v2 import profile_data_and_synthesize, detect_anomalies_pipeline

class ETLPipeline:
    """
    Automated ETL engine executing ingestion, validation, quarantine, and reconciliation.
    Ensures that data-quality signals are generated programmatically.
    """
    
    def __init__(self):
        self.db_path = config.db_path
        self.version = config.VERSION
        self.predictor_cols = ["Age", "CarVal", "Power"]  # Standard risk features

    def get_connection(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite claims warehouse."""
        return sqlite3.connect(self.db_path)

    def initialize_schemas(self):
        """
        Creates SQLite tables to enforce structured, auditable schema design.
        Separates Raw data, Quarantined anomalies (DLQ), and Clean data.
        """
        logger.info("Initializing claims warehouse SQLite database schemas...")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Raw Ingestion Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_portfolio (
                    id INTEGER PRIMARY KEY,
                    PolicyID TEXT UNIQUE,
                    Age REAL,
                    CarVal REAL,
                    Power REAL,
                    ClaimNb INTEGER,
                    ClaimAmount REAL,
                    Exposure REAL,
                    Risk_Band TEXT,
                    source_system_id TEXT,
                    ingestion_timestamp TEXT
                );
            """)

            # 2. Clean Portfolio Table (Ready for pricing)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cleaned_portfolio (
                    PolicyID TEXT PRIMARY KEY,
                    Age REAL,
                    CarVal REAL,
                    Power REAL,
                    ClaimNb INTEGER,
                    ClaimAmount REAL,
                    Exposure REAL,
                    Risk_Band TEXT,
                    transformation_version TEXT,
                    cleaned_timestamp TEXT
                );
            """)

            # 3. Dead-Letter Queue (Quarantined Anomalies)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarantine_anomalies (
                    PolicyID TEXT PRIMARY KEY,
                    Age REAL,
                    CarVal REAL,
                    Power REAL,
                    ClaimNb INTEGER,
                    ClaimAmount REAL,
                    Exposure REAL,
                    Risk_Band TEXT,
                    failure_reasons TEXT,
                    quarantine_timestamp TEXT
                );
            """)

            # 4. ETL Audit Log Ledger
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS etl_audit_log (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT,
                    total_raw_rows INTEGER,
                    clean_rows_inserted INTEGER,
                    quarantined_rows_inserted INTEGER,
                    raw_exposure_sum REAL,
                    reconciled_exposure_sum REAL,
                    raw_payout_sum REAL,
                    reconciled_payout_sum REAL,
                    reconciliation_status TEXT,
                    transformation_version TEXT,
                    error_message TEXT
                );
            """)
            conn.commit()
        logger.info("Claims warehouse schemas successfully initialized.")

    def seed_synthetic_data_if_empty(self, row_count: int = 1000):
        """
        Generates and seeds synthetic motor policy data to simulate a raw claims warehouse.
        Deliberately injects data-quality anomalies to test validation filters.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raw_portfolio")
            if cursor.fetchone()[0] > 0:
                logger.info("Raw claims warehouse already has records. Skipping synthetic data seeding.")
                return

        logger.info(f"Generating and seeding {row_count} synthetic policy records with anomalies...")
        np.random.seed(42)
        
        policy_ids = [f"POL-{i:06d}" for i in range(1, row_count + 1)]
        ages = np.random.normal(42, 12, row_count).clip(17, 85)
        car_vals = np.random.exponential(15000, row_count).clip(2000, 150000)
        powers = np.random.normal(120, 40, row_count).clip(50, 450)
        
        # Simulating claims frequency (rare count events)
        claim_nbs = np.random.poisson(0.04, row_count)
        
        # Simulating claim size severity (positive, highly skewed Gamma)
        claim_amounts = np.zeros(row_count)
        pos_claim_mask = claim_nbs > 0
        claim_amounts[pos_claim_mask] = np.random.gamma(2.0, 500, np.sum(pos_claim_mask))
        
        exposures = np.random.uniform(0.1, 1.0, row_count).clip(0.1, 1.0)
        risk_bands = np.random.choice(["Band_A", "Band_B", "Band_C"], row_count)

        df_raw = pd.DataFrame({
            "PolicyID": policy_ids,
            "Age": ages,
            "CarVal": car_vals,
            "Power": powers,
            "ClaimNb": claim_nbs,
            "ClaimAmount": claim_amounts,
            "Exposure": exposures,
            "Risk_Band": risk_bands
        })

        # Injecting Explicit Data Quality and Actuarial Anomalies (Phase 2 test)
        # 1. Negative claims count (Logical Error)
        df_raw.loc[10, "ClaimNb"] = -2
        # 2. Claims payout without claim count (Logical Error)
        df_raw.loc[25, "ClaimNb"] = 0
        df_raw.loc[25, "ClaimAmount"] = 3500.00
        # 3. Negative claim amount
        df_raw.loc[50, "ClaimAmount"] = -250.00
        # 4. Extreme multi-dimensional outliers for Isolation Forest
        df_raw.loc[100, ["Age", "CarVal", "Power"]] = [18.0, 145000.0, 440.0]
        # 5. High-influence outliers (High Leverage & High Deviance Residual)
        df_raw.loc[200, ["Age", "CarVal", "Power", "ClaimNb", "ClaimAmount"]] = [85.0, 120000.0, 420.0, 5, 85000.0]

        df_raw["source_system_id"] = "POL_ADMIN_SYS_NORTH"
        df_raw["ingestion_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Write to local database
        with self.get_connection() as conn:
            df_raw.to_sql("raw_portfolio", conn, if_exists="append", index=False)
        logger.info("Raw synthetic policy records successfully loaded.")

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Executes the main pipeline: Ingests raw data, profiles, detects anomalies,
        quarantines outliers, and performs mathematical reconciliation audits.
        """
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(f"Starting ETL Pipeline Execution at {run_timestamp}")

        # 1. Fetch Raw Ingestion Data
        with self.get_connection() as conn:
            df_raw = pd.read_sql("SELECT * FROM raw_portfolio", conn)
            
        total_raw_rows = len(df_raw)
        raw_exposure_sum = float(df_raw["Exposure"].sum())
        raw_payout_sum = float(df_raw["ClaimAmount"].sum())
        
        logger.info(f"Loaded {total_raw_rows} records from raw ingestion table.")

        # 2. Run Data Profiling & Exposure Synthesis (ASOP 23 checks)
        profile_results = profile_data_and_synthesize(
            df=df_raw,
            claim_nb_col="ClaimNb",
            claim_amount_col="ClaimAmount",
            exposure_col="Exposure"
        )
        df_profiled = profile_results["df"]
        meta_profiled = profile_results["meta"]
        
        logger.info(f"Profiling complete. Total synthesized exposures: {meta_profiled['total_exposure_years']:.2f} years.")

        # 3. Execute End-to-End Anomaly Detection Pipeline (Phase 2 diagnostics)
        anomaly_results = detect_anomalies_pipeline(
            df=df_profiled,
            predictor_cols=self.predictor_cols,
            claim_nb_col=meta_profiled["claim_nb_column_used"],
            claim_amount_col=meta_profiled["claim_amount_column_used"],
            contamination_rate=0.012  # Outlier sensitivity
        )
        df_analyzed = anomaly_results["df"]
        metrics = anomaly_results["metrics"]

        logger.info(f"Data audit complete. Total anomalies flagged: {metrics['global_flagged_count']} ({metrics['global_flagged_pct']:.2f}% of portfolio)")

        # 4. Separate Clean Portfolio and Quarantined Anomalies
        df_clean = df_analyzed[df_analyzed["global_anomaly_flag"] == False].copy()
        df_quarantine = df_analyzed[df_analyzed["global_anomaly_flag"] == True].copy()

        # Consolidate failure reasons for quarantine logs
        def get_failure_reasons(row):
            reasons = []
            if row["dq_flag"]:
                reasons.append("Logical DQ Violation (negative or zero-contradiction)")
            if row["tail_outlier_flag"]:
                reasons.append("Extreme Tail Outlier (>P99.9 Payout)")
            if row["iso_outlier_flag"]:
                reasons.append("Multi-Dimensional Feature Outlier (Isolation Forest)")
            if row["influence_flag"]:
                reasons.append("High-Influence Regression Outlier (Leverage/Deviance)")
            return ", ".join(reasons)

        df_quarantine["failure_reasons"] = df_quarantine.apply(get_failure_reasons, axis=1)
        
        # Clean columns to match database schema signatures
        df_clean_db = df_clean[["PolicyID", "Age", "CarVal", "Power", "ClaimNb", "ClaimAmount", "Exposure", "Risk_Band"]].copy()
        df_clean_db["transformation_version"] = self.version
        df_clean_db["cleaned_timestamp"] = run_timestamp

        df_quarantine_db = df_quarantine[["PolicyID", "Age", "CarVal", "Power", "ClaimNb", "ClaimAmount", "Exposure", "Risk_Band", "failure_reasons"]].copy()
        df_quarantine_db["quarantine_timestamp"] = run_timestamp

        # Establish transaction boundary for safe writes
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # 5. Idempotent Ingestion (Clear target tables first to prevent duplicate keys)
            cursor.execute("DELETE FROM cleaned_portfolio")
            cursor.execute("DELETE FROM quarantine_anomalies")
            conn.commit()

            # Write clean data
            df_clean_db.to_sql("cleaned_portfolio", conn, if_exists="append", index=False)
            # Write quarantined data (DLQ)
            df_quarantine_db.to_sql("quarantine_anomalies", conn, if_exists="append", index=False)
            
            # 6. Execute Strict Mathematical Reconciliation Audits (Reconciliation Gate)
            clean_rows = len(df_clean_db)
            quarantine_rows = len(df_quarantine_db)
            
            recon_row_sum = clean_rows + quarantine_rows
            recon_exposure_sum = float(df_clean_db["Exposure"].sum() + df_quarantine_db["Exposure"].sum())
            recon_payout_sum = float(df_clean_db["ClaimAmount"].sum() + df_quarantine_db["ClaimAmount"].sum())

            # Perform assertions (floating precision tolerance)
            row_match = total_raw_rows == recon_row_sum
            exposure_match = abs(raw_exposure_sum - recon_exposure_sum) < 1e-4
            payout_match = abs(raw_payout_sum - recon_payout_sum) < 1e-4

            if row_match and exposure_match and payout_match:
                reconciliation_status = "SUCCESS"
                error_msg = None
                logger.info("=== RECONCILIATION PASSED SUCCESSFULLY ===")
                logger.info(f"  - Rows: Ingested={total_raw_rows} | Reconciled={recon_row_sum}")
                logger.info(f"  - Exposures: Ingested={raw_exposure_sum:.4f} | Reconciled={recon_exposure_sum:.4f}")
                logger.info(f"  - Payouts: Ingested=GBP {raw_payout_sum:.2f} | Reconciled=GBP {recon_payout_sum:.2f}")
            else:
                reconciliation_status = "FAILED_RECONCILIATION"
                error_msg = f"Reconciliation balance mismatch. Rows: {row_match}, Exposure: {exposure_match}, Payouts: {payout_match}"
                raise ValueError(error_msg)

            # Log execution details to the Audit Log Ledger (Provenance)
            cursor.execute("""
                INSERT INTO etl_audit_log (
                    run_timestamp, total_raw_rows, clean_rows_inserted, quarantined_rows_inserted,
                    raw_exposure_sum, reconciled_exposure_sum, raw_payout_sum, reconciled_payout_sum,
                    reconciliation_status, transformation_version, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                run_timestamp, total_raw_rows, clean_rows, quarantine_rows,
                raw_exposure_sum, recon_exposure_sum, raw_payout_sum, recon_payout_sum,
                reconciliation_status, self.version, error_msg
            ))
            conn.commit()

        except Exception as e:
            conn.rollback()
            reconciliation_status = "FAILED_RECONCILIATION"
            logger.error(f"Reconciliation failed or database transaction aborted. Rolling back changes: {e}")
            
            # Log failure to ledger
            cursor.execute("""
                INSERT INTO etl_audit_log (
                    run_timestamp, total_raw_rows, clean_rows_inserted, quarantined_rows_inserted,
                    raw_exposure_sum, reconciled_exposure_sum, raw_payout_sum, reconciled_payout_sum,
                    reconciliation_status, transformation_version, error_message
                ) VALUES (?, 0, 0, 0, ?, 0, ?, 0, ?, ?, ?);
            """, (
                run_timestamp, total_raw_rows, raw_exposure_sum, raw_payout_sum,
                reconciliation_status, self.version, str(e)
            ))
            conn.commit()
            raise e
        finally:
            conn.close()

        return {
            "run_timestamp": run_timestamp,
            "status": reconciliation_status,
            "metrics": {
                "total_rows_ingested": total_raw_rows,
                "clean_rows_saved": clean_rows,
                "quarantined_rows_saved": quarantine_rows,
                "exposure_reconciled": exposure_match,
                "payouts_reconciled": payout_match
            }
        }

if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.initialize_schemas()
    pipeline.seed_synthetic_data_if_empty()
    pipeline_result = pipeline.run_pipeline()
    print("ETL Pipeline Execution Complete:", pipeline_result)
