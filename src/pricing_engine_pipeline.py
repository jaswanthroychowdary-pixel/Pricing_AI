"""
pricing_engine_pipeline.py

Phase 4 of the Production Pricing Engine Architecture.
Implements Bühlmann Empirical Bayes Credibility calibration across risk bands,
applies revenue-neutral portfolio balancing, executes the final Commercial Premium Formula
with large loss loadings and profit margins, writes final_priced_portfolio to SQLite,
and exports an auditable multi-sheet Excel production ledger to outputs/FINAL_priced.xlsx.
Complies with ASOP 12 (Risk Classification) and ASOP 56 (Modeling).
"""

import os
import sys
import json
import sqlite3
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure local module access
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_setup import config, logger
from pricing_functions_v2 import calibrate_buhlmann_credibility, calculate_commercial_premium

class PricingEnginePipeline:
    """
    Automated Credibility Calibration and Commercial Premium Engine (Phase 4).
    Transforms modeled risk costs into fully-calibrated, commercial premiums.
    """

    def __init__(self):
        self.db_path = config.db_path
        self.version = config.VERSION
        self.large_loss_loading = config.large_loss_loading
        self.profit_margin = config.profit_margin
        self.premium_floor = config.premium_floor
        self.premium_cap = config.premium_cap
        self.K_credibility = 50.0  # Buhlmann parameter
        self.outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Establishes connection to SQLite claims warehouse."""
        return sqlite3.connect(self.db_path)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Executes end-to-end Phase 4:
        1. Loads scored_portfolio from SQLite claims warehouse.
        2. Calibrates Buhlmann credibility and revenue-neutral correction across Risk Bands.
        3. Evaluates Commercial Pricing formula with Floor/Cap boundaries.
        4. Writes final_priced_portfolio table to SQLite.
        5. Exports multi-sheet production ledger to outputs/FINAL_priced.xlsx.
        6. Updates pricing_registry.json.
        """
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(f"Starting Phase 4 Commercial Pricing Pipeline at {run_timestamp}")

        # 1. Fetch Scored Portfolio from SQLite
        with self.get_connection() as conn:
            df_scored = pd.read_sql("SELECT * FROM scored_portfolio", conn)
            
        n_policies = len(df_scored)
        logger.info(f"Loaded {n_policies} scored policies from claims warehouse.")

        # 2. Buhlmann Credibility Calibration across Risk Bands
        logger.info(f"Calibrating Buhlmann Empirical Bayes Credibility (K = {self.K_credibility})...")
        cred_results = calibrate_buhlmann_credibility(
            df=df_scored,
            segment_col="Risk_Band",
            exposure_col="Exposure",
            observed_loss_col="ClaimAmount",
            predicted_loss_col="pure_premium",
            K=self.K_credibility
        )

        segment_metrics = cred_results["segment_metrics"]
        correction_factor = cred_results["correction_factor"]
        logger.info(f"Buhlmann Calibration complete. Revenue-Neutral Correction Factor = {correction_factor:.5f}")

        # Map adjusted RAF back to individual policies
        raf_map = dict(zip(segment_metrics["Risk_Band"], segment_metrics["adjusted_RAF"]))
        df_scored["RAF"] = df_scored["Risk_Band"].map(raf_map).fillna(1.0)

        # 3. Commercial Premium Formula Evaluation
        logger.info("Evaluating Commercial Premium Formula...")
        premium_results = calculate_commercial_premium(
            predicted_freq=df_scored["pred_frequency"].values,
            predicted_sev=df_scored["pred_severity"].values,
            risk_adjustment_factor=df_scored["RAF"].values,
            large_loss_loading=self.large_loss_loading,
            profit_margin=self.profit_margin,
            premium_floor=self.premium_floor,
            premium_cap=self.premium_cap
        )

        df_scored["gross_premium"] = premium_results["gross_premium"]
        df_scored["final_premium"] = premium_results["final_premium"]
        df_scored["pricing_timestamp"] = run_timestamp

        portfolio_metrics = premium_results["portfolio_metrics"]
        logger.info(f"Commercial pricing complete. Total Billed Premium: GBP {portfolio_metrics['Total_Premium_Collected_GBP']:,.2f}")
        logger.info(f"Average Policy Premium: GBP {portfolio_metrics['Average_Final_Premium_GBP']:.2f}")

        # 4. Write final_priced_portfolio to SQLite
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS final_priced_portfolio")
            conn.commit()
            df_scored.to_sql("final_priced_portfolio", conn, if_exists="replace", index=False)
            logger.info(f"Saved {n_policies} priced policies to table 'final_priced_portfolio' in claims warehouse.")

        # 5. Export Multi-Sheet Production Excel Ledger
        excel_path = os.path.join(self.outputs_dir, "FINAL_priced.xlsx")
        logger.info(f"Exporting production Excel ledger to {excel_path}...")

        audit_summary_df = pd.DataFrame([
            {"Metric": "Pipeline Version", "Value": self.version},
            {"Metric": "Execution Timestamp (UTC)", "Value": run_timestamp},
            {"Metric": "Total Policies Priced", "Value": n_policies},
            {"Metric": "Total Premium Collected (GBP)", "Value": round(portfolio_metrics["Total_Premium_Collected_GBP"], 2)},
            {"Metric": "Average Premium Billed (GBP)", "Value": round(portfolio_metrics["Average_Final_Premium_GBP"], 2)},
            {"Metric": "Average Pure Premium (GBP)", "Value": round(portfolio_metrics["Average_Pure_Premium_GBP"], 2)},
            {"Metric": "Large Loss Loading (L)", "Value": self.large_loss_loading},
            {"Metric": "Profit Margin (M)", "Value": self.profit_margin},
            {"Metric": "Revenue-Neutral Correction Factor", "Value": round(correction_factor, 5)},
            {"Metric": "Premium Floor (GBP)", "Value": self.premium_floor},
            {"Metric": "Premium Cap (GBP)", "Value": self.premium_cap},
            {"Metric": "Policies at Floor (Count)", "Value": portfolio_metrics["Policies_at_Floor_Count"]},
            {"Metric": "Policies at Floor (%)", "Value": round(portfolio_metrics["Policies_at_Floor_Pct"], 2)},
            {"Metric": "Policies at Cap (Count)", "Value": portfolio_metrics["Policies_at_Cap_Count"]},
            {"Metric": "Policies at Cap (%)", "Value": round(portfolio_metrics["Policies_at_Cap_Pct"], 2)}
        ])

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_scored.to_excel(writer, sheet_name="Priced_Portfolio", index=False)
            segment_metrics.to_excel(writer, sheet_name="Buhlmann_Segment_Metrics", index=False)
            audit_summary_df.to_excel(writer, sheet_name="Audit_Summary", index=False)

        logger.info("Multi-sheet Excel ledger successfully generated.")

        # 6. Update Pricing Registry and Credibility Results
        registry_path = os.path.join(self.outputs_dir, "pricing_registry.json")
        registry_data = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    registry_data = json.load(f)
            except Exception:
                pass

        registry_data.update({
            "large_loss_loading": self.large_loss_loading,
            "profit_margin": self.profit_margin,
            "premium_floor": self.premium_floor,
            "premium_cap": self.premium_cap,
            "buhlmann_k": self.K_credibility,
            "correction_factor": round(correction_factor, 5),
            "total_premium_billed": round(portfolio_metrics["Total_Premium_Collected_GBP"], 2),
            "mean_train_premium": round(portfolio_metrics["Average_Final_Premium_GBP"], 2),
            "mean_submit_premium": round(portfolio_metrics["Average_Final_Premium_GBP"], 2)
        })

        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        return {
            "status": "SUCCESS",
            "total_policies_priced": n_policies,
            "total_premium_collected_gbp": round(portfolio_metrics["Total_Premium_Collected_GBP"], 2),
            "average_premium_billed_gbp": round(portfolio_metrics["Average_Final_Premium_GBP"], 2),
            "revenue_neutral_correction_factor": round(correction_factor, 5),
            "excel_ledger_exported": excel_path
        }

if __name__ == "__main__":
    pipeline = PricingEnginePipeline()
    result = pipeline.run_pipeline()
    print("\nPhase 4 Commercial Pricing Pipeline Completed Successfully:")
    print(json.dumps(result, indent=2))
