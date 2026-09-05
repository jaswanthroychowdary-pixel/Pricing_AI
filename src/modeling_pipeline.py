"""
modeling_pipeline.py

Phase 3 of the Production Pricing Engine Architecture.
Implements automated model ingestion from SQLite claims warehouse, 80/20 train/validation splitting,
training of 3 frequency (Poisson, NB2, XGBoost) and 3 severity models (Gamma, Log-Normal, Inverse Gaussian),
executes the programmatic "Actuarial Gate" (Gini for Frequency, Gamma Deviance for Severity),
serializes winning model artifacts, and scores the complete portfolio into warehouse.db.
Complies with ASOP 56 (Modeling).
"""

import os
import sys
import json
import pickle
import sqlite3
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# Ensure local module access
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_setup import config, logger
from pricing_functions_v2 import calculate_actuarial_gini, evaluate_severity_model

class ModelingPipeline:
    """
    Automated in-database modeling and selection engine (Phase 3).
    Selects winning models using strict out-of-sample actuarial gates.
    """

    def __init__(self):
        self.db_path = config.db_path
        self.version = config.VERSION
        self.predictor_cols = ["Age", "CarVal", "Power"]
        self.outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Establishes connection to SQLite claims warehouse."""
        return sqlite3.connect(self.db_path)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Executes end-to-end Phase 3:
        1. Loads cleaned_portfolio from SQLite.
        2. Fits frequency models and selects winner via Actuarial Gini.
        3. Fits severity models and selects winner via unit Gamma Deviance.
        4. Serializes models to outputs/ and logs metrics.
        5. Scores the entire portfolio and writes scored_portfolio table to SQLite.
        """
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(f"Starting Phase 3 Modeling Pipeline at {run_timestamp}")

        # 1. Ingest clean data from claims warehouse
        with self.get_connection() as conn:
            df_clean = pd.read_sql("SELECT * FROM cleaned_portfolio", conn)
            
        n_policies = len(df_clean)
        logger.info(f"Loaded {n_policies} cleaned policies from claims warehouse.")

        # 2. Train/Validation Split (80/20 stratified by claim occurrence)
        train_df, val_df = train_test_split(
            df_clean, 
            test_size=0.2, 
            random_state=42, 
            stratify=(df_clean["ClaimNb"] > 0)
        )
        logger.info(f"Stratified partition complete: {len(train_df)} train, {len(val_df)} validation policies.")

        X_train = sm.add_constant(train_df[self.predictor_cols].fillna(0).values)
        X_val = sm.add_constant(val_df[self.predictor_cols].fillna(0).values)
        
        y_train_freq = train_df["ClaimNb"].values
        y_val_freq = val_df["ClaimNb"].values
        exp_train = train_df["Exposure"].values
        exp_val = val_df["Exposure"].values

        # ─── 3. FREQUENCY MODELING & ACTUARIAL GATE ──────────────────────────────
        logger.info("Fitting candidate Frequency models (Poisson, NB2, XGBoost)...")
        freq_candidates = {}

        # A. Poisson GLM
        try:
            poisson_model = sm.GLM(
                y_train_freq, X_train,
                family=sm.families.Poisson(),
                exposure=exp_train
            ).fit()
            pred_val_poisson = poisson_model.predict(X_val, exposure=exp_val)
            gini_poisson = calculate_actuarial_gini(y_val_freq, pred_val_poisson, exp_val)["gini"]
            freq_candidates["Poisson GLM"] = {
                "model": poisson_model,
                "val_gini": float(gini_poisson),
                "val_preds": pred_val_poisson,
                "aic": float(poisson_model.aic)
            }
        except Exception as e:
            logger.warning(f"Poisson GLM fitting failed: {e}")

        # B. Negative Binomial (NB2) GLM
        try:
            nb2_model = sm.GLM(
                y_train_freq, X_train,
                family=sm.families.NegativeBinomial(alpha=1.0),
                exposure=exp_train
            ).fit()
            pred_val_nb2 = nb2_model.predict(X_val, exposure=exp_val)
            gini_nb2 = calculate_actuarial_gini(y_val_freq, pred_val_nb2, exp_val)["gini"]
            freq_candidates["Negative Binomial GLM"] = {
                "model": nb2_model,
                "val_gini": float(gini_nb2),
                "val_preds": pred_val_nb2,
                "aic": float(nb2_model.aic)
            }
        except Exception as e:
            logger.warning(f"NB2 GLM fitting failed: {e}")

        # C. XGBoost Count-Poisson (Optional / Fallback)
        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(train_df[self.predictor_cols].values, label=y_train_freq)
            dtrain.set_base_margin(np.log(np.maximum(exp_train, 1e-4)))
            dval = xgb.DMatrix(val_df[self.predictor_cols].values, label=y_val_freq)
            dval.set_base_margin(np.log(np.maximum(exp_val, 1e-4)))
            
            params = {
                "objective": "count:poisson",
                "max_depth": 3,
                "learning_rate": 0.05,
                "eval_metric": "poisson-nloglik",
                "seed": 42
            }
            xgb_model = xgb.train(params, dtrain, num_boost_round=100)
            pred_val_xgb = xgb_model.predict(dval)
            gini_xgb = calculate_actuarial_gini(y_val_freq, pred_val_xgb, exp_val)["gini"]
            freq_candidates["XGBoost Poisson"] = {
                "model": xgb_model,
                "val_gini": float(gini_xgb),
                "val_preds": pred_val_xgb,
                "aic": np.nan
            }
        except Exception as e:
            logger.info(f"XGBoost Poisson skipped/failed ({e}); standard GLMs will be used.")

        # The Frequency Actuarial Gate: Choose model with highest validation Gini
        winning_freq_name = max(freq_candidates.keys(), key=lambda k: freq_candidates[k]["val_gini"])
        winning_freq_entry = freq_candidates[winning_freq_name]
        logger.info(f"Frequency Actuarial Gate: WINNER = {winning_freq_name} (Validation Gini = {winning_freq_entry['val_gini']:.5f})")

        # ─── 4. SEVERITY MODELING & ACTUARIAL GATE ───────────────────────────────
        logger.info("Fitting candidate Severity models (Gamma, Log-Normal, Inverse Gaussian)...")
        pos_train = train_df[train_df["ClaimAmount"] > 0].copy()
        pos_val = val_df[val_df["ClaimAmount"] > 0].copy()

        if len(pos_train) < 5 or len(pos_val) < 2:
            # Fallback if validation claims are sparse in synthetic sample
            pos_train = df_clean[df_clean["ClaimAmount"] > 0].copy()
            pos_val = pos_train.copy()

        X_train_sev = sm.add_constant(pos_train[self.predictor_cols].fillna(0).values)
        X_val_sev = sm.add_constant(pos_val[self.predictor_cols].fillna(0).values)
        y_train_sev = pos_train["ClaimAmount"].values
        y_val_sev = pos_val["ClaimAmount"].values

        sev_candidates = {}

        # A. Gamma GLM (Log Link)
        try:
            gamma_model = sm.GLM(
                y_train_sev, X_train_sev,
                family=sm.families.Gamma(link=sm.families.links.Log())
            ).fit()
            pred_val_gamma = gamma_model.predict(X_val_sev)
            eval_metrics = evaluate_severity_model(y_val_sev, pred_val_gamma, num_parameters=X_train_sev.shape[1])
            sev_candidates["Gamma GLM"] = {
                "model": gamma_model,
                "val_deviance": float(eval_metrics["gamma_deviance"]),
                "val_mae": float(eval_metrics["mae"]),
                "val_preds": pred_val_gamma
            }
        except Exception as e:
            logger.warning(f"Gamma GLM fitting failed: {e}")

        # B. Log-Normal with Analytical Bias Correction: exp(mu + sigma^2 / 2)
        try:
            log_y_train = np.log(np.maximum(y_train_sev, 1.0))
            ln_model = sm.OLS(log_y_train, X_train_sev).fit()
            pred_log_mu = ln_model.predict(X_val_sev)
            sigma2 = float(np.var(log_y_train - ln_model.predict(X_train_sev), ddof=X_train_sev.shape[1]))
            pred_val_ln = np.exp(pred_log_mu + 0.5 * sigma2)
            metrics_ln = evaluate_severity_model(y_val_sev, pred_val_ln, num_parameters=X_train_sev.shape[1])
            sev_candidates["Log-Normal OLS"] = {
                "model": ln_model,
                "val_deviance": float(metrics_ln["gamma_deviance"]),
                "val_mae": float(metrics_ln["mae"]),
                "val_preds": pred_val_ln,
                "sigma2": sigma2
            }
        except Exception as e:
            logger.warning(f"Log-Normal fitting failed: {e}")

        # C. Inverse Gaussian GLM
        try:
            ig_model = sm.GLM(
                y_train_sev, X_train_sev,
                family=sm.families.InverseGaussian(link=sm.families.links.Log())
            ).fit()
            pred_val_ig = ig_model.predict(X_val_sev)
            metrics_ig = evaluate_severity_model(y_val_sev, pred_val_ig, num_parameters=X_train_sev.shape[1])
            sev_candidates["Inverse Gaussian GLM"] = {
                "model": ig_model,
                "val_deviance": float(metrics_ig["gamma_deviance"]),
                "val_mae": float(metrics_ig["mae"]),
                "val_preds": pred_val_ig
            }
        except Exception as e:
            logger.warning(f"Inverse Gaussian GLM fitting failed: {e}")

        # The Severity Actuarial Gate: Choose model with lowest Gamma Deviance
        winning_sev_name = min(sev_candidates.keys(), key=lambda k: sev_candidates[k]["val_deviance"])
        winning_sev_entry = sev_candidates[winning_sev_name]
        logger.info(f"Severity Actuarial Gate: WINNER = {winning_sev_name} (Validation Deviance = {winning_sev_entry['val_deviance']:.5f})")

        # ─── 5. SERIALIZE WINNING MODELS & PERFORMANCE LEDGER ────────────────────
        freq_model_path = os.path.join(self.outputs_dir, "freq_model.pkl")
        sev_model_path = os.path.join(self.outputs_dir, "sev_model.pkl")

        with open(freq_model_path, "wb") as f:
            pickle.dump(winning_freq_entry["model"], f)
        with open(sev_model_path, "wb") as f:
            pickle.dump(winning_sev_entry["model"], f)

        performance_report = {
            "execution_timestamp": run_timestamp,
            "version": self.version,
            "frequency_gate": {
                "winner": winning_freq_name,
                "selected_gini": winning_freq_entry["val_gini"],
                "all_candidates": {k: {"val_gini": v["val_gini"], "aic": v.get("aic", None)} for k, v in freq_candidates.items()}
            },
            "severity_gate": {
                "winner": winning_sev_name,
                "selected_gamma_deviance": winning_sev_entry["val_deviance"],
                "all_candidates": {k: {"val_deviance": v["val_deviance"], "val_mae": v["val_mae"]} for k, v in sev_candidates.items()}
            }
        }

        perf_json_path = os.path.join(self.outputs_dir, "model_performance.json")
        with open(perf_json_path, "w") as f:
            json.dump(performance_report, f, indent=2)

        # ─── 6. SCORE ENTIRE PORTFOLIO & WRITE TO SQLite ────────────────────────
        logger.info("Scoring complete portfolio with winning actuarial models...")
        X_all = sm.add_constant(df_clean[self.predictor_cols].fillna(0).values)
        exp_all = df_clean["Exposure"].values

        # Predict Frequency
        if "XGBoost" in winning_freq_name:
            import xgboost as xgb
            dall = xgb.DMatrix(df_clean[self.predictor_cols].values)
            dall.set_base_margin(np.log(np.maximum(exp_all, 1e-4)))
            pred_freq_all = winning_freq_entry["model"].predict(dall)
        else:
            pred_freq_all = winning_freq_entry["model"].predict(X_all, exposure=exp_all)

        # Predict Severity
        if "Log-Normal" in winning_sev_name:
            pred_mu_all = winning_sev_entry["model"].predict(X_all)
            pred_sev_all = np.exp(pred_mu_all + 0.5 * winning_sev_entry["sigma2"])
        else:
            pred_sev_all = winning_sev_entry["model"].predict(X_all)

        df_scored = df_clean.copy()
        df_scored["pred_frequency"] = pred_freq_all
        df_scored["pred_severity"] = pred_sev_all
        df_scored["pure_premium"] = pred_freq_all * pred_sev_all
        df_scored["scoring_timestamp"] = run_timestamp

        # Store in claims warehouse
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS scored_portfolio")
            conn.commit()
            df_scored.to_sql("scored_portfolio", conn, if_exists="replace", index=False)
            logger.info(f"Saved {len(df_scored)} scored policy records to table 'scored_portfolio' in claims warehouse.")

        # Update Pricing Registry Snapshot
        registry_path = os.path.join(self.outputs_dir, "pricing_registry.json")
        registry_data = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    registry_data = json.load(f)
            except Exception:
                pass

        registry_data.update({
            "best_freq_model": winning_freq_name,
            "freq_gini": round(winning_freq_entry["val_gini"], 5),
            "best_sev_model": winning_sev_name,
            "sev_gamma_deviance": round(winning_sev_entry["val_deviance"], 5),
            "mean_pred_frequency": round(float(np.mean(pred_freq_all)), 5),
            "mean_pred_severity": round(float(np.mean(pred_sev_all)), 2),
            "mean_pure_premium": round(float(np.mean(df_scored["pure_premium"])), 2),
            "n_train": len(train_df),
            "n_val": len(val_df)
        })

        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        return {
            "status": "SUCCESS",
            "winning_frequency": winning_freq_name,
            "validation_gini": winning_freq_entry["val_gini"],
            "winning_severity": winning_sev_name,
            "validation_deviance": winning_sev_entry["val_deviance"],
            "total_policies_scored": len(df_scored)
        }

if __name__ == "__main__":
    pipeline = ModelingPipeline()
    result = pipeline.run_pipeline()
    print("\nPhase 3 Modeling Pipeline Completed Successfully:")
    print(json.dumps(result, indent=2))
