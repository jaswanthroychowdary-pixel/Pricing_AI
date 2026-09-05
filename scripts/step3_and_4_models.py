"""
step3_and_4_models.py

Executes Frequency Modeling (Step 3) and Severity Modeling (Step 4) pipelines:
1. Frequency: Poisson GLM, Negative Binomial GLM, and XGBoost
2. Evaluates out-of-sample Actuarial Gini, AIC, and Deviance
3. Severity: Gamma GLM, Log-Normal (with Smearing), and Inverse Gaussian GLM
4. Evaluates Gamma Deviance, MAE, RMSE, and A/E ratios
5. Serializes selected models and saves datasets for Pure Premium synthesis
"""

import os
import sys
import json
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pricing_functions_v2 import fit_and_evaluate_frequency_models, fit_and_evaluate_severity_models


def run_step3_and_4(
    clean_path: str = "outputs/df_clean.parquet",
    meta_path: str = "outputs/meta.json",
    output_dir: str = "outputs",
    best_freq_choice: str = "Poisson GLM",
    best_sev_choice: str = "Gamma GLM"
):
    os.makedirs(output_dir, exist_ok=True)
    df_clean = pd.read_parquet(clean_path)
    meta = json.load(open(meta_path))
    
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    
    # Encode features
    df_enc = df_clean[num_cols + cat_cols].copy()
    for c in num_cols:
        df_enc[c] = df_enc[c].fillna(df_enc[c].median())
    df_enc = pd.get_dummies(df_enc, columns=cat_cols, drop_first=True, dtype=float)
    feature_cols = list(df_enc.columns)
    
    df_model_input = pd.concat([df_clean[["ClaimNb", "ClaimAmount", "Exposure"]], df_enc], axis=1)
    
    # ── Step 3: Frequency Models ───────────────────────────────────────────────
    print("=== Step 3: Frequency Modeling Pipeline ===")
    freq_eval = fit_and_evaluate_frequency_models(
        df=df_model_input,
        predictor_cols=feature_cols,
        claim_nb_col="ClaimNb",
        exposure_col="Exposure"
    )
    print("Frequency Models Comparison:")
    print(freq_eval["summary"])
    
    # Fit selected frequency model on full dataset
    X_full = df_enc.to_numpy(dtype=float)
    X_full_sm = sm.add_constant(X_full)
    y_freq = df_clean["ClaimNb"].values.astype(float)
    exposure_full = np.clip(df_clean["Exposure"].values.astype(float), 1e-12, None)
    off_full = np.log(exposure_full)
    
    if best_freq_choice == "Poisson GLM":
        freq_model = sm.GLM(
            y_freq, X_full_sm,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=off_full
        ).fit(maxiter=100)
        freq_preds = freq_model.predict(X_full_sm, offset=off_full)
    elif best_freq_choice == "NegBinomial":
        freq_model = sm.GLM(
            y_freq, X_full_sm,
            family=sm.families.NegativeBinomial(alpha=1.0),
            offset=off_full
        ).fit(maxiter=100)
        freq_preds = freq_model.predict(X_full_sm, offset=off_full)
    else:
        import xgboost as xgb
        freq_model = xgb.XGBRegressor(
            objective="count:poisson", n_estimators=100,
            learning_rate=0.05, max_depth=4, random_state=42
        )
        freq_model.fit(X_full, y_freq, base_margin=off_full)
        freq_preds = freq_model.predict(X_full, base_margin=off_full)
        
    df_clean["Freq_Pred"] = freq_preds
    df_clean.to_parquet(os.path.join(output_dir, "df_with_freq.parquet"), index=False)
    
    with open(os.path.join(output_dir, "freq_model.pkl"), "wb") as f:
        pickle.dump({"model": freq_model, "model_name": best_freq_choice, "features": feature_cols}, f)
        
    freq_summary_dict = freq_eval["summary"].to_dict()
    freq_summary_dict["best_model"] = best_freq_choice
    with open(os.path.join(output_dir, "frequency_results.json"), "w") as f:
        json.dump(freq_summary_dict, f, indent=2)
        
    print(f"✓ Selected Frequency Model: {best_freq_choice}")
    print(f"✓ Saved {output_dir}/freq_model.pkl and {output_dir}/df_with_freq.parquet")
    
    # ── Step 4: Severity Models ────────────────────────────────────────────────
    print("\n=== Step 4: Severity Modeling Pipeline ===")
    sev_eval = fit_and_evaluate_severity_models(
        df=df_model_input,
        predictor_cols=feature_cols,
        claim_amount_col="ClaimAmount"
    )
    print("Severity Models Comparison:")
    print(sev_eval["summary"])
    
    # Fit selected severity model on positive claims
    pos_mask = df_clean["ClaimAmount"] > 0
    df_pos = df_clean[pos_mask]
    X_pos_sm = sm.add_constant(df_enc[pos_mask].to_numpy(dtype=float))
    y_pos = df_pos["ClaimAmount"].values.astype(float)
    
    if best_sev_choice == "Gamma GLM":
        sev_model = sm.GLM(
            y_pos, X_pos_sm,
            family=sm.families.Gamma(link=sm.families.links.Log())
        ).fit(maxiter=100)
        sev_preds = sev_model.predict(X_full_sm)
        sigma2 = None
    elif best_sev_choice == "Log-Normal":
        log_y_pos = np.log(y_pos)
        sev_model = sm.OLS(log_y_pos, X_pos_sm).fit()
        sigma2 = float(sev_model.mse_resid)
        sev_preds = np.exp(sev_model.predict(X_full_sm) + 0.5 * sigma2)
    else:
        sev_model = sm.GLM(
            y_pos, X_pos_sm,
            family=sm.families.InverseGaussian(link=sm.families.links.Log())
        ).fit(maxiter=100)
        sev_preds = sev_model.predict(X_full_sm)
        sigma2 = None
        
    df_clean["Sev_Pred"] = sev_preds
    df_clean.to_parquet(os.path.join(output_dir, "df_with_severity.parquet"), index=False)
    
    with open(os.path.join(output_dir, "sev_model.pkl"), "wb") as f:
        pickle.dump({"model": sev_model, "model_name": best_sev_choice, "features": feature_cols, "sigma2": sigma2}, f)
        
    sev_summary_dict = sev_eval["summary"].to_dict()
    sev_summary_dict["best_model"] = best_sev_choice
    sev_summary_dict["mean_pred_severity"] = float(df_clean["Sev_Pred"].mean())
    sev_summary_dict["n_train"] = int(pos_mask.sum())
    with open(os.path.join(output_dir, "severity_results.json"), "w") as f:
        json.dump(sev_summary_dict, f, indent=2)
        
    print(f"✓ Selected Severity Model: {best_sev_choice}")
    print(f"✓ Saved {output_dir}/sev_model.pkl and {output_dir}/df_with_severity.parquet")
    print("=== Steps 3 & 4 Complete ===")


if __name__ == "__main__":
    run_step3_and_4()
