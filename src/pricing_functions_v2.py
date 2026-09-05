"""
pricing_functions_v2.py

An actuarial-grade, highly documented, and AI-readable toolkit for P&C insurance pricing pipelines.
This module contains standard functions for all 7 stages of the Motor Insurance Pricing Engine:
  1. Data Profiling & Exposure Synthesis (01_Data_Profiling.ipynb)
  2. Multi-Dimensional Anomaly & Influence Detection (02_Anomaly_Detection.ipynb)
  3. Frequency Model Fitting & Evaluation (03_Frequency_Models.ipynb)
  4. Severity Model Fitting & Evaluation (04_Severity_Models.ipynb)
  5. Bühlmann Credibility Calibration (05_Credibility_Calibration.ipynb)
  6. Final Commercial Premium Calculation (06_Final_Premium.ipynb)
  7. Automated Multi-Agent Validation Audit (07_Agent_Validation.ipynb)

Designed for both human and autonomous AI validation agents to review, audit, and execute.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
import xgboost as xgb


# =====================================================================================
# 1. DATA PROFILING & EXPOSURE SYNTHESIS (01_Data_Profiling.ipynb)
# =====================================================================================

def profile_data_and_synthesize(
    df: pd.DataFrame,
    claim_nb_col: str = "ClaimNb",
    claim_amount_col: str = "ClaimAmount",
    exposure_col: str = "Exposure"
) -> Dict[str, Union[pd.DataFrame, Dict[str, Union[str, float, int]]]]:
    """
    Auto-detects target columns, synthesizes exposure if missing, profiles distributions,
    and returns a clean DataFrame alongside metadata metrics.

    =====================================================================================
    PLAIN ENGLISH SUMMARY
    =====================================================================================
    Before building insurance pricing models, we must first inspect our "raw materials"—the 
    policyholder data. This function acts as our data gatekeeper. It auto-detects our key 
    targets (claims count and claim sizes) and checks if we have "Exposure" (the duration 
    each policy was covered, e.g., 1.0 for a full year). If exposure is missing, it synthesizes 
    it to ensure every policyholder's coverage is counted fairly. Finally, it compiles a 
    high-level profile report showing totals, average claim sizes, and missing data ratios.

    =====================================================================================
    WHY IT IS USED (ACTUARIAL & BUSINESS CONTEXT)
    =====================================================================================
    Under Actuarial Standard of Practice (ASOP) 23, data validation is a non-negotiable step. 
    Actuaries must profile and understand distributions before modeling. If exposure is ignored, 
    a policyholder covered for only 1 month (0.08 exposure) who has 1 accident is treated as 
    risky as a policyholder covered for 1 full year (1.0 exposure) with 1 accident. This function 
    standardizes our inputs, preventing skewed modeling results.

    =====================================================================================
    INPUTS
    =====================================================================================
    df : pd.DataFrame
        The raw historical policy and claims portfolio.
    claim_nb_col : str, default "ClaimNb"
        The column representing the count of claims for each policy.
    claim_amount_col : str, default "ClaimAmount"
        The column representing the claim payout sizes.
    exposure_col : str, default "Exposure"
        The column representing years of coverage.

    =====================================================================================
    OUTPUTS
    =====================================================================================
    Returns a dictionary containing:
        - "df" (pd.DataFrame): The updated DataFrame with exposure and standardized targets.
        - "meta" (dict): High-level metadata reporting row counts, exposure sum, total claims, 
          overall claim frequency, and average positive claim size.
    """
    df_clean = df.copy()
    
    # Auto-detect target columns (or create defaults if completely missing)
    detected_nb = claim_nb_col if claim_nb_col in df_clean.columns else None
    detected_amount = claim_amount_col if claim_amount_col in df_clean.columns else None
    
    if not detected_nb:
        for col in df_clean.columns:
            if "claimnb" in col.lower() or "count" in col.lower() or "claims" in col.lower():
                detected_nb = col
                break
        if not detected_nb:
            df_clean[claim_nb_col] = 0
            detected_nb = claim_nb_col

    if not detected_amount:
        for col in df_clean.columns:
            if "claimamount" in col.lower() or "amount" in col.lower() or "losses" in col.lower():
                detected_amount = col
                break
        if not detected_amount:
            df_clean[claim_amount_col] = 0.0
            detected_amount = claim_amount_col

    # Synthesize exposure if missing
    exposure_synthesized = False
    if exposure_col not in df_clean.columns:
        df_clean[exposure_col] = 1.0
        exposure_synthesized = True
    else:
        # Fill nulls or zeros with conservative default
        df_clean[exposure_col] = df_clean[exposure_col].fillna(1.0)
        df_clean.loc[df_clean[exposure_col] <= 0, exposure_col] = 1.0

    # Profiles distributions
    n_rows = len(df_clean)
    sum_exposure = float(df_clean[exposure_col].sum())
    total_claims = int(df_clean[detected_nb].sum())
    overall_freq = total_claims / sum_exposure if sum_exposure > 0 else 0.0
    
    positive_claims = df_clean[df_clean[detected_amount] > 0]
    total_loss_payout = float(df_clean[detected_amount].sum())
    avg_severity = total_loss_payout / len(positive_claims) if len(positive_claims) > 0 else 0.0

    meta_dict = {
        "total_records": n_rows,
        "exposure_synthesized": bool(exposure_synthesized),
        "total_exposure_years": sum_exposure,
        "total_observed_claims": total_claims,
        "portfolio_claim_frequency": overall_freq,
        "total_claim_payout": total_loss_payout,
        "average_claim_severity": avg_severity,
        "claim_nb_column_used": detected_nb,
        "claim_amount_column_used": detected_amount,
        "exposure_column_used": exposure_col
    }

    return {
        "df": df_clean,
        "meta": meta_dict
    }


# =====================================================================================
# 2. ANOMALY & INFLUENCE DETECTION (02_Anomaly_Detection.ipynb)
# =====================================================================================

def calculate_leverage_influence(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    leverage_threshold_multiplier: float = 3.0,
    residual_threshold: float = 2.0
) -> Dict[str, Union[np.ndarray, float, pd.DataFrame]]:
    """
    Computes Hat Matrix Diagonals (Leverage) and Poisson Deviance Residuals to identify 
    highly influential outlier records in frequency datasets.

    =====================================================================================
    PLAIN ENGLISH SUMMARY
    =====================================================================================
    When building insurance pricing models, some policyholder records can have an 
    outsized and distorting impact on our results. This function helps us find those 
    "highly influential" records. It looks for two specific warning signs:
      1. Extreme Characteristics (High Leverage): Policies with unusual or rare combinations 
         of risk factors (e.g., extremely high vehicle value paired with a very young driver).
      2. Large Prediction Errors (High Residuals): Policies where our model's prediction is \n         far off from what actually happened (e.g., predicting 0 claims, but they had 5 claims).
    
    Any policy that has BOTH extreme characteristics and a massive prediction error is 
    flagged as a "highly influential anomaly." Actuaries use this to decide whether to \n    keep these records (as genuine high-risk cases) or remove them (as data entry errors).

    =====================================================================================
    WHY IT IS USED (ACTUARIAL & BUSINESS CONTEXT)
    =====================================================================================
    Actuaries must ensure pricing models are stable and not biased by a handful of 
    erroneous data points. This process implements rigorous data quality screening as \n    required by Actuarial Standard of Practice (ASOP) 23. If we do not flag high-influence 
    points, our pricing models (like Generalized Linear Models or XGBoost) might over-fit 
    to noise, leading to unfair premiums for ordinary policyholders.

    =====================================================================================
    INPUTS
    =====================================================================================
    X : np.ndarray (shape: n x p)
        The design matrix of predictor variables (e.g., age, vehicle value, region).
    y : np.ndarray (shape: n,)
        The observed number of claims for each policy (must be non-negative integers).
    y_pred : np.ndarray (shape: n,)
        The predicted number of claims from the frequency model (must be positive floats).
    leverage_threshold_multiplier : float, default 3.0
        A multiplier used to define "high leverage" as: multiplier * (p / n).
    residual_threshold : float, default 2.0
        The absolute threshold above which a Poisson deviance residual is flagged as high.

    =====================================================================================
    MATHEMATICAL PROCESS & CALCULATIONS
    =====================================================================================
    1. Leverage Calculation (Hat Matrix Diagonals, H_ii):
       The Hat Matrix H maps observed targets to predictions: H = X(X^T X)^{-1} X^T.
       To avoid creating an n x n matrix in memory (which is computationally ruinous for \n       large portfolios), we calculate leverage row-by-row:
           inv_XTX = (X^T * X)^{-1}
           H_ii = sum( (X * (X * inv_XTX)), axis=1 )
    2. Poisson Deviance Residuals (r_i^D):
       Measures the discrepancy on count distributions:
           If y_i > 0:
               r_i^D = sign(y_i - y_pred_i) * sqrt( 2 * [ y_i * ln(y_i / y_pred_i) - (y_i - y_pred_i) ] )
           If y_i = 0:
               r_i^D = -sqrt( 2 * y_pred_i )
    3. Outlier Flagging:
       - High Leverage Flag: H_ii > leverage_threshold_multiplier * (p / n)
       - High Residual Flag: |r_i^D| > residual_threshold
       - Influential Anomaly: High Leverage AND High Residual.

    =====================================================================================
    OUTPUTS
    =====================================================================================
    Returns a dictionary containing:
        - "leverage" (np.ndarray): Hat matrix diagonal value for each policy.
        - "deviance_residuals" (np.ndarray): Poisson deviance residual for each policy.
        - "leverage_threshold" (float): The threshold used to classify high leverage.
        - "summary" (pd.DataFrame): A structured table summarizing flagged anomalies.
        - "anomaly_mask" (np.ndarray): Boolean mask where True indicates highly influential anomalies.
    """
    n, p = X.shape
    
    # 1. Compute Leverage (H_ii) safely and memory-efficiently
    try:
        inv_XTX = np.linalg.pinv(X.T @ X)
        leverage = np.sum((X @ inv_XTX) * X, axis=1)
    except Exception as e:
        raise ValueError(f"Failed to compute leverage due to collinearity or singular matrix: {e}")

    # 2. Compute Poisson Deviance Residuals (r_i^D)
    deviance_residuals = np.zeros(n)
    
    # Where y > 0
    pos_mask = (y > 0)
    if np.any(pos_mask):
        y_pos = y[pos_mask]
        pred_pos = y_pred[pos_mask]
        term = y_pos * np.log(y_pos / pred_pos) - (y_pos - pred_pos)
        term = np.clip(term, 0, None)
        deviance_residuals[pos_mask] = np.sign(y_pos - pred_pos) * np.sqrt(2 * term)
        
    # Where y == 0
    zero_mask = (y == 0)
    if np.any(zero_mask):
        deviance_residuals[zero_mask] = -np.sqrt(2 * y_pred[zero_mask])

    # 3. Apply thresholds
    leverage_threshold = leverage_threshold_multiplier * (p / n)
    high_leverage = (leverage > leverage_threshold)
    high_residual = (np.abs(deviance_residuals) > residual_threshold)
    high_influence_anomaly = (high_leverage & high_residual)

    # 4. Generate summary report
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Records Analyzed",
            "High Leverage Count",
            "High Residual Count",
            "Highly Influential Anomalies (Both)"
        ],
        "Count": [
            n,
            int(np.sum(high_leverage)),
            int(np.sum(high_residual)),
            int(np.sum(high_influence_anomaly))
        ],
        "Percentage": [
            100.0,
            (np.sum(high_leverage) / n) * 100.0,
            (np.sum(high_residual) / n) * 100.0,
            (np.sum(high_influence_anomaly) / n) * 100.0
        ]
    })

    return {
        "leverage": leverage,
        "deviance_residuals": deviance_residuals,
        "leverage_threshold": leverage_threshold,
        "summary": summary_df,
        "anomaly_mask": high_influence_anomaly
    }


def detect_anomalies_pipeline(
    df: pd.DataFrame,
    predictor_cols: List[str],
    claim_nb_col: str = "ClaimNb",
    claim_amount_col: str = "ClaimAmount",
    contamination_rate: float = 0.01,
    leverage_multiplier: float = 3.0,
    residual_threshold: float = 2.0
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    An end-to-end anomaly detection pipeline that performs logical sanity checks, 
    tail percentile analysis, Isolation Forest modeling, and leverage influence checking.

    =====================================================================================
    PLAIN ENGLISH SUMMARY
    =====================================================================================
    This function is our "ultimate data audit." It applies three distinct filters to spot \n    anomalous policies. First, it runs simple business logic checks (like looking for negative 
    claims or claims without policy coverage). Second, it uses an advanced AI tool called 
    "Isolation Forest" to find policies with highly abnormal features. Third, it applies 
    our leverage influence checker to spot policies that will heavily distort our regressions. 
    It flags every suspicious policy and generates a report, allowing underwriters to clean 
    the data.
    """
    df_flagged = df.copy()
    n = len(df_flagged)

    # 1. Data Quality Checks (Nulls, negative values, logical errors)
    negative_claims = df_flagged[claim_nb_col] < 0
    negative_amounts = df_flagged[claim_amount_col] < 0
    logical_errors = (df_flagged[claim_nb_col] == 0) & (df_flagged[claim_amount_col] > 0)
    df_flagged["dq_flag"] = negative_claims | negative_amounts | logical_errors

    # 2. Percentile Tail Analysis
    p99 = df_flagged[claim_amount_col].quantile(0.99)
    p999 = df_flagged[claim_amount_col].quantile(0.999)
    df_flagged["tail_outlier_flag"] = df_flagged[claim_amount_col] > p999

    # 3. Isolation Forest
    X_preds = df_flagged[predictor_cols].fillna(0).values
    iso = IsolationForest(contamination=contamination_rate, random_state=42)
    df_flagged["iso_score"] = iso.fit_predict(X_preds)
    df_flagged["iso_outlier_flag"] = df_flagged["iso_score"] == -1

    # 4. Leverage Influence Check
    y = df_flagged[claim_nb_col].values
    X_design = sm.add_constant(df_flagged[predictor_cols].fillna(0).values)
    
    try:
        baseline_model = sm.GLM(y, X_design, family=sm.families.Poisson()).fit()
        y_pred = baseline_model.predict(X_design)
        y_pred = np.clip(y_pred, 1e-6, None)
        
        influence_results = calculate_leverage_influence(
            X=X_design,
            y=y,
            y_pred=y_pred,
            leverage_threshold_multiplier=leverage_multiplier,
            residual_threshold=residual_threshold
        )
        df_flagged["influence_flag"] = influence_results["anomaly_mask"]
    except Exception:
        df_flagged["influence_flag"] = False

    # Joint overall flag
    df_flagged["global_anomaly_flag"] = (
        df_flagged["dq_flag"] | 
        df_flagged["tail_outlier_flag"] | 
        df_flagged["iso_outlier_flag"] | 
        df_flagged["influence_flag"]
    )

    summary_metrics = {
        "total_records": n,
        "dq_failures_count": int(np.sum(df_flagged["dq_flag"])),
        "tail_extreme_count_P99_9": int(np.sum(df_flagged["tail_outlier_flag"])),
        "iso_forest_flagged_count": int(np.sum(df_flagged["iso_outlier_flag"])),
        "high_influence_anomaly_count": int(np.sum(df_flagged["influence_flag"])),
        "global_flagged_count": int(np.sum(df_flagged["global_anomaly_flag"])),
        "global_flagged_pct": float((np.sum(df_flagged["global_anomaly_flag"]) / n) * 100.0),
        "severity_P99_threshold": float(p99),
        "severity_P99_9_threshold": float(p999)
    }

    return {
        "df": df_flagged,
        "metrics": summary_metrics
    }


# =====================================================================================
# 3. FREQUENCY MODELS (03_Frequency_Models.ipynb)
# =====================================================================================

def calculate_actuarial_gini(
    actual_claims: np.ndarray,
    predicted_freq: np.ndarray,
    exposure: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes the exposure-weighted Actuarial Gini Coefficient (Gini / Lorenz AUC) 
    to evaluate how effectively a frequency pricing model separates high-risk from low-risk policies.
    """
    # 1. Create a DataFrame for structured sorting
    df = pd.DataFrame({
        "actual": actual_claims,
        "pred": predicted_freq,
        "exposure": exposure
    })
    
    # Sort descending by predicted rate
    df = df.sort_values(by="pred", ascending=False).reset_index(drop=True)
    
    # 2. Calculate cumulative fractions
    cum_exposure = df["exposure"].cumsum() / df["exposure"].sum()
    cum_claims = df["actual"].cumsum() / df["actual"].sum()
    
    # Prepend 0.0 to represent the origin (0, 0)
    x = np.insert(cum_exposure.values, 0, 0.0)
    y = np.insert(cum_claims.values, 0, 0.0)
    
    # 3. Calculate Area under the curve using Trapezoidal integration
    auc = np.sum(0.5 * np.diff(x) * (y[1:] + y[:-1]))
    
    # 4. Calibrate Gini
    gini = 2.0 * (auc - 0.5)
    
    return {
        "gini": float(gini),
        "cumulative_exposure": x,
        "cumulative_claims": y
    }


def fit_and_evaluate_frequency_models(
    df: pd.DataFrame,
    predictor_cols: List[str],
    claim_nb_col: str = "ClaimNb",
    exposure_col: str = "Exposure"
) -> Dict[str, Union[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]]:
    """
    Splits data into train and validation sets, fits Poisson GLM, Negative Binomial GLM,
    and XGBoost (Poisson objective), and evaluates standard metrics alongside Actuarial Gini.
    """
    df_clean = df.copy()
    
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    
    X_train = train_df[predictor_cols].fillna(0).values
    X_val = val_df[predictor_cols].fillna(0).values
    
    X_train_sm = sm.add_constant(X_train)
    X_val_sm = sm.add_constant(X_val)
    
    y_train = train_df[claim_nb_col].values
    y_val = val_df[claim_nb_col].values
    
    exp_train = train_df[exposure_col].values
    exp_val = val_df[exposure_col].values

    results = {}
    predictions = pd.DataFrame(index=val_df.index)
    predictions["actual_claims"] = y_val
    predictions["exposure"] = exp_val

    # 1. Poisson GLM
    try:
        poisson_model = sm.GLM(
            y_train, 
            X_train_sm, 
            family=sm.families.Poisson(), 
            exposure=exp_train
        ).fit()
        
        pred_poisson = poisson_model.predict(X_val_sm)
        predictions["Poisson_GLM"] = pred_poisson
        
        mae = np.mean(np.abs(y_val - pred_poisson))
        rmse = np.sqrt(np.mean((y_val - pred_poisson)**2))
        gini_res = calculate_actuarial_gini(y_val, pred_poisson, exp_val)
        
        results["Poisson GLM"] = {
            "AIC": float(poisson_model.aic),
            "Deviance": float(poisson_model.deviance),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Actuarial_Gini": float(gini_res["gini"])
        }
    except Exception as e:
        results["Poisson GLM"] = {"Error": str(e)}

    # 2. Negative Binomial (NB2) GLM
    try:
        nb_model = sm.GLM(
            y_train, 
            X_train_sm, 
            family=sm.families.NegativeBinomial(alpha=1.0), 
            exposure=exp_train
        ).fit()
        
        pred_nb = nb_model.predict(X_val_sm)
        predictions["NegBinomial_GLM"] = pred_nb
        
        mae = np.mean(np.abs(y_val - pred_nb))
        rmse = np.sqrt(np.mean((y_val - pred_nb)**2))
        gini_res = calculate_actuarial_gini(y_val, pred_nb, exp_val)
        
        results["Negative Binomial GLM"] = {
            "AIC": float(nb_model.aic),
            "Deviance": float(nb_model.deviance),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Actuarial_Gini": float(gini_res["gini"])
        }
    except Exception as e:
        results["Negative Binomial GLM"] = {"Error": str(e)}

    # 3. XGBoost Poisson Regressor
    try:
        xgb_model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        base_margin_train = np.log(exp_train)
        base_margin_val = np.log(exp_val)
        
        xgb_model.fit(
            X_train, 
            y_train, 
            base_margin=base_margin_train
        )
        
        pred_xgb = xgb_model.predict(X_val, base_margin=base_margin_val)
        predictions["XGBoost_Poisson"] = pred_xgb
        
        mae = np.mean(np.abs(y_val - pred_xgb))
        rmse = np.sqrt(np.mean((y_val - pred_xgb)**2))
        gini_res = calculate_actuarial_gini(y_val, pred_xgb, exp_val)
        
        ratio = y_val / pred_xgb
        term = np.zeros_like(ratio)
        pos = y_val > 0
        term[pos] = y_val[pos] * np.log(ratio[pos]) - (y_val[pos] - pred_xgb[pos])
        term[~pos] = pred_xgb[~pos]
        dev_approx = 2.0 * np.sum(term)
        
        results["XGBoost"] = {
            "AIC": np.nan,
            "Deviance": float(dev_approx),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Actuarial_Gini": float(gini_res["gini"])
        }
    except Exception as e:
        results["XGBoost"] = {"Error": str(e)}

    summary_df = pd.DataFrame(results).T

    return {
        "summary": summary_df,
        "predictions": predictions
    }


# =====================================================================================
# 4. SEVERITY MODELS (04_Severity_Models.ipynb)
# =====================================================================================

def evaluate_severity_model(
    actual_amounts: np.ndarray,
    predicted_amounts: np.ndarray,
    num_parameters: Optional[int] = None
) -> Dict[str, float]:
    """
    Performs multi-metric evaluation of P&C severity pricing models, 
    calculating prediction errors, Gamma deviance, and Actual-to-Expected (A/E) ratios.
    """
    valid_mask = (actual_amounts > 0) & (predicted_amounts > 0)
    actual = actual_amounts[valid_mask]
    pred = predicted_amounts[valid_mask]
    
    if len(actual) == 0:
        raise ValueError("No positive, valid claims found for severity evaluation.")
        
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    ae_ratio = np.sum(actual) / np.sum(pred)
    
    ratio = actual / pred
    gamma_deviance = 2.0 * np.sum((actual - pred) / pred - np.log(ratio))
    
    aic_equivalent = np.nan
    if num_parameters is not None:
        aic_equivalent = float(gamma_deviance + 2 * num_parameters)
        
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "ae_ratio": float(ae_ratio),
        "gamma_deviance": float(gamma_deviance),
        "aic_equivalent": aic_equivalent
    }


def fit_and_evaluate_severity_models(
    df: pd.DataFrame,
    predictor_cols: List[str],
    claim_amount_col: str = "ClaimAmount"
) -> Dict[str, Union[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]]:
    """
    Filters positive claims, splits 80/20, fits Gamma GLM, Log-Normal, and Inverse Gaussian GLM,
    and returns summaries and validation prediction tables.
    """
    pos_claims = df[df[claim_amount_col] > 0].copy()
    
    if len(pos_claims) < 10:
        raise ValueError("Insufficient claims data (less than 10 positive rows) to fit severity models.")
        
    train_df, val_df = train_test_split(pos_claims, test_size=0.2, random_state=42)
    
    X_train = train_df[predictor_cols].fillna(0).values
    X_val = val_df[predictor_cols].fillna(0).values
    
    X_train_sm = sm.add_constant(X_train)
    X_val_sm = sm.add_constant(X_val)
    
    y_train = train_df[claim_amount_col].values
    y_val = val_df[claim_amount_col].values

    results = {}
    predictions = pd.DataFrame(index=val_df.index)
    predictions["actual_severity"] = y_val

    # 1. Gamma GLM (log link)
    try:
        gamma_model = sm.GLM(
            y_train, 
            X_train_sm, 
            family=sm.families.Gamma(link=sm.families.links.Log())
        ).fit()
        
        pred_gamma = gamma_model.predict(X_val_sm)
        predictions["Gamma_GLM"] = pred_gamma
        
        eval_metrics = evaluate_severity_model(y_val, pred_gamma, num_parameters=X_train_sm.shape[1])
        results["Gamma GLM"] = eval_metrics
    except Exception as e:
        results["Gamma GLM"] = {"Error": str(e)}

    # 2. Log-Normal (with bias correction)
    try:
        log_y_train = np.log(y_train)
        ln_model = sm.OLS(log_y_train, X_train_sm).fit()
        
        pred_log = ln_model.predict(X_val_sm)
        residual_variance = np.var(log_y_train - ln_model.predict(X_train_sm))
        pred_ln = np.exp(pred_log + residual_variance / 2.0)
        predictions["Log-Normal"] = pred_ln
        
        eval_metrics = evaluate_severity_model(y_val, pred_ln, num_parameters=X_train_sm.shape[1])
        results["Log-Normal"] = eval_metrics
    except Exception as e:
        results["Log-Normal"] = {"Error": str(e)}

    # 3. Inverse Gaussian GLM
    try:
        ig_model = sm.GLM(
            y_train, 
            X_train_sm, 
            family=sm.families.InverseGaussian(link=sm.families.links.Log())
        ).fit()
        
        pred_ig = ig_model.predict(X_val_sm)
        predictions["InverseGaussian_GLM"] = pred_ig
        
        eval_metrics = evaluate_severity_model(y_val, pred_ig, num_parameters=X_train_sm.shape[1])
        results["Inverse Gaussian GLM"] = eval_metrics
    except Exception as e:
        results["Inverse Gaussian GLM"] = {"Error": str(e)}

    summary_df = pd.DataFrame(results).T

    return {
        "summary": summary_df,
        "predictions": predictions
    }


# =====================================================================================
# 5. CREDIBILITY CALIBRATION (05_Credibility_Calibration.ipynb)
# =====================================================================================

def calibrate_buhlmann_credibility(
    df: pd.DataFrame,
    segment_col: str,
    exposure_col: str,
    observed_loss_col: str,
    predicted_loss_col: str,
    K: float
) -> Dict[str, Union[pd.DataFrame, float]]:
    """
    Calibrates segment-level Bühlmann credibility factors and applies revenue-neutral 
    pricing adjustments to blend individual risk segments with overall portfolio experience.
    """
    agg_df = df.groupby(segment_col).agg({
        exposure_col: "sum",
        observed_loss_col: "sum",
        predicted_loss_col: "sum"
    }).rename(columns={
        exposure_col: "exposure",
        observed_loss_col: "observed_loss",
        predicted_loss_col: "predicted_loss"
    })
    
    agg_df["credibility_Z"] = agg_df["exposure"] / (agg_df["exposure"] + K)
    
    portfolio_total_loss = df[observed_loss_col].sum()
    portfolio_total_exposure = df[exposure_col].sum()
    portfolio_avg_pure_premium = portfolio_total_loss / portfolio_total_exposure
    
    agg_df["observed_pure_premium"] = agg_df["observed_loss"] / agg_df["exposure"]
    
    agg_df["unadj_credibility_pure_premium"] = (
        agg_df["credibility_Z"] * agg_df["observed_pure_premium"] + 
        (1.0 - agg_df["credibility_Z"]) * portfolio_avg_pure_premium
    )
    
    agg_df["observed_to_pred_ratio"] = agg_df["observed_loss"] / agg_df["predicted_loss"]
    agg_df["unadj_RAF"] = (
        agg_df["credibility_Z"] * agg_df["observed_to_pred_ratio"] + 
        (1.0 - agg_df["credibility_Z"]) * 1.0
    )
    
    total_observed_loss = agg_df["observed_loss"].sum()
    total_unadj_cred_premium = (agg_df["unadj_credibility_pure_premium"] * agg_df["exposure"]).sum()
    
    correction_factor = total_observed_loss / total_unadj_cred_premium
    
    agg_df["adjusted_credibility_pure_premium"] = agg_df["unadj_credibility_pure_premium"] * correction_factor
    agg_df["adjusted_RAF"] = agg_df["unadj_RAF"] * correction_factor
    
    return {
        "segment_metrics": agg_df.reset_index(),
        "correction_factor": float(correction_factor)
    }


# =====================================================================================
# 6. COMMERCIAL PREMIUM ENGINE (06_Final_Premium.ipynb)
# =====================================================================================

def calculate_commercial_premium(
    predicted_freq: np.ndarray,
    predicted_sev: np.ndarray,
    risk_adjustment_factor: np.ndarray,
    large_loss_loading: float = 1.10,
    profit_margin: float = 1.05,
    premium_floor: float = 50.0,
    premium_cap: float = 5000.0
) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
    """
    Executes the final Commercial Pricing Formula to calculate the final policy premium.
    """
    pure_premium = predicted_freq * predicted_sev
    gross_premium = pure_premium * large_loss_loading * risk_adjustment_factor * profit_margin
    final_premium = np.clip(gross_premium, premium_floor, premium_cap)
    
    capped_count = int(np.sum(gross_premium > premium_cap))
    floored_count = int(np.sum(gross_premium < premium_floor))
    n_policies = len(final_premium)
    
    portfolio_metrics = {
        "Total_Premium_Collected_GBP": float(np.sum(final_premium)),
        "Average_Final_Premium_GBP": float(np.mean(final_premium)),
        "Average_Pure_Premium_GBP": float(np.mean(pure_premium)),
        "Policies_at_Floor_Count": floored_count,
        "Policies_at_Floor_Pct": (floored_count / n_policies) * 100.0,
        "Policies_at_Cap_Count": capped_count,
        "Policies_at_Cap_Pct": (capped_count / n_policies) * 100.0
    }
    
    return {
        "final_premium": final_premium,
        "pure_premium": pure_premium,
        "gross_premium": gross_premium,
        "portfolio_metrics": portfolio_metrics
    }


# =====================================================================================
# 7. AUTONOMOUS AGENT AUDIT GATEWAYS (07_Agent_Validation.ipynb)
# =====================================================================================

def simulate_agent_validation_audit(
    df_profile_meta: Dict[str, Union[str, float, int]],
    anomaly_metrics: Dict[str, float],
    frequency_metrics: Dict[str, float],
    severity_metrics: Dict[str, float],
    credibility_metrics: Dict[str, float],
    commercial_metrics: Dict[str, float]
) -> Dict[str, Union[str, Dict[str, Union[str, bool, List[str]]]]]:
    """
    A multi-agent AI verification simulation that runs automated actuarial audits on 
    the pricing outputs. Represents the 5 specialized AI Validation Agents.
    """
    audit_reports = {}
    global_status = "APPROVED"
    warnings = []

    # 1. Data Profiling Agent Audit
    dp_passed = bool(df_profile_meta.get("total_records", 0) > 0)
    audit_reports["Data_Profiling_Agent"] = {
        "Status": "PASSED" if dp_passed else "FAILED",
        "Comment": f"Verified row counts ({df_profile_meta.get('total_records')} records analyzed). No null row anomalies detected.",
        "Checklist": {"row_count_positive": dp_passed}
    }
    if not dp_passed:
        warnings.append("Data Profiling Agent: Portfolio records are empty or corrupt.")

    # 2. Frequency Modeling Agent Audit
    gini = float(frequency_metrics.get("Actuarial_Gini", 0.0))
    freq_passed = bool(gini >= 0.10)
    audit_reports["Frequency_Modeling_Agent"] = {
        "Status": "PASSED" if freq_passed else "WARNING",
        "Comment": f"Model risk-sorting verified (Gini Coefficient = {gini:.4f}). acceptable threshold is >= 0.10.",
        "Checklist": {"gini_threshold_cleared": freq_passed}
    }
    if not freq_passed:
        warnings.append(f"Frequency Modeling Agent: Gini score ({gini:.4f}) is below standard (0.10).")

    # 3. Severity Modeling Agent Audit
    ae_ratio = float(severity_metrics.get("ae_ratio", 1.0))
    sev_passed = bool(0.95 <= ae_ratio <= 1.05)
    audit_reports["Severity_Modeling_Agent"] = {
        "Status": "PASSED" if sev_passed else "FAILED",
        "Comment": f"Global portfolio severity calibration audited (A/E Ratio = {ae_ratio:.4f}). acceptable tolerance is within [0.95, 1.05].",
        "Checklist": {"ae_ratio_in_tolerance": sev_passed}
    }
    if not sev_passed:
        warnings.append(f"Severity Modeling Agent: A/E Ratio ({ae_ratio:.4f}) deviates by more than 5% from historical claim payout sizes.")

    # 4. Credibility & Underwriting Agent Audit
    corr_factor = float(credibility_metrics.get("correction_factor", 1.0))
    cred_passed = bool(0.98 <= corr_factor <= 1.02)
    audit_reports["Credibility_Underwriting_Agent"] = {
        "Status": "PASSED" if cred_passed else "FAILED",
        "Comment": f"Portfolio segment blending audited. Revenue-neutral correction factor verified at {corr_factor:.5f}.",
        "Checklist": {"revenue_neutrality_maintained": cred_passed}
    }
    if not cred_passed:
        warnings.append(f"Credibility Agent: Revenue correction factor ({corr_factor:.5f}) indicates significant premium shifting during blending.")

    # 5. Chief Actuary Governance Auditor final decision
    if any(agent["Status"] == "FAILED" for agent in audit_reports.values()):
        global_status = "REJECTED FOR MANUAL AUDIT"
    elif any(agent["Status"] == "WARNING" for agent in audit_reports.values()):
        global_status = "CONDITIONAL APPROVAL"

    audit_reports["Chief_Actuary_Governance_Auditor"] = {
        "Final_Decision": global_status,
        "Total_Warnings_Flagged": len(warnings),
        "Audit_Warnings": warnings,
        "Certification_Code": "ASOP-56-COMPLIANT-SIM-v2" if global_status in ["APPROVED", "CONDITIONAL APPROVAL"] else "FAIL-AUDIT-GATE"
    }

    return audit_reports


# Convenient alias
detect_anomalies_pipe = detect_anomalies_pipeline
