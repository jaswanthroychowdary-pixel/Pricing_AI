"""
severity_tools.py — Actuarial tools for Notebook 04: Severity Modeling & Diagnostics.
Contains 4 focused tools (Constraint: <= 6 tools per file).
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm


def tool_filter_positive_claims(
    df: pd.DataFrame,
    claim_amount_col: str,
    claim_count_col: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Isolates the positive claims cohort and computes observation weights.

    What this tool does:
        Filters the dataset strictly to policies where ClaimAmount > 0, and extracts
        claim counts as regression observation weights.

    Why it is used:
        Severity models the conditional distribution of loss given that a claim has occurred.
        Policies with multiple claims are weighted proportionally to their count.

    Underlying Process:
        Subsets df[df[claim_amount_col] > 0] and sets weights = df_pos[claim_count_col].
    """
    df_pos = df[df[claim_amount_col] > 0].copy()
    weights = df_pos[claim_count_col].values.astype(float)
    return df_pos, weights


def tool_fit_severity_models(
    X_sm: np.ndarray,
    y_sev: np.ndarray,
    weights: np.ndarray
) -> Dict[str, Tuple[object, np.ndarray, Optional[float], Optional[float]]]:
    """
    Fits candidate severity models (Gamma GLM, Log-Normal, Inverse Gaussian).

    What this tool does:
        Estimates claim payout severity using:
        1. Gamma GLM (log link, constant coefficient of variation)
        2. Log-Normal regression (WLS with smearing mean correction: exp(mu + 0.5*sigma^2))
        3. Inverse Gaussian GLM (log link, heavy-tail assumption)

    Why it is used:
        Loss distributions are right-skewed and strictly positive. Fitting multiple families
        ensures the actuary can evaluate tail weight and variance assumptions.

    Underlying Process:
        Fits GLMs with statsmodels and applies smearing bias correction for Log-Normal.
    """
    results = {}

    # 1. Gamma GLM
    gamma = sm.GLM(
        y_sev, X_sm,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=weights
    ).fit(maxiter=100)
    pred_gamma = gamma.predict(X_sm)
    results["Gamma GLM"] = (gamma, pred_gamma, float(gamma.aic), float(gamma.deviance))

    # 2. Log-Normal WLS
    log_y = np.log(y_sev)
    lognorm = sm.WLS(log_y, X_sm, weights=weights).fit()
    sigma2 = float(lognorm.scale)
    pred_lognorm = np.exp(lognorm.predict(X_sm) + 0.5 * sigma2)
    results["Log-Normal"] = (lognorm, pred_lognorm, float(lognorm.aic), None)

    # 3. Inverse Gaussian GLM
    try:
        invgauss = sm.GLM(
            y_sev, X_sm,
            family=sm.families.InverseGaussian(link=sm.families.links.Log()),
            var_weights=weights
        ).fit(maxiter=100)
        pred_invgauss = invgauss.predict(X_sm)
        results["Inv Gaussian"] = (invgauss, pred_invgauss, float(invgauss.aic), float(invgauss.deviance))
    except Exception:
        results["Inv Gaussian"] = (None, None, None, None)

    return results


def tool_compare_severity_models(
    models_dict: Dict[str, Tuple[Optional[object], Optional[np.ndarray], Optional[float], Optional[float]]],
    y_true: np.ndarray
) -> pd.DataFrame:
    """
    Compiles the severity model comparison table for user selection.

    What this tool does:
        Evaluates AIC, GLM deviance, dollar error (MAE, RMSE), and overall Actual-to-Expected
        (A/E) payout ratio across candidate severity models.

    Why it is used:
        Ensures the selected severity model accurately reflects historical loss costs
        and achieves global financial adequacy (A/E ≈ 1.0).

    Underlying Process:
        Calculates MAE, RMSE, and global A/E = sum(y_true) / sum(y_pred).
    """
    rows = []
    for name, (model_obj, pred_v, aic, dev) in models_dict.items():
        if pred_v is not None:
            mae = mean_absolute_error(y_true, pred_v)
            rmse = float(np.sqrt(mean_squared_error(y_true, pred_v)))
            ae = np.sum(y_true) / np.sum(pred_v)
            rows.append({
                "Model": name,
                "AIC": round(aic, 1) if aic is not None else "—",
                "Deviance": round(dev, 1) if dev is not None else "—",
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "Overall A/E": round(ae, 4)
            })
    return pd.DataFrame(rows).set_index("Model")


def tool_calculate_severity_residuals(
    model: object,
    X_sm: np.ndarray,
    y_true: np.ndarray
) -> np.ndarray:
    """
    Computes deviance residuals for severity regression diagnostic plots.

    What this tool does:
        Extracts deviance residuals from the fitted GLM to assess homoscedasticity
        and residual normality across the predicted severity range.

    Why it is used:
        Confirms whether the variance assumption (e.g., Var proportional to mu^2) holds true.
    """
    if hasattr(model, "resid_deviance"):
        return model.resid_deviance
    pred = model.predict(X_sm)
    return (y_true - pred) / np.std(y_true - pred)
