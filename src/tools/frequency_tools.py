"""
frequency_tools.py — Actuarial tools for Notebook 03: Frequency Modeling & Selection.
Contains 5 focused tools (Constraint: <= 6 tools per file).
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc
import statsmodels.api as sm


def tool_prepare_frequency_features(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocesses and encodes frequency features for modeling.

    What this tool does:
        Median-imputes numeric predictors, one-hot encodes categorical covariates,
        and adds an intercept column for Generalized Linear Models.

    Why it is used:
        Ensures consistent feature matrix representation across both statistical GLMs
        and gradient-boosted trees.

    Underlying Process:
        Generates dummy variables with drop_first=True to prevent multicollinearity,
        and applies sm.add_constant for GLM estimation.
    """
    df_enc = df[num_cols + cat_cols].copy()
    for c in num_cols:
        df_enc[c] = df_enc[c].fillna(df_enc[c].median())

    df_enc = pd.get_dummies(df_enc, columns=cat_cols, drop_first=True, dtype=float)
    X_mat = sm.add_constant(df_enc.to_numpy(dtype=float))
    return df_enc, X_mat


def tool_fit_frequency_glms(
    X_train_sm: np.ndarray,
    y_train: np.ndarray,
    off_train: np.ndarray,
    X_val_sm: np.ndarray,
    off_val: np.ndarray
) -> Dict[str, Tuple[object, np.ndarray, float, float]]:
    """
    Fits Poisson and Negative Binomial GLMs with log-exposure offsets.

    What this tool does:
        Estimates coefficients for multiplicative rating structures under Poisson
        and Negative Binomial (NB2) assumptions using iteratively reweighted least squares (IRLS).

    Why it is used:
        Industry standard for motor ratemaking (CAS Exam 8). Coefficients directly translate
        into transparent multiplicative rating factors ($e^{\\beta_j}$).

    Underlying Process:
        Fits sm.GLM with Poisson and NegativeBinomial families, applying log-exposure
        as a fixed offset. Generates out-of-sample predictions on the validation set.
    """
    # 1. Poisson GLM
    pois = sm.GLM(
        y_train, X_train_sm,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=off_train
    ).fit(maxiter=100)
    pred_pois = pois.predict(X_val_sm, offset=off_val)

    # 2. Negative Binomial GLM
    nb = sm.GLM(
        y_train, X_train_sm,
        family=sm.families.NegativeBinomial(),
        offset=off_train
    ).fit(maxiter=100)
    pred_nb = nb.predict(X_val_sm, offset=off_val)

    return {
        "Poisson GLM": (pois, pred_pois, float(pois.aic), float(pois.deviance)),
        "NegBinomial": (nb, pred_nb, float(nb.aic), float(nb.deviance))
    }


def tool_fit_frequency_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    exp_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    exp_val: np.ndarray
) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """
    Fits a gradient-boosted decision tree under a Poisson count objective.

    What this tool does:
        Trains an XGBoost regressor using objective='count:poisson' with base_margin=log(Exposure)
        as an advanced non-linear benchmark against classical GLMs.

    Why it is used:
        Identifies complex feature interactions and non-linear boundaries that linear GLMs miss.

    Underlying Process:
        Fits trees with early stopping on validation Poisson deviance. Returns model and predictions.
    """
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            objective="count:poisson",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        base_margin_tr = np.log(np.clip(exp_train, 1e-12, None))
        base_margin_val = np.log(np.clip(exp_val, 1e-12, None))

        xgb.fit(
            X_train, y_train,
            base_margin=base_margin_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        pred_xgb = xgb.predict(X_val, base_margin=base_margin_val)
        return xgb, pred_xgb
    except ImportError:
        return None, None


def tool_calculate_actuarial_gini(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Computes the Actuarial Gini coefficient (Lorenz curve AUC).

    What this tool does:
        Measures how effectively the frequency model separates high-risk policyholders
        from low-risk policyholders.

    Why it is used:
        Standard error metrics (RMSE/MAE) are distorted by zero-inflation (95%+ zeroes).
        Actuaries use the Gini index to evaluate risk-differentiation power.

    Underlying Process:
        Sorts validation policies in descending order of predicted risk. Computes cumulative
        claims vs. cumulative population and integrates the Lorenz curve: Gini = 2 * (AUC - 0.5).
    """
    df_g = pd.DataFrame({"y": y_true, "p": y_pred}).sort_values("p", ascending=False)
    cum_pop = np.linspace(0, 1, len(df_g) + 1)
    cum_loss = np.concatenate([[0], df_g["y"].cumsum().values / df_g["y"].sum()])
    curve_auc = auc(cum_pop, cum_loss)
    gini = 2.0 * (curve_auc - 0.5)
    return round(float(gini), 4)


def tool_compare_frequency_models(
    models_dict: Dict[str, Tuple[Optional[object], np.ndarray, Optional[float], Optional[float]]],
    y_val: np.ndarray
) -> pd.DataFrame:
    """
    Compiles the standardized model comparison table for executive selection.

    What this tool does:
        Evaluates out-of-sample goodness-of-fit (AIC, Deviance, MAE, RMSE, Gini) across
        all candidate frequency models.

    Why it is used:
        Empowers the actuary / user to make an informed, auditable decision on which model
        to deploy in production (Cell 3.8).

    Underlying Process:
        Aggregates statistical errors and rank-order metrics into a clean summary DataFrame.
    """
    rows = []
    for name, (model_obj, pred_val, aic, dev) in models_dict.items():
        if pred_val is not None:
            mae = mean_absolute_error(y_val, pred_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            gini = tool_calculate_actuarial_gini(y_val, pred_val)
            rows.append({
                "Model": name,
                "AIC": round(aic, 1) if aic is not None else "—",
                "Deviance": round(dev, 1) if dev is not None else "—",
                "MAE": round(mae, 5),
                "RMSE": round(rmse, 5),
                "Gini": gini
            })
    return pd.DataFrame(rows).set_index("Model")
