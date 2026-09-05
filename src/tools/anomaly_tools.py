"""
anomaly_tools.py — Actuarial tools for Notebook 02: Anomaly Detection.
Contains 8 focused tools (Special permission granted: up to 8 functions for Notebook 2).
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm


def tool_check_data_quality(
    df: pd.DataFrame,
    num_cols: Union[List[str], str],
    claim_count_col: Optional[str] = None,
    claim_amount_col: Optional[str] = None
):
    """
    Performs logical sanity checks on raw claims and rating data.
    """
    if isinstance(num_cols, str) and claim_amount_col is None:
        claim_nb_col = num_cols
        claim_amt_col = claim_count_col
        neg_counts = df[claim_nb_col] < 0 if claim_nb_col in df.columns else pd.Series(False, index=df.index)
        neg_amounts = df[claim_amt_col] < 0 if claim_amt_col in df.columns else pd.Series(False, index=df.index)
        unlinked = (df[claim_nb_col] == 0) & (df[claim_amt_col] > 0) if (claim_nb_col in df.columns and claim_amt_col in df.columns) else pd.Series(False, index=df.index)
        return neg_counts | neg_amounts | unlinked

    issues = {}
    cols = num_cols if isinstance(num_cols, list) else []
    for col in cols:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues[f"{col}_negative"] = int(neg)

    if claim_amount_col and claim_amount_col in df.columns:
        neg_amt = (df[claim_amount_col] < 0).sum()
        if neg_amt > 0:
            issues["negative_claim_amounts"] = int(neg_amt)

    if claim_count_col and claim_count_col in df.columns:
        neg_cnt = (df[claim_count_col] < 0).sum()
        if neg_cnt > 0:
            issues["negative_claim_counts"] = int(neg_cnt)

    if claim_count_col and claim_amount_col and claim_count_col in df.columns and claim_amount_col in df.columns:
        zero_claims_pos_amt = ((df[claim_count_col] == 0) & (df[claim_amount_col] > 0)).sum()
        if zero_claims_pos_amt > 0:
            issues["zero_claims_positive_amount"] = int(zero_claims_pos_amt)

    return issues


def tool_analyze_tail_advanced(
    df_subset: pd.DataFrame,
    claim_amount_col: str,
    claim_count_col: str,
    name: str = "Portfolio"
) -> Dict[str, Union[str, float, int, pd.DataFrame]]:
    """
    Analyzes distribution tails separating zero-claim vs. active-claim cohorts.
    """
    zero_pct = (df_subset[claim_amount_col] == 0).mean() * 100 if claim_amount_col in df_subset.columns else 0.0
    pos_claims = df_subset[df_subset[claim_amount_col] > 0] if claim_amount_col in df_subset.columns else df_subset

    pcts = [0.50, 0.75, 0.90, 0.95, 0.99, 0.992, 0.994, 0.996, 0.998, 0.999]
    idx_labels = ["P50", "P75", "P90", "P95", "P99", "P99.2", "P99.4", "P99.6", "P99.8", "P99.9"]

    sev_series = pos_claims[claim_amount_col] if len(pos_claims) > 0 and claim_amount_col in pos_claims.columns else pd.Series([0])
    tbl_sev = pd.DataFrame({
        "Conditional Claim Amount ($)": [sev_series.quantile(p) for p in pcts]
    }, index=idx_labels)

    tail_vars = [c for c in ["CarAge", "DriverAge", "Density"] if c in df_subset.columns]
    tbl_overall = pd.DataFrame({
        col: [df_subset[col].quantile(p) for p in pcts] for col in tail_vars
    }, index=idx_labels)

    return {
        "name": name,
        "zero_pct": zero_pct,
        "n_pos_claims": len(pos_claims),
        "severity_tail": tbl_sev,
        "overall_tail": tbl_overall
    }


def tool_flag_tail_outliers(
    df: pd.DataFrame,
    tail_cols: List[str],
    quantile: float = 0.99
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """
    Flags univariate tail outliers exceeding specified quantile thresholds.
    """
    p99 = df[tail_cols].quantile(quantile)
    df_out = df.copy()
    df_out["tail_flag"] = False
    df_out["tail_reason"] = ""

    for col in tail_cols:
        mask = df_out[col] > p99[col]
        df_out.loc[mask, "tail_flag"] = True
        df_out.loc[mask, "tail_reason"] += f"{col}>P{int(quantile*100)} "

    n_tail = int(df_out["tail_flag"].sum())
    return df_out, p99, n_tail


def tool_run_isolation_forest(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42
):
    """
    Detects multivariate feature anomalies using an Isolation Forest pipeline.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    X_processed = preprocessor.fit_transform(df)

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    iso.fit(X_processed)

    df_out = df.copy()
    df_out["IF_Score"] = -iso.decision_function(X_processed)
    df_out["IF_Label"] = iso.predict(X_processed)

    return df_out, preprocessor, iso, X_processed.shape


def tool_calculate_leverage(
    df: pd.DataFrame,
    num_cols: List[str],
    leverage_multiplier: float = 2.0
) -> Tuple[pd.DataFrame, float, int, int]:
    """
    Computes Hat Matrix diagonal leverage (h_ii) in the predictor space.
    """
    X_num = df[num_cols].fillna(df[num_cols].median()).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    X_sm = sm.add_constant(X_scaled)

    try:
        XtX_inv = np.linalg.pinv(X_sm.T @ X_sm)
        H_diag  = np.einsum("ij,jk,ik->i", X_sm, XtX_inv, X_sm)
    except Exception:
        H_diag = np.full(len(df), np.nan)

    n = len(df)
    p = X_sm.shape[1]
    threshold = leverage_multiplier * (p / n)

    df_out = df.copy()
    df_out["Leverage"] = H_diag
    df_out["high_leverage"] = df_out["Leverage"] > threshold

    return df_out, float(threshold), p, n


def tool_calculate_deviance_residuals(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    claim_count_col: str,
    exposure_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Fits diagnostic Poisson GLM and computes signed deviance residuals.
    """
    df_out = df.copy()
    if not claim_count_col or claim_count_col not in df.columns:
        df_out["GLM_Pred"]     = np.nan
        df_out["Residual_Dev"] = np.nan
        df_out["Residual_Raw"] = np.nan
        return df_out, None

    df_enc = df[num_cols].fillna(df[num_cols].median())
    df_enc_cat = pd.get_dummies(df[cat_cols].fillna("Unknown"), drop_first=True, dtype=float)
    X_glm = pd.concat([df_enc, df_enc_cat], axis=1)
    X_glm_sm = sm.add_constant(X_glm.astype(float))

    y_glm = df[claim_count_col].values.astype(float)
    if exposure_col and exposure_col in df.columns:
        offset_glm = np.log(np.clip(df[exposure_col].values.astype(float), 1e-12, None))
    else:
        offset_glm = np.zeros(len(df))

    glm_res = sm.GLM(
        y_glm, X_glm_sm,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=offset_glm
    ).fit(maxiter=50)

    df_out["GLM_Pred"]     = glm_res.fittedvalues
    df_out["Residual_Dev"] = glm_res.resid_deviance
    df_out["Residual_Raw"] = y_glm - glm_res.fittedvalues

    return df_out, glm_res


def tool_identify_influential_points(
    df: pd.DataFrame,
    resid_threshold: float = 3.0
) -> Tuple[pd.DataFrame, int]:
    """
    Identifies influential points combining high leverage with large deviance residuals.
    """
    df_out = df.copy()
    df_out["high_residual"] = df_out["Residual_Dev"].abs() > resid_threshold
    df_out["influential"]   = df_out.get("high_leverage", False) & df_out["high_residual"]
    n_inf = int(df_out["influential"].sum())
    return df_out, n_inf


def tool_classify_business_review(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Categorizes flagged observations into the 4-tier Actuarial Business Review Matrix.
    """
    df_out = df.copy()

    def classify_row(row):
        is_if   = row.get("IF_Label") == -1
        is_tail = bool(row.get("tail_flag", False))
        is_inf  = bool(row.get("influential", False))
        is_lev  = bool(row.get("high_leverage", False))

        if is_if and is_tail and is_inf:
            return "🔴 Likely Error"
        elif is_if and is_inf:
            return "🟠 Suspicious"
        elif is_if and is_tail:
            return "🟠 Suspicious"
        elif is_if:
            return "🟡 Rare (IF only)"
        elif is_tail:
            return "🟡 Rare (Tail only)"
        elif is_lev:
            return "🟡 Unusual predictor"
        else:
            return "🟢 Normal"

    df_out["Review_Class"] = df_out.apply(classify_row, axis=1)
    return df_out


# --- Backward compatibility helpers ---
def tool_calculate_tail_percentiles(df: pd.DataFrame, cols: List[str], percentiles: Optional[List[float]] = None) -> pd.DataFrame:
    if percentiles is None:
        percentiles = [0.50, 0.75, 0.90, 0.95, 0.99, 0.992, 0.994, 0.996, 0.998, 0.999]
    pct_dict = {}
    for col in cols:
        if col in df.columns:
            series = df[col].dropna()
            pct_dict[col] = [round(float(series.quantile(p)), 2) for p in percentiles]
    idx_labels = [f"P{p*100:.1f}".replace(".0", "") for p in percentiles]
    return pd.DataFrame(pct_dict, index=idx_labels)


def tool_calculate_leverage_and_residuals(X_design: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, leverage_multiplier: float = 2.0, residual_threshold: float = 3.0):
    n, p = X_design.shape
    inv_xtx = np.linalg.pinv(X_design.T @ X_design)
    leverage = np.sum((X_design @ inv_xtx) * X_design, axis=1)
    y_pos = np.where(y_true > 0, y_true, 1.0)
    mu = np.clip(y_pred, 1e-6, None)
    term1 = np.where(y_true > 0, y_true * np.log(y_pos / mu), 0.0)
    term2 = y_true - mu
    dev_term = 2.0 * np.clip(term1 - term2, 0.0, None)
    deviance_residuals = np.sign(y_true - mu) * np.sqrt(dev_term)
    lev_thresh = leverage_multiplier * (p / n)
    high_leverage = leverage > lev_thresh
    high_residual = np.abs(deviance_residuals) > residual_threshold
    influential = high_leverage & high_residual
    return {
        "leverage": leverage,
        "deviance_residuals": deviance_residuals,
        "leverage_threshold": float(lev_thresh),
        "influential_mask": influential
    }


def tool_classify_business_actions(df: pd.DataFrame, dq_flag, tail_flag, iso_flag, inf_flag):
    df_audit = df.copy()
    df_audit["DQ_Flag"] = dq_flag
    df_audit["Tail_Flag"] = tail_flag
    df_audit["ISO_Flag"] = iso_flag
    df_audit["Influence_Flag"] = inf_flag
    conditions = [
        df_audit["DQ_Flag"],
        df_audit["Influence_Flag"],
        df_audit["Tail_Flag"] | df_audit["ISO_Flag"]
    ]
    choices_cat = ["Likely Data Error", "High Influence Outlier", "Rare Genuine Risk"]
    choices_act = ["REMOVE", "REMOVE", "KEEP"]
    df_audit["Anomaly_Category"] = np.select(conditions, choices_cat, default="Normal")
    df_audit["Business_Action"] = np.select(conditions, choices_act, default="KEEP")
    df_clean = df_audit[df_audit["Business_Action"] == "KEEP"].copy()
    return df_clean, df_audit
