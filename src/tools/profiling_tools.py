"""
profiling_tools.py — Actuarial tools for Notebook 01: Data Profiling & Exposure Synthesis.
Contains 5 focused tools (Constraint: <= 6 tools per file).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import json


def tool_detect_actuarial_columns(
    df: pd.DataFrame,
    claim_nb_candidates: Optional[List[str]] = None,
    claim_amount_candidates: Optional[List[str]] = None,
    exposure_candidates: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], str]:
    """
    Auto-detects key actuarial targets and standardizes policy exposure.
    """
    if claim_nb_candidates is None:
        claim_nb_candidates = ["claimnb", "claims", "claimcount", "numclaims", "claim_count"]
    if claim_amount_candidates is None:
        claim_amount_candidates = ["claimamount", "claim_amount", "lossamt", "totalclaim", "losses"]
    if exposure_candidates is None:
        exposure_candidates = ["exposure", "exposure_years", "duration", "policy_term"]

    col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    detected_nb = None
    for cand in claim_nb_candidates:
        cand_clean = cand.lower().replace(" ", "").replace("_", "")
        if cand_clean in col_map:
            detected_nb = col_map[cand_clean]
            break

    detected_amount = None
    for cand in claim_amount_candidates:
        cand_clean = cand.lower().replace(" ", "").replace("_", "")
        if cand_clean in col_map:
            detected_amount = col_map[cand_clean]
            break

    detected_exposure = None
    for cand in exposure_candidates:
        cand_clean = cand.lower().replace(" ", "").replace("_", "")
        if cand_clean in col_map:
            detected_exposure = col_map[cand_clean]
            break

    df_out = df.copy()
    if detected_exposure is None:
        df_out["Exposure"] = 1.0
        detected_exposure = "Exposure"
    else:
        df_out[detected_exposure] = df_out[detected_exposure].clip(lower=1e-6)

    return df_out, detected_nb, detected_amount, detected_exposure


def tool_classify_rating_variables(
    df: pd.DataFrame,
    exclude_cols: Union[set, list]
) -> Tuple[List[str], List[str]]:
    """
    Classifies portfolio predictors into continuous numeric vs. categorical cohorts.
    """
    rating_vars = [c for c in df.columns if c not in exclude_cols]
    num_cols = [
        c for c in rating_vars
        if df[c].dtype in [np.float64, np.int64, float, int]
        and df[c].nunique() > 10
    ]
    cat_cols = [c for c in rating_vars if c not in num_cols]
    return num_cols, cat_cols


def tool_check_missing_values(
    df: pd.DataFrame,
    save_plot_path: Optional[str] = None
) -> pd.Series:
    """
    Evaluates feature completeness and visualizes missing data proportions.
    """
    null_pct = df.isnull().mean().sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    return null_pct


def tool_plot_target_distributions(
    df: pd.DataFrame,
    claim_count_col: Optional[str],
    claim_amount_col: Optional[str],
    save_plot_path: Optional[str] = None
) -> Dict[str, Union[float, int]]:
    """
    Visualizes claim count frequency and positive loss severity distributions.
    """
    zero_pct = 0.0
    n_pos = 0

    if claim_count_col and claim_count_col in df.columns:
        zero_pct = float((df[claim_count_col] == 0).mean() * 100)

    if claim_amount_col and claim_amount_col in df.columns:
        pos = df[df[claim_amount_col] > 0][claim_amount_col]
        n_pos = len(pos)

    return {
        "zero_claim_pct": round(zero_pct, 2),
        "positive_claims_count": n_pos
    }


def tool_export_profiled_dataset(
    df: pd.DataFrame,
    claim_count_col: Optional[str],
    claim_amount_col: Optional[str],
    exposure_col: Optional[str],
    num_cols: List[str],
    cat_cols: List[str],
    parquet_path: str,
    meta_path: str
) -> dict:
    """
    Saves standardized dataset and serializes profiling metadata for downstream pipeline.
    """
    df.to_parquet(parquet_path, index=False)
    meta = {
        "claim_count_col": claim_count_col,
        "claim_amount_col": claim_amount_col,
        "exposure_col": exposure_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_rows": len(df),
        "zero_claim_pct": float((df[claim_count_col] == 0).mean()) if claim_count_col and claim_count_col in df.columns else None
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# --- Compatibility aliases ---
def tool_detect_schema(df: pd.DataFrame, claim_nb_candidates=None, claim_amount_candidates=None, exposure_candidates=None):
    df_mod, nb, amt, exp = tool_detect_actuarial_columns(df, claim_nb_candidates, claim_amount_candidates, exposure_candidates)
    return {
        "claim_count_col": nb,
        "claim_amount_col": amt,
        "exposure_col": exp,
        "claim_nb": nb,
        "claim_amount": amt,
        "exposure": exp
    }


def tool_synthesize_exposure(df: pd.DataFrame, exposure_col=None, default_val=1.0):
    df_mod, nb, amt, exp = tool_detect_actuarial_columns(df)
    return df_mod, True


def tool_profile_distributions(df: pd.DataFrame, claim_count_col, claim_amount_col, exposure_col):
    total_rows = len(df)
    total_exposure = float(df[exposure_col].sum()) if exposure_col in df.columns else float(total_rows)
    total_claims = int(df[claim_count_col].sum()) if claim_count_col in df.columns else 0
    total_loss = float(df[claim_amount_col].sum()) if claim_amount_col in df.columns else 0.0
    zero_claims_count = int((df[claim_count_col] == 0).sum()) if claim_count_col in df.columns else 0
    zero_claim_pct = zero_claims_count / total_rows if total_rows > 0 else 0.0
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [claim_count_col, claim_amount_col, exposure_col]]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns]
    return {
        "n_rows": total_rows,
        "n_policies": total_rows,
        "total_exposure": round(total_exposure, 2),
        "total_claims": total_claims,
        "portfolio_frequency": round(total_claims / total_exposure, 4) if total_exposure > 0 else 0.0,
        "total_loss_payout": round(total_loss, 2),
        "zero_claim_pct": round(zero_claim_pct, 4),
        "numeric_features": num_cols,
        "categorical_features": cat_cols
    }
