"""
credibility_tools.py — Actuarial tools for Notebook 05: Credibility & Calibration.
Contains 4 focused tools (Constraint: <= 6 tools per file).
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


def tool_calculate_pure_premium(
    freq_pred: np.ndarray,
    sev_pred: np.ndarray
) -> np.ndarray:
    """
    Multiplies expected frequency by expected severity to compute Pure Premium.

    What this tool does:
        Calculates the expected annual loss cost per policy: E[Pure Premium] = E[Freq] * E[Sev].

    Why it is used:
        Fundamental actuarial equation of risk cost. Represents the break-even technical price.
    """
    return freq_pred * sev_pred


def tool_segment_risk_bands(
    df: pd.DataFrame,
    pure_premium: np.ndarray,
    n_bands: int = 5
) -> pd.DataFrame:
    """
    Bins the portfolio into quantile risk tiers based on pure premium indications.

    What this tool does:
        Segments continuous risk scores into discrete bands (e.g., Tier 1 to Tier 5)
        to facilitate credibility group analysis.

    Why it is used:
        Allows underwriters and actuaries to examine cohort-level loss ratios and credibility.
    """
    df_out = df.copy()
    df_out["Pure_Premium"] = pure_premium
    df_out["Risk_Band"] = pd.qcut(df_out["Pure_Premium"].rank(method="first"), q=n_bands, labels=False) + 1
    return df_out


def tool_calibrate_buhlmann_credibility(
    df_segmented: pd.DataFrame,
    band_col: str,
    exposure_col: str,
    actual_loss_col: str,
    exp_loss_col: str,
    K: float = 1000.0
) -> Tuple[pd.DataFrame, float]:
    """
    Applies Bühlmann empirical Bayes credibility to blend group and portfolio loss experience.

    What this tool does:
        Calculates the credibility weight Z = n / (n + K) for each band and computes
        unadjusted Risk Adjustment Factors (RAF).

    Why it is used:
        Optimally balances the bias-variance tradeoff under classical credibility theory.
        Thin segments are blended toward the portfolio mean; large cohorts rely on their own data.
    """
    agg = df_segmented.groupby(band_col).agg(
        exposure=(exposure_col, "sum"),
        observed_loss=(actual_loss_col, "sum"),
        expected_loss=(exp_loss_col, "sum")
    ).reset_index()

    agg["credibility_Z"] = agg["exposure"] / (agg["exposure"] + K)
    agg["observed_ratio"] = agg["observed_loss"] / agg["expected_loss"]
    agg["unadj_RAF"] = agg["credibility_Z"] * agg["observed_ratio"] + (1.0 - agg["credibility_Z"]) * 1.0

    total_observed = agg["observed_loss"].sum()
    total_unadj_expected = (agg["expected_loss"] * agg["unadj_RAF"]).sum()
    correction_factor = total_observed / total_unadj_expected if total_unadj_expected > 0 else 1.0

    agg["adj_RAF"] = agg["unadj_RAF"] * correction_factor
    return agg, float(correction_factor)


def tool_enforce_revenue_neutrality(
    df_segmented: pd.DataFrame,
    band_col: str,
    band_cred_df: pd.DataFrame
) -> pd.Series:
    """
    Maps revenue-neutral Risk Adjustment Factors (RAF) back to individual policies.

    What this tool does:
        Applies the credibility-adjusted, balanced multiplier to each policyholder based on
        their assigned risk tier.

    Why it is used:
        Guarantees that credibility blending does not arbitrarily inflate or deflate overall
        portfolio revenue.
    """
    raf_map = band_cred_df.set_index(band_col)["adj_RAF"].to_dict()
    return df_segmented[band_col].map(raf_map)
