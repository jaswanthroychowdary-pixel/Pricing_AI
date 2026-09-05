"""
premium_tools.py — Actuarial tools for Notebook 06: Final Commercial Premium.
Contains 4 focused tools (Constraint: <= 6 tools per file).
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
import json


def tool_calculate_commercial_premium(
    pure_premium: np.ndarray,
    large_loss_loading: float = 1.10,
    risk_adj_factor: Optional[np.ndarray] = None,
    profit_margin: float = 1.05,
    floor: float = 50.0,
    cap: float = 5000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes the Commercial Pricing Equation to compute final customer premiums.

    What this tool does:
        Multiplies pure premium by catastrophic safety loading (1.10), underwriting margin (1.05),
        and credibility factor, then clips the result between the minimum floor and maximum cap.

    Why it is used:
        Transforms mathematical cost into marketable, solvent commercial premiums.

    Underlying Process:
        Gross = Pure_Premium * L * RAF * M.
        Final = clip(Gross, floor, cap). Returns (final_premium, gross_premium).
    """
    raf = 1.0 if risk_adj_factor is None else np.asarray(risk_adj_factor)
    gross = pure_premium * large_loss_loading * raf * profit_margin
    final = np.clip(gross, floor, cap)
    return final, gross


def tool_compute_premium_diagnostics(
    gross_premium: np.ndarray,
    final_premium: np.ndarray,
    floor: float,
    cap: float
) -> Dict[str, Union[float, int]]:
    """
    Audits portfolio-level rate distributions and boundary clipping impacts.

    What this tool does:
        Measures the count and percentage of policies constrained by the commercial floor
        or cap, and computes average premium and total revenue.

    Why it is used:
        Protects against adverse selection (too many capped high-risk drivers) and uncompetitive
        pricing (too many floored low-risk drivers).
    """
    n = len(final_premium)
    n_floor = int(np.sum(gross_premium < floor))
    n_cap = int(np.sum(gross_premium > cap))

    return {
        "n_policies": n,
        "total_revenue": round(float(np.sum(final_premium)), 2),
        "mean_premium": round(float(np.mean(final_premium)), 2),
        "median_premium": round(float(np.median(final_premium)), 2),
        "policies_at_floor": n_floor,
        "pct_at_floor": round((n_floor / n) * 100.0, 2),
        "policies_at_cap": n_cap,
        "pct_at_cap": round((n_cap / n) * 100.0, 2)
    }


def tool_compute_decile_ae_chart(
    df: pd.DataFrame,
    actual_loss_col: str,
    premium_col: str,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Computes Actual-to-Expected (A/E) loss ratios across pricing deciles.

    What this tool does:
        Ranks policies into 10 premium deciles and computes Actual Claims / Expected Claims
        to evaluate rate adequacy across all customer risk tiers.

    Why it is used:
        Standard actuarial validation requirement before rate filing. A flat curve near 1.0
        proves absence of segment cross-subsidies.
    """
    df_eval = df.copy()
    df_eval["Decile"] = pd.qcut(df_eval[premium_col].rank(method="first", ascending=False), q=n_deciles, labels=False) + 1

    dec_stats = df_eval.groupby("Decile").agg(
        actual_total=(actual_loss_col, "sum"),
        premium_total=(premium_col, "sum"),
        count=(premium_col, "count")
    ).reset_index()

    dec_stats["AE_Ratio"] = round(dec_stats["actual_total"] / dec_stats["premium_total"], 4)
    return dec_stats


def tool_export_pricing_portfolio(
    df_priced: pd.DataFrame,
    registry_data: dict,
    excel_path: str,
    json_path: str,
    parquet_path: str
) -> None:
    """
    Exports commercial pricing deliverables and registers parameters for governance.

    What this tool does:
        Writes FINAL_priced.xlsx for underwriting, df_final_premiums.parquet for data pipelines,
        and pricing_registry.json for regulatory audits.
    """
    df_priced.to_parquet(parquet_path, index=False)
    df_priced.to_excel(excel_path, index=False)
    with open(json_path, "w") as f:
        json.dump(registry_data, f, indent=2)
