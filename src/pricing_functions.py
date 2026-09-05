"""
pricing_functions.py — Central Actuarial Pricing Facade.
Re-exports all discrete tools across the 7 pipeline notebooks.
"""

from src.tools.profiling_tools import (
    tool_detect_schema,
    tool_synthesize_exposure,
    tool_profile_distributions,
)
from src.tools.anomaly_tools import (
    tool_check_data_quality,
    tool_analyze_tail_advanced,
    tool_flag_tail_outliers,
    tool_run_isolation_forest,
    tool_calculate_leverage,
    tool_calculate_deviance_residuals,
    tool_identify_influential_points,
    tool_classify_business_review,
    tool_calculate_tail_percentiles,
    tool_calculate_leverage_and_residuals,
    tool_classify_business_actions,
)
from src.tools.frequency_tools import (
    tool_prepare_frequency_features,
    tool_fit_frequency_glms,
    tool_fit_frequency_xgboost,
    tool_calculate_actuarial_gini,
    tool_compare_frequency_models,
)
from src.tools.severity_tools import (
    tool_filter_positive_claims,
    tool_fit_severity_models,
    tool_compare_severity_models,
    tool_calculate_severity_residuals,
)
from src.tools.credibility_tools import (
    tool_calculate_pure_premium,
    tool_segment_risk_bands,
    tool_calibrate_buhlmann_credibility,
    tool_enforce_revenue_neutrality,
)
from src.tools.premium_tools import (
    tool_calculate_commercial_premium,
    tool_compute_premium_diagnostics,
    tool_compute_decile_ae_chart,
    tool_export_pricing_portfolio,
)
from src.tools.validation_tools import (
    tool_build_agent_dossiers,
    tool_run_agentic_audit_simulation,
    tool_save_validation_report,
)

__all__ = [
    # Profiling (3)
    "tool_detect_schema",
    "tool_synthesize_exposure",
    "tool_profile_distributions",
    # Anomaly (8 tools)
    "tool_check_data_quality",
    "tool_analyze_tail_advanced",
    "tool_flag_tail_outliers",
    "tool_run_isolation_forest",
    "tool_calculate_leverage",
    "tool_calculate_deviance_residuals",
    "tool_identify_influential_points",
    "tool_classify_business_review",
    "tool_calculate_tail_percentiles",
    "tool_calculate_leverage_and_residuals",
    "tool_classify_business_actions",
    # Frequency (5)
    "tool_prepare_frequency_features",
    "tool_fit_frequency_glms",
    "tool_fit_frequency_xgboost",
    "tool_calculate_actuarial_gini",
    "tool_compare_frequency_models",
    # Severity (4)
    "tool_filter_positive_claims",
    "tool_fit_severity_models",
    "tool_compare_severity_models",
    "tool_calculate_severity_residuals",
    # Credibility (4)
    "tool_calculate_pure_premium",
    "tool_segment_risk_bands",
    "tool_calibrate_buhlmann_credibility",
    "tool_enforce_revenue_neutrality",
    # Premium (4)
    "tool_calculate_commercial_premium",
    "tool_compute_premium_diagnostics",
    "tool_compute_decile_ae_chart",
    "tool_export_pricing_portfolio",
    # Validation (3)
    "tool_build_agent_dossiers",
    "tool_run_agentic_audit_simulation",
    "tool_save_validation_report",
]
