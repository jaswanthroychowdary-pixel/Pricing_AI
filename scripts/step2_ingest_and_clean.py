"""
step2_ingest_and_clean.py

Step 2: Ingest & Clean Pipeline using pricing_functions_v2.
1. Ingests raw Excel portfolio (data/raw/Pricing_Data.xlsx)
2. Runs profile_data_and_synthesize to create baseline metadata (outputs/meta.json)
3. Runs detect_anomalies_pipe to generate binary global_anomaly_flag
4. Filters out anomalies to produce clean modeling dataset (outputs/df_clean.parquet)
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from pricing_functions_v2 import profile_data_and_synthesize, detect_anomalies_pipe


def run_step2_ingest_and_clean(
    raw_path: str = "data/raw/Pricing_Data.xlsx",
    output_dir: str = "outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Step 2: Ingest & Clean Pipeline ===")
    print(f"1. Ingesting raw portfolio from: {raw_path}")
    df_raw = pd.read_excel(raw_path)
    print(f"   Loaded: {len(df_raw):,} policies with {len(df_raw.columns)} columns")
    
    # 2. Data Profiling & Exposure Synthesis
    print("2. Running profile_data_and_synthesize...")
    profile_res = profile_data_and_synthesize(
        df_raw,
        claim_nb_col="ClaimNb",
        claim_amount_col="ClaimAmount",
        exposure_col="Exposure"
    )
    df_profiled = profile_res["df"]
    meta = profile_res["meta"]
    
    # Identify rating features
    num_cols = [c for c in ["CarAge", "DriverAge", "Density"] if c in df_profiled.columns]
    cat_cols = [c for c in ["Power", "Brand", "Gas", "Region"] if c in df_profiled.columns]
    
    meta["num_cols"] = num_cols
    meta["cat_cols"] = cat_cols
    meta["claim_count_col"] = meta["claim_nb_column_used"]
    meta["claim_amount_col"] = meta["claim_amount_column_used"]
    meta["exposure_col"] = meta["exposure_column_used"]
    
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   ✓ Saved metadata to: {meta_path}")
    
    profiled_path = os.path.join(output_dir, "df_profiled.parquet")
    df_profiled.to_parquet(profiled_path, index=False)
    print(f"   ✓ Saved profiled dataset to: {profiled_path}")
    
    # 3. Anomaly Detection Pipeline
    print("3. Running detect_anomalies_pipeline...")
    anomaly_res = detect_anomalies_pipe(
        df=df_profiled,
        predictor_cols=num_cols,
        claim_nb_col=meta["claim_count_col"],
        claim_amount_col=meta["claim_amount_col"],
        contamination_rate=0.01,
        leverage_multiplier=3.0,
        residual_threshold=2.0
    )
    df_flagged = anomaly_res["df"]
    anomaly_metrics = anomaly_res["metrics"]
    
    print(f"   Total anomalies flagged: {anomaly_metrics['global_flagged_count']:,} ({anomaly_metrics['global_flagged_pct']:.2f}%)")
    print(f"     - Tail extremes (P99.9): {anomaly_metrics['tail_extreme_count_P99_9']:,}")
    print(f"     - Isolation Forest:     {anomaly_metrics['iso_forest_flagged_count']:,}")
    print(f"     - High Influence GLM:   {anomaly_metrics['high_influence_anomaly_count']:,}")
    
    # 4. Clean dataset for modeling
    print("4. Filtering anomalies to build modeling dataset...")
    df_clean = df_flagged[~df_flagged["global_anomaly_flag"]].copy()
    clean_path = os.path.join(output_dir, "df_clean.parquet")
    df_clean.to_parquet(clean_path, index=False)
    print(f"   ✓ Clean dataset saved to: {clean_path} ({len(df_clean):,} policies)")
    print("=== Step 2 Complete ===")
    
    return df_clean, meta, anomaly_metrics


if __name__ == "__main__":
    run_step2_ingest_and_clean()
