import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import re

# Set page configurations
st.set_page_config(
    page_title="🛡️ Agentic Actuarial Pricing Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import pricing functions
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pricing_functions_v2 import (
        profile_data_and_synthesize,
        detect_anomalies_pipeline,
        fit_and_evaluate_frequency_models,
        fit_and_evaluate_severity_models,
        calibrate_buhlmann_credibility,
        calculate_commercial_premium,
        simulate_agent_validation_audit
    )
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# -----------------------------------------------------------------------------
# 1. HELPERS & SYNTHETIC DATA GENERATOR
# -----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_rows=1000):
    """Generates realistic synthetic policy and claims data for demo purposes."""
    np.random.seed(42)
    
    age = np.random.randint(18, 75, n_rows)
    veh_value = np.random.uniform(5000, 60000, n_rows)
    region = np.random.choice(["Urban", "Suburban", "Rural"], n_rows, p=[0.4, 0.4, 0.2])
    risk_band = np.random.choice(["Band_A", "Band_B", "Band_C"], n_rows, p=[0.7, 0.2, 0.1])
    exposure = np.random.choice([1.0, 0.5, 0.25, 0.08], n_rows, p=[0.8, 0.1, 0.07, 0.03])
    
    base_lambda = 0.03
    age_factor = np.where(age < 25, 2.0, np.where(age > 60, 1.2, 1.0))
    region_factor = np.where(region == "Urban", 1.5, np.where(region == "Rural", 0.8, 1.0))
    val_factor = 1.0 + (veh_value / 30000) * 0.5
    
    lambda_i = base_lambda * age_factor * region_factor * val_factor * exposure
    claim_nb = np.random.poisson(lambda_i)
    
    claim_amount = np.zeros(n_rows)
    has_claim = (claim_nb > 0)
    
    shape = 2.0
    scale = np.where(risk_band == "Band_C", 1500, np.where(risk_band == "Band_B", 800, 400))
    
    for i in range(n_rows):
        if has_claim[i]:
            claim_amount[i] = float(np.sum(np.random.gamma(shape, scale[i], size=claim_nb[i])))
            
    df = pd.DataFrame({
        "PolicyID": [f"POL-{i+10000:05d}" for i in range(n_rows)],
        "Age": age,
        "VehicleValue": veh_value,
        "Region": region,
        "Risk_Band": risk_band,
        "Exposure": exposure,
        "ClaimNb": claim_nb,
        "ClaimAmount": claim_amount
    })
    
    df.loc[10, "ClaimNb"] = 0
    df.loc[10, "ClaimAmount"] = 12000.0
    df.loc[20, "ClaimNb"] = -1
    df.loc[20, "ClaimAmount"] = 0.0
    df.loc[30, "ClaimNb"] = 1
    df.loc[30, "ClaimAmount"] = 95000.0
    
    return df

# -----------------------------------------------------------------------------
# 2. LOCAL KNOWLEDGE REFERENCE (RAG CHAT ASSISTANT)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_rag_knowledge():
    """Loads and indexes chapters of the actuarial manuals for zero-hallucination lookup."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "docs", "actuarial_functions_doc_v2.md"),
        os.path.join(os.path.dirname(__file__), "docs", "actuarial_functions_doc.md"),
        os.path.join(os.path.dirname(__file__), "actuarial_functions_doc_v2.md"),
    ]
    
    docs_path = None
    for p in possible_paths:
        if os.path.exists(p):
            docs_path = p
            break
        
    if docs_path and os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        sections = re.split(r'\n## ', content)
        indexed_knowledge = []
        for sec in sections:
            lines = sec.strip().split("\n")
            if not lines:
                continue
            title = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:])
            indexed_knowledge.append({
                "title": title,
                "body": body,
                "full_text": f"## {title}\n{body}"
            })
        return indexed_knowledge
    return []

def search_rag_knowledge(query, knowledge_base, top_k=1):
    """Retrieves concise, targeted actuarial manual passages without flooding raw text."""
    if not knowledge_base:
        return "No local reference documentation loaded."
        
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_sections = []
    
    for sec in knowledge_base:
        text_lower = sec["full_text"].lower()
        score = 0
        for word in query_words:
            if len(word) > 3:
                if word in text_lower:
                    score += 2
                if word in sec["title"].lower():
                    score += 6
        scored_sections.append((score, sec["title"], sec["body"]))
        
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top relevant section
    best_matches = [s for s in scored_sections if s[0] > 0]
    if not best_matches:
        best_matches = [scored_sections[0]]
        
    formatted_passages = []
    for score, title, body in best_matches[:top_k]:
        # Extract the core introductory paragraphs (up to 500 chars)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        concise_body = "\n\n".join(paragraphs[:2]) if paragraphs else body[:400]
        formatted_passages.append(f"### {title}\n{concise_body}")
        
    return "\n\n---\n\n".join(formatted_passages)

# -----------------------------------------------------------------------------
# 3. STREAMLIT APP CORE LAYOUT
# -----------------------------------------------------------------------------
st.title("🛡️ Agentic Actuarial Pricing Engine")
st.markdown("---")

if not IMPORTS_OK:
    st.error(f"⚠️ **Import Error Detected:** Failed to import custom pricing functions module.\nError detail: `{IMPORT_ERROR}`")
    st.stop()

st.sidebar.markdown("## 🛡️ Pricing Control Center")

if st.sidebar.button("🔄 Start Fresh Session", help="Clears stored session state and resets the dashboard."):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 AI Agent Access")
api_key_box = st.sidebar.text_input(
    "Gemini API Key:",
    type="password",
    value=st.session_state.get("gemini_api_key", ""),
    placeholder="Paste API key here...",
    help="Enter your Gemini API key to activate live LLM audits. Left blank, the engine runs in simulated mode."
)
if api_key_box:
    st.session_state["gemini_api_key"] = api_key_box.strip()
    st.sidebar.success("API Key loaded for this session!")

st.sidebar.markdown("---")
st.sidebar.header("📁 Data Source Selection")
data_mode = st.sidebar.radio(
    "Choose Dataset:",
    ["None (Awaiting Selection)", "Upload Custom Portfolio", "Generate Synthetic Demo Portfolio"],
    index=0
)

raw_df = None

if data_mode == "Upload Custom Portfolio":
    uploaded_file = st.sidebar.file_uploader("Upload Excel / CSV / Parquet:", type=["xlsx", "csv", "parquet"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                raw_df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_parquet(uploaded_file)
            st.sidebar.success(f"Successfully loaded {len(raw_df):,} records!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            raw_df = None
    else:
        st.sidebar.info("Awaiting file upload...")
elif data_mode == "Generate Synthetic Demo Portfolio":
    demo_size = st.sidebar.slider("Synthetic Portfolio Size:", 500, 5000, 1000, step=100)
    raw_df = generate_synthetic_data(demo_size)
    st.sidebar.success(f"Generated {len(raw_df):,} synthetic policies!")

# If no data has been selected or uploaded, show the clean Landing / Start Screen
if raw_df is None:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border-radius: 12px; padding: 2.2rem; border: 1px solid #334155; margin-bottom: 2rem;">
        <h2 style="color: #F8FAFC; margin-top: 0; font-size: 1.9rem;">🛡️ Actuarial Pricing Control Center</h2>
        <p style="color: #94A3B8; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;">
            A clean session has been initialized. To begin auditing, rate-making, and multi-agent validation, please select or provide a policy portfolio in the left sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 1️⃣ Ingest & Audit
        - Select **Generate Synthetic Demo Portfolio** to simulate 1,000+ motor policies.
        - Or choose **Upload Custom Portfolio** (CSV, Excel, Parquet).
        - Multi-layer anomaly detection & ASOP 23 exposure verification run automatically upon ingestion.
        """)
    with col2:
        st.markdown("""
        ### 2️⃣ Pricing & Credibility
        - Dual **Frequency (Poisson) & Severity (Gamma)** generalized linear models.
        - **Bühlmann Empirical Bayes** credibility weighting.
        - Commercial tariff engine with profit margin and large-loss loading controls.
        """)
    with col3:
        st.markdown("""
        ### 3️⃣ Multi-Agent Governance
        - **5 Autonomous Validation Agents**: Data, Statistical, Financial, Underwriting, and Chief Actuary.
        - Grounded **AI Actuarial Assistant** (Gemini LLM) tracking data provenance facts vs assumptions.
        """)
        
    st.info("👈 **To Start:** Choose a dataset option from the sidebar on the left.")
    st.stop()

# Consolidated 5-Tab Executive Workflow
tab1, tab2, tab3, tab4, tab_assistant = st.tabs([
    "01 Ingestion & Quality Audit",
    "02 Risk Models (Freq + Sev)",
    "03 Pricing & Credibility",
    "04 Multi-Agent Audit",
    "💬 AI Actuarial Assistant"
])

# ─── TAB 1: DATA INGESTION & QUALITY AUDIT (COMBINED 01 + 02) ──────────────────
with tab1:
    st.header("📊 Step 1: Data Ingestion & Quality Audit")
    
    # A. Ingestion & Exposure Profiling
    with st.spinner("Profiling dataset & checking exposure..."):
        profile_results = profile_data_and_synthesize(raw_df)
        df_profiled = profile_results["df"]
        meta_json = profile_results["meta"]
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{meta_json['total_records']:,}")
    with col2:
        st.metric("Total Exposure Years", f"{meta_json['total_exposure_years']:.2f}")
    with col3:
        st.metric("Observed Claims", f"{meta_json['total_observed_claims']:,}")
    with col4:
        st.metric("Observed Severity (Avg)", f"£{meta_json['average_claim_severity']:,.2f}")
        
    # Provenance Alert
    if meta_json.get("exposure_synthesized"):
        st.warning("⚠️ **Actuarial Assumption Active (ASOP 23):** The raw dataset did not contain an Exposure column. The pipeline applied a standardized assumption of **1.0 annual earned exposure** per policy to ensure frequency model stability.")
    else:
        st.success(f"✅ **Dataset Fact:** Verified exposure column '{meta_json.get('exposure_column_used', 'Exposure')}' sourced directly from the raw dataset.")
        
    st.subheader("📋 Ingested Dataset Preview")
    st.dataframe(df_profiled.head(6), use_container_width=True)
    
    st.markdown("---")
    
    # B. Multi-Layer Anomaly & Influence Filtration
    st.subheader("🛡️ Multi-Layer Anomaly & Influence Filtration")
    num_cand = [c for c in ["Age", "CarAge", "DriverAge", "VehicleValue", "Density", "CarVal", "Power"] if c in df_profiled.columns]
    predictor_cols = num_cand if num_cand else [c for c in df_profiled.columns if df_profiled[c].dtype in ['int64', 'float64'] and c not in ['ClaimNb', 'ClaimAmount', 'Exposure', 'PolicyID']]
    
    with st.spinner("Executing multi-layer anomaly detection..."):
        anomaly_results = detect_anomalies_pipeline(df_profiled, predictor_cols=predictor_cols)
        df_flagged = anomaly_results["df"]
        anomaly_metrics = anomaly_results["metrics"]
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Quality Failures", f"{anomaly_metrics['dq_failures_count']} policies")
    with col2:
        st.metric("Isolation Forest Outliers", f"{anomaly_metrics['iso_forest_flagged_count']} policies")
    with col3:
        st.metric("High-Influence Outliers", f"{anomaly_metrics['high_influence_anomaly_count']} policies")
        
    st.info(f"The pipeline flagged **{anomaly_metrics['global_flagged_count']:,}** policies as anomalous ({anomaly_metrics['global_flagged_pct']:.2f}% of the portfolio).")
    
    filter_mode = st.radio("Pipeline Filtration Action:", [
        "EXCLUDE Flagged Anomalies from Downstream Modeling (Recommended Actuarial Path)",
        "KEEP Anomalies to test Model Volatility"
    ])
    
    if "EXCLUDE" in filter_mode:
        df_clean = df_flagged[df_flagged["global_anomaly_flag"] == False].copy()
        st.success(f"Filtration complete! Modeling dataset locked at {len(df_clean):,} clean policy rows.")
    else:
        df_clean = df_flagged.copy()
        st.warning("Anomalies retained. Downstream models might exhibit high variance.")

# ─── TAB 2: ACTUARIAL RISK MODELING (COMBINED 03 + 04) ────────────────────────
with tab2:
    st.header("📈 Step 2: Actuarial Risk Modeling (Frequency + Severity)")
    st.markdown("Fit and evaluate GLMs on clean portfolio data under **ASOP 56 (Modeling)** standards.")
    
    col_freq, col_sev = st.columns(2)
    
    # Left Column: Frequency Modeling
    with col_freq:
        st.subheader("1️⃣ Claim Frequency Modeling")
        with st.spinner("Fitting Frequency GLMs & XGBoost..."):
            freq_results = fit_and_evaluate_frequency_models(df_clean, predictor_cols=predictor_cols)
            freq_summary_df = freq_results["summary"]
            freq_preds = freq_results["predictions"]
            
        st.dataframe(freq_summary_df, use_container_width=True)
        valid_freq_models = [m for m in freq_summary_df.index if m in freq_preds.columns]
        chosen_freq_model = st.selectbox(
            "Select winning Frequency model:",
            options=valid_freq_models if valid_freq_models else list(freq_summary_df.index)
        )
        if chosen_freq_model in freq_preds.columns:
            df_clean["pred_freq"] = freq_preds[chosen_freq_model]
        else:
            df_clean["pred_freq"] = freq_preds.iloc[:, 0]
        st.success(f"Locked Frequency Model: **{chosen_freq_model}**")

    # Right Column: Severity Modeling
    with col_sev:
        st.subheader("2️⃣ Claim Severity Modeling")
        with st.spinner("Fitting Severity GLMs..."):
            try:
                sev_results = fit_and_evaluate_severity_models(df_clean, predictor_cols=predictor_cols)
                sev_summary_df = sev_results["summary"]
                sev_preds = sev_results["predictions"]
                
                st.dataframe(sev_summary_df, use_container_width=True)
                valid_sev_models = [m for m in sev_summary_df.index if m in sev_preds.columns]
                chosen_sev_model = st.selectbox(
                    "Select winning Severity model:",
                    options=valid_sev_models if valid_sev_models else list(sev_summary_df.index)
                )
                if chosen_sev_model in sev_preds.columns:
                    df_clean["pred_sev"] = sev_preds[chosen_sev_model]
                else:
                    df_clean["pred_sev"] = sev_preds.iloc[:, 0]
                st.success(f"Locked Severity Model: **{chosen_sev_model}**")
                SEV_SUCCESS = True
            except Exception as e:
                st.error(f"Failed to fit severity models: {e}")
                SEV_SUCCESS = False
                
    if SEV_SUCCESS:
        st.markdown("---")
        mean_freq = float(df_clean["pred_freq"].mean())
        mean_sev = float(df_clean["pred_sev"].mean())
        mean_pure = mean_freq * mean_sev
        st.info(f"💡 **Combined Modeled Pure Premium:** Mean Annual Frequency ({mean_freq:.4f}) × Mean Severity (£{mean_sev:,.2f}) = **£{mean_pure:.2f} per policy year**")

# ─── TAB 3: COMMERCIAL PRICING & CREDIBILITY (COMBINED 05 + 06) ───────────────
with tab3:
    st.header("⚖️ Step 3: Credibility Calibration & Commercial Tariff Engine")
    
    if not SEV_SUCCESS:
        st.warning("Please satisfy preceding modeling steps.")
    else:
        # A. Bühlmann Credibility
        st.subheader("1️⃣ Bühlmann Empirical Bayes Credibility")
        col_k, col_rev = st.columns([1, 2])
        with col_k:
            K_value = st.number_input("Bühlmann Constant K:", min_value=1.0, value=500.0, step=10.0)
            
        segment_var = "Risk_Band" if "Risk_Band" in df_clean.columns else "Region" if "Region" in df_clean.columns else df_clean.columns[0]
        df_clean["prior_pure_premium"] = df_clean["pred_freq"] * df_clean["pred_sev"]
        df_clean["observed_loss"] = df_clean["ClaimAmount"]
        
        with st.spinner("Calibrating credibility adjustments..."):
            cred_results = calibrate_buhlmann_credibility(
                df=df_clean,
                segment_col=segment_var,
                exposure_col="Exposure",
                observed_loss_col="observed_loss",
                predicted_loss_col="prior_pure_premium",
                K=K_value
            )
            segment_df = cred_results["segment_metrics"]
            correction_factor = cred_results["correction_factor"]
            
        st.dataframe(segment_df, use_container_width=True)
        st.metric("Portfolio Revenue-Neutral Correction Factor", f"{correction_factor:.5f}")
        
        raf_map = dict(zip(segment_df[segment_var], segment_df["adjusted_RAF"]))
        df_clean["adjusted_RAF"] = df_clean[segment_var].map(raf_map).fillna(1.0)
        
        st.markdown("---")
        
        # B. Commercial Premium Formula
        st.subheader("2️⃣ Commercial Pricing Engine (Loadings & Bounds)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            L_load = st.slider("Large Loss Loading (L):", 1.0, 1.5, 1.10, step=0.01)
        with col2:
            M_load = st.slider("Profit Margin (M):", 1.0, 1.3, 1.05, step=0.01)
        with col3:
            prem_floor = st.number_input("Premium Floor (£):", min_value=1.0, value=50.0)
        with col4:
            prem_cap = st.number_input("Premium Cap (£):", min_value=100.0, value=5000.0)
            
        with st.spinner("Evaluating final commercial premiums..."):
            premium_results = calculate_commercial_premium(
                predicted_freq=df_clean["pred_freq"].values,
                predicted_sev=df_clean["pred_sev"].values,
                risk_adjustment_factor=df_clean["adjusted_RAF"].values,
                large_loss_loading=L_load,
                profit_margin=M_load,
                premium_floor=prem_floor,
                premium_cap=prem_cap
            )
            df_clean["Final_Premium"] = premium_results["final_premium"]
            df_clean["Gross_Premium"] = premium_results["gross_premium"]
            commercial_metrics = premium_results["portfolio_metrics"]
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Portfolio Premium", f"£{commercial_metrics['Total_Premium_Collected_GBP']:,.2f}")
        with col2:
            st.metric("Average Final Premium", f"£{commercial_metrics['Average_Final_Premium_GBP']:.2f}")
        with col3:
            st.metric("Policies at Floor", f"{commercial_metrics['Policies_at_Floor_Count']} ({commercial_metrics['Policies_at_Floor_Pct']:.2f}%)")
        with col4:
            st.metric("Policies at Cap", f"{commercial_metrics['Policies_at_Cap_Count']} ({commercial_metrics['Policies_at_Cap_Pct']:.2f}%)")
            
        cols_to_export = [c for c in ["PolicyID", "Age", "CarAge", "DriverAge", "VehicleValue", "Density", "Region", "Risk_Band", "Exposure", "ClaimNb", "ClaimAmount", "pred_freq", "pred_sev", "adjusted_RAF", "Gross_Premium", "Final_Premium"] if c in df_clean.columns]
        csv_data = df_clean[cols_to_export].to_csv(index=False)
        st.download_button(
            label="📥 Download Final Priced Portfolio CSV",
            data=csv_data,
            file_name="FINAL_priced_portfolio.csv",
            mime="text/csv"
        )

# ─── TAB 4: MULTI-AGENT GOVERNANCE AUDIT ───────────────────────────────────────
with tab4:
    st.header("🤖 Step 4: Autonomous Multi-Agent AI Audit Gate")
    if not SEV_SUCCESS:
        st.warning("Please complete preceding calculation steps.")
    else:
        with st.spinner("Running multi-agent audit review..."):
            freq_gini_val = float(freq_summary_df.loc[chosen_freq_model, "Actuarial_Gini"]) if "Actuarial_Gini" in freq_summary_df.columns and chosen_freq_model in freq_summary_df.index else 0.16
            sev_ae_val = float(sev_summary_df.loc[chosen_sev_model, "ae_ratio"]) if "ae_ratio" in sev_summary_df.columns and chosen_sev_model in sev_summary_df.index else 1.0
            
            audit_report = simulate_agent_validation_audit(
                df_profile_meta=meta_json,
                anomaly_metrics=anomaly_metrics,
                frequency_metrics={"Actuarial_Gini": freq_gini_val},
                severity_metrics={"ae_ratio": sev_ae_val},
                credibility_metrics={"correction_factor": correction_factor},
                commercial_metrics=commercial_metrics
            )
            
        for agent_name, status_dict in audit_report.items():
            if agent_name == "Chief_Actuary_Governance_Auditor":
                continue
            status_color = "green" if status_dict["Status"] == "PASSED" else "orange" if status_dict["Status"] == "WARNING" else "red"
            with st.expander(f"🕵️ **{agent_name.replace('_', ' ')}** — Status: :{status_color}[{status_dict['Status']}]"):
                st.markdown(f"**Audit Findings:** {status_dict['Comment']}")
                
        st.markdown("---")
        chief_dict = audit_report["Chief_Actuary_Governance_Auditor"]
        decision = chief_dict["Final_Decision"]
        if decision == "APPROVED":
            cert_code = chief_dict.get("Certification_Code", "ASOP-41-VERIFIED")
            st.success(f"✅ **Governance Verdict: APPROVED FOR PRODUCTION**\n\nCertification Code: `{cert_code}`")
        elif decision == "CONDITIONAL APPROVAL":
            review_items = "\n- ".join(chief_dict.get("Audit_Warnings", []))
            st.warning(f"⚠️ **Governance Verdict: CONDITIONAL APPROVAL WITH WARNINGS**\n\nTotal Warnings: {chief_dict.get('Total_Warnings_Flagged', 0)}\n\nReview Items:\n- {review_items}")
        else:
            review_items = "\n- ".join(chief_dict.get("Audit_Warnings", []))
            st.error(f"❌ **Governance Verdict: REJECTED FOR MANUAL AUDIT**\n\nWarnings Flagged:\n- {review_items}")

with tab_assistant:
    st.header("💬 AI Actuarial Reference Assistant")
    st.markdown("""
    Ask the AI anything about your pricing engine, model selection, formulas, 
    or regulatory guidelines under **strict closed-world grounding and provenance tracking**.
    """)
    
    # Check if API key is configured
    active_api_key = st.session_state.get("gemini_api_key", "").strip()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    kb = load_rag_knowledge()
    
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citation"):
                with st.expander("📚 Actuarial Citation & Grounding"):
                    st.markdown(msg["citation"])
                    
    # Chat Input
    if user_query := st.chat_input("Ask an actuarial question... (e.g. 'How was exposure determined?', 'Why Gamma deviance?')"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        retrieved_context = search_rag_knowledge(user_query, kb)
        
        with st.chat_message("assistant"):
            # Check for API Key
            if not active_api_key:
                st.warning("⚠️ **Live AI Reasoning Offline**: No Gemini API key detected.")
                st.info("👉 Please paste your Gemini API key into the **'🔑 AI Agent Access'** box in the left sidebar to activate live agent reasoning and tool synthesis.")
                
                # Honest local fallback: Show retrieved reference excerpt directly without fake AI text
                st.markdown("### 📖 Retrieved Reference Manual Section:")
                st.markdown(retrieved_context)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ *API key not provided. Retrieved reference manual documentation shown below.*",
                    "citation": retrieved_context
                })
            else:
                with st.spinner("🤖 Consulting Actuarial Manuals & Auditing Pipeline State..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=active_api_key)
                        
                        # Build Grounding Facts from live pipeline state
                        current_facts = {
                            "total_records": meta_json.get("total_records", len(raw_df)),
                            "total_exposure_years": meta_json.get("total_exposure_years", 0.0),
                            "exposure_synthesized": meta_json.get("exposure_synthesized", False),
                            "assumptions_log": meta_json.get("assumptions_log", [
                                {
                                    "parameter": "Exposure",
                                    "provenance": "ACTUARIAL_ASSUMPTION" if meta_json.get("exposure_synthesized", False) else "DATASET_FACT",
                                    "value": 1.0 if meta_json.get("exposure_synthesized", False) else "Sourced from data",
                                    "justification": "Raw dataset lacked exposure column. Assumed 1.0 annual earned exposure per policy under ASOP 23 proxy ratemaking convention." if meta_json.get("exposure_synthesized", False) else "Found in raw data."
                                }
                            ]),
                            "selected_frequency_model": chosen_freq_model,
                            "selected_severity_model": chosen_sev_model,
                            "large_loss_loading": l_loading,
                            "profit_margin": p_margin,
                            "premium_floor": p_floor,
                            "premium_cap": p_cap,
                            "buhlmann_correction_factor": correction_factor
                        }
                        
                        system_prompt = f"""You are a senior, highly professional Actuarial AI Assistant embedded in a Motor Insurance Pricing Engine.
Your users are credentialed pricing actuaries and underwriters.

CRITICAL OPERATIONAL CONSTRAINTS:
1. Answer in a direct, concise, professional tone (3 to 5 sentences maximum or a tight bullet list).
2. MANDATORY PROVENANCE TAGGING:
   - When discussing data fields, explicitly state whether they are a [DATASET FACT] (found in original raw data) or an [ACTUARIAL ASSUMPTION] (synthesized/imputed by the pipeline).
   - If asked about Exposure, you MUST explicitly state that the raw dataset lacked an exposure column, so the pipeline assumed Exposure = 1.0 per policy under ASOP 23 ratemaking rules.
3. NEVER hallucinate metrics. Use the provided Pipeline State and Reference Manual below.
4. Do not dump entire paragraphs of reference text. Cite the relevant Section Title concisely.

=== CURRENT PIPELINE RUNTIME FACTS ===
{json.dumps(current_facts, indent=2)}

=== ACTUARIAL REFERENCE MANUAL PASSAGES ===
{retrieved_context}
"""
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        prompt = f"{system_prompt}\n\nActuary Question: {user_query}\n\nActuarial AI Response:"
                        response = model.generate_content(prompt)
                        answer_text = response.text
                        
                        st.markdown(answer_text)
                        with st.expander("📚 Actuarial Citation & Grounding"):
                            st.markdown(f"**Referenced Documentation Section:**\n\n{retrieved_context}")
                            
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "citation": retrieved_context
                        })
                    except Exception as e:
                        st.error(f"❌ Error communicating with Gemini API: {e}")
                        st.info("Showing reference passage directly:")
                        st.markdown(retrieved_context)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {e}",
                            "citation": retrieved_context
                        })
            st.rerun()
