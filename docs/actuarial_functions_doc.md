# Actuarial Reference Manual: Modular Tool Library

This reference manual documents the complete suite of **31 discrete actuarial tools** developed for the **Motor Insurance Pricing Engine**, organized across the 7 pipeline notebooks.

In strict compliance with architectural guidelines, notebooks contain focused modular tools (Notebook 02 contains 8 specialized tools under explicit permission, while all other notebooks strictly maintain <= 6 tools). This modular architecture eliminates monolithic black-box code, provides an auditable trail under **Actuarial Standards of Practice (ASOP 23, 41, and 56)**, and enables interactive, step-by-step user execution directly within notebook cells.

---

## Architectural Summary & Module Map

| Notebook / Lifecycle Stage | Dedicated Module File | Tools Count | Primary Purpose |
| :--- | :--- | :---: | :--- |
| **01. Data Profiling** | `src/tools/profiling_tools.py` | **3** | Schema identification, variable classification, distribution profiling |
| **02. Anomaly Detection** | `src/tools/anomaly_tools.py` | **8** | Logical sanity gates, tail analysis, P99 outlier flags, Isolation Forest, leverage, deviance residuals, influence check, 4-tier business review matrix |
| **03. Frequency Models** | `src/tools/frequency_tools.py` | **5** | Feature engineering, Poisson GLM, NB GLM, XGBoost, model evaluation & Gini lift |
| **04. Severity Models** | `src/tools/severity_tools.py` | **4** | Positive cohort extraction, Gamma GLM, Log-Normal, A/E model evaluation |
| **05. Credibility** | `src/tools/credibility_tools.py` | **3** | Pure premium synthesis, Bühlmann credibility calibration, revenue neutrality |
| **06. Final Premium** | `src/tools/premium_tools.py` | **3** | Commercial formula execution, decile A/E validation, artifact export |
| **07. Validation Audit** | `src/tools/validation_tools.py` | **2** | Dossier compilation, 5-agent governance sign-off report |
| **Unified Facade** | `src/pricing_functions.py` | **31** | Central unified re-export of all discrete tools |

---

## 1. Data Profiling & Exposure Synthesis (`profiling_tools.py`) - Notebook 01

### `tool_detect_schema`
* **What the Tool Does:** Automatically inspects the column headers of an incoming policy DataFrame and maps non-standard column names to standard actuarial target names for claim counts, claim amounts, and exposure durations.
* **Why It Is Used (Actuarial Context):** Core insurance transaction databases and policy administration systems use varying field naming conventions. This tool establishes consistent programmatic column references so downstream pipelines execute without manual column renaming.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Raw policy and claims portfolio.
  * `claim_nb_candidates` (*Optional[List[str]]*): Candidate column names for claim counts.
  * `claim_amount_candidates` (*Optional[List[str]]*): Candidate column names for claim loss amounts.
  * `exposure_candidates` (*Optional[List[str]]*): Candidate column names for policy exposure duration.
* **Underlying Process & Math:** Case-insensitive string normalization removing whitespace and underscores, followed by ordered priority matching against candidate token dictionaries.
* **What Output It Produces:** A dictionary mapping standardized keys (`claim_count_col`, `claim_amount_col`, `exposure_col`, and aliases) to the detected DataFrame column strings.
* **How to Interpret Results:** If a value is `None`, the respective actuarial dimension was absent from the dataset and must be synthesized or populated before modeling.

### `tool_synthesize_exposure`
* **What the Tool Does:** Validates the exposure duration column in the policy dataset. If the exposure column is absent, it creates a synthesized exposure column set to a standard default; if present, it replaces invalid non-positive durations with safe values.
* **Why It Is Used (Actuarial Context):** Under ASOP 23, loss rates must be modeled relative to earned risk units. Modeling raw counts without exposure would erroneously treat a policy active for 5 days identically to a policy active for a full 365 days.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Policy dataset.
  * `exposure_col` (*Optional[str]*): Column name representing policy duration.
  * `default_val` (*float*): Default exposure duration to assign if missing (default = 1.0 year).
* **Underlying Process & Math:**
  $$Exposure_i = \begin{cases} default\_val & \text{if column missing} \\ \max(Exposure_i, \epsilon) & \text{if } Exposure_i \le 0 \\ Exposure_i & \text{otherwise} \end{cases}$$
* **What Output It Produces:** A tuple `(df_modified, synthesized_flag)` indicating whether synthetic exposure was generated.
* **How to Interpret Results:** A `synthesized_flag = True` alerts the actuary that exposure was artificially generated, requiring disclosure in the actuarial memorandum.

### `tool_profile_distributions`
* **What the Tool Does:** Compiles fundamental portfolio-level distribution metrics, feature data types, missing value frequencies, portfolio claim rate, and zero-claim proportion.
* **Why It Is Used (Actuarial Context):** Establishes an empirical baseline prior to statistical modeling, identifying zero-inflation severity and confirming whether sample size satisfies statistical credibility requirements.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Standardized policy dataset.
  * `claim_count_col` (*str*): Target column name for claim counts.
  * `claim_amount_col` (*str*): Target column name for claim payouts.
  * `exposure_col` (*str*): Column name for policy exposure.
* **Underlying Process & Math:**
  $$\text{Portfolio Frequency} = \frac{\sum \text{ClaimCount}_i}{\sum \text{Exposure}_i}, \quad \text{Zero Ratio} = \frac{\sum \mathbb{I}(\text{ClaimCount}_i = 0)}{N}$$
* **What Output It Produces:** A dictionary containing policy counts, total earned exposure years, total claims, portfolio frequency, aggregate claim payout, zero-claim percentage, list of numeric features, list of categorical features, and missing count mappings.
* **How to Interpret Results:** A high zero-claim ratio (e.g., >90%) justifies using Poisson/Negative Binomial distribution assumptions rather than Ordinary Least Squares.

---

## 2. Anomaly Detection & Influence Auditing (`anomaly_tools.py`) - Notebook 02

### `tool_check_data_quality`
* **What the Tool Does:** Enforces physical and logical consistency rules across claims and policy records, flagging negative rating values, negative claim counts/amounts, or nonzero payouts with zero recorded claims.
* **Why It Is Used (Actuarial Context):** Prevents impossible corrupted records from contaminating statistical regressions, fulfilling data scrubbing duties mandated by ASOP 23.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Policy dataset.
  * `num_cols` (*List[str]*): Numeric feature column names to evaluate.
  * `claim_count_col` (*Optional[str]*): Column name for claim count.
  * `claim_amount_col` (*Optional[str]*): Column name for claim loss amount.
* **Underlying Process & Math:**
  $$\text{Issues} = \{ c : \sum \mathbb{I}(X_{ic} < 0) > 0 \} \cup \{ \text{Unlinked} : \sum \mathbb{I}(\text{ClaimNb}_i = 0 \land \text{ClaimAmount}_i > 0) > 0 \}$$
* **What Output It Produces:** A dictionary recording each data quality issue type and its affected policyholder count.
* **How to Interpret Results:** An empty dictionary signifies a 100% clean dataset. Any detected issues must be quarantined before modeling.

### `tool_analyze_tail_advanced`
* **What the Tool Does:** Analyzes distribution tails by separating zero-claim mass from the active positive claims cohort (conditional severity), computing percentiles from P50 to P99.9.
* **Why It Is Used (Actuarial Context):** General insurance risks exhibit heavy-tailed distributions and extreme zero-inflation. Separating frequency from severity prevents tail percentiles from being collapsed to zero.
* **What Inputs It Requires:**
  * `df_subset` (*pd.DataFrame*): Policy cohort (e.g., Paris density cluster vs. rest of portfolio).
  * `claim_amount_col` (*str*): Column name for claim payout amount.
  * `claim_count_col` (*str*): Column name for claim count.
  * `name` (*str*): Descriptive name of the analyzed cohort.
* **Underlying Process & Math:** Subsets active claims and computes quantiles $Q(p) = \inf \{ x : F(x) \ge p \}$ across percentiles `[P50, P75, P90, P95, P99, P99.2, P99.4, P99.6, P99.8, P99.9]`.
* **What Output It Produces:** A dictionary containing zero-claim rate, positive claim count, conditional severity tail table, and overall feature tail table.
* **How to Interpret Results:** Highlights extreme loss concentrations and sharp jumps in exposure variables that indicate distinct regional risk profiles.

### `tool_flag_tail_outliers`
* **What the Tool Does:** Flags observations exceeding specified univariate quantile thresholds (default P99) across rating features and logs the specific reason.
* **Why It Is Used (Actuarial Context):** Identifies extreme single-feature policies (e.g., vehicles older than P99) that exert high univariate leverage on model fits.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Dataset containing continuous features.
  * `tail_cols` (*List[str]*): Continuous feature columns to check.
  * `quantile` (*float*): Quantile threshold cutoff (default = 0.99).
* **Underlying Process & Math:** Computes cutoff vector $q = \text{quantile}(df[tail\_cols], p)$ and sets $\text{tail\_flag}_i = \bigvee_j (X_{ij} > q_j)$.
* **What Output It Produces:** A tuple `(df_flagged, p99_threshold_series, n_flagged_count)`.
* **How to Interpret Results:** Policies flagged with `tail_flag = True` are retained for multi-variable evaluation in the review matrix.

### `tool_run_isolation_forest`
* **What the Tool Does:** Fits an unsupervised ensemble of isolation trees across numeric and categorical rating variables to score multivariate feature anomalies.
* **Why It Is Used (Actuarial Context):** Univariate checks fail to detect observations that are normal in single dimensions but highly anomalous in combination (e.g., an inexperienced driver operating an ultra-high-power sports vehicle in an extreme-density urban postal code).
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Dataset containing predictors.
  * `num_cols` (*List[str]*): List of numeric predictor column names.
  * `cat_cols` (*List[str]*): List of categorical predictor column names.
  * `contamination` (*float*): Expected proportion of outliers in the portfolio (default = 0.02).
  * `n_estimators` (*int*): Number of isolation trees to construct (default = 200).
  * `random_state` (*int*): Random seed for reproducibility.
* **Underlying Process & Math:** Median-imputes numerics, one-hot encodes categoricals, and measures average tree path length $h(x)$. Anomaly score $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$. Negates decision function so higher score indicates higher anomaly.
* **What Output It Produces:** A tuple `(df_scored, preprocessor_pipeline, fitted_isolation_forest, feature_matrix_shape)`.
* **How to Interpret Results:** `IF_Label == -1` designates multi-dimensional anomaly candidates for review.

### `tool_calculate_leverage`
* **What the Tool Does:** Computes Hat Matrix diagonal leverage values ($H_{ii}$) in the predictor space and flags observations exceeding the statistical threshold $2p/n$.
* **Why It Is Used (Actuarial Context):** Identifies observations positioned far from the predictor centroid. In linear and generalized linear modeling, high-leverage points can pull regression planes away from the true population parameters.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Dataset containing predictors.
  * `num_cols` (*List[str]*): Numeric feature columns.
  * `leverage_multiplier` (*float*): Multiplier $k$ over average leverage (default = 2.0).
* **Underlying Process & Math:**
  $$X_{\text{scaled}} = \text{StandardScaler}(X_{\text{num}}), \quad X_{\text{sm}} = [\mathbf{1} \quad X_{\text{scaled}}]$$
  $$H = X_{\text{sm}} (X_{\text{sm}}^T X_{\text{sm}})^{-1} X_{\text{sm}}^T, \quad H_{ii} = \text{diag}(H)$$
  $$\text{Leverage Threshold} = k \times \frac{p}{n}$$
* **What Output It Produces:** A tuple `(df_with_leverage, threshold, p_parameters, n_observations)`.
* **How to Interpret Results:** Policies with `high_leverage == True` have unusual feature configurations that must be checked for modeling stability.

### `tool_calculate_deviance_residuals`
* **What the Tool Does:** Fits a baseline Poisson GLM with log-exposure offsets and calculates signed Poisson deviance residuals ($r_i^D$).
* **Why It Is Used (Actuarial Context):** Evaluates model prediction error on count data. Standard raw residuals ($y - \hat{\mu}$) are heteroscedastic for Poisson counts; deviance residuals normalize error variance across predicted risk levels.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Policy dataset.
  * `num_cols` (*List[str]*): Numeric predictors.
  * `cat_cols` (*List[str]*): Categorical predictors.
  * `claim_count_col` (*str*): Target claim count column.
  * `exposure_col` (*Optional[str]*): Policy exposure duration column.
* **Underlying Process & Math:**
  $$\ln(\hat{\mu}_i) = \ln(\text{Exposure}_i) + \beta_0 + \sum \beta_j X_{ij}$$
  $$r_i^D = \text{sign}(y_i - \hat{\mu}_i) \sqrt{2 \left[ y_i \ln\left(\frac{y_i}{\hat{\mu}_i}\right) - (y_i - \hat{\mu}_i) \right]}$$
* **What Output It Produces:** A tuple `(df_with_residuals, fitted_glm_model)`.
* **How to Interpret Results:** Residuals clustering near zero indicate good fit; points with $|r_i^D| > 3.0$ represent severe under- or over-predictions.

### `tool_identify_influential_points`
* **What the Tool Does:** Cross-references high leverage with large deviance residuals ($|r_i^D| > 3.0$) to identify statistically influential policies (Cook's distance proxy).
* **Why It Is Used (Actuarial Context):** An observation with high leverage but low residual does not distort the model; an observation with high residual but low leverage adds mere noise. Only observations possessing BOTH high leverage and large error pull model coefficients away from true risk relativities.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Dataset with `Leverage`, `high_leverage`, and `Residual_Dev`.
  * `resid_threshold` (*float*): Absolute cutoff for deviance residuals (default = 3.0).
* **Underlying Process & Math:**
  $$\text{High Residual}_i = (|r_i^D| > \text{resid\_threshold})$$
  $$\text{Influential}_i = \text{high\_leverage}_i \land \text{High Residual}_i$$
* **What Output It Produces:** A tuple `(df_updated, n_influential_count)`.
* **How to Interpret Results:** The upper-right quadrant of the influence plot contains the small number of influential policies that threaten GLM parameter integrity.

### `tool_classify_business_review`
* **What the Tool Does:** Synthesizes multi-dimensional anomaly flags into the 4-tier Actuarial Business Review Matrix: `🔴 Likely Error`, `🟠 Suspicious`, `🟡 Rare / Unusual`, and `🟢 Normal`.
* **Why It Is Used (Actuarial Context):** Operationalizes statistical diagnostics into actionable underwriting and governance decisions, ensuring legitimate high risks are preserved while true data corruptions are purged under ASOP 23.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): DataFrame containing `IF_Label`, `tail_flag`, `influential`, and `high_leverage`.
* **Underlying Process & Math:** Hierarchical priority decision matrix:
  1. If `IF_Label == -1` AND `tail_flag` AND `influential` -> `🔴 Likely Error`
  2. Else if `IF_Label == -1` AND (`influential` OR `tail_flag`) -> `🟠 Suspicious`
  3. Else if any single flag is True (`IF only`, `Tail only`, `high_leverage only`) -> `🟡 Rare / Unusual`
  4. Otherwise -> `🟢 Normal`
* **What Output It Produces:** The DataFrame augmented with `Review_Class`.
* **How to Interpret Results:** `🔴 Likely Error` records are removed by default before rate making; `🟠 Suspicious` records are presented for underwriting review; `🟡 Rare` policies are preserved in training data; `🟢 Normal` policies form the modeling baseline.

---

## 3. Frequency Modeling & Selection (`frequency_tools.py`) - Notebook 03

### `tool_prepare_frequency_features`
* **What the Tool Does:** Preprocesses numeric and categorical predictors into design matrices suitable for Generalized Linear Models and tree algorithms.
* **Why It Is Used (Actuarial Context):** Ensures identical feature transformations and baseline reference dummy encodings across both classical GLMs and machine learning benchmarks.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Policy dataset.
  * `num_cols` (*List[str]*): Numeric feature column names.
  * `cat_cols` (*List[str]*): Categorical feature column names.
* **Underlying Process & Math:** Imputes missing numeric values with training medians, applies `pd.get_dummies(..., drop_first=True)` to prevent collinearity, and prepends an intercept constant column for GLM estimation.
* **What Output It Produces:** A tuple `(df_encoded, X_design_matrix)` containing the encoded DataFrame and the numerical GLM matrix.
* **How to Interpret Results:** Ready for direct ingestion into statsmodels GLM or gradient boosted trees.

### `tool_fit_frequency_glms`
* **What the Tool Does:** Fits Poisson and Negative Binomial (NB2) Generalized Linear Models using Iteratively Reweighted Least Squares (IRLS) with log-exposure offsets.
* **Why It Is Used (Actuarial Context):** The industry standard for insurance frequency modeling (CAS Exam 8). Multiplicative log-link structures yield clear, regulator-friendly base rates and rating relativities.
* **What Inputs It Requires:**
  * `X_train_sm` (*np.ndarray*): Training feature matrix with intercept.
  * `y_train` (*np.ndarray*): Training claim counts.
  * `off_train` (*np.ndarray*): Training offset vector, equal to $\ln(\text{Exposure})$.
  * `X_val_sm` (*np.ndarray*): Validation feature matrix with intercept.
  * `off_val` (*np.ndarray*): Validation offset vector.
* **Underlying Process & Math:**
  $$\ln(E[Y_i]) = \ln(\text{Exposure}_i) + \beta_0 + \sum_{j=1}^p \beta_j X_{ij}$$
  $$\text{Poisson: } \text{Var}(Y_i) = \mu_i, \quad \text{Negative Binomial: } \text{Var}(Y_i) = \mu_i + \alpha \mu_i^2$$
* **What Output It Produces:** A dictionary mapping model names to tuples of `(fitted_model_object, validation_predictions, AIC, Deviance)`.
* **How to Interpret Results:** Lower AIC and deviance indicate superior fit; if the Negative Binomial dispersion parameter is significant, overdispersion is present.

### `tool_fit_frequency_xgboost`
* **What the Tool Does:** Fits an advanced gradient-boosted decision tree under a Poisson count objective with base margins set to log-exposure.
* **Why It Is Used (Actuarial Context):** Serves as a non-linear machine learning benchmark to test whether GLMs miss significant non-linear risk patterns or cross-variable interactions.
* **What Inputs It Requires:**
  * `X_train` (*pd.DataFrame*): Encoded training predictors.
  * `y_train` (*np.ndarray*): Training claim counts.
  * `exp_train` (*np.ndarray*): Training exposure durations.
  * `X_val` (*pd.DataFrame*): Encoded validation predictors.
  * `y_val` (*np.ndarray*): Validation claim counts.
  * `exp_val` (*np.ndarray*): Validation exposure durations.
* **Underlying Process & Math:** Minimizes Poisson deviance objective with base margin $\ln(\text{Exposure})$ using validation early stopping:
  $$\mathcal{L}(y, \hat{y}) = 2 \sum \left( \hat{y}_i - y_i \ln(\hat{y}_i) \right)$$
* **What Output It Produces:** A tuple `(fitted_xgb_model, validation_predictions)`.
* **How to Interpret Results:** If XGBoost outperforms GLMs substantially on validation Gini, underwriters should consider adding interaction terms into the GLM.

### `tool_calculate_actuarial_gini`
* **What the Tool Does:** Computes the Actuarial Gini coefficient based on the Area Under the Lorenz Curve (AUC).
* **Why It Is Used (Actuarial Context):** Traditional regression error metrics (like $R^2$ or RMSE) are uninformative for zero-inflated count data (>90% zeroes). Actuaries evaluate frequency models primarily on their ability to rank-order policyholders from lowest to highest risk.
* **What Inputs It Requires:**
  * `y_true` (*np.ndarray*): Observed validation target counts.
  * `y_pred` (*np.ndarray*): Out-of-sample predicted frequency rates.
* **Underlying Process & Math:** Sorts observations in descending order of predicted risk. Computes cumulative loss share against cumulative population share, and calculates:
  $$\text{Gini} = 2 \times (\text{AUC} - 0.5)$$
* **What Output It Produces:** A positive float representing the Actuarial Gini coefficient.
* **How to Interpret Results:** Higher Gini indicates stronger risk-differentiation and lift. A model with higher Gini is more effective at separating safe drivers from hazardous drivers.

### `tool_compare_frequency_models`
* **What the Tool Does:** Compiles out-of-sample goodness-of-fit, error, and lift metrics across all candidate frequency models into a comparative table.
* **Why It Is Used (Actuarial Context):** Enables the actuary to review competitive performance metrics (AIC, Deviance, MAE, RMSE, Gini) and choose the winning frequency model for production rate deployment.
* **What Inputs It Requires:**
  * `models_dict` (*Dict*): Dictionary of candidate model outputs and predictions.
  * `y_val` (*np.ndarray*): Observed validation counts.
* **Underlying Process & Math:** Computes MAE, RMSE, and Gini on validation data for each model and tabulates comparative performance.
* **What Output It Produces:** A Pandas DataFrame indexed by model name displaying all evaluation metrics.
* **How to Interpret Results:** Guides the actuary to select the model achieving optimal balance between statistical accuracy and business explainability.

---

## 4. Severity Modeling & Diagnostics (`severity_tools.py`) - Notebook 04

### `tool_filter_positive_claims`
* **What the Tool Does:** Extracts the positive claims cohort ($Y_i > 0$) from the dataset and computes regression observation weights based on claim count.
* **Why It Is Used (Actuarial Context):** Claim severity models the conditional distribution of loss amount given that a claim has occurred ($f(Y | Y > 0)$). Policies with multiple claims must be weighted proportionally to their claim count.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Cleaned policy dataset.
  * `claim_amount_col` (*str*): Column name for claim payout amount.
  * `claim_count_col` (*str*): Column name for claim count.
* **Underlying Process & Math:**
  $$\text{Subset: } \{ i : \text{ClaimAmount}_i > 0 \}, \quad w_i = \text{ClaimCount}_i$$
* **What Output It Produces:** A tuple `(df_positive, observation_weights)`.
* **How to Interpret Results:** The isolated cohort contains only nonzero loss events for conditional severity estimation.

### `tool_fit_severity_models`
* **What the Tool Does:** Fits three candidate severity distributions: Gamma GLM (log link), Log-Normal regression (with smearing mean correction), and Inverse Gaussian GLM.
* **Why It Is Used (Actuarial Context):** Claim sizes are strictly positive, continuously distributed, and right-skewed. Testing multiple error structures identifies the optimal model for loss variance and tail skewness.
* **What Inputs It Requires:**
  * `X_sm` (*np.ndarray*): Feature design matrix for positive claims cohort.
  * `y_sev` (*np.ndarray*): Observed positive claim amounts.
  * `weights` (*np.ndarray*): Claim count observation weights.
* **Underlying Process & Math:**
  * **Gamma GLM:** $\text{Var}(Y) = \phi \mu^2$, log link $\ln(\mu) = X \beta$.
  * **Log-Normal:** Fits WLS on $\ln(Y)$; predicts using smearing retransformation correction:
    $$\hat{Y} = \exp\left( X \hat{\beta} + \frac{1}{2} s^2 \right)$$
  * **Inverse Gaussian:** $\text{Var}(Y) = \phi \mu^3$, accommodating heavier right tails.
* **What Output It Produces:** A dictionary mapping model names to tuples of `(model_object, predictions, AIC, Deviance)`.
* **How to Interpret Results:** Compares model fits across variance assumptions to select the most realistic loss distribution.

### `tool_compare_severity_models`
* **What the Tool Does:** Compiles statistical fit, dollar error, and portfolio-wide financial adequacy (Actual-to-Expected payout ratio) for severity candidates.
* **Why It Is Used (Actuarial Context):** In addition to minimizing prediction error, a production severity model must be financially balanced—total predicted claims must equal total historical claims ($A/E \approx 1.00$).
* **What Inputs It Requires:**
  * `models_dict` (*Dict*): Fitted severity models dictionary.
  * `y_true` (*np.ndarray*): Observed positive claim amounts.
* **Underlying Process & Math:**
  $$\text{Overall A/E} = \frac{\sum_{i=1}^m y_i}{\sum_{i=1}^m \hat{y}_i}$$
* **What Output It Produces:** A summary Pandas DataFrame showing AIC, Deviance, MAE, RMSE, and Overall A/E ratio.
* **How to Interpret Results:** An $A/E > 1.0$ indicates systemic under-prediction; an $A/E < 1.0$ indicates systemic over-prediction. Actuaries favor models where $A/E \in [0.98, 1.02]$.

### `tool_calculate_severity_residuals`
* **What the Tool Does:** Computes deviance residuals from the fitted severity model to evaluate heteroscedasticity and distributional assumptions.
* **Why It Is Used (Actuarial Context):** Validates whether residual variance is constant across predicted claim sizes or whether systematic skewness remains in the model errors.
* **What Inputs It Requires:**
  * `model` (*object*): Fitted GLM or regression model object.
  * `X_sm` (*np.ndarray*): Design matrix.
  * `y_true` (*np.ndarray*): Observed positive claim amounts.
* **Underlying Process & Math:** Extracts standardized deviance residuals $r_i^D$ directly from the fitted GLM results.
* **What Output It Produces:** A 1D NumPy array of residuals.
* **How to Interpret Results:** Normal Q-Q plots and scatter plots of residuals against fitted values should show no pronounced curvature or funnel shapes.

---

## 5. Credibility & Calibration (`credibility_tools.py`) - Notebook 05

### `tool_calculate_pure_premium`
* **What the Tool Does:** Multiplies modeled annual claim frequency by modeled conditional claim severity to generate unadjusted Pure Premium (Risk Premium).
* **Why It Is Used (Actuarial Context):** The fundamental actuarial equation of risk cost. Represents the break-even technical price required to fund expected loss payouts per exposure unit.
* **What Inputs It Requires:**
  * `freq_pred` (*np.ndarray*): Modeled claim frequency per exposure unit.
  * `sev_pred` (*np.ndarray*): Modeled claim severity per claim.
* **Underlying Process & Math:**
  $$\text{Pure Premium}_i = \hat{\lambda}_i \times \hat{\mu}_i$$
* **What Output It Produces:** A 1D NumPy array of policy-level pure premiums.
* **How to Interpret Results:** The baseline technical price prior to segment credibility adjustments, capital loadings, or commercial limits.

### `tool_segment_risk_bands`
* **What the Tool Does:** Quantile-segments policyholders into homogeneous risk bands based on continuous pure premium rankings.
* **Why It Is Used (Actuarial Context):** Aggregates continuous statistical scores into underwriting cohorts to evaluate actual vs. expected performance and apply credibility smoothing.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Policy dataset.
  * `pure_premium` (*np.ndarray*): Modeled pure premium array.
  * `n_bands` (*int*): Number of quantile bands (default = 5).
* **Underlying Process & Math:** Quantile cut partitioning based on rank ordering:
  $$\text{Risk Band}_i = \text{qcut}(\text{Rank}(\text{PurePremium}_i), q)$$
* **What Output It Produces:** The DataFrame augmented with `Pure_Premium` and `Risk_Band` columns.
* **How to Interpret Results:** Band 1 represents the lowest-risk cohort; Band $q$ represents the highest-risk cohort.

### `tool_calibrate_buhlmann_credibility`
* **What the Tool Does:** Calculates Bühlmann empirical Bayes credibility weights ($Z$) and Risk Adjustment Factors (RAF) for each risk segment, enforcing portfolio-level revenue neutrality.
* **Why It Is Used (Actuarial Context):** Implements classical credibility theory to resolve the bias-variance tradeoff. Thin segments with limited exposure are shrunk toward the portfolio average ($Z \to 0$), while large segments rely on their own empirical experience ($Z \to 1$).
* **What Inputs It Requires:**
  * `df_segmented` (*pd.DataFrame*): DataFrame containing risk bands and loss data.
  * `band_col` (*str*): Risk band grouping column.
  * `exposure_col` (*str*): Policy exposure column.
  * `actual_loss_col` (*str*): Observed claims column.
  * `exp_loss_col` (*str*): Expected pure premium losses column.
  * `K` (*float*): Bühlmann credibility parameter, representing expected process variance divided by variance of hypothetical means (default = 1000.0).
* **Underlying Process & Math:**
  $$Z_g = \frac{n_g}{n_g + K}$$
  $$\text{Unadjusted RAF}_g = Z_g \left( \frac{\text{Observed Loss}_g}{\text{Expected Loss}_g} \right) + (1 - Z_g) \times 1.0$$
  $$\text{Revenue Correction Factor } c = \frac{\sum \text{Observed Loss}}{\sum (\text{Expected Loss}_g \times \text{Unadjusted RAF}_g)}$$
  $$\text{Adjusted RAF}_g = \text{Unadjusted RAF}_g \times c$$
* **What Output It Produces:** A tuple `(credibility_summary_df, revenue_correction_factor)`.
* **How to Interpret Results:** The adjusted RAF ensures that credibility adjustments neither inflate nor deflate the overall portfolio premium volume.

### `tool_enforce_revenue_neutrality`
* **What the Tool Does:** Maps the calibrated, balanced Risk Adjustment Factors back to individual policyholder records.
* **Why It Is Used (Actuarial Context):** Operationalizes the group-level credibility calibration into policy-level commercial rating adjustments.
* **What Inputs It Requires:**
  * `df_segmented` (*pd.DataFrame*): Segmented policy dataset.
  * `band_col` (*str*): Column identifying the policy risk band.
  * `band_cred_df` (*pd.DataFrame*): Credibility table containing `adj_RAF`.
* **Underlying Process & Math:** Relational mapping linking each policy's risk band to the corresponding adjusted RAF factor.
* **What Output It Produces:** A Pandas Series containing the policy-level Risk Adjustment Factor.
* **How to Interpret Results:** A factor of 1.00 represents neutral pricing; >1.00 applies a credibility surcharge; <1.00 applies a credibility credit.

---

## 6. Final Commercial Premium (`premium_tools.py`) - Notebook 06

### `tool_calculate_commercial_premium`
* **What the Tool Does:** Executes the Actuarial Commercial Pricing Equation, combining pure premium, catastrophic loadings, credibility adjustments, profit margins, and regulatory boundary limits.
* **Why It Is Used (Actuarial Context):** Transforms theoretical loss costs into marketable, solvent commercial premiums that satisfy underwriter targets and competitive boundaries.
* **What Inputs It Requires:**
  * `pure_premium` (*np.ndarray*): Base modeled loss cost.
  * `large_loss_loading` (*float*): Catastrophic / large-loss loading multiplier $L$ (default = 1.10).
  * `risk_adj_factor` (*Optional[np.ndarray]*): Policy credibility factor $\text{RAF}$.
  * `profit_margin` (*float*): Underwriting profit and expense margin multiplier $M$ (default = 1.05).
  * `floor` (*float*): Minimum rate limit to ensure solvent transaction costs (default = 50.00).
  * `cap` (*float*): Maximum rate limit to prevent rate shock and extreme prices (default = 5000.00).
* **Underlying Process & Math:**
  $$\text{Gross Premium}_i = \text{Pure Premium}_i \times L \times \text{RAF}_i \times M$$
  $$\text{Final Commercial Premium}_i = \min(\max(\text{Gross Premium}_i, \text{floor}), \text{cap})$$
* **What Output It Produces:** A tuple `(final_premium, gross_premium)`.
* **How to Interpret Results:** `gross_premium` reflects the unconstrained commercial rate; `final_premium` is the billed customer price.

### `tool_compute_premium_diagnostics`
* **What the Tool Does:** Audits portfolio rate distribution, aggregate earned revenue, mean/median rate levels, and the proportion of policies constrained by commercial bounds.
* **Why It Is Used (Actuarial Context):** Protects against rate capping distortion. Excessive policies constrained at the cap indicate adverse selection risk; excessive policies at the floor suggest uncompetitive minimum rates.
* **What Inputs It Requires:**
  * `gross_premium` (*np.ndarray*): Unconstrained commercial premiums.
  * `final_premium` (*np.ndarray*): Bounded customer premiums.
  * `floor` (*float*): Commercial rate floor.
  * `cap` (*float*): Commercial rate cap.
* **Underlying Process & Math:**
  $$\text{Pct at Floor} = \frac{\sum \mathbb{I}(\text{Gross}_i < \text{floor})}{N} \times 100\%, \quad \text{Pct at Cap} = \frac{\sum \mathbb{I}(\text{Gross}_i > \text{cap})}{N} \times 100\%$$
* **What Output It Produces:** A dictionary containing policy counts, total portfolio revenue, mean premium, median premium, counts at boundaries, and boundary percentages.
* **How to Interpret Results:** Clipping proportions should remain low (typically <1%) so that individual risk differentiation is preserved.

### `tool_compute_decile_ae_chart`
* **What the Tool Does:** Ranks the portfolio into premium deciles and computes Actual-to-Expected (A/E) loss ratios across each pricing tier.
* **Why It Is Used (Actuarial Context):** Essential pre-filing actuarial diagnostic. Verifies that the premium structure is fair across all risk strata and that low-risk drivers are not cross-subsidizing high-risk drivers.
* **What Inputs It Requires:**
  * `df` (*pd.DataFrame*): Evaluated dataset.
  * `actual_loss_col` (*str*): Column containing observed claims.
  * `premium_col` (*str*): Column containing billed commercial premium.
  * `n_deciles` (*int*): Number of decile groupings (default = 10).
* **Underlying Process & Math:** Groups by premium decile and computes:
  $$\text{Decile A/E} = \frac{\sum_{i \in \text{Decile}} \text{Actual Loss}_i}{\sum_{i \in \text{Decile}} \text{Commercial Premium}_i}$$
* **What Output It Produces:** A Pandas DataFrame showing decile-by-decile actual losses, collected premiums, policy counts, and A/E loss ratios.
* **How to Interpret Results:** A flat decile A/E curve close to the target loss ratio proves consistent rating equity across all risk segments.

### `tool_export_pricing_portfolio`
* **What the Tool Does:** Exports the finalized pricing deliverables in multi-format outputs: formatted Excel workbook for underwriters, Apache Parquet table for high-speed cloud ingestion, and JSON parameter registry for regulatory compliance.
* **Why It Is Used (Actuarial Context):** Bridges technical actuarial modeling with underwriting operations and regulatory filing requirements.
* **What Inputs It Requires:**
  * `df_priced` (*pd.DataFrame*): Final DataFrame containing policy data, pure premium, rating factors, and final commercial premium.
  * `registry_data` (*dict*): Metadata dictionary recording all commercial parameters, dates, and author sign-offs.
  * `excel_path` (*str*): File path for `.xlsx` export.
  * `json_path` (*str*): File path for `.json` registry export.
  * `parquet_path` (*str*): File path for `.parquet` binary export.
* **Underlying Process & Math:** Writes serialized parquet file, Excel sheet, and formatted JSON registry to disk.
* **What Output It Produces:** Serialized disk artifacts ready for operational deployment.
* **How to Interpret Results:** Provides permanent, auditable governance artifacts for rate filings.

---

## 7. Multi-Agent Validation Audit (`validation_tools.py`) - Notebook 07

### `tool_build_agent_dossiers`
* **What the Tool Does:** Compiles structured parameter dossiers summarizing pipeline results across data profiling, frequency modeling, severity modeling, credibility calibration, and commercial rate limits.
* **Why It Is Used (Actuarial Context):** Packages quantitative results into specialized audit prompts tailored for each of the 5 AI peer-review agents.
* **What Inputs It Requires:**
  * `meta` (*dict*): Profiling metadata.
  * `freq_results` (*dict*): Frequency model evaluation metrics and selection.
  * `sev_results` (*dict*): Severity model evaluation metrics and selection.
  * `cred_results` (*dict*): Credibility calibration metrics.
  * `pricing_registry` (*dict*): Commercial formula parameters.
* **Underlying Process & Math:** Assembles domain-specific structured strings for:
  1. Data Profiling Agent
  2. Frequency Modeling Agent
  3. Severity Modeling Agent
  4. Credibility & Underwriting Agent
  5. Chief Actuary Auditor
* **What Output It Produces:** A dictionary mapping agent roles to formatted audit dossiers.
* **How to Interpret Results:** Used directly as prompt payloads for multi-agent LLM peer review or deterministic audit simulations.

### `tool_run_agentic_audit_simulation`
* **What the Tool Does:** Executes deterministic heuristic actuarial validation rules that simulate the 5 specialized AI review agents, verifying compliance with ASOP standards.
* **Why It Is Used (Actuarial Context):** Provides an automated CI/CD actuarial safety gate. Verifies that all statistical and commercial requirements are met before rate filing.
* **What Inputs It Requires:**
  * `meta` (*dict*): Profiling metadata.
  * `freq_results` (*dict*): Frequency modeling results.
  * `sev_results` (*dict*): Severity modeling results.
  * `cred_results` (*dict*): Credibility results.
  * `pricing_registry` (*dict*): Commercial parameters.
* **Underlying Process & Math:** Evaluates 5 actuarial gates:
  * Gate 1 (Profiling): Portfolio volume and exposure sufficiency ($N > 1,000$).
  * Gate 2 (Frequency): Model selection validity and Gini positive rank-ordering.
  * Gate 3 (Severity): Conditional positive cohort isolation and financial balance.
  * Gate 4 (Credibility): Bühlmann calibration and revenue neutrality.
  * Gate 5 (Chief Actuary): Sign-off on commercial loadings and boundary limits.
* **What Output It Produces:** A structured dictionary containing validation status (`PASSED`/`WARNING`/`APPROVED`) and qualitative findings for each agent.
* **How to Interpret Results:** An all-passed status clears the portfolio for executive review and production deployment.

### `tool_save_validation_report`
* **What the Tool Does:** Serializes the complete multi-agent validation audit and sign-offs to a permanent JSON governance artifact.
* **Why It Is Used (Actuarial Context):** Satisfies Actuarial Standard of Practice (ASOP) 41 (Actuarial Communications), maintaining an auditable digital record of automated review findings.
* **What Inputs It Requires:**
  * `validation_report` (*dict*): Consolidate validation findings from `tool_run_agentic_audit_simulation`.
  * `output_path` (*str*): File path destination for the JSON report.
* **Underlying Process & Math:** Formatted JSON serialization with two-space indentation.
* **What Output It Produces:** A persistent JSON audit report file.
* **How to Interpret Results:** Archived as regulatory compliance documentation supporting commercial rate submissions.\n