# tests/test_all_tools.py
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

import src.pricing_functions as pf


class TestActuarialTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 200
        cls.df = pd.DataFrame({
            'ClaimNb': np.random.poisson(0.1, n),
            'ClaimAmount': np.where(np.random.rand(n) < 0.1, np.random.exponential(1500, n), 0.0),
            'Exposure': np.random.uniform(0.1, 1.0, n),
            'DrivAge': np.random.randint(18, 75, n),
            'VehAge': np.random.randint(0, 20, n),
            'BonusMalus': np.random.randint(50, 150, n),
            'Density': np.random.uniform(10, 5000, n),
            'VehPower': np.random.randint(4, 12, n),
            'Brand': np.random.choice(['Fiat', 'Renault', 'Mercedes'], n),
            'Gas': np.random.choice(['Regular', 'Diesel'], n),
            'Region': np.random.choice(['R11', 'R24', 'R82'], n),
            'Area': np.random.choice(['A', 'B', 'C', 'D'], n)
        })
        cls.num_cols = ['DrivAge', 'VehAge', 'BonusMalus', 'Density', 'VehPower']
        cls.cat_cols = ['Brand', 'Gas', 'Region']

    def test_01_profiling_tools(self):
        # 1.1 tool_detect_schema
        schema = pf.tool_detect_schema(self.df)
        self.assertEqual(schema['claim_nb'], 'ClaimNb')
        self.assertEqual(schema['claim_amount'], 'ClaimAmount')
        self.assertEqual(schema['exposure'], 'Exposure')

        # 1.2 tool_synthesize_exposure
        df_no_exp = self.df.drop(columns=['Exposure'])
        df_syn, syn_flag = pf.tool_synthesize_exposure(df_no_exp)
        self.assertTrue('Exposure' in df_syn.columns)
        self.assertTrue(syn_flag)

        # 1.3 tool_profile_distributions
        metrics = pf.tool_profile_distributions(self.df, 'ClaimNb', 'ClaimAmount', 'Exposure')
        self.assertEqual(metrics['n_policies'], len(self.df))
        self.assertGreater(metrics['zero_claim_pct'], 0.70)

    def test_02_anomaly_tools(self):
        # 2.1 tool_check_data_quality
        dq_flags = pf.tool_check_data_quality(self.df, 'ClaimNb', 'ClaimAmount')
        self.assertEqual(len(dq_flags), len(self.df))

        # 2.2 tool_calculate_tail_percentiles
        tail_df = pf.tool_calculate_tail_percentiles(self.df, ['ClaimAmount', 'Density'])
        self.assertIn('ClaimAmount', tail_df.columns)
        self.assertIn('P99', tail_df.index)

        # 2.3 tool_run_isolation_forest
        df_iso, preprocessor, iso, proc_shape = pf.tool_run_isolation_forest(
            self.df, self.num_cols, self.cat_cols, contamination=0.02
        )
        iso_preds = df_iso['IF_Label']
        self.assertEqual(len(iso_preds), len(self.df))

        # 2.4 tool_calculate_leverage
        df_lev, lev_thresh, p, n = pf.tool_calculate_leverage(self.df, self.num_cols)
        self.assertIn('Leverage', df_lev.columns)
        self.assertIn('high_leverage', df_lev.columns)

        # 2.5 tool_calculate_deviance_residuals
        df_res, glm_res = pf.tool_calculate_deviance_residuals(
            self.df, self.num_cols, self.cat_cols, 'ClaimNb', 'Exposure'
        )
        self.assertIn('Residual_Dev', df_res.columns)

        # 2.6 tool_identify_influential_points
        df_inf, n_inf = pf.tool_identify_influential_points(df_res)
        self.assertIn('influential', df_inf.columns)

        # 2.7 tool_classify_business_review
        df_rev = pf.tool_classify_business_review(df_iso)
        self.assertIn('Review_Class', df_rev.columns)

        # 2.8 tool_classify_business_actions (compatibility)
        tail_flag = self.df['ClaimAmount'] > self.df['ClaimAmount'].quantile(0.99)
        iso_flag = iso_preds == -1
        df_clean, df_audit = pf.tool_classify_business_actions(
            self.df, dq_flags, tail_flag, iso_flag, df_inf['influential']
        )
        self.assertIn('Anomaly_Category', df_audit.columns)
        self.assertIn('Business_Action', df_audit.columns)
        self.assertLessEqual(len(df_clean), len(self.df))

    def test_03_frequency_tools(self):
        # 3.1 tool_prepare_frequency_features
        df_enc, X_mat = pf.tool_prepare_frequency_features(self.df, self.num_cols, self.cat_cols)
        self.assertEqual(X_mat.shape[0], len(self.df))

        # 3.2 tool_fit_frequency_glms
        y = self.df['ClaimNb'].values.astype(float)
        off = np.log(np.clip(self.df['Exposure'].values, 1e-6, None))
        split = 150
        glm_res = pf.tool_fit_frequency_glms(
            X_mat[:split], y[:split], off[:split],
            X_mat[split:], off[split:]
        )
        self.assertIn('Poisson GLM', glm_res)
        self.assertIn('NegBinomial', glm_res)

        # 3.3 tool_fit_frequency_xgboost
        xgb_mod, xgb_pred = pf.tool_fit_frequency_xgboost(
            df_enc.iloc[:split], y[:split], self.df['Exposure'].values[:split],
            df_enc.iloc[split:], y[split:], self.df['Exposure'].values[split:]
        )

        # 3.4 tool_calculate_actuarial_gini
        pred_pois = glm_res['Poisson GLM'][1]
        gini = pf.tool_calculate_actuarial_gini(y[split:], pred_pois)
        self.assertIsInstance(gini, float)

        # 3.5 tool_compare_frequency_models
        models_dict = {
            'Poisson GLM': glm_res['Poisson GLM'],
            'NegBinomial': glm_res['NegBinomial']
        }
        if xgb_mod is not None:
            models_dict['XGBoost Poisson'] = (xgb_mod, xgb_pred, None, None)
        comp_df = pf.tool_compare_frequency_models(models_dict, y[split:])
        self.assertIn('Gini', comp_df.columns)
        self.assertIn('MAE', comp_df.columns)

    def test_04_severity_tools(self):
        # 4.1 tool_filter_positive_claims
        df_pos, weights = pf.tool_filter_positive_claims(self.df, 'ClaimAmount', 'ClaimNb')
        if len(df_pos) < 5:
            # Ensure at least 10 positive claims for testing
            extra = self.df.iloc[:10].copy()
            extra['ClaimAmount'] = np.random.uniform(500, 3000, 10)
            extra['ClaimNb'] = 1
            df_pos, weights = pf.tool_filter_positive_claims(
                pd.concat([self.df, extra]), 'ClaimAmount', 'ClaimNb'
            )

        # Prepare features
        df_enc_pos = pd.get_dummies(df_pos[self.num_cols + self.cat_cols], drop_first=True, dtype=float)
        X_pos = np.column_stack([np.ones(len(df_enc_pos)), df_enc_pos.values])
        y_sev = df_pos['ClaimAmount'].values

        # 4.2 tool_fit_severity_models
        sev_res = pf.tool_fit_severity_models(X_pos, y_sev, weights)
        self.assertIn('Gamma GLM', sev_res)
        self.assertIn('Log-Normal', sev_res)

        # 4.3 tool_compare_severity_models
        sev_comp = pf.tool_compare_severity_models(sev_res, y_sev)
        self.assertIn('Overall A/E', sev_comp.columns)

        # 4.4 tool_calculate_severity_residuals
        gamma_model = sev_res['Gamma GLM'][0]
        resids = pf.tool_calculate_severity_residuals(gamma_model, X_pos, y_sev)
        self.assertEqual(len(resids), len(y_sev))

    def test_05_credibility_tools(self):
        # 5.1 tool_calculate_pure_premium
        freq_p = np.full(len(self.df), 0.08)
        sev_p = np.full(len(self.df), 1200.0)
        pure_prem = pf.tool_calculate_pure_premium(freq_p, sev_p)
        self.assertEqual(len(pure_prem), len(self.df))
        self.assertAlmostEqual(pure_prem[0], 96.0)

        # 5.2 tool_segment_risk_bands
        df_seg = pf.tool_segment_risk_bands(self.df, pure_prem, n_bands=5)
        self.assertIn('Risk_Band', df_seg.columns)

        # 5.3 tool_calibrate_buhlmann_credibility
        df_seg['ExpectedLoss'] = pure_prem * df_seg['Exposure']
        cred_table, corr = pf.tool_calibrate_buhlmann_credibility(
            df_seg, 'Risk_Band', 'Exposure', 'ClaimAmount', 'ExpectedLoss', K=500.0
        )
        self.assertIn('adj_RAF', cred_table.columns)

        # 5.4 tool_enforce_revenue_neutrality
        raf_series = pf.tool_enforce_revenue_neutrality(df_seg, 'Risk_Band', cred_table)
        self.assertEqual(len(raf_series), len(df_seg))

    def test_06_premium_tools(self):
        pure_prem = np.full(len(self.df), 100.0)
        raf = np.ones(len(self.df))

        # 6.1 tool_calculate_commercial_premium
        final_p, gross_p = pf.tool_calculate_commercial_premium(
            pure_prem, large_loss_loading=1.10, risk_adj_factor=raf, profit_margin=1.05,
            floor=50.0, cap=5000.0
        )
        expected_gross = 100.0 * 1.10 * 1.05
        self.assertAlmostEqual(gross_p[0], expected_gross)
        self.assertAlmostEqual(final_p[0], expected_gross)

        # 6.2 tool_compute_premium_diagnostics
        diag = pf.tool_compute_premium_diagnostics(gross_p, final_p, 50.0, 5000.0)
        self.assertEqual(diag['n_policies'], len(self.df))
        self.assertIn('mean_premium', diag)

        # 6.3 tool_compute_decile_ae_chart
        df_dec = self.df.copy()
        df_dec['FinalPremium'] = final_p
        dec_chart = pf.tool_compute_decile_ae_chart(df_dec, 'ClaimAmount', 'FinalPremium', 5)
        self.assertEqual(len(dec_chart), 5)
        self.assertIn('AE_Ratio', dec_chart.columns)

        # 6.4 tool_export_pricing_portfolio
        os.makedirs('outputs/test_run', exist_ok=True)
        pf.tool_export_pricing_portfolio(
            df_dec, {'test': 1},
            'outputs/test_run/test.xlsx',
            'outputs/test_run/test.json',
            'outputs/test_run/test.parquet'
        )
        self.assertTrue(os.path.exists('outputs/test_run/test.xlsx'))
        self.assertTrue(os.path.exists('outputs/test_run/test.json'))
        self.assertTrue(os.path.exists('outputs/test_run/test.parquet'))

    def test_07_validation_tools(self):
        meta = {'n_rows': 200, 'zero_claim_pct': 92.5}
        freq_res = {'best_model': 'Poisson GLM'}
        sev_res = {'best_model': 'Gamma GLM'}
        cred_res = {'n_segments': 5}
        pricing_reg = {'floor': 50.0, 'cap': 5000.0}

        # 7.1 tool_build_agent_dossiers
        dossiers = pf.tool_build_agent_dossiers(meta, freq_res, sev_res, cred_res, pricing_reg)
        self.assertEqual(len(dossiers), 5)

        # 7.2 tool_run_agentic_audit_simulation
        sim = pf.tool_run_agentic_audit_simulation(meta, freq_res, sev_res, cred_res, pricing_reg)
        self.assertEqual(len(sim), 5)

        # 7.3 tool_save_validation_report
        pf.tool_save_validation_report(sim, 'outputs/test_run/val_report.json')
        self.assertTrue(os.path.exists('outputs/test_run/val_report.json'))


if __name__ == '__main__':
    unittest.main()
