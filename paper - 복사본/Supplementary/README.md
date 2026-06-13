# Supplementary Data — File Index

This archive contains the raw CSV outputs from the v12.1 modeling pipeline and the additional threshold-decomposition analyses referenced in the paper. Each file is mapped below to the paper table or figure it supports.

## Mapping to paper tables and figures

| Supplementary table | CSV file | Description |
|---|---|---|
| Table S2 (full 225 results) | `table_main_default_threshold.csv` | Per-configuration metrics (MCC, bootstrap CI, ROC-AUC, PR-AUC, BAcc, sens, spec, Brier, AD coverage) for all 225 configurations. |
| Table S2 (locked cascade) | `all_locked_test.csv` | Locked-configuration cascade results (Levels 1–3, Table 4 source). |
| Table S2 (CV selection state) | `scenario_cv_selection.csv` | Per-scenario CV-selection state across the 225 grid. |
| Table S3 | `table_hp_tuning_comparison.csv` | Default vs. tuned MCC plus selected hyperparameter values. |
| Table S4 | `ad_coverage_summary.csv` | AD coverage and mean Tanimoto similarity per configuration. |
| Table S5 | `cross_endpoint_overlap.csv` | Pairwise compound overlap and Cohen's κ between Ames / in-vitro CA / in-vivo MN. |
| Table S6 (full SHAP) | `shap_all_experiments.csv` | Mean \|SHAP\| values for all features × all tree-based models. |
| Table S6 (per-endpoint top-20) | `shap_top20_ames.csv`, `shap_top20_invitro.csv`, `shap_top20_invivo.csv` | Top-20 SHAP features per endpoint (Figure 6 source). |
| Table S6 (statistical comparison) | `mcnemar_tests.csv` | Cross-endpoint McNemar summary referenced in the main text. |
| Table S6 (per-endpoint stats) | `statistical_comparison_{ames,invitro,invitro_sampling,invivo,invivo_sampling}.csv` | Pairwise McNemar / Cochran's Q results within each endpoint. |
| Table S7 | (separate analysis, not in v12.1) | LDO threshold-tuning on Morgan FP-1024 + 13 physchem features. The values in the paper's Table S7 are inline; raw numbers can be regenerated from the pipeline by re-running the LDO threshold scan with the FP-1024 feature set. |
| Table S8 | `table_s8_ldo_threshold_analysis.csv` | LDO threshold decomposition on the locked-configuration compact features (reduced cross-source partition; see paper for the methodological note). |
| **Table S9 (NEW)** | `ames_leave_domain_out.csv` | **Per-algorithm LDO results with 95% bootstrap CIs (the raw source for Table 5).** Includes MCC, BAcc, ROC-AUC, sens, spec, Brier with [lo, hi] bounds. |
| Figure 7 (learning curves) | `learning_curves.csv` | Per-fraction MCC across 5-fold scaffold CV at 10–100 % training fractions. |
| Figure 5 (salt-stripping effect) | derived from `table_main_default_threshold.csv` | Mean MCC per scenario × endpoint, computed by averaging the rows in `table_main_default_threshold.csv` grouped by (endpoint, scenario, model, feature_mode). |

## Notes on a few files worth flagging

### `table_supplementary_threshold_tuning.csv`

This is a v12.1 product but is **not** the paper's Table S7. It reports threshold-tuning results on the **in-domain scaffold-split test set** (e.g., test_n = 2,712 for Ames), not on LDO test partitions. It is included here because some readers may wish to compare in-domain threshold behavior to the LDO threshold behavior in Table S7 / S8; it should not be cited as the source of Table S7.

### `ames_leave_domain_out.csv` (new prominence)

Although this file existed in v12.1, the original paper draft did not surface its CIs in the main-text Table 5. The current revision reports CIs in Table 5 and provides the full per-row breakdown in Table S9. Anyone reproducing the figures or re-tabulating Table 5 should treat this file as authoritative for the LDO numbers.

## Reproduction and contact

Pipeline version: v12.1, generated 2026-03-30. Global seed 42; scaffold-split assignments documented in the code repository linked from the paper's Data and Code Availability section.

Questions on the supplementary data should be directed to the corresponding author at hpjeon@ks.ac.kr.
