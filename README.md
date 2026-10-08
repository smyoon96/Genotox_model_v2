# Genotox_model_v2

**"Data Source Heterogeneity, Not Algorithm Choice, Drives Performance Gaps in QSAR for Mutagenicity"**  
Submitted to *Journal of Cheminformatics* (Manuscript ID: ci-2026-01929c)

---

## Repository Structure

```
Genotox_model_v2/
├── dataset/
│   ├── data/
│   │   ├── ames_combine.xlsx          # Raw Ames mutagenicity dataset
│   │   ├── invitro_pre.csv            # Preprocessed in vitro chromosomal aberration dataset
│   │   ├── invivo_pre.csv             # Preprocessed in vivo micronucleus dataset
│   │   └── fg_descriptor_analysis/   # Functional group & descriptor analysis outputs
│   └── preprocess.py                  # Preprocessing script
│
├── genotox_pipeline_v2/
│   ├── genotox_pipeline.py            # Main pipeline entry point
│   ├── pipeline_v2_core.py            # Core modeling functions
│   ├── config.py                      # Configuration (datasets, algorithms, FP settings)
│   ├── negative_sampling.py           # Negative sampling strategies (KMeans medoid, etc.)
│   ├── run_all.py                     # Full pipeline runner
│   ├── launcher.ipynb                 # Jupyter notebook launcher
│   ├── step1_data_audit.py            # Data quality audit
│   ├── step2_feature_build.py         # Feature construction (Morgan FP, compact descriptors)
│   ├── step2b_preprocessing_impact.py # Preprocessing impact analysis
│   ├── step3_train.py                 # Model training (RF, XGB, LGBM, SVM, LR, ANN, GNN)
│   ├── step4_feature_extraction.py    # Feature importance & SHAP extraction
│   ├── step4_leakage_report.py        # Data leakage detection
│   ├── step5_ablation.py              # Ablation study
│   ├── step6_imbalance_study.py       # Class imbalance analysis
│   ├── step7_dashboard.py             # Results dashboard
│   ├── step8_comprehensive_analysis.py# Comprehensive statistical analysis
│   ├── step9_external_benchmark.py    # External validation & AD analysis
│   ├── runs/                          # Experiment outputs (v11, v12, v12.1)
│   │   └── YYYYMMDD_HHMMSS_vXX/      # Per-run results: metrics, predictions, split metadata
│   └── broadfp/                       # Broad fingerprint feature files (train/test splits)
│
└── paper/                             # LaTeX manuscript files
```

---

## Datasets

Three genotoxicity endpoints are evaluated:

| Dataset | n | Positive rate | Source type |
|---|---|---|---|
| Ames mutagenicity | 14,771 | Mixed (drug/industrial) | Multi-source compiled |
| *In vitro* chromosomal aberration | 1,656 | Imbalanced | Industrial-focused |
| *In vivo* micronucleus | 2,199 | Imbalanced | Industrial-focused |

Each dataset includes a `source` column distinguishing drug-domain and industrial-domain compounds, which is the primary variable of interest in this study.

---

## Reproducing Results

### Requirements

```
Python >= 3.9
rdkit >= 2023.09
scikit-learn >= 1.3
xgboost
lightgbm
torch (for GNN)
pandas, numpy, scipy
```

### Run

```bash
# Full pipeline (reproduces paper results)
python genotox_pipeline_v2/run_all.py

# Or step-by-step
python genotox_pipeline_v2/step1_data_audit.py
python genotox_pipeline_v2/step2_feature_build.py
python genotox_pipeline_v2/step3_train.py
...
```

Key reproducibility parameters (locked across all experiments):

- Global random seed: **42**
- Split method: scaffold-based (Bemis-Murcko)
- Preprocessing scenarios: `raw_all`, `salt_stripped`, `no_metal`
- Fingerprint sizes: 512, 1024, 2048 bits (Morgan radius=2)
- Reference run: `runs/20260330_101505_v12.1/`

All locked test split assignments are in `runs/20260330_101505_v12.1/{endpoint}/fixed_split.csv`.


---



## Contact

Seongmin Yoon (smyoon@ks.ac.kr)  
Department of Biosafety, Kyungsung University
