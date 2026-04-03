# Genotox Endpoint-Specific Modeling Pipeline

유전독성(Genotoxicity) Endpoint별 예측 모델 구축 파이프라인

## Endpoints
| Endpoint | 설명 | 주요 metric |
|----------|------|------------|
| **Ames** | 박테리아 복귀돌연변이 시험 | MCC |
| **in vitro** | 시험관 내 염색체이상 시험 | MCC |
| **in vivo** | 생체 내 소핵 시험 | Balanced Accuracy (specificity floor) |

---

## 빠른 시작

```bash
# 1. 필수 패키지 설치
pip install pandas numpy scikit-learn xgboost imbalanced-learn rdkit shap matplotlib joblib

# 2. 기본 실행 (shortlist 4개 조합 + leakage report + dashboard)
python run_all.py

# 3. 전체 조합 실행 (24개 조합)
python run_all.py --all

# 4. 논문용 Full Study (ablation + imbalance 전략 비교 포함)
python run_all.py --full-study

# 5. 전체 조합 + Full Study
python run_all.py --all --full-study

# 6. 개별 단계 실행
python step1_data_audit.py          # 데이터 감사
python step2_feature_build.py       # Feature 빌드
python step3_train.py               # 학습 & 평가
python step4_leakage_report.py      # Leakage 보고서
python step5_ablation.py            # Ablation study
python step6_imbalance_study.py     # Imbalance 전략 비교
python step7_dashboard.py           # 종합 Dashboard
```

또는 `launcher.ipynb`에서 셀 단위로 실행할 수 있습니다.

---

## 파이프라인 7단계

| Step | 스크립트 | 설명 | 자동실행 |
|------|----------|------|----------|
| 1 | `step1_data_audit.py` | Raw 데이터 감사 & 스키마 통합 | ✓ |
| 2 | `step2_feature_build.py` | Broad feature table 생성 | ✓ |
| 3 | `step3_train.py` | Leakage-free 학습 & 평가 | ✓ |
| 4 | `step4_leakage_report.py` | Reviewer 방어용 leakage 보고서 | ✓ |
| 5 | `step5_ablation.py` | Ablation / Robustness study | `--full-study` |
| 6 | `step6_imbalance_study.py` | Imbalance 전략 비교 | `--full-study` |
| 7 | `step7_dashboard.py` | 종합 Dashboard & 논문 도표 | ✓ |

---

## 디렉토리 구조

```
genotox_pipeline/
├── config.py                    # 전역 설정 (경로, 모델, 전략 등)
├── step1_data_audit.py          # [1] Raw 데이터 감사 & 스키마 통합
├── step2_feature_build.py       # [2] Broad feature table 생성
├── step3_train.py               # [3] Leakage-free 학습 & 평가
├── step4_leakage_report.py      # [4] Reviewer 방어용 leakage report
├── step5_ablation.py            # [5] Ablation / Robustness study
├── step6_imbalance_study.py     # [6] Imbalance 전략 비교
├── step7_dashboard.py           # [7] 종합 Dashboard
├── run_all.py                   # 마스터 실행기
├── launcher.ipynb               # Launcher notebook (얇은 wrapper)
├── requirements.txt
├── README.md
└── utils/
    ├── data_utils.py            # 데이터 로딩, merge key 정규화
    ├── feature_utils.py         # Feature 추출, FP 생성, bit selection
    ├── cv_utils.py              # Leakage-free CV, threshold tuning
    ├── resample_utils.py        # Alert bootstrap, SMOTE, hybrid
    ├── model_utils.py           # 모델 빌더, 전처리, SHAP
    └── viz_utils.py             # 시각화 (CM, ROC, PR, threshold sweep)
```

---

## 출력 구조

```
output/
├── run_config.json
├── logs/
│   ├── run_all.log
│   ├── step1_data_audit.log
│   ├── step2_feature_build.log
│   ├── step3_train.log
│   ├── step4_leakage_report.log
│   ├── step5_ablation.log
│   ├── step6_imbalance.log
│   ├── step7_dashboard.log
│   └── {endpoint}_{model}_{strategy}.log   # 실험별 로그
├── broadfp/
│   ├── ames_broadfp_train.csv
│   ├── ames_broadfp_test.csv
│   └── ...
├── summary/
│   ├── data_audit_summary.csv
│   ├── schema_manifest.json
│   ├── feature_audit_summary.csv
│   ├── summary_metrics.csv        ← 전체 실험 결과
│   ├── best_by_endpoint.csv       ← endpoint별 최고 조합
│   ├── completed_jobs.csv
│   ├── failed_jobs.csv
│   ├── leakage_report/            ← Reviewer 방어 보고서
│   │   ├── leakage_control_report.md
│   │   ├── per_fold_selected_features.csv
│   │   ├── per_fold_resampling_report.csv
│   │   └── all_selected_fp_manifests.csv
│   ├── ablation/                  ← 논문용 ablation
│   │   ├── ablation_summary.csv
│   │   ├── robustness_ci_summary.csv
│   │   ├── calibration_summary.csv
│   │   ├── applicability_domain_report.csv
│   │   ├── ablation_heatmap.png
│   │   └── {endpoint}_ablation_barplot.png
│   ├── imbalance/                 ← 전략 비교
│   │   ├── strategy_comparison_summary.csv
│   │   ├── best_strategy_by_endpoint.csv
│   │   └── false_positive_comparison.csv
│   └── dashboard/                 ← 종합 도표
│       ├── overview_table.csv
│       ├── endpoint_comparison.png
│       ├── strategy_heatmap.png
│       └── best_models_gallery/
└── artifacts/
    └── {endpoint}_{model}_{strategy}/
        ├── run_info.json
        ├── model.joblib
        ├── predictions.csv
        ├── misclassifications.csv
        ├── feature_importance.csv
        ├── selected_fp_manifest.csv
        ├── cv_fold_report.csv
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── pr_curve.png
        ├── threshold_sweep.png
        ├── feature_importance.png
        └── shap_summary_bar.png
```

---

## Leakage-Free 구조 (Reviewer 방어)

이 파이프라인은 아래 원칙을 엄격히 준수합니다.

### 1. Fingerprint Bit Selection — Fold 내부에서만

```
for each CV fold:
    1) train fold에서 prevalence filter → variance filter → MI/chi2
    2) 상위 k개 bit만 선택
    3) validation fold에는 선택된 bit만 적용
    ★ selection 기준은 train fold에서만 도출
```

**구현 위치**: `utils/feature_utils.py → select_fp_bits_in_fold()`

### 2. Resampling — Fold 내부에서만

```
for each CV fold:
    1) train fold 내부에서 resampling (alert_bootstrap / SMOTE / hybrid)
    2) validation fold는 원본 그대로
    ★ resampled 데이터가 validation이나 test에 유출되지 않음
```

**구현 위치**: `utils/resample_utils.py`, `utils/cv_utils.py → run_single_fold()`

### 3. Hyperparameter / Threshold Tuning 분리

```
Inner CV (3-fold):  hyperparameter search (RandomizedSearchCV)
Outer fold:         threshold tuning (probability sweep)
Test set:           최종 평가만 (untouched)
```

### 4. Final Refit

```
1) Outer-train 전체로 FP bit selection 재수행
2) Outer-train 전체로 resampling 재수행
3) Best hyperparameters로 최종 모델 학습
4) Test set에 적용 → 최종 metric 보고
```

### 5. Test Set = 완전히 Untouched

Test set은 **최종 모델의 평가에만** 사용됩니다.
- Feature selection 기준 도출에 사용하지 않음
- Resampling에 포함하지 않음
- Threshold tuning에 사용하지 않음

---

## Feature Block 구성

| Block | 설명 | 예시 |
|-------|------|------|
| **physchem** | 물리화학적 기술자 | MW, logP, TPSA, rot_bonds, fraction_csp3 |
| **fg_present** | 작용기 존재 여부 | fg_nitro_present, fg_amine_present |
| **fg_count** | 작용기 개수 | fg_nitro_count, fg_halogen_count |
| **rule** | 경험적 규칙 | bb_n_genotox_alerts, nitro_aromatic |
| **fingerprint** | Morgan FP | fp_0 ~ fp_255 (fold 내부에서 선별) |
| **qm** (optional) | QM 기술자 | HOMO, LUMO, gap, dipole_moment |

---

## Imbalance 대응 전략

| 전략 | 설명 |
|------|------|
| `none` | resampling 없음, class_weight='balanced' 또는 scale_pos_weight |
| `alert_bootstrap` | 고신뢰 positive family 내부 bootstrap (제한적) |
| `smotenc` | SMOTE-NC (categorical 안전 변형, FP 전체에는 미적용) |
| `hybrid` | alert_bootstrap → SMOTE 순차 적용 |

---

## Shortlist (최우선 재현 대상)

| Endpoint | Strategy | Model | 비고 |
|----------|----------|-------|------|
| Ames | none | XGBoost | 기본 |
| in vitro | alert_bootstrap | XGBoost | 기본 |
| in vitro | none | LogReg | compact benchmark |
| in vivo | alert_bootstrap | XGBoost | 기본 |

---

## 평가 지표

- **MCC** (Matthews Correlation Coefficient): 불균형 데이터에서 가장 신뢰할 수 있는 단일 지표
- **Balanced Accuracy**: sensitivity와 specificity의 평균
- **ROC-AUC / PR-AUC**: 확률 기반 성능
- **Sensitivity / Specificity**: 양성/음성 각각의 탐지율
- **Threshold sweep**: 다양한 threshold에서의 지표 변화 시각화

---

## 설정 변경

`config.py`에서 주요 설정을 변경할 수 있습니다:

- `PROJECT_ROOT`: 프로젝트 루트 경로
- `FP_RADIUS`, `FP_NBITS`: Morgan fingerprint 파라미터
- `FP_SELECT_K`: fold 내부에서 선택할 FP bit 수
- `CV_FOLDS`: CV fold 수
- `N_RANDOM_SEARCH`: RandomizedSearch 반복 수
- `SHORTLIST`: 최우선 실험 조합
- `COMPACT_FEATURES`: endpoint별 compact feature set

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| `broadfp files dirty or missing` | `python step2_feature_build.py --force` |
| `RDKit not installed` | `pip install rdkit` 또는 `conda install -c conda-forge rdkit` |
| `All folds failed` | 로그 확인: `output/logs/{experiment}.log` |
| Kernel crash | `run_all.py`는 subprocess로 분리 실행하므로 kernel 안정 |
| Memory 부족 | `config.py`에서 `FP_NBITS` 줄이거나 `FP_SELECT_K` 줄이기 |
