# Leakage Control Report

Generated: 2026-05-08 04:20:51

## 1. 파이프라인 개요

이 파이프라인은 유전독성(genotoxicity) endpoint-specific 예측 모델을 구축합니다.
세 endpoint (Ames, in vitro chromosome aberration, in vivo micronucleus)를 분리하여 처리하며,
data leakage를 방지하기 위한 엄격한 구조를 따릅니다.

## 2. Leakage 방지 구조

### 2.1 Fingerprint Bit Selection -- Fold 내부 한정

**원칙**: Morgan fingerprint의 bit selection은 반드시 CV train fold 내부에서만 수행합니다.

**구현 (`utils/feature_utils.py → select_fp_bits_in_fold()`)**:
1. Train fold에서 prevalence filter (최소 0.01)
2. Train fold에서 variance filter (최소 0.0001)
3. Train fold에서 mutual information 기반 상위 128개 bit 선택
4. Validation fold에는 선택된 bit만 적용
5. Test set에는 final refit에서 결정된 bit만 적용

**Leakage 차단 근거**:
- Selection 기준(prevalence, variance, MI score)은 train fold에서만 계산
- Validation fold의 데이터는 selection 과정에 참여하지 않음
- Test set은 최종 모델 평가에만 사용

### 2.2 Resampling -- Fold 내부 한정

**원칙**: 모든 resampling(alert bootstrap, SMOTE, hybrid)은 CV train fold 내부에서만 수행합니다.

**구현 (`utils/resample_utils.py`, `utils/cv_utils.py → run_single_fold()`)**:
1. Resampling 함수는 train fold의 X, y만 입력받음
2. Validation fold의 데이터는 원본 그대로 유지
3. `__bootstrap__` 내부 마커는 feature에서 자동 제외
4. Synthetic sample은 validation/test에 유출되지 않음

**SMOTE 특이사항**:
- Fingerprint 전체 sparse bit에 무차별 SMOTE를 적용하지 않음
- Non-FP 컬럼에만 SMOTE를 적용하고, FP 컬럼은 0으로 채움
- Categorical 컬럼이 있으면 SMOTE-NC를 사용

### 2.3 Hyperparameter Tuning vs Threshold Tuning 분리

| 단계 | 데이터 소스 | 목적 |
|------|------------|------|
| Inner CV (3-fold) | Train fold 내부 | Hyperparameter search (RandomizedSearchCV) |
| Outer fold validation | Validation fold | Threshold tuning (probability sweep) |
| Test evaluation | Test set (untouched) | 최종 성능 보고 |

**Leakage 차단 근거**:
- Hyperparameter는 train fold 내부의 inner CV로만 결정
- Threshold는 validation fold에서만 tuning
- Test set은 threshold 결정에 사용되지 않음

### 2.4 Final Refit

**순서**:
1. Outer-train 전체에서 FP bit selection 재수행
2. Outer-train 전체에서 resampling 재수행
3. CV에서 찾은 best hyperparameters로 모델 학습
4. CV에서 찾은 median threshold로 test set 평가

### 2.5 Test Set -- 완전히 Untouched

Test set은 아래 어떤 과정에도 사용되지 않습니다:
- [X] Feature selection 기준 도출
- [X] Resampling
- [X] Hyperparameter tuning
- [X] Threshold tuning
- [OK] 최종 모델 평가만

## 3. CV Protocol Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    FULL DATASET                          │
│  ┌──────────────────────────┐  ┌───────────────────────┐│
│  │      TRAIN SET           │  │     TEST SET          ││
│  │  (scaffold-aware split)  │  │  (completely held out)││
│  └────────────┬─────────────┘  └───────────────────────┘│
│               │                                          │
│     ┌─────────▼──────────┐                               │
│     │  Scaffold GroupKFold │  ← 5 folds     │
│     │  (or StratifiedKFold)│                              │
│     └─────────┬──────────┘                               │
│               │                                          │
│    ┌──────────▼────────────────────────┐                 │
│    │          PER FOLD                 │                  │
│    │  ┌──────────────────────────────┐ │                  │
│    │  │ 1. FP bit selection          │ │ ← train fold    │
│    │  │    (prevalence→variance→MI)  │ │   ONLY          │
│    │  ├──────────────────────────────┤ │                  │
│    │  │ 2. Resampling                │ │ ← train fold    │
│    │  │    (bootstrap/SMOTE/hybrid)  │ │   ONLY          │
│    │  ├──────────────────────────────┤ │                  │
│    │  │ 3. Inner CV (3-fold)         │ │ ← train fold    │
│    │  │    → hyperparameter search   │ │   ONLY          │
│    │  ├──────────────────────────────┤ │                  │
│    │  │ 4. Threshold tuning          │ │ ← validation    │
│    │  │    (probability sweep)       │ │   fold ONLY     │
│    │  └──────────────────────────────┘ │                  │
│    └───────────────────────────────────┘                  │
│               │                                           │
│     ┌─────────▼──────────┐                                │
│     │  FINAL REFIT        │                                │
│     │  1. FP selection    │ ← outer-train ONLY            │
│     │  2. Resampling      │ ← outer-train ONLY            │
│     │  3. Model fit       │ ← outer-train ONLY            │
│     └─────────┬──────────┘                                │
│               │                                           │
│     ┌─────────▼──────────┐                                │
│     │  TEST EVALUATION    │ ← test set (untouched)        │
│     │  (final metrics)    │                                │
│     └────────────────────┘                                │
└──────────────────────────────────────────────────────────┘
```

## 4. 내부 마커 / Leakage 컬럼 방어

아래 컬럼들은 자동으로 feature에서 제외됩니다:
- `__bootstrap__`: resampling 내부 마커
- `__internal__`, `__aug__`: 기타 내부 마커
- `No`, `label`, `scaffold_group`, `canonical_smiles`: 메타 컬럼
- High-cardinality string 컬럼 (nunique > 50)
- `*_x`, `*_y`: merge 잔여 suffix

**구현**: `utils/feature_utils.py → is_leakage_col()`, `utils/data_utils.py → classify_columns()`

## 5. Scaffold-Aware Evaluation

- `scaffold_group` 컬럼이 존재하면 `GroupKFold`를 사용하여 같은 scaffold가 train/validation에 동시에 나타나지 않도록 합니다.
- `scaffold_group`이 없으면 `StratifiedKFold`로 fallback합니다.
- Train/test split은 원본 데이터의 scaffold-aware split을 그대로 사용합니다.

## 6. 결론

이 파이프라인은 reviewer가 가장 먼저 점검할 세 가지 leakage 지점을 구조적으로 차단합니다:
1. **Feature selection leakage** → fold 내부 bit selection
2. **Resampling leakage** → fold 내부 resampling
3. **Evaluation leakage** → test set 완전 분리

모든 관련 코드에는 `* train fold 내부에서만 호출할 것 *` 주석이 명시되어 있으며,
per-fold report에서 각 fold의 selection/resampling 결과를 개별 확인할 수 있습니다.
