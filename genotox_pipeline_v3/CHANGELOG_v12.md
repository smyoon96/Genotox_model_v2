# CHANGELOG: v11 → v12 → v12.1 → v12.2

## v12.2 (2026-04-30) — 신규 화학공간 예측력 개선

### [FIX-1] Mode B: OOF Youden's J threshold
**Problem:** `step9_external_benchmark.py` Mode B가 threshold=0.5 고정 사용.
신규 화학공간에서 모델이 낮은 확률값을 출력하는 경향이 있어
threshold=0.5는 과도하게 보수적 → FN 폭증 원인.

**Fix:** `find_oof_threshold()` 함수 추가.
- 학습 데이터 5-fold OOF probability 수집 후 Youden's J (TPR-FPR) 최대화 threshold 반환
- Mode B에서 fixed_05(0.5)와 oof_optimal 두 threshold 비교 출력
- 결과 CSV에 `threshold_type`, `threshold_value` 컬럼 추가

### [NEW-2] Mode B: AD-stratified sensitivity/specificity
**Added:** `compute_ad_stratified()` 함수.
- `compute_ad()`의 `per_sample_max_sim`을 활용하여
  AD 내(in_ad, T≥threshold) / AD 외(out_ad, T<threshold) 분리 평가
- sensitivity 저하가 화학공간 이격(AD 외) 때문인지 정량적으로 분리
- 결과 CSV에 `in_ad_sensitivity`, `out_ad_sensitivity` 등 컬럼 추가

### [NEW-3] Mode B: Chemical space analysis
**Added:** `chemical_space_analysis()` 함수.
- TP/FN/TN/FP 별 학습 데이터와의 max Tanimoto 유사도 분포 계산
- 출력: `chemical_space_{endpoint}.csv` (per-sample) +
        `chemical_space_summary_{endpoint}.csv` (quartile 집계)
- 논문 Figure 근거: FN의 sim_median < TP의 sim_median → 화학공간 이격이 sensitivity 저하 원인
- `pct_below_05/04/03` 컬럼: 논문 "self-Tanimoto" 분석과 직결

### [NEW-4] Mode D: FN Augmentation Loop
**Added:** `fn_augmentation_loop()` 함수 + `--fn-augmentation` / `--fn-rounds` 옵션.
- 라운드마다 FN 물질을 train에 편입하고 sensitivity 변화 추적
- 각 라운드에서 OOF threshold 재계산
- AD coverage / FN의 max_sim_to_train 함께 기록
- 출력: `fn_augmentation_{endpoint}.csv` (라운드별 집계) +
        `fn_augmentation_{endpoint}_detail.csv` (편입된 물질 SMILES + sim)
- 논문 활용: ICH M7 기준(sensitivity 0.70) 달성에 필요한 편입 물질 수 추정

### [FIX] `pipeline_v2_core.compute_ad()`: per_sample_max_sim 반환 추가
- 반환 dict에 `"per_sample_max_sim": ms` (shape: n_test,) 추가
- NEW-2, NEW-3, NEW-4의 선결 조건
- 기존 반환값(coverage, n_in, n_out, mean_sim, median_sim) 유지 — 하위 호환

### CLI 변경 (step9)
```
# Mode B (신규 화학공간 검증)
python step9_external_benchmark.py --run-dir runs/RUN_ID \
  --external-ames path/to/eu_ntp_ames.csv \
  --external-invitro path/to/eu_ntp_invitro.csv \
  --external-invivo path/to/eu_ntp_invivo.csv \
  --ad-threshold 0.4

# Mode D (FN augmentation)
python step9_external_benchmark.py --run-dir runs/RUN_ID \
  --external-ames path/to/eu_ntp_ames.csv \
  --fn-augmentation --fn-rounds 3
```

---

## v12.1 Fixes (post Phase 1 verification)

### FP bit selection differentiation
**Problem:** `top_k=min(nbits//2, 256)` capped at 256 for all modes (fp512, fp1024, fp2048),
making all broad modes select identical 256 FP bits → 394 total features regardless of mode.

**Fix:** Scale top_k proportionally: fp512→128, fp1024→256, fp2048→384.
Now modes produce 266, 394, 522 features respectively — proper differentiation.

### SHAP expanded to all tree-based models
**Problem:** SHAP only computed for `is_rec` (CV-selected compact XGBoost), which missed
the actual best-performing models. Also failed silently when `shap` not installed.

**Fix:** SHAP now computed for all tree-based models (XGB, LGBM, RF) on non-sampling
endpoints with ≥50 test samples. Added SHAP summary aggregation step that creates
`shap_top20_{endpoint}.csv` for each endpoint's best model.

### External benchmark comparison script (step9_external_benchmark.py)
New standalone script with three modes:
- **Mode A**: Literature comparison table (published benchmarks vs our results)
- **Mode B**: External validation (train on our data, test on external dataset)
- **Mode C**: Random vs scaffold split comparison (quantify overestimation)

---

## Critical Bug Fixes

### [CRITICAL-1] `salt_stripped` scenario was identical to `raw_all`
**Fix:**
- `apply_scenario()` now creates a new column `_analysis_smiles`
- `find_smi()` now checks for `_analysis_smiles` first
- Feature extraction now happens **per scenario**

### [CRITICAL-2] GNN was not a real Graph Convolutional Network
**Fix:** Added `GCNLayer` (proper `σ(A_norm · X · W)`) and `GCNModel`.

### [CRITICAL-3] AD similarity calculation was incorrect
**Fix:** Explicit binarization + vectorized Tanimoto + increased `max_tr` to 1000.

### [CRITICAL-4] HP tuning data leakage
**Fix:** Documented separation of `hp_cv_mcc` vs `mcc_hp_tuned`.

---

## Methodological Improvements (v12)

- [METHOD-5] McNemar pairwise model comparison
- [METHOD-7] LDO with multiple models
- [METHOD-8] OOF threshold uses 5-fold
- [METHOD-10] Learning curves
- [METHOD-11] SHAP analysis
- [METHOD-14] FP bit selection in broad modes
