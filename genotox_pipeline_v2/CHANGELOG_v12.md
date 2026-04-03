# CHANGELOG: v11 → v12 → v12.1

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
**File:** `pipeline_v2_core.py` → `apply_scenario()`

**Problem:** The `salt_stripped` branch only replaced SMILES values in the DataFrame column,
but all downstream feature extraction functions (`extract_fg_features`, `extract_physchem_features`,
`extract_fingerprint_features`) independently searched for the SMILES column using `find_smi()`,
which found the *original* column name (e.g., `SMILES`) — not the replaced values.
Result: 120/120 raw_all vs salt_stripped comparisons produced **identical MCC values**.

**Fix:**
- `apply_scenario()` now creates a new column `_analysis_smiles` with the properly transformed SMILES
- `find_smi()` now checks for `_analysis_smiles` first
- All feature extraction functions use `find_smi()` consistently
- `strip_salts()` function added using RDKit's `LargestFragmentChooser` + `Uncharger`
- Feature extraction now happens **per scenario** (not pre-computed once for all scenarios)

**Verification:** The pipeline now checks `salt_stripped_verified` in the checklist — confirms
that salt_stripped results differ from raw_all.

---

### [CRITICAL-2] GNN was not a real Graph Convolutional Network
**File:** `genotox_pipeline.py` → `MolGCN`, `GCNLayer`, `GCNModel`

**Problem:** The `_graph_forward()` method did:
```python
for _ in range(2):
    x = torch.matmul(a, x)  # raw adjacency multiplication
return model(x.mean(dim=0))  # model = nn.Sequential(Linear→ReLU→...)
```
This is NOT a GCN. Real GCN applies `X' = σ(A·X·W)` where W is a **learnable weight
inside the message passing loop**. The old code was effectively an MLP on mean atom features
with a fixed neighborhood aggregation. This explains why GNN MCC was near 0 on most endpoints.

**Fix:**
- Added `GCNLayer` class: proper `σ(A_norm · X · W)` with learnable `nn.Linear`, LayerNorm, dropout
- Added `GCNModel` class: multi-layer GCN with global mean pooling → classifier head
- Added gradient clipping (`clip_grad_norm_`) and weight decay for training stability

---

### [CRITICAL-3] AD similarity calculation was incorrect
**File:** `pipeline_v2_core.py` → `compute_ad()`

**Problem:** Used element-wise multiplication on float arrays for Tanimoto. While correct for
strict binary (0/1) vectors, the computation was fragile:
- `max_tr=300` made AD coverage estimates unstable
- No explicit binarization step

**Fix:**
- Explicit binarization: `(fp > 0).astype(float32)` before computation
- Vectorized Tanimoto: `|A∩B| / |A∪B|` with proper zero-division handling
- Increased `max_tr` from 300 to 1000
- Added `median_sim` to output

---

### [CRITICAL-4] HP tuning data leakage
**File:** `genotox_pipeline.py` → `tune_model()`

**Problem:** `RandomizedSearchCV` with `refit=True` refits on the full training set.
The `best_score_` reflects internal CV, but the model is then evaluated on the same test set
used for the fixed-param model. This makes `mcc_hp_tuned` vs `mcc` comparison unfair because
HP selection optimized for performance on internal folds of the same train data.

**Fix:**
- Documented that `hp_cv_mcc` is the honest internal estimate
- The test evaluation is legitimate (held-out test set was never seen)
- Added clear separation: `mcc` = fixed params on test, `mcc_hp_tuned` = tuned params on test
- The paper should report both with proper context

---

## Methodological Improvements

### [METHOD-5] Statistical tests for model comparison
**File:** `pipeline_v2_core.py`

- `mcnemar_test()`: McNemar's chi-squared test with continuity correction for pairwise classifier comparison
- `pairwise_model_comparison()`: generates comparison matrix for all model pairs within an endpoint
- Results saved to `mcnemar_tests.csv` and `statistical_comparison_{endpoint}.csv`

### [METHOD-7] Domain confounding tested with multiple models
**File:** `genotox_pipeline.py` → Step 6

- LDO now runs with XGBoost, LightGBM, and Random Forest (was XGBoost only)
- Results include model name in the output CSV

### [METHOD-8] OOF threshold uses 5-fold (was 3)
**File:** `genotox_pipeline.py` → `oof_threshold()`

- Changed `n_folds=3` → `n_folds=5` for more stable threshold estimation

### [METHOD-10] Learning curves added
**File:** `pipeline_v2_core.py` → `learning_curve()`

- Trains on increasing fractions [10%, 20%, 30%, 50%, 70%, 100%] of training data
- Uses scaffold-aware splitting
- Results saved to `learning_curves.csv`
- Helps distinguish "not enough data" from "inherently hard prediction"

### [METHOD-11] SHAP analysis for interpretability
**File:** `genotox_pipeline.py` → `compute_shap_importance()`

- TreeExplainer for tree-based models (XGB, LGBM, RF)
- KernelExplainer fallback for other models
- Computed for CV-selected best model per endpoint
- Results saved to `shap_importance.csv` per experiment

### [METHOD-14] FP bit selection in broad modes
**File:** `step4_feature_extraction.py` → `select_fingerprint_bits()`

- Mutual information-based bit selection (train-only, no leakage)
- Prevalence filter (1%–99%) + top-k by MI score
- Applied to all broad_fp modes: selects top `min(nbits/2, 256)` bits
- May improve FP-2048 performance (was degraded vs FP-512 due to noise)

---

## Other Improvements

### Interpretation & Warnings
- `interpret_result()` now flags sampling endpoints explicitly
- Warns when MCC < 0.1 on datasets with >50 test samples

### Bootstrap CI
- Increased from 200 to 500 bootstrap iterations for tighter CIs

### Code Cleanup
- Removed dead code paths
- Added proper error handling with `traceback.print_exc()`
- Config version tracking in output JSON
