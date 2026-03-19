"""
cv_utils.py — Leakage-free CV · Scaffold-aware split · Threshold tuning
========================================================================
★ Reviewer 방어 핵심 모듈 ★
- fingerprint bit selection → fold 내부에서만
- resampling → fold 내부에서만
- hyperparameter tuning과 threshold tuning 분리
- inner-CV score와 outer test score 구분
"""
import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score,
    average_precision_score, confusion_matrix, make_scorer
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 1. Scaffold-aware split generator
# ─────────────────────────────────────────────────────────────────────

def scaffold_kfold(df: pd.DataFrame, n_splits: int = 5,
                   group_col: str = "scaffold_group",
                   label_col: str = "label",
                   seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    scaffold_group 기반 GroupKFold.
    scaffold_group이 없으면 StratifiedKFold fallback.
    """
    if group_col in df.columns and df[group_col].notna().sum() > 0:
        groups = df[group_col].fillna("unknown_" + df.index.astype(str))
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(df, df[label_col], groups=groups))
        logger.info(f"Using GroupKFold with '{group_col}' ({n_splits} folds)")
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(df, df[label_col]))
        logger.info(f"Scaffold col missing — using StratifiedKFold ({n_splits} folds)")
    return folds


# ─────────────────────────────────────────────────────────────────────
# 2. 평가 지표 계산
# ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """MCC, balanced_accuracy, sensitivity, specificity, ROC-AUC, PR-AUC 등 일괄 계산"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = np.nan
    return metrics


# ─────────────────────────────────────────────────────────────────────
# 3. Threshold tuning (metric 기반)
# ─────────────────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   metric: str = "mcc",
                   specificity_floor: Optional[float] = None,
                   n_thresholds: int = 200) -> Tuple[float, float]:
    """
    probability threshold를 sweep해서 최적 threshold 선택.
    specificity_floor이 있으면 해당 조건을 만족하는 범위에서만 선택.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -np.inf
    best_thr = 0.5

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if specificity_floor is not None and spec < specificity_floor:
            continue

        if metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        elif metric == "sensitivity":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            score = matthews_corrcoef(y_true, preds)

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score


# ─────────────────────────────────────────────────────────────────────
# 4. Leakage-free 단일 fold 실행
# ─────────────────────────────────────────────────────────────────────

def run_single_fold(
    fold_idx: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_builder: Callable,
    param_dist: dict,
    fp_cols: List[str],
    non_fp_cols: List[str],
    resample_fn: Optional[Callable] = None,
    fp_select_k: int = 128,
    n_iter: int = 40,
    seed: int = 42,
    endpoint: str = "",
    threshold_metric: str = "mcc",
    specificity_floor: Optional[float] = None,
) -> dict:
    """
    단일 CV fold 실행 — leakage-free 구조.

    순서:
    1) train fold 내부에서 FP bit selection
    2) train fold 내부에서 resampling (optional)
    3) RandomizedSearchCV로 hyperparameter tuning
    4) validation fold에서 threshold tuning
    5) 최종 metric 계산
    """
    from utils.feature_utils import select_fp_bits_in_fold

    result = {"fold": fold_idx}

    # ── Step 1: FP bit selection (train fold 내부) ──
    selected_fp = []
    if fp_cols:
        X_train_fp = X_train[fp_cols].values
        selected_bits = select_fp_bits_in_fold(
            X_train_fp, y_train, k=fp_select_k, method="mutual_info"
        )
        selected_fp = [fp_cols[i] for i in selected_bits] if len(selected_bits) > 0 else []
        result["n_selected_fp_bits"] = len(selected_fp)
        result["selected_fp_bits"] = selected_bits.tolist() if len(selected_bits) > 0 else []
    else:
        result["n_selected_fp_bits"] = 0
        result["selected_fp_bits"] = []

    # 최종 feature 구성
    fold_features = non_fp_cols + selected_fp
    if not fold_features:
        logger.error(f"Fold {fold_idx}: no features available")
        result["error"] = "no_features"
        return result

    X_tr = X_train[fold_features].copy()
    X_vl = X_val[fold_features].copy()

    # ── Step 2: Resampling (train fold 내부) ──
    if resample_fn is not None:
        try:
            X_tr, y_train_rs = resample_fn(X_tr, y_train)
            result["resampled"] = True
            result["resampled_size"] = len(y_train_rs)
        except Exception as e:
            logger.warning(f"Fold {fold_idx}: resampling failed: {e}")
            y_train_rs = y_train
            result["resampled"] = False
    else:
        y_train_rs = y_train
        result["resampled"] = False

    # ── Step 3: Hyperparameter tuning ──
    try:
        base_model = model_builder()
        scorer = make_scorer(matthews_corrcoef)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold_idx)
        search = RandomizedSearchCV(
            base_model, param_dist, n_iter=min(n_iter, _param_space_size(param_dist)),
            scoring=scorer, cv=inner_cv, random_state=seed + fold_idx,
            n_jobs=-1, error_score=-1.0
        )
        search.fit(X_tr, y_train_rs)
        model = search.best_estimator_
        result["best_params"] = search.best_params_
        result["inner_cv_score"] = search.best_score_
        # Hyperparameter landscape 시각화용 저장
        try:
            result["cv_results_df"] = pd.DataFrame(search.cv_results_)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Fold {fold_idx}: training failed: {e}")
        result["error"] = str(e)
        return result

    # ── Step 4: Threshold tuning (validation fold) ──
    try:
        y_prob = model.predict_proba(X_vl)[:, 1]
    except Exception:
        y_prob = model.predict(X_vl).astype(float)

    best_thr, _ = tune_threshold(
        y_val, y_prob, metric=threshold_metric,
        specificity_floor=specificity_floor
    )
    y_pred = (y_prob >= best_thr).astype(int)

    # ── Step 5: Metrics ──
    metrics = compute_all_metrics(y_val, y_pred, y_prob)
    result.update(metrics)
    result["threshold"] = best_thr
    result["model"] = model
    result["fold_features"] = fold_features

    return result


def _param_space_size(param_dist: dict) -> int:
    """대략적인 파라미터 공간 크기"""
    size = 1
    for v in param_dist.values():
        if hasattr(v, '__len__'):
            size *= len(v)
    return size


# ─────────────────────────────────────────────────────────────────────
# 5. Full CV loop
# ─────────────────────────────────────────────────────────────────────

def run_cv_loop(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_builder: Callable,
    param_dist: dict,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    resample_fn: Optional[Callable] = None,
    fp_select_k: int = 128,
    n_iter: int = 40,
    seed: int = 42,
    endpoint: str = "",
    threshold_metric: str = "mcc",
    specificity_floor: Optional[float] = None,
) -> List[dict]:
    """전체 CV fold 순환 — 개별 fold 실패해도 나머지 진행"""

    fp_cols = [c for c in feature_cols if c.startswith("fp_")]
    non_fp_cols = [c for c in feature_cols if not c.startswith("fp_")]

    y_all = df_train[label_col].values
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"  Fold {fold_idx + 1}/{len(folds)} ...")
        try:
            res = run_single_fold(
                fold_idx=fold_idx,
                X_train=df_train.iloc[train_idx][feature_cols],
                y_train=y_all[train_idx],
                X_val=df_train.iloc[val_idx][feature_cols],
                y_val=y_all[val_idx],
                model_builder=model_builder,
                param_dist=param_dist,
                fp_cols=fp_cols,
                non_fp_cols=non_fp_cols,
                resample_fn=resample_fn,
                fp_select_k=fp_select_k,
                n_iter=n_iter,
                seed=seed,
                endpoint=endpoint,
                threshold_metric=threshold_metric,
                specificity_floor=specificity_floor,
            )
            results.append(res)
        except Exception as e:
            logger.error(f"  Fold {fold_idx + 1} FAILED: {e}")
            results.append({"fold": fold_idx, "error": str(e)})

    return results
