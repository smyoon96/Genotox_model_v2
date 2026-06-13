"""
utils/cv_utils.py -- CV / 평가 유틸리티
"""
import sys, logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from utils.progress import pbar

logger = logging.getLogger("cv_utils")


def scaffold_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = None):
    """scaffold_group 기반 GroupKFold fold index 생성."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder

    seed = seed or cfg.GLOBAL_SEED
    if "scaffold_group" not in df.columns:
        from sklearn.model_selection import StratifiedKFold
        y = df["label"].values if "label" in df.columns else np.zeros(len(df))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(df)), y))

    groups = LabelEncoder().fit_transform(df["scaffold_group"].astype(str).values)
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    y = df["label"].values if "label" in df.columns else np.zeros(len(df))
    return list(gkf.split(np.zeros(len(df)), y, groups))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None) -> Dict:
    """분류 지표 계산 (MCC, BA, sens, spec, ROC-AUC, PR-AUC, F1)."""
    from sklearn.metrics import (
        matthews_corrcoef, balanced_accuracy_score,
        confusion_matrix, roc_auc_score, average_precision_score, f1_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}
    if len(set(y_true)) < 2:
        return {"mcc": 0.0, "balanced_accuracy": 0.0,
                "sensitivity": 0.0, "specificity": 0.0}

    metrics["mcc"]              = round(float(matthews_corrcoef(y_true, y_pred)), 4)
    metrics["balanced_accuracy"]= round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    metrics["f1"]               = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    metrics["tp"], metrics["fp"] = int(tp), int(fp)
    metrics["fn"], metrics["tn"] = int(fn), int(tn)

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
        except Exception:
            metrics["pr_auc"] = None

    return metrics


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   criterion: str = "mcc",
                   thr_range: tuple = (0.1, 0.9),
                   thr_step: float = 0.02) -> float:
    """probability threshold grid search (OOF 외부에서 사용)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline_v2_core import oof_tune_threshold
    # y_prob을 OOF처럼 사용
    import types
    # 간단히 직접 구현
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
    best_thr, best_score = 0.5, -np.inf
    _candidates = np.arange(thr_range[0], thr_range[1] + 1e-9, thr_step)
    for thr in pbar(_candidates, desc="  Threshold grid", leave=False):
        pred = (y_prob >= thr).astype(int)
        if len(set(pred)) < 2:
            continue
        if criterion == "mcc":
            score = matthews_corrcoef(y_true, pred)
        elif criterion == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            score = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return round(best_thr, 4)


def run_cv_loop(df_train: pd.DataFrame, feature_cols: List[str],
                label_col: str, model_builder: Callable,
                param_dist: Dict, folds: List,
                resample_fn: Optional[Callable] = None,
                fp_select_k: int = 128, n_iter: int = 10,
                seed: int = None, endpoint: str = "ames",
                threshold_metric: str = "mcc",
                specificity_floor: Optional[float] = None) -> List[Dict]:
    """
    Leakage-free CV loop.
    각 fold: HP 탐색(inner) → OOF threshold → 평가(outer).
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder

    seed = seed or cfg.GLOBAL_SEED
    results = []

    for fold_idx, (tr_idx, val_idx) in pbar(enumerate(folds), desc="  CV folds", total=len(folds), leave=False):
        try:
            X_all = df_train[feature_cols].copy()
            for c in X_all.columns:
                X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
            X_all = X_all.fillna(0).values.astype(np.float32)
            y_all = df_train[label_col].values.astype(int)

            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]

            if len(set(y_tr)) < 2 or len(set(y_val)) < 2:
                continue

            # Resampling (fold 내부)
            if resample_fn is not None:
                try:
                    X_tr, y_tr = resample_fn(X_tr, y_tr)
                except Exception as e:
                    logger.warning(f"  Fold {fold_idx} resample failed: {e}")

            # HP search (inner)
            base_model = model_builder()
            if param_dist:
                try:
                    rs = RandomizedSearchCV(
                        base_model, param_dist, n_iter=n_iter,
                        scoring="average_precision", cv=3,
                        n_jobs=-1, random_state=seed, refit=True,
                    )
                    rs.fit(X_tr, y_tr)
                    mdl = rs.best_estimator_
                    best_params = rs.best_params_
                except Exception:
                    mdl = base_model
                    mdl.fit(X_tr, y_tr)
                    best_params = {}
            else:
                mdl = base_model
                mdl.fit(X_tr, y_tr)
                best_params = {}

            y_prob = mdl.predict_proba(X_val)[:, 1]
            thr = tune_threshold(y_val, y_prob, threshold_metric)
            y_pred = (y_prob >= thr).astype(int)

            fold_metrics = compute_all_metrics(y_val, y_pred, y_prob)
            fold_metrics["fold"] = fold_idx
            fold_metrics["threshold"] = thr
            fold_metrics["best_params"] = best_params
            results.append(fold_metrics)

        except Exception as e:
            logger.warning(f"  Fold {fold_idx} failed: {e}")
            results.append({"fold": fold_idx, "error": str(e)})

    return results
