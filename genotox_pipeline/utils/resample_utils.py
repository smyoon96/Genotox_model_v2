"""
resample_utils.py — Class imbalance 대응 전략
=============================================
★ 모든 resampling은 반드시 CV train fold 내부에서만 수행 ★
★ validation fold / test set에는 절대 적용하지 않음 ★
"""
import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 1. Alert-family Bootstrap
# ─────────────────────────────────────────────────────────────────────

def alert_bootstrap(X: pd.DataFrame, y: np.ndarray,
                    alert_cols: Optional[List[str]] = None,
                    target_ratio: float = 0.5,
                    max_factor: int = 3,
                    seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    High-confidence positive family 내부에서 제한적 bootstrap.
    alert 컬럼이 없으면 전체 positive에서 bootstrap.

    ★ train fold 내부에서만 호출할 것 ★
    """
    rng = np.random.RandomState(seed)

    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_pos >= n_neg:
        return X, y  # 이미 균형 또는 positive 없음

    # 목표 positive 수
    target_n_pos = min(int(n_neg * target_ratio), n_pos * max_factor)
    n_to_add = target_n_pos - n_pos

    if n_to_add <= 0:
        return X, y

    # alert family가 있으면 해당 그룹에서만 bootstrap
    pos_X = X[pos_mask]
    if alert_cols:
        available = [c for c in alert_cols if c in X.columns]
        if available:
            alert_score = pos_X[available].sum(axis=1)
            high_conf_mask = alert_score > 0
            if high_conf_mask.sum() > 0:
                pos_X = pos_X[high_conf_mask]

    # bootstrap sampling
    boot_idx = rng.choice(pos_X.index, size=n_to_add, replace=True)
    boot_X = pos_X.loc[boot_idx].copy()
    boot_X.index = range(len(X), len(X) + n_to_add)

    # __bootstrap__ marker 추가 (feature에서는 제외됨)
    X_out = pd.concat([X, boot_X], axis=0)
    y_out = np.concatenate([y, np.ones(n_to_add, dtype=y.dtype)])

    # 내부 마커 (나중에 제거)
    X_out["__bootstrap__"] = 0
    X_out.iloc[len(X):, X_out.columns.get_loc("__bootstrap__")] = 1

    logger.info(f"Alert bootstrap: {n_pos} → {target_n_pos} positives (+{n_to_add})")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────
# 2. SMOTE-NC (categorical 안전 변형)
# ─────────────────────────────────────────────────────────────────────

def safe_smotenc(X: pd.DataFrame, y: np.ndarray,
                 cat_cols: Optional[List[str]] = None,
                 seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    SMOTE-NC 또는 KMeans-SMOTE를 안전하게 적용.
    fingerprint 전체 sparse bit에 무차별 적용하지 않음.

    ★ train fold 내부에서만 호출할 것 ★
    """
    try:
        from imblearn.over_sampling import SMOTENC, SMOTE, KMeansSMOTE
    except ImportError:
        logger.error("imbalanced-learn not installed — SMOTE disabled")
        return X, y

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_pos >= n_neg:
        return X, y

    # k_neighbors 조정 (소수 클래스가 적을 때)
    k = min(5, n_pos - 1)
    if k < 1:
        logger.warning("Too few positive samples for SMOTE")
        return X, y

    # fingerprint bit 컬럼을 제외한 subset에서 SMOTE 적용
    fp_cols = [c for c in X.columns if c.startswith("fp_")]
    non_fp_cols = [c for c in X.columns if not c.startswith("fp_") and c != "__bootstrap__"]

    if not non_fp_cols:
        logger.warning("No non-FP columns for SMOTE — skip")
        return X, y

    X_for_smote = X[non_fp_cols].copy()

    # categorical 인덱스 찾기
    cat_indices = []
    if cat_cols:
        for i, col in enumerate(non_fp_cols):
            if col in cat_cols:
                cat_indices.append(i)

    try:
        if cat_indices:
            smote = SMOTENC(
                categorical_features=cat_indices,
                k_neighbors=k, random_state=seed
            )
        else:
            smote = SMOTE(k_neighbors=k, random_state=seed)

        X_res, y_res = smote.fit_resample(X_for_smote, y)
        X_out = pd.DataFrame(X_res, columns=non_fp_cols)

        # FP 컬럼은 nearest neighbor에서 가져오기 (synthetic에는 0으로)
        if fp_cols:
            n_original = len(X)
            n_synthetic = len(X_out) - n_original
            fp_original = X[fp_cols].values
            fp_synthetic = np.zeros((n_synthetic, len(fp_cols)), dtype=np.int8)
            fp_all = np.vstack([fp_original, fp_synthetic])
            for i, fc in enumerate(fp_cols):
                X_out[fc] = fp_all[:, i]

        logger.info(f"SMOTE: {n_pos} → {(y_res == 1).sum()} positives")
        return X_out, y_res

    except Exception as e:
        logger.warning(f"SMOTE failed: {e}")
        return X, y


# ─────────────────────────────────────────────────────────────────────
# 3. Hybrid (alert bootstrap + SMOTE)
# ─────────────────────────────────────────────────────────────────────

def hybrid_resample(X: pd.DataFrame, y: np.ndarray,
                    alert_cols: Optional[List[str]] = None,
                    cat_cols: Optional[List[str]] = None,
                    seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Alert bootstrap → SMOTE 순서로 hybrid 적용.
    ★ train fold 내부에서만 호출할 것 ★
    """
    # Step 1: alert bootstrap (moderate)
    X_ab, y_ab = alert_bootstrap(X, y, alert_cols=alert_cols,
                                  target_ratio=0.35, max_factor=2, seed=seed)
    # __bootstrap__ 제거 후 SMOTE
    if "__bootstrap__" in X_ab.columns:
        X_ab = X_ab.drop(columns=["__bootstrap__"])

    # Step 2: SMOTE
    X_out, y_out = safe_smotenc(X_ab, y_ab, cat_cols=cat_cols, seed=seed)
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────
# 4. Strategy dispatcher
# ─────────────────────────────────────────────────────────────────────

def get_resample_fn(strategy: str,
                    alert_cols: Optional[List[str]] = None,
                    cat_cols: Optional[List[str]] = None,
                    seed: int = 42):
    """
    strategy 이름 → resampling 함수 반환.
    'none'이면 None 반환 (resampling 안 함).
    """
    if strategy == "none":
        return None
    elif strategy == "alert_bootstrap":
        return lambda X, y: alert_bootstrap(X, y, alert_cols=alert_cols, seed=seed)
    elif strategy in ("smotenc", "smote"):
        return lambda X, y: safe_smotenc(X, y, cat_cols=cat_cols, seed=seed)
    elif strategy == "hybrid":
        return lambda X, y: hybrid_resample(X, y, alert_cols=alert_cols,
                                             cat_cols=cat_cols, seed=seed)
    else:
        logger.warning(f"Unknown strategy '{strategy}' — no resampling")
        return None
