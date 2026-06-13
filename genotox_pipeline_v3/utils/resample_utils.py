"""utils/resample_utils.py -- 클래스 불균형 대응 전략"""
import sys, logging
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np

logger = logging.getLogger("resample_utils")


def get_resample_fn(strategy: str,
                    alert_cols: List[str] = None,
                    cat_cols: List[str] = None,
                    seed: int = 42) -> Optional[Callable]:
    """
    strategy → (X, y) → (X_res, y_res) 함수 반환.
    None 반환 시 resampling 없음.
    """
    if strategy in ("none", "class_weight_only"):
        return None

    if strategy == "coverage_under":
        def _under(X, y):
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=seed)
            return rus.fit_resample(X, y)
        return _under

    if strategy == "hybrid_negative_pool":
        def _hybrid(X, y):
            try:
                from imblearn.combine import SMOTETomek
                sm = SMOTETomek(random_state=seed)
                return sm.fit_resample(X, y)
            except ImportError:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=seed)
                return rus.fit_resample(X, y)
        return _hybrid

    if strategy == "hard_negative_enriched":
        def _hard(X, y):
            from imblearn.under_sampling import NearMiss
            try:
                nm = NearMiss(version=3)
                return nm.fit_resample(X, y)
            except Exception:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=seed)
                return rus.fit_resample(X, y)
        return _hard

    logger.warning(f"Unknown strategy: {strategy} -- no resampling")
    return None
