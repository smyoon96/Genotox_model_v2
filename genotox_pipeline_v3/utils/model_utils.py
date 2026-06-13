"""utils/model_utils.py -- 모델 팩토리 / SHAP / refit 유틸리티"""
import sys, logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

logger = logging.getLogger("model_utils")


def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg) / max(n_pos, 1)


def get_model_builder(model_name: str) -> Callable:
    """model_name → () → sklearn estimator 팩토리 반환."""
    def _builder():
        spw = 1.0  # 호출 시 설정 불가, 기본값 사용 (run_cv_loop에서 override)
        if model_name == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  eval_metric="logloss", random_state=cfg.GLOBAL_SEED,
                                  n_jobs=-1, verbosity=0)
        elif model_name == "lgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                   verbose=-1, random_state=cfg.GLOBAL_SEED, n_jobs=-1)
        elif model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, max_depth=10,
                                           class_weight="balanced",
                                           random_state=cfg.GLOBAL_SEED, n_jobs=-1)
        elif model_name == "svm":
            from sklearn.svm import SVC
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([("sc", StandardScaler()),
                              ("clf", SVC(probability=True, kernel="rbf",
                                          class_weight="balanced",
                                          random_state=cfg.GLOBAL_SEED))])
        elif model_name == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=1000,
                                                          class_weight="balanced",
                                                          random_state=cfg.GLOBAL_SEED))])
        elif model_name == "ann":
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([("sc", StandardScaler()),
                              ("clf", MLPClassifier(hidden_layer_sizes=(128, 64),
                                                     max_iter=300,
                                                     random_state=cfg.GLOBAL_SEED))])
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return _builder


def get_param_dist(model_name: str) -> Dict:
    """config.HP_GRIDS에서 HP 탐색 공간 반환."""
    return cfg.HP_GRIDS.get(model_name, {})


def final_refit(X_train, y_train, model_builder: Callable,
                best_params: Dict, fp_cols: List[str],
                fp_select_k: int = 128,
                resample_fn=None) -> Dict:
    """outer-train 전체로 최종 모델 학습."""
    from sklearn.metrics import mutual_info_score

    X = X_train.copy() if hasattr(X_train, "copy") else X_train
    if hasattr(X, "fillna"):
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0).values.astype(np.float32)
    else:
        X = np.array(X, dtype=np.float32)

    y = np.array(y_train, dtype=int)

    if resample_fn is not None:
        try:
            X, y = resample_fn(X, y)
        except Exception as e:
            logger.warning(f"  Final refit resample failed: {e}")

    mdl = model_builder()
    if best_params:
        try:
            mdl.set_params(**best_params)
        except Exception:
            pass

    mdl.fit(X, y)

    # feature list
    if hasattr(X_train, "columns"):
        features = list(X_train.columns)
    else:
        features = [str(i) for i in range(X.shape[1])]

    return {
        "model": mdl,
        "features": features,
        "selected_fp_bits": None,
    }


def compute_shap(model, X: pd.DataFrame, save_path: Optional[str] = None):
    """SHAP bar plot 저장 (tree-based 모델만)."""
    try:
        import shap
        import matplotlib.pyplot as plt
        explainer = shap.TreeExplainer(model)
        X_np = X.fillna(0).values.astype(np.float32)
        sv = explainer.shap_values(X_np[:min(200, len(X_np))])
        if isinstance(sv, list):
            sv = sv[1]
        shap.summary_plot(sv, X_np, feature_names=list(X.columns),
                          plot_type="bar", show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close("all")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
