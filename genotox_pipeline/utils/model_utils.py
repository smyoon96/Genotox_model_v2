"""
model_utils.py — 모델 빌더 · 전처리 파이프라인 · SHAP
=====================================================
"""
import logging
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, LogisticRegression

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# 1. Preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────

def build_preprocessor(numeric_cols: List[str],
                       categorical_cols: List[str]) -> ColumnTransformer:
    """
    숫자: median imputer + StandardScaler
    범주형: most_frequent imputer + OneHotEncoder
    """
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        # passthrough 반환
        return ColumnTransformer([], remainder="passthrough")

    return ColumnTransformer(transformers, remainder="drop")


# ─────────────────────────────────────────────────────────────────────
# 2. Model builders
# ─────────────────────────────────────────────────────────────────────

def build_logreg() -> LogisticRegression:
    """Elastic-net logistic regression builder"""
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )


def build_xgb():
    """XGBoost classifier builder"""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        logger.error("xgboost not installed")
        raise


def get_model_builder(model_name: str) -> Callable:
    """모델 이름 → builder 함수"""
    if model_name == "logreg":
        return build_logreg
    elif model_name == "xgb":
        return build_xgb
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_param_dist(model_name: str) -> dict:
    """모델 이름 → hyperparameter distribution"""
    import config as cfg
    if model_name == "logreg":
        return cfg.LOGREG_PARAM_DIST
    elif model_name == "xgb":
        return cfg.XGB_PARAM_DIST
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────
# 3. Full pipeline builder (preprocessor + model)
# ─────────────────────────────────────────────────────────────────────

def build_full_pipeline(model_name: str,
                        numeric_cols: List[str],
                        categorical_cols: List[str]) -> Pipeline:
    """전처리 + 모델을 결합한 완전한 Pipeline"""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = get_model_builder(model_name)()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ─────────────────────────────────────────────────────────────────────
# 4. Imbalance-aware scale_pos_weight (XGBoost)
# ─────────────────────────────────────────────────────────────────────

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost scale_pos_weight 계산"""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# ─────────────────────────────────────────────────────────────────────
# 5. Final refit (outer-train 전체)
# ─────────────────────────────────────────────────────────────────────

def final_refit(X_train: pd.DataFrame, y_train: np.ndarray,
                model_builder: Callable, best_params: dict,
                fp_cols: List[str], fp_select_k: int = 128,
                resample_fn: Optional[Callable] = None) -> dict:
    """
    최종 모델 학습: outer-train 전체로 FP bit selection → resampling → fit.
    ★ test는 완전히 untouched ★
    """
    from utils.feature_utils import select_fp_bits_in_fold

    non_fp_cols = [c for c in X_train.columns if not c.startswith("fp_")]

    # FP bit selection (outer-train 전체)
    selected_fp = []
    if fp_cols:
        X_fp = X_train[fp_cols].values
        bits = select_fp_bits_in_fold(X_fp, y_train, k=fp_select_k)
        selected_fp = [fp_cols[i] for i in bits] if len(bits) > 0 else []

    final_features = non_fp_cols + selected_fp

    X_tr = X_train[final_features].copy()

    # Resampling
    if resample_fn is not None:
        try:
            X_tr, y_train = resample_fn(X_tr, y_train)
            # __bootstrap__ 제거
            if "__bootstrap__" in X_tr.columns:
                X_tr = X_tr.drop(columns=["__bootstrap__"])
                if "__bootstrap__" in final_features:
                    final_features.remove("__bootstrap__")
        except Exception as e:
            logger.warning(f"Final refit resampling failed: {e}")

    # Model fit
    model = model_builder()
    model.set_params(**best_params)
    model.fit(X_tr, y_train)

    return {
        "model": model,
        "features": final_features,
        "selected_fp_bits": [fp_cols.index(c) for c in selected_fp],
    }


# ─────────────────────────────────────────────────────────────────────
# 6. SHAP 분석 (best model만)
# ─────────────────────────────────────────────────────────────────────

def compute_shap(model, X: pd.DataFrame, max_samples: int = 500,
                 save_path: Optional[str] = None) -> Optional[np.ndarray]:
    """SHAP summary 계산 (best model만 경량 실행)"""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skip")
        return None

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    try:
        # XGBoost or tree → TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            try:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
                return None

    if save_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_sample, show=False,
                              plot_type="bar", max_display=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

    return shap_values
