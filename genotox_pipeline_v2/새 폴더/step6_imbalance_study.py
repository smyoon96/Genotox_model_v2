"""
step6_imbalance_study.py — Class Imbalance 대응 전략 비교
=========================================================
Prompt 8: none vs alert_bootstrap vs smotenc vs hybrid 비교.

출력:
  - strategy_comparison_summary.csv
  - endpoint별 best strategy
  - false positive 증가 여부 표
  - strategy별 confusion matrix & threshold sweep
"""
import sys, json, logging, time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from utils.data_utils import safe_read_csv, check_broadfp_dirty
from utils.feature_utils import get_clean_feature_cols, separate_num_cat
from utils.cv_utils import scaffold_kfold, run_cv_loop, compute_all_metrics, tune_threshold
from utils.model_utils import get_model_builder, get_param_dist, final_refit
from utils.resample_utils import get_resample_fn
from utils.viz_utils import save_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step6_imbalance.log", mode="w"),
    ]
)
logger = logging.getLogger("step6")


# ─────────────────────────────────────────────────────────────────────
# Endpoint별 권장 전략 우선순위
# ─────────────────────────────────────────────────────────────────────
ENDPOINT_STRATEGIES = {
    "ames":    ["none", "alert_bootstrap"],
    "invitro": ["none", "alert_bootstrap", "smotenc", "hybrid"],
    "invivo":  ["none", "alert_bootstrap"],
}


def run_imbalance_experiment(endpoint: str, strategy: str,
                             model_name: str = "xgb") -> dict:
    """단일 (endpoint, strategy, model) imbalance 실험"""
    result = {
        "endpoint": endpoint,
        "strategy": strategy,
        "model": model_name,
    }

    try:
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        test_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_test.csv"

        if check_broadfp_dirty(train_path):
            result["status"] = "skip_dirty"
            return result

        df_train = safe_read_csv(train_path)
        df_test = safe_read_csv(test_path)

        label_col = cfg.ENDPOINTS[endpoint]["label_col"]
        y_train = df_train[label_col].values.astype(int)
        y_test = df_test[label_col].values.astype(int)

        # Feature setup
        feature_cols = get_clean_feature_cols(df_train, include_fp=True)
        feature_cols = [c for c in feature_cols
                        if c in df_train.columns
                        and df_train[c].dtype != "object"
                        or c == "scaffold_group_type"]

        safe_features = []
        for col in feature_cols:
            if df_train[col].dtype == "object":
                if col == "scaffold_group_type" and df_train[col].nunique() <= 20:
                    safe_features.append(col)
            else:
                safe_features.append(col)
        feature_cols = safe_features

        fp_cols = [c for c in feature_cols if c.startswith("fp_")]
        alert_cols = [c for c in feature_cols if "alert" in c.lower() or "genotox" in c.lower()]
        _, cat_cols = separate_num_cat(df_train, feature_cols)

        result["n_features"] = len(feature_cols)
        result["n_fp_bits"] = len(fp_cols)
        result["train_pos"] = int(y_train.sum())
        result["train_neg"] = int((y_train == 0).sum())
        result["train_pos_rate"] = round(y_train.mean(), 4)

        # CV
        folds = scaffold_kfold(df_train, n_splits=cfg.CV_FOLDS, seed=cfg.RANDOM_SEED)
        model_builder = get_model_builder(model_name)
        param_dist = get_param_dist(model_name)

        resample_fn = get_resample_fn(
            strategy, alert_cols=alert_cols, cat_cols=cat_cols,
            seed=cfg.RANDOM_SEED
        )

        threshold_metric = cfg.THRESHOLD_METRIC.get(endpoint, "mcc")
        spec_floor = cfg.SPECIFICITY_FLOOR.get(endpoint)

        cv_results = run_cv_loop(
            df_train=df_train,
            feature_cols=feature_cols,
            label_col=label_col,
            model_builder=model_builder,
            param_dist=param_dist,
            folds=folds,
            resample_fn=resample_fn,
            fp_select_k=cfg.FP_SELECT_K,
            n_iter=min(cfg.N_RANDOM_SEARCH, 25),
            seed=cfg.RANDOM_SEED,
            endpoint=endpoint,
            threshold_metric=threshold_metric,
            specificity_floor=spec_floor,
        )

        valid = [r for r in cv_results if "error" not in r]
        if not valid:
            result["status"] = "all_folds_failed"
            return result

        cv_df = pd.DataFrame(valid)
        for m in ["mcc", "balanced_accuracy", "sensitivity", "specificity",
                   "roc_auc", "pr_auc"]:
            if m in cv_df.columns:
                result[f"cv_{m}_mean"] = round(cv_df[m].mean(), 4)
                result[f"cv_{m}_std"] = round(cv_df[m].std(), 4)

        # Final refit + test
        best_params = valid[0].get("best_params", {})
        refit = final_refit(
            df_train[feature_cols], y_train, model_builder,
            best_params, fp_cols, cfg.FP_SELECT_K, resample_fn
        )

        final_features = refit["features"]
        X_test_final = df_test[final_features].copy()

        try:
            y_prob = refit["model"].predict_proba(X_test_final)[:, 1]
        except Exception:
            y_prob = refit["model"].predict(X_test_final).astype(float)

        median_thr = cv_df["threshold"].median() if "threshold" in cv_df.columns else 0.5
        y_pred = (y_prob >= median_thr).astype(int)

        test_metrics = compute_all_metrics(y_test, y_pred, y_prob)
        for k, v in test_metrics.items():
            result[f"test_{k}"] = v
        result["test_threshold"] = float(median_thr)

        # Plots
        exp_dir = cfg.ARTIFACT_DIR / f"imbalance_{endpoint}_{model_name}_{strategy}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        feat_imp = {}
        try:
            if hasattr(refit["model"], "feature_importances_"):
                feat_imp = dict(zip(final_features, refit["model"].feature_importances_))
            elif hasattr(refit["model"], "coef_"):
                feat_imp = dict(zip(final_features, refit["model"].coef_[0]))
        except Exception:
            pass

        save_all_plots(y_test, y_pred, y_prob, feat_imp, exp_dir,
                        prefix=f"{endpoint}_{strategy}")

        result["status"] = "completed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"  FAILED: {endpoint}/{strategy}/{model_name}: {e}")

    return result


def run_imbalance_study():
    """전체 imbalance 전략 비교 실행"""
    logger.info("=" * 60)
    logger.info("Step 6: Imbalance Strategy Comparison")
    logger.info("=" * 60)

    imbalance_dir = cfg.SUMMARY_DIR / "imbalance"
    imbalance_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for endpoint in cfg.ENDPOINTS:
        strategies = ENDPOINT_STRATEGIES.get(endpoint, ["none"])
        logger.info(f"\n--- {endpoint} (strategies: {strategies}) ---")

        for strategy in strategies:
            for model_name in ["xgb"]:  # XGB 중심
                logger.info(f"  {strategy} / {model_name} ...")
                t0 = time.time()
                result = run_imbalance_experiment(endpoint, strategy, model_name)
                elapsed = time.time() - t0
                result["elapsed_sec"] = round(elapsed, 1)
                all_results.append(result)

                mcc = result.get("test_mcc", "?")
                status = result.get("status", "?")
                logger.info(f"    → {status}, MCC={mcc} ({elapsed:.0f}s)")

    # 결과 저장
    df = pd.DataFrame(all_results)
    summary_path = imbalance_dir / "strategy_comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"\nStrategy comparison saved: {summary_path}")

    # Best strategy per endpoint
    if not df.empty and "test_mcc" in df.columns:
        best_rows = []
        for ep in cfg.ENDPOINTS:
            ep_df = df[(df["endpoint"] == ep) & (df["status"] == "completed")]
            if not ep_df.empty:
                best_idx = ep_df["test_mcc"].idxmax()
                best_rows.append(ep_df.loc[best_idx])
        if best_rows:
            best_df = pd.DataFrame(best_rows)
            best_path = imbalance_dir / "best_strategy_by_endpoint.csv"
            best_df.to_csv(best_path, index=False)
            logger.info(f"Best strategy saved: {best_path}")

    # False positive 비교 표
    if not df.empty and "test_fp" in df.columns:
        fp_table = df[["endpoint", "strategy", "model",
                        "test_fp", "test_tp", "test_fn", "test_tn",
                        "test_sensitivity", "test_specificity", "test_mcc"]].copy()
        fp_path = imbalance_dir / "false_positive_comparison.csv"
        fp_table.to_csv(fp_path, index=False)
        logger.info(f"FP comparison saved: {fp_path}")

    logger.info(f"\nImbalance study complete. Output: {imbalance_dir}")


if __name__ == "__main__":
    run_imbalance_study()
