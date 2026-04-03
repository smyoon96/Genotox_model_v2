"""
step5_ablation.py — 논문용 Ablation / Robustness Study
======================================================
Prompt 10: compact vs broad vs broad_fp vs broad_fp+QM 비교,
           bootstrap CI, applicability domain, calibration curve.

출력:
  - ablation_summary.csv
  - robustness_ci_summary.csv
  - calibration_summary.csv
  - applicability_domain_report.csv
  - 논문 그림용 종합 도표 PNG
"""
import sys, json, logging, traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from utils.data_utils import safe_read_csv, check_broadfp_dirty, classify_columns
from utils.feature_utils import (
    get_clean_feature_cols, separate_num_cat, classify_feature_blocks
)
from utils.cv_utils import scaffold_kfold, compute_all_metrics, tune_threshold
from utils.model_utils import get_model_builder, get_param_dist, compute_scale_pos_weight
from utils.resample_utils import get_resample_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step5_ablation.log", mode="w"),
    ]
)
logger = logging.getLogger("step5")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────
# 1. Feature block ablation
# ─────────────────────────────────────────────────────────────────────

def get_feature_subsets(df: pd.DataFrame, endpoint: str) -> dict:
    """ablation 실험용 feature subset 정의"""
    all_features = get_clean_feature_cols(df, include_fp=True, include_qm=True)
    blocks = classify_feature_blocks(all_features)

    subsets = {}

    # compact: endpoint별 수동 feature
    compact_feats = cfg.COMPACT_FEATURES.get(endpoint, [])
    compact_available = [c for c in compact_feats if c in df.columns]
    if compact_available:
        subsets["compact"] = compact_available

    # broad_tabular: physchem + fg + rules (FP 제외)
    broad_tab = blocks["physchem"] + blocks["fg_present"] + blocks["fg_count"] + blocks["rule"] + blocks["other"]
    if broad_tab:
        subsets["broad_tabular"] = broad_tab

    # broad_fp: broad_tabular + fingerprint
    broad_fp = broad_tab + blocks["fingerprint"]
    if broad_fp:
        subsets["broad_fp"] = broad_fp

    # broad_fp + QM
    if blocks["qm"]:
        broad_fp_qm = broad_fp + blocks["qm"]
        subsets["broad_fp_qm"] = broad_fp_qm

    return subsets


# ─────────────────────────────────────────────────────────────────────
# 2. Bootstrap Confidence Interval
# ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray = None,
                 n_bootstrap: int = 200, ci: float = 0.95,
                 seed: int = 42) -> dict:
    """Bootstrap으로 주요 metric의 CI 계산"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1 - ci) / 2

    metric_samples = {
        "mcc": [], "balanced_accuracy": [], "sensitivity": [], "specificity": [],
    }
    if y_prob is not None:
        metric_samples["roc_auc"] = []
        metric_samples["pr_auc"] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]

        # 최소 양성/음성 확인
        if len(np.unique(yt)) < 2:
            continue

        metrics = compute_all_metrics(yt, yp, y_prob[idx] if y_prob is not None else None)
        for k in metric_samples:
            if k in metrics and not np.isnan(metrics.get(k, np.nan)):
                metric_samples[k].append(metrics[k])

    result = {}
    for k, samples in metric_samples.items():
        if len(samples) < 10:
            continue
        arr = np.array(samples)
        result[f"{k}_mean"] = round(np.mean(arr), 4)
        result[f"{k}_ci_low"] = round(np.percentile(arr, alpha * 100), 4)
        result[f"{k}_ci_high"] = round(np.percentile(arr, (1 - alpha) * 100), 4)
        result[f"{k}_std"] = round(np.std(arr), 4)

    return result


# ─────────────────────────────────────────────────────────────────────
# 3. Applicability Domain (Distance to Train)
# ─────────────────────────────────────────────────────────────────────

def compute_applicability_domain(X_train: np.ndarray, X_test: np.ndarray,
                                  percentile_threshold: float = 95) -> dict:
    """
    Applicability domain proxy:
    - distance to nearest train sample
    - OOD flag (distance > percentile threshold of train distances)
    """
    from sklearn.metrics import pairwise_distances

    # train 내부 거리 분포
    try:
        # 큰 데이터셋은 샘플링
        n_sample = min(2000, len(X_train))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=n_sample, replace=False)
        X_sub = X_train[idx]

        # train 내부 거리
        train_dists = pairwise_distances(X_sub, metric="euclidean")
        np.fill_diagonal(train_dists, np.inf)
        train_nn_dists = train_dists.min(axis=1)
        threshold = np.percentile(train_nn_dists, percentile_threshold)

        # test → train 최근접 거리
        test_dists = pairwise_distances(X_test, X_sub, metric="euclidean")
        test_nn_dists = test_dists.min(axis=1)

        n_ood = (test_nn_dists > threshold).sum()

        return {
            "train_nn_dist_median": round(float(np.median(train_nn_dists)), 4),
            "train_nn_dist_p95": round(float(threshold), 4),
            "test_nn_dist_median": round(float(np.median(test_nn_dists)), 4),
            "test_nn_dist_mean": round(float(np.mean(test_nn_dists)), 4),
            "n_test": int(len(X_test)),
            "n_ood": int(n_ood),
            "ood_fraction": round(float(n_ood / len(X_test)), 4),
            "threshold_percentile": percentile_threshold,
        }
    except Exception as e:
        logger.warning(f"AD computation failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# 4. Calibration Curve / Brier Score
# ─────────────────────────────────────────────────────────────────────

def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10, save_path: str = None) -> dict:
    """Calibration 분석: Brier score + calibration curve"""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(y_true, y_prob)
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
    except Exception:
        return {"brier_score": round(brier, 4)}

    result = {
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
    }

    if save_path and HAS_MPL:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration Curve (Brier={brier:.3f})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Calibration plot saved: {save_path}")

    return result


# ─────────────────────────────────────────────────────────────────────
# 5. 단일 ablation 실험
# ─────────────────────────────────────────────────────────────────────

def run_ablation_single(endpoint: str, subset_name: str,
                        feature_cols: list, model_name: str = "xgb",
                        strategy: str = "none") -> dict:
    """단일 (endpoint, feature_subset, model, strategy) ablation"""
    import joblib
    from utils.cv_utils import run_cv_loop
    from utils.model_utils import final_refit

    result = {
        "endpoint": endpoint,
        "feature_set": subset_name,
        "model": model_name,
        "strategy": strategy,
        "n_features": len(feature_cols),
    }

    try:
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        test_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_test.csv"

        if not train_path.exists():
            result["status"] = "skip_no_data"
            return result

        df_train = safe_read_csv(train_path)
        df_test = safe_read_csv(test_path)

        label_col = cfg.ENDPOINTS[endpoint]["label_col"]
        y_train = df_train[label_col].values.astype(int)
        y_test = df_test[label_col].values.astype(int)

        # feature 필터: 실제 존재하는 컬럼만
        actual_features = [c for c in feature_cols if c in df_train.columns]
        if not actual_features:
            result["status"] = "skip_no_features"
            return result

        result["n_actual_features"] = len(actual_features)

        # CV
        folds = scaffold_kfold(df_train, n_splits=cfg.CV_FOLDS, seed=cfg.RANDOM_SEED)
        model_builder = get_model_builder(model_name)
        param_dist = get_param_dist(model_name)

        alert_cols = [c for c in actual_features if "alert" in c.lower()]
        _, cat_cols = separate_num_cat(df_train, actual_features)
        resample_fn = get_resample_fn(strategy, alert_cols=alert_cols,
                                       cat_cols=cat_cols, seed=cfg.RANDOM_SEED)

        fp_cols = [c for c in actual_features if c.startswith("fp_")]
        non_fp = [c for c in actual_features if not c.startswith("fp_")]

        cv_results = run_cv_loop(
            df_train=df_train,
            feature_cols=actual_features,
            label_col=label_col,
            model_builder=model_builder,
            param_dist=param_dist,
            folds=folds,
            resample_fn=resample_fn,
            fp_select_k=min(cfg.FP_SELECT_K, len(fp_cols)) if fp_cols else 0,
            n_iter=min(cfg.N_RANDOM_SEARCH, 20),
            seed=cfg.RANDOM_SEED,
            endpoint=endpoint,
        )

        valid = [r for r in cv_results if "error" not in r]
        if not valid:
            result["status"] = "all_folds_failed"
            return result

        cv_df = pd.DataFrame(valid)
        for m in ["mcc", "balanced_accuracy", "roc_auc", "pr_auc"]:
            if m in cv_df.columns:
                result[f"cv_{m}_mean"] = round(cv_df[m].mean(), 4)
                result[f"cv_{m}_std"] = round(cv_df[m].std(), 4)

        # Final refit + test
        best_params = valid[0].get("best_params", {})
        refit = final_refit(
            df_train[actual_features], y_train, model_builder,
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

        # Bootstrap CI
        ci = bootstrap_ci(y_test, y_pred, y_prob, n_bootstrap=200)
        for k, v in ci.items():
            result[f"ci_{k}"] = v

        # Calibration
        cal = compute_calibration(y_test, y_prob)
        for k, v in cal.items():
            result[f"cal_{k}"] = v

        # Applicability domain
        try:
            X_train_num = df_train[final_features].select_dtypes(include=[np.number]).values
            X_test_num = df_test[final_features].select_dtypes(include=[np.number]).values
            if X_train_num.shape[1] > 0:
                ad = compute_applicability_domain(X_train_num, X_test_num)
                for k, v in ad.items():
                    result[f"ad_{k}"] = v
        except Exception as e:
            logger.warning(f"AD failed for {endpoint}/{subset_name}: {e}")

        result["status"] = "completed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Ablation failed: {endpoint}/{subset_name}/{model_name}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────
# 6. 종합 도표 생성
# ─────────────────────────────────────────────────────────────────────

def plot_ablation_summary(df: pd.DataFrame, save_dir: Path):
    """논문 그림용 ablation 종합 도표"""
    if not HAS_MPL or df.empty:
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # MCC 비교 barplot (endpoint별)
    metric_col = "test_mcc"
    if metric_col not in df.columns:
        return

    endpoints = df["endpoint"].unique()
    for ep in endpoints:
        ep_df = df[df["endpoint"] == ep].copy()
        if ep_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(ep_df))
        labels = ep_df["feature_set"].values
        values = ep_df[metric_col].values

        # CI bars
        ci_low = ep_df.get("ci_mcc_ci_low", pd.Series(values)).values
        ci_high = ep_df.get("ci_mcc_ci_high", pd.Series(values)).values
        yerr_low = values - ci_low
        yerr_high = ci_high - values
        yerr = np.array([np.maximum(yerr_low, 0), np.maximum(yerr_high, 0)])

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(x)]
        ax.bar(x, values, color=colors, yerr=yerr, capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Test MCC")
        ax.set_title(f"{ep.upper()} — Feature Set Ablation")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"{ep}_ablation_barplot.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {ep}_ablation_barplot.png")

    # 전체 heatmap (endpoint × feature_set → MCC)
    try:
        pivot = df.pivot_table(index="endpoint", columns="feature_set",
                               values=metric_col, aggfunc="first")
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            fig, ax = plt.subplots(figsize=(8, 3))
            im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=10)

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

            plt.colorbar(im, ax=ax, label="Test MCC")
            ax.set_title("Ablation Heatmap (Test MCC)")
            plt.tight_layout()
            plt.savefig(save_dir / "ablation_heatmap.png", dpi=200, bbox_inches="tight")
            plt.close()
            logger.info("Saved: ablation_heatmap.png")
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")


# ─────────────────────────────────────────────────────────────────────
# 7. 메인 실행
# ─────────────────────────────────────────────────────────────────────

def run_ablation():
    """전체 ablation study 실행"""
    logger.info("=" * 60)
    logger.info("Step 5: Ablation / Robustness Study")
    logger.info("=" * 60)

    ablation_dir = cfg.SUMMARY_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for endpoint in cfg.ENDPOINTS:
        logger.info(f"\n--- {endpoint} ---")

        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        if not train_path.exists():
            logger.warning(f"  broadfp not found — skip")
            continue

        df_train = safe_read_csv(train_path)
        subsets = get_feature_subsets(df_train, endpoint)
        logger.info(f"  Feature subsets: {list(subsets.keys())}")

        for subset_name, features in subsets.items():
            for model_name in ["xgb", "logreg"]:
                logger.info(f"  Running: {subset_name} / {model_name}")
                result = run_ablation_single(
                    endpoint, subset_name, features,
                    model_name=model_name, strategy="none"
                )
                all_results.append(result)
                status = result.get("status", "?")
                mcc = result.get("test_mcc", "?")
                logger.info(f"    → {status}, MCC={mcc}")

    # 결과 저장
    df_results = pd.DataFrame(all_results)

    ablation_path = ablation_dir / "ablation_summary.csv"
    df_results.to_csv(ablation_path, index=False)
    logger.info(f"\nAblation summary saved: {ablation_path}")

    # CI summary
    ci_cols = [c for c in df_results.columns if c.startswith("ci_")]
    if ci_cols:
        id_cols = ["endpoint", "feature_set", "model"]
        ci_df = df_results[id_cols + ci_cols].copy()
        ci_path = ablation_dir / "robustness_ci_summary.csv"
        ci_df.to_csv(ci_path, index=False)
        logger.info(f"CI summary saved: {ci_path}")

    # Calibration summary
    cal_cols = [c for c in df_results.columns if c.startswith("cal_")]
    if cal_cols:
        cal_df = df_results[["endpoint", "feature_set", "model"] + cal_cols].copy()
        cal_path = ablation_dir / "calibration_summary.csv"
        cal_df.to_csv(cal_path, index=False)
        logger.info(f"Calibration summary saved: {cal_path}")

    # AD report
    ad_cols = [c for c in df_results.columns if c.startswith("ad_")]
    if ad_cols:
        ad_df = df_results[["endpoint", "feature_set", "model"] + ad_cols].copy()
        ad_path = ablation_dir / "applicability_domain_report.csv"
        ad_df.to_csv(ad_path, index=False)
        logger.info(f"AD report saved: {ad_path}")

    # 도표
    plot_ablation_summary(df_results, ablation_dir)

    logger.info(f"\nAblation study complete. Output: {ablation_dir}")


if __name__ == "__main__":
    run_ablation()
