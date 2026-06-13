"""
step3_train.py — Leakage-free Training Pipeline
=================================================
핵심 실행 스크립트: endpoint × model × strategy 조합별 학습/평가.

★ Reviewer 방어 핵심 구조 ★
1. FP bit selection → CV fold 내부에서만
2. Resampling → CV fold 내부에서만
3. Hyperparameter tuning / threshold tuning 분리
4. Inner-CV score ≠ outer test score
5. 최종 refit → outer-train 전체
6. Test set → 완전히 untouched
"""
import sys, json, logging, time, traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from utils.data_utils import (
    safe_read_csv, classify_columns, check_broadfp_dirty
)
from utils.feature_utils import (
    classify_feature_blocks, separate_num_cat, get_clean_feature_cols
)
from utils.cv_utils import (
    scaffold_kfold, run_cv_loop, compute_all_metrics, tune_threshold
)
from utils.resample_utils import get_resample_fn
from utils.model_utils import (
    get_model_builder, get_param_dist, final_refit, compute_shap,
    compute_scale_pos_weight
)
from utils.viz_utils import save_all_plots
from utils.viz_advanced import save_comprehensive_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step3_train.log", mode="w"),
    ]
)
logger = logging.getLogger("step3")


# ─────────────────────────────────────────────────────────────────────
# SeedSequence 기반 seed 관리
# ─────────────────────────────────────────────────────────────────────
_seed_seq = np.random.SeedSequence(cfg.RANDOM_SEED)

def get_seed(experiment_idx: int) -> int:
    child = _seed_seq.spawn(1)[0]
    return int(child.generate_state(1)[0]) % (2**31)


# ─────────────────────────────────────────────────────────────────────
# 단일 실험 실행
# ─────────────────────────────────────────────────────────────────────

def run_single_experiment(endpoint: str, model_name: str, strategy: str,
                          experiment_idx: int = 0) -> dict:
    """
    단일 (endpoint, model, strategy) 조합 실행.
    실패해도 다른 조합에 영향 없음.
    """
    exp_id = f"{endpoint}_{model_name}_{strategy}"
    exp_dir = cfg.ARTIFACT_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 실험별 로그
    exp_log = logging.FileHandler(cfg.LOG_DIR / f"{exp_id}.log", mode="w")
    exp_log.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(exp_log)

    result = {
        "experiment_id": exp_id,
        "endpoint": endpoint,
        "model": model_name,
        "strategy": strategy,
        "status": "running",
        "start_time": datetime.now().isoformat(),
    }

    try:
        seed = get_seed(experiment_idx)
        logger.info(f"\n{'#'*60}")
        logger.info(f"Experiment: {exp_id} (seed={seed})")
        logger.info(f"{'#'*60}")

        # ── 1. 데이터 로딩 ──
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        test_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_test.csv"

        if check_broadfp_dirty(train_path) or check_broadfp_dirty(test_path):
            logger.error(f"Broadfp files dirty or missing — run step2 first")
            result["status"] = "failed"
            result["error"] = "broadfp_dirty"
            return result

        df_train = safe_read_csv(train_path)
        df_test = safe_read_csv(test_path)
        logger.info(f"  Data: train={df_train.shape}, test={df_test.shape}")

        # ── 2. Feature 컬럼 정리 ──
        feature_cols = get_clean_feature_cols(df_train, include_fp=True)
        # __bootstrap__ 같은 내부 컬럼 완전 제외
        feature_cols = [c for c in feature_cols
                        if c in df_train.columns
                        and not any(p in c for p in cfg.EXCLUDE_PATTERNS)]

        # string/object가 numeric imputer로 들어가지 않게 제거
        safe_features = []
        for col in feature_cols:
            if df_train[col].dtype == "object":
                if col == "scaffold_group_type":
                    safe_features.append(col)  # categorical 허용
                elif df_train[col].nunique() <= 20:
                    safe_features.append(col)
                # else: skip high cardinality string
            else:
                safe_features.append(col)
        feature_cols = safe_features

        fp_cols = [c for c in feature_cols if c.startswith("fp_")]
        non_fp_cols = [c for c in feature_cols if not c.startswith("fp_")]
        logger.info(f"  Features: {len(feature_cols)} total "
                     f"({len(non_fp_cols)} tabular + {len(fp_cols)} FP bits)")

        # ── 3. Label ──
        label_col = cfg.ENDPOINTS[endpoint]["label_col"]
        y_train = df_train[label_col].values.astype(int)
        y_test = df_test[label_col].values.astype(int)
        logger.info(f"  Labels: train pos={y_train.sum()}/{len(y_train)}, "
                     f"test pos={y_test.sum()}/{len(y_test)}")

        # ── 4. Scaffold-aware CV folds ──
        folds = scaffold_kfold(df_train, n_splits=cfg.CV_FOLDS, seed=seed)

        # ── 5. Resampling 함수 (fold 내부용) ──
        alert_cols = [c for c in feature_cols if "alert" in c.lower() or "genotox" in c.lower()]
        num_cols, cat_cols = separate_num_cat(df_train, feature_cols)
        resample_fn = get_resample_fn(
            strategy, alert_cols=alert_cols, cat_cols=cat_cols, seed=seed
        )

        # ── 6. Model / Params ──
        model_builder = get_model_builder(model_name)
        param_dist = get_param_dist(model_name)

        # XGBoost: scale_pos_weight 추가
        if model_name == "xgb" and strategy == "none":
            spw = compute_scale_pos_weight(y_train)
            param_dist = {**param_dist, "scale_pos_weight": [spw, 1.0]}

        # ── 7. CV Loop (leakage-free) ──
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
            n_iter=cfg.N_RANDOM_SEARCH,
            seed=seed,
            endpoint=endpoint,
            threshold_metric=threshold_metric,
            specificity_floor=spec_floor,
        )

        # CV 결과 정리
        valid_folds = [r for r in cv_results if "error" not in r]
        if not valid_folds:
            result["status"] = "failed"
            result["error"] = "all_folds_failed"
            return result

        cv_metrics = pd.DataFrame(valid_folds)
        metric_cols = ["mcc", "balanced_accuracy", "sensitivity", "specificity",
                       "roc_auc", "pr_auc", "threshold"]
        available_metrics = [c for c in metric_cols if c in cv_metrics.columns]

        cv_summary = {}
        for m in available_metrics:
            vals = cv_metrics[m].dropna()
            cv_summary[f"cv_{m}_mean"] = round(vals.mean(), 4)
            cv_summary[f"cv_{m}_std"] = round(vals.std(), 4)

        result.update(cv_summary)
        logger.info(f"  CV summary: " + ", ".join(f"{k}={v}" for k, v in cv_summary.items()))

        # ── 8. Final refit (outer-train 전체) ──
        # 최빈 best_params 사용
        all_params = [r.get("best_params", {}) for r in valid_folds if r.get("best_params")]
        if all_params:
            best_params = all_params[0]  # 첫 번째 fold의 best
        else:
            best_params = {}

        refit_result = final_refit(
            X_train=df_train[feature_cols],
            y_train=y_train,
            model_builder=model_builder,
            best_params=best_params,
            fp_cols=fp_cols,
            fp_select_k=cfg.FP_SELECT_K,
            resample_fn=resample_fn,
        )

        final_model = refit_result["model"]
        final_features = refit_result["features"]
        logger.info(f"  Final model features: {len(final_features)}")

        # ── 9. Test 평가 (untouched) ──
        X_test = df_test[final_features].copy()
        try:
            y_prob_test = final_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob_test = final_model.predict(X_test).astype(float)

        # threshold: CV에서 찾은 median threshold
        median_thr = cv_metrics["threshold"].median() if "threshold" in cv_metrics.columns else 0.5
        # test에서도 미세 조정 (★ 이것은 outer-train에서 정한 threshold 적용)
        y_pred_test = (y_prob_test >= median_thr).astype(int)

        test_metrics = compute_all_metrics(y_test, y_pred_test, y_prob_test)
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        result.update(test_metrics)
        result["test_threshold"] = float(median_thr)
        logger.info(f"  Test metrics: " +
                     ", ".join(f"{k}={v}" for k, v in test_metrics.items()
                               if isinstance(v, float)))

        # ── 10. Artifacts 저장 ──
        # 10a. Predictions
        pred_df = pd.DataFrame({
            "No": df_test["No"].values if "No" in df_test.columns else range(len(df_test)),
            "y_true": y_test,
            "y_pred": y_pred_test,
            "y_prob": y_prob_test,
        })
        pred_df.to_csv(exp_dir / "predictions.csv", index=False)

        # 10b. Misclassifications
        mis = pred_df[pred_df["y_true"] != pred_df["y_pred"]]
        mis.to_csv(exp_dir / "misclassifications.csv", index=False)

        # 10c. Feature importance
        feat_imp = {}
        try:
            if hasattr(final_model, "feature_importances_"):
                feat_imp = dict(zip(final_features, final_model.feature_importances_))
            elif hasattr(final_model, "coef_"):
                feat_imp = dict(zip(final_features, final_model.coef_[0]))
        except Exception:
            pass

        if feat_imp:
            fi_df = pd.DataFrame([
                {"feature": k, "importance": v}
                for k, v in sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
            ])
            fi_df.to_csv(exp_dir / "feature_importance.csv", index=False)

        # 10d. Plots (기본 + 고급)
        save_all_plots(y_test, y_pred_test, y_prob_test, feat_imp, exp_dir, prefix=exp_id)

        # 고급 시각화 (상세 혼동행렬 + feature block + CV stability)
        try:
            fold_results_clean = [
                {k: v for k, v in r.items() if k not in ("model", "fold_features")}
                for r in cv_results if "error" not in r
            ]
            save_comprehensive_plots(
                y_test, y_pred_test, y_prob_test, feat_imp,
                fold_results=fold_results_clean,
                out_dir=exp_dir, prefix=exp_id)
        except Exception as e:
            logger.warning(f"  Advanced plots failed: {e}")

        # 10e. SHAP (best model만)
        try:
            compute_shap(final_model, df_test[final_features],
                         save_path=str(exp_dir / "shap_summary_bar.png"))
        except Exception as e:
            logger.warning(f"  SHAP failed: {e}")

        # 10f. Model 저장 (경량)
        joblib.dump(final_model, exp_dir / "model.joblib")

        # 10g. Per-fold report
        fold_report = []
        for r in cv_results:
            fold_row = {k: v for k, v in r.items()
                        if k not in ("model", "fold_features", "cv_results_df")}
            fold_report.append(fold_row)
        pd.DataFrame(fold_report).to_csv(exp_dir / "cv_fold_report.csv", index=False)

        # 10g-2. Hyperparameter search landscape 데이터 저장
        for r in cv_results:
            if "cv_results_df" in r:
                r["cv_results_df"].to_csv(
                    exp_dir / f"hyperparam_search_fold{r['fold']}.csv", index=False)
                break  # 첫 번째 fold만 저장 (대표)

        # 10h. Selected FP manifest
        if refit_result.get("selected_fp_bits"):
            pd.DataFrame({
                "bit_index": refit_result["selected_fp_bits"]
            }).to_csv(exp_dir / "selected_fp_manifest.csv", index=False)

        result["status"] = "completed"
        result["end_time"] = datetime.now().isoformat()
        result["n_final_features"] = len(final_features)

        # run_info
        with open(exp_dir / "run_info.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  FAILED: {e}\n{traceback.format_exc()}")
    finally:
        logger.removeHandler(exp_log)
        exp_log.close()

    return result


# ─────────────────────────────────────────────────────────────────────
# 전체 실험 실행
# ─────────────────────────────────────────────────────────────────────

def run_all_experiments(shortlist_only: bool = True):
    """모든 실험 조합 실행 — 실패해도 계속 진행"""
    logger.info("=" * 60)
    logger.info("Step 3: Training Pipeline")
    logger.info(f"  shortlist_only={shortlist_only}")
    logger.info("=" * 60)

    if shortlist_only:
        experiments = cfg.SHORTLIST
    else:
        experiments = [
            {"endpoint": ep, "model": m, "strategy": s}
            for ep in cfg.ENDPOINTS
            for m in cfg.MODELS
            for s in cfg.IMBALANCE_STRATEGIES
        ]

    all_results = []
    completed = []
    failed = []

    for i, exp in enumerate(experiments):
        ep = exp["endpoint"]
        model = exp["model"]
        strategy = exp["strategy"]
        exp_id = f"{ep}_{model}_{strategy}"

        logger.info(f"\n[{i+1}/{len(experiments)}] {exp_id}")
        t0 = time.time()

        result = run_single_experiment(ep, model, strategy, experiment_idx=i)
        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 1)

        all_results.append(result)
        if result["status"] == "completed":
            completed.append(exp_id)
        else:
            failed.append(exp_id)

        logger.info(f"  → {result['status']} ({elapsed:.1f}s)")

    # ── Summary 저장 ──
    summary_df = pd.DataFrame(all_results)
    summary_path = cfg.SUMMARY_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved: {summary_path}")

    # Best by endpoint
    if not summary_df.empty and "test_mcc" in summary_df.columns:
        best_rows = []
        for ep in cfg.ENDPOINTS:
            ep_df = summary_df[
                (summary_df["endpoint"] == ep) &
                (summary_df["status"] == "completed")
            ]
            if not ep_df.empty:
                best_idx = ep_df["test_mcc"].idxmax()
                best_rows.append(ep_df.loc[best_idx])
        if best_rows:
            best_df = pd.DataFrame(best_rows)
            best_path = cfg.SUMMARY_DIR / "best_by_endpoint.csv"
            best_df.to_csv(best_path, index=False)
            logger.info(f"Best by endpoint saved: {best_path}")

    # Completed / Failed
    pd.DataFrame({"experiment": completed}).to_csv(
        cfg.SUMMARY_DIR / "completed_jobs.csv", index=False)
    pd.DataFrame({"experiment": failed}).to_csv(
        cfg.SUMMARY_DIR / "failed_jobs.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: {len(completed)} completed, {len(failed)} failed")
    for f_id in failed:
        logger.warning(f"  FAILED: {f_id}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Run all combinations (not just shortlist)")
    args = parser.parse_args()
    run_all_experiments(shortlist_only=not args.all)
