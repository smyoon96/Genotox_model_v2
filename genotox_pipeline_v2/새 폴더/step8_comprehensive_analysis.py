"""
step8_comprehensive_analysis.py — 종합 분석 및 시각화
=====================================================
모든 실험 결과를 종합하여 논문급 시각화를 생성한다.

출력:
  analysis/
  ├── confusion_matrices/        # endpoint별 상세 혼동행렬
  ├── model_comparison/          # 다중 모델 ROC/PR overlay + radar
  ├── feature_analysis/          # Feature 블록 기여도 + QM 상관
  ├── chemical_space/            # t-SNE/PCA 화학공간 + AD
  ├── qm_analysis/               # QM descriptor 공간 + 분포
  ├── cv_stability/              # CV fold 안정성
  ├── hyperparam/                # Hyperparameter landscape
  └── summary_tables/            # 종합 결과 표 (CSV + 논문용)
"""
import sys, json, logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from utils.data_utils import safe_read_csv, check_broadfp_dirty, classify_columns
from utils.feature_utils import classify_feature_blocks, get_clean_feature_cols
from utils.cv_utils import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step8_analysis.log", mode="w"),
    ]
)
logger = logging.getLogger("step8")


# ═══════════════════════════════════════════════════════════════════════
# A. 상세 혼동행렬 (모든 실험)
# ═══════════════════════════════════════════════════════════════════════

def generate_confusion_matrices(out_dir: Path):
    """모든 실험의 predictions.csv → 상세 혼동행렬 생성"""
    from utils.viz_advanced import plot_detailed_confusion_matrix
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(cfg.ARTIFACT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        pred_csv = exp_dir / "predictions.csv"
        if not pred_csv.exists():
            continue

        df = pd.read_csv(pred_csv)
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            continue

        name = exp_dir.name
        y_prob = df["y_prob"].values if "y_prob" in df.columns else None

        plot_detailed_confusion_matrix(
            df["y_true"].values, df["y_pred"].values, y_prob,
            title=name.replace("_", " / "),
            save_path=str(cm_dir / f"{name}_cm.png"))
        logger.info(f"  CM: {name}")


# ═══════════════════════════════════════════════════════════════════════
# B. 다중 모델 비교 (ROC/PR overlay + Radar)
# ═══════════════════════════════════════════════════════════════════════

def generate_model_comparisons(out_dir: Path):
    """endpoint별 모든 모델의 ROC/PR overlay + metric radar"""
    from utils.viz_advanced import plot_multi_model_roc_pr, plot_metric_radar
    mc_dir = out_dir / "model_comparison"
    mc_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#1976D2", "#F57C00", "#388E3C", "#D32F2F", "#7B1FA2", "#00796B"]

    for endpoint in cfg.ENDPOINTS:
        # 해당 endpoint의 모든 predictions 수집
        roc_data = []
        radar_data = []

        for exp_dir in sorted(cfg.ARTIFACT_DIR.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith(endpoint):
                continue
            pred_csv = exp_dir / "predictions.csv"
            info_json = exp_dir / "run_info.json"
            if not pred_csv.exists():
                continue

            df = pd.read_csv(pred_csv)
            if "y_true" not in df.columns or "y_prob" not in df.columns:
                continue

            name = exp_dir.name.replace(f"{endpoint}_", "")
            roc_data.append({
                "name": name,
                "y_true": df["y_true"].values,
                "y_prob": df["y_prob"].values,
            })

            # radar용 metrics
            metrics = compute_all_metrics(
                df["y_true"].values,
                df["y_pred"].values if "y_pred" in df.columns else (df["y_prob"].values > 0.5).astype(int),
                df["y_prob"].values)
            metrics["name"] = name
            radar_data.append(metrics)

        if roc_data:
            plot_multi_model_roc_pr(
                roc_data,
                save_path=str(mc_dir / f"{endpoint}_roc_pr_overlay.png"),
                title_prefix=endpoint.upper())
            logger.info(f"  ROC/PR overlay: {endpoint} ({len(roc_data)} models)")

        if radar_data:
            plot_metric_radar(
                radar_data,
                save_path=str(mc_dir / f"{endpoint}_metric_radar.png"),
                title=f"{endpoint.upper()} — Model Comparison")
            logger.info(f"  Radar chart: {endpoint}")


# ═══════════════════════════════════════════════════════════════════════
# C. Feature 블록 기여도 분석
# ═══════════════════════════════════════════════════════════════════════

def generate_feature_analysis(out_dir: Path):
    """Feature importance를 블록별로 분석"""
    from utils.viz_advanced import plot_feature_block_importance
    fa_dir = out_dir / "feature_analysis"
    fa_dir.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(cfg.ARTIFACT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        fi_csv = exp_dir / "feature_importance.csv"
        if not fi_csv.exists():
            continue

        df = pd.read_csv(fi_csv)
        if "feature" not in df.columns or "importance" not in df.columns:
            continue

        importances = dict(zip(df["feature"], df["importance"]))
        name = exp_dir.name

        plot_feature_block_importance(
            importances,
            save_path=str(fa_dir / f"{name}_feature_blocks.png"),
            title=name.replace("_", " / "))
        logger.info(f"  Feature blocks: {name}")


# ═══════════════════════════════════════════════════════════════════════
# D. 화학공간 분석 (t-SNE / PCA / AD)
# ═══════════════════════════════════════════════════════════════════════

def generate_chemical_space_analysis(out_dir: Path):
    """endpoint별 화학공간 시각화"""
    from utils.chemical_space import (
        plot_chemical_space_tsne, plot_chemical_space_pca,
        plot_property_distributions, plot_applicability_domain,
    )
    cs_dir = out_dir / "chemical_space"
    cs_dir.mkdir(parents=True, exist_ok=True)

    for endpoint in cfg.ENDPOINTS:
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        test_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_test.csv"
        if not train_path.exists() or not test_path.exists():
            continue

        logger.info(f"  Chemical space: {endpoint}")
        df_train = safe_read_csv(train_path)
        df_test = safe_read_csv(test_path)

        label_col = cfg.ENDPOINTS[endpoint]["label_col"]
        y_train = df_train[label_col].values.astype(int)
        y_test = df_test[label_col].values.astype(int)

        # numeric feature만 추출
        feature_cols = get_clean_feature_cols(df_train, include_fp=True, include_qm=True)
        num_cols = [c for c in feature_cols
                    if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c])]

        if len(num_cols) < 5:
            logger.warning(f"  Too few numeric features ({len(num_cols)}) — skip")
            continue

        X_train = df_train[num_cols].values
        X_test = df_test[num_cols].values

        # t-SNE (FP bits만 사용하면 더 깔끔)
        fp_cols = [c for c in num_cols if c.startswith("fp_")]
        if len(fp_cols) >= 20:
            plot_chemical_space_tsne(
                df_train[fp_cols].values, df_test[fp_cols].values,
                y_train, y_test,
                title=f"{endpoint.upper()} — FP Chemical Space (t-SNE)",
                save_path=str(cs_dir / f"{endpoint}_tsne_fp.png"))

        # PCA (전체 feature)
        plot_chemical_space_pca(
            X_train, X_test, y_train, y_test,
            title=f"{endpoint.upper()} — Chemical Space (PCA)",
            save_path=str(cs_dir / f"{endpoint}_pca_all.png"))

        # 성질 분포
        prop_list = [p for p in cfg.PROPERTY_VIZ_LIST if p in df_train.columns]
        if prop_list:
            plot_property_distributions(
                df_train, df_test, prop_list,
                label_col=label_col,
                title=f"{endpoint.upper()} — Property Distributions",
                save_path=str(cs_dir / f"{endpoint}_property_dist.png"))

        # Applicability Domain
        # best 실험의 predictions 로드
        best_pred = None
        for exp_dir in cfg.ARTIFACT_DIR.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith(endpoint):
                pred_csv = exp_dir / "predictions.csv"
                if pred_csv.exists():
                    best_pred = pd.read_csv(pred_csv)
                    break

        if best_pred is not None and "y_true" in best_pred.columns:
            plot_applicability_domain(
                X_train, X_test,
                best_pred["y_pred"].values if "y_pred" in best_pred.columns else y_test,
                y_test,
                title=f"{endpoint.upper()} — Applicability Domain",
                save_path=str(cs_dir / f"{endpoint}_ad.png"))


# ═══════════════════════════════════════════════════════════════════════
# E. QM 전자적 기술자 분석
# ═══════════════════════════════════════════════════════════════════════

def generate_qm_analysis(out_dir: Path):
    """QM descriptor 상관관계 + 공간 시각화"""
    from utils.chemical_space import plot_qm_correlation_heatmap, plot_qm_space
    from utils.qm_descriptors import qm_descriptor_summary, QM_RELEVANCE
    qm_dir = out_dir / "qm_analysis"
    qm_dir.mkdir(parents=True, exist_ok=True)

    for endpoint in cfg.ENDPOINTS:
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        if not train_path.exists():
            continue

        df = safe_read_csv(train_path)
        label_col = cfg.ENDPOINTS[endpoint]["label_col"]

        qm_cols = [c for c in df.columns if c.startswith(cfg.QM_PREFIX)]
        if not qm_cols:
            logger.info(f"  [{endpoint}] No QM columns — skip")
            continue

        logger.info(f"  QM analysis: {endpoint} ({len(qm_cols)} descriptors)")

        # QM descriptor 품질 요약
        summary = qm_descriptor_summary(df, prefix=cfg.QM_PREFIX)
        if not summary.empty:
            summary.to_csv(qm_dir / f"{endpoint}_qm_summary.csv", index=False)

        # 상관 히트맵
        plot_qm_correlation_heatmap(
            df, prefix=cfg.QM_PREFIX, label_col=label_col,
            top_n=20,
            save_path=str(qm_dir / f"{endpoint}_qm_corr_heatmap.png"))

        # QM 공간 (HOMO vs LUMO 등)
        plot_qm_space(
            df, label_col=label_col, prefix=cfg.QM_PREFIX,
            save_path=str(qm_dir / f"{endpoint}_qm_space.png"))

        # Endpoint별 QM 관련성 리포트
        if endpoint in QM_RELEVANCE:
            rel = QM_RELEVANCE[endpoint]
            rel_data = {
                "endpoint": endpoint,
                "reason": rel["reason"],
                "high_relevance": rel["high"],
                "medium_relevance": rel["medium"],
            }
            with open(qm_dir / f"{endpoint}_qm_relevance.json", "w") as f:
                json.dump(rel_data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# F. CV 안정성 분석
# ═══════════════════════════════════════════════════════════════════════

def generate_cv_stability_analysis(out_dir: Path):
    """모든 실험의 CV fold 결과를 boxplot으로"""
    from utils.viz_advanced import plot_cv_fold_stability
    cv_dir = out_dir / "cv_stability"
    cv_dir.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(cfg.ARTIFACT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        fold_csv = exp_dir / "cv_fold_report.csv"
        if not fold_csv.exists():
            continue

        df = pd.read_csv(fold_csv, low_memory=False)
        fold_results = df.to_dict("records")
        name = exp_dir.name

        plot_cv_fold_stability(
            fold_results,
            save_path=str(cv_dir / f"{name}_cv_stability.png"),
            title=name.replace("_", " / "))
        logger.info(f"  CV stability: {name}")


# ═══════════════════════════════════════════════════════════════════════
# G. 종합 Endpoint 패널
# ═══════════════════════════════════════════════════════════════════════

def generate_endpoint_panel(out_dir: Path):
    """summary_metrics.csv → 종합 endpoint 비교 패널"""
    from utils.viz_advanced import plot_endpoint_summary_panel
    panel_dir = out_dir / "summary_tables"
    panel_dir.mkdir(parents=True, exist_ok=True)

    summary_path = cfg.SUMMARY_DIR / "summary_metrics.csv"
    if not summary_path.exists():
        logger.warning("summary_metrics.csv not found")
        return

    df = pd.read_csv(summary_path, low_memory=False)
    plot_endpoint_summary_panel(
        df, save_path=str(panel_dir / "endpoint_summary_panel.png"))
    logger.info("  Endpoint summary panel generated")

    # 논문용 정리 표 (LaTeX-friendly)
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return

    paper_cols = [
        "endpoint", "model", "strategy",
        "test_mcc", "test_balanced_accuracy", "test_roc_auc",
        "test_pr_auc", "test_sensitivity", "test_specificity",
        "cv_mcc_mean", "cv_mcc_std", "n_final_features",
    ]
    avail = [c for c in paper_cols if c in completed.columns]
    paper_df = completed[avail].sort_values(["endpoint", "test_mcc"], ascending=[True, False])

    paper_df.to_csv(panel_dir / "paper_results_table.csv", index=False)
    logger.info("  Paper results table saved")

    # Feature block 요약 표
    block_rows = []
    for endpoint in cfg.ENDPOINTS:
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        if not train_path.exists():
            continue
        df_train = pd.read_csv(train_path, nrows=0, low_memory=False)
        col_class = classify_columns(df_train)
        blocks = classify_feature_blocks(col_class["features"])
        row = {"endpoint": endpoint, "total": len(col_class["features"])}
        for blk, cols in blocks.items():
            row[blk] = len(cols)
        block_rows.append(row)

    if block_rows:
        block_df = pd.DataFrame(block_rows)
        block_df.to_csv(panel_dir / "feature_block_counts.csv", index=False)
        logger.info("  Feature block counts saved")


# ═══════════════════════════════════════════════════════════════════════
# G-2. Hyperparameter Landscape 분석
# ═══════════════════════════════════════════════════════════════════════

def generate_hyperparam_analysis(out_dir: Path):
    """모든 실험의 hyperparameter search 결과를 시각화"""
    from utils.viz_advanced import plot_hyperparam_landscape
    hp_dir = out_dir / "hyperparam"
    hp_dir.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(cfg.ARTIFACT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        # 첫 번째 fold의 search CSV
        hp_csv = exp_dir / "hyperparam_search_fold0.csv"
        if not hp_csv.exists():
            continue

        df = pd.read_csv(hp_csv, low_memory=False)
        name = exp_dir.name

        # XGB landscape
        if "param_max_depth" in df.columns and "param_learning_rate" in df.columns:
            plot_hyperparam_landscape(
                df, param_x="max_depth", param_y="learning_rate",
                save_path=str(hp_dir / f"{name}_hp_depth_lr.png"),
                title=f"{name} — Depth vs LR")
            logger.info(f"  Hyperparam: {name} (depth vs lr)")

        if "param_n_estimators" in df.columns and "param_max_depth" in df.columns:
            plot_hyperparam_landscape(
                df, param_x="n_estimators", param_y="max_depth",
                save_path=str(hp_dir / f"{name}_hp_nest_depth.png"),
                title=f"{name} — N_est vs Depth")

        # LogReg landscape
        if "param_C" in df.columns and "param_l1_ratio" in df.columns:
            plot_hyperparam_landscape(
                df, param_x="C", param_y="l1_ratio",
                save_path=str(hp_dir / f"{name}_hp_C_l1.png"),
                title=f"{name} — C vs L1 Ratio")
            logger.info(f"  Hyperparam: {name} (C vs l1_ratio)")


# ═══════════════════════════════════════════════════════════════════════
# H. 메인 실행
# ═══════════════════════════════════════════════════════════════════════

def run_comprehensive_analysis():
    """전체 종합 분석 실행"""
    logger.info("=" * 60)
    logger.info("Step 8: Comprehensive Analysis & Visualization")
    logger.info("=" * 60)

    analysis_dir = cfg.OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        ("Confusion Matrices", generate_confusion_matrices),
        ("Model Comparisons", generate_model_comparisons),
        ("Feature Analysis", generate_feature_analysis),
        ("Chemical Space", generate_chemical_space_analysis),
        ("QM Analysis", generate_qm_analysis),
        ("CV Stability", generate_cv_stability_analysis),
        ("Hyperparam Landscape", generate_hyperparam_analysis),
        ("Endpoint Panel", generate_endpoint_panel),
    ]

    results = {}
    for name, func in steps:
        logger.info(f"\n--- {name} ---")
        try:
            func(analysis_dir)
            results[name] = "ok"
        except Exception as e:
            results[name] = f"failed: {e}"
            logger.error(f"  FAILED: {e}")

    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    for name, status in results.items():
        symbol = "✓" if status == "ok" else "✗"
        logger.info(f"  {symbol} {name}: {status}")
    logger.info(f"\nOutput: {analysis_dir}")

    # 파일 목록
    total_files = 0
    for d in analysis_dir.rglob("*"):
        if d.is_file():
            total_files += 1
    logger.info(f"Total files generated: {total_files}")


if __name__ == "__main__":
    run_comprehensive_analysis()
