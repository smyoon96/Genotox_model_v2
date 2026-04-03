"""
step7_dashboard.py — 최종 Summary Dashboard 생성
=================================================
모든 실험 결과를 종합하여 논문/보고용 도표와 요약 테이블을 생성한다.

출력:
  - dashboard/overview_table.csv
  - dashboard/endpoint_comparison.png
  - dashboard/strategy_comparison.png
  - dashboard/best_models_gallery/ (best model 시각화 모음)
"""
import sys, logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step7_dashboard.log", mode="w"),
    ]
)
logger = logging.getLogger("step7")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_all_results() -> pd.DataFrame:
    """모든 결과 CSV를 통합"""
    frames = []

    # step3 summary
    s3 = cfg.SUMMARY_DIR / "summary_metrics.csv"
    if s3.exists():
        df = pd.read_csv(s3, low_memory=False)
        df["source"] = "training"
        frames.append(df)

    # step5 ablation
    s5 = cfg.SUMMARY_DIR / "ablation" / "ablation_summary.csv"
    if s5.exists():
        df = pd.read_csv(s5, low_memory=False)
        df["source"] = "ablation"
        frames.append(df)

    # step6 imbalance
    s6 = cfg.SUMMARY_DIR / "imbalance" / "strategy_comparison_summary.csv"
    if s6.exists():
        df = pd.read_csv(s6, low_memory=False)
        df["source"] = "imbalance"
        frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.DataFrame()


def make_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """논문용 overview table (completed만)"""
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return completed

    display_cols = [
        "endpoint", "model", "strategy", "source",
        "test_mcc", "test_balanced_accuracy", "test_roc_auc", "test_pr_auc",
        "test_sensitivity", "test_specificity",
        "cv_mcc_mean", "cv_mcc_std",
        "n_features", "test_threshold",
    ]
    available = [c for c in display_cols if c in completed.columns]

    overview = completed[available].copy()
    # 정렬: endpoint → test_mcc 내림차순
    if "test_mcc" in overview.columns:
        overview = overview.sort_values(["endpoint", "test_mcc"], ascending=[True, False])

    return overview


def plot_endpoint_comparison(df: pd.DataFrame, save_dir: Path):
    """Endpoint별 best model 성능 비교"""
    if not HAS_MPL or df.empty or "test_mcc" not in df.columns:
        return

    completed = df[df["status"] == "completed"]
    metrics = ["test_mcc", "test_balanced_accuracy", "test_sensitivity", "test_specificity"]
    available = [m for m in metrics if m in completed.columns]

    if not available:
        return

    # endpoint별 best MCC 행
    best_rows = []
    for ep in cfg.ENDPOINTS:
        ep_df = completed[completed["endpoint"] == ep]
        if not ep_df.empty:
            best_idx = ep_df["test_mcc"].idxmax()
            row = ep_df.loc[best_idx].copy()
            best_rows.append(row)

    if not best_rows:
        return

    best_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best_df))
    width = 0.18
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    for i, metric in enumerate(available):
        offset = (i - len(available) / 2 + 0.5) * width
        values = best_df[metric].values
        label = metric.replace("test_", "").upper()
        ax.bar(x + offset, values, width, label=label, color=colors[i % len(colors)], alpha=0.85)

    labels = []
    for _, row in best_df.iterrows():
        model = row.get("model", "?")
        strategy = row.get("strategy", "?")
        ep = row.get("endpoint", "?")
        labels.append(f"{ep}\n({model}/{strategy})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Best Model per Endpoint — Key Metrics")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "endpoint_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: endpoint_comparison.png")


def plot_strategy_heatmap(df: pd.DataFrame, save_dir: Path):
    """Strategy × Endpoint heatmap"""
    if not HAS_MPL or df.empty or "test_mcc" not in df.columns:
        return

    completed = df[df["status"] == "completed"]
    try:
        pivot = completed.pivot_table(
            index="endpoint", columns="strategy",
            values="test_mcc", aggfunc="max"
        )
    except Exception:
        return

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > 0.4 else "black")

    plt.colorbar(im, ax=ax, label="Test MCC", shrink=0.8)
    ax.set_title("Strategy × Endpoint — Best Test MCC")
    plt.tight_layout()
    plt.savefig(save_dir / "strategy_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: strategy_heatmap.png")


def collect_best_gallery(save_dir: Path):
    """Best model 시각화 모음"""
    gallery_dir = save_dir / "best_models_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    best_path = cfg.SUMMARY_DIR / "best_by_endpoint.csv"
    if not best_path.exists():
        logger.warning("best_by_endpoint.csv not found")
        return

    best_df = pd.read_csv(best_path, low_memory=False)

    import shutil
    for _, row in best_df.iterrows():
        exp_id = row.get("experiment_id", "")
        if not exp_id:
            continue
        exp_dir = cfg.ARTIFACT_DIR / exp_id
        if not exp_dir.is_dir():
            continue

        ep = row.get("endpoint", "unknown")
        target_dir = gallery_dir / ep
        target_dir.mkdir(parents=True, exist_ok=True)

        for png in exp_dir.glob("*.png"):
            shutil.copy2(png, target_dir / png.name)
        for csv_f in exp_dir.glob("*.csv"):
            shutil.copy2(csv_f, target_dir / csv_f.name)

        logger.info(f"  Gallery: {ep} ← {exp_id}")


def run_dashboard():
    """dashboard 생성"""
    logger.info("=" * 60)
    logger.info("Step 7: Summary Dashboard")
    logger.info("=" * 60)

    dash_dir = cfg.SUMMARY_DIR / "dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)

    # 전체 결과 로딩
    df = load_all_results()
    if df.empty:
        logger.warning("No results found — run training steps first")
        return

    logger.info(f"Loaded {len(df)} experiment results")

    # Overview table
    overview = make_overview_table(df)
    if not overview.empty:
        overview_path = dash_dir / "overview_table.csv"
        overview.to_csv(overview_path, index=False)
        logger.info(f"Overview table saved: {overview_path} ({len(overview)} rows)")

        # 콘솔 출력
        logger.info("\n" + "=" * 80)
        logger.info("OVERVIEW TABLE")
        logger.info("=" * 80)
        display_cols = ["endpoint", "model", "strategy", "source",
                        "test_mcc", "test_balanced_accuracy", "test_sensitivity",
                        "test_specificity"]
        avail = [c for c in display_cols if c in overview.columns]
        for _, row in overview.iterrows():
            parts = [f"{c}={row.get(c, '?')}" for c in avail]
            logger.info("  " + " | ".join(parts))

    # Plots
    plot_endpoint_comparison(df, dash_dir)
    plot_strategy_heatmap(df, dash_dir)

    # Best gallery
    collect_best_gallery(dash_dir)

    # Final best summary
    if not df.empty and "test_mcc" in df.columns:
        completed = df[df["status"] == "completed"]
        logger.info("\n" + "=" * 80)
        logger.info("BEST MODEL PER ENDPOINT")
        logger.info("=" * 80)
        for ep in cfg.ENDPOINTS:
            ep_df = completed[completed["endpoint"] == ep]
            if ep_df.empty:
                logger.info(f"  {ep}: no completed experiments")
                continue
            best_idx = ep_df["test_mcc"].idxmax()
            best = ep_df.loc[best_idx]
            model = best.get("model", "?")
            strategy = best.get("strategy", "?")
            mcc = best.get("test_mcc", "?")
            ba = best.get("test_balanced_accuracy", "?")
            sens = best.get("test_sensitivity", "?")
            spec = best.get("test_specificity", "?")
            logger.info(f"  {ep}: {model}/{strategy} "
                         f"| MCC={mcc} BA={ba} Sens={sens} Spec={spec}")

    logger.info(f"\nDashboard complete. Output: {dash_dir}")


if __name__ == "__main__":
    run_dashboard()
