"""
viz_advanced.py — 종합 시각화 모듈
===================================
논문/보고서급 시각화:
  1. 상세 혼동행렬 (count + rate + 지표 표)
  2. 다중 모델 ROC/PR overlay
  3. Metric radar chart
  4. Feature 블록별 기여도 분석
  5. Hyperparameter search landscape
  6. CV fold 안정성 boxplot
  7. Threshold 최적화 상세 분석
  8. Endpoint 종합 비교 패널
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score,
        matthews_corrcoef, balanced_accuracy_score
    )
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


PALETTE = {
    "blue": "#1976D2", "green": "#388E3C", "orange": "#F57C00",
    "red": "#D32F2F", "purple": "#7B1FA2", "teal": "#00796B",
    "pink": "#C2185B", "amber": "#FFA000", "gray": "#616161",
}


def _setup_style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "figure.dpi": 150,
        "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15,
    })


# ═══════════════════════════════════════════════════════════════════════
# 1. 상세 혼동행렬
# ═══════════════════════════════════════════════════════════════════════

def plot_detailed_confusion_matrix(
    y_true, y_pred, y_prob=None,
    title: str = "", save_path: Optional[str] = None,
):
    """4칸 혼동행렬 + count/rate + 하단 지표 요약 표"""
    if not HAS_MPL:
        return
    _setup_style()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    fig = plt.figure(figsize=(8, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.3], hspace=0.35)

    ax_cm = fig.add_subplot(gs[0])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("cm", ["#FFFFFF", "#1565C0"])
    ax_cm.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    labels_desc = ["Negative (0)", "Positive (1)"]
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(labels_desc, fontsize=10)
    ax_cm.set_yticklabels(labels_desc, fontsize=10)
    ax_cm.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax_cm.set_ylabel("Actual", fontsize=11, fontweight="bold")

    cells = [
        [f"TN\n{tn}\n({tn/total*100:.1f}%)", f"FP\n{fp}\n({fp/total*100:.1f}%)"],
        [f"FN\n{fn}\n({fn/total*100:.1f}%)", f"TP\n{tp}\n({tp/total*100:.1f}%)"],
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax_cm.text(j, i, cells[i][j], ha="center", va="center",
                       fontsize=12, fontweight="bold", color=color)

    ax_cm.set_title(title or "Detailed Confusion Matrix", fontsize=13, fontweight="bold")

    # 지표 표
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    mcc = matthews_corrcoef(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    data = [
        ["MCC", f"{mcc:.4f}", "Balanced Acc", f"{ba:.4f}"],
        ["Sensitivity (Recall)", f"{sens:.4f}", "Specificity (TNR)", f"{spec:.4f}"],
        ["PPV (Precision)", f"{ppv:.4f}", "NPV", f"{npv:.4f}"],
        ["Total", f"{total}", "Pos Rate (Actual)", f"{(tp+fn)/total:.3f}"],
    ]
    table = ax_tbl.table(cellText=data, loc="center", cellLoc="center",
                         colWidths=[0.25, 0.13, 0.25, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        if c in (0, 2):
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 2. 다중 모델 ROC / PR Overlay
# ═══════════════════════════════════════════════════════════════════════

def plot_multi_model_roc_pr(
    results: List[Dict], save_path: Optional[str] = None,
    title_prefix: str = "",
):
    """
    results: list of {name, y_true, y_prob, color(optional)}
    → 2-panel ROC + PR overlay
    """
    if not HAS_MPL or not results:
        return
    _setup_style()

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = list(PALETTE.values())

    for i, r in enumerate(results):
        name, yt, yp = r["name"], r["y_true"], r["y_prob"]
        c = r.get("color", colors[i % len(colors)])

        fpr, tpr, _ = roc_curve(yt, yp)
        ax_roc.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUC={auc(fpr,tpr):.3f})")

        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax_pr.plot(rec, prec, lw=2, color=c, label=f"{name} (AP={ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"{title_prefix} ROC Curves", fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=8)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{title_prefix} PR Curves", fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 3. Metric Radar Chart
# ═══════════════════════════════════════════════════════════════════════

def plot_metric_radar(
    results: List[Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "",
):
    """
    Spider/radar chart — 여러 모델의 metric 프로필 비교.
    results: list of dict with 'name' and metric keys.
    """
    if not HAS_MPL or not results:
        return
    _setup_style()

    if metrics is None:
        metrics = ["mcc", "balanced_accuracy", "sensitivity",
                    "specificity", "roc_auc", "pr_auc"]

    avail = [m for m in metrics if m in results[0]]
    if len(avail) < 3:
        return

    n = len(avail)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = list(PALETTE.values())

    for i, r in enumerate(results):
        values = [max(0, r.get(m, 0)) for m in avail]
        values += values[:1]
        c = colors[i % len(colors)]
        ax.plot(angles, values, "o-", lw=2, color=c, label=r["name"], markersize=5)
        ax.fill(angles, values, alpha=0.1, color=c)

    ax.set_xticks(angles[:-1])
    labels = [m.replace("_", " ").title() for m in avail]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Model Metric Comparison", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 4. Feature 블록별 기여도 분석
# ═══════════════════════════════════════════════════════════════════════

def plot_feature_block_importance(
    importances: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "",
):
    """
    Feature importance를 블록별로 집계하여 stacked bar + 개별 top feature 표시.
    블록: physchem, fg_present, fg_count, rule, fingerprint, qm, other
    """
    if not HAS_MPL or not importances:
        return
    _setup_style()

    # 블록 분류
    blocks = {}
    for feat, imp in importances.items():
        fl = feat.lower()
        if fl.startswith("fp_"):
            blk = "Fingerprint"
        elif fl.startswith("qm_") or fl.startswith("ext_qm_"):
            blk = "QM / Electronic"
        elif fl.startswith("fg_") and "present" in fl:
            blk = "FG Present"
        elif fl.startswith("fg_") and "count" in fl:
            blk = "FG Count"
        elif any(k in fl for k in ["alert", "rule", "genotox", "nitro", "epoxide",
                                     "amine", "halogen", "hydrazine"]):
            blk = "Alert / Rule"
        elif any(k in fl for k in ["mw", "logp", "tpsa", "rot_bonds", "fraction_csp3",
                                     "hba", "hbd", "ring_count"]):
            blk = "Physicochemical"
        else:
            blk = "Other"

        if blk not in blocks:
            blocks[blk] = {"total": 0, "features": {}}
        blocks[blk]["total"] += abs(imp)
        blocks[blk]["features"][feat] = abs(imp)

    fig, (ax_block, ax_top) = plt.subplots(1, 2, figsize=(14, 6),
                                             gridspec_kw={"width_ratios": [1, 1.3]})

    # 왼쪽: 블록별 총 importance
    block_colors = {
        "Physicochemical": "#1976D2", "FG Present": "#388E3C", "FG Count": "#4CAF50",
        "Alert / Rule": "#F57C00", "Fingerprint": "#7B1FA2",
        "QM / Electronic": "#00796B", "Other": "#9E9E9E",
    }
    sorted_blocks = sorted(blocks.items(), key=lambda x: x[1]["total"], reverse=True)
    names = [b[0] for b in sorted_blocks]
    totals = [b[1]["total"] for b in sorted_blocks]
    colors = [block_colors.get(n, "#9E9E9E") for n in names]
    n_feats = [len(b[1]["features"]) for b in sorted_blocks]

    y_pos = range(len(names))
    bars = ax_block.barh(y_pos, totals, color=colors, alpha=0.85)
    ax_block.set_yticks(y_pos)
    ax_block.set_yticklabels([f"{n} ({nf})" for n, nf in zip(names, n_feats)], fontsize=9)
    ax_block.invert_yaxis()
    ax_block.set_xlabel("Total |Importance|")
    ax_block.set_title("Feature Block Contribution", fontweight="bold")

    # 오른쪽: Top 25 개별 feature
    all_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:25]
    feat_names, feat_vals = zip(*all_feats) if all_feats else ([], [])

    feat_colors = []
    for fn in feat_names:
        fl = fn.lower()
        if fl.startswith("fp_"):
            feat_colors.append(block_colors["Fingerprint"])
        elif fl.startswith("qm_") or fl.startswith("ext_qm_"):
            feat_colors.append(block_colors["QM / Electronic"])
        elif fl.startswith("fg_"):
            feat_colors.append(block_colors["FG Present"])
        elif any(k in fl for k in ["alert", "rule", "genotox"]):
            feat_colors.append(block_colors["Alert / Rule"])
        elif any(k in fl for k in ["mw", "logp", "tpsa", "rot"]):
            feat_colors.append(block_colors["Physicochemical"])
        else:
            feat_colors.append(block_colors["Other"])

    y_pos2 = range(len(feat_names))
    ax_top.barh(y_pos2, [abs(v) for v in feat_vals], color=feat_colors, alpha=0.85)
    ax_top.set_yticks(y_pos2)
    short_names = [fn.replace("qm_", "").replace("gasteiger_", "gast_")[:30]
                    for fn in feat_names]
    ax_top.set_yticklabels(short_names, fontsize=7.5)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("|Importance|")
    ax_top.set_title("Top 25 Individual Features", fontweight="bold")

    fig.suptitle(title or "Feature Importance Analysis", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 5. Hyperparameter Search Landscape
# ═══════════════════════════════════════════════════════════════════════

def plot_hyperparam_landscape(
    cv_results_df: pd.DataFrame,
    param_x: str, param_y: str,
    score_col: str = "mean_test_score",
    save_path: Optional[str] = None,
    title: str = "",
):
    """
    RandomizedSearchCV의 cv_results_ DataFrame에서
    2D hyperparameter landscape 시각화.
    """
    if not HAS_MPL or cv_results_df.empty:
        return
    _setup_style()

    px = f"param_{param_x}" if f"param_{param_x}" in cv_results_df.columns else param_x
    py = f"param_{param_y}" if f"param_{param_y}" in cv_results_df.columns else param_y

    if px not in cv_results_df.columns or py not in cv_results_df.columns:
        logger.warning(f"Param columns not found: {px}, {py}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (0) scatter: param_x vs score
    ax = axes[0]
    x_vals = pd.to_numeric(cv_results_df[px], errors="coerce")
    scores = cv_results_df[score_col]
    ax.scatter(x_vals, scores, c="#1976D2", alpha=0.6, s=30, edgecolors="white", lw=0.5)
    ax.set_xlabel(param_x)
    ax.set_ylabel("CV Score (MCC)")
    ax.set_title(f"{param_x} vs Score", fontweight="bold")

    # (1) scatter: param_y vs score
    ax = axes[1]
    y_vals = pd.to_numeric(cv_results_df[py], errors="coerce")
    ax.scatter(y_vals, scores, c="#F57C00", alpha=0.6, s=30, edgecolors="white", lw=0.5)
    ax.set_xlabel(param_y)
    ax.set_ylabel("CV Score (MCC)")
    ax.set_title(f"{param_y} vs Score", fontweight="bold")

    # (2) 2D heatmap
    ax = axes[2]
    try:
        pivot = cv_results_df.pivot_table(
            index=py, columns=px, values=score_col, aggfunc="mean")
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=7, rotation=45)
        ax.set_yticklabels([f"{v}" for v in pivot.index], fontsize=7)
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Score")
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data\nfor heatmap",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("2D Landscape", fontweight="bold")

    fig.suptitle(title or "Hyperparameter Search Landscape", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 6. CV Fold 안정성 Boxplot
# ═══════════════════════════════════════════════════════════════════════

def plot_cv_fold_stability(
    fold_results: List[Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "",
):
    """CV fold별 metric 분포를 boxplot + strip으로 표시."""
    if not HAS_MPL or not fold_results:
        return
    _setup_style()

    if metrics is None:
        metrics = ["mcc", "balanced_accuracy", "sensitivity", "specificity",
                    "roc_auc", "pr_auc"]

    df = pd.DataFrame(fold_results)
    avail = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not avail:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(avail) * 1.5), 5))
    colors = ["#1976D2", "#388E3C", "#F57C00", "#D32F2F", "#7B1FA2", "#00796B"]

    positions = range(len(avail))
    bp = ax.boxplot([df[m].dropna().values for m in avail],
                    positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # strip (개별 fold 점)
    for i, m in enumerate(avail):
        vals = df[m].dropna().values
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   c=colors[i % len(colors)], s=40, alpha=0.7,
                   edgecolors="white", lw=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("_", " ").title() for m in avail],
                        fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title or "CV Fold Stability", fontsize=13, fontweight="bold")

    # mean 표시
    for i, m in enumerate(avail):
        mean_v = df[m].mean()
        std_v = df[m].std()
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"μ={mean_v:.3f}\nσ={std_v:.3f}",
                ha="center", va="top", fontsize=7, color=colors[i % len(colors)])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 7. Endpoint 종합 비교 패널
# ═══════════════════════════════════════════════════════════════════════

def plot_endpoint_summary_panel(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    summary_metrics.csv에서 endpoint × model × strategy 전체 결과를
    하나의 종합 패널로 시각화.
    """
    if not HAS_MPL or summary_df.empty:
        return
    _setup_style()

    completed = summary_df[summary_df["status"] == "completed"].copy()
    if completed.empty:
        return

    endpoints = completed["endpoint"].unique()
    n_ep = len(endpoints)

    fig = plt.figure(figsize=(18, 5 * n_ep))
    outer_gs = gridspec.GridSpec(n_ep, 1, hspace=0.4)

    metric_cols = ["test_mcc", "test_balanced_accuracy", "test_sensitivity",
                   "test_specificity", "test_roc_auc", "test_pr_auc"]
    avail_metrics = [m for m in metric_cols if m in completed.columns]

    for ep_idx, ep in enumerate(sorted(endpoints)):
        ep_df = completed[completed["endpoint"] == ep].copy()

        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[ep_idx],
                                                      width_ratios=[1.5, 1, 1])

        # (a) Grouped bar: metric별 각 실험
        ax = fig.add_subplot(inner_gs[0])
        n_exp = len(ep_df)
        x = np.arange(n_exp)
        width = 0.12
        colors_m = ["#1976D2", "#388E3C", "#F57C00", "#D32F2F", "#7B1FA2", "#00796B"]

        for j, m in enumerate(avail_metrics[:6]):
            offset = (j - len(avail_metrics[:6]) / 2 + 0.5) * width
            vals = ep_df[m].values
            ax.bar(x + offset, vals, width, color=colors_m[j],
                   alpha=0.8, label=m.replace("test_", ""))

        labels = [f"{r.get('model','?')}\n{r.get('strategy','?')}"
                  for _, r in ep_df.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title(f"{ep.upper()} — All Experiments", fontweight="bold")
        if ep_idx == 0:
            ax.legend(fontsize=6, ncol=3, loc="upper right")
        ax.set_ylim(0, 1.05)

        # (b) Best model confusion matrix
        ax_cm = fig.add_subplot(inner_gs[1])
        if "test_mcc" in ep_df.columns:
            best = ep_df.loc[ep_df["test_mcc"].idxmax()]
            cm_vals = np.array([
                [int(best.get("test_tn", 0)), int(best.get("test_fp", 0))],
                [int(best.get("test_fn", 0)), int(best.get("test_tp", 0))],
            ])
            cm_norm = cm_vals / cm_vals.sum(axis=1, keepdims=True).clip(1)
            cmap = LinearSegmentedColormap.from_list("cm", ["#FFFFFF", "#1565C0"])
            ax_cm.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    c = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax_cm.text(j, i, f"{cm_vals[i,j]}", ha="center", va="center",
                               fontsize=12, fontweight="bold", color=c)
            ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Neg", "Pos"], fontsize=9)
            ax_cm.set_yticklabels(["Neg", "Pos"], fontsize=9)
            bm = best.get("model", "?")
            bs = best.get("strategy", "?")
            ax_cm.set_title(f"Best: {bm}/{bs}\nMCC={best.get('test_mcc',0):.3f}",
                            fontweight="bold", fontsize=10)

        # (c) CV stability mini-boxplot
        ax_cv = fig.add_subplot(inner_gs[2])
        cv_cols = [c for c in ep_df.columns if c.startswith("cv_") and c.endswith("_mean")]
        if cv_cols:
            cv_vals = ep_df[cv_cols].values.T
            bp = ax_cv.boxplot(cv_vals.tolist(), patch_artist=True)
            short = [c.replace("cv_", "").replace("_mean", "") for c in cv_cols]
            ax_cv.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
            ax_cv.set_title("CV Mean Scores", fontweight="bold", fontsize=10)
            for patch in bp["boxes"]:
                patch.set_facecolor("#E3F2FD")
        else:
            ax_cv.text(0.5, 0.5, "No CV data", ha="center", va="center",
                       transform=ax_cv.transAxes)
            ax_cv.set_title("CV Scores", fontweight="bold", fontsize=10)

    plt.suptitle("Endpoint Summary Panel", fontsize=15, fontweight="bold", y=1.01)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 8. 종합 저장 함수 (기존 save_all_plots 대체)
# ═══════════════════════════════════════════════════════════════════════

def save_comprehensive_plots(
    y_true, y_pred, y_prob,
    feature_importances: dict,
    fold_results: List[Dict] = None,
    out_dir: Path = None,
    prefix: str = "",
):
    """모든 고급 시각화를 한 번에 저장."""
    if not HAS_MPL:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    # 상세 혼동행렬
    plot_detailed_confusion_matrix(
        y_true, y_pred, y_prob,
        title=f"{prefix} Confusion Matrix",
        save_path=str(out_dir / f"{p}detailed_confusion_matrix.png"))

    # Feature block importance
    if feature_importances:
        plot_feature_block_importance(
            feature_importances,
            title=f"{prefix} Feature Analysis",
            save_path=str(out_dir / f"{p}feature_block_importance.png"))

    # CV fold stability
    if fold_results:
        plot_cv_fold_stability(
            fold_results,
            title=f"{prefix} CV Fold Stability",
            save_path=str(out_dir / f"{p}cv_fold_stability.png"))
