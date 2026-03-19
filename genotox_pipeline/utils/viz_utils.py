"""
viz_utils.py — 시각화 유틸리티
==============================
confusion matrix, ROC, PR curve, threshold sweep, feature importance
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
    from sklearn.metrics import (
        confusion_matrix, ConfusionMatrixDisplay,
        roc_curve, auc, precision_recall_curve, average_precision_score,
        matthews_corrcoef, balanced_accuracy_score
    )
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — plots disabled")


def plot_confusion_matrix(y_true, y_pred, title: str = "",
                          save_path: Optional[str] = None):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, title: str = "",
                   save_path: Optional[str] = None):
    if not HAS_MPL:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true, y_prob, title: str = "",
                  save_path: Optional[str] = None):
    if not HAS_MPL:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curve")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(y_true, y_prob, title: str = "",
                         save_path: Optional[str] = None):
    """Threshold vs MCC, balanced accuracy, sensitivity, specificity"""
    if not HAS_MPL:
        return
    thresholds = np.linspace(0.01, 0.99, 100)
    mccs, baccs, sens, specs = [], [], [], []
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        mccs.append(matthews_corrcoef(y_true, preds))
        baccs.append(balanced_accuracy_score(y_true, preds))
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, mccs, label="MCC", lw=2)
    ax.plot(thresholds, baccs, label="Balanced Acc", lw=2)
    ax.plot(thresholds, sens, label="Sensitivity", lw=1.5, ls="--")
    ax.plot(thresholds, specs, label="Specificity", lw=1.5, ls="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title or "Threshold Sweep")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances: dict, top_n: int = 20,
                            title: str = "",
                            save_path: Optional[str] = None):
    """Feature importance bar chart"""
    if not HAS_MPL:
        return
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, values = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(7, max(4, len(names) * 0.3)))
    y_pos = range(len(names))
    ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title or "Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_all_plots(y_true, y_pred, y_prob, feature_importances: dict,
                   out_dir: Path, prefix: str = ""):
    """모든 시각화를 한 번에 저장"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    plot_confusion_matrix(y_true, y_pred,
                          title=f"{prefix} Confusion Matrix",
                          save_path=str(out_dir / f"{p}confusion_matrix.png"))
    if y_prob is not None:
        plot_roc_curve(y_true, y_prob,
                       title=f"{prefix} ROC",
                       save_path=str(out_dir / f"{p}roc_curve.png"))
        plot_pr_curve(y_true, y_prob,
                      title=f"{prefix} PR",
                      save_path=str(out_dir / f"{p}pr_curve.png"))
        plot_threshold_sweep(y_true, y_prob,
                             title=f"{prefix} Threshold Sweep",
                             save_path=str(out_dir / f"{p}threshold_sweep.png"))
    if feature_importances:
        plot_feature_importance(feature_importances,
                                title=f"{prefix} Feature Importance",
                                save_path=str(out_dir / f"{p}feature_importance.png"))
