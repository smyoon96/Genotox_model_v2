"""utils/viz_utils.py — 기본 시각화 유틸리티"""
import sys, logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("viz_utils")


def save_all_plots(y_true, y_pred, y_prob, feat_imp: Dict,
                   out_dir: Path, prefix: str = ""):
    """혼동행렬 / ROC / PR / Feature importance 기본 플롯 저장."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            confusion_matrix, ConfusionMatrixDisplay,
            roc_curve, auc, precision_recall_curve,
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"]).plot(ax=axes[0])
        axes[0].set_title("Confusion Matrix")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        axes[1].plot([0,1],[0,1],"k--")
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC Curve"); axes[1].legend()

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        axes[2].plot(rec, prec)
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
        axes[2].set_title("PR Curve")

        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_overview.png", dpi=120, bbox_inches="tight")
        plt.close("all")

    except Exception as e:
        logger.warning(f"save_all_plots failed: {e}")
