"""
chemical_space.py — 화학공간 분석 모듈
======================================
Chemical space 시각화 및 분석:
  - t-SNE / PCA 기반 화학공간 맵
  - Train vs Test 분포 비교
  - Positive vs Negative 분리 시각화
  - Scaffold 클러스터링 시각화
  - Applicability domain (AD) 시각화
  - 물리화학적 성질 분포 비교
  - QM descriptor 공간 분석
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
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# ═══════════════════════════════════════════════════════════════════════
# 1. 화학공간 t-SNE / PCA 맵
# ═══════════════════════════════════════════════════════════════════════

def plot_chemical_space_tsne(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_names: List[str] = None,
    title: str = "",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    seed: int = 42,
):
    """
    t-SNE 기반 화학공간 2D 맵:
      - Train positive / negative
      - Test positive / negative (다른 마커)
      - AD 경계 표시
    """
    if not HAS_VIZ:
        return
    logger.info("Computing t-SNE chemical space map...")

    # 결합 + 스케일링
    X_all = np.vstack([X_train, X_test])
    n_train = len(X_train)

    # NaN/Inf 처리
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X_all) // 4),
                random_state=seed, n_iter=1000)
    X_emb = tsne.fit_transform(X_scaled)

    X_train_emb = X_emb[:n_train]
    X_test_emb = X_emb[n_train:]

    # ── 시각화: 2x2 grid ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0) Train/Test 분리
    ax = axes[0, 0]
    ax.scatter(X_train_emb[:, 0], X_train_emb[:, 1],
               c="#90CAF9", s=10, alpha=0.4, label="Train", edgecolors="none")
    ax.scatter(X_test_emb[:, 0], X_test_emb[:, 1],
               c="#EF5350", s=14, alpha=0.6, label="Test", marker="^", edgecolors="none")
    ax.set_title("Train vs Test", fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # (0,1) Label 분리 (전체)
    ax = axes[0, 1]
    y_all = np.concatenate([y_train, y_test])
    neg_mask = y_all == 0
    pos_mask = y_all == 1
    ax.scatter(X_emb[neg_mask, 0], X_emb[neg_mask, 1],
               c="#81C784", s=10, alpha=0.4, label="Negative", edgecolors="none")
    ax.scatter(X_emb[pos_mask, 0], X_emb[pos_mask, 1],
               c="#E53935", s=14, alpha=0.6, label="Positive", edgecolors="none")
    ax.set_title("Positive vs Negative", fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # (1,0) Train label
    ax = axes[1, 0]
    tn_mask = y_train == 0
    tp_mask = y_train == 1
    ax.scatter(X_train_emb[tn_mask, 0], X_train_emb[tn_mask, 1],
               c="#A5D6A7", s=10, alpha=0.4, label="Train Neg", edgecolors="none")
    ax.scatter(X_train_emb[tp_mask, 0], X_train_emb[tp_mask, 1],
               c="#C62828", s=14, alpha=0.6, label="Train Pos", edgecolors="none")
    ax.set_title("Train: Label distribution", fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # (1,1) Test label + misclassification
    ax = axes[1, 1]
    test_neg = y_test == 0
    test_pos = y_test == 1
    ax.scatter(X_test_emb[test_neg, 0], X_test_emb[test_neg, 1],
               c="#A5D6A7", s=14, alpha=0.5, label="Test Neg", marker="^", edgecolors="none")
    ax.scatter(X_test_emb[test_pos, 0], X_test_emb[test_pos, 1],
               c="#C62828", s=18, alpha=0.7, label="Test Pos", marker="^", edgecolors="none")
    ax.set_title("Test: Label distribution", fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.suptitle(title or "Chemical Space (t-SNE)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_chemical_space_pca(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_names: List[str] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """PCA 기반 화학공간 맵 + 분산 설명력 barplot"""
    if not HAS_VIZ:
        return
    logger.info("Computing PCA chemical space map...")

    X_all = np.nan_to_num(np.vstack([X_train, X_test]), nan=0, posinf=0, neginf=0)
    n_train = len(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=min(10, X_all.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) PC1 vs PC2 — Train/Test
    ax = axes[0]
    ax.scatter(X_pca[:n_train, 0], X_pca[:n_train, 1],
               c="#90CAF9", s=10, alpha=0.4, label="Train", edgecolors="none")
    ax.scatter(X_pca[n_train:, 0], X_pca[n_train:, 1],
               c="#EF5350", s=14, alpha=0.6, label="Test", marker="^", edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA: Train vs Test", fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)

    # (1) PC1 vs PC2 — Label
    ax = axes[1]
    y_all = np.concatenate([y_train, y_test])
    for label, color, name in [(0, "#81C784", "Negative"), (1, "#E53935", "Positive")]:
        mask = y_all == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, s=10, alpha=0.5, label=name, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA: Positive vs Negative", fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)

    # (2) Variance explained
    ax = axes[2]
    n_comp = len(pca.explained_variance_ratio_)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
           color="#1976D2", alpha=0.7, label="Individual")
    ax.plot(range(1, n_comp + 1), cumvar, "r-o", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Variance", fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(y=80, color="gray", ls="--", lw=0.8, alpha=0.5)

    fig.suptitle(title or "Chemical Space (PCA)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 2. 물리화학적 / QM 성질 분포 비교
# ═══════════════════════════════════════════════════════════════════════

def plot_property_distributions(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    properties: List[str],
    label_col: str = "label",
    title: str = "",
    save_path: Optional[str] = None,
    ncols: int = 4,
):
    """
    물리화학적/QM 성질의 Train vs Test, Positive vs Negative 분포를
    histogram으로 한 번에 비교.
    """
    if not HAS_VIZ:
        return

    available = [p for p in properties if p in df_train.columns and p in df_test.columns]
    if not available:
        logger.warning("No properties available for distribution plot")
        return

    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, prop in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        train_vals = df_train[prop].dropna()
        test_vals = df_test[prop].dropna()

        # combined range
        all_vals = pd.concat([train_vals, test_vals])
        bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 40)

        ax.hist(train_vals, bins=bins, alpha=0.5, color="#1976D2",
                density=True, label="Train")
        ax.hist(test_vals, bins=bins, alpha=0.5, color="#F57C00",
                density=True, label="Test")

        # Positive 오버레이 (train만)
        if label_col in df_train.columns:
            pos_vals = df_train.loc[df_train[label_col] == 1, prop].dropna()
            if len(pos_vals) > 5:
                ax.hist(pos_vals, bins=bins, alpha=0.3, color="#D32F2F",
                        density=True, label="Positive", histtype="step", lw=2)

        ax.set_title(prop.replace("qm_", ""), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    # 빈 subplot 숨기기
    for idx in range(len(available), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title or "Property Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 3. QM Descriptor 상관 히트맵
# ═══════════════════════════════════════════════════════════════════════

def plot_qm_correlation_heatmap(
    df: pd.DataFrame,
    prefix: str = "qm_",
    label_col: str = "label",
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """
    QM descriptor 간 상관관계 + label과의 상관관계 히트맵.
    """
    if not HAS_VIZ:
        return

    qm_cols = [c for c in df.columns if c.startswith(prefix)]
    if not qm_cols:
        logger.warning("No QM columns for heatmap")
        return

    # label과의 상관이 높은 순으로 top_n
    if label_col in df.columns:
        corr_with_label = df[qm_cols].corrwith(df[label_col]).abs()
        top_cols = corr_with_label.nlargest(top_n).index.tolist()
    else:
        top_cols = qm_cols[:top_n]

    if len(top_cols) < 2:
        return

    corr = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(top_cols) * 0.5),
                                     max(6, len(top_cols) * 0.4)))

    cmap = LinearSegmentedColormap.from_list("corr", ["#1565C0", "#FFFFFF", "#C62828"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    short_labels = [c.replace(prefix, "").replace("gasteiger_", "gast_") for c in top_cols]
    ax.set_xticks(range(len(top_cols)))
    ax.set_yticks(range(len(top_cols)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    # 수치 표시 (작은 행렬에서만)
    if len(top_cols) <= 15:
        for i in range(len(top_cols)):
            for j in range(len(top_cols)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title(f"QM Descriptor Correlation (top {len(top_cols)} by |corr with label|)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 4. Applicability Domain 시각화
# ═══════════════════════════════════════════════════════════════════════

def plot_applicability_domain(
    X_train: np.ndarray, X_test: np.ndarray,
    y_pred: np.ndarray, y_true: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    AD 시각화: test 샘플의 train nearest-neighbor 거리 vs 예측 확신도.
    OOD 샘플과 오분류 패턴을 보여준다.
    """
    if not HAS_VIZ:
        return
    from sklearn.metrics import pairwise_distances

    logger.info("Computing applicability domain visualization...")

    X_train_clean = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test_clean = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_clean)
    X_te_s = scaler.transform(X_test_clean)

    # 샘플링 (계산 비용)
    n_sub = min(2000, len(X_tr_s))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr_s), n_sub, replace=False)
    X_tr_sub = X_tr_s[idx]

    # test → train 최근접 거리
    dists = pairwise_distances(X_te_s, X_tr_sub, metric="euclidean")
    nn_dist = dists.min(axis=1)

    # train 내부 거리 (threshold용)
    tr_dists = pairwise_distances(X_tr_sub, metric="euclidean")
    np.fill_diagonal(tr_dists, np.inf)
    tr_nn = tr_dists.min(axis=1)
    ad_threshold = np.percentile(tr_nn, 95)

    correct = y_pred == y_true
    ood = nn_dist > ad_threshold

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (0) 거리 분포
    ax = axes[0]
    ax.hist(tr_nn, bins=50, alpha=0.6, color="#1976D2", density=True, label="Train NN dist")
    ax.hist(nn_dist, bins=50, alpha=0.6, color="#F57C00", density=True, label="Test NN dist")
    ax.axvline(ad_threshold, color="#D32F2F", ls="--", lw=2, label=f"AD threshold (95%)")
    ax.set_xlabel("Nearest Neighbor Distance")
    ax.set_ylabel("Density")
    ax.set_title("AD: Distance Distribution", fontweight="bold")
    ax.legend(fontsize=8)

    # (1) 거리 vs 오분류
    ax = axes[1]
    ax.scatter(nn_dist[correct], np.zeros_like(nn_dist[correct]) + 0.1,
               c="#4CAF50", s=12, alpha=0.4, label="Correct")
    ax.scatter(nn_dist[~correct], np.zeros_like(nn_dist[~correct]) - 0.1,
               c="#F44336", s=20, alpha=0.7, label="Misclassified", marker="x")
    ax.axvline(ad_threshold, color="gray", ls="--", lw=1)
    ax.set_xlabel("Distance to Training Set")
    ax.set_title("AD: Misclassification Pattern", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yticks([])

    # (2) AD summary
    ax = axes[2]
    ax.axis("off")
    n_total = len(nn_dist)
    n_ood = ood.sum()
    n_mis_in = (~correct & ~ood).sum()
    n_mis_out = (~correct & ood).sum()
    n_cor_in = (correct & ~ood).sum()
    n_cor_out = (correct & ood).sum()

    summary_data = [
        ["", "In AD", "Out of AD", "Total"],
        ["Correct", f"{n_cor_in}", f"{n_cor_out}", f"{correct.sum()}"],
        ["Misclassified", f"{n_mis_in}", f"{n_mis_out}", f"{(~correct).sum()}"],
        ["Total", f"{n_total - n_ood}", f"{n_ood}", f"{n_total}"],
        ["", "", "", ""],
        ["OOD fraction", f"{n_ood/n_total*100:.1f}%", "", ""],
        ["Mis. rate (in AD)", f"{n_mis_in/(n_total-n_ood)*100:.1f}%" if n_total > n_ood else "N/A", "", ""],
        ["Mis. rate (out AD)", f"{n_mis_out/n_ood*100:.1f}%" if n_ood > 0 else "N/A", "", ""],
    ]

    table = ax.table(cellText=summary_data, loc="center",
                     cellLoc="center", colWidths=[0.32, 0.22, 0.22, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        if r == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(fontweight="bold")
    ax.set_title("AD Summary", fontweight="bold", pad=15)

    fig.suptitle(title or "Applicability Domain Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 5. QM descriptor 공간 시각화 (HOMO-LUMO plot 등)
# ═══════════════════════════════════════════════════════════════════════

def plot_qm_space(
    df: pd.DataFrame,
    label_col: str = "label",
    prefix: str = "qm_",
    save_path: Optional[str] = None,
):
    """
    QM 전자적 기술자 공간 시각화:
    - HOMO_proxy vs LUMO_proxy
    - gap_proxy vs electrophilicity_proxy
    - charge_range vs chemical_softness
    """
    if not HAS_VIZ:
        return

    pairs = [
        (f"{prefix}HOMO_proxy", f"{prefix}LUMO_proxy", "HOMO proxy", "LUMO proxy"),
        (f"{prefix}gap_proxy", f"{prefix}electrophilicity_proxy", "Gap proxy", "Electrophilicity"),
        (f"{prefix}gasteiger_charge_range", f"{prefix}chemical_softness_proxy", "Charge range", "Softness"),
        (f"{prefix}gasteiger_max_charge", f"{prefix}gasteiger_min_charge", "Max charge", "Min charge"),
    ]

    available_pairs = [(x, y, xl, yl) for x, y, xl, yl in pairs
                       if x in df.columns and y in df.columns]

    if not available_pairs:
        logger.warning("No QM pair columns available")
        return

    fig, axes = plt.subplots(1, len(available_pairs),
                              figsize=(len(available_pairs) * 4.5, 4.5))
    if len(available_pairs) == 1:
        axes = [axes]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, available_pairs):
        for label, color, name, marker, size in [
            (0, "#4CAF50", "Negative", "o", 10),
            (1, "#E53935", "Positive", "^", 16),
        ]:
            if label_col in df.columns:
                mask = df[label_col] == label
                xv = df.loc[mask, xcol].values
                yv = df.loc[mask, ycol].values
            else:
                xv = df[xcol].values
                yv = df[ycol].values

            # NaN 제거
            valid = np.isfinite(xv) & np.isfinite(yv)
            ax.scatter(xv[valid], yv[valid], c=color, s=size, alpha=0.4,
                       label=name, marker=marker, edgecolors="none")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle("QM Descriptor Space", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
