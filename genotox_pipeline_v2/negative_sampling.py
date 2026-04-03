"""
negative_sampling.py — 음성 대표 샘플링 (Auto-Optimized)
=========================================================================
3가지 방법 × 다수 비율(1:1~1:5)을 CV로 평가하여 최적 조합을 자동 선택.

방법:
  1. Descriptor (KMeans Medoid)  — 물리화학 descriptor 공간 클러스터링
  2. FG-Stratified              — 작용기 프로파일 분포 보존 층화추출
  3. Combined                   — Descriptor + FG 하이브리드

Usage:
    python negative_sampling.py [--data-dir ./data]
"""

import sys, logging, argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR, GLOBAL_SEED, FG_SMARTS, ALERT_FAMILIES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

RATIO_CANDIDATES = [1, 2, 3, 5]
METHOD_CANDIDATES = ["descriptor", "fg", "combined"]


# ═══════════════════════════════════════════════════════
#  Descriptor 계산
# ═══════════════════════════════════════════════════════

def compute_descriptors(df, smi_col="SMILES"):
    desc_data = []
    for _, row in df.iterrows():
        smi = str(row[smi_col])
        mol = Chem.MolFromSmiles(smi) if pd.notna(smi) else None
        if mol is None:
            desc_data.append({k: 0.0 for k in [
                "mw","logp","hbd","hba","tpsa","rot_bonds",
                "num_rings","num_arom_rings","fraction_csp3",
                "num_heteroatoms","num_heavy_atoms","mr"]})
            continue
        desc_data.append({
            "mw":Descriptors.MolWt(mol), "logp":Descriptors.MolLogP(mol),
            "hbd":rdMolDescriptors.CalcNumHBD(mol), "hba":rdMolDescriptors.CalcNumHBA(mol),
            "tpsa":Descriptors.TPSA(mol), "rot_bonds":rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_rings":rdMolDescriptors.CalcNumRings(mol),
            "num_arom_rings":rdMolDescriptors.CalcNumAromaticRings(mol),
            "fraction_csp3":rdMolDescriptors.CalcFractionCSP3(mol),
            "num_heteroatoms":rdMolDescriptors.CalcNumHeteroatoms(mol),
            "num_heavy_atoms":mol.GetNumHeavyAtoms(), "mr":Descriptors.MolMR(mol),
        })
    return pd.DataFrame(desc_data, index=df.index)


def compute_fg_profile(df, endpoint, smi_col="SMILES"):
    mols = df[smi_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
    alert_pos = ALERT_FAMILIES.get(endpoint, {}).get("positive", [])
    common_fgs = [k for k in FG_SMARTS if k.startswith("fg_")]
    all_fgs = sorted(set(alert_pos + common_fgs))

    fg_bits = {}
    for fg_name in all_fgs:
        smarts = FG_SMARTS.get(fg_name, "")
        if not smarts: continue
        pat = Chem.MolFromSmarts(smarts)
        if pat is None: continue
        fg_bits[fg_name] = mols.apply(
            lambda m: int(len(m.GetSubstructMatches(pat)) > 0) if m else 0)

    fg_df = pd.DataFrame(fg_bits, index=df.index)
    profiles = []
    for idx in fg_df.index:
        active = [col for col in fg_df.columns if fg_df.loc[idx, col] == 1]
        profiles.append("|".join(sorted(active)) if active else "none")
    fg_df["fg_profile"] = profiles
    return fg_df


# ═══════════════════════════════════════════════════════
#  3가지 샘플링 방법
# ═══════════════════════════════════════════════════════

def descriptor_kmeans_sampling(df, n_select, seed=GLOBAL_SEED):
    neg_df = df[df["label"] == 0].copy()
    if len(neg_df) <= n_select:
        return neg_df.index
    desc = compute_descriptors(neg_df)
    desc_vals = np.nan_to_num(StandardScaler().fit_transform(desc.values.astype(np.float64)), 0)
    km = KMeans(n_clusters=n_select, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(desc_vals)
    centroids = km.cluster_centers_
    selected = []
    for c in range(n_select):
        mask = labels == c
        if not mask.any(): continue
        dists = np.linalg.norm(desc_vals[mask] - centroids[c], axis=1)
        selected.append(neg_df.index[mask][np.argmin(dists)])
    return pd.Index(selected)


def fg_stratified_sampling(df, endpoint, n_select, seed=GLOBAL_SEED):
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0].copy()
    if len(neg_df) <= n_select:
        return neg_df.index

    fg_pos = compute_fg_profile(pos_df, endpoint)
    fg_neg = compute_fg_profile(neg_df, endpoint)
    pos_profile_counts = fg_pos["fg_profile"].value_counts(normalize=True)

    rng = np.random.RandomState(seed)
    selected = []
    for profile, proportion in pos_profile_counts.items():
        n_from = max(1, int(round(proportion * n_select)))
        candidates = neg_df.index[fg_neg["fg_profile"] == profile]
        if len(candidates) == 0: continue
        n_actual = min(n_from, len(candidates))
        selected.extend(rng.choice(candidates, size=n_actual, replace=False).tolist())

    selected = list(dict.fromkeys(selected))
    remaining = n_select - len(selected)
    if remaining > 0:
        unselected = neg_df.drop(index=selected, errors="ignore")
        if len(unselected) > 0:
            n_fill = min(remaining, len(unselected))
            if n_fill >= 2:
                fill_idx = descriptor_kmeans_sampling(unselected.assign(label=0), n_fill, seed)
                selected.extend(fill_idx.tolist())
            else:
                selected.extend(rng.choice(unselected.index, size=n_fill, replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    return pd.Index(selected[:n_select])


def combined_sampling(df, endpoint, n_select, seed=GLOBAL_SEED, fg_weight=0.5):
    n_fg = max(1, int(round(fg_weight * n_select)))
    n_desc = n_select - n_fg
    fg_idx = fg_stratified_sampling(df, endpoint, n_fg, seed)
    neg_remaining = df[(df["label"] == 0) & (~df.index.isin(fg_idx))].copy()
    if len(neg_remaining) > 0 and n_desc > 0:
        desc_idx = descriptor_kmeans_sampling(neg_remaining, min(n_desc, len(neg_remaining)), seed)
    else:
        desc_idx = pd.Index([])
    all_sel = list(dict.fromkeys(fg_idx.tolist() + desc_idx.tolist()))
    return pd.Index(all_sel[:n_select])


SAMPLING_FN = {
    "descriptor": lambda df, ep, n, s: descriptor_kmeans_sampling(df, n, s),
    "fg":         lambda df, ep, n, s: fg_stratified_sampling(df, ep, n, s),
    "combined":   lambda df, ep, n, s: combined_sampling(df, ep, n, s),
}


# ═══════════════════════════════════════════════════════
#  Coverage
# ═══════════════════════════════════════════════════════

def compute_coverage(df_all_neg, df_selected_neg):
    desc_all = compute_descriptors(df_all_neg).values.astype(np.float64)
    desc_sel = compute_descriptors(df_selected_neg).values.astype(np.float64)
    scaler = StandardScaler()
    desc_all_s = np.nan_to_num(scaler.fit_transform(desc_all), 0)
    desc_sel_s = np.nan_to_num(scaler.transform(desc_sel), 0)
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(desc_all_s, desc_sel_s, metric="euclidean")
    min_dists = dists.min(axis=1)
    return {
        "coverage_mean_dist":   round(float(min_dists.mean()), 4),
        "coverage_median_dist": round(float(np.median(min_dists)), 4),
        "coverage_95th_dist":   round(float(np.percentile(min_dists, 95)), 4),
        "coverage_max_dist":    round(float(min_dists.max()), 4),
    }


# ═══════════════════════════════════════════════════════
#  CV 평가
# ═══════════════════════════════════════════════════════

def cv_evaluate(sampled_df, endpoint, n_folds=5, seed=GLOBAL_SEED):
    """5-fold Stratified CV → MCC, BAcc, Sens, Spec"""
    from step4_feature_extraction import extract_fg_features, extract_physchem_features
    from xgboost import XGBClassifier

    fg = extract_fg_features(sampled_df, endpoint)
    ph = extract_physchem_features(sampled_df, endpoint)
    fgp = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
    feat = pd.concat([fgp, ph], axis=1)
    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat = feat.fillna(0)
    X = feat.values.astype(np.float32)
    y = sampled_df["label"].values.astype(int)

    if len(set(y)) < 2 or len(y) < n_folds * 2:
        return {"mcc":0.0,"mcc_std":0.0,"bacc":0.5,"bacc_std":0.0,
                "sens":0.0,"sens_std":0.0,"spec":0.0,"spec_std":0.0}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mccs, baccs, senss, specs = [], [], [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val, y_tr, y_val = X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]
        if len(set(y_tr)) < 2 or len(set(y_val)) < 2: continue
        spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
        mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=seed, n_jobs=-1, verbosity=0)
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_val)
        mccs.append(matthews_corrcoef(y_val, pred))
        baccs.append(balanced_accuracy_score(y_val, pred))
        tp = ((pred==1)&(y_val==1)).sum(); tn = ((pred==0)&(y_val==0)).sum()
        fn = ((pred==0)&(y_val==1)).sum(); fp = ((pred==1)&(y_val==0)).sum()
        senss.append(tp/max(tp+fn,1)); specs.append(tn/max(tn+fp,1))

    if not mccs:
        return {"mcc":0.0,"mcc_std":0.0,"bacc":0.5,"bacc_std":0.0,
                "sens":0.0,"sens_std":0.0,"spec":0.0,"spec_std":0.0}
    return {
        "mcc":round(float(np.mean(mccs)),4), "mcc_std":round(float(np.std(mccs)),4),
        "bacc":round(float(np.mean(baccs)),4), "bacc_std":round(float(np.std(baccs)),4),
        "sens":round(float(np.mean(senss)),4), "sens_std":round(float(np.std(senss)),4),
        "spec":round(float(np.mean(specs)),4), "spec_std":round(float(np.std(specs)),4),
    }


# ═══════════════════════════════════════════════════════
#  메인: 전체 탐색 + 최적 선택
# ═══════════════════════════════════════════════════════

def run_sampling(data_dir=None):
    dp = Path(data_dir) if data_dir else DATA_DIR
    datasets = {
        "invitro": {"file":"invitro.csv","output":"invitro_sampling.csv","endpoint":"invitro"},
        "invivo":  {"file":"invivo.csv", "output":"invivo_sampling.csv", "endpoint":"invivo"},
    }

    all_results = []
    best_configs = {}

    for name, cfg in datasets.items():
        fp = dp / cfg["file"]
        if not fp.exists():
            fp_xlsx = dp / cfg["file"].replace(".csv",".xlsx")
            if fp_xlsx.exists():
                df = pd.read_excel(fp_xlsx, engine="openpyxl")
            else:
                logger.warning(f"  [{name}] not found: {fp}"); continue
        else:
            df = pd.read_csv(fp, encoding="utf-8-sig")

        df["label"] = df["label"].astype(int)
        n_pos = (df["label"]==1).sum()
        n_neg = (df["label"]==0).sum()
        max_ratio = min(max(RATIO_CANDIDATES), n_neg // n_pos)

        logger.info(f"\n{'='*70}")
        logger.info(f"  [{name}] Total={len(df)}, Pos={n_pos}, Neg={n_neg} (1:{n_neg/n_pos:.1f})")
        logger.info(f"  Grid: {len(METHOD_CANDIDATES)} methods × "
                     f"{len([r for r in RATIO_CANDIDATES if r<=max_ratio])} ratios")
        logger.info(f"{'='*70}")

        ep_results = []
        for ratio in RATIO_CANDIDATES:
            if ratio > max_ratio:
                logger.info(f"  ratio=1:{ratio} — skipped (max=1:{max_ratio})")
                continue
            n_select = n_pos * ratio
            for method in METHOD_CANDIDATES:
                neg_idx = SAMPLING_FN[method](df, cfg["endpoint"], n_select, GLOBAL_SEED)
                pos_df = df[df["label"]==1]
                sampled = pd.concat([pos_df, df.loc[neg_idx]]).sort_index().reset_index(drop=True)

                cv = cv_evaluate(sampled, cfg["endpoint"])
                row = {"endpoint":name, "method":method, "ratio":f"1:{ratio}",
                       "ratio_int":ratio, "n_pos":n_pos, "n_neg":len(neg_idx),
                       "n_total":len(sampled), **cv}
                ep_results.append(row)
                all_results.append(row)

                logger.info(f"  1:{ratio}  {method:12s}  n={len(sampled):4d}  "
                            f"MCC={cv['mcc']:.4f}±{cv['mcc_std']:.4f}  "
                            f"BAcc={cv['bacc']:.4f}  Sens={cv['sens']:.4f}  Spec={cv['spec']:.4f}")

        # ── 최적 선택: MCC 기준 (tie-break: BAcc) ──
        ep_df = pd.DataFrame(ep_results)
        best_row = ep_df.sort_values(["mcc","bacc"], ascending=[False,False]).iloc[0]
        best_method = best_row["method"]
        best_ratio = best_row["ratio_int"]
        n_select_best = n_pos * best_ratio

        logger.info(f"\n  ★ [{name}] Best: method={best_method}, ratio=1:{best_ratio}, "
                     f"MCC={best_row['mcc']:.4f}, BAcc={best_row['bacc']:.4f}")

        # ── 최종 샘플링 & 저장 ──
        neg_idx_best = SAMPLING_FN[best_method](df, cfg["endpoint"], n_select_best, GLOBAL_SEED)
        pos_df = df[df["label"]==1]
        neg_best = df.loc[neg_idx_best]
        final = pd.concat([pos_df, neg_best]).sort_index()

        coverage = compute_coverage(df[df["label"]==0], neg_best)

        output_cols = ["No","SMILES","label"]
        for c in output_cols:
            if c not in final.columns and c=="No": final["No"] = final.index
        out_path = dp / cfg["output"]
        final[output_cols].to_csv(out_path, index=False, encoding="utf-8-sig")

        logger.info(f"  Saved: {out_path}  (pos={pos_df['label'].sum()}, neg={len(neg_best)}, "
                     f"total={len(final)})")
        logger.info(f"  Coverage: mean={coverage['coverage_mean_dist']:.4f}, "
                     f"95th={coverage['coverage_95th_dist']:.4f}")

        best_configs[name] = {
            "method":best_method, "ratio":f"1:{best_ratio}",
            "mcc":best_row["mcc"], "bacc":best_row["bacc"],
            "sens":best_row["sens"], "spec":best_row["spec"],
            "n_pos":n_pos, "n_neg":len(neg_best), "n_total":len(final),
            "coverage":coverage, "output":str(out_path),
        }

    # ── 비교 리포트 저장 ──
    report_df = pd.DataFrame(all_results)
    report_df.to_csv(dp / "negative_sampling_cv_report.csv", index=False)
    with open(dp / "negative_sampling_best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n{'='*70}")
    logger.info("  FINAL SUMMARY")
    logger.info(f"{'='*70}")
    for ep, c in best_configs.items():
        logger.info(f"  [{ep}] {c['method']} @ {c['ratio']}  "
                     f"MCC={c['mcc']:.4f}  BAcc={c['bacc']:.4f}  "
                     f"pos={c['n_pos']}  neg={c['n_neg']}  total={c['n_total']}")

    return {"report": report_df, "best": best_configs}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Negative Sampling (Auto-Optimized)")
    p.add_argument("--data-dir", default=None)
    a = p.parse_args()
    run_sampling(data_dir=a.data_dir)
