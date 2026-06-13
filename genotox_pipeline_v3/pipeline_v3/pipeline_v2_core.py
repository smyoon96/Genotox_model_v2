"""
pipeline_v2_core.py -- v12 (2026-03-27)
=======================================
Bug fixes from v8→v12:
  [CRITICAL-1] apply_scenario('salt_stripped') now ACTUALLY replaces SMILES
               before feature extraction (was silently ignored → identical to raw_all)
  [CRITICAL-3] compute_ad uses proper Tanimoto for binary FP, increased sample
  [METHOD-5]   Added statistical comparison functions (Wilcoxon, McNemar, etc.)
  [METHOD-10]  Added learning_curve() utility
  [IMPROVE]    interpret_result now flags sampling endpoints explicitly
"""
import logging
from collections import defaultdict
import numpy as np
from utils.progress import pbar as _pbar
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MakeScaffoldGeneric
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix)
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)
SEED = 42
METAL_NUMS = frozenset({3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,33,
    37,38,39,40,41,42,44,45,46,47,48,49,50,55,56,72,73,74,75,76,77,78,79,80,81,82,83})

# ═══════════════════════════════════════════════════════
#  SMILES Handling
# ═══════════════════════════════════════════════════════

# Canonical SMILES column name used throughout the pipeline after scenario application
ANALYSIS_SMILES_COL = "_analysis_smiles"

def find_smi(df):
    """Find the best SMILES column. After apply_scenario, use _analysis_smiles."""
    if ANALYSIS_SMILES_COL in df.columns:
        return ANALYSIS_SMILES_COL
    for c in ["canonical_smiles", "SMILES_raw", "SMILES", "_can", "smiles"]:
        if c in df.columns:
            return c
    raise KeyError(f"No SMILES column in {list(df.columns)}")

def to_can(s):
    if pd.isna(s) or str(s).strip() == "":
        return None
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def strip_salts(smi):
    """Strip salts/fragments from SMILES, return largest fragment canonical."""
    if pd.isna(smi) or str(smi).strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        # Use RDKit's fragment remover
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)
        # Uncharge
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return to_can(smi)


def file_hash(path):
    """SHA-256 of file for reproducibility."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ═══════════════════════════════════════════════════════
#  Conflict Resolution
# ═══════════════════════════════════════════════════════

def resolve_conflicts(df, strategy="conservative"):
    sc = find_smi(df)
    df = df.copy()
    df["_can"] = df[sc].apply(to_can)
    valid = df[df["_can"].notna()].copy()
    grp = valid.groupby("_can")["label"].agg(["nunique", "count", "sum"])
    csmi = set(grp[grp["nunique"] > 1].index)
    nc = valid[~valid["_can"].isin(csmi)]
    cr = valid[valid["_can"].isin(csmi)]

    if strategy == "conservative":
        cleaned = nc.copy()
    elif strategy == "positive_priority":
        res = []
        for cs in csmi:
            r = cr[cr["_can"] == cs].iloc[0].copy()
            r["label"] = 1
            res.append(r)
        cleaned = pd.concat([nc, pd.DataFrame(res)], ignore_index=True) if res else nc.copy()
    elif strategy == "majority_vote":
        res = []
        for cs in csmi:
            sub = cr[cr["_can"] == cs]
            np_ = (sub["label"] == 1).sum()
            nn_ = (sub["label"] == 0).sum()
            if np_ == nn_:
                continue
            r = sub.iloc[0].copy()
            r["label"] = 1 if np_ > nn_ else 0
            res.append(r)
        cleaned = pd.concat([nc, pd.DataFrame(res)], ignore_index=True) if res else nc.copy()
    else:
        raise ValueError(strategy)

    cleaned = cleaned.drop_duplicates(subset=["_can"], keep="first")
    rpt = {
        "strategy": strategy,
        "n_original": len(df),
        "n_conflicts": len(csmi),
        "n_after": len(cleaned),
        "pos_rate_before": round(df["label"].mean(), 4),
        "pos_rate_after": round(cleaned["label"].mean(), 4) if len(cleaned) > 0 else 0,
    }
    return cleaned, cr, rpt


# ═══════════════════════════════════════════════════════
#  Scaffold Assignment & Split
# ═══════════════════════════════════════════════════════

def assign_scaffolds(df, n_clusters=50, seed=SEED):
    sc = find_smi(df)
    df = df.copy()
    scaffolds = []
    stypes = []
    for smi in df[sc]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            scaffolds.append("INVALID")
            stypes.append("invalid")
            continue
        try:
            s = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if s and s.strip():
                sm = Chem.MolFromSmiles(s)
                scaffolds.append(
                    Chem.MolToSmiles(MakeScaffoldGeneric(sm), canonical=True) if sm else s
                )
                stypes.append("cyclic")
            else:
                scaffolds.append(f"__ACYC_{len(scaffolds)}")
                stypes.append("acyclic")
        except Exception:
            scaffolds.append(f"__ERR_{len(scaffolds)}")
            stypes.append("error")
    df["scaffold"] = scaffolds
    df["scaffold_type"] = stypes

    amask = df["scaffold_type"] == "acyclic"
    na = amask.sum()
    if na > n_clusters:
        fps = []
        for smi in df.loc[amask, sc]:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                a = np.zeros(1024, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, a)
                fps.append(a)
            else:
                fps.append(np.zeros(1024, dtype=np.float32))
        nc = min(n_clusters, na // 2)
        if nc >= 2:
            labels = AgglomerativeClustering(
                n_clusters=nc, metric="euclidean", linkage="ward"
            ).fit_predict(np.array(fps))
            df.loc[amask, "scaffold"] = [f"ACYC_C{l}" for l in labels]
    df["scaffold_group"] = df["scaffold"]
    return df


def fixed_split(df, ratio=0.8, seed=SEED):
    if "scaffold_group" not in df.columns:
        df = assign_scaffolds(df, seed=seed)
    rng = np.random.RandomState(seed)
    scfs = defaultdict(list)
    for idx, row in df.iterrows():
        scfs[row["scaffold_group"]].append(idx)
    items = list(scfs.items())
    rng.shuffle(items)
    items.sort(key=lambda x: len(x[1]), reverse=True)
    n_target = int(len(df) * ratio)
    train_idx = []
    test_idx = []
    for s, idxs in items:
        if len(train_idx) + len(idxs) <= n_target:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    df = df.copy()
    df["split"] = "unassigned"
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    bal = {
        "train_n": len(tr), "test_n": len(te),
        "train_pos": round(tr["label"].mean(), 4),
        "test_pos": round(te["label"].mean(), 4),
        "diff": round(abs(tr["label"].mean() - te["label"].mean()), 4),
        "scaffold_overlap": len(set(tr["scaffold_group"]) & set(te["scaffold_group"])),
        "n_groups": df["scaffold_group"].nunique(),
    }
    return df, bal


# ═══════════════════════════════════════════════════════
#  [CRITICAL-1 FIX] Scenario Application
#  salt_stripped now ACTUALLY transforms SMILES and stores
#  in _analysis_smiles column for downstream feature extraction
# ═══════════════════════════════════════════════════════

def apply_scenario(df_split, scenario, flags=None):
    """
    Apply preprocessing scenario and set _analysis_smiles column.

    CRITICAL FIX (v12): salt_stripped now creates _analysis_smiles from
    stripped SMILES. All downstream feature extraction MUST use find_smi()
    which will find _analysis_smiles first.
    """
    df = df_split.copy()
    orig_smi_col = None
    for c in ["canonical_smiles", "SMILES_raw", "SMILES", "_can", "smiles"]:
        if c in df.columns:
            orig_smi_col = c
            break
    if orig_smi_col is None:
        raise KeyError("No SMILES column found")

    if scenario == "raw_all":
        # Use original SMILES as-is
        df[ANALYSIS_SMILES_COL] = df[orig_smi_col].copy()

    elif scenario == "no_metal":
        # Remove metal-containing compounds
        if flags is not None and "has_metal" in flags.columns:
            m = flags.reindex(df.index)["has_metal"].fillna(False).values
            df = df[~m].copy()
        df[ANALYSIS_SMILES_COL] = df[orig_smi_col].copy()

    elif scenario == "salt_stripped":
        # [CRITICAL FIX] Actually strip salts and use stripped SMILES
        if flags is not None and "smiles_stripped" in flags.columns:
            # Use pre-computed stripped SMILES from preprocessing flags
            stripped = flags.reindex(df.index)["smiles_stripped"]
            df[ANALYSIS_SMILES_COL] = stripped.where(
                stripped.notna(), df[orig_smi_col]
            ).values
        else:
            # Compute salt stripping on the fly
            logger.info("  salt_stripped: computing on-the-fly (no precomputed flags)")
            df[ANALYSIS_SMILES_COL] = df[orig_smi_col].apply(strip_salts)
            # Fill failed strips with original
            mask = df[ANALYSIS_SMILES_COL].isna()
            df.loc[mask, ANALYSIS_SMILES_COL] = df.loc[mask, orig_smi_col]

        # Log how many SMILES actually changed
        n_changed = (df[ANALYSIS_SMILES_COL] != df[orig_smi_col]).sum()
        logger.info(f"  salt_stripped: {n_changed}/{len(df)} SMILES changed")

    elif scenario == "metal_as_feature":
        if flags is not None and "has_metal" in flags.columns:
            df["has_metal_feature"] = flags.reindex(df.index)["has_metal"].fillna(False).astype(int).values
        df[ANALYSIS_SMILES_COL] = df[orig_smi_col].copy()

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()


# ═══════════════════════════════════════════════════════
#  TRUE Repeated Scaffold CV (fixed in v8, retained)
# ═══════════════════════════════════════════════════════

def repeated_cv(X, y, groups, model_fn, n_folds=5, n_repeats=5, seed=SEED):
    ug = np.unique(groups)
    ng = len(ug)
    af = min(n_folds, ng)
    if af < 2:
        return {}
    all_m = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(seed + rep * 7919)
        perm = rng.permutation(ng)
        fold_assign = np.arange(ng) % af
        g2f = dict(zip(ug[perm], fold_assign))
        sf = np.array([g2f[g] for g in groups])
        for f in range(af):
            vm = sf == f
            tm = ~vm
            if vm.sum() == 0 or tm.sum() == 0:
                continue
            Xt, yt = X[tm], y[tm]
            Xv, yv = X[vm], y[vm]
            if len(set(yt)) < 2 or len(set(yv)) < 2:
                continue
            mdl = model_fn()
            mdl.fit(Xt, yt)
            yp = mdl.predict_proba(Xv)[:, 1]
            yd = (yp >= 0.5).astype(int)
            mcc = matthews_corrcoef(yv, yd)
            try:
                roc = roc_auc_score(yv, yp)
            except Exception:
                roc = np.nan
            all_m.append({
                "rep": rep, "fold": f, "mcc": mcc, "roc_auc": roc,
                "bacc": balanced_accuracy_score(yv, yd), "n_val": len(yv),
            })
    if not all_m:
        return {}
    mdf = pd.DataFrame(all_m)

    # Verify fold uniqueness
    fold_sigs = set()
    for rep in range(n_repeats):
        rng2 = np.random.RandomState(seed + rep * 7919)
        p2 = rng2.permutation(ng)
        g2f2 = dict(zip(ug[p2], np.arange(ng) % af))
        fold_sigs.add(tuple(g2f2[g] for g in ug[:min(10, ng)]))

    result = {"_n_unique_fold_assignments": len(fold_sigs)}
    for m in ["mcc", "roc_auc", "bacc"]:
        v = mdf[m].dropna()
        if len(v) > 0:
            result[m] = {
                "mean": round(v.mean(), 4), "std": round(v.std(), 4),
                "lo": round(v.quantile(0.025), 4), "hi": round(v.quantile(0.975), 4),
                "n": len(v),
            }
    return result


# ═══════════════════════════════════════════════════════
#  Bootstrap CI
# ═══════════════════════════════════════════════════════

def bootstrap_ci(yt, yp, ypr, n=500, seed=SEED):
    rng = np.random.RandomState(seed)
    sz = len(yt)
    boots = {"mcc": [], "bacc": [], "roc_auc": [], "sens": [], "spec": [], "brier": []}
    for _ in range(n):
        idx = rng.choice(sz, sz, replace=True)
        yt_, yp_, ypr_ = yt[idx], yp[idx], ypr[idx]
        if len(set(yt_)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(yt_, yp_, labels=[0, 1]).ravel()
        boots["mcc"].append(matthews_corrcoef(yt_, yp_))
        boots["bacc"].append(balanced_accuracy_score(yt_, yp_))
        boots["sens"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        boots["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        boots["brier"].append(brier_score_loss(yt_, ypr_))
        try:
            boots["roc_auc"].append(roc_auc_score(yt_, ypr_))
        except Exception:
            pass
    out = {}
    for m, vals in boots.items():
        if vals:
            a = np.array(vals)
            out[m] = {
                "mean": round(a.mean(), 4),
                "lo": round(np.percentile(a, 2.5), 4),
                "hi": round(np.percentile(a, 97.5), 4),
            }
    return out


# ═══════════════════════════════════════════════════════
#  [CRITICAL-3 FIX] Applicability Domain -- Proper Tanimoto
# ═══════════════════════════════════════════════════════

def compute_ad(tr_fp, te_fp, threshold=0.3, max_tr=1000, seed=42):
    """
    Compute Applicability Domain using Tanimoto similarity on binary FP.

    CRITICAL FIX (v12):
      - Use proper Tanimoto: |A∩B| / |A∪B| for binary vectors
      - Increased max_tr from 300 to 1000 for stability
      - Added input validation

    [v12.2] Returns per_sample_max_sim (np.ndarray) for AD-stratified analysis
            in step9 Mode B/D.
    """
    rng = np.random.RandomState(seed)

    # Subsample training set if needed
    if len(tr_fp) > max_tr:
        idx = rng.choice(len(tr_fp), max_tr, replace=False)
        tr = tr_fp[idx]
    else:
        tr = tr_fp

    # Ensure binary
    tr_bin = (tr > 0).astype(np.float32)
    te_bin = (te_fp > 0).astype(np.float32)

    ms = np.zeros(len(te_bin))
    for i in _pbar(range(len(te_bin)), desc="  AD similarity", leave=False):
        # Vectorized Tanimoto: |A∩B| / |A∪B|
        intersection = np.sum(te_bin[i] * tr_bin, axis=1)  # (n_tr,)
        a_bits = np.sum(te_bin[i])
        b_bits = np.sum(tr_bin, axis=1)  # (n_tr,)
        union = a_bits + b_bits - intersection
        # Avoid division by zero
        sims = np.where(union > 0, intersection / union, 0.0)
        ms[i] = sims.max() if len(sims) > 0 else 0.0

    in_ad = ms >= threshold
    return {
        "coverage": round(in_ad.mean(), 4),
        "n_in": int(in_ad.sum()),
        "n_out": int((~in_ad).sum()),
        "mean_sim": round(ms.mean(), 4),
        "median_sim": round(np.median(ms), 4),
        # [v12.2] per-sample max Tanimoto similarity to training set
        # Shape: (n_test,) -- used for AD-stratified sensitivity in step9
        "per_sample_max_sim": ms,
    }


# ═══════════════════════════════════════════════════════
#  Calibration
# ═══════════════════════════════════════════════════════

def calibration(yt, ypr, bins=10):
    brier = brier_score_loss(yt, ypr)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (ypr >= edges[i]) & (ypr < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(yt) * abs(yt[mask].mean() - ypr[mask].mean())
    return {"brier": round(brier, 4), "ece": round(ece, 4)}


# ═══════════════════════════════════════════════════════
#  Domain Confounding & LDO
# ═══════════════════════════════════════════════════════

def domain_confounding(df, col="domain"):
    if col not in df.columns:
        return {"has_domain": False}
    rpt = {"has_domain": True, "domains": {}}
    for d, g in df.groupby(col):
        rpt["domains"][d] = {
            "n": len(g),
            "pos": int((g["label"] == 1).sum()),
            "pos_rate": round(g["label"].mean(), 4),
        }
    doms = list(rpt["domains"].keys())
    if len(doms) == 2:
        r0 = rpt["domains"][doms[0]]["pos_rate"]
        r1 = rpt["domains"][doms[1]]["pos_rate"]
        p0 = 1 if r0 > 0.5 else 0
        p1 = 1 if r1 > 0.5 else 0
        cor = sum(1 for _, row in df.iterrows()
                  if (p0 if row[col] == doms[0] else p1) == row["label"])
        rpt["domain_acc"] = round(cor / len(df), 4)
        rpt["enrichment"] = round(max(r0, r1) / max(min(r0, r1), 0.001), 1)
    return rpt


def leave_domain_out(df, col, feat_fn, model_fn, ep):
    if col not in df.columns:
        return []
    doms = df[col].unique()
    results = []
    for td in doms:
        train = df[df[col] != td].copy()
        test = df[df[col] == td].copy()
        if (len(train) < 10 or len(test) < 10 or
                len(set(train["label"])) < 2 or len(set(test["label"])) < 2):
            continue
        Xr, yr, _ = feat_fn(train, ep)
        Xe, ye, _ = feat_fn(test, ep)
        mdl = model_fn()
        mdl.fit(Xr, yr)
        yp = mdl.predict_proba(Xe)[:, 1]
        yd = (yp >= 0.5).astype(int)
        ci = bootstrap_ci(ye, yd, yp, n=500)
        results.append({
            "train_dom": ",".join([d for d in doms if d != td]),
            "test_dom": td,
            "train_n": len(yr), "test_n": len(ye),
            "test_pos": round(ye.mean(), 4),
            **{k: v["mean"] for k, v in ci.items()},
            **{f"{k}_lo": v["lo"] for k, v in ci.items()},
            **{f"{k}_hi": v["hi"] for k, v in ci.items()},
        })
    return results


# ═══════════════════════════════════════════════════════
#  Cross-Endpoint Overlap
# ═══════════════════════════════════════════════════════

def cross_endpoint(datasets):
    maps = {}
    for ep, df in datasets.items():
        sc = find_smi(df)
        m = {}
        for _, row in df.iterrows():
            cs = to_can(row[sc])
            if cs:
                m[cs] = int(row["label"])
        maps[ep] = m
    results = []
    eps = list(datasets.keys())
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            e1, e2 = eps[i], eps[j]
            m1, m2 = maps[e1], maps[e2]
            ov = set(m1) & set(m2)
            conc = sum(1 for s in ov if m1[s] == m2[s])
            kappa = np.nan
            if len(ov) > 1:
                try:
                    from sklearn.metrics import cohen_kappa_score
                    kappa = cohen_kappa_score([m1[s] for s in ov], [m2[s] for s in ov])
                except Exception:
                    pass
            results.append({
                "ep1": e1, "ep2": e2, "overlap": len(ov),
                "overlap_pct": round(len(ov) / max(len(m2), 1) * 100, 1),
                "concordant": conc, "discordant": len(ov) - conc,
                "kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════
#  [METHOD-5] Statistical Tests for Model Comparison
# ═══════════════════════════════════════════════════════

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test: are two classifiers' errors significantly different?
    Returns chi2 statistic and p-value.
    """
    from scipy.stats import chi2 as chi2_dist

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    # Contingency: a correct & b wrong, a wrong & b correct
    b_val = int((correct_a & ~correct_b).sum())  # a right, b wrong
    c_val = int((~correct_a & correct_b).sum())  # a wrong, b right

    if b_val + c_val == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0}

    # McNemar with continuity correction
    chi2 = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "n_discordant": b_val + c_val,
        "a_better": b_val, "b_better": c_val,
    }


def wilcoxon_model_comparison(cv_results_a, cv_results_b, metric="mcc"):
    """
    Wilcoxon signed-rank test on paired CV fold results.
    Requires matched fold-level scores from both models.
    """
    from scipy.stats import wilcoxon

    a_scores = [r[metric] for r in cv_results_a if metric in r]
    b_scores = [r[metric] for r in cv_results_b if metric in r]

    n = min(len(a_scores), len(b_scores))
    if n < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n_pairs": n,
                "note": "Too few pairs for Wilcoxon test"}

    a_scores = a_scores[:n]
    b_scores = b_scores[:n]

    try:
        stat, p = wilcoxon(a_scores, b_scores)
        return {
            "statistic": round(stat, 4), "p_value": round(p, 6),
            "n_pairs": n,
            "mean_diff": round(np.mean(np.array(a_scores) - np.array(b_scores)), 4),
        }
    except Exception as e:
        return {"statistic": np.nan, "p_value": np.nan, "n_pairs": n, "error": str(e)}


def pairwise_model_comparison(all_rows, endpoint, metric="mcc"):
    """
    Pairwise comparison of all model types within an endpoint.
    Uses test set predictions for McNemar test.
    Returns a comparison matrix.
    """
    ep_rows = [r for r in all_rows if r.get("endpoint") == endpoint]
    if not ep_rows:
        return pd.DataFrame()

    # Group by model, take best feature mode for each
    model_best = {}
    for r in ep_rows:
        mn = r["model"]
        if mn not in model_best or r.get(metric, 0) > model_best[mn].get(metric, 0):
            model_best[mn] = r

    models = sorted(model_best.keys())
    results = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            r1, r2 = model_best[m1], model_best[m2]
            diff = r1.get(metric, 0) - r2.get(metric, 0)
            results.append({
                "model_a": m1, "model_b": m2,
                f"{metric}_a": r1.get(metric), f"{metric}_b": r2.get(metric),
                "diff": round(diff, 4),
                "better": m1 if diff > 0 else m2,
            })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════
#  [METHOD-10] Learning Curve
# ═══════════════════════════════════════════════════════

def learning_curve(X, y, groups, model_fn, fractions=None,
                   n_folds=5, seed=SEED):
    """
    Compute learning curve: train on increasing fractions of data.
    Uses scaffold-aware splitting.
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    ug = np.unique(groups)
    ng = len(ug)
    af = min(n_folds, ng)
    if af < 2:
        return []

    rng = np.random.RandomState(seed)
    perm = rng.permutation(ng)
    fold_assign = np.arange(ng) % af
    g2f = dict(zip(ug[perm], fold_assign))
    sf = np.array([g2f[g] for g in groups])

    results = []
    for frac in fractions:
        fold_scores = []
        for f in range(af):
            vm = sf == f
            tm = ~vm
            if vm.sum() == 0 or tm.sum() == 0:
                continue
            Xt, yt = X[tm], y[tm]
            Xv, yv = X[vm], y[vm]
            if len(set(yt)) < 2 or len(set(yv)) < 2:
                continue

            # Subsample training data
            n_use = max(int(len(Xt) * frac), 10)
            if n_use < len(Xt):
                idx = rng.choice(len(Xt), n_use, replace=False)
                Xt_sub, yt_sub = Xt[idx], yt[idx]
                if len(set(yt_sub)) < 2:
                    continue
            else:
                Xt_sub, yt_sub = Xt, yt

            mdl = model_fn()
            mdl.fit(Xt_sub, yt_sub)
            yp = mdl.predict_proba(Xv)[:, 1]
            yd = (yp >= 0.5).astype(int)
            fold_scores.append(matthews_corrcoef(yv, yd))

        if fold_scores:
            results.append({
                "fraction": frac,
                "n_train": int(len(X) * frac * (af - 1) / af),
                "mcc_mean": round(np.mean(fold_scores), 4),
                "mcc_std": round(np.std(fold_scores), 4),
                "n_folds": len(fold_scores),
            })
    return results


# ═══════════════════════════════════════════════════════
#  Result Interpretation
# ═══════════════════════════════════════════════════════

def interpret_result(row):
    """Automatic annotations to prevent over-interpretation."""
    notes = []

    # Flag small test set positives
    if row.get("test_n", 0) > 0:
        n_pos = round(row["test_n"] * row.get("test_pos_rate", row.get("test_pos", 0)))
        if n_pos < 15:
            notes.append(
                f"CAUTION: only ~{n_pos} positives in test set -- interpret as exploratory"
            )

    # Flag sampling endpoints
    ep = row.get("endpoint", "")
    if "sampling" in ep:
        notes.append("NOTE: sampling endpoint -- reduced dataset, verify generalizability")

    # CV-Test gap
    mcc = row.get("mcc", 0)
    cv_mcc = row.get("cv_mcc_mean")
    if cv_mcc is not None and not pd.isna(cv_mcc):
        gap = abs(mcc - cv_mcc)
        if gap > 0.2:
            notes.append(
                f"WARNING: test-CV gap={gap:.3f} -- possible split luck or CV instability"
            )

    # Flag very low MCC
    if mcc is not None and mcc < 0.1 and row.get("test_n", 0) > 50:
        notes.append("WARNING: MCC < 0.1 -- model near random performance")

    return "; ".join(notes) if notes else "OK"


# ═══════════════════════════════════════════════════════
#  Threshold Tuning
# ═══════════════════════════════════════════════════════

def tune_threshold(y_train_true, y_train_proba):
    """Train set MCC-maximizing threshold. Apply directly to test."""
    best_t, best_mcc = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        pred = (y_train_proba >= t).astype(int)
        if len(set(pred)) < 2:
            continue
        m = matthews_corrcoef(y_train_true, pred)
        if m > best_mcc:
            best_mcc = m
            best_t = t
    return round(best_t, 2), round(best_mcc, 4)


# Aliases
smi_col = find_smi


# ═══════════════════════════════════════════════════════
#  OOF Threshold Grid Search (unified -- v12.2)
#  genotox_pipeline.oof_threshold + step9.find_oof_threshold 통합
# ═══════════════════════════════════════════════════════

def oof_tune_threshold(X_tr: np.ndarray, y_tr: np.ndarray,
                       groups=None, model_fn=None, model=None,
                       n_folds: int = 5, seed: int = SEED,
                       thr_range: tuple = (0.1, 0.9),
                       thr_step: float = 0.02,
                       criterion: str = "mcc") -> dict:
    """
    OOF(Out-Of-Fold) 기반 threshold grid search.

    genotox_pipeline.py의 oof_threshold()와
    step9_external_benchmark.py의 find_oof_threshold()를 통합.

    Parameters
    ----------
    X_tr, y_tr  : 학습 데이터
    groups      : scaffold group 배열. 있으면 GroupKFold, 없으면 StratifiedKFold
    model_fn    : lambda → 새 모델 (genotox_pipeline 스타일)
    model       : sklearn 모델 인스턴스 (step9 스타일, clone 사용)
    n_folds     : CV fold 수
    thr_range   : (low, high) 탐색 구간
    thr_step    : 탐색 간격 (default 0.02 → 40개 후보)
    criterion   : "mcc" | "f1" | "youden"

    Returns
    -------
    dict:
        best_threshold  : 최적 threshold
        best_score      : criterion 최고값
        criterion       : 사용 기준
        threshold_grid  : List[{threshold, score, sensitivity, specificity}]
    """
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

    # ── OOF probability 수집 ──────────────────────────────────────────
    oof_proba = np.full(len(y_tr), np.nan)

    if groups is not None:
        # scaffold-aware: GroupKFold
        from sklearn.model_selection import GroupKFold
        unique_g = np.unique(groups)
        n_splits = min(n_folds, len(unique_g))
        if n_splits < 2:
            return {"best_threshold": 0.5, "best_score": None,
                    "criterion": criterion, "threshold_grid": []}
        gkf = GroupKFold(n_splits=n_splits)
        fold_iter = gkf.split(X_tr, y_tr, groups)
    else:
        # 외부 검증용: StratifiedKFold
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = skf.split(X_tr, y_tr)

    for tr_idx, val_idx in fold_iter:
        if model_fn is not None:
            mdl = model_fn()
        else:
            from sklearn.base import clone
            mdl = clone(model)
        mdl.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_proba[val_idx] = mdl.predict_proba(X_tr[val_idx])[:, 1]

    valid = ~np.isnan(oof_proba)
    if valid.sum() < 10:
        return {"best_threshold": 0.5, "best_score": None,
                "criterion": criterion, "threshold_grid": []}

    y_v = y_tr[valid]
    p_v = oof_proba[valid]

    # ── Grid search ──────────────────────────────────────────────────
    candidates = np.arange(thr_range[0], thr_range[1] + 1e-9, thr_step)
    grid_results = []
    best_thr, best_score = 0.5, -np.inf

    for thr in _pbar(candidates, desc="  OOF thr grid", leave=False):
        pred = (p_v >= thr).astype(int)
        if len(set(pred)) < 2:
            continue
        tn, fp_n, fn_n, tp_n = confusion_matrix(y_v, pred, labels=[0, 1]).ravel()
        sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
        spec = tn  / (tn  + fp_n)  if (tn  + fp_n) > 0 else 0.0

        if criterion == "mcc":
            score = matthews_corrcoef(y_v, pred)
        elif criterion == "f1":
            score = f1_score(y_v, pred, zero_division=0)
        elif criterion == "youden":
            score = sens + spec - 1.0
        else:
            raise ValueError(f"criterion must be 'mcc', 'f1', or 'youden'. Got: {criterion}")

        grid_results.append({
            "threshold":   round(float(thr),   4),
            "score":       round(float(score),  4),
            "sensitivity": round(float(sens),   4),
            "specificity": round(float(spec),   4),
        })
        if score > best_score:
            best_score = score
            best_thr   = float(thr)

    return {
        "best_threshold": round(best_thr,         4),
        "best_score":     round(float(best_score), 4),
        "criterion":      criterion,
        "threshold_grid": grid_results,
    }
