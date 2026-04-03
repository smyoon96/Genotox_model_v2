"""
step9_external_benchmark.py — External Validation & Benchmark Comparison
=========================================================================
Phase 2: Compare our models against published benchmarks and validate
on external datasets.

Two modes:
  A) Literature comparison: compare our scaffold-split results vs published
     random-split results → quantify overestimation
  B) External validation: train on our data, test on external dataset
     (e.g., Hansen 2009 Ames, ToxCast invitro)

Usage:
  # Mode A: Literature comparison (no external data needed)
  python step9_external_benchmark.py --run-dir runs/20260327_100253_v12

  # Mode B: External validation (requires external CSV with SMILES + label)
  python step9_external_benchmark.py --run-dir runs/20260327_100253_v12 \
    --external-ames path/to/hansen2009.csv \
    --external-smiles-col SMILES --external-label-col Ames

  # Mode C: Random vs scaffold split comparison on OUR data
  python step9_external_benchmark.py --run-dir runs/20260327_100253_v12 --split-comparison
"""

import sys, json, logging, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_v2_core import (
    to_can, find_smi, bootstrap_ci, compute_ad, calibration,
    ANALYSIS_SMILES_COL, SEED,
)
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features,
    extract_fingerprint_features, select_fingerprint_bits,
)
from config import GLOBAL_SEED, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
)
lg = logging.getLogger("benchmark")


# ═══════════════════════════════════════════════════════
#  MODE A: Literature Comparison Table
# ═══════════════════════════════════════════════════════

PUBLISHED_BENCHMARKS = {
    "ames": [
        {
            "reference": "Hansen et al. 2009 (JCIM)",
            "dataset_size": 6512,
            "split": "random 80/20",
            "model": "SVM (ECFP6)",
            "metric": "accuracy",
            "value": 0.83,
            "mcc": None,
            "notes": "Random split, no scaffold awareness",
        },
        {
            "reference": "Xu et al. 2012 (JCIM)",
            "dataset_size": 7617,
            "model": "kNN + SVM consensus",
            "split": "random 5-fold CV",
            "metric": "accuracy",
            "value": 0.86,
            "mcc": None,
            "notes": "Random split CV, consensus model",
        },
        {
            "reference": "Honma et al. 2019 (Genes Environ)",
            "dataset_size": 12140,
            "model": "Random Forest",
            "split": "random 80/20",
            "metric": "balanced_accuracy",
            "value": 0.80,
            "mcc": None,
            "notes": "Largest public Ames dataset study",
        },
        {
            "reference": "Yang et al. 2019 (JCIM) - ChemProp",
            "dataset_size": 7255,
            "model": "D-MPNN (ChemProp)",
            "split": "scaffold split",
            "metric": "roc_auc",
            "value": 0.853,
            "mcc": None,
            "notes": "Scaffold split — more comparable to our work",
        },
        {
            "reference": "Li et al. 2021 (Chem Res Toxicol)",
            "dataset_size": 8576,
            "model": "XGBoost + ECFP4",
            "split": "random 80/20",
            "metric": "mcc",
            "value": 0.71,
            "mcc": 0.71,
            "notes": "Random split, same metric as ours",
        },
    ],
    "invitro": [
        {
            "reference": "ICH M7 assessment (FDA 2017)",
            "dataset_size": None,
            "model": "Expert rule + QSAR",
            "split": "regulatory",
            "metric": "sensitivity",
            "value": 0.70,
            "mcc": None,
            "notes": "Regulatory target: ≥70% sensitivity for genotoxicity",
        },
    ],
    "invivo": [
        {
            "reference": "Benigni et al. 2010 (Mutat Res)",
            "dataset_size": 850,
            "model": "Structural alerts only",
            "split": "leave-one-out",
            "metric": "sensitivity",
            "value": 0.65,
            "mcc": None,
            "notes": "LOO, limited to known structural alerts",
        },
    ],
}


def generate_literature_comparison(run_dir: Path):
    """Generate comparison table between our results and published benchmarks."""
    lg.info("=== Mode A: Literature Comparison ===")

    # Load our results
    locked = pd.read_csv(run_dir / "all_locked_test.csv")

    results = []
    for ep, benchmarks in PUBLISHED_BENCHMARKS.items():
        # Our best model for this endpoint
        ep_df = locked[locked["endpoint"] == ep]
        if ep_df.empty:
            continue
        our_best = ep_df.nlargest(1, "mcc").iloc[0]

        for bm in benchmarks:
            row = {
                "endpoint": ep,
                "source": "published",
                "reference": bm["reference"],
                "dataset_size": bm["dataset_size"],
                "split_method": bm["split"],
                "model": bm["model"],
                "metric": bm["metric"],
                "value": bm["value"],
                "mcc_reported": bm.get("mcc"),
                "notes": bm.get("notes", ""),
            }
            results.append(row)

        # Our result
        results.append({
            "endpoint": ep,
            "source": "this_work",
            "reference": "This work (v12)",
            "dataset_size": our_best["train_n"] + our_best["test_n"],
            "split_method": "scaffold split",
            "model": f"{our_best['model']} ({our_best['feature_mode']})",
            "metric": "mcc",
            "value": our_best["mcc"],
            "mcc_reported": our_best["mcc"],
            "notes": f"Scaffold split, AD_cov={our_best.get('ad_coverage', 'N/A')}",
        })

        # Also add our random-split equivalent for fair comparison
        # (will be computed in split_comparison mode)

    comp_df = pd.DataFrame(results)
    out_path = run_dir / "literature_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    lg.info(f"  Saved: {out_path} ({len(comp_df)} entries)")

    # Print summary
    for ep in comp_df["endpoint"].unique():
        lg.info(f"\n  [{ep}]")
        ep_comp = comp_df[comp_df["endpoint"] == ep]
        for _, r in ep_comp.iterrows():
            src = "★ " if r["source"] == "this_work" else "  "
            lg.info(f"    {src}{r['reference']}: {r['metric']}={r['value']:.3f} "
                    f"({r['split_method']}, {r['model']})")

    return comp_df


# ═══════════════════════════════════════════════════════
#  MODE B: External Dataset Validation
# ═══════════════════════════════════════════════════════

def external_validation(run_dir: Path, external_path: Path,
                        smiles_col: str = "SMILES", label_col: str = "label",
                        endpoint: str = "ames"):
    """
    Train best model on our data, test on external dataset.

    This is the strongest form of validation: the external dataset
    was never seen during any part of model development.
    """
    from sklearn.metrics import matthews_corrcoef
    from xgboost import XGBClassifier

    lg.info(f"=== Mode B: External Validation ({endpoint}) ===")
    lg.info(f"  External data: {external_path}")

    # Load external data
    ext = pd.read_csv(external_path)
    lg.info(f"  External: n={len(ext)}, pos={ext[label_col].mean():.4f}")

    # Canonicalize external SMILES
    ext["_can"] = ext[smiles_col].apply(to_can)
    ext = ext[ext["_can"].notna()].copy()
    ext[ANALYSIS_SMILES_COL] = ext["_can"]
    lg.info(f"  After canonicalization: n={len(ext)}")

    # Load our training data (best scenario)
    best_sc_path = run_dir / "best_scenario_by_cv.json"
    with open(best_sc_path) as f:
        best_sc = json.load(f)
    sc = best_sc.get(endpoint, "raw_all")

    # Find our split data
    ep_dir = run_dir / endpoint
    split_path = ep_dir / "fixed_split.csv"
    if not split_path.exists():
        lg.error(f"  Split file not found: {split_path}")
        return None
    our_data = pd.read_csv(split_path)
    our_train = our_data[our_data["split"] == "train"].copy()
    our_train[ANALYSIS_SMILES_COL] = our_train[find_smi(our_train)]

    # Remove overlap: exclude external compounds that appear in our data
    our_smiles = set(our_train[ANALYSIS_SMILES_COL].apply(to_can).dropna())
    ext_before = len(ext)
    ext = ext[~ext["_can"].isin(our_smiles)].copy()
    lg.info(f"  Overlap removed: {ext_before - len(ext)} compounds")
    lg.info(f"  External after dedup: n={len(ext)}, pos={ext[label_col].mean():.4f}")

    if len(ext) < 20:
        lg.error("  Too few external compounds after dedup")
        return None

    # Extract features for both
    lg.info("  Extracting features...")
    fg_tr = extract_fg_features(our_train, endpoint)
    ph_tr = extract_physchem_features(our_train, endpoint)
    fgp_tr = fg_tr[[c for c in fg_tr.columns if c.endswith("_present") or c.startswith("bb_")]]
    fp_tr = extract_fingerprint_features(our_train, endpoint, n_bits=512)

    fg_te = extract_fg_features(ext, endpoint)
    ph_te = extract_physchem_features(ext, endpoint)
    fgp_te = fg_te[[c for c in fg_te.columns if c.endswith("_present") or c.startswith("bb_")]]
    fp_te = extract_fingerprint_features(ext, endpoint, n_bits=512)

    # Select FP bits from train
    selected = select_fingerprint_bits(fp_tr, our_train["label"].values, top_k=128, method="mi")

    # Build feature matrices
    X_tr = pd.concat([fgp_tr, ph_tr, fp_tr[selected]], axis=1).fillna(0)
    X_te = pd.concat([fgp_te, ph_te, fp_te[selected]], axis=1).fillna(0)
    for c in X_tr.columns:
        X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce")
        X_te[c] = pd.to_numeric(X_te[c], errors="coerce")
    X_tr = X_tr.fillna(0).values.astype(np.float32)
    X_te = X_te.fillna(0).values.astype(np.float32)

    y_tr = our_train["label"].values.astype(int)
    y_te = ext[label_col].values.astype(int)

    # Train models
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    models = {
        "xgb": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=GLOBAL_SEED, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMClassifier
        models["lgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, verbose=-1,
            random_state=GLOBAL_SEED, n_jobs=-1)
    except ImportError:
        pass
    from sklearn.ensemble import RandomForestClassifier
    models["rf"] = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=-1)

    results = []
    for mn, mdl in models.items():
        lg.info(f"  Training {mn}...")
        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        ci = bootstrap_ci(y_te, pred, proba, 500)
        cal = calibration(y_te, proba)

        # AD coverage on external set
        ad = compute_ad(
            fp_tr[selected].values.astype(np.float32),
            fp_te[selected].values.astype(np.float32),
        )

        row = {
            "endpoint": endpoint, "model": mn,
            "external_dataset": str(external_path.name),
            "train_n": len(y_tr), "test_n": len(y_te),
            "test_pos_rate": round(y_te.mean(), 4),
            "ad_coverage": ad["coverage"],
            **{k: v["mean"] for k, v in ci.items()},
            **{f"{k}_lo": v["lo"] for k, v in ci.items()},
            **{f"{k}_hi": v["hi"] for k, v in ci.items()},
            "brier": cal["brier"], "ece": cal["ece"],
        }
        results.append(row)
        lg.info(f"    {mn}: MCC={row['mcc']:.4f} [{row['mcc_lo']:.4f},{row['mcc_hi']:.4f}] "
                f"AUC={row['roc_auc']:.4f} AD_cov={ad['coverage']:.3f}")

    ext_df = pd.DataFrame(results)
    out_path = run_dir / f"external_validation_{endpoint}.csv"
    ext_df.to_csv(out_path, index=False)
    lg.info(f"  Saved: {out_path}")
    return ext_df


# ═══════════════════════════════════════════════════════
#  MODE C: Random vs Scaffold Split Comparison
# ═══════════════════════════════════════════════════════

def split_comparison(run_dir: Path, n_repeats: int = 5):
    """
    Compare random split vs scaffold split on OUR data.
    Shows how much performance is overestimated by random split.
    This is a key argument for the paper.
    """
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import matthews_corrcoef
    from xgboost import XGBClassifier

    lg.info("=== Mode C: Random vs Scaffold Split Comparison ===")

    results = []
    for ep in ["ames", "invitro", "invivo", "invitro_sampling", "invivo_sampling"]:
        ep_dir = run_dir / ep
        split_path = ep_dir / "fixed_split.csv"
        if not split_path.exists():
            continue

        data = pd.read_csv(split_path)
        data[ANALYSIS_SMILES_COL] = data[find_smi(data)]

        # Extract compact features
        fg = extract_fg_features(data, ep)
        ph = extract_physchem_features(data, ep)
        fgp = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
        feat = pd.concat([fgp, ph], axis=1).fillna(0)
        for c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        X = feat.fillna(0).values.astype(np.float32)
        y = data["label"].values.astype(int)
        groups = data["scaffold_group"].astype(str).values

        if len(set(y)) < 2:
            continue

        # Guard: need enough data for meaningful CV
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if min(n_pos, n_neg) < 10:
            lg.info(f"  [{ep}] Skipping — too few minority class samples ({min(n_pos, n_neg)})")
            continue

        spw = (y == 0).sum() / max((y == 1).sum(), 1)

        # A) Scaffold split CV (as in our pipeline)
        from sklearn.preprocessing import LabelEncoder
        groups_le = LabelEncoder().fit_transform(groups)
        ug = np.unique(groups_le)
        ng = len(ug)
        af = min(5, ng)

        scaffold_mccs = []
        for rep in range(n_repeats):
            rng = np.random.RandomState(GLOBAL_SEED + rep * 7919)
            perm = rng.permutation(ng)
            fold_assign = np.arange(ng) % af
            g2f = dict(zip(ug[perm], fold_assign))
            sf = np.array([g2f[g] for g in groups_le])
            for f in range(af):
                vm = sf == f; tm = ~vm
                if vm.sum() == 0 or tm.sum() == 0:
                    continue
                Xt, yt = X[tm], y[tm]; Xv, yv = X[vm], y[vm]
                if len(set(yt)) < 2 or len(set(yv)) < 2:
                    continue
                mdl = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    scale_pos_weight=spw, eval_metric="logloss",
                    random_state=GLOBAL_SEED, n_jobs=-1)
                mdl.fit(Xt, yt)
                pred = (mdl.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                scaffold_mccs.append(matthews_corrcoef(yv, pred))

        # B) Random split CV (what most papers do)
        random_mccs = []
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=5, shuffle=True,
                                   random_state=GLOBAL_SEED + rep)
            for tr_idx, val_idx in skf.split(X, y):
                Xt, yt = X[tr_idx], y[tr_idx]
                Xv, yv = X[val_idx], y[val_idx]
                if len(set(yt)) < 2 or len(set(yv)) < 2:
                    continue
                mdl = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    scale_pos_weight=spw, eval_metric="logloss",
                    random_state=GLOBAL_SEED, n_jobs=-1)
                mdl.fit(Xt, yt)
                pred = (mdl.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                random_mccs.append(matthews_corrcoef(yv, pred))

        scaffold_mean = np.mean(scaffold_mccs)
        random_mean = np.mean(random_mccs)
        overestimation = random_mean - scaffold_mean

        row = {
            "endpoint": ep,
            "n_total": len(y),
            "pos_rate": round(y.mean(), 4),
            "random_split_mcc_mean": round(random_mean, 4),
            "random_split_mcc_std": round(np.std(random_mccs), 4),
            "scaffold_split_mcc_mean": round(scaffold_mean, 4),
            "scaffold_split_mcc_std": round(np.std(scaffold_mccs), 4),
            "overestimation": round(overestimation, 4),
            "overestimation_pct": round(overestimation / max(scaffold_mean, 0.01) * 100, 1),
            "n_random_folds": len(random_mccs),
            "n_scaffold_folds": len(scaffold_mccs),
        }
        results.append(row)
        lg.info(f"  [{ep}] Random MCC={random_mean:.4f} vs Scaffold MCC={scaffold_mean:.4f} "
                f"→ overestimation={overestimation:+.4f} ({row['overestimation_pct']:+.1f}%)")

    comp_df = pd.DataFrame(results)
    out_path = run_dir / "random_vs_scaffold_split.csv"
    comp_df.to_csv(out_path, index=False)
    lg.info(f"  Saved: {out_path}")
    return comp_df


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="External benchmark comparison for genotox QSAR"
    )
    p.add_argument("--run-dir", required=True, help="Path to pipeline run directory")
    p.add_argument("--external-ames", default=None,
                   help="Path to external Ames CSV (Mode B)")
    p.add_argument("--external-smiles-col", default="SMILES")
    p.add_argument("--external-label-col", default="label")
    p.add_argument("--split-comparison", action="store_true",
                   help="Run random vs scaffold split comparison (Mode C)")
    p.add_argument("--skip-literature", action="store_true",
                   help="Skip literature comparison table")
    args = p.parse_args()

    rd = Path(args.run_dir)
    if not rd.exists():
        raise FileNotFoundError(f"Run directory not found: {rd}")

    # Mode A: always run unless skipped
    if not args.skip_literature:
        generate_literature_comparison(rd)

    # Mode B: external validation
    if args.external_ames:
        external_validation(
            rd, Path(args.external_ames),
            smiles_col=args.external_smiles_col,
            label_col=args.external_label_col,
            endpoint="ames",
        )

    # Mode C: split comparison
    if args.split_comparison:
        split_comparison(rd)

    lg.info("\nDone.")


if __name__ == "__main__":
    main()
