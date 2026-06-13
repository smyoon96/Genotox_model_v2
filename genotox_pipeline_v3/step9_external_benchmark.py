"""
step9_external_benchmark.py -- External Validation & Benchmark Comparison
=========================================================================
v12.2 Changes:
  [FIX-1]  Mode B: OOF Youden-J threshold (was fixed 0.5)
  [NEW-2]  Mode B: AD-stratified sensitivity/specificity reporting
  [NEW-3]  Mode B: Chemical space analysis (Tanimoto dist by TP/FN/TN/FP)
  [NEW-4]  Mode D: FN augmentation loop

Four modes:
  A) Literature comparison
  B) External validation + OOF threshold + AD-stratified + chemical space
  C) Random vs scaffold split comparison
  D) FN augmentation loop

Usage:
  python step9_external_benchmark.py --run-dir runs/RUN_ID
  python step9_external_benchmark.py --run-dir runs/RUN_ID \\
    --external-ames path/to/external.csv --external-invitro ... --external-invivo ...
  python step9_external_benchmark.py --run-dir runs/RUN_ID --split-comparison
  python step9_external_benchmark.py --run-dir runs/RUN_ID \\
    --external-ames path/to/external.csv --fn-augmentation --fn-rounds 3
"""

import sys, json, logging, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.progress import pbar, step_header, task_done
from pipeline_v2_core import (
    to_can, find_smi, bootstrap_ci, compute_ad, calibration,
    oof_tune_threshold,     # [v12.2] unified OOF threshold grid search
    ANALYSIS_SMILES_COL, SEED,
)
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features,
    extract_fingerprint_features, select_fingerprint_bits, build_all_features,
)
from config import GLOBAL_SEED, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
lg = logging.getLogger("benchmark")


# ═══════════════════════════════════════════════════════
#  MODE A: Literature Comparison
# ═══════════════════════════════════════════════════════

PUBLISHED_BENCHMARKS = {
    "ames": [
        {"reference": "Hansen et al. 2009 (JCIM)", "dataset_size": 6512,
         "split": "random 80/20", "model": "SVM (ECFP6)", "metric": "accuracy",
         "value": 0.83, "mcc": None, "notes": "Random split, no scaffold awareness"},
        {"reference": "Xu et al. 2012 (JCIM)", "dataset_size": 7617,
         "model": "kNN + SVM consensus", "split": "random 5-fold CV",
         "metric": "accuracy", "value": 0.86, "mcc": None, "notes": "Random split CV, consensus model"},
        {"reference": "Honma et al. 2019 (Genes Environ)", "dataset_size": 12140,
         "model": "Random Forest", "split": "random 80/20", "metric": "balanced_accuracy",
         "value": 0.80, "mcc": None, "notes": "Largest public Ames dataset study"},
        {"reference": "Yang et al. 2019 (JCIM) - ChemProp", "dataset_size": 7255,
         "model": "D-MPNN (ChemProp)", "split": "scaffold split", "metric": "roc_auc",
         "value": 0.853, "mcc": None, "notes": "Scaffold split -- more comparable to our work"},
        {"reference": "Li et al. 2021 (Chem Res Toxicol)", "dataset_size": 8576,
         "model": "XGBoost + ECFP4", "split": "random 80/20", "metric": "mcc",
         "value": 0.71, "mcc": 0.71, "notes": "Random split, same metric as ours"},
    ],
    "invitro": [
        {"reference": "ICH M7 assessment (FDA 2017)", "dataset_size": None,
         "model": "Expert rule + QSAR", "split": "regulatory", "metric": "sensitivity",
         "value": 0.70, "mcc": None, "notes": "Regulatory target: >=70% sensitivity"},
    ],
    "invivo": [
        {"reference": "Benigni et al. 2010 (Mutat Res)", "dataset_size": 850,
         "model": "Structural alerts only", "split": "leave-one-out", "metric": "sensitivity",
         "value": 0.65, "mcc": None, "notes": "LOO, limited to known structural alerts"},
    ],
}


def generate_literature_comparison(run_dir: Path):
    lg.info("=== Mode A: Literature Comparison ===")
    locked = pd.read_csv(run_dir / "all_locked_test.csv")
    results = []
    for ep, benchmarks in PUBLISHED_BENCHMARKS.items():
        ep_df = locked[locked["endpoint"] == ep]
        if ep_df.empty:
            continue
        our_best = ep_df.nlargest(1, "mcc").iloc[0]
        for bm in benchmarks:
            results.append({
                "endpoint": ep, "source": "published", "reference": bm["reference"],
                "dataset_size": bm["dataset_size"], "split_method": bm["split"],
                "model": bm["model"], "metric": bm["metric"], "value": bm["value"],
                "mcc_reported": bm.get("mcc"), "notes": bm.get("notes", ""),
            })
        results.append({
            "endpoint": ep, "source": "this_work", "reference": "This work (v12)",
            "dataset_size": our_best["train_n"] + our_best["test_n"],
            "split_method": "scaffold split",
            "model": f"{our_best['model']} ({our_best['feature_mode']})",
            "metric": "mcc", "value": our_best["mcc"], "mcc_reported": our_best["mcc"],
            "notes": f"Scaffold split, AD_cov={our_best.get('ad_coverage', 'N/A')}",
        })
    comp_df = pd.DataFrame(results)
    out_path = run_dir / "literature_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    lg.info(f"  Saved: {out_path}")
    return comp_df


# [v12.2 FIX-1] → oof_tune_threshold() imported from pipeline_v2_core


# ═══════════════════════════════════════════════════════
#  [v12.2 NEW-2] AD-Stratified Performance
# ═══════════════════════════════════════════════════════

def compute_ad_stratified(y_te: np.ndarray, pred: np.ndarray,
                           per_sample_sim: np.ndarray,
                           ad_threshold: float = 0.4) -> dict:
    """
    AD 내(in_ad) / AD 외(out_ad) 분리 sensitivity/specificity.

    per_sample_sim: compute_ad()['per_sample_max_sim']
    논문 근거: sensitivity 저하가 AD 외 물질(화학공간 이격) 때문임을 분리 증명.
    """
    from sklearn.metrics import matthews_corrcoef, confusion_matrix

    result = {}
    for label, mask in [
        ("in_ad",  per_sample_sim >= ad_threshold),
        ("out_ad", per_sample_sim <  ad_threshold),
        ("all",    np.ones(len(y_te), dtype=bool)),
    ]:
        n = int(mask.sum())
        if n == 0:
            result[label] = {"n": 0}
            continue
        yt = y_te[mask]
        yp = pred[mask]
        n_pos = int((yt == 1).sum())
        n_neg = int((yt == 0).sum())
        if n_pos == 0 or n_neg == 0:
            result[label] = {"n": n, "n_pos": n_pos, "n_neg": n_neg,
                              "sensitivity": None, "specificity": None, "mcc": None}
            continue
        tn, fp_n, fn_n, tp_n = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        result[label] = {
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "tp": int(tp_n), "fp": int(fp_n), "fn": int(fn_n), "tn": int(tn),
            "sensitivity": round(tp_n / (tp_n + fn_n), 4) if (tp_n + fn_n) > 0 else None,
            "specificity": round(tn / (tn + fp_n), 4) if (tn + fp_n) > 0 else None,
            "mcc": round(matthews_corrcoef(yt, yp), 4) if len(set(yp)) > 1 else None,
            "ad_threshold": ad_threshold,
            "sim_median": round(float(np.median(per_sample_sim[mask])), 4),
        }
        lg.info(f"    AD [{label:7s}] n={n:4d} pos={n_pos} "
                f"sens={result[label]['sensitivity']} "
                f"spec={result[label]['specificity']} "
                f"mcc={result[label]['mcc']} "
                f"sim_med={result[label]['sim_median']:.3f}")
    return result


# ═══════════════════════════════════════════════════════
#  [v12.2 NEW-3] Chemical Space Analysis
# ═══════════════════════════════════════════════════════

def chemical_space_analysis(y_te: np.ndarray, pred: np.ndarray,
                             per_sample_sim: np.ndarray,
                             endpoint: str, run_dir: Path) -> pd.DataFrame:
    """
    TP/FN/TN/FP 별 학습 데이터와의 Tanimoto 유사도 분포.

    논문 Figure:
      FN의 sim_median < TP의 sim_median → 화학공간 이격이 sensitivity 저하 원인
      FN에서 T<0.5 비율이 높음 → 논문 self-Tanimoto 분석과 직결
    """
    outcomes = []
    for i in range(len(y_te)):
        actual, predicted = int(y_te[i]), int(pred[i])
        if   actual == 1 and predicted == 1: outcome = "TP"
        elif actual == 1 and predicted == 0: outcome = "FN"
        elif actual == 0 and predicted == 0: outcome = "TN"
        else:                                outcome = "FP"
        outcomes.append({
            "sample_idx": i,
            "actual": actual, "predicted": predicted, "outcome": outcome,
            "max_tanimoto_to_train": round(float(per_sample_sim[i]), 4),
        })

    detail_df = pd.DataFrame(outcomes)
    detail_df.to_csv(run_dir / f"chemical_space_{endpoint}.csv", index=False)

    summary_rows = []
    for outcome in ["TP", "FN", "TN", "FP"]:
        sub = detail_df[detail_df["outcome"] == outcome]["max_tanimoto_to_train"]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "endpoint": endpoint, "outcome": outcome, "n": len(sub),
            "sim_mean":      round(sub.mean(), 4),
            "sim_median":    round(sub.median(), 4),
            "sim_q25":       round(sub.quantile(0.25), 4),
            "sim_q75":       round(sub.quantile(0.75), 4),
            "pct_below_05":  round((sub < 0.5).mean(), 4),
            "pct_below_04":  round((sub < 0.4).mean(), 4),
            "pct_below_03":  round((sub < 0.3).mean(), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / f"chemical_space_summary_{endpoint}.csv", index=False)

    lg.info(f"  [Chemical Space] {endpoint}")
    for _, r in summary_df.iterrows():
        lg.info(f"    {r['outcome']:3s}: n={r['n']:4d}  sim_median={r['sim_median']:.3f}  "
                f"T<0.5:{r['pct_below_05']*100:.1f}%  T<0.4:{r['pct_below_04']*100:.1f}%")
    return summary_df


# ═══════════════════════════════════════════════════════
#  Shared: Feature Extraction Helper
# ═══════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, endpoint: str, n_bits: int = 512,
                   top_k: int = 128, selected_bits=None):
    """FGP + physchem + MI-selected FP. selected_bits=None: train용 (selection 수행)."""
    fg = extract_fg_features(df, endpoint)
    ph = extract_physchem_features(df, endpoint)
    fgp = fg  # v12.2: FG 전체 컬럼 사용
    fp_full = extract_fingerprint_features(df, endpoint, n_bits=n_bits)
    if selected_bits is None:
        y_tmp = df["label"].values.astype(int)
        selected_bits = select_fingerprint_bits(fp_full, y_tmp, top_k=top_k, method="mi")
    feat = pd.concat([fgp, ph, fp_full[selected_bits]], axis=1).fillna(0)
    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    X = feat.fillna(0).values.astype(np.float32)
    return X, fp_full, selected_bits


# ═══════════════════════════════════════════════════════
#  MODE B: External Dataset Validation  [v12.2]
# ═══════════════════════════════════════════════════════

def external_validation(run_dir: Path, external_path: Path,
                        smiles_col: str = "SMILES", label_col: str = "label",
                        endpoint: str = "ames", ad_threshold: float = 0.4,
                        n_bits: int = 512, top_k: int = 128,
                        thr_range: tuple = (0.1, 0.9), thr_step: float = 0.02,
                        criterion: str = "mcc"):
    """
    Train on our data, test on external dataset.

    v12.2:
      [FIX-1] OOF Youden's J threshold (was fixed 0.5)
      [NEW-2] AD-stratified sensitivity/specificity
      [NEW-3] Chemical space similarity distribution by TP/FN/TN/FP
    """
    from sklearn.metrics import matthews_corrcoef
    from xgboost import XGBClassifier

    lg.info(f"=== Mode B: External Validation ({endpoint}) ===")
    ext = pd.read_csv(external_path)
    ext["_can"] = ext[smiles_col].apply(to_can)
    ext = ext[ext["_can"].notna()].copy()
    ext[ANALYSIS_SMILES_COL] = ext["_can"]
    ext["label"] = ext[label_col].astype(int)
    lg.info(f"  External: n={len(ext)}, pos={ext['label'].mean():.4f}")

    split_path = run_dir / endpoint / "fixed_split.csv"
    if not split_path.exists():
        lg.error(f"  Split file not found: {split_path}"); return None
    our_data = pd.read_csv(split_path)
    our_train = our_data[our_data["split"] == "train"].copy()
    our_train[ANALYSIS_SMILES_COL] = our_train[find_smi(our_train)]

    our_smiles = set(our_train[ANALYSIS_SMILES_COL].apply(to_can).dropna())
    ext_before = len(ext)
    ext = ext[~ext["_can"].isin(our_smiles)].copy()
    lg.info(f"  Overlap removed: {ext_before - len(ext)} | Remaining: n={len(ext)}, "
            f"pos={ext['label'].mean():.4f}")

    if len(ext) < 20:
        lg.error("  Too few external compounds after dedup"); return None

    lg.info("  Extracting features...")
    X_tr, fp_tr_full, selected_bits = build_features(our_train, endpoint, n_bits, top_k)
    X_te, fp_te_full, _             = build_features(ext, endpoint, n_bits, top_k,
                                                      selected_bits=selected_bits)
    y_tr = our_train["label"].values.astype(int)
    y_te = ext["label"].values.astype(int)

    # AD -- [v12.2] per_sample_max_sim 사용
    fp_tr_sel = fp_tr_full[selected_bits].values.astype(np.float32)
    fp_te_sel = fp_te_full[selected_bits].values.astype(np.float32)
    ad_result = compute_ad(fp_tr_sel, fp_te_sel, threshold=ad_threshold)
    per_sample_sim = ad_result["per_sample_max_sim"]

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    models = {
        "xgb": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              scale_pos_weight=spw, eval_metric="logloss",
                              random_state=GLOBAL_SEED, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMClassifier
        models["lgbm"] = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                         scale_pos_weight=spw, verbose=-1,
                                         random_state=GLOBAL_SEED, n_jobs=-1)
    except ImportError:
        pass
    from sklearn.ensemble import RandomForestClassifier
    models["rf"] = RandomForestClassifier(n_estimators=200, max_depth=10,
                                           class_weight="balanced",
                                           random_state=GLOBAL_SEED, n_jobs=-1)

    results = []
    all_preds_by_model = {}

    for mn, mdl in pbar(models.items(), desc="  Models", total=len(models)):
        lg.info(f"\n  [{mn}]")

        # [v12.2 FIX-1] OOF threshold grid search
        thr_result = oof_tune_threshold(
            X_tr, y_tr, model=mdl, n_folds=5,
            thr_range=thr_range, thr_step=thr_step, criterion=criterion)
        opt_thr   = thr_result["best_threshold"]
        grid_rows = thr_result["threshold_grid"]

        # 그리드 탐색 결과 저장 (진단 / Figure용)
        grid_df = pd.DataFrame(grid_rows)
        grid_df.insert(0, "endpoint", endpoint)
        grid_df.insert(1, "model", mn)
        grid_path = run_dir / f"threshold_grid_{endpoint}_{mn}.csv"
        grid_df.to_csv(grid_path, index=False)
        lg.info(f"    Threshold grid saved: {grid_path} ({len(grid_df)} candidates)")

        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_te)[:, 1]

        # fixed_05 vs oof_optimal 나란히 평가 (재현성 확보)
        for thr_label, thr in [("fixed_05", 0.5), ("oof_optimal", opt_thr)]:
            pred = (proba >= thr).astype(int)
            ci = bootstrap_ci(y_te, pred, proba, 500)
            cal = calibration(y_te, proba)

            lg.info(f"    --- threshold={thr:.4f} ({thr_label}) ---")
            ad_strat = compute_ad_stratified(y_te, pred, per_sample_sim, ad_threshold)

            row = {
                "endpoint": endpoint, "model": mn,
                "threshold_type": thr_label, "threshold_value": round(thr, 4),
                "oof_criterion": criterion,
                "oof_best_score": thr_result["best_score"] if thr_label == "oof_optimal" else None,
                "external_dataset": str(external_path.name),
                "train_n": len(y_tr), "test_n": len(y_te),
                "test_pos_rate": round(y_te.mean(), 4),
                "ad_coverage": ad_result["coverage"],
                "ad_mean_sim": ad_result["mean_sim"],
                "ad_median_sim": ad_result["median_sim"],
                **{k: v["mean"] for k, v in ci.items()},
                **{f"{k}_lo": v["lo"] for k, v in ci.items()},
                **{f"{k}_hi": v["hi"] for k, v in ci.items()},
                "brier": cal["brier"], "ece": cal["ece"],
            }
            for zone in ["in_ad", "out_ad", "all"]:
                z = ad_strat.get(zone, {})
                for metric in ["n", "n_pos", "sensitivity", "specificity", "mcc", "sim_median"]:
                    row[f"{zone}_{metric}"] = z.get(metric)
            results.append(row)

        all_preds_by_model[mn] = (proba >= opt_thr).astype(int)

    # [NEW-3] Chemical space analysis (best model)
    best_mn = list(all_preds_by_model.keys())[0]
    lg.info(f"\n  [Chemical Space -- {best_mn}]")
    chemical_space_analysis(y_te, all_preds_by_model[best_mn], per_sample_sim,
                             endpoint, run_dir)

    ext_df = pd.DataFrame(results)
    out_path = run_dir / f"external_validation_{endpoint}.csv"
    ext_df.to_csv(out_path, index=False)
    lg.info(f"\n  Saved: {out_path}")
    return ext_df


# ═══════════════════════════════════════════════════════
#  MODE C: Random vs Scaffold Split Comparison
# ═══════════════════════════════════════════════════════

def split_comparison(run_dir: Path, n_repeats: int = 5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import matthews_corrcoef
    from xgboost import XGBClassifier

    lg.info("=== Mode C: Random vs Scaffold Split Comparison ===")
    results = []
    _eps = ["ames", "invitro", "invivo", "invitro_sampling", "invivo_sampling"]
    for ep in pbar(_eps, desc="Mode C: Split comparison", colour="blue"):
        ep_dir = run_dir / ep
        split_path = ep_dir / "fixed_split.csv"
        if not split_path.exists():
            continue
        data = pd.read_csv(split_path)
        data[ANALYSIS_SMILES_COL] = data[find_smi(data)]
        fg = extract_fg_features(data, ep)
        ph = extract_physchem_features(data, ep)
        fgp = fg  # v12.2: FG 전체 컬럼 사용
        feat = pd.concat([fgp, ph], axis=1).fillna(0)
        for c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        X = feat.fillna(0).values.astype(np.float32)
        y = data["label"].values.astype(int)
        groups = data["scaffold_group"].astype(str).values
        if len(set(y)) < 2:
            continue
        n_pos = y.sum(); n_neg = len(y) - n_pos
        if min(n_pos, n_neg) < 10:
            lg.info(f"  [{ep}] Skipping -- too few minority ({min(n_pos, n_neg)})"); continue
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        from sklearn.preprocessing import LabelEncoder
        groups_le = LabelEncoder().fit_transform(groups)
        ug = np.unique(groups_le); ng = len(ug); af = min(5, ng)

        scaffold_mccs, random_mccs = [], []
        for rep in range(n_repeats):
            rng = np.random.RandomState(GLOBAL_SEED + rep * 7919)
            perm = rng.permutation(ng); fold_assign = np.arange(ng) % af
            g2f = dict(zip(ug[perm], fold_assign))
            sf = np.array([g2f[g] for g in groups_le])
            for f in range(af):
                vm = sf == f; tm = ~vm
                if vm.sum() == 0 or tm.sum() == 0: continue
                Xt, yt = X[tm], y[tm]; Xv, yv = X[vm], y[vm]
                if len(set(yt)) < 2 or len(set(yv)) < 2: continue
                mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     scale_pos_weight=spw, eval_metric="logloss",
                                     random_state=GLOBAL_SEED, n_jobs=-1)
                mdl.fit(Xt, yt)
                pred = (mdl.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                scaffold_mccs.append(matthews_corrcoef(yv, pred))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED + rep)
            for tr_idx, val_idx in skf.split(X, y):
                Xt, yt = X[tr_idx], y[tr_idx]; Xv, yv = X[val_idx], y[val_idx]
                if len(set(yt)) < 2 or len(set(yv)) < 2: continue
                mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     scale_pos_weight=spw, eval_metric="logloss",
                                     random_state=GLOBAL_SEED, n_jobs=-1)
                mdl.fit(Xt, yt)
                pred = (mdl.predict_proba(Xv)[:, 1] >= 0.5).astype(int)
                random_mccs.append(matthews_corrcoef(yv, pred))

        scaffold_mean = np.mean(scaffold_mccs); random_mean = np.mean(random_mccs)
        overestimation = random_mean - scaffold_mean
        row = {
            "endpoint": ep, "n_total": len(y), "pos_rate": round(y.mean(), 4),
            "random_split_mcc_mean": round(random_mean, 4),
            "random_split_mcc_std": round(np.std(random_mccs), 4),
            "scaffold_split_mcc_mean": round(scaffold_mean, 4),
            "scaffold_split_mcc_std": round(np.std(scaffold_mccs), 4),
            "overestimation": round(overestimation, 4),
            "overestimation_pct": round(overestimation / max(scaffold_mean, 0.01) * 100, 1),
            "n_random_folds": len(random_mccs), "n_scaffold_folds": len(scaffold_mccs),
        }
        results.append(row)
        lg.info(f"  [{ep}] Random={random_mean:.4f} Scaffold={scaffold_mean:.4f} "
                f"overest={overestimation:+.4f} ({row['overestimation_pct']:+.1f}%)")

    comp_df = pd.DataFrame(results)
    comp_df.to_csv(run_dir / "random_vs_scaffold_split.csv", index=False)
    lg.info(f"  Saved: {run_dir / 'random_vs_scaffold_split.csv'}")
    return comp_df


# ═══════════════════════════════════════════════════════
#  [v12.2 NEW-4] MODE D: FN Augmentation Loop
# ═══════════════════════════════════════════════════════

def fn_augmentation_loop(run_dir: Path, external_path: Path,
                          smiles_col: str = "SMILES", label_col: str = "label",
                          endpoint: str = "ames", n_rounds: int = 3,
                          ad_threshold: float = 0.4,
                          n_bits: int = 512, top_k: int = 128,
                          thr_range: tuple = (0.1, 0.9), thr_step: float = 0.02,
                          criterion: str = "mcc"):
    """
    FN 물질 반복 편입 실험.

    라운드마다:
      1. 현재 train으로 외부 데이터 예측 (OOF threshold)
      2. FN 추출 → train에 추가
      3. sensitivity / AD coverage 변화 추적

    논문 활용:
      - 몇 개의 FN 편입 시 sensitivity가 ICH M7 기준(0.70)에 도달하는지
      - AD coverage와 sensitivity의 라운드별 변화 (Figure)
      - 편입된 FN 물질의 구조적 특성 (sim_median으로 정량화)

    출력:
      fn_augmentation_{endpoint}.csv        -- 라운드별 집계
      fn_augmentation_{endpoint}_detail.csv -- 편입된 물질 목록 (SMILES + sim)
    """
    from sklearn.metrics import confusion_matrix
    from xgboost import XGBClassifier

    lg.info(f"=== Mode D: FN Augmentation ({endpoint}, {n_rounds} rounds) ===")

    # 외부 데이터 로드
    ext_raw = pd.read_csv(external_path)
    ext_raw["_can"] = ext_raw[smiles_col].apply(to_can)
    ext_raw = ext_raw[ext_raw["_can"].notna()].copy()
    ext_raw[ANALYSIS_SMILES_COL] = ext_raw["_can"]
    ext_raw["label"] = ext_raw[label_col].astype(int)
    lg.info(f"  External: n={len(ext_raw)}, pos_rate={ext_raw['label'].mean():.4f}")

    # 원본 학습 데이터
    split_path = run_dir / endpoint / "fixed_split.csv"
    if not split_path.exists():
        lg.error(f"  Split file not found: {split_path}"); return None
    our_data = pd.read_csv(split_path)
    our_train_orig = our_data[our_data["split"] == "train"].copy()
    our_train_orig[ANALYSIS_SMILES_COL] = our_train_orig[find_smi(our_train_orig)]

    orig_smiles = set(our_train_orig[ANALYSIS_SMILES_COL].apply(to_can).dropna())
    ext_pool = ext_raw[~ext_raw["_can"].isin(orig_smiles)].copy().reset_index(drop=True)
    lg.info(f"  External after dedup vs train: n={len(ext_pool)}")
    if len(ext_pool) < 20:
        lg.error("  Too few external compounds"); return None

    augmented_train = our_train_orig.copy()
    remaining_ext   = ext_pool.copy()
    round_results   = []
    added_compounds = []

    for rnd in pbar(range(n_rounds + 1), desc="  FN aug rounds", colour="yellow"):
        lg.info(f"\n  --- Round {rnd} | train_n={len(augmented_train)} ---")
        if len(remaining_ext) < 10:
            lg.info("  Remaining pool too small, stopping."); break

        X_tr, fp_tr_full, selected_bits = build_features(augmented_train, endpoint, n_bits, top_k)
        X_te, fp_te_full, _             = build_features(remaining_ext, endpoint, n_bits, top_k,
                                                          selected_bits=selected_bits)
        y_tr = augmented_train["label"].values.astype(int)
        y_te = remaining_ext["label"].values.astype(int)

        fp_tr_sel = fp_tr_full[selected_bits].values.astype(np.float32)
        fp_te_sel = fp_te_full[selected_bits].values.astype(np.float32)
        ad_result     = compute_ad(fp_tr_sel, fp_te_sel, threshold=ad_threshold)
        per_sample_sim = ad_result["per_sample_max_sim"]

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=GLOBAL_SEED, n_jobs=-1)
        thr_result = oof_tune_threshold(
            X_tr, y_tr, model=mdl, n_folds=5,
            thr_range=thr_range, thr_step=thr_step, criterion=criterion)
        opt_thr = thr_result["best_threshold"]
        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_te)[:, 1]
        pred  = (proba >= opt_thr).astype(int)

        n_pos = int((y_te == 1).sum()); n_neg = int((y_te == 0).sum())
        if n_pos == 0 or n_neg == 0:
            lg.warning("  Skipping round -- no pos or neg in remaining pool"); break

        tn, fp_n, fn_n, tp_n = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        sensitivity  = tp_n / (tp_n + fn_n)  if (tp_n + fn_n) > 0 else 0.0
        specificity  = tn  / (tn  + fp_n)   if (tn  + fp_n) > 0 else 0.0

        ad_strat = compute_ad_stratified(y_te, pred, per_sample_sim, ad_threshold)

        fn_mask = (pred == 0) & (y_te == 1)
        fn_sim_values = per_sample_sim[fn_mask] if fn_mask.sum() > 0 else np.array([])

        round_results.append({
            "round": rnd,
            "train_n": len(augmented_train),
            "train_pos_n": int((y_tr == 1).sum()),
            "remaining_ext_n": len(remaining_ext),
            "remaining_ext_pos_n": int(n_pos),
            "threshold_used": round(opt_thr, 4),
            "oof_criterion": criterion,
            "oof_best_score": round(float(thr_result["best_score"]), 4),
            "tp": int(tp_n), "fp": int(fp_n), "fn": int(fn_n), "tn": int(tn),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "fn_to_add": int(fn_mask.sum()),
            "ad_coverage": ad_result["coverage"],
            "ad_mean_sim": ad_result["mean_sim"],
            "fn_sim_median": round(float(np.median(fn_sim_values)), 4) if len(fn_sim_values) > 0 else None,
            "in_ad_sensitivity":  ad_strat.get("in_ad", {}).get("sensitivity"),
            "out_ad_sensitivity": ad_strat.get("out_ad", {}).get("sensitivity"),
            "in_ad_n":  ad_strat.get("in_ad", {}).get("n"),
            "out_ad_n": ad_strat.get("out_ad", {}).get("n"),
        })
        lg.info(f"    sens={sensitivity:.4f}  spec={specificity:.4f}  "
                f"fn_added={fn_mask.sum()}  ad_cov={ad_result['coverage']:.3f}")

        if rnd == n_rounds:
            break  # 마지막 라운드: 평가만 하고 편입 안 함
        if fn_mask.sum() == 0:
            lg.info("  No FN compounds -- stopping early."); break

        fn_df = remaining_ext[fn_mask].copy()
        fn_df["split"] = "train"
        fn_df["augmentation_round"] = rnd + 1
        fn_df["max_sim_to_train"] = per_sample_sim[fn_mask]
        added_compounds.append(fn_df)

        augmented_train = pd.concat([augmented_train, fn_df], ignore_index=True)
        remaining_ext   = remaining_ext[~fn_mask].copy().reset_index(drop=True)

    round_df = pd.DataFrame(round_results)
    round_df.to_csv(run_dir / f"fn_augmentation_{endpoint}.csv", index=False)
    lg.info(f"\n  Saved: fn_augmentation_{endpoint}.csv")

    if added_compounds:
        detail_df = pd.concat(added_compounds, ignore_index=True)
        detail_df[[ANALYSIS_SMILES_COL, "label", "augmentation_round",
                   "max_sim_to_train"]].to_csv(
            run_dir / f"fn_augmentation_{endpoint}_detail.csv", index=False)
        lg.info(f"  Saved: fn_augmentation_{endpoint}_detail.csv ({len(detail_df)} compounds)")

    return round_df


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="External benchmark -- genotox QSAR v12.2")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--external-ames",    default=None)
    p.add_argument("--external-invitro", default=None)
    p.add_argument("--external-invivo",  default=None)
    p.add_argument("--external-smiles-col", default="SMILES")
    p.add_argument("--external-label-col",  default="label")
    p.add_argument("--split-comparison",  action="store_true")
    p.add_argument("--fn-augmentation",   action="store_true")
    p.add_argument("--fn-rounds",   type=int,   default=3)
    p.add_argument("--skip-literature",   action="store_true")
    p.add_argument("--ad-threshold", type=float, default=0.4,
                   help="Tanimoto threshold for AD in/out (default: 0.4)")
    p.add_argument("--n-bits", type=int, default=512)
    p.add_argument("--top-k",  type=int, default=128)
    # Threshold grid search
    p.add_argument("--thr-min",   type=float, default=0.1,
                   help="Threshold grid lower bound (default: 0.1)")
    p.add_argument("--thr-max",   type=float, default=0.9,
                   help="Threshold grid upper bound (default: 0.9)")
    p.add_argument("--thr-step",  type=float, default=0.02,
                   help="Threshold grid step size (default: 0.02 = 40 candidates)")
    p.add_argument("--criterion", type=str,   default="mcc",
                   choices=["mcc", "f1", "youden"],
                   help="OOF threshold selection criterion (default: mcc)")
    args = p.parse_args()

    rd = Path(args.run_dir)
    if not rd.exists():
        raise FileNotFoundError(f"Run directory not found: {rd}")

    if not args.skip_literature:
        generate_literature_comparison(rd)

    ext_map = {
        "ames":    args.external_ames,
        "invitro": args.external_invitro,
        "invivo":  args.external_invivo,
    }
    for endpoint, ext_path in ext_map.items():
        if not ext_path:
            continue
        ep = Path(ext_path)
        external_validation(rd, ep,
                            smiles_col=args.external_smiles_col,
                            label_col=args.external_label_col,
                            endpoint=endpoint,
                            ad_threshold=args.ad_threshold,
                            n_bits=args.n_bits, top_k=args.top_k,
                            thr_range=(args.thr_min, args.thr_max),
                            thr_step=args.thr_step,
                            criterion=args.criterion)
        if args.fn_augmentation:
            fn_augmentation_loop(rd, ep,
                                  smiles_col=args.external_smiles_col,
                                  label_col=args.external_label_col,
                                  endpoint=endpoint,
                                  n_rounds=args.fn_rounds,
                                  ad_threshold=args.ad_threshold,
                                  n_bits=args.n_bits, top_k=args.top_k,
                                  thr_range=(args.thr_min, args.thr_max),
                                  thr_step=args.thr_step,
                                  criterion=args.criterion)

    if args.split_comparison:
        split_comparison(rd)

    lg.info("\nDone.")


if __name__ == "__main__":
    main()
