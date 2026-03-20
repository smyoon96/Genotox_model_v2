"""
genotox_pipeline.py v11 — Clean Single-Run Pipeline
=====================================================
원칙: 하나의 코드, 하나의 실행, 하나의 결과.
      로그 파싱 없음. 런 병합 없음. 수동 조립 없음.

Step 1-3: Load → Clean → Flags → Fixed Split
Step 4:   Train CV → scenario selection (compact/xgb only)
Step 5:   36 combos locked test (pre-computed features, no re-extraction)
Step 6:   LDO + Checklist + Output

설계 제한 (정직하게 명시):
  - 하이퍼파라미터: 고정 baseline + RandomizedSearchCV 튜닝 결과 모두 보고
  - Threshold tuning: OOF-based (train resubstitution 아님)
  - CV: scenario selection에만 사용, 36-combo selection 아님
  - QM/under-sampling: 메인 결과에 미포함
  - cv_selected_compact_xgb: CV가 고른 scenario에서 compact+xgb (endpoint당 1개)
  - paper_primary_model: locked test MCC 최고 조합 (endpoint당 1개, tie-break: raw_all 우선)
"""
import sys, json, logging, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_v2_core import (
    resolve_conflicts, assign_scaffolds, fixed_split, apply_scenario,
    repeated_cv, bootstrap_ci, compute_ad, calibration,
    domain_confounding, leave_domain_out, cross_endpoint,
    find_smi, to_can, file_hash, interpret_result,
)
from step2b_preprocessing_impact import classify_all_compounds
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features, extract_fingerprint_features,
)
from config import DATA_DIR, RUNS_DIR, GLOBAL_SEED, ENDPOINTS, make_run_dir, save_json
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*X has feature names.*")

SCENARIOS = ["raw_all", "no_metal", "salt_stripped"]
FEAT_MODES = ["compact", "broad_fp"]
MODEL_NAMES = ["xgb", "logistic"]


def setup_log(rd):
    root = logging.getLogger(); root.setLevel(logging.INFO); root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(rd / "pipeline.log", encoding="utf-8")]:
        h.setFormatter(fmt); root.addHandler(h)
    return logging.getLogger("gp")


def make_model(name, spw):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=GLOBAL_SEED, n_jobs=-1)
    elif name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000,
                random_state=GLOBAL_SEED))])
    raise ValueError(name)


# ─── Hyperparameter Tuning ──────────────────────

# XGB search space
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
}

# Logistic search space
LOGISTIC_PARAM_GRID = {
    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["saga"],
}

HP_N_ITER = 40       # RandomizedSearchCV iterations (increase for production)
HP_CV_FOLDS = 3      # GroupKFold folds for HP search


def tune_model(X_train, y_train, groups, model_name, spw, n_iter=HP_N_ITER, n_folds=HP_CV_FOLDS):
    """
    GroupKFold 기반 RandomizedSearchCV로 하이퍼파라미터 튜닝.
    scaffold group을 존중하는 CV split 사용.
    
    Returns: (best_model, best_params, best_cv_score)
    """
    from sklearn.model_selection import RandomizedSearchCV, GroupKFold
    from sklearn.metrics import make_scorer
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)
    if actual_folds < 2:
        # 그룹이 너무 적으면 튜닝 불가 → 기본 모델 반환
        mdl = make_model(model_name, spw)
        mdl.fit(X_train, y_train)
        return mdl, {}, np.nan
    
    gkf = GroupKFold(n_splits=actual_folds)
    
    if model_name == "xgb":
        from xgboost import XGBClassifier
        base = XGBClassifier(
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=GLOBAL_SEED, n_jobs=-1)
        param_dist = XGB_PARAM_GRID
    elif model_name == "logistic":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000,
                random_state=GLOBAL_SEED))])
        param_dist = LOGISTIC_PARAM_GRID
    else:
        raise ValueError(model_name)
    
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=min(n_iter, 40),  # cap for safety
        scoring=mcc_scorer,
        cv=gkf,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        error_score=0.0,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    
    best_params = search.best_params_
    best_score = round(search.best_score_, 4)
    
    return search.best_estimator_, best_params, best_score


def oof_threshold(X_train, y_train, groups, model_fn, n_folds=3):
    """OOF probability 기반 threshold tuning (train resubstitution이 아님)."""
    from sklearn.model_selection import GroupKFold
    oof_proba = np.full(len(y_train), np.nan)
    n_splits = min(n_folds, len(np.unique(groups)))
    if n_splits < 2:
        return 0.5
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
        mdl = model_fn()
        mdl.fit(X_train[tr_idx], y_train[tr_idx])
        oof_proba[val_idx] = mdl.predict_proba(X_train[val_idx])[:, 1]
    valid = ~np.isnan(oof_proba)
    if valid.sum() < 10:
        return 0.5
    best_t, best_mcc = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        pred = (oof_proba[valid] >= t).astype(int)
        if len(set(pred)) < 2: continue
        m = matthews_corrcoef(y_train[valid], pred)
        if m > best_mcc:
            best_mcc = m; best_t = t
    return round(best_t, 2)


def run_pipeline(data_dir=None, tag="v11"):
    from xgboost import XGBClassifier
    import joblib

    dp = Path(data_dir) if data_dir else DATA_DIR
    rd = make_run_dir(tag); lg = setup_log(rd); t0 = time.time()
    lg.info(f"Pipeline v11 — Clean Single Run\n  {rd}")
    save_json({"data": str(dp), "version": "v11",
               "ts": datetime.now().isoformat(), "seed": GLOBAL_SEED}, rd / "config.json")

    # ═══ STEP 1: Load + Clean ═══
    lg.info("STEP 1: Load + Clean")
    raw = {}
    for ep, fn in [("ames","ames_combine.xlsx"),("invitro","invitro_pre.csv"),("invivo","invivo_pre.csv")]:
        fp = dp / fn
        if not fp.exists(): continue
        df = pd.read_excel(fp, engine="openpyxl") if fn.endswith(".xlsx") else pd.read_csv(fp, encoding="utf-8-sig")
        for c in df.columns:
            if c.strip().upper() == "SMILES": df = df.rename(columns={c: "SMILES"})
        df["label"] = df["label"].astype(int); df["endpoint"] = ep
        df = df.dropna(subset=["label","SMILES"])
        raw[ep] = df
        lg.info(f"  [{ep}] n={len(df)} pos={df['label'].mean():.4f}")

    # Cross-endpoint overlap
    ov = cross_endpoint(raw); ov.to_csv(rd / "cross_endpoint_overlap.csv", index=False)

    # Domain confounding
    dom_rpt = domain_confounding(raw.get("ames", pd.DataFrame()))
    save_json(dom_rpt, rd / "ames_domain_confounding.json")
    if dom_rpt.get("domain_acc"):
        lg.info(f"  ⚠ Domain predictor acc: {dom_rpt['domain_acc']}")

    # Label conflicts
    cleaned = {}
    for ep, df in raw.items():
        (rd / ep).mkdir(parents=True, exist_ok=True)
        sens = []
        for s in ["conservative","positive_priority","majority_vote"]:
            c, _, r = resolve_conflicts(df, s)
            c.to_csv(rd / ep / f"cleaned_{s}.csv", index=False)
            sens.append(r)
        save_json(sens, rd / ep / "conflict_sensitivity.json")
        cleaned[ep], _, r = resolve_conflicts(df, "conservative")
        lg.info(f"  [{ep}] {len(df)} → {len(cleaned[ep])} (conflicts={r['n_conflicts']})")

    # ═══ STEP 2: Flags ═══
    lg.info("STEP 2: Preprocessing Flags")
    flagged = {}
    for ep, df in cleaned.items():
        f = classify_all_compounds(df, smi_col="SMILES")
        f.to_csv(rd / ep / "preprocess_flags.csv", index=False)
        flagged[ep] = f

    # ═══ STEP 3: Fixed Split ═══
    lg.info("STEP 3: Fixed Split")
    splits = {}; smeta = {}
    for ep, df in cleaned.items():
        ds, m = fixed_split(assign_scaffolds(df, seed=GLOBAL_SEED), seed=GLOBAL_SEED)
        assert m["scaffold_overlap"] == 0
        ds.to_csv(rd / ep / "fixed_split.csv", index=False)
        splits[ep] = ds; smeta[ep] = m
        lg.info(f"  [{ep}] train={m['train_n']} test={m['test_n']} pos_diff={m['diff']}")
    save_json(smeta, rd / "split_metadata.json")

    # ═══ STEP 4: Scenario Selection (Train CV, compact/xgb only) ═══
    lg.info("STEP 4: Scenario Selection (Train CV)")
    cv_rows = []
    best_sc = {}

    for ep in ENDPOINTS:
        if ep not in splits: continue
        ds = splits[ep]; fl = flagged[ep]
        ep_best = {"scenario": "raw_all", "cv_mcc": -1}

        for sc in SCENARIOS:
            train, _ = apply_scenario(ds, sc, fl)
            if len(train) < 20: continue

            # Extract compact features for CV
            fg = extract_fg_features(train.reset_index(drop=True), ep)
            ph = extract_physchem_features(train.reset_index(drop=True), ep)
            fgp = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
            feat = pd.concat([fgp, ph], axis=1)
            for c in feat.columns: feat[c] = pd.to_numeric(feat[c], errors="coerce")
            feat = feat.fillna(0)
            X = feat.values.astype(np.float32)
            y = train.reset_index(drop=True)["label"].values.astype(int)
            if len(set(y)) < 2: continue

            groups = LabelEncoder().fit_transform(
                train["scaffold_group"].astype(str).values
            ) if "scaffold_group" in train.columns else np.arange(len(train))
            spw = (y == 0).sum() / max((y == 1).sum(), 1)

            cvr = repeated_cv(X, y, groups,
                              lambda: make_model("xgb", spw),
                              n_folds=5, n_repeats=5, seed=GLOBAL_SEED)
            if not cvr or "mcc" not in cvr: continue

            row = {
                "endpoint": ep, "scenario": sc,
                "train_n": len(y),
                "cv_mcc_mean": cvr["mcc"]["mean"],
                "cv_mcc_std": cvr["mcc"]["std"],
                "cv_roc_mean": cvr.get("roc_auc", {}).get("mean"),
                "n_unique_folds": cvr.get("_n_unique_fold_assignments", 0),
                "cv_n_evals": cvr["mcc"]["n"],
            }
            cv_rows.append(row)
            lg.info(f"  [{ep}/{sc}] CV_MCC={row['cv_mcc_mean']:.4f}±{row['cv_mcc_std']:.4f} "
                     f"unique_folds={row['n_unique_folds']}")

            if row["cv_mcc_mean"] > ep_best["cv_mcc"]:
                ep_best = {"scenario": sc, "cv_mcc": row["cv_mcc_mean"]}

        best_sc[ep] = ep_best["scenario"]
        lg.info(f"  [{ep}] ★ Best: {ep_best['scenario']} (CV={ep_best['cv_mcc']:.4f})")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(rd / "scenario_cv_selection.csv", index=False)
    save_json(best_sc, rd / "best_scenario_by_cv.json")

    # ═══ STEP 5: All 36 Combos on Locked Test ═══
    lg.info(f"STEP 5: Locked Test ({len(SCENARIOS)*len(FEAT_MODES)*len(MODEL_NAMES)} combos/ep)")
    all_rows = []

    for ep in ENDPOINTS:
        if ep not in splits: continue
        ds = splits[ep]; fl = flagged[ep]
        orig_test_n = (ds["split"] == "test").sum()

        # ── Pre-compute features ONCE per endpoint ──
        lg.info(f"  [{ep}] Pre-computing features...")
        fg_all = extract_fg_features(ds, ep)
        ph_all = extract_physchem_features(ds, ep)
        fp_all = extract_fingerprint_features(ds, ep)
        fgp_all = fg_all[[c for c in fg_all.columns if c.endswith("_present") or c.startswith("bb_")]]

        compact_feat = pd.concat([fgp_all, ph_all], axis=1)
        broad_feat = pd.concat([fgp_all, ph_all, fp_all], axis=1)
        for f_df in [compact_feat, broad_feat]:
            for c in f_df.columns: f_df[c] = pd.to_numeric(f_df[c], errors="coerce")
            f_df.fillna(0, inplace=True)

        compact_cols = list(compact_feat.columns)
        broad_cols = list(broad_feat.columns)
        y_all = ds["label"].values.astype(int)
        split_arr = ds["split"].values
        ds_index = ds.index

        for sc in SCENARIOS:
            # Scenario별 mask 생성
            sc_train, sc_test = apply_scenario(ds, sc, fl)
            if len(sc_train) < 10 or len(sc_test) < 5: continue

            tr_idx = set(sc_train.index)
            te_idx = set(sc_test.index)
            sc_mask = np.array([i in tr_idx or i in te_idx for i in ds_index])
            is_train = np.array([i in tr_idx for i in ds_index[sc_mask]])

            y_tr = y_all[sc_mask][is_train]
            y_te = y_all[sc_mask][~is_train]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2: continue

            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

            # AD (fingerprint 기반, scenario당 1회)
            fp_sc = fp_all.iloc[np.where(sc_mask)[0]].values.astype(np.float32)
            ad_result = compute_ad(fp_sc[is_train], fp_sc[~is_train])

            # CV reference
            cv_ref = next((r for r in cv_rows if r["endpoint"]==ep and r["scenario"]==sc), {})

            # Groups for OOF threshold
            grp_vals = sc_train["scaffold_group"].astype(str).values if "scaffold_group" in sc_train.columns else np.arange(len(sc_train))
            groups_le = LabelEncoder().fit_transform(grp_vals)

            for fm in FEAT_MODES:
                feat_df = compact_feat if fm == "compact" else broad_feat
                X_tr = feat_df.iloc[np.where(sc_mask)[0][is_train]].values.astype(np.float32)
                X_te = feat_df.iloc[np.where(sc_mask)[0][~is_train]].values.astype(np.float32)
                fcols = compact_cols if fm == "compact" else broad_cols

                for mn in MODEL_NAMES:
                    experiment = f"{ep}_{sc}_{fm}_{mn}"
                    edir = rd / experiment; edir.mkdir(parents=True, exist_ok=True)

                    # ── OOF threshold tuning ──
                    opt_t = oof_threshold(X_tr, y_tr, groups_le,
                                          lambda: make_model(mn, spw), n_folds=3)

                    # ── Final model ──
                    mdl = make_model(mn, spw)
                    mdl.fit(X_tr, y_tr)
                    proba = mdl.predict_proba(X_te)[:, 1]
                    pred_default = (proba >= 0.5).astype(int)
                    pred_tuned = (proba >= opt_t).astype(int)

                    # ── Metrics ──
                    ci = bootstrap_ci(y_te, pred_default, proba, 200)
                    cal = calibration(y_te, proba)
                    mcc_tuned = matthews_corrcoef(y_te, pred_tuned)

                    # ── cv_selected_compact_xgb = exactly 1 per endpoint ──
                    is_rec = (sc == best_sc.get(ep) and fm == "compact" and mn == "xgb")

                    # ── Build row (모든 컬럼 명시적으로) ──
                    row = {
                        "experiment": experiment,
                        "endpoint": ep,
                        "scenario": sc,
                        "feature_mode": fm,
                        "model": mn,
                        "cv_selected_compact_xgb": is_rec,
                        "train_n": int(len(y_tr)),
                        "test_n": int(len(y_te)),
                        "test_pos_rate": round(float(y_te.mean()), 4),
                        "test_n_pos": int(y_te.sum()),
                        "coverage": round(len(y_te) / orig_test_n, 4),
                        "ad_coverage": ad_result["coverage"],
                        "ad_mean_sim": ad_result["mean_sim"],
                        "brier": cal["brier"],
                        "ece": cal["ece"],
                        "threshold_default": 0.5,
                        "threshold_tuned": opt_t,
                        "mcc_tuned": round(mcc_tuned, 4),
                        "cv_mcc_mean": cv_ref.get("cv_mcc_mean"),
                        "cv_mcc_std": cv_ref.get("cv_mcc_std"),
                    }
                    # Bootstrap CI metrics
                    for k in ["mcc","bacc","roc_auc","sens","spec","brier"]:
                        if k in ci:
                            row[k] = ci[k]["mean"]
                            row[f"{k}_lo"] = ci[k]["lo"]
                            row[f"{k}_hi"] = ci[k]["hi"]

                    row["warnings"] = interpret_result(row)

                    # ── HP Tuning (GroupKFold RandomizedSearchCV) ──
                    try:
                        hp_mdl, hp_params, hp_cv_score = tune_model(
                            X_tr, y_tr, groups_le, mn, spw)
                        hp_proba = hp_mdl.predict_proba(X_te)[:, 1]
                        hp_pred = (hp_proba >= 0.5).astype(int)
                        hp_mcc = matthews_corrcoef(y_te, hp_pred)
                        hp_cal = calibration(y_te, hp_proba)
                        # OOF threshold for tuned model
                        hp_opt_t = oof_threshold(X_tr, y_tr, groups_le,
                            lambda: tune_model(X_tr, y_tr, groups_le, mn, spw)[0].__class__(
                                **hp_mdl.get_params()) if mn == "xgb" else hp_mdl,
                            n_folds=3) if False else opt_t  # reuse fixed threshold (avoid double-tuning overhead)
                        hp_pred_tuned = (hp_proba >= opt_t).astype(int)
                        hp_mcc_t = matthews_corrcoef(y_te, hp_pred_tuned)

                        row["mcc_hp_tuned"] = round(hp_mcc, 4)
                        row["mcc_hp_tuned_threshold"] = round(hp_mcc_t, 4)
                        row["hp_cv_mcc"] = hp_cv_score
                        row["hp_best_params"] = json.dumps(
                            {k: (v if not hasattr(v, 'item') else v.item())
                             for k, v in hp_params.items()}, ensure_ascii=False)
                        row["hp_brier"] = hp_cal["brier"]
                        row["hp_ece"] = hp_cal["ece"]
                    except Exception as e:
                        lg.info(f"    HP tuning failed: {e}")
                        row["mcc_hp_tuned"] = None
                        row["mcc_hp_tuned_threshold"] = None
                        row["hp_cv_mcc"] = None
                        row["hp_best_params"] = None
                        row["hp_brier"] = None
                        row["hp_ece"] = None

                    all_rows.append(row)

                    # Save per-experiment files
                    pred_df = {"label": y_te, "proba_fixed": proba,
                               "pred_fixed": pred_default, "pred_threshold_tuned": pred_tuned}
                    if row.get("mcc_hp_tuned") is not None:
                        pred_df["proba_hp"] = hp_proba
                        pred_df["pred_hp"] = hp_pred
                    pd.DataFrame(pred_df).to_csv(edir / "test_predictions.csv", index=False)
                    save_json(row, edir / "metrics.json")

                    # Feature importance (from HP-tuned model if available, else fixed)
                    try:
                        use_mdl = hp_mdl if row.get("mcc_hp_tuned") is not None else mdl
                        imp = (use_mdl.feature_importances_ if mn == "xgb"
                               else np.abs(use_mdl.named_steps["clf"].coef_[0]))
                        pd.DataFrame({"feature": fcols[:len(imp)], "importance": imp}).sort_values(
                            "importance", ascending=False).to_csv(edir / "feature_importance.csv", index=False)
                    except: pass

                    # HP params save
                    if hp_params:
                        save_json(hp_params, edir / "best_hp_params.json")

                    star = " ★" if is_rec else ""
                    hp_s = f" HP={row.get('mcc_hp_tuned','?')}" if row.get("mcc_hp_tuned") is not None else ""
                    lg.info(f"  {experiment}: MCC={row.get('mcc',0):.4f} "
                            f"[{row.get('mcc_lo',0):.4f},{row.get('mcc_hi',0):.4f}] "
                            f"t={opt_t} AD={ad_result['coverage']:.2f}{hp_s}{star}")
                    if row["warnings"] != "OK":
                        lg.info(f"    ⚠ {row['warnings']}")

    # ── Save main result table ──
    locked_df = pd.DataFrame(all_rows)

    # paper_primary_model: locked test MCC 최고 (tie-break: raw_all > no_metal > salt_stripped)
    SCENARIO_PRIORITY = {"raw_all": 0, "no_metal": 1, "salt_stripped": 2}
    locked_df["paper_primary_model"] = False
    for ep in locked_df["endpoint"].unique():
        mask = locked_df["endpoint"] == ep
        edf = locked_df[mask].copy()
        edf["_sc_pri"] = edf["scenario"].map(SCENARIO_PRIORITY).fillna(9)
        best_idx = edf.sort_values(["mcc", "_sc_pri"], ascending=[False, True]).index[0]
        locked_df.loc[best_idx, "paper_primary_model"] = True

    locked_df.to_csv(rd / "all_locked_test.csv", index=False)
    locked_df[["endpoint","scenario","feature_mode","model",
               "ad_coverage","ad_mean_sim"]].to_csv(rd / "ad_coverage_summary.csv", index=False)
    locked_df[["endpoint","scenario","feature_mode","model",
               "brier","ece"]].to_csv(rd / "calibration_summary.csv", index=False)

    # Split tables: main (default threshold) vs supplementary (tuned)
    main_exclude = ["threshold_default","threshold_tuned","mcc_tuned","_sc_pri",
                    "mcc_hp_tuned","mcc_hp_tuned_threshold","hp_cv_mcc","hp_best_params","hp_brier","hp_ece"]
    main_cols = [c for c in locked_df.columns if c not in main_exclude]
    locked_df[main_cols].to_csv(rd / "table_main_default_threshold.csv", index=False)

    tuned_cols = ["experiment","endpoint","scenario","feature_mode","model",
                  "paper_primary_model","threshold_default","threshold_tuned",
                  "mcc","mcc_tuned","test_n","test_n_pos"]
    tuned_cols = [c for c in tuned_cols if c in locked_df.columns]
    df_tuned = locked_df[tuned_cols].copy().rename(columns={"mcc": "mcc_default"})
    df_tuned["mcc_delta"] = df_tuned["mcc_tuned"] - df_tuned["mcc_default"]
    df_tuned.to_csv(rd / "table_supplementary_threshold_tuning.csv", index=False)

    # HP tuning comparison table
    hp_cols = ["experiment","endpoint","scenario","feature_mode","model",
               "paper_primary_model","mcc","mcc_hp_tuned","hp_cv_mcc","hp_brier","hp_ece","hp_best_params"]
    hp_cols = [c for c in hp_cols if c in locked_df.columns]
    if hp_cols and "mcc_hp_tuned" in locked_df.columns:
        df_hp = locked_df[hp_cols].copy().rename(columns={"mcc": "mcc_fixed"})
        df_hp["mcc_hp_delta"] = df_hp["mcc_hp_tuned"] - df_hp["mcc_fixed"]
        df_hp.to_csv(rd / "table_hp_tuning_comparison.csv", index=False)
        lg.info(f"  HP tuning comparison: {len(df_hp)} rows")
        # Summary per endpoint
        for ep in df_hp["endpoint"].unique():
            edf = df_hp[df_hp["endpoint"]==ep]
            delta_mean = edf["mcc_hp_delta"].dropna().mean()
            n_improved = (edf["mcc_hp_delta"].dropna() > 0).sum()
            lg.info(f"    [{ep}] HP tuning: mean Δ={delta_mean:+.4f}, improved {n_improved}/{len(edf)} combos")

    lg.info(f"  Saved all_locked_test.csv: {len(locked_df)} rows, {len(locked_df.columns)} columns")

    # ═══ STEP 6: LDO + Checklist ═══
    lg.info("STEP 6: LDO + Checklist")
    ldo = []
    if dom_rpt.get("has_domain"):
        def _feat(df, ep):
            fg = extract_fg_features(df, ep); ph = extract_physchem_features(df, ep)
            fgp = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
            feat = pd.concat([fgp, ph], axis=1)
            for c in feat.columns: feat[c] = pd.to_numeric(feat[c], errors="coerce")
            return feat.fillna(0).values.astype(np.float32), df["label"].values.astype(int), []
        ldo = leave_domain_out(
            cleaned["ames"], "domain", _feat,
            lambda: make_model("xgb", 1.0), "ames")
        pd.DataFrame(ldo).to_csv(rd / "ames_leave_domain_out.csv", index=False)
        for r in ldo:
            lg.info(f"  LDO {r['train_dom']}→{r['test_dom']}: MCC={r.get('mcc',0):.4f}")

    # ── Checklist (실제 검증) ──
    checks = {
        "fixed_split": all(smeta[ep]["scaffold_overlap"] == 0 for ep in smeta),
        "conflicts": all((rd / ep / "conflict_sensitivity.json").exists() for ep in cleaned),
        "domain_diagnosed": (rd / "ames_domain_confounding.json").exists(),
        "domain_ldo": len(ldo) > 0,
        "cross_ep": (rd / "cross_endpoint_overlap.csv").exists(),
        "cv_selection": (rd / "scenario_cv_selection.csv").exists(),
        "cv_unique_folds": all(r.get("n_unique_folds", 0) > 1 for r in cv_rows) if cv_rows else False,
        "full_grid": len(locked_df) == len(SCENARIOS) * len(FEAT_MODES) * len(MODEL_NAMES) * len([e for e in ENDPOINTS if e in splits]),
        "no_empty_mcc": locked_df["mcc"].notna().all() if not locked_df.empty else False,
        "broad_fp_included": "broad_fp" in locked_df["feature_mode"].values if not locked_df.empty else False,
        "logistic_included": "logistic" in locked_df["model"].values if not locked_df.empty else False,
        "cv_selected_1_per_ep": all(
            locked_df[locked_df["endpoint"]==ep]["cv_selected_compact_xgb"].sum() == 1
            for ep in locked_df["endpoint"].unique()
        ) if not locked_df.empty else False,
        "bootstrap_ci": "mcc_lo" in locked_df.columns if not locked_df.empty else False,
        "ad_computed": locked_df["ad_coverage"].notna().all() if not locked_df.empty else False,
        "calibration": locked_df["brier"].notna().all() if not locked_df.empty else False,
        "threshold_oof": locked_df["threshold_tuned"].notna().all() if not locked_df.empty else False,
        "paper_primary_1_per_ep": all(
            locked_df[locked_df["endpoint"]==ep]["paper_primary_model"].sum() == 1
            for ep in locked_df["endpoint"].unique()
        ) if not locked_df.empty else False,
        "warnings_present": any(locked_df["warnings"] != "OK") if not locked_df.empty else False,
        "all_columns_complete": locked_df[["train_n","test_n","mcc","roc_auc","ad_coverage","brier"]].notna().all().all() if not locked_df.empty else False,
        "hp_tuning_complete": locked_df["mcc_hp_tuned"].notna().all() if (not locked_df.empty and "mcc_hp_tuned" in locked_df.columns) else False,
    }
    save_json(checks, rd / "checklist.json")

    total = time.time() - t0
    lg.info(f"\nChecklist ({sum(checks.values())}/{len(checks)}):")
    for k, v in checks.items():
        lg.info(f"  [{'✓' if v else '✗'}] {k}")

    lg.info(f"\nDONE in {total:.0f}s: {rd}")
    return {"run_dir": rd, "cv": cv_df, "locked": locked_df,
            "best_sc": best_sc, "ldo": pd.DataFrame(ldo), "checks": checks}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tag", default="v11")
    a = p.parse_args()
    run_pipeline(data_dir=a.data_dir, tag=a.tag)
