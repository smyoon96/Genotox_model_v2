"""
deploy_pipeline.py — 배포용 모델 재학습 + SHAP + Threshold 최적화
===================================================================
genotox_pipeline.py (v4)의 결과를 바탕으로 best model을 전체 데이터로
재학습하여 배포용 모델을 생성한다.

처리 흐름 (genotox_pipeline과 동일한 전처리 보장):
  STEP 1: Load + resolve_conflicts (label 충돌 해소)
  STEP 2: classify_all_compounds (전처리 플래그)
  STEP 3: apply_scenario (best scenario 적용)
  STEP 4: Feature 추출 (FG + PhysChem + SA + Delta + Electronic + Route + FP)
  STEP 5: OOF CV → Threshold 최적화 (sensitivity 우선)
  STEP 6: 전체 데이터 재학습
  STEP 7: SHAP 분석 + 저장
  STEP 8: Model + Metadata 저장

Usage:
    python deploy_pipeline.py \\
        --results-dir ./runs/20260527_091038_v12_extened \\
        --data-dir ./data \\
        --output-dir ./output/deploy

배포 후 예측:
    from deploy_pipeline import predict_from_smiles
    results = predict_from_smiles(["CCO", "c1ccccc1N"], Path("./output/deploy"))
"""

import sys, json, logging, argparse, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
lg = logging.getLogger("deploy")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ── v4 파이프라인 모듈 import ──
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from pipeline_v2_core import (
    resolve_conflicts, find_smi, METAL_NUMS, ANALYSIS_SMILES_COL,
    apply_scenario as core_apply_scenario,
    bootstrap_ci, calibration,
)
from step2b_preprocessing_impact import classify_all_compounds
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features,
    extract_fingerprint_features, select_fingerprint_bits,
    extract_multi_fp_union, extract_qm_features,
)
from step4b_sa_features import extract_sa_features
from step4c_delta_descriptors import (
    extract_delta_features, extract_extended_descriptors,
    extract_electronic_descriptors,
)
from step0_conditional_router import ConditionalRouter
from config import (
    DATA_DIR, GLOBAL_SEED, ENDPOINTS,
    CROSS_ENDPOINT_STACKING, ENSEMBLE_CONFIG,
)

SCENARIOS = ["raw_all", "no_metal", "salt_stripped", "conditional"]
QM_DIR = DATA_DIR / "qm" if (DATA_DIR / "qm").exists() else None

SPEC_FLOORS = {
    "ames": 0.70, "invitro": 0.65, "invivo": 0.75,
    "invitro_sampling": 0.65, "invivo_sampling": 0.75,
}


# ═══════════════════════════════════════════════════════
#  STEP 1: Best model 식별
# ═══════════════════════════════════════════════════════

def identify_best_models(results_dir: Path) -> dict:
    """all_locked_test.csv에서 endpoint별 best config 추출."""
    csv_path = results_dir / "all_locked_test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    best = {}
    for ep in df["endpoint"].unique():
        row = df[df["endpoint"] == ep].sort_values("mcc", ascending=False).iloc[0]
        best[ep] = {
            "experiment": row["experiment"],
            "scenario": row["scenario"],
            "feature_mode": row["feature_mode"],
            "model": row["model"],
            "mcc": float(row["mcc"]),
        }
        lg.info(f"  [{ep}] {row['model']} / {row['scenario']} / "
                f"{row['feature_mode']} (MCC={row['mcc']:.4f})")
    return best


# ═══════════════════════════════════════════════════════
#  STEP 1+2: Load + Clean + Preprocess Flags
# ═══════════════════════════════════════════════════════

def load_and_clean(data_dir: Path, endpoint: str) -> tuple:
    """
    데이터 로드 → resolve_conflicts → classify_all_compounds.
    genotox_pipeline.py STEP 1 + STEP 2 동일 로직.

    Returns: (cleaned_df, flag_df)
    """
    # Load
    df = None
    for ext in [".csv", ".xlsx"]:
        fp = data_dir / f"{endpoint}{ext}"
        if fp.exists():
            df = (pd.read_csv(fp, encoding="utf-8-sig") if ext == ".csv"
                  else pd.read_excel(fp, engine="openpyxl"))
            break
    if df is None:
        raise FileNotFoundError(f"Data not found: {endpoint} in {data_dir}")

    for c in df.columns:
        if c.strip().upper() == "SMILES":
            df = df.rename(columns={c: "SMILES"})
    df["label"] = df["label"].astype(int)
    df["endpoint"] = endpoint
    df = df.dropna(subset=["label", "SMILES"])
    n_raw = len(df)

    # STEP 1: resolve_conflicts (conservative)
    cleaned, _, rpt = resolve_conflicts(df, "conservative")
    n_conflicts = rpt.get("n_conflicts", 0)
    lg.info(f"  [{endpoint}] Load: {n_raw} → {len(cleaned)} "
            f"(conflicts={n_conflicts})")

    # STEP 2: classify_all_compounds
    try:
        flags = classify_all_compounds(cleaned, smi_col="SMILES")
        n_metal = flags["has_metal"].sum() if "has_metal" in flags.columns else 0
        n_salt = flags["has_salt"].sum() if "has_salt" in flags.columns else 0
        lg.info(f"  [{endpoint}] Flags: metals={n_metal}, salts={n_salt}")
    except Exception as e:
        lg.warning(f"  [{endpoint}] Flags skipped: {e}")
        flags = pd.DataFrame(index=cleaned.index)

    return cleaned, flags


# ═══════════════════════════════════════════════════════
#  STEP 3: Scenario Application
# ═══════════════════════════════════════════════════════

def apply_scenario_deploy(df: pd.DataFrame, flags: pd.DataFrame,
                           scenario: str, router=None) -> pd.DataFrame:
    """
    genotox_pipeline STEP 5의 apply_scenario 재현.
    flags에서 has_metal 등 활용.
    """
    df = df.copy()

    if scenario == "raw_all":
        df["_analysis_smiles"] = df["SMILES"]
        return df

    if scenario in ("salt_stripped", "conditional"):
        from rdkit.Chem.MolStandardize import rdMolStandardize
        stripped = []
        for smi in df["SMILES"]:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    stripped.append(smi); continue
                parent = rdMolStandardize.FragmentParent(mol)
                stripped.append(Chem.MolToSmiles(parent, canonical=True))
            except Exception:
                stripped.append(smi)
        df["_analysis_smiles"] = stripped
        return df

    if scenario == "no_metal":
        if "has_metal" in flags.columns:
            keep_mask = ~flags["has_metal"].values[:len(df)]
            df = df[keep_mask].copy()
        else:
            keep = []
            for smi in df["SMILES"]:
                try:
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol is None: keep.append(False); continue
                    keep.append(not any(a.GetAtomicNum() in METAL_NUMS
                                       for a in mol.GetAtoms()))
                except Exception:
                    keep.append(True)
            df = df[keep].copy()
        df["_analysis_smiles"] = df["SMILES"]
        return df.reset_index(drop=True)

    df["_analysis_smiles"] = df["SMILES"]
    return df


# ═══════════════════════════════════════════════════════
#  STEP 4: Feature Building (v4 전체 재현)
# ═══════════════════════════════════════════════════════

def build_features_v4(df: pd.DataFrame, endpoint: str,
                       feature_mode: str, scenario: str,
                       router=None, ames_model=None) -> pd.DataFrame:
    """
    genotox_pipeline.py STEP 5의 feature 빌드 로직 완전 재현.
    compact / extended / broad_fpN / multi_fp 모두 지원.
    """
    # ── 기본 feature 추출 ──
    fg = extract_fg_features(df, endpoint)
    ph = extract_physchem_features(df, endpoint)

    # ── SA features ──
    try:
        sa = extract_sa_features(df, endpoint)
    except Exception:
        sa = pd.DataFrame(index=df.index)

    # ── Delta features (전처리 전후 차이) ──
    try:
        smi_col = find_smi(df)
        delta = extract_delta_features(
            df, raw_smi_col="SMILES", pre_smi_col=smi_col, endpoint=endpoint)
    except Exception:
        delta = pd.DataFrame(index=df.index)

    # ── Electronic descriptors (확장 VSA, EState, Gasteiger, Kier) ──
    try:
        elec = extract_electronic_descriptors(df, endpoint)
    except Exception:
        elec = pd.DataFrame(index=df.index)

    # ── Route features (conditional only) ──
    route = pd.DataFrame(index=df.index)
    if scenario == "conditional" and router is not None:
        try:
            route = router.get_route_features(df)
        except Exception:
            pass

    # ── QM features (optional) ──
    try:
        qm, _ = extract_qm_features(df, endpoint, qm_dir=QM_DIR)
    except Exception:
        qm = pd.DataFrame(index=df.index)

    # ── Feature mode 조립 ──
    if feature_mode == "compact":
        parts = [fg, ph]
        if not qm.empty: parts.append(qm)
        feat = pd.concat(parts, axis=1)

    elif feature_mode == "extended":
        parts = [fg, ph, sa, delta, elec]
        if not qm.empty: parts.append(qm)
        if not route.empty: parts.append(route)
        feat = pd.concat(parts, axis=1)

    elif feature_mode.startswith("broad_fp"):
        nbits = int(feature_mode.replace("broad_fp", ""))
        fp = extract_fingerprint_features(df, endpoint, n_bits=nbits)
        y = df["label"].values.astype(int)
        top_k = {512: 128, 1024: 256, 2048: 384}.get(nbits, nbits // 4)
        selected = select_fingerprint_bits(fp, y, top_k=top_k, method="mi")
        fp_sel = fp[selected]
        parts = [fg, ph, sa, fp_sel]
        if not qm.empty: parts.append(qm)
        feat = pd.concat(parts, axis=1)

    elif feature_mode == "multi_fp":
        mfp = extract_multi_fp_union(df, endpoint,
                                      morgan_bits=1024, ap_bits=512, tt_bits=512)
        y = df["label"].values.astype(int)
        selected = select_fingerprint_bits(mfp, y, top_k=384, method="mi")
        mfp_sel = mfp[selected]
        parts = [fg, ph, sa, mfp_sel]
        if not qm.empty: parts.append(qm)
        feat = pd.concat(parts, axis=1)

    else:
        lg.warning(f"  Unknown feature_mode: {feature_mode}, using compact")
        feat = pd.concat([fg, ph], axis=1)

    # ── Ames cross-endpoint stacking feature ──
    if ames_model is not None and endpoint != "ames":
        try:
            mdl_info = ames_model
            ames_mdl = mdl_info["model"]
            ames_fcols = mdl_info["feature_cols"]
            ames_feat = feat.reindex(columns=ames_fcols, fill_value=0)
            ames_proba = ames_mdl.predict_proba(
                ames_feat.values.astype(np.float32))[:, 1]

            # Domain weight 적용
            dw = mdl_info.get("domain_weights")
            if dw and "domain" in df.columns:
                dom = df["domain"].str.lower().str.strip().values
                w = np.ones(len(ames_proba), dtype=np.float32)
                w[dom == "industrial"] = dw["w_industrial"]
                w[dom == "drug"] = dw["w_drug"]
                ames_proba = np.clip(ames_proba * w, 0, 1)

            feat["ames_proba"] = ames_proba
            lg.info(f"  Ames stacking added (mean={ames_proba.mean():.3f})")
        except Exception as e:
            lg.warning(f"  Ames stacking failed: {e}")

    # ── 정리 ──
    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat = feat.fillna(0).replace([np.inf, -np.inf], 0)

    # constant column 제거
    nunique = feat.nunique()
    feat = feat.loc[:, nunique > 1]

    lg.info(f"  Features: {feat.shape[1]} cols")
    return feat


# ═══════════════════════════════════════════════════════
#  모델 생성 (genotox_pipeline.make_model 동일)
# ═══════════════════════════════════════════════════════

def make_model(name: str, spw: float, seed: int = GLOBAL_SEED):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0)
    elif name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.1,
            scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1)
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced",
            random_state=seed, n_jobs=-1)
    elif name == "svm":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.preprocessing import StandardScaler
        return SKPipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(max_iter=5000,
                                                       random_state=seed)))])
    elif name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.preprocessing import StandardScaler
        return SKPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="saga",
                                        penalty="l2", random_state=seed))])
    else:
        raise ValueError(f"Unsupported model: {name}")


# ═══════════════════════════════════════════════════════
#  Domain Weight 로드
# ═══════════════════════════════════════════════════════

def load_domain_weights(results_dir: Path, scenario: str,
                         feature_mode: str) -> dict:
    """결과에서 CV 최적화된 domain weight 로드."""
    pattern = f"domain_weight_grid_ames_{scenario}_{feature_mode}.csv"
    dw_path = results_dir / pattern
    if not dw_path.exists():
        lg.info(f"  Domain weight not found: {pattern}")
        return {"w_industrial": 1.0, "w_drug": 1.0}

    dw = pd.read_csv(dw_path)
    best = dw.sort_values("score", ascending=False).iloc[0]
    w = {"w_industrial": float(best["w_industrial"]),
         "w_drug": float(best["w_drug"])}
    lg.info(f"  Domain weights: ind={w['w_industrial']:.2f}, "
            f"drug={w['w_drug']:.2f}")
    return w


# ═══════════════════════════════════════════════════════
#  Threshold 최적화 (sensitivity 우선)
# ═══════════════════════════════════════════════════════

def optimize_threshold(y_true, proba, spec_floor=0.70,
                       sens_weight=2.0) -> dict:
    """
    score = sens_weight × sensitivity + specificity 최대화.
    단, specificity ≥ spec_floor.
    """
    results = []
    for t in np.arange(0.10, 0.90, 0.005):
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        mcc = matthews_corrcoef(y_true, pred) if len(set(pred)) > 1 else 0.0
        results.append({"threshold": round(float(t), 4),
                        "sensitivity": round(sens, 4),
                        "specificity": round(spec, 4),
                        "mcc": round(mcc, 4),
                        "bacc": round((sens + spec) / 2, 4),
                        "score": round(sens_weight * sens + spec, 4)})

    df = pd.DataFrame(results)
    valid = df[df["specificity"] >= spec_floor]
    best = (valid.loc[valid["score"].idxmax()] if not valid.empty
            else df.loc[df["mcc"].idxmax()])
    return {"optimal_threshold": float(best["threshold"]),
            "sensitivity": float(best["sensitivity"]),
            "specificity": float(best["specificity"]),
            "mcc": float(best["mcc"]),
            "bacc": float(best["bacc"]),
            "all_thresholds": df}


# ═══════════════════════════════════════════════════════
#  SHAP 분석
# ═══════════════════════════════════════════════════════

def compute_shap(model, X, feature_cols, out_dir: Path,
                  model_name: str, y=None, max_display=25,
                  max_samples=2000):
    """
    SHAP 분석 + 시각화 (XGBoost/LightGBM/RF/Pipeline 모두 지원).

    출력:
      shap_summary.png / shap_bar.png / shap_waterfall.png /
      shap_top5_dep.png / shap_values.csv / shap_full_values.csv.gz
    """
    import shap
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_total = len(X)
    X_arr = X if isinstance(X, np.ndarray) else X.values

    # ── 대용량 샘플링 ──
    if n_total > max_samples:
        lg.info(f"  SHAP: sampling {max_samples}/{n_total}...")
        rng = np.random.RandomState(GLOBAL_SEED)
        if y is not None and len(np.unique(y)) == 2:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            pos_rate = len(pos_idx) / n_total
            n_pos_s = max(10, int(max_samples * pos_rate))
            n_neg_s = max_samples - n_pos_s
            sel = np.sort(np.concatenate([
                rng.choice(pos_idx, min(n_pos_s, len(pos_idx)), replace=False),
                rng.choice(neg_idx, min(n_neg_s, len(neg_idx)), replace=False),
            ]))
        else:
            sel = np.sort(rng.choice(n_total, max_samples, replace=False))
        X_shap = X_arr[sel].astype(np.float32)
        y_shap = y[sel] if y is not None else None
    else:
        X_shap = X_arr.astype(np.float32)
        y_shap = y
        sel = np.arange(n_total)

    lg.info(f"  SHAP: {len(X_shap)} samples × {len(feature_cols)} features")

    # ── SHAP values 계산 (XGBoost booster 우선) ──
    sv = None
    base_value = 0.0

    # DataFrame으로 통일
    X_shap_df = pd.DataFrame(X_shap, columns=feature_cols)

    # Method 1a: XGBoost native SHAP (pred_contribs — 가장 안정적)
    if model_name == "xgb" and sv is None:
        try:
            import xgboost
            lg.info(f"  Trying XGBoost native pred_contribs...")
            booster = model.get_booster()
            dmat = xgboost.DMatrix(X_shap, feature_names=feature_cols)
            # pred_contribs: (n_samples, n_features+1), 마지막 열 = bias
            contribs = booster.predict(dmat, pred_contribs=True)
            sv = contribs[:, :-1]       # feature contributions
            base_value = float(contribs[0, -1])  # bias term
            lg.info(f"  XGB native OK: shape={sv.shape}, base={base_value:.4f}")
        except Exception as e0:
            lg.warning(f"  XGB native failed: {e0}")
            sv = None

    # Method 1b: TreeExplainer (LGBM, RF)
    if model_name in ("lgbm", "rf") and sv is None:
        try:
            lg.info(f"  Trying TreeExplainer...")
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(X_shap_df)

            if isinstance(raw, list):
                sv = np.array(raw[1] if len(raw) == 2 else raw[0])
            elif isinstance(raw, np.ndarray):
                sv = raw[:, :, 1] if raw.ndim == 3 else raw
            else:
                sv = np.array(raw)

            ev = explainer.expected_value
            base_value = float(ev[1] if isinstance(ev, (list, np.ndarray))
                               and len(ev) > 1 else ev)
            lg.info(f"  TreeExplainer OK: shape={sv.shape}")
        except Exception as e1:
            lg.warning(f"  TreeExplainer failed: {e1}")
            sv = None

    # Method 2: shap.Explainer (unified API — 모든 모델 fallback)
    if sv is None:
        try:
            lg.info(f"  Trying shap.Explainer...")
            explainer2 = shap.Explainer(model, X_shap_df)
            explanation = explainer2(X_shap_df)
            sv = explanation.values
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            base_value = float(explanation.base_values.mean()
                               if hasattr(explanation.base_values, 'mean')
                               else explanation.base_values[0])
            lg.info(f"  shap.Explainer OK: shape={sv.shape}")
        except Exception as e2:
            lg.warning(f"  shap.Explainer failed: {e2}")
            sv = None

    # Method 3: KernelExplainer (최후 수단, 샘플 축소)
    if sv is None:
        try:
            lg.info(f"  Trying KernelExplainer (last resort)...")
            n_kern = min(300, len(X_shap))
            bg = shap.sample(X_shap_df, min(50, n_kern))
            ke = shap.KernelExplainer(model.predict_proba, bg)
            raw = ke.shap_values(X_shap_df.iloc[:n_kern])
            sv = raw[1] if isinstance(raw, list) else raw
            base_value = float(ke.expected_value[1]
                               if isinstance(ke.expected_value, (list, np.ndarray))
                               else ke.expected_value)
            # X_shap도 축소
            X_shap = X_shap[:n_kern]
            X_shap_df = X_shap_df.iloc[:n_kern]
            if y_shap is not None:
                y_shap = y_shap[:n_kern]
            lg.info(f"  KernelExplainer OK: shape={sv.shape}")
        except Exception as e3:
            lg.error(f"  All SHAP methods failed: {e3}")
            import traceback; traceback.print_exc()
            return None

    sv = np.array(sv, dtype=np.float64)
    X_shap_df = pd.DataFrame(X_shap, columns=feature_cols)

    # ── 1. Summary beeswarm ──
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_shap_df, max_display=max_display,
                          show=False, plot_size=None)
        plt.title("SHAP Summary (Beeswarm)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        lg.info(f"  ✓ shap_summary.png")
    except Exception as e:
        lg.warning(f"  Beeswarm failed: {e}")
        plt.close("all")

    # ── 2. Bar plot ──
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_shap_df, plot_type="bar",
                          max_display=max_display, show=False, plot_size=None)
        plt.title("SHAP Feature Importance", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        lg.info(f"  ✓ shap_bar.png")
    except Exception as e:
        lg.warning(f"  Bar plot failed: {e}")
        plt.close("all")

    # ── 3. Waterfall ──
    try:
        # 양성 중 가장 확신 높은 건 선택
        idx = 0
        if y_shap is not None and (y_shap == 1).any():
            try:
                proba_s = model.predict_proba(X_shap)[:, 1]
                proba_s[y_shap != 1] = -1
                idx = int(np.argmax(proba_s))
            except Exception:
                idx = int(np.where(y_shap == 1)[0][0])

        exp_obj = shap.Explanation(
            values=sv[idx],
            base_values=base_value,
            data=X_shap[idx],
            feature_names=feature_cols,
        )
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(exp_obj, max_display=15, show=False)
        plt.title("SHAP Waterfall (positive prediction)", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        lg.info(f"  ✓ shap_waterfall.png")
    except Exception as e:
        lg.warning(f"  Waterfall failed: {e}")
        plt.close("all")

    # ── 4. Top 5 dependence ──
    try:
        mean_abs = np.abs(sv).mean(axis=0)
        top5 = np.argsort(mean_abs)[::-1][:5]
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i, fi in enumerate(top5):
            ax = axes[i]
            ax.scatter(X_shap_df.iloc[:, fi].values, sv[:, fi],
                       c=sv[:, fi], cmap="RdBu_r", s=8, alpha=0.6,
                       edgecolors="none")
            ax.set_xlabel(feature_cols[fi], fontsize=9)
            ax.set_ylabel("SHAP" if i == 0 else "", fontsize=9)
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.set_title(f"#{i+1}", fontsize=11)
        plt.suptitle("Top 5 Feature Dependence", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_top5_dep.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        lg.info(f"  ✓ shap_top5_dep.png")
    except Exception as e:
        lg.warning(f"  Dependence failed: {e}")
        plt.close("all")

    # ── 5. CSV ──
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(out_dir / "shap_values.csv", index=False)

    pd.DataFrame(sv, columns=feature_cols).to_csv(
        out_dir / "shap_full_values.csv.gz", index=False, compression="gzip")

    lg.info(f"  SHAP complete: {len(shap_df)} features")
    return shap_df


# ═══════════════════════════════════════════════════════
#  Threshold curve 시각화
# ═══════════════════════════════════════════════════════

def plot_threshold_curve(thresh: dict, out_dir: Path, ep: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = thresh["all_thresholds"]
    opt_t = thresh["optimal_threshold"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold"], df["sensitivity"], "b-", lw=2, label="Sensitivity")
    ax.plot(df["threshold"], df["specificity"], "r-", lw=2, label="Specificity")
    ax.plot(df["threshold"], df["mcc"], "g--", lw=1.5, label="MCC")
    ax.axvline(x=opt_t, color="k", ls="--", alpha=0.7,
               label=f"Optimal t={opt_t:.3f}")
    ax.axvline(x=0.5, color="gray", ls=":", alpha=0.5, label="Default t=0.5")

    opt_row = df.iloc[(df["threshold"] - opt_t).abs().idxmin()]
    ax.annotate(f"t={opt_t:.3f}\nSens={opt_row['sensitivity']:.3f}\n"
                f"Spec={opt_row['specificity']:.3f}\nMCC={opt_row['mcc']:.3f}",
                xy=(opt_t, opt_row["sensitivity"]),
                xytext=(opt_t + 0.08, opt_row["sensitivity"] - 0.1),
                fontsize=9, arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"{ep} — Threshold (sensitivity-prioritized)")
    ax.legend(fontsize=10); ax.set_xlim(0.1, 0.9); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════
#  단일 endpoint 배포 빌드
# ═══════════════════════════════════════════════════════

def build_endpoint(ep, cfg, data_dir, results_dir, output_dir,
                    router=None, ames_model=None):
    """
    한 endpoint에 대해:
      Load+Clean → Scenario → Features → OOF CV → Threshold →
      Full retrain → SHAP → Save
    """
    t0 = time.time()
    ep_dir = output_dir / ep
    ep_dir.mkdir(parents=True, exist_ok=True)

    scenario = cfg["scenario"]
    feature_mode = cfg["feature_mode"]
    model_name = cfg["model"]
    lg.info(f"  Config: {model_name} / {scenario} / {feature_mode}")

    # ── STEP 1+2: Load + Clean + Flags ──
    df, flags = load_and_clean(data_dir, ep)

    # ── STEP 3: Scenario ──
    df = apply_scenario_deploy(df, flags, scenario, router)
    lg.info(f"  After {scenario}: {len(df)} rows")

    # ── STEP 4: Features ──
    feat = build_features_v4(df, ep, feature_mode, scenario,
                              router=router, ames_model=ames_model)
    feature_cols = list(feat.columns)
    X = feat.values.astype(np.float32)
    y = df["label"].values.astype(int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    spw = n_neg / max(n_pos, 1)

    # ── Domain weight ──
    dw = {"w_industrial": 1.0, "w_drug": 1.0}
    if ep == "ames" and results_dir:
        dw = load_domain_weights(results_dir, scenario, feature_mode)

    def _make_sw(y_subset_idx=None):
        if ep != "ames" or "domain" not in df.columns:
            return {}
        if model_name not in ("xgb", "lgbm", "rf"):
            return {}
        dom = df["domain"].str.lower().str.strip().values
        sw = np.ones(len(y), dtype=np.float32)
        sw[dom == "industrial"] = dw["w_industrial"]
        sw[dom == "drug"] = dw["w_drug"]
        sw /= sw.mean()
        if y_subset_idx is not None:
            return {"sample_weight": sw[y_subset_idx]}
        return {"sample_weight": sw}

    # ── STEP 5: OOF CV → Threshold ──
    lg.info(f"  5-fold OOF CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    oof_proba = np.zeros(len(y))
    oof_mccs = []

    for tr_idx, val_idx in skf.split(X, y):
        mdl = make_model(model_name, spw)
        mdl.fit(X[tr_idx], y[tr_idx], **_make_sw(tr_idx))
        oof_proba[val_idx] = mdl.predict_proba(X[val_idx])[:, 1]
        oof_mccs.append(matthews_corrcoef(
            y[val_idx], (oof_proba[val_idx] >= 0.5).astype(int)))

    cv_mcc = float(np.mean(oof_mccs))
    lg.info(f"  CV MCC: {cv_mcc:.4f} ± {np.std(oof_mccs):.4f}")

    spec_floor = SPEC_FLOORS.get(ep, 0.70)
    thresh = optimize_threshold(y, oof_proba, spec_floor, sens_weight=2.0)
    opt_t = thresh["optimal_threshold"]
    lg.info(f"  Threshold: {opt_t:.3f} → sens={thresh['sensitivity']:.3f} "
            f"spec={thresh['specificity']:.3f} mcc={thresh['mcc']:.3f}")

    thresh["all_thresholds"].to_csv(ep_dir / "threshold_search.csv", index=False)
    plot_threshold_curve(thresh, ep_dir, ep)

    # ── STEP 6: 전체 데이터 재학습 ──
    lg.info(f"  Full retrain on {len(X)} samples...")
    final_model = make_model(model_name, spw)
    final_model.fit(X, y, **_make_sw())

    # ── STEP 7: SHAP ──
    try:
        shap_df = compute_shap(final_model, X, feature_cols,
                                ep_dir, model_name, y=y)
    except Exception as e:
        lg.warning(f"  SHAP failed: {e}")
        import traceback; traceback.print_exc()
        shap_df = None

    # ── STEP 8: Save ──
    joblib.dump(final_model, ep_dir / "model.joblib")
    with open(ep_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "endpoint": ep,
        "model_type": model_name,
        "scenario": scenario,
        "feature_mode": feature_mode,
        "n_features": len(feature_cols),
        "n_total": len(X),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "cv_mcc": round(cv_mcc, 4),
        "cv_mcc_std": round(float(np.std(oof_mccs)), 4),
        "mcc_default_05": round(float(matthews_corrcoef(
            y, (oof_proba >= 0.5).astype(int))), 4),
        "optimal_threshold": round(opt_t, 4),
        "sensitivity": round(thresh["sensitivity"], 4),
        "specificity": round(thresh["specificity"], 4),
        "mcc_at_threshold": round(thresh["mcc"], 4),
        "bacc_at_threshold": round(thresh["bacc"], 4),
        "spec_floor": spec_floor,
        "domain_weights": dw if ep == "ames" else None,
        "has_ames_stacking": "ames_proba" in feature_cols,
        "elapsed_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
        "top_features": (shap_df[["feature", "mean_abs_shap"]]
                         .head(20).to_dict("records") if shap_df is not None else []),
    }
    with open(ep_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    lg.info(f"  Saved: {ep_dir} ({time.time()-t0:.0f}s)")

    return {
        "model": final_model,
        "feature_cols": feature_cols,
        "domain_weights": dw if ep == "ames" else None,
        "metadata": metadata,
    }


# ═══════════════════════════════════════════════════════
#  메인 실행
# ═══════════════════════════════════════════════════════

def run_deploy(results_dir: Path, data_dir: Path, output_dir: Path = None):
    t0 = time.time()
    output_dir = output_dir or (SCRIPT_DIR / "output" / "deploy")
    output_dir.mkdir(parents=True, exist_ok=True)

    lg.info("=" * 60)
    lg.info("  DEPLOYMENT PIPELINE (v4-compatible)")
    lg.info(f"  Results: {results_dir}")
    lg.info(f"  Data:    {data_dir}")
    lg.info(f"  Output:  {output_dir}")
    lg.info("=" * 60)

    # ── Best models ──
    lg.info("\n[1] Identifying best models...")
    best = identify_best_models(results_dir)

    # ── Conditional router ──
    router = ConditionalRouter()

    # ── Ames first (stacking source) ──
    ames_info = None
    if "ames" in best:
        lg.info(f"\n{'─'*50}")
        lg.info("[2] Ames (stacking source)")
        ames_info = build_endpoint(
            "ames", best["ames"], data_dir, results_dir, output_dir,
            router=router)

    # ── Remaining endpoints ──
    all_meta = {}
    if ames_info:
        all_meta["ames"] = ames_info["metadata"]

    for ep, cfg in best.items():
        if ep == "ames":
            continue
        lg.info(f"\n{'─'*50}")
        lg.info(f"[3] {ep}")
        info = build_endpoint(
            ep, cfg, data_dir, results_dir, output_dir,
            router=router, ames_model=ames_info)
        all_meta[ep] = info["metadata"]

    # ── Summary ──
    with open(output_dir / "deployment_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False, default=str)

    # ── Feature 사용 현황 CSV ──
    lg.info("\n[4] Generating feature usage report...")
    _build_feature_usage_csv(output_dir, all_meta)

    lg.info(f"\n{'='*60}")
    lg.info("  DEPLOYMENT COMPLETE")
    lg.info(f"{'='*60}")
    for ep, m in all_meta.items():
        lg.info(f"  [{ep:22s}] t={m['optimal_threshold']:.3f} "
                f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                f"mcc={m['mcc_at_threshold']:.3f} (CV={m['cv_mcc']:.3f})")
    lg.info(f"\n  Total: {time.time()-t0:.0f}s")


def _build_feature_usage_csv(output_dir: Path, all_meta: dict):
    """
    모든 endpoint의 feature 사용 현황을 하나의 CSV로 저장.

    출력: feature_usage_matrix.csv
      - 행: feature 이름
      - 열: endpoint별 사용 여부 (1/0) + feature 카테고리
    """
    # 각 endpoint의 feature 목록 수집
    ep_features = {}
    for ep_name, meta in all_meta.items():
        fcols_path = output_dir / ep_name / "feature_columns.json"
        if fcols_path.exists():
            with open(fcols_path) as f:
                ep_features[ep_name] = set(json.load(f))
        else:
            ep_features[ep_name] = set()

    # 전체 feature union
    all_feats = sorted(set().union(*ep_features.values()))
    if not all_feats:
        lg.warning("  No features found")
        return

    # Feature 카테고리 분류
    def categorize(fname):
        # BB alerts (check before generic fg_ prefix)
        if fname.startswith("fg_bb_"):     return "benigni_bossa_fg"
        if fname.startswith("rule_bb_"):   return "benigni_bossa_rule"
        if fname.startswith("bb_"):        return "benigni_bossa_summary"
        # Specific alert families (check before generic fg_)
        if fname.startswith("fg_kz_"):     return "kazius_alert"
        if fname.startswith("fg_mn_"):     return "micronucleus_alert"
        if fname.startswith("fg_iss_"):    return "in_silico_alert"
        if fname.startswith("fg_met_"):    return "metal_flag"
        # Generic FG
        if fname.startswith("fg_"):        return "functional_group"
        # Fingerprints
        if fname.startswith("fp_"):        return "morgan_fp"
        if fname.startswith("maccs_"):     return "maccs_key"
        if fname.startswith("ap_"):        return "atompair_fp"
        if fname.startswith("tt_"):        return "toptorsion_fp"
        # Step4b structural alerts
        if fname.startswith("SA"):         return "structural_alert_step4b"
        # Extended descriptors
        if fname.startswith("delta_"):     return "delta_descriptor"
        if any(fname.startswith(p) for p in
               ["slogp_vsa","smr_vsa","peoe_vsa"]): return "vsa_descriptor"
        if fname.startswith("estate_"):    return "estate_descriptor"
        if fname.startswith("gasteiger_"): return "gasteiger_charge"
        if fname.startswith("chi") or fname.startswith("kappa"):
            return "connectivity_index"
        if fname == "hall_kier_alpha":     return "connectivity_index"
        # Others
        if fname == "ames_proba":          return "cross_endpoint_stacking"
        if fname.startswith("route_"):     return "conditional_route"
        if fname.startswith("positive_alert") or fname.startswith("negative_alert"):
            return "alert_score"
        if fname.startswith("sa_total") or fname.startswith("n_metals"):
            return "alert_score"
        return "physicochemical"

    rows = []
    for feat in all_feats:
        row = {"feature": feat, "category": categorize(feat)}
        for ep_name in ep_features:
            row[ep_name] = 1 if feat in ep_features[ep_name] else 0
        row["n_endpoints"] = sum(row[ep] for ep in ep_features)
        rows.append(row)

    usage_df = pd.DataFrame(rows)

    # 정렬: 카테고리 → 사용 endpoint 수(내림차순) → 이름
    usage_df = usage_df.sort_values(
        ["category", "n_endpoints", "feature"],
        ascending=[True, False, True])

    usage_df.to_csv(output_dir / "feature_usage_matrix.csv", index=False)

    # 요약
    cat_summary = usage_df.groupby("category").agg(
        n_features=("feature", "count"),
        **{ep: (ep, "sum") for ep in ep_features}
    ).sort_values("n_features", ascending=False)
    cat_summary.to_csv(output_dir / "feature_category_summary.csv")

    lg.info(f"  ✓ feature_usage_matrix.csv ({len(usage_df)} features × "
            f"{len(ep_features)} endpoints)")
    lg.info(f"  ✓ feature_category_summary.csv ({len(cat_summary)} categories)")

    # 콘솔 요약
    for _, row in cat_summary.iterrows():
        lg.info(f"    {row.name:28s}: {int(row['n_features']):4d} features")


# ═══════════════════════════════════════════════════════
#  배포 후 예측 함수
# ═══════════════════════════════════════════════════════

def predict_from_smiles(smiles_list: list, model_dir: Path) -> pd.DataFrame:
    """배포된 모델로 SMILES 리스트 예측."""
    results = pd.DataFrame({"SMILES": smiles_list})

    for ep_dir in sorted(model_dir.iterdir()):
        if not ep_dir.is_dir(): continue
        meta_path = ep_dir / "metadata.json"
        model_path = ep_dir / "model.joblib"
        fcols_path = ep_dir / "feature_columns.json"
        if not all(p.exists() for p in [meta_path, model_path, fcols_path]):
            continue

        with open(meta_path) as f: meta = json.load(f)
        with open(fcols_path) as f: fcols = json.load(f)

        mdl = joblib.load(model_path)
        ep = meta["endpoint"]
        threshold = meta["optimal_threshold"]

        df = pd.DataFrame({"SMILES": smiles_list, "label": 0,
                           "_analysis_smiles": smiles_list})
        feat = build_features_v4(df, ep, meta["feature_mode"],
                                  meta["scenario"])
        feat = feat.reindex(columns=fcols, fill_value=0)
        proba = mdl.predict_proba(feat.values.astype(np.float32))[:, 1]
        pred = (proba >= threshold).astype(int)

        results[f"{ep}_proba"] = proba.round(4)
        results[f"{ep}_pred"] = pred
        results[f"{ep}_threshold"] = threshold

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Deployment Pipeline (v4)")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    run_deploy(
        Path(args.results_dir), Path(args.data_dir),
        Path(args.output_dir) if args.output_dir else None)
