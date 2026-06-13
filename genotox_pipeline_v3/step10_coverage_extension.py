"""
step10_coverage_extension.py -- 화학공간 커버리지 확장 및 FN 감소 전략
=======================================================================
v12.2 신규 step. 이전 step9 외부 검증에서 확인된 sensitivity 저하 문제를
구조적으로 보완하기 위한 전략 A~J 구현.

전략 목록:
  A) Diversity 기반 대표물질 샘플링 (Butina clustering)
  B) 강화된 AD: leverage + conformal prediction
  D) Cost-sensitive learning (focal loss, FN penalty sample_weight)
  F) SMOTE in fingerprint space
  I) Multi-model soft voting ensemble
  J) Endpoint별 meta-model (stacking)

사용:
  python step10_coverage_extension.py --run-dir runs/RUN_ID
  python step10_coverage_extension.py --run-dir runs/RUN_ID --strategy all
  python step10_coverage_extension.py --run-dir runs/RUN_ID --strategy A B D
"""

import sys, json, logging, argparse, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as cfg
from pipeline_v2_core import (
    to_can, find_smi, bootstrap_ci, compute_ad,
    oof_tune_threshold, ANALYSIS_SMILES_COL, SEED,
)
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features,
    extract_fingerprint_features, select_fingerprint_bits, build_all_features,
)
from utils.progress import pbar, step_header, task_done

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step10_coverage.log",
                            mode="w", encoding="utf-8"),
    ]
)
lg = logging.getLogger("step10")


# ═══════════════════════════════════════════════════════
#  공통 유틸
# ═══════════════════════════════════════════════════════

def _get_run_dir(run_dir_arg: str = None) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg)
    if cfg.LATEST_TXT.exists():
        p = Path(cfg.LATEST_TXT.read_text().strip())
        if p.exists():
            return p
    subdirs = sorted(cfg.RUNS_DIR.iterdir(),
                     key=lambda x: x.stat().st_mtime, reverse=True)
    for d in subdirs:
        if d.is_dir():
            return d
    raise FileNotFoundError("run_dir를 찾을 수 없음 -- --run-dir 지정 필요")


def _load_split(run_dir: Path, endpoint: str):
    p = run_dir / endpoint / "fixed_split.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    smi_col = find_smi(df)
    df[ANALYSIS_SMILES_COL] = df[smi_col]
    return df


def _fp_matrix(df: pd.DataFrame, n_bits: int = 1024) -> np.ndarray:
    """SMILES → Morgan FP numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in df[ANALYSIS_SMILES_COL]:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise ValueError
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            fps.append(list(fp))
        except Exception:
            fps.append([0] * n_bits)
    return np.array(fps, dtype=np.float32)


def _build_Xty(df: pd.DataFrame, endpoint: str,
               n_bits: int = 512, top_k: int = 128,
               fp_selected_bits=None):
    """
    v12.2: FG 전체 (615개) + Physchem (13개) + FP (MI-selected) 통합.
    기존: fgp = fg[[_present or bb_]] → ~123개만 사용 → 492개 누락
    수정: build_all_features()로 모든 블록 포함.
    """
    result = build_all_features(
        df, endpoint,
        include_fp=True, fp_n_bits=n_bits,
        fp_selected_bits=fp_selected_bits,
        fp_top_k=top_k, fp_method="mi",
        return_fp_full=True,
    )
    y = df["label"].values.astype(int) if "label" in df.columns else np.zeros(len(df), dtype=int)
    return result["X"], y, result["fp_selected"]


# ═══════════════════════════════════════════════════════
#  전략 A: Diversity 기반 대표물질 샘플링
# ═══════════════════════════════════════════════════════

def strategy_A_diversity_sampling(run_dir: Path,
                                   endpoints: list = None,
                                   n_clusters: int = 200,
                                   n_bits: int = 1024) -> dict:
    """
    Butina clustering으로 화학공간을 n_clusters 개 클러스터로 분할,
    각 클러스터에서 대표 물질 1개씩 샘플링.

    목적:
      - 현재 학습 데이터의 화학공간 편향 정량화
      - 커버리지가 낮은 클러스터 식별 → 외부 데이터 추가 우선순위 결정

    출력: diversity_coverage_{endpoint}.csv
      - cluster_id, n_train, n_test, representative_smiles, coverage_ratio
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    lg.info("=== Strategy A: Diversity-based Coverage Analysis ===")
    endpoints = endpoints or ["ames", "invitro", "invivo"]
    results = {}

    for ep in pbar(endpoints, desc="[A] Diversity sampling"):
        df = _load_split(run_dir, ep)
        if df is None:
            lg.warning(f"  [{ep}] split file not found -- skipping")
            continue

        t0 = time.time()
        lg.info(f"  [{ep}] Computing FP for {len(df)} compounds...")

        # Morgan FP → RDKit FP objects (Butina 필요)
        fps_rdkit = []
        valid_idx = []
        for i, smi in enumerate(df[ANALYSIS_SMILES_COL]):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                fps_rdkit.append(fp)
                valid_idx.append(i)
            except Exception:
                continue

        df_valid = df.iloc[valid_idx].copy().reset_index(drop=True)

        # Butina clustering (Tanimoto distance threshold=0.6)
        from rdkit.ML.Cluster import Butina
        dists = []
        n = len(fps_rdkit)
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps_rdkit[i], fps_rdkit[:i])
            dists.extend([1 - s for s in sims])

        tanimoto_threshold = 0.4  # 거리 > 0.4 → 다른 클러스터
        clusters = Butina.ClusterData(dists, n, tanimoto_threshold, isDistData=True)
        lg.info(f"  [{ep}] Butina: {n} compounds → {len(clusters)} clusters "
                f"(thr={tanimoto_threshold}) [{time.time()-t0:.0f}s]")

        # 클러스터별 train/test 분포
        cluster_rows = []
        train_mask = df_valid["split"] == "train"
        test_mask  = df_valid["split"] == "test"

        for ci, cluster_ids in enumerate(pbar(clusters, desc=f"  {ep} clusters", leave=False)):
            cluster_ids = list(cluster_ids)
            cl_df = df_valid.iloc[cluster_ids]
            n_tr = int(train_mask.iloc[cluster_ids].sum())
            n_te = int(test_mask.iloc[cluster_ids].sum())
            n_pos = int((cl_df["label"] == 1).sum())

            # 대표 물질: 클러스터 centroid에 가장 가까운 물질
            if len(cluster_ids) == 1:
                rep_idx = cluster_ids[0]
            else:
                cl_fps = [fps_rdkit[i] for i in cluster_ids]
                mean_sim = [
                    np.mean(DataStructs.BulkTanimotoSimilarity(cl_fps[j], cl_fps))
                    for j in range(len(cl_fps))
                ]
                rep_idx = cluster_ids[int(np.argmax(mean_sim))]

            cluster_rows.append({
                "endpoint": ep,
                "cluster_id": ci,
                "cluster_size": len(cluster_ids),
                "n_train": n_tr,
                "n_test": n_te,
                "n_positive": n_pos,
                "pos_rate": round(n_pos / max(len(cluster_ids), 1), 4),
                "coverage_ratio": round(n_tr / max(len(cluster_ids), 1), 4),
                "representative_smiles": df_valid[ANALYSIS_SMILES_COL].iloc[rep_idx],
                "train_only": n_tr > 0 and n_te == 0,
                "test_only":  n_tr == 0 and n_te > 0,   # ← 외부 데이터 추가 우선순위
            })

        cov_df = pd.DataFrame(cluster_rows)
        out_path = run_dir / f"diversity_coverage_{ep}.csv"
        cov_df.to_csv(out_path, index=False)

        # 요약
        n_test_only = int(cov_df["test_only"].sum())
        n_train_only = int(cov_df["train_only"].sum())
        coverage = round(1 - n_test_only / max(len(clusters), 1), 4)
        lg.info(f"  [{ep}] Coverage: {coverage:.3f} | "
                f"test-only clusters: {n_test_only} | "
                f"train-only clusters: {n_train_only}")
        lg.info(f"  [{ep}] → {out_path}")

        results[ep] = {
            "n_clusters": len(clusters),
            "coverage": coverage,
            "n_test_only": n_test_only,
        }

    return results


# ═══════════════════════════════════════════════════════
#  전략 B: 강화된 AD (leverage + conformal prediction)
# ═══════════════════════════════════════════════════════

def strategy_B_enhanced_ad(run_dir: Path,
                             endpoints: list = None,
                             n_bits: int = 512,
                             top_k: int = 128) -> dict:
    """
    3가지 AD 방법 비교:
      1) distance-based (k-NN Tanimoto) -- 기존
      2) leverage method (hat matrix diagonal)
      3) conformal prediction (nonconformity score 기반 p-value)

    출력: ad_comparison_{endpoint}.csv
      - per-compound: in_knn_ad, in_leverage_ad, conformal_pvalue, actual_label, predicted
    """
    from sklearn.metrics import confusion_matrix
    from xgboost import XGBClassifier

    lg.info("=== Strategy B: Enhanced Applicability Domain ===")
    endpoints = endpoints or ["ames", "invitro", "invivo"]
    results = {}

    for ep in pbar(endpoints, desc="[B] Enhanced AD"):
        df = _load_split(run_dir, ep)
        if df is None:
            continue

        df_tr = df[df["split"] == "train"].copy().reset_index(drop=True)
        df_te = df[df["split"] == "test"].copy().reset_index(drop=True)

        if len(df_tr) < 50 or len(df_te) < 10:
            lg.warning(f"  [{ep}] Too small -- skipping")
            continue

        lg.info(f"  [{ep}] Building features...")
        X_tr, y_tr, selected = _build_Xty(df_tr, ep, n_bits, top_k)
        X_te, y_te, _        = _build_Xty(df_te, ep, n_bits, top_k,
                                           ) if False else (None, None, None)
        # build X_te with same selected bits
        _te_res = build_all_features(df_te, ep, include_fp=True,
                                      fp_n_bits=n_bits, fp_selected_bits=selected,
                                      fp_top_k=top_k)
        X_te = _te_res["X"]
        y_te = df_te["label"].values.astype(int)

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=SEED, n_jobs=-1, verbosity=0)
        mdl.fit(X_tr, y_tr)
        proba_te = mdl.predict_proba(X_te)[:, 1]

        # ── AD 1: distance-based (k-NN Tanimoto) ──────────────────────
        fp_tr_sel = extract_fingerprint_features(df_tr, ep, n_bits=n_bits)[selected].values.astype(np.float32)
        fp_te_sel = extract_fingerprint_features(df_te, ep, n_bits=n_bits)[selected].values.astype(np.float32)
        ad_knn = compute_ad(fp_tr_sel, fp_te_sel, threshold=0.4)
        knn_sim = ad_knn["per_sample_max_sim"]
        in_knn  = knn_sim >= 0.4

        # ── AD 2: leverage method (hat matrix) ─────────────────────────
        # h_i = x_i^T (X^T X)^{-1} x_i
        # warning leverage: h_i > 3p/n (p=n_features, n=n_train)
        try:
            XtX = X_tr.T @ X_tr + np.eye(X_tr.shape[1]) * 1e-6  # ridge
            XtX_inv = np.linalg.inv(XtX)
            h_te = np.array([X_te[i] @ XtX_inv @ X_te[i]
                             for i in pbar(range(len(X_te)),
                                           desc=f"  {ep} leverage",
                                           leave=False)])
            h_tr = np.array([X_tr[i] @ XtX_inv @ X_tr[i]
                             for i in range(len(X_tr))])
            h_star = 3 * X_tr.shape[1] / len(X_tr)   # warning threshold
            in_leverage = h_te <= h_star
            lg.info(f"  [{ep}] Leverage h*={h_star:.4f} | "
                    f"in_AD: {in_leverage.mean()*100:.1f}%")
        except Exception as e:
            lg.warning(f"  [{ep}] Leverage failed: {e}")
            in_leverage = np.ones(len(X_te), dtype=bool)
            h_te = np.zeros(len(X_te))

        # ── AD 3: Conformal prediction (inductive) ─────────────────────
        # calibration set = 20% of train (stratified)
        # nonconformity score = 1 - p_hat(true_class)
        from sklearn.model_selection import train_test_split
        try:
            X_prop, X_cal, y_prop, y_cal = train_test_split(
                X_tr, y_tr, test_size=0.2, stratify=y_tr,
                random_state=SEED)
            cal_mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     scale_pos_weight=spw, eval_metric="logloss",
                                     random_state=SEED, n_jobs=-1, verbosity=0)
            cal_mdl.fit(X_prop, y_prop)
            cal_proba = cal_mdl.predict_proba(X_cal)
            # nonconformity: 1 - P(true_class)
            nc_scores = np.array([1 - cal_proba[i, y_cal[i]]
                                   for i in range(len(y_cal))])
            # test p-values: p(y) = #{cal: nc >= nc_test} / (n_cal + 1)
            te_proba = cal_mdl.predict_proba(X_te)
            conformal_pvals = {}
            for y_hyp in [0, 1]:
                nc_te = 1 - te_proba[:, y_hyp]
                pvals = np.array([
                    (nc_scores >= nc_te[i]).sum() / (len(nc_scores) + 1)
                    for i in range(len(nc_te))
                ])
                conformal_pvals[y_hyp] = pvals

            # significance: reject prediction if p-val < 0.1
            conf_pred = np.where(conformal_pvals[1] >= 0.1, 1,
                         np.where(conformal_pvals[0] >= 0.1, 0, -1))  # -1 = uncertain
            n_uncertain = (conf_pred == -1).sum()
            lg.info(f"  [{ep}] Conformal: uncertain={n_uncertain} "
                    f"({n_uncertain/len(conf_pred)*100:.1f}%)")
        except Exception as e:
            lg.warning(f"  [{ep}] Conformal failed: {e}")
            conformal_pvals = {0: np.zeros(len(y_te)), 1: np.zeros(len(y_te))}
            conf_pred = (proba_te >= 0.5).astype(int)

        # ── 결과 저장 ───────────────────────────────────────────────────
        thr_result = oof_tune_threshold(X_tr, y_tr, model=mdl, n_folds=5,
                                         thr_range=(0.1, 0.9), thr_step=0.02,
                                         criterion="mcc")
        opt_thr = thr_result["best_threshold"]
        pred = (proba_te >= opt_thr).astype(int)

        ad_df = pd.DataFrame({
            "smiles": df_te[ANALYSIS_SMILES_COL].values,
            "actual": y_te,
            "predicted": pred,
            "proba": proba_te.round(4),
            "knn_max_sim": knn_sim.round(4),
            "in_knn_ad": in_knn,
            "leverage_h": h_te.round(6),
            "in_leverage_ad": in_leverage,
            "conformal_pval_pos": conformal_pvals[1].round(4),
            "conformal_pval_neg": conformal_pvals[0].round(4),
            "conformal_pred": conf_pred,
            # AD consensus: 2/3 방법에서 in_AD면 in
            "in_consensus_ad": (in_knn.astype(int) + in_leverage.astype(int) >= 1),
        })

        # AD별 sensitivity 비교
        for ad_col, ad_label in [
            ("in_knn_ad",      "kNN"),
            ("in_leverage_ad", "leverage"),
            ("in_consensus_ad","consensus"),
        ]:
            mask = ad_df[ad_col].values
            yt, yp = y_te[mask], pred[mask]
            if (yt == 1).sum() == 0:
                continue
            tn, fp_n, fn_n, tp_n = confusion_matrix(yt, yp, labels=[0,1]).ravel()
            sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
            spec = tn  / (tn + fp_n)   if (tn + fp_n)  > 0 else 0
            lg.info(f"  [{ep}] AD={ad_label:10s} n={mask.sum():4d} "
                    f"sens={sens:.3f} spec={spec:.3f}")

        out_path = run_dir / f"ad_comparison_{ep}.csv"
        ad_df.to_csv(out_path, index=False)
        lg.info(f"  [{ep}] → {out_path}")
        results[ep] = {"n_test": len(y_te), "opt_threshold": opt_thr}

    return results


# ═══════════════════════════════════════════════════════
#  전략 D: Cost-sensitive learning (FN penalty)
# ═══════════════════════════════════════════════════════

def strategy_D_cost_sensitive(run_dir: Path,
                               endpoints: list = None,
                               n_bits: int = 512, top_k: int = 128,
                               fn_penalty: float = 3.0) -> dict:
    """
    FN(위음성)에 높은 penalty를 부여하는 cost-sensitive learning.

    방법:
      1) XGBoost scale_pos_weight × fn_penalty
      2) Focal loss (gamma=2, alpha=0.75): easy negative 다운웨이팅
      3) 비교: standard vs cost-sensitive sensitivity/specificity

    fn_penalty: FN 비용 배율 (default=3.0 → FN이 FP보다 3배 불이익)

    출력: cost_sensitive_{endpoint}.csv
    """
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    from xgboost import XGBClassifier

    lg.info(f"=== Strategy D: Cost-Sensitive Learning (fn_penalty={fn_penalty}) ===")
    endpoints = endpoints or ["ames", "invitro", "invivo"]
    all_rows = []

    for ep in pbar(endpoints, desc="[D] Cost-sensitive"):
        df = _load_split(run_dir, ep)
        if df is None:
            continue
        df_tr = df[df["split"] == "train"].copy().reset_index(drop=True)
        df_te = df[df["split"] == "test"].copy().reset_index(drop=True)
        if len(df_tr) < 50:
            continue

        X_tr, y_tr, selected = _build_Xty(df_tr, ep, n_bits, top_k)
        fg = extract_fg_features(df_te, ep)
        ph = extract_physchem_features(df_te, ep)
        fgp = fg  # v12.2: FG 전체 컬럼 사용
        fp_te = extract_fingerprint_features(df_te, ep, n_bits=n_bits)
        feat_te = pd.concat([fgp, ph, fp_te[selected]], axis=1).fillna(0)
        for c in feat_te.columns:
            feat_te[c] = pd.to_numeric(feat_te[c], errors="coerce")
        X_te = feat_te.fillna(0).values.astype(np.float32)
        y_te = df_te["label"].values.astype(int)

        n_neg = (y_tr == 0).sum(); n_pos = (y_tr == 1).sum()
        base_spw = n_neg / max(n_pos, 1)

        configs = {
            "standard":      {"scale_pos_weight": base_spw},
            "fn_penalty":    {"scale_pos_weight": base_spw * fn_penalty},
            "focal_gamma2":  {"scale_pos_weight": base_spw},   # focal 적용
        }

        for cfg_name, xgb_kwargs in pbar(configs.items(),
                                          desc=f"  {ep} configs", leave=False):
            t0 = time.time()
            mdl = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=SEED,
                n_jobs=-1, verbosity=0, **xgb_kwargs,
            )

            # Focal loss via sample_weight 근사:
            # w_i = (1 - p_i)^gamma (iterative 근사 -- 1 pass)
            _sw = None
            if cfg_name == "focal_gamma2":
                # warm-up: 기본 모델로 확률 추정 후 focal weight 계산
                warm = XGBClassifier(n_estimators=50, max_depth=4,
                                      learning_rate=0.1, eval_metric="logloss",
                                      scale_pos_weight=base_spw,
                                      random_state=SEED, n_jobs=-1, verbosity=0)
                warm.fit(X_tr, y_tr)
                p_hat = warm.predict_proba(X_tr)[:, 1]
                p_true = np.where(y_tr == 1, p_hat, 1 - p_hat)
                _sw = ((1 - p_true) ** 2).astype(np.float32)  # gamma=2
                _sw = _sw / _sw.mean()  # normalize
                mdl.fit(X_tr, y_tr, sample_weight=_sw)
            else:
                mdl.fit(X_tr, y_tr)

            proba = mdl.predict_proba(X_te)[:, 1]
            thr = oof_tune_threshold(X_tr, y_tr, model=mdl, n_folds=5,
                                      thr_range=(0.1, 0.9), thr_step=0.02,
                                      criterion="mcc")["best_threshold"]
            pred = (proba >= thr).astype(int)

            tn, fp_n, fn_n, tp_n = confusion_matrix(y_te, pred, labels=[0,1]).ravel()
            sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
            spec = tn  / (tn  + fp_n)  if (tn  + fp_n)  > 0 else 0.0
            mcc  = matthews_corrcoef(y_te, pred) if len(set(pred)) > 1 else 0.0

            lg.info(f"  [{ep}/{cfg_name}] "
                    f"sens={sens:.3f} spec={spec:.3f} mcc={mcc:.3f} "
                    f"thr={thr:.3f} [{time.time()-t0:.0f}s]")
            all_rows.append({
                "endpoint": ep, "config": cfg_name,
                "fn_penalty": fn_penalty,
                "scale_pos_weight": xgb_kwargs["scale_pos_weight"],
                "tp": int(tp_n), "fp": int(fp_n),
                "fn": int(fn_n), "tn": int(tn),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "mcc": round(mcc, 4),
                "threshold": round(thr, 4),
            })

    out_df = pd.DataFrame(all_rows)
    out_path = run_dir / "cost_sensitive_results.csv"
    out_df.to_csv(out_path, index=False)
    lg.info(f"Saved: {out_path}")
    return out_df.to_dict("records")


# ═══════════════════════════════════════════════════════
#  전략 F: SMOTE in fingerprint space
# ═══════════════════════════════════════════════════════

def strategy_F_smote(run_dir: Path,
                     endpoints: list = None,
                     n_bits: int = 512, top_k: int = 128) -> dict:
    """
    SMOTE(Synthetic Minority Over-sampling)를 FP 공간에서 적용.

    주의: SMOTE 생성 샘플의 화학적 타당성은 보장 안 됨.
    논문에서 반드시 언급 필요:
      "SMOTE-generated samples are used only as auxiliary training signal
       and do not represent validated chemical structures."

    비교: no-SMOTE vs SMOTE vs ADASYN

    출력: smote_comparison_{endpoint}.csv
    """
    lg.info("=== Strategy F: SMOTE in Fingerprint Space ===")

    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.pipeline import make_pipeline as imb_pipeline
    except ImportError:
        lg.warning("imbalanced-learn not installed: pip install imbalanced-learn")
        return {}

    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    from xgboost import XGBClassifier

    endpoints = endpoints or ["ames", "invitro", "invivo"]
    all_rows = []

    for ep in pbar(endpoints, desc="[F] SMOTE"):
        df = _load_split(run_dir, ep)
        if df is None:
            continue
        df_tr = df[df["split"] == "train"].copy().reset_index(drop=True)
        df_te = df[df["split"] == "test"].copy().reset_index(drop=True)
        y_tr = df_tr["label"].values.astype(int)

        if (y_tr == 1).sum() < 6:
            lg.warning(f"  [{ep}] Too few positives for SMOTE -- skipping")
            continue

        X_tr, y_tr, selected = _build_Xty(df_tr, ep, n_bits, top_k)
        fg = extract_fg_features(df_te, ep)
        ph = extract_physchem_features(df_te, ep)
        fgp = fg  # v12.2: FG 전체 컬럼 사용
        fp_te = extract_fingerprint_features(df_te, ep, n_bits=n_bits)
        feat_te = pd.concat([fgp, ph, fp_te[selected]], axis=1).fillna(0)
        for c in feat_te.columns:
            feat_te[c] = pd.to_numeric(feat_te[c], errors="coerce")
        X_te = feat_te.fillna(0).values.astype(np.float32)
        y_te = df_te["label"].values.astype(int)

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        for sampler_name, sampler in pbar(
            [("no_smote", None),
             ("smote",    SMOTE(random_state=SEED, k_neighbors=min(5, (y_tr==1).sum()-1))),
             ("adasyn",   ADASYN(random_state=SEED))],
            desc=f"  {ep} samplers", leave=False,
        ):
            try:
                if sampler is not None:
                    X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                    lg.info(f"  [{ep}/{sampler_name}] "
                            f"{len(y_tr)} → {len(y_res)} "
                            f"(+{len(y_res)-len(y_tr)} synthetic)")
                else:
                    X_res, y_res = X_tr, y_tr

                mdl = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    scale_pos_weight=spw, eval_metric="logloss",
                    random_state=SEED, n_jobs=-1, verbosity=0)
                mdl.fit(X_res, y_res)
                proba = mdl.predict_proba(X_te)[:, 1]
                thr = oof_tune_threshold(X_tr, y_tr, model=mdl, n_folds=5,
                                          thr_range=(0.1, 0.9), thr_step=0.02,
                                          criterion="mcc")["best_threshold"]
                pred = (proba >= thr).astype(int)

                tn, fp_n, fn_n, tp_n = confusion_matrix(y_te, pred, labels=[0,1]).ravel()
                sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
                spec = tn  / (tn  + fp_n)  if (tn  + fp_n)  > 0 else 0.0
                mcc  = matthews_corrcoef(y_te, pred) if len(set(pred)) > 1 else 0.0

                lg.info(f"  [{ep}/{sampler_name}] "
                        f"sens={sens:.3f} spec={spec:.3f} mcc={mcc:.3f}")
                all_rows.append({
                    "endpoint": ep, "sampler": sampler_name,
                    "n_train_after": len(y_res),
                    "n_synthetic": len(y_res) - len(y_tr),
                    "tp": int(tp_n), "fp": int(fp_n),
                    "fn": int(fn_n), "tn": int(tn),
                    "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4),
                    "mcc": round(mcc, 4),
                    "threshold": round(thr, 4),
                })
            except Exception as e:
                lg.warning(f"  [{ep}/{sampler_name}] failed: {e}")

    out_df = pd.DataFrame(all_rows)
    out_path = run_dir / "smote_comparison.csv"
    out_df.to_csv(out_path, index=False)
    lg.info(f"Saved: {out_path}")
    return out_df.to_dict("records")


# ═══════════════════════════════════════════════════════
#  전략 I: Multi-model Soft Voting Ensemble
# ═══════════════════════════════════════════════════════

def strategy_I_ensemble(run_dir: Path,
                         endpoints: list = None,
                         n_bits: int = 512, top_k: int = 128) -> dict:
    """
    XGB + LGBM + RF soft voting ensemble.

    논문 근거: 단일 모델의 화학공간 편향을 다양한 알고리즘의
    앙상블로 완화 (bias-variance tradeoff).

    출력: ensemble_results.csv
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier

    lg.info("=== Strategy I: Multi-model Soft Voting Ensemble ===")
    endpoints = endpoints or ["ames", "invitro", "invivo"]
    all_rows = []

    for ep in pbar(endpoints, desc="[I] Ensemble"):
        df = _load_split(run_dir, ep)
        if df is None:
            continue
        df_tr = df[df["split"] == "train"].copy().reset_index(drop=True)
        df_te = df[df["split"] == "test"].copy().reset_index(drop=True)

        X_tr, y_tr, selected = _build_Xty(df_tr, ep, n_bits, top_k)
        fg = extract_fg_features(df_te, ep)
        ph = extract_physchem_features(df_te, ep)
        fgp = fg  # v12.2: FG 전체 컬럼 사용
        fp_te = extract_fingerprint_features(df_te, ep, n_bits=n_bits)
        feat_te = pd.concat([fgp, ph, fp_te[selected]], axis=1).fillna(0)
        for c in feat_te.columns:
            feat_te[c] = pd.to_numeric(feat_te[c], errors="coerce")
        X_te = feat_te.fillna(0).values.astype(np.float32)
        y_te = df_te["label"].values.astype(int)

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        base_models = [
            ("xgb",  XGBClassifier(n_estimators=200, max_depth=5,
                                    scale_pos_weight=spw, eval_metric="logloss",
                                    random_state=SEED, n_jobs=-1, verbosity=0)),
            ("rf",   RandomForestClassifier(n_estimators=200, max_depth=10,
                                              class_weight="balanced",
                                              random_state=SEED, n_jobs=-1)),
        ]
        try:
            from lightgbm import LGBMClassifier
            base_models.append(
                ("lgbm", LGBMClassifier(n_estimators=200, max_depth=5,
                                         scale_pos_weight=spw, verbose=-1,
                                         random_state=SEED, n_jobs=-1))
            )
        except ImportError:
            pass

        # 개별 모델 + ensemble 비교
        for name, mdl in pbar(base_models + [("ensemble", None)],
                               desc=f"  {ep} models", leave=False):
            if name == "ensemble":
                mdl = VotingClassifier(estimators=base_models, voting="soft", n_jobs=-1)

            mdl.fit(X_tr, y_tr)
            proba = mdl.predict_proba(X_te)[:, 1]
            thr = oof_tune_threshold(X_tr, y_tr, model=base_models[0][1],
                                      n_folds=5, thr_range=(0.1,0.9),
                                      thr_step=0.02, criterion="mcc")["best_threshold"]
            pred = (proba >= thr).astype(int)

            tn, fp_n, fn_n, tp_n = confusion_matrix(y_te, pred, labels=[0,1]).ravel()
            sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
            spec = tn  / (tn  + fp_n)  if (tn  + fp_n)  > 0 else 0.0
            mcc  = matthews_corrcoef(y_te, pred) if len(set(pred)) > 1 else 0.0

            lg.info(f"  [{ep}/{name:8s}] "
                    f"sens={sens:.3f} spec={spec:.3f} mcc={mcc:.3f}")
            all_rows.append({
                "endpoint": ep, "model": name,
                "tp": int(tp_n), "fp": int(fp_n),
                "fn": int(fn_n), "tn": int(tn),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "mcc": round(mcc, 4), "threshold": round(thr, 4),
            })

    out_df = pd.DataFrame(all_rows)
    out_path = run_dir / "ensemble_results.csv"
    out_df.to_csv(out_path, index=False)
    lg.info(f"Saved: {out_path}")
    return out_df.to_dict("records")


# ═══════════════════════════════════════════════════════
#  전략 J: Endpoint별 meta-model (stacking)
# ═══════════════════════════════════════════════════════

def strategy_J_stacking(run_dir: Path,
                         endpoints: list = None,
                         n_bits: int = 512, top_k: int = 128) -> dict:
    """
    Ames → InVitro → InVivo 순으로 stacking.

    생물학적 연관성:
      - Ames(유전독성) 확률을 InVitro/InVivo 모델의 추가 feature로 사용
      - 각 endpoint 독립 최적화 후 meta-learner로 통합

    출력: stacking_results.csv
    """
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_predict

    lg.info("=== Strategy J: Endpoint Meta-model (Stacking) ===")

    # Step 1: 각 endpoint에서 OOF probability 생성
    oof_probas = {}
    models = {}
    selected_bits = {}

    for ep in pbar(["ames", "invitro", "invivo"], desc="[J] Base models"):
        df = _load_split(run_dir, ep)
        if df is None:
            continue
        df_tr = df[df["split"] == "train"].copy().reset_index(drop=True)

        X_tr, y_tr, sel = _build_Xty(df_tr, ep, n_bits, top_k)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        mdl = XGBClassifier(n_estimators=200, max_depth=5, scale_pos_weight=spw,
                             eval_metric="logloss", random_state=SEED,
                             n_jobs=-1, verbosity=0)

        if len(set(y_tr)) < 2:
            continue

        # OOF prediction for stacking feature
        oof_p = cross_val_predict(mdl, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
        oof_probas[ep] = (oof_p, y_tr, df_tr)
        mdl.fit(X_tr, y_tr)
        models[ep] = (mdl, sel)
        selected_bits[ep] = sel
        lg.info(f"  [{ep}] OOF proba computed (n={len(y_tr)})")

    # Step 2: InVitro/InVivo meta-model with Ames probability as extra feature
    stacking_rows = []
    for target_ep in pbar(["invitro", "invivo"], desc="[J] Meta-models"):
        if target_ep not in oof_probas or "ames" not in models:
            continue

        df_target = _load_split(run_dir, target_ep)
        if df_target is None:
            continue

        df_tr_tgt = df_target[df_target["split"] == "train"].copy().reset_index(drop=True)
        df_te_tgt = df_target[df_target["split"] == "test"].copy().reset_index(drop=True)

        X_tr_tgt, y_tr_tgt, sel_tgt = _build_Xty(df_tr_tgt, target_ep, n_bits, top_k)

        # Ames model prediction for target train/test
        ames_mdl, ames_sel = models["ames"]

        def _ames_proba(df_ep):
            fg_ = extract_fg_features(df_ep, "ames")
            ph_ = extract_physchem_features(df_ep, "ames")
            _r_ = build_all_features(df_ep, "ames", include_fp=True,
                                     fp_n_bits=n_bits, fp_selected_bits=ames_sel)
            return ames_mdl.predict_proba(_r_["X"])[:, 1]

        try:
            ames_p_tr = _ames_proba(df_tr_tgt).reshape(-1, 1)
            # Augment features
            X_tr_aug = np.hstack([X_tr_tgt, ames_p_tr])

            # Build test features
            fg_te = extract_fg_features(df_te_tgt, target_ep)
            ph_te = extract_physchem_features(df_te_tgt, target_ep)
            _rte = build_all_features(df_te_tgt, target_ep, include_fp=True,
                                      fp_n_bits=n_bits, fp_selected_bits=sel_tgt)
            X_te_tgt = _rte["X"]
            ames_p_te = _ames_proba(df_te_tgt).reshape(-1, 1)
            X_te_aug = np.hstack([X_te_tgt, ames_p_te])
            y_te_tgt = df_te_tgt["label"].values.astype(int)

            spw = (y_tr_tgt == 0).sum() / max((y_tr_tgt == 1).sum(), 1)

            for name, X_tr_use, X_te_use in [
                ("base",    X_tr_tgt, X_te_tgt),
                ("stacked", X_tr_aug, X_te_aug),
            ]:
                mdl = XGBClassifier(n_estimators=200, max_depth=5,
                                     scale_pos_weight=spw, eval_metric="logloss",
                                     random_state=SEED, n_jobs=-1, verbosity=0)
                mdl.fit(X_tr_use, y_tr_tgt)
                proba = mdl.predict_proba(X_te_use)[:, 1]
                thr = oof_tune_threshold(X_tr_use, y_tr_tgt, model=mdl,
                                          n_folds=5, criterion="mcc")["best_threshold"]
                pred = (proba >= thr).astype(int)

                tn, fp_n, fn_n, tp_n = confusion_matrix(y_te_tgt, pred, labels=[0,1]).ravel()
                sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
                spec = tn  / (tn  + fp_n)  if (tn  + fp_n)  > 0 else 0.0
                mcc  = matthews_corrcoef(y_te_tgt, pred) if len(set(pred)) > 1 else 0.0

                lg.info(f"  [{target_ep}/{name:7s}] "
                        f"sens={sens:.3f} spec={spec:.3f} mcc={mcc:.3f}")
                stacking_rows.append({
                    "endpoint": target_ep, "model": name,
                    "uses_ames_feature": name == "stacked",
                    "tp": int(tp_n), "fp": int(fp_n),
                    "fn": int(fn_n), "tn": int(tn),
                    "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4),
                    "mcc": round(mcc, 4), "threshold": round(thr, 4),
                })
        except Exception as e:
            lg.warning(f"  [{target_ep}] Stacking failed: {e}")

    out_df = pd.DataFrame(stacking_rows)
    out_path = run_dir / "stacking_results.csv"
    out_df.to_csv(out_path, index=False)
    lg.info(f"Saved: {out_path}")
    return out_df.to_dict("records")


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

STRATEGY_MAP = {
    "A": ("Diversity Coverage Analysis",     strategy_A_diversity_sampling),
    "B": ("Enhanced AD (leverage+conformal)", strategy_B_enhanced_ad),
    "D": ("Cost-sensitive Learning",          strategy_D_cost_sensitive),
    "F": ("SMOTE in FP space",                strategy_F_smote),
    "I": ("Multi-model Ensemble",             strategy_I_ensemble),
    "J": ("Endpoint Stacking",                strategy_J_stacking),
}


def main():
    parser = argparse.ArgumentParser(
        description="Step 10: Coverage Extension & FN Reduction Strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None,
                        help="Pipeline run 디렉토리 (미지정 시 LATEST_TXT 자동 탐색)")
    parser.add_argument("--strategy", nargs="+", default=["all"],
                        choices=list(STRATEGY_MAP.keys()) + ["all"],
                        help="실행할 전략 (기본: all)")
    parser.add_argument("--endpoints", nargs="+",
                        default=["ames", "invitro", "invivo"],
                        help="평가 endpoint 목록")
    parser.add_argument("--n-bits",      type=int,   default=512)
    parser.add_argument("--top-k",       type=int,   default=128)
    parser.add_argument("--fn-penalty",  type=float, default=3.0,
                        help="Strategy D: FN penalty 배율")
    parser.add_argument("--n-clusters",  type=int,   default=200,
                        help="Strategy A: Butina cluster 수 목표")
    args = parser.parse_args()

    run_dir = _get_run_dir(args.run_dir)
    lg.info(f"Run dir: {run_dir}")

    to_run = list(STRATEGY_MAP.keys()) if "all" in args.strategy else args.strategy

    summary = {}
    total_t0 = time.time()

    for key in pbar(to_run, desc="Strategies", colour="magenta"):
        label, fn = STRATEGY_MAP[key]
        step_header(int(key, 36) if key.isdigit() else 0, len(to_run), f"[{key}] {label}")
        t0 = time.time()
        try:
            kwargs = dict(run_dir=run_dir, endpoints=args.endpoints)
            if key == "A": kwargs["n_clusters"] = args.n_clusters
            if key == "D": kwargs["fn_penalty"] = args.fn_penalty
            kwargs.update({"n_bits": args.n_bits, "top_k": args.top_k})
            # A does not take n_bits/top_k
            if key == "A":
                kwargs = dict(run_dir=run_dir, endpoints=args.endpoints,
                              n_clusters=args.n_clusters, n_bits=args.n_bits)
            result = fn(**kwargs)
            summary[key] = {"status": "ok", "elapsed": round(time.time()-t0, 1)}
            task_done(f"[{key}] {label}", time.time()-t0)
        except Exception as e:
            lg.error(f"[{key}] FAILED: {e}")
            import traceback; traceback.print_exc()
            summary[key] = {"status": "failed", "error": str(e)}

    lg.info(f"\n{'='*50}")
    lg.info(f"Step 10 complete [{time.time()-total_t0:.0f}s]")
    for k, v in summary.items():
        sym = "V" if v["status"] == "ok" else "X"
        lg.info(f"  [{sym}] {k}: {v['status']} {v.get('elapsed','')}")


if __name__ == "__main__":
    main()
