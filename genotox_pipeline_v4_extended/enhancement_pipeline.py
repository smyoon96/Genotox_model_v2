"""
enhancement_pipeline.py — 모델 개선 5가지 전략 통합 파이프라인
================================================================
deploy_pipeline.py 실행 후 사용. 기존 모델을 개선하거나 보완하는 전략들.

전략:
  1. AD 기반 신뢰도 등급 분류
  2. Conformal Prediction (예측 불확실성 정량화)
  3. SMILES Augmentation (데이터 확장)
  4. Multi-task Learning (endpoint 간 공유 학습)
  5. External Validation 프레임워크

Usage:
    python enhancement_pipeline.py \\
        --deploy-dir ./output/deploy \\
        --data-dir ./data \\
        --output-dir ./output/enhanced
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
lg = logging.getLogger("enhance")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from config import GLOBAL_SEED

# deploy_pipeline의 함수 재사용
from deploy_pipeline import (
    load_and_clean, apply_scenario_deploy, build_features_v4,
    make_model, optimize_threshold,
)


# ═══════════════════════════════════════════════════════
#  전략 1: AD 기반 신뢰도 등급 분류
# ═══════════════════════════════════════════════════════

def strategy_ad_confidence(deploy_dir: Path, data_dir: Path, output_dir: Path):
    """
    Tanimoto 유사도 기반 AD 판별 → 예측을 High/Medium/Low confidence로 분류.
    AD 내부 화합물만의 성능을 별도 보고.
    """
    lg.info("\n" + "="*60)
    lg.info("  STRATEGY 1: AD-based Confidence Stratification")
    lg.info("="*60)

    from step4_feature_extraction import extract_fingerprint_features
    from sklearn.metrics import pairwise_distances

    results = []

    for ep_dir in sorted(deploy_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        ep = meta["endpoint"]
        lg.info(f"\n  [{ep}]")

        # 데이터 로드 + 전처리
        df, flags = load_and_clean(data_dir, ep)
        df = apply_scenario_deploy(df, flags, meta["scenario"])

        # FP 추출 (AD 계산용)
        fp = extract_fingerprint_features(df, ep, n_bits=1024)
        fp_arr = fp.values.astype(np.float32)

        # 모델 로드
        model = joblib.load(ep_dir / "model.joblib")
        with open(ep_dir / "feature_columns.json") as f:
            fcols = json.load(f)

        # Feature 추출
        feat = build_features_v4(df, ep, meta["feature_mode"], meta["scenario"])
        feat = feat.reindex(columns=fcols, fill_value=0)
        X = feat.values.astype(np.float32)
        y = df["label"].values.astype(int)
        threshold = meta["optimal_threshold"]

        # OOF predictions
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        oof_proba = np.zeros(len(y))
        for tr_idx, val_idx in skf.split(X, y):
            spw = (y[tr_idx]==0).sum() / max((y[tr_idx]==1).sum(), 1)
            mdl = make_model(meta["model_type"], spw)
            mdl.fit(X[tr_idx], y[tr_idx])
            oof_proba[val_idx] = mdl.predict_proba(X[val_idx])[:, 1]

        # AD: 각 화합물의 train 내 최근접 Tanimoto 유사도 (LOO 방식)
        tani_sim = 1 - pairwise_distances(fp_arr, metric="jaccard")
        np.fill_diagonal(tani_sim, 0)  # 자기 자신 제외
        max_sim = tani_sim.max(axis=1)  # 가장 가까운 이웃 유사도

        # 신뢰도 등급
        q33 = np.percentile(max_sim, 33)
        q66 = np.percentile(max_sim, 66)

        confidence = np.where(max_sim >= q66, "high",
                     np.where(max_sim >= q33, "medium", "low"))

        # 등급별 성능
        lg.info(f"  Thresholds: low<{q33:.3f}, medium<{q66:.3f}, high>={q66:.3f}")

        for level in ["high", "medium", "low"]:
            mask = confidence == level
            if mask.sum() < 10:
                continue
            y_sub = y[mask]
            p_sub = (oof_proba[mask] >= threshold).astype(int)
            if len(set(y_sub)) < 2 or len(set(p_sub)) < 2:
                mcc = 0.0
            else:
                mcc = matthews_corrcoef(y_sub, p_sub)
            tp = ((p_sub==1)&(y_sub==1)).sum()
            fn = ((p_sub==0)&(y_sub==1)).sum()
            tn = ((p_sub==0)&(y_sub==0)).sum()
            fp_c = ((p_sub==1)&(y_sub==0)).sum()
            sens = tp / max(tp+fn, 1)
            spec = tn / max(tn+fp_c, 1)

            lg.info(f"    {level:6s}: n={mask.sum():5d}  "
                    f"sens={sens:.3f} spec={spec:.3f} mcc={mcc:.3f}")

            results.append({
                "endpoint": ep, "confidence": level,
                "n": int(mask.sum()),
                "n_pos": int(y_sub.sum()),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "mcc": round(mcc, 4),
                "sim_threshold": round(float(q66 if level=="high" else q33), 4),
            })

    out_path = output_dir / "ad_confidence_stratification.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    lg.info(f"\n  Saved: {out_path}")
    return results


# ═══════════════════════════════════════════════════════
#  전략 2: Conformal Prediction (수동 구현, mapie 불요)
# ═══════════════════════════════════════════════════════

def strategy_conformal(deploy_dir: Path, data_dir: Path, output_dir: Path,
                        alphas: list = None):
    """
    Conformal Prediction (수동 구현).
    OOF nonconformity score 기반으로 multiple alpha에서 uncertain 비율과
    certain-only 성능을 측정.

    nonconformity score = 1 - P(true class)
    threshold = alpha-quantile of calibration scores
    prediction set = {class c : 1 - P(c) <= threshold}
    """
    lg.info("\n" + "="*60)
    lg.info("  STRATEGY 2: Conformal Prediction (manual)")
    lg.info("="*60)

    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    results = []

    for ep_dir in sorted(deploy_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        ep = meta["endpoint"]
        lg.info(f"\n  [{ep}]")

        # 데이터 + feature
        df, flags = load_and_clean(data_dir, ep)
        df = apply_scenario_deploy(df, flags, meta["scenario"])
        with open(ep_dir / "feature_columns.json") as f:
            fcols = json.load(f)
        feat = build_features_v4(df, ep, meta["feature_mode"], meta["scenario"])
        feat = feat.reindex(columns=fcols, fill_value=0)
        X = feat.values.astype(np.float32)
        y = df["label"].values.astype(int)
        threshold = meta["optimal_threshold"]

        # OOF: 각 sample의 P(class=0), P(class=1) 수집
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        oof_p0 = np.zeros(len(y))
        oof_p1 = np.zeros(len(y))

        for tr_idx, val_idx in skf.split(X, y):
            spw = (y[tr_idx]==0).sum() / max((y[tr_idx]==1).sum(), 1)
            mdl = make_model(meta["model_type"], spw)
            mdl.fit(X[tr_idx], y[tr_idx])
            proba = mdl.predict_proba(X[val_idx])
            oof_p0[val_idx] = proba[:, 0]
            oof_p1[val_idx] = proba[:, 1]

        # Nonconformity score = 1 - P(true class)
        nc_scores = np.where(y == 1, 1 - oof_p1, 1 - oof_p0)

        for alpha in alphas:
            # Conformal threshold: (1-alpha) quantile of nc scores
            q = np.quantile(nc_scores, 1 - alpha)

            # Prediction set: include class c if 1 - P(c) <= q
            set_0 = (1 - oof_p0) <= q  # class 0 in set?
            set_1 = (1 - oof_p1) <= q  # class 1 in set?

            uncertain = set_0 & set_1    # both classes
            empty = ~set_0 & ~set_1      # no class
            certain = ~uncertain & ~empty

            n_uncertain = uncertain.sum()
            n_empty = empty.sum()
            n_certain = certain.sum()
            pct_uncertain = (n_uncertain + n_empty) / len(y) * 100

            # Certain-only 성능
            if n_certain > 10:
                # certain predictions: single class in set
                cert_pred = np.where(set_1[certain] & ~set_0[certain], 1, 0)
                y_cert = y[certain]
                mcc_cert = (matthews_corrcoef(y_cert, cert_pred)
                           if len(set(cert_pred)) > 1 else 0)
                tp = ((cert_pred==1)&(y_cert==1)).sum()
                fn = ((cert_pred==0)&(y_cert==1)).sum()
                tn = ((cert_pred==0)&(y_cert==0)).sum()
                fp_c = ((cert_pred==1)&(y_cert==0)).sum()
                sens_cert = tp / max(tp+fn, 1)
                spec_cert = tn / max(tn+fp_c, 1)
            else:
                mcc_cert = sens_cert = spec_cert = 0.0

            lg.info(f"  alpha={alpha:.2f}: certain={n_certain} "
                    f"({100-pct_uncertain:.0f}%) uncertain={n_uncertain} "
                    f"empty={n_empty} | MCC_cert={mcc_cert:.3f} "
                    f"sens={sens_cert:.3f} spec={spec_cert:.3f}")

            results.append({
                "endpoint": ep, "alpha": alpha,
                "n_total": len(y),
                "n_certain": int(n_certain),
                "n_uncertain": int(n_uncertain),
                "n_empty": int(n_empty),
                "pct_uncertain": round(pct_uncertain, 1),
                "conformal_q": round(float(q), 4),
                "mcc_certain": round(mcc_cert, 4),
                "sens_certain": round(sens_cert, 4),
                "spec_certain": round(spec_cert, 4),
                "mcc_all": round(float(meta["mcc_at_threshold"]), 4),
            })

    out_path = output_dir / "conformal_prediction.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    lg.info(f"\n  Saved: {out_path}")
    return results


# ═══════════════════════════════════════════════════════
#  전략 3: SMILES Augmentation
# ═══════════════════════════════════════════════════════

def augment_smiles(smi: str, n_aug: int = 10, seed: int = 42) -> list:
    """한 SMILES에서 n_aug개의 랜덤 SMILES 생성."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    aug = set()
    aug.add(Chem.MolToSmiles(mol, canonical=True))
    rng = np.random.RandomState(seed)
    atoms = list(range(mol.GetNumAtoms()))
    for _ in range(n_aug * 3):  # oversample, deduplicate
        rng.shuffle(atoms)
        new_smi = Chem.MolToSmiles(mol, rootedAtAtom=int(atoms[0]),
                                    canonical=False)
        aug.add(new_smi)
        if len(aug) >= n_aug + 1:
            break
    return list(aug)


def strategy_smiles_augmentation(deploy_dir: Path, data_dir: Path,
                                  output_dir: Path, n_aug: int = 10):
    """
    양성 화합물의 SMILES를 augment하여 학습 데이터 확장 후 재학습.
    음성은 그대로 유지 (이미 충분).
    """
    lg.info("\n" + "="*60)
    lg.info(f"  STRATEGY 3: SMILES Augmentation (n_aug={n_aug})")
    lg.info("="*60)

    results = []

    for ep_dir in sorted(deploy_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        ep = meta["endpoint"]

        # 양성이 적은 endpoint만 대상
        if meta["n_pos"] > 1000:
            lg.info(f"  [{ep}] Skipping (n_pos={meta['n_pos']} sufficient)")
            continue

        lg.info(f"\n  [{ep}] Augmenting {meta['n_pos']} positives × {n_aug}...")

        # 데이터 로드
        df, flags = load_and_clean(data_dir, ep)
        df = apply_scenario_deploy(df, flags, meta["scenario"])

        # 양성만 augment
        pos_df = df[df["label"] == 1].copy()
        neg_df = df[df["label"] == 0].copy()

        aug_rows = []
        for _, row in pos_df.iterrows():
            smi = row.get("_analysis_smiles", row["SMILES"])
            aug_smiles = augment_smiles(str(smi), n_aug=n_aug)
            for a_smi in aug_smiles[1:]:  # 원본 제외
                new_row = row.copy()
                new_row["SMILES"] = a_smi
                new_row["_analysis_smiles"] = a_smi
                aug_rows.append(new_row)

        if not aug_rows:
            continue

        aug_df = pd.DataFrame(aug_rows)
        df_aug = pd.concat([df, aug_df], ignore_index=True)
        n_aug_added = len(aug_df)

        lg.info(f"  Augmented: {len(df)} → {len(df_aug)} "
                f"(+{n_aug_added} positive variants)")

        # Feature + molecule-level GroupKFold (누수 방지)
        # 같은 분자의 모든 SMILES variant가 같은 fold에 배치
        from sklearn.model_selection import GroupKFold

        with open(ep_dir / "feature_columns.json") as f:
            fcols = json.load(f)

        feat = build_features_v4(df_aug, ep, meta["feature_mode"], meta["scenario"])
        feat = feat.reindex(columns=fcols, fill_value=0)
        X = feat.values.astype(np.float32)
        y = df_aug["label"].values.astype(int)

        # Group ID: canonical SMILES → 같은 분자 = 같은 그룹
        canon_smiles = []
        for smi in df_aug["SMILES"]:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                canon_smiles.append(Chem.MolToSmiles(mol, canonical=True)
                                   if mol else str(smi))
            except Exception:
                canon_smiles.append(str(smi))
        df_aug["_mol_group"] = pd.Categorical(canon_smiles).codes
        groups = df_aug["_mol_group"].values

        n_unique_mols = len(set(groups))
        n_folds = min(5, n_unique_mols)
        lg.info(f"  GroupKFold: {n_unique_mols} unique molecules, {n_folds} folds")

        gkf = GroupKFold(n_splits=n_folds)
        oof = np.zeros(len(y))
        mccs = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            # 검증: train/test 분자 겹침 확인
            tr_groups = set(groups[tr_idx])
            val_groups = set(groups[val_idx])
            overlap = tr_groups & val_groups
            assert len(overlap) == 0, f"Leakage! {len(overlap)} molecules overlap"

            spw = (y[tr_idx]==0).sum() / max((y[tr_idx]==1).sum(), 1)
            mdl = make_model(meta["model_type"], spw)
            mdl.fit(X[tr_idx], y[tr_idx])
            oof[val_idx] = mdl.predict_proba(X[val_idx])[:, 1]
            if len(set(y[val_idx])) >= 2:
                mccs.append(matthews_corrcoef(
                    y[val_idx], (oof[val_idx] >= 0.5).astype(int)))

        cv_mcc = np.mean(mccs)
        thresh = optimize_threshold(y, oof, spec_floor=0.65, sens_weight=2.0)

        lg.info(f"  CV MCC: {meta['cv_mcc']:.4f} → {cv_mcc:.4f} "
                f"(Δ={cv_mcc - meta['cv_mcc']:+.4f})")
        lg.info(f"  Threshold: {thresh['optimal_threshold']:.3f} "
                f"sens={thresh['sensitivity']:.3f} spec={thresh['specificity']:.3f}")

        results.append({
            "endpoint": ep,
            "n_original": meta["n_total"],
            "n_augmented": len(df_aug),
            "n_pos_original": meta["n_pos"],
            "n_pos_augmented": int(y.sum()),
            "cv_mcc_before": meta["cv_mcc"],
            "cv_mcc_after": round(cv_mcc, 4),
            "delta_mcc": round(cv_mcc - meta["cv_mcc"], 4),
            "sens_before": meta["sensitivity"],
            "sens_after": round(thresh["sensitivity"], 4),
            "spec_after": round(thresh["specificity"], 4),
        })

    out_path = output_dir / "smiles_augmentation.csv"
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)
    lg.info(f"\n  Saved: {out_path}")
    return results


# ═══════════════════════════════════════════════════════
#  전략 4: Multi-task Learning
# ═══════════════════════════════════════════════════════

def strategy_multitask(deploy_dir: Path, data_dir: Path, output_dir: Path):
    """
    Multi-task: 공통 feature space에서 5개 endpoint를 동시에 학습.
    XGBoost multi-output wrapper + 공유 feature 사용.
    """
    lg.info("\n" + "="*60)
    lg.info("  STRATEGY 4: Multi-task Learning")
    lg.info("="*60)

    from step4_feature_extraction import (
        extract_fg_features, extract_physchem_features
    )
    from step4b_sa_features import extract_sa_features

    # 모든 endpoint 데이터 로드
    all_data = {}
    for ep_dir in sorted(deploy_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        ep = meta["endpoint"]
        # sampling endpoint 제외 (원본과 중복)
        if "sampling" in ep:
            continue
        df, flags = load_and_clean(data_dir, ep)
        all_data[ep] = {"df": df, "flags": flags, "meta": meta}

    if len(all_data) < 2:
        lg.warning("  Not enough endpoints for multi-task")
        return None

    # SMILES 기반 화합물 매칭
    lg.info(f"  Endpoints: {list(all_data.keys())}")

    # 공통 feature set: FG + PhysChem + SA (FP 제외 — 길이 일관성)
    all_smiles = set()
    for ep, data in all_data.items():
        smi_col = "_analysis_smiles" if "_analysis_smiles" in data["df"].columns else "SMILES"
        all_smiles.update(data["df"][smi_col].astype(str).tolist())

    # 공통 DataFrame 구축
    smi_list = sorted(all_smiles)
    master = pd.DataFrame({"SMILES": smi_list, "_analysis_smiles": smi_list})
    master["label"] = 0  # dummy
    master["endpoint"] = "multi"

    lg.info(f"  Unique compounds: {len(master)}")

    # 공통 feature 추출
    fg = extract_fg_features(master, "ames")
    ph = extract_physchem_features(master, "ames")
    try:
        sa = extract_sa_features(master, "ames")
    except Exception:
        sa = pd.DataFrame(index=master.index)

    shared_feat = pd.concat([fg, ph, sa], axis=1)
    for c in shared_feat.columns:
        shared_feat[c] = pd.to_numeric(shared_feat[c], errors="coerce")
    shared_feat = shared_feat.fillna(0)

    # constant 제거
    shared_feat = shared_feat.loc[:, shared_feat.nunique() > 1]
    shared_cols = list(shared_feat.columns)
    lg.info(f"  Shared features: {len(shared_cols)}")

    # SMILES → index 매핑
    smi_to_idx = {s: i for i, s in enumerate(smi_list)}

    # Multi-task labels: endpoint별 label 매핑
    results = []
    for ep, data in all_data.items():
        df = data["df"]
        smi_col = "_analysis_smiles" if "_analysis_smiles" in df.columns else "SMILES"

        # 이 endpoint에 존재하는 화합물만 추출
        ep_smis = df[smi_col].astype(str).values
        ep_labels = df["label"].values.astype(int)
        ep_idx = [smi_to_idx[s] for s in ep_smis if s in smi_to_idx]
        ep_y = [ep_labels[i] for i, s in enumerate(ep_smis) if s in smi_to_idx]

        X_ep = shared_feat.iloc[ep_idx].values.astype(np.float32)
        y_ep = np.array(ep_y, dtype=int)

        # 다른 endpoint의 label을 보조 feature로 추가
        aux_feats = []
        for other_ep, other_data in all_data.items():
            if other_ep == ep:
                continue
            other_df = other_data["df"]
            other_smi_col = "_analysis_smiles" if "_analysis_smiles" in other_df.columns else "SMILES"
            other_label_map = dict(zip(
                other_df[other_smi_col].astype(str),
                other_df["label"].astype(int)
            ))
            # -1 = unknown (이 화합물이 다른 endpoint에 없음)
            aux = np.array([other_label_map.get(s, -1) for s in ep_smis
                           if s in smi_to_idx], dtype=np.float32)
            aux_feats.append(aux.reshape(-1, 1))

        if aux_feats:
            X_mt = np.hstack([X_ep] + aux_feats)
        else:
            X_mt = X_ep

        # OOF CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        oof = np.zeros(len(y_ep))
        mccs = []

        for tr_idx, val_idx in skf.split(X_mt, y_ep):
            spw = (y_ep[tr_idx]==0).sum() / max((y_ep[tr_idx]==1).sum(), 1)
            mdl = make_model("xgb", spw)
            mdl.fit(X_mt[tr_idx], y_ep[tr_idx])
            oof[val_idx] = mdl.predict_proba(X_mt[val_idx])[:, 1]
            mccs.append(matthews_corrcoef(
                y_ep[val_idx], (oof[val_idx] >= 0.5).astype(int)))

        cv_mcc = np.mean(mccs)
        orig_mcc = data["meta"]["cv_mcc"]

        lg.info(f"  [{ep}] Multi-task CV MCC: {orig_mcc:.4f} → {cv_mcc:.4f} "
                f"(Δ={cv_mcc - orig_mcc:+.4f})")

        results.append({
            "endpoint": ep,
            "n": len(y_ep),
            "n_shared_features": len(shared_cols),
            "n_aux_endpoints": len(all_data) - 1,
            "cv_mcc_single": round(orig_mcc, 4),
            "cv_mcc_multitask": round(cv_mcc, 4),
            "delta_mcc": round(cv_mcc - orig_mcc, 4),
        })

    out_path = output_dir / "multitask_learning.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    lg.info(f"\n  Saved: {out_path}")
    return results


# ═══════════════════════════════════════════════════════
#  전략 5: External Validation 프레임워크
# ═══════════════════════════════════════════════════════

def strategy_external_validation(deploy_dir: Path, data_dir: Path,
                                  output_dir: Path):
    """
    외부 검증 데이터가 있으면 자동 평가.
    data_dir/external/ 아래에 {endpoint}_external.csv (SMILES, label) 배치.
    없으면 프레임워크만 생성하고 안내.
    """
    lg.info("\n" + "="*60)
    lg.info("  STRATEGY 5: External Validation")
    lg.info("="*60)

    ext_dir = data_dir / "external"
    results = []

    for ep_dir in sorted(deploy_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        ep = meta["endpoint"]

        # 외부 데이터 확인
        ext_path = ext_dir / f"{ep}_external.csv" if ext_dir.exists() else None
        if ext_path is None or not ext_path.exists():
            lg.info(f"  [{ep}] No external data found")
            lg.info(f"    → Place file at: data/external/{ep}_external.csv")
            lg.info(f"    → Required columns: SMILES, label")
            results.append({
                "endpoint": ep, "status": "no_data",
                "expected_path": f"data/external/{ep}_external.csv",
            })
            continue

        # 외부 데이터 로드
        ext_df = pd.read_csv(ext_path)
        if "SMILES" not in ext_df.columns or "label" not in ext_df.columns:
            lg.warning(f"  [{ep}] Invalid format: need SMILES, label columns")
            continue

        ext_df["label"] = ext_df["label"].astype(int)
        ext_df["endpoint"] = ep
        ext_df["_analysis_smiles"] = ext_df["SMILES"]
        n_ext = len(ext_df)
        n_pos = ext_df["label"].sum()

        lg.info(f"  [{ep}] External: {n_ext} compounds "
                f"(pos={n_pos}, neg={n_ext - n_pos})")

        # 모델 로드 + 예측
        model = joblib.load(ep_dir / "model.joblib")
        with open(ep_dir / "feature_columns.json") as f:
            fcols = json.load(f)

        feat = build_features_v4(ext_df, ep, meta["feature_mode"],
                                  meta["scenario"])
        feat = feat.reindex(columns=fcols, fill_value=0)
        X_ext = feat.values.astype(np.float32)
        y_ext = ext_df["label"].values.astype(int)
        threshold = meta["optimal_threshold"]

        proba = model.predict_proba(X_ext)[:, 1]
        pred = (proba >= threshold).astype(int)

        # 성능 계산
        mcc = matthews_corrcoef(y_ext, pred) if len(set(pred)) > 1 else 0
        tp = ((pred==1)&(y_ext==1)).sum()
        fn = ((pred==0)&(y_ext==1)).sum()
        tn = ((pred==0)&(y_ext==0)).sum()
        fp_c = ((pred==1)&(y_ext==0)).sum()
        sens = tp / max(tp+fn, 1)
        spec = tn / max(tn+fp_c, 1)
        bacc = (sens + spec) / 2
        try:
            auc = roc_auc_score(y_ext, proba)
        except Exception:
            auc = 0

        lg.info(f"  Results: sens={sens:.3f} spec={spec:.3f} "
                f"mcc={mcc:.3f} AUC={auc:.3f}")
        lg.info(f"  vs internal: sens={meta['sensitivity']:.3f} "
                f"spec={meta['specificity']:.3f}")

        # 예측 결과 저장
        ext_df["proba"] = proba.round(4)
        ext_df["pred"] = pred
        ext_df.to_csv(output_dir / f"external_{ep}_predictions.csv", index=False)

        results.append({
            "endpoint": ep, "status": "validated",
            "n_external": n_ext, "n_pos": int(n_pos),
            "ext_sensitivity": round(sens, 4),
            "ext_specificity": round(spec, 4),
            "ext_mcc": round(mcc, 4),
            "ext_bacc": round(bacc, 4),
            "ext_auc": round(auc, 4),
            "int_sensitivity": meta["sensitivity"],
            "int_specificity": meta["specificity"],
            "int_mcc": meta["mcc_at_threshold"],
            "delta_mcc": round(mcc - meta["mcc_at_threshold"], 4),
        })

    out_path = output_dir / "external_validation.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    lg.info(f"\n  Saved: {out_path}")
    return results


# ═══════════════════════════════════════════════════════
#  메인 실행
# ═══════════════════════════════════════════════════════

def run_all(deploy_dir: Path, data_dir: Path, output_dir: Path = None):
    t0 = time.time()
    output_dir = output_dir or (SCRIPT_DIR / "output" / "enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    lg.info("="*60)
    lg.info("  ENHANCEMENT PIPELINE — 5 Strategies")
    lg.info(f"  Deploy:  {deploy_dir}")
    lg.info(f"  Data:    {data_dir}")
    lg.info(f"  Output:  {output_dir}")
    lg.info("="*60)

    # 1. AD Confidence
    r1 = strategy_ad_confidence(deploy_dir, data_dir, output_dir)

    # 2. Conformal Prediction
    r2 = strategy_conformal(deploy_dir, data_dir, output_dir)

    # 3. SMILES Augmentation
    r3 = strategy_smiles_augmentation(deploy_dir, data_dir, output_dir)

    # 4. Multi-task Learning
    r4 = strategy_multitask(deploy_dir, data_dir, output_dir)

    # 5. External Validation
    r5 = strategy_external_validation(deploy_dir, data_dir, output_dir)

    # Summary
    summary = {
        "ad_confidence": r1,
        "conformal": r2,
        "smiles_augmentation": r3,
        "multitask": r4,
        "external_validation": r5,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(output_dir / "enhancement_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    lg.info(f"\n{'='*60}")
    lg.info(f"  ENHANCEMENT COMPLETE ({time.time()-t0:.0f}s)")
    lg.info(f"  Results: {output_dir}")
    lg.info(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Enhancement Pipeline")
    p.add_argument("--deploy-dir", required=True,
                   help="deploy_pipeline 출력 디렉토리")
    p.add_argument("--data-dir", required=True,
                   help="원본 데이터 디렉토리")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    run_all(
        Path(args.deploy_dir), Path(args.data_dir),
        Path(args.output_dir) if args.output_dir else None,
    )
