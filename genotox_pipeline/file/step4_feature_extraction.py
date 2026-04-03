"""
step4_feature_extraction.py – 작용기 / 물리화학 / QM / fingerprint 추출 및 dataset 구축
======================================================================================
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

from config import (
    FG_SMARTS, ALERT_FAMILIES, FP_RADIUS, FP_NBITS,
    ENDPOINTS, save_json,
)

logger = logging.getLogger(__name__)


# =============================================
#  4-1. Functional Group / Structural Alert
# =============================================

def _count_smarts(mol, smarts_str: str) -> int:
    """mol에서 SMARTS 패턴 매칭 횟수 반환"""
    try:
        pat = Chem.MolFromSmarts(smarts_str)
        if pat is None:
            return 0
        return len(mol.GetSubstructMatches(pat))
    except:
        return 0


def extract_fg_features(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """작용기 presence/count/rule features 추출"""
    smi_col = next((c for c in ["canonical_smiles","SMILES_raw","SMILES","_can"] if c in df.columns), "SMILES")
    mols = df[smi_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    fg_data = {}
    for fg_name, smarts in FG_SMARTS.items():
        counts = mols.apply(lambda m: _count_smarts(m, smarts) if m else 0)
        fg_data[f"fg_{fg_name}_count"]   = counts
        fg_data[f"fg_{fg_name}_present"] = (counts > 0).astype(int)

        # binned rules
        fg_data[f"rule_{fg_name}_ge1"] = (counts >= 1).astype(int)
        fg_data[f"rule_{fg_name}_ge2"] = (counts >= 2).astype(int)
        fg_data[f"rule_{fg_name}_ge3"] = (counts >= 3).astype(int)

    fg_df = pd.DataFrame(fg_data, index=df.index)

    # aggregate alert features
    alert_pos = ALERT_FAMILIES.get(endpoint, {}).get("positive", [])
    alert_cols = [f"fg_{a}_count" for a in alert_pos if f"fg_{a}_count" in fg_df.columns]
    if alert_cols:
        fg_df["bb_n_genotox_alerts"]  = fg_df[alert_cols].sum(axis=1)
        fg_df["bb_any_genotox_alert"] = (fg_df["bb_n_genotox_alerts"] > 0).astype(int)

        # positive alert score
        alert_present = [f"fg_{a}_present" for a in alert_pos
                         if f"fg_{a}_present" in fg_df.columns]
        fg_df["positive_alert_score"] = fg_df[alert_present].sum(axis=1)
    else:
        fg_df["bb_n_genotox_alerts"]  = 0
        fg_df["bb_any_genotox_alert"] = 0
        fg_df["positive_alert_score"] = 0

    logger.info(f"  [{endpoint}] FG features: {fg_df.shape[1]} columns")
    return fg_df


# =============================================
#  4-2. Physicochemical Block
# =============================================

def extract_physchem_features(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """RDKit 기반 물리화학적 특성 계산"""
    smi_col = next((c for c in ["canonical_smiles","SMILES_raw","SMILES","_can"] if c in df.columns), "SMILES")
    mols = df[smi_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    records = []
    for mol in mols:
        if mol is None:
            records.append({k: np.nan for k in [
                "MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
                "FractionCSP3", "AromaticRingCount", "RingCount",
                "HeteroatomCount", "HeavyAtomCount", "FormalCharge",
                "NumValenceElectrons",
            ]})
            continue
        try:
            rec = {
                "MW":                    Descriptors.MolWt(mol),
                "LogP":                  Descriptors.MolLogP(mol),
                "TPSA":                  Descriptors.TPSA(mol),
                "HBD":                   Descriptors.NumHDonors(mol),
                "HBA":                   Descriptors.NumHAcceptors(mol),
                "RotatableBonds":        Descriptors.NumRotatableBonds(mol),
                "FractionCSP3":          Descriptors.FractionCSP3(mol),
                "AromaticRingCount":     rdMolDescriptors.CalcNumAromaticRings(mol),
                "RingCount":             rdMolDescriptors.CalcNumRings(mol),
                "HeteroatomCount":       rdMolDescriptors.CalcNumHeteroatoms(mol),
                "HeavyAtomCount":        mol.GetNumHeavyAtoms(),
                "FormalCharge":          Chem.GetFormalCharge(mol),
                "NumValenceElectrons":   Descriptors.NumValenceElectrons(mol),
            }
        except Exception:
            rec = {k: np.nan for k in [
                "MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
                "FractionCSP3", "AromaticRingCount", "RingCount",
                "HeteroatomCount", "HeavyAtomCount", "FormalCharge",
                "NumValenceElectrons",
            ]}
        records.append(rec)

    phys_df = pd.DataFrame(records, index=df.index)
    logger.info(f"  [{endpoint}] Physchem features: {phys_df.shape[1]} columns")
    return phys_df


# =============================================
#  4-3. Quantum Descriptor Block (optional)
# =============================================

def extract_qm_features(
    df: pd.DataFrame,
    endpoint: str,
    qm_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    QM descriptor CSV가 있으면 merge, 없으면 빈 DataFrame 반환.
    """
    report = {"endpoint": endpoint, "qm_available": False, "merge_rate": 0.0}

    if qm_dir is None:
        return pd.DataFrame(index=df.index), report

    # 가능한 QM 파일 탐색
    qm_candidates = list(qm_dir.glob(f"*{endpoint}*qm*.csv")) + \
                    list(qm_dir.glob(f"*qm*{endpoint}*.csv"))

    if not qm_candidates:
        logger.info(f"  [{endpoint}] No QM descriptor file found. Skipping.")
        return pd.DataFrame(index=df.index), report

    qm_raw = pd.read_csv(qm_candidates[0])
    report["qm_available"] = True

    # merge key: No or canonical_smiles
    merge_key = "No" if "No" in qm_raw.columns else None
    if merge_key is None:
        for c in ["canonical_smiles", "SMILES", "smiles"]:
            if c in qm_raw.columns:
                merge_key = c
                break

    if merge_key is None:
        logger.warning(f"  [{endpoint}] QM file has no merge key. Skipping.")
        return pd.DataFrame(index=df.index), report

    # merge
    qm_raw[merge_key] = qm_raw[merge_key].astype(str)
    df_key = df[merge_key].astype(str) if merge_key in df.columns else None
    if df_key is None:
        return pd.DataFrame(index=df.index), report

    merged = df[[merge_key]].merge(qm_raw, on=merge_key, how="left")
    # QM 컬럼만 추출
    qm_cols = [c for c in merged.columns
               if c not in [merge_key] and c not in df.columns]
    qm_df = merged[qm_cols].copy()
    qm_df.index = df.index

    report["merge_rate"] = round(qm_df.dropna(how="all").shape[0] / len(df), 4)
    logger.info(f"  [{endpoint}] QM merge rate: {report['merge_rate']:.2%}")

    return qm_df, report


# =============================================
#  4-4. Fingerprint Block
# =============================================

def extract_fingerprint_features(
    df: pd.DataFrame,
    endpoint: str,
    radius: int = FP_RADIUS,
    n_bits: int = FP_NBITS,
) -> pd.DataFrame:
    """Morgan fingerprint 생성"""
    smi_col = next((c for c in ["canonical_smiles","SMILES_raw","SMILES","_can"] if c in df.columns), "SMILES")

    fp_matrix = np.zeros((len(df), n_bits), dtype=np.int8)
    for i, smi in enumerate(df[smi_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                fp_matrix[i] = arr
        except:
            pass

    fp_cols = [f"fp_{i:04d}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    logger.info(f"  [{endpoint}] Fingerprint: {n_bits} bits")
    return fp_df


def select_fingerprint_bits(
    fp_train: pd.DataFrame,
    top_k: int = 64,
) -> List[str]:
    """
    train set 내부에서만 bit selection:
    prevalence/variance 기반 top-k 선택.
    """
    variances = fp_train.var()
    # prevalence: 최소 1% ~ 최대 99%
    prev = fp_train.mean()
    mask = (prev >= 0.01) & (prev <= 0.99)
    filtered = variances[mask]
    selected = filtered.nlargest(min(top_k, len(filtered))).index.tolist()
    return selected


# =============================================
#  4-5. 최종 Feature Dataset Build
# =============================================

# 제외할 meta/id 컬럼 패턴
META_COLUMNS = {
    "No", "label", "SMILES_raw", "SMILES", "_can", "canonical_smiles",
    "standardized_smiles", "analysis_smiles",
    "murcko_scaffold", "scaffold", "scaffold_group",
    "scaffold_group_type", "scaffold_type",
    "endpoint", "source_dataset", "split", "_scenario",
    "merge_id", "No_original", "sanitized", "salt_stripped",
    "charge_normed", "error", "domain", "status",
    "__augmented__", "has_metal_feature", "has_Sn",
}


def build_feature_datasets(
    df: pd.DataFrame,
    fg_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    qm_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    endpoint: str,
    fp_selected_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """compact / broad_tabular / broad_fp 데이터셋 빌드"""

    # ID + label 기본
    id_label = df[["No", "label"]].copy()

    # FG에서 핵심 feature만 (presence + aggregate)
    fg_core = fg_df[[c for c in fg_df.columns
                      if c.endswith("_present") or c.startswith("bb_") or
                      c == "positive_alert_score"]].copy()

    # FG 전체 (presence + count + rule)
    fg_full = fg_df.copy()

    # physchem 핵심
    phys_core_cols = ["MW", "LogP", "TPSA", "HBD", "HBA",
                      "RotatableBonds", "FractionCSP3", "AromaticRingCount",
                      "RingCount", "HeavyAtomCount"]
    phys_core = phys_df[[c for c in phys_core_cols if c in phys_df.columns]].copy()
    phys_full = phys_df.copy()

    # --- compact ---
    compact = pd.concat([id_label, fg_core, phys_core], axis=1)

    # --- broad_tabular ---
    parts = [id_label, fg_full, phys_full]
    if qm_df is not None and not qm_df.empty and qm_df.shape[1] > 0:
        parts.append(qm_df)
    broad_tabular = pd.concat(parts, axis=1)

    # --- broad_fp ---
    if fp_selected_cols and len(fp_selected_cols) > 0:
        fp_subset = fp_df[fp_selected_cols].copy()
    else:
        fp_subset = fp_df.copy()
    broad_fp = pd.concat([broad_tabular, fp_subset], axis=1)

    # dtype 안전성: numeric coercion
    for ds_name, ds in [("compact", compact), ("broad_tabular", broad_tabular),
                         ("broad_fp", broad_fp)]:
        feat_cols = [c for c in ds.columns if c not in META_COLUMNS]
        for c in feat_cols:
            ds[c] = pd.to_numeric(ds[c], errors="coerce")

    return {
        "compact":       compact,
        "broad_tabular": broad_tabular,
        "broad_fp":      broad_fp,
    }


# =============================================
#  통합 실행 함수
# =============================================

def run_step4(
    splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    out_dir: Path,
    qm_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Step 4 전체 실행:
      각 endpoint × train/test에 대해 feature 추출 → dataset build
    Returns:
      endpoint → { feature_mode → (train_df, test_df) }
    """
    all_datasets = {}
    manifest = {}
    qm_reports = []

    for ep in ENDPOINTS:
        if ep not in splits:
            continue
        logger.info(f"=== Step 4: Feature extraction for {ep} ===")
        train_df, test_df = splits[ep]

        ep_dir = out_dir / ep
        ep_dir.mkdir(parents=True, exist_ok=True)

        # 합쳐서 feature 추출 (train/test 구분은 나중에)
        combined = pd.concat([train_df, test_df], ignore_index=True)
        train_mask = combined["split"] == "train"

        # 4-1. FG
        fg_all = extract_fg_features(combined, ep)
        fg_all.to_csv(ep_dir / f"{ep}_fg_features.csv", index=False)

        # alert family map
        save_json(ALERT_FAMILIES.get(ep, {}),
                  ep_dir / f"{ep}_alert_family_map.json")

        # 4-2. Physchem
        phys_all = extract_physchem_features(combined, ep)
        phys_all.to_csv(ep_dir / f"{ep}_physchem_features.csv", index=False)

        # 4-3. QM
        qm_all, qm_report = extract_qm_features(combined, ep, qm_dir)
        qm_reports.append(qm_report)
        if not qm_all.empty:
            qm_all.to_csv(ep_dir / f"{ep}_qm_features.csv", index=False)

        # 4-4. Fingerprint
        fp_all = extract_fingerprint_features(combined, ep)
        fp_all.to_csv(ep_dir / f"{ep}_fingerprint_features.csv", index=False)

        # fingerprint bit selection (train 내부에서만)
        fp_train = fp_all[train_mask]
        selected_bits = select_fingerprint_bits(fp_train, top_k=64)
        fp_sel_report = {
            "total_bits": fp_all.shape[1],
            "selected_bits": len(selected_bits),
            "selection_method": "variance_top64_train_only",
            "selected_columns": selected_bits,
        }
        save_json(fp_sel_report, ep_dir / f"{ep}_fingerprint_selection_report.json")

        # 4-5. Dataset build
        feature_sets = build_feature_datasets(
            combined, fg_all, phys_all, qm_all, fp_all,
            ep, fp_selected_cols=selected_bits,
        )

        ep_datasets = {}
        for mode, full_ds in feature_sets.items():
            train_part = full_ds[train_mask].reset_index(drop=True)
            test_part  = full_ds[~train_mask].reset_index(drop=True)

            train_part.to_csv(ep_dir / f"{ep}_{mode}_train.csv", index=False)
            test_part.to_csv(ep_dir / f"{ep}_{mode}_test.csv", index=False)

            ep_datasets[mode] = (train_part, test_part)

            feat_cols = [c for c in train_part.columns if c not in META_COLUMNS]
            manifest[f"{ep}_{mode}"] = {
                "n_features": len(feat_cols),
                "train_rows": len(train_part),
                "test_rows":  len(test_part),
                "columns":    feat_cols,
            }

        all_datasets[ep] = ep_datasets

    save_json(manifest, out_dir / "feature_manifest.json")
    pd.DataFrame(qm_reports).to_csv(out_dir / "qm_merge_report.csv", index=False)

    logger.info("Step 4 complete.")
    return all_datasets
