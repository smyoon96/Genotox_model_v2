"""
step4_feature_extraction.py — v12 (2026-03-27)
================================================
FIX: All extraction functions now use find_smi() which respects
     _analysis_smiles column set by apply_scenario().
     This ensures salt_stripped SMILES are actually used for feature extraction.

ADD: FP bit selection integrated into broad modes.
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
from pipeline_v2_core import find_smi, ANALYSIS_SMILES_COL

logger = logging.getLogger(__name__)


# =============================================
#  4-1. Functional Group / Structural Alert
# =============================================

def _count_smarts(mol, smarts_str: str) -> int:
    try:
        pat = Chem.MolFromSmarts(smarts_str)
        if pat is None:
            return 0
        return len(mol.GetSubstructMatches(pat))
    except Exception:
        return 0


def extract_fg_features(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """
    Extract functional group presence/count/rule features.
    Uses find_smi() to respect _analysis_smiles from scenario application.
    """
    smi_col = find_smi(df)
    mols = df[smi_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    fg_data = {}
    for fg_name, smarts in FG_SMARTS.items():
        counts = mols.apply(lambda m: _count_smarts(m, smarts) if m else 0)
        fg_data[f"fg_{fg_name}_count"] = counts
        fg_data[f"fg_{fg_name}_present"] = (counts > 0).astype(int)
        fg_data[f"rule_{fg_name}_ge1"] = (counts >= 1).astype(int)
        fg_data[f"rule_{fg_name}_ge2"] = (counts >= 2).astype(int)
        fg_data[f"rule_{fg_name}_ge3"] = (counts >= 3).astype(int)

    fg_df = pd.DataFrame(fg_data, index=df.index)

    # Aggregate alert features
    alert_pos = ALERT_FAMILIES.get(endpoint, {}).get("positive", [])
    alert_cols = [f"fg_{a}_count" for a in alert_pos if f"fg_{a}_count" in fg_df.columns]
    if alert_cols:
        fg_df["bb_n_genotox_alerts"] = fg_df[alert_cols].sum(axis=1)
        fg_df["bb_any_genotox_alert"] = (fg_df["bb_n_genotox_alerts"] > 0).astype(int)
        alert_present = [f"fg_{a}_present" for a in alert_pos
                         if f"fg_{a}_present" in fg_df.columns]
        fg_df["positive_alert_score"] = fg_df[alert_present].sum(axis=1)
    else:
        fg_df["bb_n_genotox_alerts"] = 0
        fg_df["bb_any_genotox_alert"] = 0
        fg_df["positive_alert_score"] = 0

    logger.info(f"  [{endpoint}] FG features: {fg_df.shape[1]} columns (smi_col={smi_col})")
    return fg_df


# =============================================
#  4-2. Physicochemical Block
# =============================================

def extract_physchem_features(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Uses find_smi() to respect _analysis_smiles."""
    smi_col = find_smi(df)
    mols = df[smi_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    desc_names = [
        "MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
        "FractionCSP3", "AromaticRingCount", "RingCount",
        "HeteroatomCount", "HeavyAtomCount", "FormalCharge",
        "NumValenceElectrons",
    ]
    records = []
    for mol in mols:
        if mol is None:
            records.append({k: np.nan for k in desc_names})
            continue
        try:
            rec = {
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "RotatableBonds": Descriptors.NumRotatableBonds(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
                "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
                "HeteroatomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),
                "HeavyAtomCount": mol.GetNumHeavyAtoms(),
                "FormalCharge": Chem.GetFormalCharge(mol),
                "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
            }
        except Exception:
            rec = {k: np.nan for k in desc_names}
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
    report = {"endpoint": endpoint, "qm_available": False, "merge_rate": 0.0}
    if qm_dir is None:
        return pd.DataFrame(index=df.index), report

    qm_candidates = list(qm_dir.glob(f"*{endpoint}*qm*.csv")) + \
                    list(qm_dir.glob(f"*qm*{endpoint}*.csv"))
    if not qm_candidates:
        logger.info(f"  [{endpoint}] No QM descriptor file found. Skipping.")
        return pd.DataFrame(index=df.index), report

    qm_raw = pd.read_csv(qm_candidates[0])
    report["qm_available"] = True

    merge_key = "No" if "No" in qm_raw.columns else None
    if merge_key is None:
        for c in ["canonical_smiles", "SMILES", "smiles"]:
            if c in qm_raw.columns:
                merge_key = c
                break
    if merge_key is None:
        logger.warning(f"  [{endpoint}] QM file has no merge key. Skipping.")
        return pd.DataFrame(index=df.index), report

    qm_raw[merge_key] = qm_raw[merge_key].astype(str)
    df_key = df[merge_key].astype(str) if merge_key in df.columns else None
    if df_key is None:
        return pd.DataFrame(index=df.index), report

    merged = df[[merge_key]].merge(qm_raw, on=merge_key, how="left")
    qm_cols = [c for c in merged.columns if c not in [merge_key] and c not in df.columns]
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
    """
    Morgan fingerprint generation.
    Uses find_smi() to respect _analysis_smiles.
    """
    smi_col = find_smi(df)

    fp_matrix = np.zeros((len(df), n_bits), dtype=np.int8)
    for i, smi in enumerate(df[smi_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                fp_matrix[i] = arr
        except Exception:
            pass

    fp_cols = [f"fp_{i:04d}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    logger.info(f"  [{endpoint}] Fingerprint: {n_bits} bits (smi_col={smi_col})")
    return fp_df


def select_fingerprint_bits(
    fp_train: pd.DataFrame,
    y_train: np.ndarray = None,
    top_k: int = 128,
    method: str = "variance",
) -> List[str]:
    """
    Train-set-only FP bit selection.

    Methods:
      - 'variance': prevalence filter + top-k by variance
      - 'mi': mutual information (requires y_train)
    """
    # Prevalence filter: 1%–99%
    prev = fp_train.mean()
    mask = (prev >= 0.01) & (prev <= 0.99)
    filtered_cols = prev[mask].index.tolist()

    if method == "mi" and y_train is not None:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(
            fp_train[filtered_cols].values, y_train, random_state=42
        )
        mi_series = pd.Series(mi, index=filtered_cols)
        selected = mi_series.nlargest(min(top_k, len(mi_series))).index.tolist()
    else:
        variances = fp_train[filtered_cols].var()
        selected = variances.nlargest(min(top_k, len(variances))).index.tolist()

    return selected


# =============================================
#  Meta columns to exclude from features
# =============================================

META_COLUMNS = {
    "No", "label", "SMILES_raw", "SMILES", "_can", "canonical_smiles",
    "standardized_smiles", "analysis_smiles", ANALYSIS_SMILES_COL,
    "murcko_scaffold", "scaffold", "scaffold_group",
    "scaffold_group_type", "scaffold_type",
    "endpoint", "source_dataset", "split", "_scenario",
    "merge_id", "No_original", "sanitized", "salt_stripped",
    "charge_normed", "error", "domain", "status",
    "__augmented__", "has_metal_feature", "has_Sn",
    "smiles_stripped", "smiles_canonical",
}
