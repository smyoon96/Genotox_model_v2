"""
step4_feature_extraction.py -- v12 (2026-03-27)
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

# ─── SMARTS 패턴 캐시 ────────────────────────────────────────────────
# MolFromSmarts를 compound마다 반복 호출하면 O(n_mol × n_smarts) 재컴파일 발생
# (invivo_sampling/no_metal 123패턴 × 수백 화합물 = 수만 번 재컴파일)
# → 모듈 로드 시점에 한 번만 컴파일하여 캐시
_SMARTS_CACHE: Dict[str, object] = {}

def _get_pat(smarts_str: str):
    if smarts_str not in _SMARTS_CACHE:
        _SMARTS_CACHE[smarts_str] = Chem.MolFromSmarts(smarts_str)
    return _SMARTS_CACHE[smarts_str]


def _count_smarts(mol, smarts_str: str) -> int:
    """
    주어진 mol에서 SMARTS 패턴 매칭 수를 반환.
    maxMatches=1000 제한으로 지수적 backtracking hang 방지
    (규제독성 데이터에서 1000+ 매치는 실질적으로 발생하지 않음).
    """
    try:
        pat = _get_pat(smarts_str)
        if pat is None:
            return 0
        return len(mol.GetSubstructMatches(pat, maxMatches=1000))
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


# =============================================
#  Additional Fingerprint Types (Strategy 3)
# =============================================

def extract_maccs_features(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """MACCS keys (166 structural keys) — Morgan과 상보적인 substructure 정보."""
    from rdkit.Chem import MACCSkeys
    smi_col = find_smi(df)
    n_bits = 167  # MACCS returns 167 bits (index 0 unused)
    fp_matrix = np.zeros((len(df), n_bits), dtype=np.int8)
    for i, smi in enumerate(df[smi_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros(n_bits, dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                fp_matrix[i] = arr
        except Exception:
            pass
    fp_cols = [f"maccs_{i:03d}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    logger.info(f"  [{endpoint}] MACCS keys: {n_bits} bits")
    return fp_df


def extract_atompair_features(
    df: pd.DataFrame, endpoint: str, n_bits: int = 1024
) -> pd.DataFrame:
    """AtomPair fingerprint — 원자 쌍 간 topological distance 인코딩."""
    from rdkit.Chem import rdFingerprintGenerator
    smi_col = find_smi(df)
    fpg = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    fp_matrix = np.zeros((len(df), n_bits), dtype=np.int8)
    for i, smi in enumerate(df[smi_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = fpg.GetFingerprintAsNumPy(mol)
                fp_matrix[i] = (fp > 0).astype(np.int8)
        except Exception:
            pass
    fp_cols = [f"ap_{i:04d}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    logger.info(f"  [{endpoint}] AtomPair FP: {n_bits} bits")
    return fp_df


def extract_toptorsion_features(
    df: pd.DataFrame, endpoint: str, n_bits: int = 1024
) -> pd.DataFrame:
    """TopologicalTorsion fingerprint — rotatable torsion 패턴 인코딩."""
    from rdkit.Chem import rdFingerprintGenerator
    smi_col = find_smi(df)
    fpg = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    fp_matrix = np.zeros((len(df), n_bits), dtype=np.int8)
    for i, smi in enumerate(df[smi_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = fpg.GetFingerprintAsNumPy(mol)
                fp_matrix[i] = (fp > 0).astype(np.int8)
        except Exception:
            pass
    fp_cols = [f"tt_{i:04d}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    logger.info(f"  [{endpoint}] TopTorsion FP: {n_bits} bits")
    return fp_df


def extract_multi_fp_union(
    df: pd.DataFrame, endpoint: str, morgan_bits: int = 1024,
    ap_bits: int = 512, tt_bits: int = 512,
) -> pd.DataFrame:
    """
    Multi-FP union: Morgan + MACCS + AtomPair + TopTorsion.
    각 FP가 다른 structural 정보를 인코딩하여 상보적 커버리지 제공.
    """
    morgan = extract_fingerprint_features(df, endpoint, n_bits=morgan_bits)
    maccs = extract_maccs_features(df, endpoint)
    ap = extract_atompair_features(df, endpoint, n_bits=ap_bits)
    tt = extract_toptorsion_features(df, endpoint, n_bits=tt_bits)
    union = pd.concat([morgan, maccs, ap, tt], axis=1)
    logger.info(f"  [{endpoint}] Multi-FP union: {union.shape[1]} total features")
    return union


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


# =============================================
#  4-5. Unified Feature Builder (v12.2)
# =============================================

def build_all_features(
    df: pd.DataFrame,
    endpoint: str,
    include_fp: bool = True,
    fp_n_bits: int = 512,
    fp_selected_bits=None,
    fp_top_k: int = 128,
    fp_method: str = "mi",
    qm_dir=None,
    include_electronic: bool = True,
    electronic_levels=None,
    electronic_n_jobs: int = 1,
    return_fp_full: bool = False,
) -> dict:
    """
    모든 feature 블록을 통합하여 반환.

    Feature 블록:
      ① Functional Group (FG) -- 615개
         fg_*_count / fg_*_present / rule_*_ge1/2/3 / bb_* / positive_alert_score
      ② Physicochemical -- 13개
         MW, LogP, TPSA, HBD, HBA, RotatableBonds, FractionCSP3,
         AromaticRingCount, RingCount, HeteroatomCount, HeavyAtomCount,
         FormalCharge, NumValenceElectrons
      ③ Electronic / QM (v12.2 신규)
         Level 1 (RDKit, 항상):
           Gasteiger 부분 전하, EState, Chi, Kappa, BCUT, PEOE_VSA, ...
         Level 2 (Mordred, pip install mordred):
           확장 전자 기술자 ~200개
         Level 3 (xTB, conda install xtb-python):
           HOMO/LUMO energy, gap, 쌍극자, IP, EA, 친전자성 지수
      ④ Fingerprint (선택적)
         Morgan FP (radius=2), MI 기반 bit selection
      ⑤ 외부 QM CSV (qm_dir 제공 시)

    Parameters
    ----------
    df              : endpoint 데이터프레임 (SMILES, label 포함)
    endpoint        : "ames" | "invitro" | "invivo" | ...
    include_fp      : Morgan FP 포함 여부
    fp_n_bits       : FP bit 수
    fp_selected_bits: 미리 선택된 bit 목록 (train에서 결정 후 test에 전달)
    fp_top_k        : MI 기반 선택 bit 수 (fp_selected_bits=None일 때)
    fp_method       : bit 선택 기준 "mi" | "variance"
    qm_dir          : QM descriptor CSV 디렉토리 (Path or None)
    return_fp_full  : True면 full FP DataFrame도 반환 (AD 계산용)

    Returns
    -------
    dict:
        X             : np.ndarray (n_samples, n_features)
        feature_cols  : List[str]
        fp_selected   : List[str] | None  (train에서 선택된 FP bit 목록)
        fp_full       : pd.DataFrame | None  (return_fp_full=True일 때)
        block_sizes   : dict  각 블록의 feature 수
        qm_report     : dict
    """
    import numpy as _np
    import pandas as _pd

    # ── ① Functional Group ───────────────────────────────────────────
    fg_df = extract_fg_features(df, endpoint)
    # 모든 FG feature 포함 (count, present, rule_*, bb_*, score)
    fg_cols = [c for c in fg_df.columns
               if not c.startswith("_")]  # 내부 임시 컬럼만 제외
    fg_block = fg_df[fg_cols]

    # ── ② Physicochemical ────────────────────────────────────────────
    ph_df = extract_physchem_features(df, endpoint)
    ph_cols = list(ph_df.columns)

    # ── ③ Fingerprint ────────────────────────────────────────────────
    fp_selected = fp_selected_bits
    fp_full_df  = None
    fp_block    = _pd.DataFrame(index=df.index)

    if include_fp:
        fp_full_df = extract_fingerprint_features(df, endpoint, n_bits=fp_n_bits)
        if fp_selected is None:
            y = df["label"].values.astype(int) if "label" in df.columns else None
            fp_selected = select_fingerprint_bits(
                fp_full_df, y, top_k=fp_top_k, method=fp_method)
        fp_block = fp_full_df[fp_selected]

    # ── ③ Electronic / QM features (v12.2) ─────────────────────────
    elec_block = _pd.DataFrame(index=df.index)
    if include_electronic:
        try:
            from utils.electronic_descriptors import (
                extract_electronic_features, get_available_levels, install_guide)
            avail = get_available_levels()
            if any(avail.values()):
                elec_block = extract_electronic_features(
                    df, levels=electronic_levels, n_jobs=electronic_n_jobs)
            else:
                logger.info(f"  [{endpoint}] No QM library: {install_guide()}")
        except Exception as e:
            logger.warning(f"  [{endpoint}] Electronic features failed: {e}")

    # ── ④ 외부 QM CSV (qm_dir 제공 시) ──────────────────────────────
    qm_block  = _pd.DataFrame(index=df.index)
    qm_report = {"qm_available": False}
    if qm_dir is not None:
        try:
            qm_block, qm_report = extract_qm_features(df, endpoint,
                                                        qm_dir=qm_dir)
        except Exception as e:
            logger.warning(f"  [{endpoint}] External QM failed: {e}")

    # ── 통합 ─────────────────────────────────────────────────────────
    parts = [fg_block, ph_df]
    if not elec_block.empty:
        parts.append(elec_block)
    if include_fp and not fp_block.empty:
        parts.append(fp_block)
    if not qm_block.empty:
        parts.append(qm_block)

    feat = _pd.concat(parts, axis=1)
    for c in feat.columns:
        feat[c] = _pd.to_numeric(feat[c], errors="coerce")
    feat = feat.fillna(0.0)

    X = feat.values.astype(_np.float32)

    block_sizes = {
        "fg":         len(fg_cols),
        "physchem":   len(ph_cols),
        "electronic": len(elec_block.columns) if not elec_block.empty else 0,
        "fp":         len(fp_block.columns) if not fp_block.empty else 0,
        "qm_ext":     len(qm_block.columns) if not qm_block.empty else 0,
        "total":      feat.shape[1],
    }
    logger.info(
        f"  [{endpoint}] Features: "
        f"FG={block_sizes['fg']} | "
        f"Physchem={block_sizes['physchem']} | "
        f"Electronic={block_sizes['electronic']} | "
        f"FP={block_sizes['fp']} | "
        f"Total={block_sizes['total']}"
    )

    return {
        "X":            X,
        "feature_cols": list(feat.columns),
        "fp_selected":  fp_selected,
        "fp_full":      fp_full_df if return_fp_full else None,
        "block_sizes":  block_sizes,
        "qm_report":    qm_report,
    }
