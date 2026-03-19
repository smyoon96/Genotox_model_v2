"""
qm_descriptors.py — QM / 전자적 기술자 계산 모듈
=================================================
RDKit 기반 전자적 descriptor를 SMILES에서 직접 계산한다.
외부 QM 파일(Gaussian/ORCA 등)이 있으면 merge도 수행한다.

계산되는 descriptor:
  A. RDKit 전자적 기술자 (SMILES → 즉시 계산)
     - Gasteiger partial charges: max, min, mean, range
     - HOMO/LUMO proxy (EState index 기반)
     - 전기음성도 관련: chi, kappa
     - dipole moment proxy, polarizability proxy
  B. Frontier orbital 근사 (Extended Hückel 수준)
     - HOMO_proxy, LUMO_proxy, gap_proxy
     - electrophilicity_index, chemical_hardness, chemical_softness
  C. 외부 QM CSV merge (optional)
     - HOMO, LUMO, gap, dipole_moment, polarizability 등
"""
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, AllChem, EState, rdMolDescriptors,
        rdPartialCharges
    )
    from rdkit.Chem.EState import EState_VSA
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not installed — QM descriptor computation disabled")


# ═══════════════════════════════════════════════════════════════════════
# A. 단일 분자 전자적 기술자 계산
# ═══════════════════════════════════════════════════════════════════════

def compute_electronic_descriptors_single(mol) -> Dict[str, float]:
    """
    단일 RDKit mol 객체에서 전자적 descriptor를 계산한다.
    Returns: descriptor dict (key → float)
    """
    desc = {}
    if mol is None:
        return {k: np.nan for k in _ELECTRONIC_KEYS}

    try:
        # ── 1. Gasteiger Partial Charges ──
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            q = atom.GetDoubleProp("_GasteigerCharge")
            if np.isfinite(q):
                charges.append(q)
        if charges:
            charges = np.array(charges)
            desc["gasteiger_max_charge"] = float(np.max(charges))
            desc["gasteiger_min_charge"] = float(np.min(charges))
            desc["gasteiger_mean_charge"] = float(np.mean(charges))
            desc["gasteiger_charge_range"] = float(np.max(charges) - np.min(charges))
            desc["gasteiger_max_abs_charge"] = float(np.max(np.abs(charges)))
            desc["gasteiger_pos_charge_sum"] = float(np.sum(charges[charges > 0]))
            desc["gasteiger_neg_charge_sum"] = float(np.sum(charges[charges < 0]))
        else:
            for k in ["gasteiger_max_charge", "gasteiger_min_charge",
                       "gasteiger_mean_charge", "gasteiger_charge_range",
                       "gasteiger_max_abs_charge", "gasteiger_pos_charge_sum",
                       "gasteiger_neg_charge_sum"]:
                desc[k] = np.nan

        # ── 2. HOMO / LUMO proxy (EState 기반) ──
        # MaxAbsEStateIndex → HOMO 관련, MinAbsEStateIndex → LUMO 관련
        desc["HOMO_proxy"] = float(Descriptors.MaxAbsEStateIndex(mol))
        desc["LUMO_proxy"] = float(Descriptors.MinAbsEStateIndex(mol))
        desc["MaxEStateIndex"] = float(Descriptors.MaxEStateIndex(mol))
        desc["MinEStateIndex"] = float(Descriptors.MinEStateIndex(mol))

        # gap proxy
        homo_p = desc["HOMO_proxy"]
        lumo_p = desc["LUMO_proxy"]
        if np.isfinite(homo_p) and np.isfinite(lumo_p) and homo_p != 0:
            desc["gap_proxy"] = abs(homo_p - lumo_p)
            # 전기화학적 파라미터 근사
            mu = (homo_p + lumo_p) / 2  # chemical potential proxy
            eta = (homo_p - lumo_p) / 2  # hardness proxy
            desc["chemical_potential_proxy"] = float(mu)
            desc["chemical_hardness_proxy"] = float(abs(eta)) if eta != 0 else np.nan
            desc["chemical_softness_proxy"] = float(1 / (2 * abs(eta))) if eta != 0 else np.nan
            desc["electrophilicity_proxy"] = float(mu**2 / (2 * abs(eta))) if eta != 0 else np.nan
        else:
            desc["gap_proxy"] = np.nan
            desc["chemical_potential_proxy"] = np.nan
            desc["chemical_hardness_proxy"] = np.nan
            desc["chemical_softness_proxy"] = np.nan
            desc["electrophilicity_proxy"] = np.nan

        # ── 3. EState VSA descriptors (전자 밀도 분포) ──
        estate_vsa = EState_VSA.EState_VSA_(mol)
        for i, val in enumerate(estate_vsa):
            desc[f"EState_VSA{i+1}"] = float(val)

        # ── 4. Polarizability / 분극성 proxy ──
        desc["LabuteASA"] = float(Descriptors.LabuteASA(mol))
        desc["PEOE_VSA_sum"] = float(sum(rdMolDescriptors.PEOE_VSA_(mol)))

        # ── 5. 전기음성도 관련 chi / kappa ──
        desc["Chi0"] = float(Descriptors.Chi0(mol))
        desc["Chi0n"] = float(Descriptors.Chi0n(mol))
        desc["Chi1"] = float(Descriptors.Chi1(mol))
        desc["Chi1n"] = float(Descriptors.Chi1n(mol))
        desc["Kappa1"] = float(Descriptors.Kappa1(mol))
        desc["Kappa2"] = float(Descriptors.Kappa2(mol))
        desc["Kappa3"] = float(Descriptors.Kappa3(mol))

        # ── 6. 추가 전자적 속성 ──
        desc["NumValenceElectrons"] = float(Descriptors.NumValenceElectrons(mol))
        desc["NumRadicalElectrons"] = float(Descriptors.NumRadicalElectrons(mol))
        desc["MaxPartialCharge"] = float(Descriptors.MaxPartialCharge(mol))
        desc["MinPartialCharge"] = float(Descriptors.MinPartialCharge(mol))
        desc["MaxAbsPartialCharge"] = float(Descriptors.MaxAbsPartialCharge(mol))
        desc["MinAbsPartialCharge"] = float(Descriptors.MinAbsPartialCharge(mol))

        # ── 7. 전자 이동/반응성 관련 ──
        desc["TPSA"] = float(Descriptors.TPSA(mol))
        desc["MolLogP"] = float(Descriptors.MolLogP(mol))
        desc["MolMR"] = float(Descriptors.MolMR(mol))  # Molar refractivity → polarizability proxy

    except Exception as e:
        logger.debug(f"Electronic descriptor error: {e}")
        # 실패 시 NaN
        for k in list(desc.keys()):
            if not np.isfinite(desc.get(k, 0)):
                desc[k] = np.nan

    return desc


# 기본 electronic descriptor key 목록
_ELECTRONIC_KEYS = [
    "gasteiger_max_charge", "gasteiger_min_charge", "gasteiger_mean_charge",
    "gasteiger_charge_range", "gasteiger_max_abs_charge",
    "gasteiger_pos_charge_sum", "gasteiger_neg_charge_sum",
    "HOMO_proxy", "LUMO_proxy", "MaxEStateIndex", "MinEStateIndex",
    "gap_proxy", "chemical_potential_proxy", "chemical_hardness_proxy",
    "chemical_softness_proxy", "electrophilicity_proxy",
    "LabuteASA", "PEOE_VSA_sum",
    "Chi0", "Chi0n", "Chi1", "Chi1n", "Kappa1", "Kappa2", "Kappa3",
    "NumValenceElectrons", "NumRadicalElectrons",
    "MaxPartialCharge", "MinPartialCharge",
    "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "TPSA", "MolLogP", "MolMR",
]

# ═══════════════════════════════════════════════════════════════════════
# B. DataFrame에 전자적 기술자 일괄 추가
# ═══════════════════════════════════════════════════════════════════════

def add_electronic_descriptors(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    prefix: str = "qm_",
) -> pd.DataFrame:
    """
    DataFrame의 SMILES에서 전자적 descriptor를 계산하여 컬럼으로 추가.
    모든 컬럼명에 prefix를 붙여 block 구분.
    """
    if not HAS_RDKIT:
        logger.warning("RDKit required — electronic descriptors skipped")
        return df

    if smiles_col not in df.columns:
        logger.warning(f"SMILES column '{smiles_col}' not found")
        return df

    logger.info(f"Computing electronic descriptors from '{smiles_col}' ({len(df)} molecules)...")

    all_descs = []
    n_failed = 0
    for smi in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
            desc = compute_electronic_descriptors_single(mol)
        except Exception:
            desc = {k: np.nan for k in _ELECTRONIC_KEYS}
            n_failed += 1
        all_descs.append(desc)

    desc_df = pd.DataFrame(all_descs, index=df.index)

    # prefix 추가
    desc_df.columns = [f"{prefix}{c}" for c in desc_df.columns]

    # NaN/Inf 처리
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"  → {desc_df.shape[1]} electronic descriptors computed "
                f"(failed: {n_failed}/{len(df)})")

    return pd.concat([df, desc_df], axis=1)


# ═══════════════════════════════════════════════════════════════════════
# C. 외부 QM 파일 merge
# ═══════════════════════════════════════════════════════════════════════

def merge_external_qm(
    df: pd.DataFrame,
    qm_path,
    merge_key: str = "No",
    qm_cols: Optional[List[str]] = None,
    prefix: str = "ext_qm_",
) -> Tuple[pd.DataFrame, dict]:
    """
    외부 QM 계산 결과 CSV를 merge.
    Returns: (merged df, merge_report dict)
    """
    from pathlib import Path
    from utils.data_utils import normalize_merge_key

    report = {"status": "ok", "n_merged": 0, "n_missing": 0}
    qm_path = Path(qm_path)

    if not qm_path.exists():
        report["status"] = "file_not_found"
        logger.info(f"External QM file not found: {qm_path} — skip")
        return df, report

    try:
        df_qm = pd.read_csv(qm_path, low_memory=False)
    except Exception as e:
        report["status"] = f"read_error: {e}"
        return df, report

    if merge_key not in df_qm.columns:
        # canonical_smiles로 fallback
        for alt_key in ["canonical_smiles", "SMILES", "smiles"]:
            if alt_key in df_qm.columns and alt_key in df.columns:
                merge_key = alt_key
                break
        else:
            report["status"] = "no_merge_key"
            return df, report

    # merge key 정규화
    if merge_key == "No":
        df_qm["No"] = normalize_merge_key(df_qm["No"])

    # QM 컬럼 선택
    default_qm = ["HOMO", "LUMO", "gap", "dipole_moment", "polarizability",
                   "max_atomic_charge", "min_atomic_charge",
                   "electrophilicity_index", "hardness", "softness",
                   "homo", "lumo", "dipole", "electron_affinity",
                   "ionization_potential"]
    if qm_cols is None:
        qm_cols = [c for c in df_qm.columns if c.lower() in [q.lower() for q in default_qm]]

    if not qm_cols:
        report["status"] = "no_qm_columns"
        logger.warning("No QM columns found in external file")
        return df, report

    qm_sub = df_qm[[merge_key] + qm_cols].drop_duplicates(subset=[merge_key])
    qm_sub.columns = [merge_key] + [f"{prefix}{c}" for c in qm_cols]

    # merge
    n_before = len(df)
    df = df.merge(qm_sub, on=merge_key, how="left")

    # 통계
    first_qm_col = f"{prefix}{qm_cols[0]}"
    if first_qm_col in df.columns:
        n_filled = df[first_qm_col].notna().sum()
        report["n_merged"] = int(n_filled)
        report["n_missing"] = int(len(df) - n_filled)
        report["coverage"] = round(n_filled / len(df), 4)
        report["qm_columns"] = qm_cols

    logger.info(f"  External QM merged: {report['n_merged']}/{len(df)} "
                f"({report.get('coverage', 0)*100:.1f}% coverage)")

    return df, report


# ═══════════════════════════════════════════════════════════════════════
# D. QM descriptor 요약 / 품질 리포트
# ═══════════════════════════════════════════════════════════════════════

def qm_descriptor_summary(df: pd.DataFrame, prefix: str = "qm_") -> pd.DataFrame:
    """QM descriptor 블록의 품질 요약"""
    qm_cols = [c for c in df.columns if c.startswith(prefix)]
    if not qm_cols:
        return pd.DataFrame()

    rows = []
    for col in qm_cols:
        series = df[col]
        rows.append({
            "descriptor": col.replace(prefix, ""),
            "full_name": col,
            "count": int(series.notna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "mean": round(series.mean(), 6) if series.notna().any() else np.nan,
            "std": round(series.std(), 6) if series.notna().any() else np.nan,
            "min": round(series.min(), 6) if series.notna().any() else np.nan,
            "max": round(series.max(), 6) if series.notna().any() else np.nan,
            "has_inf": int(np.isinf(series.replace(np.nan, 0)).sum()),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# E. Endpoint별 QM feature 추천
# ═══════════════════════════════════════════════════════════════════════

# 유전독성 endpoint별 QM descriptor 관련성
QM_RELEVANCE = {
    "ames": {
        "high": ["electrophilicity_proxy", "gasteiger_max_charge",
                  "LUMO_proxy", "gap_proxy", "chemical_softness_proxy"],
        "medium": ["gasteiger_charge_range", "MaxPartialCharge",
                    "chemical_hardness_proxy", "HOMO_proxy"],
        "reason": "Ames 양성은 electrophilic reactivity와 높은 상관 — "
                  "LUMO가 낮을수록 친전자 공격 용이"
    },
    "invitro": {
        "high": ["electrophilicity_proxy", "gasteiger_max_abs_charge",
                  "gap_proxy", "HOMO_proxy", "chemical_softness_proxy"],
        "medium": ["MolMR", "LabuteASA", "PEOE_VSA_sum", "Chi1n"],
        "reason": "In vitro 염색체이상은 DNA intercalation + charge transfer 관련 — "
                  "HOMO 높으면 전자 공여 용이"
    },
    "invivo": {
        "high": ["gap_proxy", "chemical_softness_proxy", "MolMR",
                  "LabuteASA", "TPSA"],
        "medium": ["electrophilicity_proxy", "HOMO_proxy",
                    "gasteiger_charge_range", "Kappa2"],
        "reason": "In vivo 소핵은 대사 활성화 + 생체이용률 관련 — "
                  "polarizability와 molecular size가 중요"
    },
}
