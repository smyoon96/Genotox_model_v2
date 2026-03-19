"""
feature_utils.py — Feature 추출 · FP 생성 · fold 내부 bit selection
===================================================================
"""
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# RDKit import (optional — 없으면 FP 생성 skip)
# ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not installed — fingerprint generation disabled")


# ─────────────────────────────────────────────────────────────────────
# 1. Morgan Fingerprint 생성
# ─────────────────────────────────────────────────────────────────────

def smiles_to_morgan(smiles_series: pd.Series, radius: int = 2,
                     n_bits: int = 256) -> Optional[np.ndarray]:
    """SMILES → Morgan FP numpy array (N × n_bits)"""
    if not HAS_RDKIT:
        logger.error("RDKit required for fingerprint generation")
        return None

    arr = np.zeros((len(smiles_series), n_bits), dtype=np.int8)
    failed = 0
    for i, smi in enumerate(smiles_series):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                failed += 1
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr[i])
        except Exception:
            failed += 1
    if failed > 0:
        logger.warning(f"Morgan FP: {failed}/{len(smiles_series)} failed")
    return arr


def add_fp_columns(df: pd.DataFrame, smiles_col: str = "canonical_smiles",
                   radius: int = 2, n_bits: int = 256) -> pd.DataFrame:
    """DataFrame에 fp_0 .. fp_{n_bits-1} 컬럼 추가"""
    if smiles_col not in df.columns:
        logger.warning(f"SMILES column '{smiles_col}' not found — skip FP")
        return df
    arr = smiles_to_morgan(df[smiles_col], radius, n_bits)
    if arr is None:
        return df
    fp_cols = [f"fp_{i}" for i in range(n_bits)]
    fp_df = pd.DataFrame(arr, columns=fp_cols, index=df.index)
    return pd.concat([df, fp_df], axis=1)


# ─────────────────────────────────────────────────────────────────────
# 2. Valid 컬럼 boolean 변환
# ─────────────────────────────────────────────────────────────────────

def safe_bool_mask(series: pd.Series) -> pd.Series:
    """bool/int/string 혼합 valid 컬럼을 안전하게 0/1로 변환"""
    def _to_bool(v):
        if pd.isna(v):
            return 0
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v != 0)
        s = str(v).strip().lower()
        return 1 if s in ("true", "1", "yes", "valid") else 0
    return series.map(_to_bool).astype(np.int8)


# ─────────────────────────────────────────────────────────────────────
# 3. Feature block 분류
# ─────────────────────────────────────────────────────────────────────

def classify_feature_blocks(columns: List[str]) -> dict:
    """컬럼 이름으로 feature block 분류"""
    blocks = {
        "physchem": [],
        "fg_present": [],
        "fg_count": [],
        "rule": [],
        "fingerprint": [],
        "qm": [],
        "other": [],
    }
    physchem_keys = {"mw", "logp", "tpsa", "rot_bonds", "hba", "hbd",
                     "fraction_csp3", "aromatic_ring_count", "ring_count",
                     "num_heavy_atoms", "mol_weight"}
    qm_keys = {"homo", "lumo", "gap", "dipole", "polarizability",
                "electrophilicity", "hardness", "softness", "charge",
                "gasteiger", "estate", "kappa", "chi0", "chi1",
                "peoe_vsa", "labute", "molmr", "valence_electron",
                "radical_electron", "partial_charge", "chemical_potential"}

    for col in columns:
        cl = col.lower()
        if cl.startswith("fp_"):
            blocks["fingerprint"].append(col)
        elif cl.startswith("qm_") or cl.startswith("ext_qm_"):
            blocks["qm"].append(col)
        elif cl.startswith("fg_") and "present" in cl:
            blocks["fg_present"].append(col)
        elif cl.startswith("fg_") and "count" in cl:
            blocks["fg_count"].append(col)
        elif any(k in cl for k in qm_keys):
            blocks["qm"].append(col)
        elif any(k in cl for k in physchem_keys) or cl in physchem_keys:
            blocks["physchem"].append(col)
        elif "rule" in cl or "alert" in cl or "genotox" in cl:
            blocks["rule"].append(col)
        else:
            blocks["other"].append(col)
    return blocks


# ─────────────────────────────────────────────────────────────────────
# 4. Fold 내부 fingerprint bit selection (LEAKAGE-FREE)
# ─────────────────────────────────────────────────────────────────────

def select_fp_bits_in_fold(X_train_fp: np.ndarray, y_train: np.ndarray,
                           k: int = 128,
                           prevalence_min: float = 0.01,
                           variance_min: float = 0.005,
                           method: str = "mutual_info") -> np.ndarray:
    """
    CV fold 내부에서만 fingerprint bit selection 수행.
    Returns: 선택된 bit 인덱스 배열

    ★ 이 함수는 반드시 train fold에서만 호출한다.
    ★ validation/test에는 여기서 결정된 bit만 적용한다.
    """
    n_samples, n_bits = X_train_fp.shape

    # Step 1: prevalence filter
    prevalence = X_train_fp.mean(axis=0)
    prev_mask = prevalence >= prevalence_min

    # Step 2: variance filter
    variance = X_train_fp.var(axis=0)
    var_mask = variance >= variance_min

    combined_mask = prev_mask & var_mask
    surviving_idx = np.where(combined_mask)[0]

    if len(surviving_idx) == 0:
        logger.warning("No FP bits survived prevalence/variance filter")
        return np.array([], dtype=int)

    if len(surviving_idx) <= k:
        return surviving_idx

    # Step 3: MI 또는 chi2로 상위 k개 선택
    X_sub = X_train_fp[:, surviving_idx]
    if method == "mutual_info":
        scores = mutual_info_classif(X_sub, y_train, random_state=42)
    elif method == "chi2":
        scores, _ = chi2(X_sub, y_train)
    else:
        scores = mutual_info_classif(X_sub, y_train, random_state=42)

    top_k_local = np.argsort(scores)[-k:]
    selected = surviving_idx[top_k_local]
    return np.sort(selected)


# ─────────────────────────────────────────────────────────────────────
# 5. Feature matrix 준비 (numeric/categorical 분리)
# ─────────────────────────────────────────────────────────────────────

def separate_num_cat(df: pd.DataFrame, feature_cols: List[str]
                     ) -> Tuple[List[str], List[str]]:
    """숫자 / 범주형 컬럼 분리 — string, object, category 모두 categorical"""
    numeric = []
    categorical = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def is_leakage_col(col: str) -> bool:
    """target leakage / 내부 마커 컬럼인지 판별"""
    import config as cfg
    cl = col.lower()
    if col in cfg.META_COLS or col.lower() in [m.lower() for m in cfg.META_COLS]:
        return True
    if any(pat.lower() in cl for pat in cfg.EXCLUDE_PATTERNS):
        return True
    if cl in ("index", "unnamed: 0", "unnamed:_0"):
        return True
    return False


def get_clean_feature_cols(df: pd.DataFrame, include_fp: bool = True,
                           include_qm: bool = False) -> List[str]:
    """leakage-free feature 컬럼 목록 추출"""
    import config as cfg
    cols = []
    for col in df.columns:
        if is_leakage_col(col):
            continue
        if not include_fp and col.startswith("fp_"):
            continue
        if not include_qm and col.lower() in [q.lower() for q in cfg.QM_COLS]:
            continue
        cols.append(col)
    return cols
