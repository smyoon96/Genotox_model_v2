"""
utils/feature_utils.py -- Feature 컬럼 분류 / FP 생성 유틸리티
"""
import sys, logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from utils.data_utils import classify_columns

logger = logging.getLogger("feature_utils")


# ──────────────────────────────────────────────
#  Feature 블록 분류
# ──────────────────────────────────────────────

def classify_feature_blocks(feature_cols: List[str]) -> Dict[str, List[str]]:
    """feature 컬럼 목록을 블록별로 분류."""
    blocks = {
        "fingerprint": [],
        "structural_alert": [],
        "functional_group": [],
        "physchem": [],
        "qm": [],
        "other": [],
    }
    for col in feature_cols:
        if col.startswith("fp_"):
            blocks["fingerprint"].append(col)
        elif col.startswith("bb_") or col.startswith("kz_") or col.startswith("iss_") or col.startswith("mn_") or col.startswith("met_"):
            blocks["structural_alert"].append(col)
        elif col.endswith("_present") or col.startswith("fg_"):
            blocks["functional_group"].append(col)
        elif col.startswith(cfg.QM_PREFIX) or col.startswith(cfg.EXT_QM_PREFIX):
            blocks["qm"].append(col)
        elif any(col.startswith(p) for p in ["mw_", "logp_", "tpsa_", "hbd_", "hba_",
                                               "rot_", "arom_", "ring_", "heavy_", "frac_"]):
            blocks["physchem"].append(col)
        else:
            blocks["other"].append(col)
    return blocks


def get_clean_feature_cols(df: pd.DataFrame, include_fp: bool = True) -> List[str]:
    """df에서 feature 컬럼 목록 반환 (meta/excluded 제외)."""
    col_class = classify_columns(df)
    features = col_class["features"]
    if not include_fp:
        features = [c for c in features if not c.startswith("fp_")]
    return features


def separate_num_cat(df: pd.DataFrame,
                     feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    """feature 컬럼을 numeric / categorical 분리."""
    num_cols, cat_cols = [], []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype in ("object", "category", "bool"):
            cat_cols.append(col)
        else:
            try:
                df[col].astype(float)
                num_cols.append(col)
            except (ValueError, TypeError):
                cat_cols.append(col)
    return num_cols, cat_cols


# ──────────────────────────────────────────────
#  Morgan Fingerprint 추가
# ──────────────────────────────────────────────

def add_fp_columns(df: pd.DataFrame, smiles_col: str,
                   radius: int = 2, n_bits: int = 1024) -> pd.DataFrame:
    """DataFrame에 Morgan FP 컬럼 추가. 계산 실패 시 0으로 채움."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.warning("RDKit not available -- FP columns not added")
        return df

    df = df.copy()
    fp_prefix = "fp_"
    fp_cols = [f"{fp_prefix}{i}" for i in range(n_bits)]

    fp_matrix = []
    for smi in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise ValueError("Invalid SMILES")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fp_matrix.append(list(fp))
        except Exception:
            fp_matrix.append([0] * n_bits)

    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    return pd.concat([df, fp_df], axis=1)


# safe_bool_mask alias (step2 호환)
from utils.data_utils import safe_bool_mask  # re-export
