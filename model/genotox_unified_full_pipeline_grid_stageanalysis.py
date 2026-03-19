
from __future__ import annotations

import argparse
import io
import importlib.util
import json
import math
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import shutil
import subprocess
import warnings
import zipfile
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit.Chem import AllChem, rdMolDescriptors, rdMolHash
try:
    from rdkit.Chem import rdFingerprintGenerator
    HAVE_MORGAN_GENERATOR = True
except Exception:
    HAVE_MORGAN_GENERATOR = False
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*")
warnings.filterwarnings("ignore", message=".*libpng warning.*")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

RANDOM_STATE = 42
DEFAULT_FP_BITS = 256

ENDPOINTS = {
    "ames": {
        "raw_name": "ames_combine",
        "raw_file": "ames_combine.xlsx",
        "raw_sheet": "ames_pre",
        "representation": "canonical",
        "label_col": "label",
        "threshold_metric": "mcc",
        "specificity_floor": 0.0,
        "sampling": "none",
        "model": "xgb",
    },
    "invitro": {
        "raw_name": "invitro_pre",
        "raw_file": "invitro_pre.csv",
        "representation": "canonical",
        "label_col": "label",
        "threshold_metric": "mcc",
        "specificity_floor": 0.75,
        "sampling": "coverage_hardneg",
        "model": "xgb",
    },
    "invivo": {
        "raw_name": "invivo_pre",
        "raw_file": "invivo_pre.csv",
        "representation": "canonical",
        "label_col": "label",
        "threshold_metric": "balanced_accuracy",
        "specificity_floor": 0.80,
        "sampling": "coverage_hardneg",
        "model": "xgb",
    },
}

PHYS_CHEM_COLS = [
    "mw", "logp", "tpsa", "hbd", "hba", "rot_bonds", "ring_count",
    "aromatic_ring_count", "heteroatom_count", "fraction_csp3",
    "heavy_atom_count", "formal_charge_total", "has_formal_charge",
]

RULE_SPECS = {
    "ames": [
        ("fg_epoxide_count", "ge", 1),
        ("fg_epoxide_count", "ge", 2),
        ("fg_BB_SA5_nitro_aromatic_count", "ge", 2),
        ("fg_BB_SA10_N_nitroso_count", "ge", 1),
        ("fg_hydrazine_like_count", "ge", 1),
        ("fg_fluoro_count", "ge", 5),
        ("fg_sulfone_count", "ge", 2),
        ("fg_ester_count", "ge", 2),
        ("fg_BB_SA47_n_alkylcarboxylic_acid_count", "ge", 1),
    ],
    "invitro": [
        ("fg_BB_SA2_primary_aromatic_amine_count", "ge", 1),
        ("fg_BB_SA2_primary_aromatic_amine_count", "ge", 2),
        ("fg_aniline_count", "ge", 1),
        ("fg_epoxide_count", "ge", 1),
        ("fg_BB_SA20_ab_unsat_carbonyl_count", "ge", 1),
        ("fg_BB_SA47_n_alkylcarboxylic_acid_count", "ge", 1),
        ("fg_ester_count", "eq", 1),
        ("fg_alcohol_count", "ge", 2),
    ],
    "invivo": [
        ("fg_epoxide_count", "ge", 1),
        ("fg_epoxide_count", "ge", 2),
        ("fg_organometal_like_count", "ge", 1),
        ("fg_organometal_like_count", "ge", 2),
        ("fg_alcohol_count", "ge", 1),
        ("fg_fused_ring_like_count", "ge", 1),
    ],
}

EXPERT_POSITIVE = {
    "ames": {
        "fg_BB_SA10_N_nitroso_present": 1.5,
        "fg_BB_SA5_nitro_aromatic_present": 1.3,
        "fg_epoxide_present": 1.0,
        "fg_hydrazine_like_present": 0.8,
        "bb_n_genotox_alerts": 0.3,
    },
    "invitro": {
        "fg_BB_SA2_primary_aromatic_amine_present": 1.0,
        "fg_aniline_present": 0.7,
        "fg_BB_SA20_ab_unsat_carbonyl_present": 0.7,
        "fg_epoxide_present": 0.4,
        "bb_n_genotox_alerts": 0.2,
    },
    "invivo": {
        "fg_epoxide_present": 0.9,
        "fg_organometal_like_present": 0.9,
        "fg_heavy_metal_atom_present": 0.6,
        "bb_n_genotox_alerts": 0.1,
    },
}

EXPERT_NEGATIVE = {
    "ames": {
        "fg_ester_present": 0.6,
        "fg_sulfone_present": 0.6,
        "fg_BB_SA47_n_alkylcarboxylic_acid_present": 0.8,
        "fraction_csp3": 0.3,
    },
    "invitro": {
        "fg_BB_SA47_n_alkylcarboxylic_acid_present": 0.9,
        "fg_ester_present": 0.4,
        "fraction_csp3": 0.4,
        "rot_bonds": 0.2,
    },
    "invivo": {
        "fg_alcohol_present": 1.0,
        "fg_fused_ring_like_present": 0.7,
    },
}

MODEL_CONFIG = {
    "ames": {
        "logreg_distributions": {
            "C": np.logspace(-2, 1, 50),
            "l1_ratio": np.linspace(0.0, 0.8, 41),
        },
        "xgb_distributions": {
            "n_estimators": [250, 350, 450, 550],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.07],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [2, 3, 5],
            "reg_lambda": [1.0, 2.0, 4.0],
            "reg_alpha": [0.0, 0.2, 0.5],
            "gamma": [0.0, 0.1, 0.2],
        },
    },
    "invitro": {
        "logreg_distributions": {
            "C": np.logspace(-2.3, 0.7, 50),
            "l1_ratio": np.linspace(0.0, 0.8, 41),
        },
        "xgb_distributions": {
            "n_estimators": [200, 300, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.08],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [3, 4, 6],
            "reg_lambda": [2.0, 3.0, 5.0],
            "reg_alpha": [0.0, 0.3, 0.6],
            "gamma": [0.0, 0.1, 0.2],
        },
    },
    "invivo": {
        "logreg_distributions": {
            "C": np.logspace(-2.5, 0.3, 40),
            "l1_ratio": np.linspace(0.0, 0.9, 46),
        },
        "xgb_distributions": {
            "n_estimators": [150, 250, 350],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.05, 0.08],
            "subsample": [0.75, 0.85, 0.95],
            "colsample_bytree": [0.6, 0.75, 0.9],
            "min_child_weight": [4, 6, 8],
            "reg_lambda": [3.0, 5.0, 8.0],
            "reg_alpha": [0.3, 0.8, 1.2],
            "gamma": [0.0, 0.1, 0.2],
        },
    },
}


GRID_CONFIG = {
    "ames": {
        "logreg_grid": {
            "C": [0.1, 0.3, 1.0, 3.0],
            "l1_ratio": [0.0, 0.15, 0.35, 0.6],
        },
        "xgb_grid": {
            "n_estimators": [300, 450],
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.05],
            "subsample": [0.85, 1.0],
            "colsample_bytree": [0.75, 0.9],
            "min_child_weight": [2, 4],
            "reg_lambda": [1.0, 3.0],
            "reg_alpha": [0.0, 0.4],
            "gamma": [0.0, 0.1],
        },
    },
    "invitro": {
        "logreg_grid": {
            "C": [0.08, 0.2, 0.6, 1.5],
            "l1_ratio": [0.0, 0.2, 0.4, 0.7],
        },
        "xgb_grid": {
            "n_estimators": [200, 350],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.05],
            "subsample": [0.8, 0.95],
            "colsample_bytree": [0.7, 0.85],
            "min_child_weight": [3, 5],
            "reg_lambda": [2.0, 5.0],
            "reg_alpha": [0.0, 0.5],
            "gamma": [0.0, 0.1],
        },
    },
    "invivo": {
        "logreg_grid": {
            "C": [0.05, 0.15, 0.4, 1.0],
            "l1_ratio": [0.0, 0.25, 0.5, 0.8],
        },
        "xgb_grid": {
            "n_estimators": [150, 250],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.06],
            "subsample": [0.75, 0.9],
            "colsample_bytree": [0.6, 0.8],
            "min_child_weight": [4, 7],
            "reg_lambda": [3.0, 8.0],
            "reg_alpha": [0.3, 1.0],
            "gamma": [0.0, 0.15],
        },
    },
}

@dataclass
class StagePaths:
    raw_base: Path
    standardized: Path
    preprocessed: Path
    split_assignments: Path
    train_full: Path
    test_full: Path
    train_selected_raw: Path
    train_selected_full: Path


@dataclass
class RunArtifacts:
    endpoint: str
    model_name: str
    sampling_strategy: str
    representation: str
    feature_blocks: List[str]
    metrics: Dict[str, float]
    run_dir: str
    train_n: int
    test_n: int
    selected_train_n: int


def normalize_no_value(x) -> Optional[str]:
    if pd.isna(x):
        return None
    try:
        xf = float(x)
        if math.isfinite(xf) and float(int(xf)) == xf:
            return str(int(xf))
        return str(xf)
    except Exception:
        s = str(x).strip()
        if s.endswith(".0"):
            try:
                return str(int(float(s)))
            except Exception:
                return s
        return s


def normalize_no_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_no_value)


def read_csv_any(path: Path, string_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if string_cols is None:
        string_cols = []
    dtype = {c: "string" for c in string_cols}
    return pd.read_csv(path, low_memory=False, dtype=dtype)


def read_raw_endpoint(base_dir: Path, endpoint: str) -> pd.DataFrame:
    cfg = ENDPOINTS[endpoint]
    raw_path = base_dir / cfg["raw_file"]
    if not raw_path.exists():
        alt = base_dir / "data" / raw_path.name
        if alt.exists():
            raw_path = alt
    if not raw_path.exists():
        raise FileNotFoundError(f"raw file not found for {endpoint}: {cfg['raw_file']}")
    if raw_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(raw_path, sheet_name=cfg.get("raw_sheet"))
    else:
        df = read_csv_any(raw_path, string_cols=["No", "SMILES"])
    df["No_norm"] = normalize_no_series(df["No"])
    df["raw_smiles_input"] = df["SMILES"].astype(str)
    return df


def find_fg_source(base_dir: Path) -> Path:
    candidates = [
        base_dir / "fg_descriptor_analysis.zip",
        base_dir / "data" / "fg_descriptor_analysis.zip",
        base_dir / "fg_descriptor_analysis",
        base_dir / "data" / "fg_descriptor_analysis",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("fg_descriptor_analysis.zip or folder not found")


def read_csv_from_zip_or_dir(source: Path, member: str) -> pd.DataFrame:
    if source.is_dir():
        return read_csv_any(source / member, string_cols=["No", "SMILES", "raw_smiles", "canonical_smiles", "standardized_smiles"])
    with zipfile.ZipFile(source) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, low_memory=False)


def coerce_valid_mask(df: pd.DataFrame, col: str = "valid") -> pd.Series:
    if col not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype) or ptypes.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() >= 0.8:
        return s_num.fillna(0).astype(int).astype(bool)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"1", "true", "t", "yes", "y", "valid"})


def get_views_and_analysis(base_dir: Path, endpoint: str, representation: str = "canonical") -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = find_fg_source(base_dir)
    raw_stem = ENDPOINTS[endpoint]["raw_name"]
    views = read_csv_from_zip_or_dir(source, f"fg_descriptor_analysis/{raw_stem}/{raw_stem}_preprocessed_views.csv")
    analysis = read_csv_from_zip_or_dir(source, f"fg_descriptor_analysis/{raw_stem}/{raw_stem}_{representation}_analysis.csv")
    views["No_norm"] = normalize_no_series(views["No"])
    analysis["No_norm"] = normalize_no_series(analysis["No"])
    analysis = analysis.loc[coerce_valid_mask(analysis, "valid")].copy()
    return views, analysis


def sanitize_rule_name(col: str, mode: str, thr: int) -> str:
    safe = col.replace("fg_", "").replace("__", "_")
    safe = safe.replace(" ", "_")
    return f"rule_{safe}_{mode}{thr}"


def add_rule_features(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    out = df.copy()
    for col, mode, thr in RULE_SPECS[endpoint]:
        if col not in out.columns:
            continue
        name = sanitize_rule_name(col, mode, thr)
        s = pd.to_numeric(out[col], errors="coerce").fillna(0)
        if mode == "ge":
            out[name] = (s >= thr).astype(int)
        elif mode == "eq":
            out[name] = (s == thr).astype(int)
    return out


def make_scaffold_group(smiles: str, murcko: object) -> Tuple[str, str]:
    murcko_str = None if pd.isna(murcko) else str(murcko).strip()
    if murcko_str:
        return murcko_str, "murcko"
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"ACY_INVALID::{smiles}", "acyclic_singleton"
    anon = rdMolHash.MolHash(mol, rdMolHash.HashFunction.AnonymousGraph)
    if anon:
        return f"ACY_{anon}", "acyclic_anonymous_graph"
    return f"ACY_SINGLE::{smiles}", "acyclic_singleton"


def choose_best_holdout(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "scaffold_group",
    seed: int = RANDOM_STATE,
    target_test_size: float = 0.20,
    n_splits: int = 5,
) -> pd.Series:
    y = df[label_col].astype(int).values
    groups = df[group_col].astype(str).values
    overall_pos = y.mean()
    target_test_n = len(df) * target_test_size
    target_test_pos = y.sum() * target_test_size
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_mask, best_score = None, None
    for _, test_idx in sgkf.split(df, y, groups=groups):
        mask = np.zeros(len(df), dtype=bool)
        mask[test_idx] = True
        test_n = int(mask.sum())
        test_pos = int(y[mask].sum())
        test_rate = test_pos / test_n if test_n else 0.0
        size_pen = abs(test_n - target_test_n) / max(len(df), 1)
        rate_pen = abs(test_rate - overall_pos)
        pos_pen = abs(test_pos - target_test_pos) / max(int(y.sum()), 1)
        score = size_pen + 2.0 * rate_pen + pos_pen
        if best_score is None or score < best_score:
            best_score, best_mask = score, mask
    return pd.Series(np.where(best_mask, "test", "train"), index=df.index, name="scaffold_split_80_20")


def compute_morgan_bits(smiles_series: pd.Series, n_bits: int = DEFAULT_FP_BITS, radius: int = 2) -> pd.DataFrame:
    arr = np.zeros((len(smiles_series), n_bits), dtype=np.uint8)
    mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits) if HAVE_MORGAN_GENERATOR else None
    for i, smi in enumerate(smiles_series.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mgen is not None:
            fp = mgen.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        row = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, row)
        arr[i, :] = row.astype(np.uint8)
    cols = [f"fp_morgan2_{j:03d}" for j in range(n_bits)]
    return pd.DataFrame(arr, columns=cols, index=smiles_series.index)


@lru_cache(maxsize=4)
def load_qm_module(base_dir_str: str):
    """Load local qm_descriptors.py as a module."""
    base_dir = Path(base_dir_str)
    candidates = [
        base_dir / 'qm_descriptors.py',
        base_dir / 'data' / 'qm_descriptors.py',
        Path(__file__).with_name('qm_descriptors.py'),
        Path('/mnt/data/qm_descriptors.py'),
    ]
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location('genotox_local_qm', cand)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    return None


def detect_default_external_qm_file(base_dir: Path) -> Optional[Path]:
    candidates = [
        base_dir / 'qm_descriptors.csv',
        base_dir / 'qm_external.csv',
        base_dir / 'data' / 'qm_descriptors.csv',
        base_dir / 'data' / 'qm_external.csv',
        base_dir / 'data' / 'external_qm.csv',
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def attach_required_qm_block(
    df: pd.DataFrame,
    base_dir: Path,
    output_dir: Path,
    qm_file: Optional[Path] = None,
    merge_key: str = 'No_norm',
    smiles_col: str = 'canonical_smiles',
    compute_if_missing: bool = False,
) -> pd.DataFrame:
    """Always compute QM-style descriptors from qm_descriptors.py, then optionally merge external/xtb QM."""
    out = df.copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    qm_mod = load_qm_module(str(base_dir))
    if qm_mod is None:
        raise FileNotFoundError(
            'qm_descriptors.py not found. Place it in base_dir or base_dir/data to enable required QM block.'
        )

    # 1) Always compute RDKit-based electronic/QM proxy descriptors
    out = qm_mod.add_electronic_descriptors(out, smiles_col=smiles_col, prefix='qm_')

    # 2) Optionally merge external QM CSV if provided/found
    ext_path = Path(qm_file) if qm_file else detect_default_external_qm_file(base_dir)
    ext_cols = []
    if ext_path is not None and ext_path.exists():
        qm_ext = read_csv_any(ext_path, string_cols=[merge_key, 'No', 'SMILES', 'canonical_smiles', 'smiles'])
        if merge_key not in qm_ext.columns:
            if merge_key == 'No_norm' and 'No' in qm_ext.columns:
                qm_ext['No_norm'] = normalize_no_series(qm_ext['No'])
            elif 'canonical_smiles' in qm_ext.columns and 'canonical_smiles' in out.columns:
                merge_key = 'canonical_smiles'
            elif 'SMILES' in qm_ext.columns and 'SMILES' in out.columns:
                merge_key = 'SMILES'
            elif 'smiles' in qm_ext.columns and 'SMILES' in out.columns:
                qm_ext['SMILES'] = qm_ext['smiles'].astype(str)
                merge_key = 'SMILES'
        exclude = {merge_key, 'No', 'No_norm', 'SMILES', 'smiles', 'canonical_smiles', 'label'}
        ext_cols = [c for c in qm_ext.columns if c not in exclude]
        if ext_cols:
            qm_sub = qm_ext[[merge_key] + ext_cols].drop_duplicates(subset=[merge_key]).copy()
            rename_map = {c: (c if c.startswith('ext_qm_') else f'ext_qm_{c}') for c in ext_cols}
            qm_sub = qm_sub.rename(columns=rename_map)
            out = out.merge(qm_sub, on=merge_key, how='left')
            ext_cols = list(rename_map.values())

    # 3) Optionally compute xTB cache for missing molecules (supplementary)
    xtb_cols = []
    if compute_if_missing:
        cache_path = output_dir / 'qm_xtb_cache.csv'
        cache = pd.DataFrame()
        if cache_path.exists():
            cache = read_csv_any(cache_path, string_cols=[merge_key])
        missing_keys = set(out[merge_key].dropna().astype(str)) - set(cache.get(merge_key, pd.Series(dtype=str)).astype(str))
        rows = []
        if missing_keys:
            work_base = output_dir / '_xtb_work'
            work_base.mkdir(parents=True, exist_ok=True)
            for key in missing_keys:
                row = out.loc[out[merge_key].astype(str).eq(str(key))].iloc[0]
                try:
                    vals = try_compute_qm_xtb(str(row[smiles_col]), work_base / str(key))
                except Exception:
                    vals = {}
                vals[merge_key] = key
                rows.append(vals)
        if rows:
            new_cache = pd.DataFrame(rows)
            cache = pd.concat([cache, new_cache], ignore_index=True).drop_duplicates(subset=[merge_key], keep='last')
            cache.to_csv(cache_path, index=False)
        if not cache.empty:
            xtb_cols = [c for c in cache.columns if c != merge_key]
            xtb_sub = cache[[merge_key] + xtb_cols].copy()
            rename_map = {c: (c if c.startswith('xtb_qm_') else f'xtb_qm_{c}') for c in xtb_cols}
            xtb_sub = xtb_sub.rename(columns=rename_map)
            out = out.merge(xtb_sub, on=merge_key, how='left')
            xtb_cols = list(rename_map.values())

    # 4) Save summary of qm block quality
    qm_cols_all = [c for c in out.columns if c.startswith('qm_') or c.startswith('ext_qm_') or c.startswith('xtb_qm_')]
    out.attrs['qm_cols'] = qm_cols_all
    try:
        qm_summary = qm_mod.qm_descriptor_summary(out, prefix='qm_')
        if not qm_summary.empty:
            qm_summary.to_csv(output_dir / 'qm_descriptor_summary.csv', index=False)
    except Exception:
        pass

    meta = pd.DataFrame({
        'qm_block': qm_cols_all,
        'source': ['rdkit_proxy' if c.startswith('qm_') else 'external_qm' if c.startswith('ext_qm_') else 'xtb_qm' for c in qm_cols_all]
    })
    if not meta.empty:
        meta.to_csv(output_dir / 'qm_feature_manifest.csv', index=False)
    return out


def try_compute_qm_xtb(smiles: str, workdir: Path, xtb_exe: str = "xtb") -> Dict[str, float]:
    xtb_path = shutil.which(xtb_exe)
    if xtb_path is None:
        raise FileNotFoundError("xtb executable not found")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=17) != 0:
        return {}
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    xyz_path = workdir / "mol.xyz"
    Chem.MolToXYZFile(mol, str(xyz_path))
    cmd = [xtb_path, str(xyz_path), "--gfn", "2", "--opt", "loose"]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        return {}
    out_text = res.stdout + "\n" + res.stderr
    vals = {}
    for line in out_text.splitlines():
        s = line.strip()
        if "HOMO-LUMO GAP" in s and ":" in s:
            try:
                vals["qm_homo_lumo_gap_ev"] = float(s.split()[-2])
            except Exception:
                pass
        if s.startswith("| TOTAL ENERGY"):
            try:
                vals["qm_total_energy"] = float(s.split()[-3])
            except Exception:
                pass
        if "molecular dipole:" in s.lower():
            # parsing in xtb is inconsistent; handled later if possible
            pass
    return vals


def merge_or_compute_qm(
    df: pd.DataFrame,
    output_dir: Path,
    qm_file: Optional[Path] = None,
    merge_key: str = 'No_norm',
    smiles_col: str = 'standardized_smiles',
    compute_if_missing: bool = False,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return attach_required_qm_block(
        df=df,
        base_dir=base,
        output_dir=output_dir,
        qm_file=qm_file,
        merge_key=merge_key,
        smiles_col=smiles_col,
        compute_if_missing=compute_if_missing,
    )



def _safe_stage_name(name: str) -> str:
    return str(name).lower().replace(" ", "_")

def write_stage_analysis(df: pd.DataFrame, outdir: Path, endpoint: str, stage_name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stage = _safe_stage_name(stage_name)
    profile = {
        "endpoint": endpoint,
        "stage": stage,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_missing_cells": int(df.isna().sum().sum()),
        "fraction_missing_cells": float(df.isna().sum().sum() / max(df.shape[0] * max(df.shape[1],1), 1)),
        "positive_rate": float(pd.to_numeric(df["label"], errors="coerce").mean()) if "label" in df.columns else np.nan,
        "n_unique_No": int(df["No"].astype(str).nunique()) if "No" in df.columns else np.nan,
        "n_unique_SMILES": int(df["SMILES"].astype(str).nunique()) if "SMILES" in df.columns else np.nan,
        "n_unique_canonical_smiles": int(df["canonical_smiles"].astype(str).nunique()) if "canonical_smiles" in df.columns else np.nan,
        "n_unique_standardized_smiles": int(df["standardized_smiles"].astype(str).nunique()) if "standardized_smiles" in df.columns else np.nan,
    }
    pd.DataFrame([profile]).to_csv(outdir / f"{stage}_profile_summary.csv", index=False)

    miss = pd.DataFrame({
        "column": df.columns,
        "missing_n": df.isna().sum().values,
        "missing_frac": (df.isna().mean().values),
        "dtype": [str(df[c].dtype) for c in df.columns],
    }).sort_values(["missing_frac", "missing_n"], ascending=False)
    miss.to_csv(outdir / f"{stage}_missingness.csv", index=False)

    label_df = None
    if "label" in df.columns:
        label_counts = pd.Series(pd.to_numeric(df["label"], errors="coerce")).fillna(-1).astype(int).value_counts().sort_index()
        label_df = pd.DataFrame({"label": label_counts.index, "count": label_counts.values})
        label_df.to_csv(outdir / f"{stage}_label_distribution.csv", index=False)
        fig = plt.figure(figsize=(4.5, 3.5))
        plt.bar(label_df["label"].astype(str), label_df["count"].values)
        plt.title(f"{endpoint} {stage} label distribution")
        plt.xlabel("label"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / f"{stage}_label_distribution.png", dpi=150)
        plt.close(fig)

    num_cols = []
    cat_cols = []
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s):
            cat_cols.append(c)
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().mean() >= 0.8:
                num_cols.append(c)
            else:
                cat_cols.append(c)

    if num_cols:
        num_summary = df[num_cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        num_summary.to_csv(outdir / f"{stage}_numeric_summary.csv")
        # top variance plot
        var = df[num_cols].apply(pd.to_numeric, errors="coerce").var(numeric_only=True).sort_values(ascending=False).head(20)
        if len(var) > 0:
            fig = plt.figure(figsize=(7, 5))
            plt.barh(var.index[::-1], var.values[::-1])
            plt.title(f"{endpoint} {stage} top numeric variance")
            plt.tight_layout()
            plt.savefig(outdir / f"{stage}_numeric_variance_top20.png", dpi=150)
            plt.close(fig)

    if cat_cols:
        rows = []
        for c in cat_cols[:200]:
            s = df[c].astype("string")
            rows.append({
                "column": c,
                "n_unique": int(s.nunique(dropna=True)),
                "top_value": str(s.mode(dropna=True).iloc[0]) if s.nunique(dropna=True) > 0 else "",
                "top_freq": int(s.value_counts(dropna=True).iloc[0]) if s.nunique(dropna=True) > 0 else 0,
            })
        pd.DataFrame(rows).sort_values(["n_unique", "top_freq"], ascending=[False, False]).to_csv(outdir / f"{stage}_categorical_summary.csv", index=False)

    # stage comparison friendly fingerprints/feature presence audit
    key_cols = [c for c in ["SMILES", "canonical_smiles", "standardized_smiles", "murcko_scaffold", "scaffold_group_type"] if c in df.columns]
    if key_cols:
        avail = []
        for c in key_cols:
            avail.append({"column": c, "non_null_frac": float(df[c].notna().mean())})
        avail_df = pd.DataFrame(avail)
        avail_df.to_csv(outdir / f"{stage}_key_column_availability.csv", index=False)
        fig = plt.figure(figsize=(6, 3.5))
        plt.bar(avail_df["column"], avail_df["non_null_frac"].values)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.title(f"{endpoint} {stage} key column availability")
        plt.tight_layout()
        plt.savefig(outdir / f"{stage}_key_column_availability.png", dpi=150)
        plt.close(fig)

def write_feature_analysis(full_df: pd.DataFrame, outdir: Path, endpoint: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    block_map = detect_feature_blocks(full_df)
    manifest_rows = [{"block": k, "n_features": len(v)} for k, v in block_map.items()]
    pd.DataFrame(manifest_rows).to_csv(outdir / "feature_block_manifest.csv", index=False)

    fig = plt.figure(figsize=(7, 4))
    mdf = pd.DataFrame(manifest_rows).sort_values("n_features", ascending=False)
    plt.bar(mdf["block"], mdf["n_features"].values)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{endpoint} feature block sizes")
    plt.tight_layout()
    plt.savefig(outdir / "feature_block_manifest.png", dpi=150)
    plt.close(fig)

    all_features = [c for cols in block_map.values() for c in cols]
    all_features = [c for c in dict.fromkeys(all_features) if c in full_df.columns]
    if not all_features:
        return

    y = pd.to_numeric(full_df["label"], errors="coerce").fillna(0).astype(int) if "label" in full_df.columns else pd.Series(np.zeros(len(full_df), dtype=int))
    missing = pd.DataFrame({
        "feature": all_features,
        "missing_frac": [float(full_df[c].isna().mean()) for c in all_features],
        "dtype": [str(full_df[c].dtype) for c in all_features],
    }).sort_values("missing_frac", ascending=False)
    missing.to_csv(outdir / "feature_missingness.csv", index=False)

    numeric_like = []
    binary_like = []
    for c in all_features:
        s = pd.to_numeric(full_df[c], errors="coerce")
        if s.notna().mean() >= 0.9:
            numeric_like.append(c)
            uniq = pd.Series(s.dropna().unique())
            if len(uniq) <= 3 and set(pd.Series(uniq).astype(float).round(6).tolist()).issubset({0.0, 1.0}):
                binary_like.append(c)

    if numeric_like:
        diffs = abs_mean_diff_score(full_df, y, numeric_like).head(50)
        pd.DataFrame({"feature": diffs.index, "abs_mean_diff": diffs.values}).to_csv(outdir / "top_numeric_association.csv", index=False)
        fig = plt.figure(figsize=(8, 7))
        top = diffs.iloc[:20][::-1]
        plt.barh(top.index, top.values)
        plt.title(f"{endpoint} top feature-label separation")
        plt.tight_layout()
        plt.savefig(outdir / "top_numeric_association.png", dpi=150)
        plt.close(fig)

    if binary_like:
        rows = []
        for c in binary_like:
            s = pd.to_numeric(full_df[c], errors="coerce").fillna(0)
            mask = s >= 1
            if mask.sum() == 0:
                continue
            pos_rate = y[mask].mean() if mask.sum() else np.nan
            lift = float(pos_rate / max(y.mean(), 1e-9))
            rows.append({"feature": c, "support": int(mask.sum()), "positive_rate_when_present": float(pos_rate), "lift_vs_baseline": lift})
        if rows:
            bdf = pd.DataFrame(rows).sort_values("lift_vs_baseline", ascending=False)
            bdf.to_csv(outdir / "binary_feature_lift.csv", index=False)
            fig = plt.figure(figsize=(8, 7))
            top = bdf.head(20).iloc[::-1]
            plt.barh(top["feature"], top["lift_vs_baseline"])
            plt.title(f"{endpoint} top binary feature lift")
            plt.tight_layout()
            plt.savefig(outdir / "binary_feature_lift.png", dpi=150)
            plt.close(fig)

def stage_output_paths(base_dir: Path, endpoint: str, output_root: Path) -> StagePaths:
    stage_dir = output_root / endpoint
    (stage_dir / "01_raw").mkdir(parents=True, exist_ok=True)
    (stage_dir / "02_standardized").mkdir(parents=True, exist_ok=True)
    (stage_dir / "03_preprocessed").mkdir(parents=True, exist_ok=True)
    (stage_dir / "04_split").mkdir(parents=True, exist_ok=True)
    (stage_dir / "05_features").mkdir(parents=True, exist_ok=True)
    (stage_dir / "06_sampling").mkdir(parents=True, exist_ok=True)
    return StagePaths(
        raw_base=stage_dir / "01_raw" / f"{endpoint}_raw_base.csv",
        standardized=stage_dir / "02_standardized" / f"{endpoint}_standardized_view.csv",
        preprocessed=stage_dir / "03_preprocessed" / f"{endpoint}_preprocessed_feature_ready.csv",
        split_assignments=stage_dir / "04_split" / f"{endpoint}_scaffold_split_assignments.csv",
        train_full=stage_dir / "05_features" / f"{endpoint}_train_full_features.csv",
        test_full=stage_dir / "05_features" / f"{endpoint}_test_full_features.csv",
        train_selected_raw=stage_dir / "06_sampling" / f"{endpoint}_train_selected_raw.csv",
        train_selected_full=stage_dir / "06_sampling" / f"{endpoint}_train_selected_full.csv",
    )


def prepare_endpoint_dataset(
    base_dir: Path,
    output_root: Path,
    endpoint: str,
    representation: Optional[str] = None,
    fp_bits: int = DEFAULT_FP_BITS,
    qm_file: Optional[Path] = None,
    compute_qm_if_missing: bool = False,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    rep = representation or ENDPOINTS[endpoint]["representation"]
    paths = stage_output_paths(base_dir, endpoint, output_root)
    if (not force_rebuild) and paths.train_full.exists() and paths.test_full.exists():
        return {
            "paths": paths,
            "train_df": read_csv_any(paths.train_full, string_cols=["No", "No_norm", "SMILES", "raw_smiles", "canonical_smiles", "standardized_smiles", "scaffold_group", "scaffold_group_type", "scaffold_split_80_20"]),
            "test_df": read_csv_any(paths.test_full, string_cols=["No", "No_norm", "SMILES", "raw_smiles", "canonical_smiles", "standardized_smiles", "scaffold_group", "scaffold_group_type", "scaffold_split_80_20"]),
        }

    raw_df = read_raw_endpoint(base_dir, endpoint)
    views, analysis = get_views_and_analysis(base_dir, endpoint, rep)

    raw_df.to_csv(paths.raw_base, index=False)
    write_stage_analysis(raw_df, output_root / endpoint / "01_raw", endpoint, "raw")

    views_use = views.drop_duplicates(subset=["No_norm"]).copy()
    if "label" in views_use.columns:
        standardized = raw_df.merge(
            views_use,
            on=["No_norm", "label"],
            how="left",
            suffixes=("", "_view"),
        )
    else:
        standardized = raw_df.merge(
            views_use,
            on=["No_norm"],
            how="left",
            suffixes=("", "_view"),
        )
    standardized.to_csv(paths.standardized, index=False)
    write_stage_analysis(standardized, output_root / endpoint / "02_standardized", endpoint, "standardized")

    analysis = analysis.drop(columns=[c for c in ["representation", "analysis_smiles", "valid", "metal_elements"] if c in analysis.columns], errors="ignore")
    preprocessed = standardized.merge(analysis.drop_duplicates(subset=["No_norm"]), on=["No_norm", "label"], how="left", suffixes=("", "_ana"))
    preprocessed = add_rule_features(preprocessed, endpoint)

    split_smiles_col = "canonical_smiles" if rep == "canonical" else "standardized_smiles"
    groups = [make_scaffold_group(s, m) for s, m in zip(preprocessed[split_smiles_col].fillna(preprocessed["SMILES"]).astype(str), preprocessed["murcko_scaffold"])]
    preprocessed["scaffold_group"] = [g for g, _ in groups]
    preprocessed["scaffold_group_type"] = [t for _, t in groups]
    preprocessed["scaffold_split_80_20"] = choose_best_holdout(preprocessed, label_col="label", group_col="scaffold_group", seed=RANDOM_STATE)
    preprocessed.to_csv(paths.preprocessed, index=False)
    write_stage_analysis(preprocessed, output_root / endpoint / "03_preprocessed", endpoint, "preprocessed")

    preprocessed[["No", "No_norm", "label", "SMILES", "canonical_smiles", "standardized_smiles", "scaffold_group", "scaffold_group_type", "scaffold_split_80_20"]].to_csv(paths.split_assignments, index=False)

    fp_df = compute_morgan_bits(preprocessed[split_smiles_col].fillna(preprocessed["SMILES"]), n_bits=fp_bits, radius=2)
    full_df = pd.concat([preprocessed.reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)
    full_df = merge_or_compute_qm(full_df, output_root / endpoint / "05_features", qm_file=qm_file, merge_key="No_norm", smiles_col=split_smiles_col, compute_if_missing=compute_qm_if_missing, base_dir=base_dir)
    write_feature_analysis(full_df, output_root / endpoint / "05_features", endpoint)

    train_df = full_df.loc[full_df["scaffold_split_80_20"].eq("train")].copy()
    test_df = full_df.loc[full_df["scaffold_split_80_20"].eq("test")].copy()
    train_df.to_csv(paths.train_full, index=False)
    test_df.to_csv(paths.test_full, index=False)
    return {"paths": paths, "train_df": train_df, "test_df": test_df}


def detect_feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    fp_cols = [c for c in df.columns if c.startswith("fp_morgan2_")]
    qm_cols = [c for c in df.columns if c.startswith("qm_") or c.startswith("ext_qm_") or c.startswith("xtb_qm_")]
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    fg_present = [c for c in df.columns if c.startswith("fg_") and c.endswith("_present")]
    fg_count = [c for c in df.columns if c.startswith("fg_") and c.endswith("_count")]
    physchem = [c for c in PHYS_CHEM_COLS if c in df.columns]
    scaffold_meta = [c for c in ["scaffold_group_type"] if c in df.columns]
    expert = [c for c in ["expert_positive_alert_score", "expert_negative_alert_score"] if c in df.columns]
    return {
        "physchem": physchem,
        "rules": rule_cols,
        "fg_present": fg_present,
        "fg_count": fg_count,
        "fingerprint": fp_cols,
        "qm": qm_cols,
        "scaffold_meta": scaffold_meta,
        "expert": expert,
    }


def compute_alert_scores(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    out = df.copy()
    pos = np.zeros(len(out), dtype=float)
    for col, w in EXPERT_POSITIVE.get(endpoint, {}).items():
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)
            pos += w * s.to_numpy()
    neg = np.zeros(len(out), dtype=float)
    for col, w in EXPERT_NEGATIVE.get(endpoint, {}).items():
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)
            neg += w * s.to_numpy()
    out["expert_positive_alert_score"] = pos
    out["expert_negative_alert_score"] = neg
    return out


def is_internal_col(col: str) -> bool:
    if col.startswith("__"):
        return True
    return col in {
        "label", "No", "No_norm", "SMILES", "raw_smiles", "canonical_smiles", "standardized_smiles",
        "raw_smiles_input", "scaffold_group", "scaffold_split_80_20", "murcko_scaffold",
        "domain", "status", "raw_name",
    }


def abs_mean_diff_score(X: pd.DataFrame, y: pd.Series, cols: List[str]) -> pd.Series:
    pos = X.loc[y.eq(1), cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    neg = X.loc[y.eq(0), cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if len(pos) == 0 or len(neg) == 0:
        return pd.Series(0.0, index=cols)
    return (pos.mean(axis=0) - neg.mean(axis=0)).abs().sort_values(ascending=False)


def select_feature_cols(
    train_df: pd.DataFrame,
    blocks: Sequence[str],
    fp_top_k: int = 128,
    enforce_existing: Optional[Sequence[str]] = None,
) -> List[str]:
    train_df = train_df.copy()
    block_map = detect_feature_blocks(train_df)
    cols: List[str] = []
    for block in blocks:
        cols.extend(block_map.get(block, []))
    cols = [c for c in cols if (c in train_df.columns) and (not is_internal_col(c))]
    fp_cols = [c for c in cols if c.startswith("fp_morgan2_")]
    other_cols = [c for c in cols if c not in fp_cols]
    if fp_cols:
        y = train_df["label"].astype(int)
        score = abs_mean_diff_score(train_df, y, fp_cols)
        fp_keep = score.index.tolist()[: min(fp_top_k, len(score))]
    else:
        fp_keep = []
    out = other_cols + fp_keep
    if enforce_existing is not None:
        out = [c for c in out if c in enforce_existing]
    seen = set()
    final = []
    for c in out:
        if c not in seen:
            final.append(c)
            seen.add(c)
    return final


def build_preprocessor(X: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    X = X[feature_cols].copy()
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in feature_cols:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s):
            cat_cols.append(c)
            continue
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().mean() >= 0.9:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    num_pipe = Pipeline([
        ("to_numeric", FunctionTransformer(coerce_numeric_df, validate=False, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("No usable feature columns found")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        if not (ptypes.is_object_dtype(out[c]) or ptypes.is_string_dtype(out[c]) or isinstance(out[c].dtype, pd.CategoricalDtype)):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def make_model(model_name: str, endpoint: str, params: Dict, y_train: pd.Series, seed: int):
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    if model_name == "logreg":
        return LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            class_weight="balanced",
            max_iter=4000,
            C=float(params.get("C", 1.0)),
            l1_ratio=float(params.get("l1_ratio", 0.2)),
            random_state=seed,
        )
    if model_name == "xgb":
        if not HAVE_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 3)),
            reg_lambda=float(params.get("reg_lambda", 2.0)),
            reg_alpha=float(params.get("reg_alpha", 0.2)),
            gamma=float(params.get("gamma", 0.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            scale_pos_weight=float(neg / pos),
        )
    raise ValueError(model_name)


def sampling_ratio_for_endpoint(endpoint: str) -> float:
    if endpoint == "ames":
        return 1.0
    if endpoint == "invitro":
        return 1.5
    return 2.0


def select_training_panel(
    train_df: pd.DataFrame,
    endpoint: str,
    strategy: str,
    seed: int,
    target_neg_pos_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = train_df.copy()
    if "expert_positive_alert_score" not in df.columns or "expert_negative_alert_score" not in df.columns:
        df = compute_alert_scores(df, endpoint)
    pos_df = df.loc[df["label"].astype(int).eq(1)].copy()
    neg_df = df.loc[df["label"].astype(int).eq(0)].copy()
    if strategy == "none" or len(neg_df) == 0 or len(pos_df) == 0:
        manifest = pd.DataFrame({
            "strategy": [strategy],
            "n_pos": [len(pos_df)],
            "n_neg_before": [len(neg_df)],
            "n_neg_after": [len(neg_df)],
            "target_ratio": [target_neg_pos_ratio if target_neg_pos_ratio is not None else np.nan],
        })
        return df, manifest
    ratio = target_neg_pos_ratio if target_neg_pos_ratio is not None else sampling_ratio_for_endpoint(endpoint)
    target_neg = min(len(neg_df), int(math.ceil(len(pos_df) * ratio)))
    if target_neg >= len(neg_df):
        manifest = pd.DataFrame({
            "strategy": [strategy],
            "n_pos": [len(pos_df)],
            "n_neg_before": [len(neg_df)],
            "n_neg_after": [len(neg_df)],
            "target_ratio": [ratio],
        })
        return df, manifest

    sel_cols = [c for c in [
        "mw", "logp", "tpsa", "rot_bonds", "aromatic_ring_count", "fraction_csp3",
        "bb_n_genotox_alerts", "expert_positive_alert_score", "expert_negative_alert_score"
    ] if c in neg_df.columns]
    for c in ["fg_epoxide_present", "fg_BB_SA5_nitro_aromatic_present", "fg_BB_SA10_N_nitroso_present",
              "fg_BB_SA2_primary_aromatic_amine_present", "fg_BB_SA20_ab_unsat_carbonyl_present",
              "fg_organometal_like_present", "fg_alcohol_present"]:
        if c in neg_df.columns and c not in sel_cols:
            sel_cols.append(c)

    work = neg_df[sel_cols].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    rng = np.random.default_rng(seed)

    coverage_frac = 0.6 if endpoint != "invivo" else 0.5
    n_coverage = max(1, int(round(target_neg * coverage_frac)))
    n_hard = max(0, target_neg - n_coverage)

    cover_idx: List[int] = []
    if len(work) <= n_coverage:
        cover_idx = neg_df.index.tolist()
    else:
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans
            X = StandardScaler().fit_transform(work.to_numpy())
            n_clusters = min(n_coverage, max(2, int(np.sqrt(len(work)))))
            if os.name == "nt" and len(work) < 12000:
                km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            else:
                km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10, batch_size=max(4096, min(len(work), 8192)))
            labels = km.fit_predict(X)
            centers = km.cluster_centers_
            chosen = []
            for k in range(n_clusters):
                idx = np.where(labels == k)[0]
                if len(idx) == 0:
                    continue
                d = ((X[idx] - centers[k]) ** 2).sum(axis=1)
                chosen.append(idx[int(np.argmin(d))])
            cover_idx = neg_df.index[np.array(chosen)].tolist()
        except Exception:
            cover_idx = neg_df.sample(n=n_coverage, random_state=seed).index.tolist()

    hard_idx: List[int] = []
    if n_hard > 0:
        temp = neg_df.drop(index=cover_idx, errors="ignore").copy()
        if len(temp) > 0:
            score = temp["expert_positive_alert_score"].astype(float) - 0.5 * temp["expert_negative_alert_score"].astype(float)
            temp = temp.assign(_hard_score=score)
            temp = temp.sort_values(["_hard_score", "mw"], ascending=[False, True])
            hard_idx = temp.head(n_hard).index.tolist()

    selected_neg_idx = list(dict.fromkeys(cover_idx + hard_idx))
    if len(selected_neg_idx) < target_neg:
        remain = neg_df.drop(index=selected_neg_idx, errors="ignore")
        need = min(target_neg - len(selected_neg_idx), len(remain))
        if need > 0:
            extra = remain.sample(n=need, random_state=seed).index.tolist()
            selected_neg_idx.extend(extra)
    selected_neg_df = neg_df.loc[selected_neg_idx].copy()
    selected_df = pd.concat([pos_df, selected_neg_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    manifest = pd.DataFrame({
        "strategy": [strategy],
        "n_pos": [len(pos_df)],
        "n_neg_before": [len(neg_df)],
        "n_neg_after": [len(selected_neg_df)],
        "target_ratio": [ratio],
        "coverage_frac": [coverage_frac],
        "n_coverage_selected": [len(cover_idx)],
        "n_hard_selected": [len(hard_idx)],
    })
    return selected_df, manifest


def metric_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float, metric_name: str) -> float:
    pred = (probs >= threshold).astype(int)
    if metric_name == "mcc":
        return matthews_corrcoef(y_true, pred)
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, pred)
    if metric_name == "average_precision":
        return average_precision_score(y_true, probs)
    raise ValueError(metric_name)


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, metric_name: str, specificity_floor: float = 0.0) -> Tuple[float, pd.DataFrame]:
    rows = []
    best_thr, best_score = 0.5, -1e18
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if metric_name == "mcc":
            score = matthews_corrcoef(y_true, pred)
        else:
            score = balanced_accuracy_score(y_true, pred)
        rows.append({"threshold": thr, "score": score, "sensitivity": sens, "specificity": spec, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        if spec >= specificity_floor and score > best_score:
            best_score, best_thr = score, thr
    if best_score <= -1e17:
        rows_df = pd.DataFrame(rows)
        best_idx = rows_df["score"].idxmax()
        best_thr = float(rows_df.loc[best_idx, "threshold"])
    return float(best_thr), pd.DataFrame(rows)


def seed_children(base_seed: int, *labels: str, n: int = 1) -> List[int]:
    ss = np.random.SeedSequence([base_seed] + [abs(hash(x)) % (2**31-1) for x in labels])
    return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(n)]


def candidate_cv_score(
    train_df: pd.DataFrame,
    endpoint: str,
    blocks: List[str],
    fp_top_k: int,
    model_name: str,
    params: Dict,
    sampling_strategy: str,
    threshold_metric: str,
    specificity_floor: float,
    cv_splits: int,
    base_seed: int,
) -> float:
    groups = train_df["scaffold_group"].astype(str).values
    y_all = train_df["label"].astype(int).values
    sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=base_seed)
    oof = np.zeros(len(train_df), dtype=float)
    for fold, (tr, va) in enumerate(sgkf.split(train_df, y_all, groups=groups)):
        fold_seed = seed_children(base_seed, "fold", str(fold), n=1)[0]
        fold_train = train_df.iloc[tr].copy()
        fold_valid = train_df.iloc[va].copy()
        sampled_train, _ = select_training_panel(fold_train, endpoint, sampling_strategy, fold_seed)
        sampled_train = compute_alert_scores(sampled_train, endpoint)
        fold_valid = compute_alert_scores(fold_valid, endpoint)
        feature_cols = select_feature_cols(sampled_train, blocks, fp_top_k=fp_top_k)
        pre = build_preprocessor(sampled_train, feature_cols)
        Xtr = sampled_train[feature_cols]
        ytr = sampled_train["label"].astype(int)
        Xva = fold_valid[feature_cols]
        model = make_model(model_name, endpoint, params, ytr, fold_seed)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        oof[va] = pipe.predict_proba(Xva)[:, 1]
    thr, _ = tune_threshold(y_all, oof, threshold_metric, specificity_floor=specificity_floor)
    return metric_from_probs(y_all, oof, thr, threshold_metric)


def tune_params(
    train_df: pd.DataFrame,
    endpoint: str,
    blocks: List[str],
    fp_top_k: int,
    model_name: str,
    sampling_strategy: str,
    threshold_metric: str,
    specificity_floor: float,
    base_seed: int,
    cv_splits: int = 3,
    n_iter: int = 12,
    search_method: str = "grid",
) -> Tuple[Dict, pd.DataFrame]:
    if search_method == "grid":
        grid = GRID_CONFIG[endpoint]["logreg_grid" if model_name == "logreg" else "xgb_grid"]
        candidates = list(ParameterGrid(grid))
    else:
        dist = MODEL_CONFIG[endpoint]["logreg_distributions" if model_name == "logreg" else "xgb_distributions"]
        candidates = list(ParameterSampler(dist, n_iter=n_iter, random_state=base_seed))
    rows = []
    best_score, best_params = -1e18, None
    for i, cand in enumerate(candidates):
        score = candidate_cv_score(
            train_df, endpoint, blocks, fp_top_k, model_name, cand,
            sampling_strategy, threshold_metric, specificity_floor,
            cv_splits=cv_splits, base_seed=seed_children(base_seed, "cand", str(i), n=1)[0]
        )
        row = dict(cand)
        row["cv_score"] = score
        rows.append(row)
        if score > best_score:
            best_score, best_params = score, cand
    return best_params or {}, pd.DataFrame(rows).sort_values("cv_score", ascending=False)


def fit_and_threshold(
    train_df: pd.DataFrame,
    endpoint: str,
    blocks: List[str],
    fp_top_k: int,
    model_name: str,
    params: Dict,
    sampling_strategy: str,
    threshold_metric: str,
    specificity_floor: float,
    base_seed: int,
    cv_splits: int = 3,
) -> Tuple[Pipeline, List[str], float, pd.DataFrame]:
    groups = train_df["scaffold_group"].astype(str).values
    y_all = train_df["label"].astype(int).values
    sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=base_seed)
    oof = np.zeros(len(train_df), dtype=float)
    for fold, (tr, va) in enumerate(sgkf.split(train_df, y_all, groups=groups)):
        fold_seed = seed_children(base_seed, "thr", str(fold), n=1)[0]
        fold_train = train_df.iloc[tr].copy()
        fold_valid = train_df.iloc[va].copy()
        sampled_train, _ = select_training_panel(fold_train, endpoint, sampling_strategy, fold_seed)
        sampled_train = compute_alert_scores(sampled_train, endpoint)
        fold_valid = compute_alert_scores(fold_valid, endpoint)
        feature_cols = select_feature_cols(sampled_train, blocks, fp_top_k=fp_top_k)
        pre = build_preprocessor(sampled_train, feature_cols)
        model = make_model(model_name, endpoint, params, sampled_train["label"].astype(int), fold_seed)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(sampled_train[feature_cols], sampled_train["label"].astype(int))
        oof[va] = pipe.predict_proba(fold_valid[feature_cols])[:,1]
    thr, thr_df = tune_threshold(y_all, oof, threshold_metric, specificity_floor=specificity_floor)

    final_seed = seed_children(base_seed, "final", n=1)[0]
    sampled_train, _ = select_training_panel(train_df, endpoint, sampling_strategy, final_seed)
    sampled_train = compute_alert_scores(sampled_train, endpoint)
    feature_cols = select_feature_cols(sampled_train, blocks, fp_top_k=fp_top_k)
    pre = build_preprocessor(sampled_train, feature_cols)
    model = make_model(model_name, endpoint, params, sampled_train["label"].astype(int), final_seed)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(sampled_train[feature_cols], sampled_train["label"].astype(int))
    return pipe, feature_cols, thr, thr_df


def extract_importance(pipe: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    try:
        fn = pre.get_feature_names_out()
    except Exception:
        fn = np.array(feature_cols)
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": fn, "importance": model.feature_importances_})
    elif hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_).reshape(-1)
        imp = pd.DataFrame({"feature": fn, "importance": np.abs(coefs)})
    else:
        imp = pd.DataFrame({"feature": fn, "importance": np.nan})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def save_curves_and_confusion(run_dir: Path, y_true: np.ndarray, probs: np.ndarray, threshold: float, thr_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    pd.DataFrame([{"tn": tn, "fp": fp, "fn": fn, "tp": tp}]).to_csv(run_dir / "confusion_matrix_counts.csv", index=False)

    fig = plt.figure(figsize=(4,4))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, interpolation="nearest")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xticks([0,1], ["pred 0", "pred 1"])
    plt.yticks([0,1], ["true 0", "true 1"])
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(run_dir / "roc_curve.csv", index=False)
    fig = plt.figure(figsize=(4.5,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(run_dir / "roc_curve.png", dpi=150); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_df.to_csv(run_dir / "pr_curve.csv", index=False)
    fig = plt.figure(figsize=(4.5,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
    plt.tight_layout(); plt.savefig(run_dir / "pr_curve.png", dpi=150); plt.close(fig)

    thr_df.to_csv(run_dir / "threshold_sweep.csv", index=False)
    fig = plt.figure(figsize=(5,4))
    plt.plot(thr_df["threshold"], thr_df["score"])
    plt.axvline(threshold, linestyle="--")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold sweep")
    plt.tight_layout(); plt.savefig(run_dir / "threshold_sweep.png", dpi=150); plt.close(fig)

    imp_top = importance_df.head(20).iloc[::-1]
    imp_top.to_csv(run_dir / "feature_importance.csv", index=False)
    fig = plt.figure(figsize=(7,6))
    plt.barh(imp_top["feature"], imp_top["importance"])
    plt.title("Top feature importance")
    plt.tight_layout(); plt.savefig(run_dir / "feature_importance_top.png", dpi=150); plt.close(fig)


def maybe_save_shap(run_dir: Path, pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, max_background: int = 200, max_eval: int = 200) -> None:
    if not HAVE_SHAP:
        return
    model = pipe.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        return
    bg = X_train.sample(n=min(max_background, len(X_train)), random_state=17) if len(X_train) > max_background else X_train
    ev = X_test.sample(n=min(max_eval, len(X_test)), random_state=19) if len(X_test) > max_eval else X_test
    try:
        if HAVE_XGB and isinstance(model, XGBClassifier):
            Xt_bg = pipe.named_steps["pre"].transform(bg)
            Xt_ev = pipe.named_steps["pre"].transform(ev)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xt_ev)
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
        else:
            explainer = shap.Explainer(pipe, bg)
            shap_values = explainer(ev)
            feature_names = ev.columns
        fig = plt.figure()
        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (run_dir / "shap_error.txt").write_text(str(e), encoding="utf-8")


def evaluate_on_test(pipe: Pipeline, feature_cols: List[str], test_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    Xte = test_df[feature_cols]
    yte = test_df["label"].astype(int).to_numpy()
    probs = pipe.predict_proba(Xte)[:,1]
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0,1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    metrics = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "mcc": float(matthews_corrcoef(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, probs)) if len(np.unique(yte)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(yte, probs)) if len(np.unique(yte)) > 1 else np.nan,
        "threshold": float(threshold),
    }
    return metrics, probs


def save_prediction_tables(run_dir: Path, test_df: pd.DataFrame, probs: np.ndarray, threshold: float) -> None:
    out = test_df[["No", "No_norm", "SMILES", "label", "scaffold_group", "scaffold_group_type"]].copy()
    out["pred_proba"] = probs
    out["pred_label"] = (probs >= threshold).astype(int)
    out["is_error"] = (out["pred_label"] != out["label"].astype(int)).astype(int)
    out.to_csv(run_dir / "test_predictions.csv", index=False)
    out.loc[out["is_error"].eq(1)].to_csv(run_dir / "test_misclassifications.csv", index=False)


def run_model_experiment(
    base_dir: Path,
    output_root: Path,
    endpoint: str,
    representation: str,
    feature_blocks: List[str],
    sampling_strategy: str,
    model_name: str,
    qm_file: Optional[Path] = None,
    compute_qm_if_missing: bool = False,
    fp_bits: int = DEFAULT_FP_BITS,
    fp_top_k: int = 128,
    tune_iter: int = 12,
    search_method: str = "grid",
    cv_splits: int = 3,
    save_shap: bool = False,
    force_rebuild_features: bool = False,
    base_seed: int = RANDOM_STATE,
) -> RunArtifacts:
    prep = prepare_endpoint_dataset(
        base_dir, output_root, endpoint,
        representation=representation, fp_bits=fp_bits,
        qm_file=qm_file, compute_qm_if_missing=compute_qm_if_missing,
        force_rebuild=force_rebuild_features,
    )
    train_df = compute_alert_scores(prep["train_df"], endpoint)
    test_df = compute_alert_scores(prep["test_df"], endpoint)
    paths: StagePaths = prep["paths"]

    selected_train_df, selection_manifest = select_training_panel(train_df, endpoint, sampling_strategy, seed_children(base_seed, endpoint, "outer_select", n=1)[0])
    train_raw_cols = [c for c in ["No", "No_norm", "SMILES", "label", "scaffold_group", "scaffold_group_type", "canonical_smiles", "standardized_smiles"] if c in selected_train_df.columns]
    selected_train_df[train_raw_cols].to_csv(paths.train_selected_raw, index=False)
    selected_train_df.to_csv(paths.train_selected_full, index=False)
    selection_manifest.to_csv(paths.train_selected_full.with_name(paths.train_selected_full.stem + "_selection_manifest.csv"), index=False)

    threshold_metric = ENDPOINTS[endpoint]["threshold_metric"]
    specificity_floor = ENDPOINTS[endpoint]["specificity_floor"]

    tune_seed = seed_children(base_seed, endpoint, model_name, "tune", n=1)[0]
    params, tuning_df = tune_params(
        train_df=train_df,
        endpoint=endpoint,
        blocks=feature_blocks,
        fp_top_k=fp_top_k,
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        threshold_metric=threshold_metric,
        specificity_floor=specificity_floor,
        base_seed=tune_seed,
        cv_splits=cv_splits,
        n_iter=tune_iter,
        search_method=search_method,
    )

    fit_seed = seed_children(base_seed, endpoint, model_name, "fit", n=1)[0]
    pipe, feature_cols, threshold, thr_df = fit_and_threshold(
        train_df=train_df,
        endpoint=endpoint,
        blocks=feature_blocks,
        fp_top_k=fp_top_k,
        model_name=model_name,
        params=params,
        sampling_strategy=sampling_strategy,
        threshold_metric=threshold_metric,
        specificity_floor=specificity_floor,
        base_seed=fit_seed,
        cv_splits=cv_splits,
    )

    run_name = f"{endpoint}__{representation}__{sampling_strategy}__{model_name}"
    run_dir = output_root / endpoint / "07_model_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(run_dir / "tuning_results.csv", index=False)

    metrics, probs = evaluate_on_test(pipe, feature_cols, test_df, threshold)
    metrics.update({
        "endpoint": endpoint,
        "representation": representation,
        "sampling_strategy": sampling_strategy,
        "model_name": model_name,
        "feature_blocks": ",".join(feature_blocks),
        "n_features": len(feature_cols),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "selected_train_n": int(len(selected_train_df)),
        "selected_train_pos_rate": float(selected_train_df["label"].mean()),
        "test_pos_rate": float(test_df["label"].mean()),
        "search_method": search_method,
    })
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_tables(run_dir, test_df, probs, threshold)
    importance_df = extract_importance(pipe, feature_cols)
    save_curves_and_confusion(run_dir, test_df["label"].astype(int).to_numpy(), probs, threshold, thr_df, importance_df)

    bundle = {
        "pipe": pipe,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "params": params,
        "X_train_sample": train_df[feature_cols].sample(n=min(300, len(train_df)), random_state=11),
        "X_test_sample": test_df[feature_cols].sample(n=min(300, len(test_df)), random_state=13),
        "feature_blocks": feature_blocks,
    }
    joblib.dump(bundle, run_dir / "trained_bundle.joblib")
    if save_shap:
        maybe_save_shap(run_dir, pipe, train_df[feature_cols], test_df[feature_cols])

    return RunArtifacts(
        endpoint=endpoint,
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        representation=representation,
        feature_blocks=feature_blocks,
        metrics=metrics,
        run_dir=str(run_dir),
        train_n=len(train_df),
        test_n=len(test_df),
        selected_train_n=len(selected_train_df),
    )



def build_stage_overview(output_root: Path, endpoints: List[str]) -> None:
    outdir = output_root / "_summary"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in endpoints:
        for stage, rel in [("raw", "01_raw"), ("standardized", "02_standardized"), ("preprocessed", "03_preprocessed")]:
            f = output_root / ep / rel / f"{stage}_profile_summary.csv"
            if f.exists():
                df = pd.read_csv(f)
                rows.append(df)
    if not rows:
        return
    comp = pd.concat(rows, ignore_index=True)
    comp.to_csv(outdir / "stage_profile_comparison.csv", index=False)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    for ep in endpoints:
        sub = comp.loc[comp["endpoint"].eq(ep)].copy()
        if len(sub) == 0:
            continue
        ax1.plot(sub["stage"], sub["n_rows"], marker="o", label=ep)
    ax1.set_title("Rows by stage"); ax1.set_ylabel("n_rows"); ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    for ep in endpoints:
        sub = comp.loc[comp["endpoint"].eq(ep)].copy()
        if len(sub) == 0:
            continue
        ax2.plot(sub["stage"], sub["positive_rate"], marker="o", label=ep)
    ax2.set_title("Positive rate by stage"); ax2.set_ylabel("positive_rate"); ax2.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage_profile_comparison.png", dpi=150)
    plt.close(fig)

def build_dashboard(summary_df: pd.DataFrame, outdir: Path) -> None:
    if summary_df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    best = summary_df.sort_values(["endpoint", "mcc"], ascending=[True, False]).groupby("endpoint", as_index=False).head(1)
    best.to_csv(outdir / "best_by_endpoint.csv", index=False)

    fig = plt.figure(figsize=(9, 4.5))
    for i, metric in enumerate(["mcc", "balanced_accuracy"]):
        ax = fig.add_subplot(1, 2, i + 1)
        plot_df = summary_df.copy()
        labels = plot_df["endpoint"] + "\n" + plot_df["model_name"] + "\n" + plot_df["sampling_strategy"]
        ax.bar(range(len(plot_df)), plot_df[metric].values)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(outdir / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_full_pipeline(
    base_dir: Path,
    output_root: Path,
    endpoints: Optional[List[str]] = None,
    representation_map: Optional[Dict[str, str]] = None,
    feature_blocks_map: Optional[Dict[str, List[str]]] = None,
    sampling_map: Optional[Dict[str, str]] = None,
    model_map: Optional[Dict[str, str]] = None,
    qm_file: Optional[Path] = None,
    compute_qm_if_missing: bool = False,
    fp_bits: int = DEFAULT_FP_BITS,
    fp_top_k_map: Optional[Dict[str, int]] = None,
    tune_iter_map: Optional[Dict[str, int]] = None,
    search_method: str = "grid",
    cv_splits: int = 3,
    save_shap_best_only: bool = True,
    force_rebuild_features: bool = True,
    base_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    endpoints = endpoints or list(ENDPOINTS.keys())
    representation_map = representation_map or {ep: ENDPOINTS[ep]["representation"] for ep in endpoints}
    feature_blocks_map = feature_blocks_map or {
        "ames": ["physchem", "rules", "fg_present", "fg_count", "fingerprint", "qm", "expert", "scaffold_meta"],
        "invitro": ["physchem", "rules", "fg_present", "fg_count", "fingerprint", "qm", "expert", "scaffold_meta"],
        "invivo": ["physchem", "rules", "fg_present", "fg_count", "fingerprint", "qm", "expert", "scaffold_meta"],
    }
    sampling_map = sampling_map or {ep: ENDPOINTS[ep]["sampling"] for ep in endpoints}
    model_map = model_map or {ep: ENDPOINTS[ep]["model"] for ep in endpoints}
    fp_top_k_map = fp_top_k_map or {"ames": 128, "invitro": 96, "invivo": 64}
    tune_iter_map = tune_iter_map or {"ames": 16, "invitro": 12, "invivo": 10}

    artifacts: List[RunArtifacts] = []
    for ep in endpoints:
        art = run_model_experiment(
            base_dir=base_dir,
            output_root=output_root,
            endpoint=ep,
            representation=representation_map.get(ep, "canonical"),
            feature_blocks=feature_blocks_map[ep],
            sampling_strategy=sampling_map[ep],
            model_name=model_map[ep],
            qm_file=qm_file,
            compute_qm_if_missing=compute_qm_if_missing,
            fp_bits=fp_bits,
            fp_top_k=fp_top_k_map.get(ep, 128),
            tune_iter=tune_iter_map.get(ep, 12),
            search_method=search_method,
            cv_splits=cv_splits,
            save_shap=False,
            force_rebuild_features=force_rebuild_features,
            base_seed=seed_children(base_seed, ep, n=1)[0],
        )
        artifacts.append(art)

    summary = pd.DataFrame([a.metrics for a in artifacts])
    dashboard_dir = output_root / "_summary"
    build_dashboard(summary, dashboard_dir)
    build_stage_overview(output_root, endpoints)

    if save_shap_best_only and not summary.empty:
        best_rows = summary.sort_values(["endpoint", "mcc"], ascending=[True, False]).groupby("endpoint", as_index=False).head(1)
        for _, row in best_rows.iterrows():
            run_dir = Path(row["run_dir"]) if "run_dir" in row else Path()
        for art in artifacts:
            if any((summary["endpoint"].eq(art.endpoint) & summary["mcc"].eq(art.metrics["mcc"]))):
                try:
                    bundle = joblib.load(Path(art.run_dir) / "trained_bundle.joblib")
                    maybe_save_shap(Path(art.run_dir), bundle["pipe"], bundle["X_train_sample"], bundle["X_test_sample"])
                except Exception as e:
                    (Path(art.run_dir) / "shap_error.txt").write_text(str(e), encoding="utf-8")
    return summary


def default_base_dir() -> Path:
    candidates = [
        Path(r"C:\Users\Administrator\Documents\GitHub\Genotox_model\model"),
        Path("/mnt/data"),
        Path.cwd(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd()


def make_notebook_preset_json(base_dir: Path, output_root: Path) -> Dict:
    return {
        "base_dir": str(base_dir),
        "output_root": str(output_root),
        "endpoints": list(ENDPOINTS.keys()),
        "representation_map": {ep: ENDPOINTS[ep]["representation"] for ep in ENDPOINTS},
        "sampling_map": {ep: ENDPOINTS[ep]["sampling"] for ep in ENDPOINTS},
        "model_map": {ep: ENDPOINTS[ep]["model"] for ep in ENDPOINTS},
        "search_method": "grid",
    }


def main():
    p = argparse.ArgumentParser(description="Unified genotoxicity pipeline from raw data to model artifacts")
    p.add_argument("--base-dir", type=str, default=str(default_base_dir()))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--endpoints", nargs="*", default=["ames", "invitro", "invivo"])
    p.add_argument("--qm-file", type=str, default=None)
    p.add_argument("--compute-qm-if-missing", action="store_true")
    p.add_argument("--fp-bits", type=int, default=DEFAULT_FP_BITS)
    p.add_argument("--cv-splits", type=int, default=3)
    p.add_argument("--search-method", type=str, default="grid", choices=["grid","randomized"])
    p.add_argument("--force-rebuild-features", action="store_true")
    p.add_argument("--save-shap-best-only", action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    output_root = Path(args.output_dir) if args.output_dir else base_dir / "runs" / "unified_full_pipeline"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_full_pipeline(
        base_dir=base_dir,
        output_root=output_root,
        endpoints=args.endpoints,
        qm_file=Path(args.qm_file) if args.qm_file else None,
        compute_qm_if_missing=args.compute_qm_if_missing,
        fp_bits=args.fp_bits,
        search_method=args.search_method,
        cv_splits=args.cv_splits,
        save_shap_best_only=args.save_shap_best_only,
        force_rebuild_features=args.force_rebuild_features,
        base_seed=args.seed,
    )
    summary.to_csv(output_root / "_summary" / "summary_metrics.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
