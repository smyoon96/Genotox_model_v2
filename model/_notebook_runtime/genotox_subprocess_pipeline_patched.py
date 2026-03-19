
from __future__ import annotations

import argparse
import gc
import io
import itertools
import json
import math
import traceback
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zlib

import joblib
import numpy as np
import pandas as pd
from pandas.api import types as ptypes

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
from sklearn.model_selection import GroupKFold, ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Do not silence all warnings. Only hide very noisy known warnings if they appear.
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from imblearn.over_sampling import SMOTE
    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False
    SMOTE = None

TARGET_COL = "label"
GROUP_COL = "scaffold_group"
META_COLS = [
    "No",
    "SMILES",
    "label",
    "murcko_scaffold",
    "scaffold_group",
    "scaffold_group_type",
    "scaffold_split_80_20",
    "canonical_smiles",
    "standardized_smiles",
    "analysis_smiles",
    "representation",
    "metal_elements",
    "scaffold_origin",
    "scaffold_source",
    "scaffold_family",
]
RAW_TEXT_EXCLUDE = {
    "canonical_smiles",
    "standardized_smiles",
    "analysis_smiles",
    "representation",
    "metal_elements",
}
FORCE_CATEGORICAL_COLS = {
    "scaffold_group_type",
    "scaffold_origin",
    "scaffold_source",
    "scaffold_family",
}
BASE_STEMS = {"ames": "ames", "invitro": "invitro", "invivo": "invivo"}
RAW_STEMS = {"ames": "ames_combine", "invitro": "invitro_pre", "invivo": "invivo_pre"}

POS_ALERTS = {
    "ames": {
        "rule_bb_sa10_n_nitroso_ge1": 1.6,
        "rule_bb_sa5_nitro_aromatic_ge2": 1.4,
        "rule_epoxide_ge1": 1.2,
        "rule_hydrazine_like_ge1": 1.0,
        "fg_BB_SA10_N_nitroso_present": 1.3,
        "fg_BB_SA5_nitro_aromatic_present": 1.1,
        "fg_epoxide_present": 0.9,
        "bb_n_genotox_alerts": 0.6,
    },
    "invitro": {
        "rule_bb_sa2_primary_aromatic_amine_ge1": 1.4,
        "rule_aniline_ge1": 1.0,
        "rule_bb_sa20_ab_unsat_carbonyl_ge1": 1.0,
        "fg_BB_SA2_primary_aromatic_amine_present": 1.2,
        "fg_aniline_present": 0.8,
        "fg_BB_SA20_ab_unsat_carbonyl_present": 0.8,
        "bb_n_genotox_alerts": 0.5,
    },
    "invivo": {
        "rule_epoxide_ge1": 1.4,
        "rule_organometal_like_ge1": 1.2,
        "fg_epoxide_present": 1.0,
        "fg_organometal_like_present": 0.9,
        "contains_metal": 0.5,
        "contains_heavy_metal": 0.5,
    },
}
NEG_ALERTS = {
    "ames": {
        "rule_ester_ge2": 1.1,
        "rule_sulfone_ge2": 1.0,
        "rule_bb_sa47_n_alkylcarboxylic_acid_ge1": 1.2,
    },
    "invitro": {
        "rule_bb_sa47_n_alkylcarboxylic_acid_ge1": 1.1,
        "rule_ester_eq1": 0.8,
    },
    "invivo": {
        "rule_alcohol_ge1": 1.2,
        "rule_fused_ring_like_ge1": 0.9,
    },
}
SEARCH_SPACE = {
    "logreg": {
        "C": [0.15, 0.3, 0.6, 1.0, 1.5],
        "l1_ratio": [0.05, 0.15, 0.30, 0.50, 0.70],
    },
    "xgb": {
        "n_estimators": [160, 240, 320],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.7, 0.85],
        "min_child_weight": [2, 4, 6],
        "reg_lambda": [1.0, 3.0, 5.0],
        "reg_alpha": [0.0, 0.5, 1.0],
    },
}


def make_seed(root_seed: int, *parts: object) -> int:
    seq_parts = [int(root_seed)]
    for p in parts:
        seq_parts.append(zlib.crc32(str(p).encode("utf-8")) & 0xFFFFFFFF)
    ss = np.random.SeedSequence(seq_parts)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def coerce_valid_mask(df: pd.DataFrame, col: str = "valid") -> pd.Series:
    if col not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    s = df[col]
    if ptypes.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    s_num = pd.to_numeric(s, errors="coerce")
    if float(s_num.notna().mean()) >= 0.8:
        return s_num.fillna(0).astype(int).astype(bool)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"1", "true", "t", "yes", "y", "valid"})


def read_csv_from_zip_or_dir(source: Path, rel_path: str) -> pd.DataFrame:
    if source.is_dir():
        return pd.read_csv(source / rel_path)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as zf:
            with zf.open(rel_path) as fp:
                return pd.read_csv(fp)
    raise FileNotFoundError(str(source))


def make_scaffold_group_type(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    out = np.where(s.eq("") | s.eq("nan"), "acyclic_anonymous_graph", "murcko_scaffold")
    return pd.Series(out, index=series.index, name="scaffold_group_type")


def try_import_rdkit():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    return Chem, AllChem, DataStructs


def morgan_bits(smiles: pd.Series, n_bits: int = 256, radius: int = 2) -> pd.DataFrame:
    Chem, AllChem, DataStructs = try_import_rdkit()
    arr = np.zeros((len(smiles), n_bits), dtype=np.int8)
    for i, smi in enumerate(smiles.fillna("").astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        tmp = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, tmp)
        arr[i, :] = tmp
    cols = [f"fp_morgan2_{j}" for j in range(n_bits)]
    return pd.DataFrame(arr, columns=cols)


def locate_fg_source(base_dir: Path, data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / "fg_descriptor_analysis.zip",
        base_dir / "fg_descriptor_analysis.zip",
        data_dir / "fg_descriptor_analysis",
        base_dir / "fg_descriptor_analysis",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_base_split_files(endpoint: str, data_dir: Path) -> Tuple[Path, Path]:
    stem = BASE_STEMS[endpoint]
    patterns = [
        (f"{stem}_recommended_union_train.csv", f"{stem}_recommended_union_test.csv"),
        (f"{RAW_STEMS[endpoint]}_train_set_scaffold80_20.csv", f"{RAW_STEMS[endpoint]}_test_set_scaffold80_20.csv"),
    ]
    for tr, te in patterns:
        p1, p2 = data_dir / tr, data_dir / te
        if p1.exists() and p2.exists():
            return p1, p2
    raise FileNotFoundError(f"Base split files not found for {endpoint} in {data_dir}")


def build_broadfp(endpoint: str, base_dir: Path, data_dir: Path, n_bits: int = 256) -> Tuple[Path, Path]:
    out_train = data_dir / f"{BASE_STEMS[endpoint]}_broadfp_train.csv"
    out_test = data_dir / f"{BASE_STEMS[endpoint]}_broadfp_test.csv"
    if out_train.exists() and out_test.exists():
        return out_train, out_test

    fg_source = locate_fg_source(base_dir, data_dir)
    if fg_source is None:
        raise FileNotFoundError("fg_descriptor_analysis.zip (or folder) not found in model/ or model/data/")

    base_train_path, base_test_path = find_base_split_files(endpoint, data_dir)
    base_train = pd.read_csv(base_train_path)
    base_test = pd.read_csv(base_test_path)

    raw = RAW_STEMS[endpoint]
    canonical_rel = f"fg_descriptor_analysis/{raw}/{raw}_canonical_analysis.csv"
    views_rel = f"fg_descriptor_analysis/{raw}/{raw}_preprocessed_views.csv"
    canonical = read_csv_from_zip_or_dir(fg_source, canonical_rel)
    views = read_csv_from_zip_or_dir(fg_source, views_rel)

    canonical = canonical.loc[coerce_valid_mask(canonical, "valid")].copy().reset_index(drop=True)
    canonical = canonical.drop(columns=[c for c in ["representation", "analysis_smiles", "valid", "metal_elements"] if c in canonical.columns], errors="ignore")
    if "No" not in canonical.columns:
        raise ValueError(f"'No' missing in canonical analysis for {endpoint}")

    views_keep = [c for c in ["No", "SMILES", "canonical_smiles"] if c in views.columns]
    views = views[views_keep].drop_duplicates(subset=["No"])

    def merge_one(base_df: pd.DataFrame) -> pd.DataFrame:
        df = base_df.copy()
        df = df.merge(views, on="No", how="left", suffixes=("", "_view"))
        if "SMILES_view" in df.columns:
            df["SMILES"] = df["SMILES"].fillna(df["SMILES_view"])
            df = df.drop(columns=["SMILES_view"])
        df = df.merge(canonical, on="No", how="left", suffixes=("", "_canon"))
        for c in list(df.columns):
            if c.endswith("_canon") and c[:-6] in df.columns:
                df[c[:-6]] = df[c[:-6]].fillna(df[c])
                df = df.drop(columns=[c])
        if "scaffold_group_type" not in df.columns:
            base_scaffold = df["murcko_scaffold"] if "murcko_scaffold" in df.columns else pd.Series("", index=df.index)
            df["scaffold_group_type"] = make_scaffold_group_type(base_scaffold)
        smi_source = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
        fp = morgan_bits(df[smi_source], n_bits=n_bits, radius=2)
        df = pd.concat([df.reset_index(drop=True), fp.reset_index(drop=True)], axis=1)
        return df

    train_df = merge_one(base_train)
    test_df = merge_one(base_test)
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    return out_train, out_test


def detect_qm_file(config: Dict[str, object], base_dir: Path, data_dir: Path) -> Optional[Path]:
    qm_file = config.get("qm_file")
    if qm_file:
        p = Path(str(qm_file))
        if not p.is_absolute():
            p = base_dir / p
        return p if p.exists() else None
    for p in [data_dir, base_dir]:
        for cand in p.glob("*qm*.csv"):
            return cand
    return None


def merge_qm_if_available(df: pd.DataFrame, qm_path: Optional[Path], merge_key: str = "No") -> pd.DataFrame:
    if qm_path is None or not qm_path.exists():
        return df
    qm = pd.read_csv(qm_path)
    if merge_key not in df.columns:
        return df
    if merge_key not in qm.columns:
        return df
    qm_cols = [c for c in qm.columns if c != merge_key]
    rename = {c: (c if c.startswith("qm_") else f"qm_{c}") for c in qm_cols}
    qm = qm.rename(columns=rename)
    return df.merge(qm, on=merge_key, how="left")


def feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    all_cols = [c for c in df.columns if c not in META_COLS and c not in RAW_TEXT_EXCLUDE]

    blocks["rules"] = [
        c for c in all_cols
        if c.startswith("rule_") or c in ("bb_n_genotox_alerts", "bb_n_alerts", "expert_positive_alert_score", "expert_negative_alert_score")
    ]
    blocks["fg_present"] = [c for c in all_cols if c.startswith("fg_") and c.endswith("_present")]
    blocks["fg_count"] = [c for c in all_cols if c.startswith("fg_") and c.endswith("_count")]
    blocks["fingerprint"] = [c for c in all_cols if c.startswith("fp_morgan2_")]
    blocks["qm"] = [c for c in all_cols if c.startswith("qm_")]

    excluded = set(sum(blocks.values(), []))
    physchem_cols: List[str] = []
    for c in all_cols:
        if c in excluded or c in FORCE_CATEGORICAL_COLS:
            continue
        s = df[c]
        if ptypes.is_bool_dtype(s) or ptypes.is_numeric_dtype(s):
            physchem_cols.append(c)
            continue
        s_num = pd.to_numeric(s, errors="coerce")
        if float(s_num.notna().mean()) >= 0.98:
            physchem_cols.append(c)
    blocks["physchem"] = physchem_cols
    return blocks


def add_expert_scores(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    df = df.copy()
    pos_score = np.zeros(len(df), dtype=float)
    neg_score = np.zeros(len(df), dtype=float)
    for col, w in POS_ALERTS.get(endpoint, {}).items():
        if col in df.columns:
            pos_score += (pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy() > 0).astype(float) * float(w)
    for col, w in NEG_ALERTS.get(endpoint, {}).items():
        if col in df.columns:
            neg_score += (pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy() > 0).astype(float) * float(w)
    df["expert_positive_alert_score"] = pos_score
    df["expert_negative_alert_score"] = neg_score
    return df


def pick_fp_bits(train_df: pd.DataFrame, y: pd.Series, fp_cols: List[str], top_k: Optional[int]) -> List[str]:
    if top_k is None or len(fp_cols) <= top_k:
        return list(fp_cols)
    pos = train_df.loc[y == 1, fp_cols].mean(axis=0)
    neg = train_df.loc[y == 0, fp_cols].mean(axis=0)
    score = (pos - neg).abs().sort_values(ascending=False)
    return score.index[:int(top_k)].tolist()


def select_feature_cols(
    train_df: pd.DataFrame,
    block_names: List[str],
    fp_top_k: Optional[int],
    y_for_selection: Optional[pd.Series] = None,
) -> List[str]:
    blocks = feature_blocks(train_df)
    cols: List[str] = []
    for b in block_names:
        cols.extend(blocks.get(b, []))
    cols = list(dict.fromkeys([c for c in cols if c in train_df.columns]))
    if "fingerprint" in block_names:
        fp_cols = [c for c in cols if c.startswith("fp_morgan2_")]
        if y_for_selection is None:
            y_for_selection = train_df[TARGET_COL].astype(int)
        keep_fp = pick_fp_bits(train_df, y_for_selection, fp_cols, fp_top_k)
        cols = [c for c in cols if not c.startswith("fp_morgan2_")] + keep_fp
    if "scaffold_group_type" in train_df.columns and "scaffold_group_type" not in cols:
        cols = ["scaffold_group_type"] + cols
    return cols


def infer_types(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        if c in FORCE_CATEGORICAL_COLS:
            cat_cols.append(c)
            continue
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cat_cols.append(c)
            continue
        if ptypes.is_bool_dtype(s) or ptypes.is_numeric_dtype(s):
            num_cols.append(c)
            continue
        if ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s):
            # Try numeric coercion before declaring categorical.
            s_num = pd.to_numeric(s, errors="coerce")
            if float(s_num.notna().mean()) >= 0.98:
                num_cols.append(c)
            else:
                cat_cols.append(c)
            continue
        s_num = pd.to_numeric(s, errors="coerce")
        if float(s_num.notna().mean()) >= 0.98:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def _to_numeric_frame(X):
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    return X_df.apply(pd.to_numeric, errors="coerce")


def _to_categorical_frame(X):
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_df = X_df.copy()
    for c in X_df.columns:
        X_df[c] = X_df[c].astype("string")
        X_df[c] = X_df[c].replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    return X_df


def build_preprocessor(df: pd.DataFrame, cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols, cat_cols = infer_types(df, cols)
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("to_numeric", FunctionTransformer(_to_numeric_frame, validate=False, feature_names_out="one-to-one")),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("to_cat", FunctionTransformer(_to_categorical_frame, validate=False, feature_names_out="one-to-one")),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    return pre, num_cols, cat_cols


def make_model(model_name: str, endpoint: str, y: pd.Series, random_state: int, params: Dict[str, object]):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if model_name == "logreg":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=8000,
            class_weight="balanced",
            C=float(params.get("C", 0.6)),
            l1_ratio=float(params.get("l1_ratio", 0.25)),
            random_state=random_state,
        )
    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed or failed to import")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(params.get("n_estimators", 240)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.85)),
            min_child_weight=float(params.get("min_child_weight", 2)),
            reg_lambda=float(params.get("reg_lambda", 3.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            random_state=random_state,
            n_jobs=2,
            scale_pos_weight=max(1.0, neg / max(pos, 1)),
        )
    raise ValueError(model_name)


def alert_bootstrap(X_df: pd.DataFrame, y: pd.Series, endpoint: str, random_state: int, target_ratio: float = 0.45) -> Tuple[pd.DataFrame, pd.Series]:
    pos_idx = y[y == 1].index
    neg_n = int((y == 0).sum())
    pos_n = int((y == 1).sum())
    if pos_n == 0 or pos_n / max(neg_n, 1) >= target_ratio:
        return X_df, y
    add_n = int(max(0, math.ceil(target_ratio * neg_n) - pos_n))

    pos_pool = X_df.loc[pos_idx].copy()
    if "expert_positive_alert_score" in pos_pool.columns:
        weights = pd.to_numeric(pos_pool["expert_positive_alert_score"], errors="coerce").fillna(0).to_numpy()
        mask = weights > 0
        if mask.sum() > 0:
            pos_pool = pos_pool.loc[mask]
            weights = weights[mask]
        else:
            weights = np.ones(len(pos_pool), dtype=float)
    else:
        weights = np.ones(len(pos_pool), dtype=float)

    if len(pos_pool) == 0:
        return X_df, y

    rng = np.random.default_rng(make_seed(random_state, endpoint, "alert_bootstrap"))
    probs = weights / weights.sum()
    chosen = rng.choice(np.arange(len(pos_pool)), size=add_n, replace=True, p=probs)
    X_extra = pos_pool.iloc[chosen].copy()
    y_extra = pd.Series(np.ones(len(X_extra), dtype=int), index=np.arange(len(X_extra)))
    X_aug = pd.concat([X_df.reset_index(drop=True), X_extra.reset_index(drop=True)], axis=0, ignore_index=True)
    y_aug = pd.concat([y.reset_index(drop=True), y_extra.reset_index(drop=True)], axis=0, ignore_index=True)
    return X_aug, y_aug


def apply_smote(X_mat, y: pd.Series, random_state: int, sampling_ratio: float = 0.45):
    if not HAVE_IMBLEARN:
        return X_mat, y
    pos_n = int((y == 1).sum())
    if pos_n < 3:
        return X_mat, y
    k = min(5, pos_n - 1)
    sm = SMOTE(sampling_strategy=min(sampling_ratio, 0.99), random_state=random_state, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_mat, y)
    return X_res, pd.Series(y_res)


def train_once(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    X_valid_df: pd.DataFrame,
    y_valid: pd.Series,
    feature_cols: List[str],
    endpoint: str,
    model_name: str,
    params: Dict[str, object],
    strategy: str,
    random_state: int,
):
    X_work, y_work = X_train_df[feature_cols].copy(), y_train.copy()
    if strategy in ("alert_bootstrap", "hybrid"):
        X_work, y_work = alert_bootstrap(X_work, y_work, endpoint, random_state=random_state)

    pre, _, _ = build_preprocessor(X_work, feature_cols)
    Xtr = pre.fit_transform(X_work)
    Xva = pre.transform(X_valid_df[feature_cols].copy())

    if strategy in ("smote", "hybrid"):
        Xtr_dense = Xtr.toarray() if hasattr(Xtr, "toarray") else np.asarray(Xtr)
        Xtr_dense, y_work = apply_smote(Xtr_dense, y_work, random_state=random_state)
        Xtr = Xtr_dense

    model = make_model(model_name, endpoint, y_work, random_state, params)
    model.fit(Xtr, y_work)
    prob = model.predict_proba(Xva)[:, 1]
    return pre, model, prob


def best_threshold(y_true: np.ndarray, prob: np.ndarray, metric: str, specificity_floor: float = 0.0) -> Tuple[float, float]:
    thrs = np.linspace(0.05, 0.95, 91)
    best_thr, best_score = 0.5, -1e9
    for thr in thrs:
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        if spec < specificity_floor:
            continue
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            score = matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else -1.0
        if score > best_score:
            best_thr, best_score = thr, score
    return float(best_thr), float(best_score)


def feature_cols_for_fold(train_fold_df: pd.DataFrame, block_names: List[str], fp_top_k: Optional[int]) -> List[str]:
    return select_feature_cols(
        train_fold_df,
        block_names=block_names,
        fp_top_k=fp_top_k,
        y_for_selection=train_fold_df[TARGET_COL].astype(int),
    )


def candidate_cv_score(
    train_df: pd.DataFrame,
    block_names: List[str],
    fp_top_k: Optional[int],
    endpoint: str,
    model_name: str,
    params: Dict[str, object],
    strategy: str,
    cv_splits: int,
    metric: str,
    random_state: int,
) -> float:
    y = train_df[TARGET_COL].astype(int)
    groups = train_df[GROUP_COL].astype(str)
    n_splits = min(cv_splits, groups.nunique())
    if n_splits < 2:
        return -1.0

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
        fold_train = train_df.iloc[tr_idx].copy()
        fold_valid = train_df.iloc[va_idx].copy()
        fold_cols = feature_cols_for_fold(fold_train, block_names, fp_top_k)
        _, _, prob = train_once(
            fold_train,
            fold_train[TARGET_COL].astype(int),
            fold_valid,
            fold_valid[TARGET_COL].astype(int),
            fold_cols,
            endpoint,
            model_name,
            params,
            strategy,
            make_seed(random_state, endpoint, strategy, model_name, "cv", fold),
        )
        y_va = fold_valid[TARGET_COL].astype(int).to_numpy()
        spec_floor = 0.75 if endpoint == "invitro" else (0.80 if endpoint == "invivo" else 0.0)
        if metric == "average_precision":
            score = average_precision_score(y_va, prob)
        elif metric == "roc_auc":
            score = roc_auc_score(y_va, prob) if len(np.unique(y_va)) > 1 else np.nan
        else:
            _, score = best_threshold(
                y_va,
                prob,
                metric="balanced_accuracy" if endpoint == "invivo" else "mcc",
                specificity_floor=spec_floor,
            )
        fold_scores.append(score)

    arr = np.array(fold_scores, dtype=float)
    return float(np.nanmean(arr))


def tune_params(
    train_df: pd.DataFrame,
    block_names: List[str],
    fp_top_k: Optional[int],
    endpoint: str,
    model_name: str,
    strategy: str,
    cv_splits: int,
    metric: str,
    random_state: int,
    search_n_iter: int,
):
    space = SEARCH_SPACE[model_name]
    full_grid = list(ParameterGrid(space))
    total = len(full_grid)
    n_iter = min(search_n_iter, total)
    if n_iter >= total:
        candidates = full_grid
    else:
        rs = make_seed(random_state, endpoint, strategy, model_name, "search")
        candidates = list(ParameterSampler(space, n_iter=n_iter, random_state=rs))

    best_p, best_s = candidates[0], -1e9
    rows = []
    for i, params in enumerate(candidates):
        cand_seed = make_seed(random_state, endpoint, strategy, model_name, "candidate", i)
        score = candidate_cv_score(
            train_df=train_df,
            block_names=block_names,
            fp_top_k=fp_top_k,
            endpoint=endpoint,
            model_name=model_name,
            params=params,
            strategy=strategy,
            cv_splits=cv_splits,
            metric=metric,
            random_state=cand_seed,
        )
        rows.append({"candidate": i, **params, "cv_score": score})
        if score > best_s:
            best_s, best_p = score, params
    tuning_df = pd.DataFrame(rows).sort_values("cv_score", ascending=False)
    return best_p, tuning_df, float(best_s), total, n_iter


def tune_threshold_oof(
    train_df: pd.DataFrame,
    block_names: List[str],
    fp_top_k: Optional[int],
    endpoint: str,
    model_name: str,
    params: Dict[str, object],
    strategy: str,
    cv_splits: int,
    random_state: int,
):
    y = train_df[TARGET_COL].astype(int)
    groups = train_df[GROUP_COL].astype(str)
    n_splits = min(cv_splits, groups.nunique())
    if n_splits < 2:
        return 0.5, pd.DataFrame()

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(train_df), np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
        fold_train = train_df.iloc[tr_idx].copy()
        fold_valid = train_df.iloc[va_idx].copy()
        fold_cols = feature_cols_for_fold(fold_train, block_names, fp_top_k)
        _, _, prob = train_once(
            fold_train,
            fold_train[TARGET_COL].astype(int),
            fold_valid,
            fold_valid[TARGET_COL].astype(int),
            fold_cols,
            endpoint,
            model_name,
            params,
            strategy,
            make_seed(random_state, endpoint, strategy, model_name, "thr", fold),
        )
        oof[va_idx] = prob

    metric = "balanced_accuracy" if endpoint == "invivo" else "mcc"
    spec_floor = 0.75 if endpoint == "invitro" else (0.80 if endpoint == "invivo" else 0.0)
    thr, _ = best_threshold(y.to_numpy(), oof, metric, specificity_floor=spec_floor)

    sweep_rows = []
    for t in np.linspace(0.05, 0.95, 91):
        pred = (oof >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sweep_rows.append(
            {
                "threshold": t,
                "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else np.nan,
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
            }
        )
    return thr, pd.DataFrame(sweep_rows)


def get_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names = []
    if num_cols:
        names.extend(num_cols)
    if cat_cols:
        enc = pre.named_transformers_["cat"].named_steps["onehot"]
        names.extend(enc.get_feature_names_out(cat_cols).tolist())
    return names


def save_curves(y_true, prob, outdir: Path):
    fpr, tpr, _ = roc_curve(y_true, prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(outdir / "roc_curve.csv", index=False)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, prob)
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(outdir / "pr_curve.csv", index=False)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR")
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=150)
    plt.close()


def save_confusion(y_true, pred, outdir: Path):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(outdir / "confusion_matrix_counts.csv")
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.title("Confusion matrix (norm)")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix_normalized.png", dpi=150)
    plt.close()


def save_threshold_sweep(sweep_df: pd.DataFrame, outdir: Path):
    sweep_df.to_csv(outdir / "threshold_sweep.csv", index=False)
    plt.figure(figsize=(6, 4))
    for c in ["mcc", "balanced_accuracy", "sensitivity", "specificity"]:
        if c in sweep_df.columns:
            plt.plot(sweep_df["threshold"], sweep_df[c], label=c)
    plt.legend()
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Threshold sweep")
    plt.tight_layout()
    plt.savefig(outdir / "threshold_sweep.png", dpi=150)
    plt.close()


def save_importance(model, feature_names: List[str], outdir: Path):
    if hasattr(model, "coef_"):
        vals = np.abs(model.coef_.ravel())
    elif hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_)
    else:
        return
    imp = pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values("importance", ascending=False)
    imp.to_csv(outdir / "feature_importance.csv", index=False)
    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(7, max(4, 0.25 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top feature importance")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance_top.png", dpi=150)
    plt.close()


def maybe_shap_best(run_root: Path, top_n_background: int = 200, top_n_explain: int = 300):
    summary_path = run_root / "summary_metrics.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    best = summary.sort_values(["endpoint", "mcc", "balanced_accuracy", "pr_auc"], ascending=[True, False, False, False]).groupby("endpoint").head(1)

    try:
        import shap
    except Exception as e:
        (run_root / "shap_skipped.txt").write_text(str(e), encoding="utf-8")
        return

    for _, row in best.iterrows():
        exp_dir = Path(row["exp_dir"])
        try:
            bundle = joblib.load(exp_dir / "trained_bundle.joblib")
            pre = bundle["preprocessor"]
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]
            Xbg = bundle["X_background"][feature_cols].head(top_n_background)
            Xex = bundle["X_explain"][feature_cols].head(top_n_explain)
            Xt_bg = pre.transform(Xbg)
            Xt_ex = pre.transform(Xex)
            if hasattr(Xt_bg, "toarray"):
                Xt_bg = Xt_bg.toarray()
            if hasattr(Xt_ex, "toarray"):
                Xt_ex = Xt_ex.toarray()
            num_cols = bundle["num_cols"]
            cat_cols = bundle["cat_cols"]
            feature_names = get_feature_names(pre, num_cols, cat_cols)
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                raw = explainer.shap_values(Xt_ex)
            else:
                explainer = shap.Explainer(model, Xt_bg, feature_names=feature_names)
                raw = explainer(Xt_ex)
                raw = getattr(raw, "values", raw)
            vals = np.asarray(raw)
            if vals.ndim == 3:
                vals = vals[..., 1]
            mean_abs = np.abs(vals).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv(exp_dir / "shap_mean_abs.csv", index=False)
            top = shap_df.head(20).iloc[::-1]
            plt.figure(figsize=(7, max(4, 0.25 * len(top))))
            plt.barh(top["feature"], top["mean_abs_shap"])
            plt.title("SHAP mean |value|")
            plt.tight_layout()
            plt.savefig(exp_dir / "shap_summary_bar.png", dpi=150)
            plt.close()
        except Exception:
            (exp_dir / "shap_error.txt").write_text(traceback.format_exc(), encoding="utf-8")


def run_experiment(endpoint: str, strategy: str, model_name: str, config: Dict[str, object], run_root: Path):
    data_dir = Path(config["data_dir"])
    base_dir = Path(config["base_dir"])
    build_broadfp(endpoint, base_dir, data_dir, n_bits=int(config.get("fp_bits", 256)))

    train_path = data_dir / f"{BASE_STEMS[endpoint]}_broadfp_train.csv"
    test_path = data_dir / f"{BASE_STEMS[endpoint]}_broadfp_test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = add_expert_scores(train_df, endpoint)
    test_df = add_expert_scores(test_df, endpoint)

    qm_path = detect_qm_file(config, base_dir, data_dir)
    train_df = merge_qm_if_available(train_df, qm_path, merge_key=str(config.get("qm_merge_key", "No")))
    test_df = merge_qm_if_available(test_df, qm_path, merge_key=str(config.get("qm_merge_key", "No")))

    block_names = list(config["feature_blocks"])
    fp_top_k = config.get("fp_top_k")
    feature_cols = select_feature_cols(train_df, block_names, fp_top_k, y_for_selection=train_df[TARGET_COL].astype(int))

    exp_dir = run_root / f"{endpoint}__{strategy}__{model_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    tuning_metric = str(config.get("tuning_metric", "mcc"))
    search_n_iter = int(config.get(f"search_n_iter_{model_name}", config.get("search_n_iter", 24)))
    params, tuning_df, best_cv, total_grid, sampled_grid = tune_params(
        train_df=train_df,
        block_names=block_names,
        fp_top_k=fp_top_k,
        endpoint=endpoint,
        model_name=model_name,
        strategy=strategy,
        cv_splits=int(config.get("cv_splits", 3)),
        metric=tuning_metric,
        random_state=int(config.get("random_state", 42)),
        search_n_iter=search_n_iter,
    )
    tuning_df.to_csv(exp_dir / "tuning_results.csv", index=False)

    thr, sweep_df = tune_threshold_oof(
        train_df=train_df,
        block_names=block_names,
        fp_top_k=fp_top_k,
        endpoint=endpoint,
        model_name=model_name,
        params=params,
        strategy=strategy,
        cv_splits=int(config.get("cv_splits", 3)),
        random_state=int(config.get("random_state", 42)),
    )
    save_threshold_sweep(sweep_df, exp_dir)

    X_work = train_df[feature_cols].copy()
    y_work = train_df[TARGET_COL].astype(int).copy()
    if strategy in ("alert_bootstrap", "hybrid"):
        X_work, y_work = alert_bootstrap(
            X_work,
            y_work,
            endpoint,
            random_state=make_seed(int(config.get("random_state", 42)), endpoint, strategy, "fit"),
        )

    pre, num_cols, cat_cols = build_preprocessor(X_work, feature_cols)
    Xtr = pre.fit_transform(X_work)
    if strategy in ("smote", "hybrid"):
        Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else np.asarray(Xtr)
        Xtr, y_work = apply_smote(
            Xtr,
            y_work,
            random_state=make_seed(int(config.get("random_state", 42)), endpoint, strategy, "smote_fit"),
        )

    model = make_model(
        model_name,
        endpoint,
        y_work,
        make_seed(int(config.get("random_state", 42)), endpoint, strategy, model_name, "fit_model"),
        params,
    )
    model.fit(Xtr, y_work)

    Xte = pre.transform(test_df[feature_cols].copy())
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= thr).astype(int)
    y_true = test_df[TARGET_COL].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    metrics = {
        "endpoint": endpoint,
        "strategy": strategy,
        "model_name": model_name,
        "n_features": len(feature_cols),
        "threshold": float(thr),
        "tuning_metric": tuning_metric,
        "best_cv_score": float(best_cv),
        "search_candidates_total": int(total_grid),
        "search_candidates_evaluated": int(sampled_grid),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "train_pos_rate": float(train_df[TARGET_COL].mean()),
        "test_pos_rate": float(test_df[TARGET_COL].mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "exp_dir": str(exp_dir),
    }

    pd.DataFrame([metrics]).to_csv(exp_dir / "metrics.csv", index=False)
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_meta_cols = [c for c in META_COLS if c in test_df.columns]
    preds = test_df[pred_meta_cols].copy()
    preds["pred_proba"] = prob
    preds["pred_label"] = pred
    preds.to_csv(exp_dir / "test_predictions.csv", index=False)
    preds.loc[preds[TARGET_COL] != preds["pred_label"]].to_csv(exp_dir / "test_misclassifications.csv", index=False)

    save_confusion(y_true, pred, exp_dir)
    save_curves(y_true, prob, exp_dir)
    save_importance(model, get_feature_names(pre, num_cols, cat_cols), exp_dir)

    # Save a lean bundle only.
    joblib.dump(
        {
            "preprocessor": pre,
            "model": model,
            "feature_cols": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "X_background": train_df[feature_cols].head(int(config.get("shap_background_n", 200))).copy(),
            "X_explain": test_df[feature_cols].head(int(config.get("shap_explain_n", 300))).copy(),
        },
        exp_dir / "trained_bundle.joblib",
    )

    gc.collect()
    return metrics


def build_summary(run_root: Path):
    rows = []
    for p in sorted(run_root.glob("*__*__*/metrics.csv")):
        try:
            rows.append(pd.read_csv(p).iloc[0].to_dict())
        except Exception:
            continue
    if not rows:
        pd.DataFrame().to_csv(run_root / "summary_metrics.csv", index=False)
        pd.DataFrame().to_csv(run_root / "best_by_endpoint.csv", index=False)
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    ranked = summary.sort_values(["endpoint", "mcc", "balanced_accuracy", "pr_auc"], ascending=[True, False, False, False])
    ranked.to_csv(run_root / "summary_metrics_ranked.csv", index=False)
    best = ranked.groupby("endpoint").head(1)
    best.to_csv(run_root / "best_by_endpoint.csv", index=False)

    for metric in ["mcc", "balanced_accuracy", "pr_auc"]:
        plt.figure(figsize=(8, 4))
        labels = summary["endpoint"] + " | " + summary["strategy"] + " | " + summary["model_name"]
        plt.bar(range(len(summary)), summary[metric].astype(float))
        plt.xticks(range(len(summary)), labels, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(metric.upper())
        plt.tight_layout()
        plt.savefig(run_root / f"summary_{metric}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--mode", default="run", choices=["build", "run", "shap"])
    args = ap.parse_args()

    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    base_dir = Path(config["base_dir"])
    data_dir = Path(config["data_dir"])
    runs_dir = Path(config["runs_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "build":
        for ep in config["endpoints"]:
            print(f"[BUILD] {ep}")
            build_broadfp(ep, base_dir, data_dir, n_bits=int(config.get("fp_bits", 256)))
        print("[BUILD] done")
        return

    if args.mode == "shap":
        latest = runs_dir / "LATEST_SUBPROC_RUN_PATH.txt"
        if not latest.exists():
            raise FileNotFoundError("No latest run pointer found.")
        run_root = Path(latest.read_text(encoding="utf-8").strip())
        maybe_shap_best(run_root)
        print(f"[SHAP] done: {run_root}")
        return

    run_root = runs_dir / f"subproc_allinone_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    (runs_dir / "LATEST_SUBPROC_RUN_PATH.txt").write_text(str(run_root), encoding="utf-8")
    with open(run_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for ep in config["endpoints"]:
        build_broadfp(ep, base_dir, data_dir, n_bits=int(config.get("fp_bits", 256)))
        for strategy in config["strategies"]:
            for model_name in config["models"]:
                print(f"[RUN] endpoint={ep} strategy={strategy} model={model_name}")
                try:
                    metrics = run_experiment(ep, strategy, model_name, config, run_root)
                    print("[OK]", json.dumps({k: metrics[k] for k in ["endpoint", "strategy", "model_name", "mcc", "balanced_accuracy", "pr_auc"]}, ensure_ascii=False))
                except Exception:
                    err_dir = run_root / f"{ep}__{strategy}__{model_name}"
                    err_dir.mkdir(parents=True, exist_ok=True)
                    (err_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print("[ERROR]", ep, strategy, model_name)
                    print(traceback.format_exc())
                gc.collect()

    build_summary(run_root)
    print(f"[DONE] run_root={run_root}")


if __name__ == "__main__":
    main()
