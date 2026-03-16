from __future__ import annotations

import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolHash
from sklearn.model_selection import StratifiedGroupKFold

RAW_ZIP = "/mnt/data/fg_descriptor_analysis.zip"
RULE_HIGHLIGHTS = "/mnt/data/fg_count_threshold_highlights.csv"
OUTDIR = "/mnt/data/unified_train_test_pipeline_outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_MODE = "recommended_union"
USE_QM = False
QM_FILE: Optional[str] = None
MERGE_KEY = "No"
QM_COLS: Optional[List[str]] = None

ID_COLS = ["No", "SMILES", "label"]
META_COLS = [
    "murcko_scaffold",
    "scaffold_group",
    "scaffold_group_type",
    "scaffold_split_80_20",
]

ENDPOINTS: Dict[str, Dict[str, object]] = {
    "ames": {
        "raw_name": "ames_combine",
        "compact": [
            "scaffold_group_type",
            "bb_n_genotox_alerts",
            "aromatic_ring_count",
            "fraction_csp3",
            "rot_bonds",
            "tpsa",
            "rule_bb_sa5_nitro_aromatic_ge2",
            "rule_bb_sa10_n_nitroso_ge1",
            "rule_epoxide_ge1",
            "rule_hydrazine_like_ge1",
            "rule_ester_ge2",
            "rule_sulfone_ge2",
            "rule_bb_sa47_n_alkylcarboxylic_acid_ge1",
        ],
        "extended_extra": [
            "fg_BB_SA5_nitro_aromatic_present",
            "fg_aniline_present",
            "fg_heteroaromatic_ring_present",
            "fg_fused_ring_like_present",
            "fg_epoxide_present",
            "fg_BB_SA10_N_nitroso_present",
            "fg_hydrazine_like_present",
            "fg_BB_SA2c_aromatic_amine_general_present",
            "fg_ester_present",
            "fg_ether_present",
            "fg_carboxylic_acid_present",
            "fg_sulfone_present",
            "fg_sulfonic_acid_or_sulfonate_present",
            "rule_epoxide_ge2",
            "rule_fluoro_ge5",
            "bb_n_alerts",
            "fg_BB_SA2_primary_aromatic_amine_present",
            "fg_BB_SA40_alkyl_halides_nongx_present",
            "has_formal_charge",
            "ring_count",
            "heavy_atom_count",
        ],
    },
    "invitro": {
        "raw_name": "invitro_pre",
        "compact": [
            "scaffold_group_type",
            "bb_n_genotox_alerts",
            "rot_bonds",
            "logp",
            "fraction_csp3",
            "mw",
            "rule_bb_sa2_primary_aromatic_amine_ge1",
            "rule_aniline_ge1",
            "rule_bb_sa20_ab_unsat_carbonyl_ge1",
            "rule_bb_sa47_n_alkylcarboxylic_acid_ge1",
            "rule_ester_eq1",
        ],
        "extended_extra": [
            "fg_aniline_present",
            "fg_halogen_any_present",
            "fg_BB_SA2c_aromatic_amine_general_present",
            "fg_BB_SA20_ab_unsat_carbonyl_present",
            "fg_BB_SA47_n_alkylcarboxylic_acid_present",
            "rule_bb_sa2_primary_aromatic_amine_ge2",
            "rule_epoxide_ge1",
            "rule_alcohol_ge2",
            "bb_n_alerts",
            "fg_BB_SA2_primary_aromatic_amine_present",
            "fg_BB_SA40_alkyl_halides_nongx_present",
            "fg_BB_SA18_sulphonate_ester_present",
        ],
    },
    "invivo": {
        "raw_name": "invivo_pre",
        "compact": [
            "scaffold_group_type",
            "mw",
            "hbd",
            "logp",
            "heteroatom_count",
            "rot_bonds",
            "rule_epoxide_ge1",
            "rule_organometal_like_ge1",
            "rule_alcohol_ge1",
            "rule_fused_ring_like_ge1",
        ],
        "extended_extra": [
            "fg_epoxide_present",
            "fg_organometal_like_present",
            "fg_fused_ring_like_present",
            "fg_alcohol_present",
            "rule_epoxide_ge2",
            "rule_organometal_like_ge2",
            "contains_heavy_metal",
            "contains_metal",
            "fg_BB_SA16b_aziridine_present",
        ],
    },
}


def rule_col_name(rule_row: pd.Series) -> str:
    base = str(rule_row["feature_clean"])
    mode = "ge" if rule_row["mode"] == "ge" else "eq"
    thr_val = float(rule_row["threshold"])
    thr = str(int(thr_val)) if thr_val.is_integer() else str(thr_val).replace(".", "p")
    safe = (
        base.lower()
        .replace("BB_", "bb_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"rule_{safe}_{mode}{thr}"


def apply_rule(series: pd.Series, mode: str, threshold: float) -> pd.Series:
    s = series.fillna(0)
    if mode == "ge":
        return (s >= threshold).astype(int)
    if mode == "eq":
        return (s == threshold).astype(int)
    raise ValueError(f"Unsupported mode: {mode}")


def make_scaffold_group(smiles: str, murcko: object) -> Tuple[str, str]:
    murcko_str = None if pd.isna(murcko) else str(murcko).strip()
    if murcko_str:
        return murcko_str, "murcko"
    mol = Chem.MolFromSmiles(smiles)
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
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
    target_test_size: float = TEST_SIZE,
) -> pd.Series:
    y = df[label_col].astype(int).values
    groups = df[group_col].astype(str).values
    overall_pos = y.mean()
    target_test_n = len(df) * target_test_size
    target_test_pos = y.sum() * target_test_size
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_mask = None
    best_score = None
    for _, test_idx in sgkf.split(df, y, groups=groups):
        mask = np.zeros(len(df), dtype=bool)
        mask[test_idx] = True
        test_n = int(mask.sum())
        test_pos = int(y[mask].sum())
        test_pos_rate = test_pos / test_n if test_n else 0.0
        size_penalty = abs(test_n - target_test_n) / max(len(df), 1)
        rate_penalty = abs(test_pos_rate - overall_pos)
        pos_count_penalty = abs(test_pos - target_test_pos) / max(y.sum(), 1)
        score = size_penalty + 2.0 * rate_penalty + 1.0 * pos_count_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_mask = mask

    split = pd.Series(np.where(best_mask, "test", "train"), index=df.index, name="scaffold_split_80_20")
    return split


def get_rule_table(endpoint_key: str, rule_table: pd.DataFrame) -> pd.DataFrame:
    raw_name = ENDPOINTS[endpoint_key]["raw_name"]
    return rule_table.loc[rule_table["endpoint"].eq(raw_name)].copy()


def load_raw_endpoint(endpoint_key: str, raw_zip: str) -> pd.DataFrame:
    raw_name = str(ENDPOINTS[endpoint_key]["raw_name"])
    with zipfile.ZipFile(raw_zip) as zz:
        path = f"fg_descriptor_analysis/{raw_name}/{raw_name}_canonical_analysis.csv"
        df = pd.read_csv(io.BytesIO(zz.read(path)), low_memory=False)
    df = df.loc[df["valid"].eq(1)].copy()
    df["SMILES"] = df["analysis_smiles"].astype(str)
    groups = [make_scaffold_group(s, m) for s, m in zip(df["SMILES"], df["murcko_scaffold"])]
    df["scaffold_group"] = [g for g, _ in groups]
    df["scaffold_group_type"] = [t for _, t in groups]
    return df


def merge_qm_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_QM or not QM_FILE:
        return df
    qm = pd.read_csv(QM_FILE)
    if MERGE_KEY not in df.columns or MERGE_KEY not in qm.columns:
        raise KeyError(f"MERGE_KEY '{MERGE_KEY}' must exist in both tables")
    if QM_COLS is None:
        exclude = {MERGE_KEY, "label", "No", "SMILES", "analysis_smiles"}
        qm_cols = [c for c in qm.columns if c not in exclude]
    else:
        qm_cols = list(QM_COLS)
    use_cols = [MERGE_KEY] + qm_cols
    qm_use = qm[use_cols].copy()
    return df.merge(qm_use, on=MERGE_KEY, how="left")


def dedupe_keep_order(cols: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def build_endpoint_table(endpoint_key: str, rule_table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = load_raw_endpoint(endpoint_key, RAW_ZIP)
    rules = get_rule_table(endpoint_key, rule_table)
    rule_cols = []
    for _, row in rules.iterrows():
        out_col = rule_col_name(row)
        source_col = str(row["feature"])
        if source_col not in df.columns:
            raise KeyError(f"Missing source feature '{source_col}' for endpoint '{endpoint_key}'")
        df[out_col] = apply_rule(df[source_col], str(row["mode"]), float(row["threshold"]))
        rule_cols.append(out_col)

    df = merge_qm_if_needed(df)
    df["scaffold_split_80_20"] = choose_best_holdout(df)

    compact = list(ENDPOINTS[endpoint_key]["compact"])
    extended = dedupe_keep_order(compact + list(ENDPOINTS[endpoint_key]["extended_extra"]))
    union_features = dedupe_keep_order(extended)
    if USE_QM and QM_FILE:
        if QM_COLS is None:
            qm = pd.read_csv(QM_FILE)
            exclude = {MERGE_KEY, "label", "No", "SMILES", "analysis_smiles"}
            qm_use_cols = [c for c in qm.columns if c not in exclude]
        else:
            qm_use_cols = list(QM_COLS)
        union_features = dedupe_keep_order(union_features + qm_use_cols)

    missing = [c for c in union_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing selected feature columns for {endpoint_key}: {missing}")

    keep_cols = dedupe_keep_order(ID_COLS + META_COLS + union_features)
    result = df[keep_cols].copy()

    feature_manifest = {
        "compact": compact,
        "extended": extended,
        "union": union_features,
        "all_rule_cols": rule_cols,
    }
    return result, feature_manifest


def endpoint_summary(df: pd.DataFrame, endpoint_key: str, feature_manifest: Dict[str, List[str]]) -> Dict[str, object]:
    train_mask = df["scaffold_split_80_20"].eq("train")
    test_mask = df["scaffold_split_80_20"].eq("test")
    shared_groups = sorted(set(df.loc[train_mask, "scaffold_group"]) & set(df.loc[test_mask, "scaffold_group"]))
    return {
        "endpoint": endpoint_key,
        "n_total": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "pos_rate_total": float(df["label"].mean()),
        "pos_rate_train": float(df.loc[train_mask, "label"].mean()),
        "pos_rate_test": float(df.loc[test_mask, "label"].mean()),
        "n_features_compact": len(feature_manifest["compact"]),
        "n_features_extended": len(feature_manifest["extended"]),
        "n_features_union": len(feature_manifest["union"]),
        "shared_scaffold_groups": len(shared_groups),
    }


def save_outputs(df: pd.DataFrame, endpoint_key: str, outdir: str) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    combined_path = out / f"{endpoint_key}_recommended_union_with_split.csv"
    train_path = out / f"{endpoint_key}_recommended_union_train.csv"
    test_path = out / f"{endpoint_key}_recommended_union_test.csv"
    assign_path = out / f"{endpoint_key}_split_assignments.csv"

    df.to_csv(combined_path, index=False)
    df.loc[df["scaffold_split_80_20"].eq("train")].to_csv(train_path, index=False)
    df.loc[df["scaffold_split_80_20"].eq("test")].to_csv(test_path, index=False)
    df[ID_COLS + META_COLS].to_csv(assign_path, index=False)

    return {
        "combined": str(combined_path),
        "train": str(train_path),
        "test": str(test_path),
        "assignments": str(assign_path),
    }


def bundle_outputs(outdir: str, bundle_name: str = "unified_train_test_pipeline_outputs.zip") -> str:
    out_path = Path(outdir)
    bundle_path = out_path.parent / bundle_name
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_path.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_path.parent))
    return str(bundle_path)


def main() -> None:
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    rule_table = pd.read_csv(RULE_HIGHLIGHTS)

    summaries = []
    all_paths = {}
    manifest = {
        "settings": {
            "raw_zip": RAW_ZIP,
            "rule_highlights": RULE_HIGHLIGHTS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "feature_mode": FEATURE_MODE,
            "use_qm": USE_QM,
            "qm_file": QM_FILE,
            "merge_key": MERGE_KEY,
            "qm_cols": QM_COLS,
        },
        "endpoints": {},
    }

    leakage_rows = []
    for endpoint_key in ENDPOINTS:
        df, feature_manifest = build_endpoint_table(endpoint_key, rule_table)
        paths = save_outputs(df, endpoint_key, OUTDIR)
        all_paths[endpoint_key] = paths
        summaries.append(endpoint_summary(df, endpoint_key, feature_manifest))
        manifest["endpoints"][endpoint_key] = {**feature_manifest, **paths}

        train_groups = set(df.loc[df["scaffold_split_80_20"].eq("train"), "scaffold_group"])
        test_groups = set(df.loc[df["scaffold_split_80_20"].eq("test"), "scaffold_group"])
        shared = train_groups & test_groups
        leakage_rows.append(
            {
                "endpoint": endpoint_key,
                "shared_scaffold_groups": len(shared),
                "shared_example": next(iter(shared)) if shared else "",
            }
        )

    summary_df = pd.DataFrame(summaries)
    leakage_df = pd.DataFrame(leakage_rows)
    summary_df.to_csv(Path(OUTDIR) / "split_summary.csv", index=False)
    leakage_df.to_csv(Path(OUTDIR) / "leakage_check.csv", index=False)

    manifest_path = Path(OUTDIR) / "feature_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = Path(OUTDIR) / "README.txt"
    readme.write_text(
        "Unified scaffold-aware train/test pipeline outputs\n\n"
        "Each endpoint now has only three main files for direct use:\n"
        "- *_recommended_union_train.csv\n"
        "- *_recommended_union_test.csv\n"
        "- *_recommended_union_with_split.csv\n\n"
        "The train/test files already include recommended endpoint-specific multiblock features\n"
        "(compact + extended union), scaffold metadata, and label.\n"
        "Use feature_manifest.json to recover compact vs extended subsets during training.\n",
        encoding="utf-8",
    )

    bundle_path = bundle_outputs(OUTDIR)
    print(summary_df)
    print(f"\\nBundle saved to: {bundle_path}")


if __name__ == "__main__":
    main()
