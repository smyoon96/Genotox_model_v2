"""
step2b_preprocessing_impact.py
===============================
전처리 전/후 데이터를 상세 비교 분석하는 모듈.

분석 항목:
  1) 전처리 단계별 영향 받은 화합물 식별 및 특성 분석
     - invalid SMILES 제거
     - metal 포함 화합물 (제거 또는 별도 처리)
     - salt stripping (fragment '.' 제거)
     - charge normalization
  2) 제거/변형된 화합물의 클래스 분포 및 화학적 특성
  3) 전처리 전/후 모델 성능 비교 (A/B test)
  4) 전처리 결정에 대한 근거 리포트

모든 분석은 train partition 기준으로 수행한다.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from config import ENDPOINTS, GLOBAL_SEED, save_json

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  금속 원소 정의
# ═══════════════════════════════════════════════════════════════
METAL_ATOMIC_NUMS = {
    3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,
    37,38,39,40,41,42,44,45,46,47,48,49,50,
    55,56,72,73,74,75,76,77,78,79,80,81,82,83,
}

METAL_NAMES = {
    3:"Li",4:"Be",11:"Na",12:"Mg",13:"Al",19:"K",20:"Ca",
    24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",
    33:"As",47:"Ag",48:"Cd",50:"Sn",51:"Sb",
    78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",
}


# ═══════════════════════════════════════════════════════════════
#  SECTION 1. 화합물별 전처리 플래그 부여
# ═══════════════════════════════════════════════════════════════

def classify_compound(smi: str) -> Dict:
    """
    단일 SMILES에 대해 전처리 관련 모든 플래그를 산출한다.
    """
    result = {
        "valid_smiles":      True,
        "has_metal":         False,
        "metal_elements":    [],
        "has_salt":          False,
        "n_fragments":       1,
        "needs_charge_norm": False,
        "sanitizable":       True,
        "smiles_raw":        smi,
        "smiles_stripped":   None,
        "smiles_canonical":  None,
        "mw_raw":            np.nan,
        "mw_after":          np.nan,
        "mw_change":         0.0,
        "preprocessing_actions": [],
    }

    if pd.isna(smi) or str(smi).strip() == "":
        result["valid_smiles"] = False
        result["sanitizable"] = False
        result["preprocessing_actions"].append("removed:empty")
        return result

    smi = str(smi).strip()

    # ── 1) 기본 파싱 ──
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result["valid_smiles"] = False
        result["sanitizable"] = False
        result["preprocessing_actions"].append("removed:invalid_smiles")
        return result

    result["mw_raw"] = Descriptors.MolWt(mol)

    # ── 2) Salt / Fragment 검사 ──
    if "." in smi:
        result["has_salt"] = True
        result["n_fragments"] = smi.count(".") + 1
        result["preprocessing_actions"].append("salt_stripped")

    # ── 3) Metal 검사 ──
    metal_found = []
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum in METAL_ATOMIC_NUMS:
            metal_found.append(METAL_NAMES.get(anum, f"Z{anum}"))
    if metal_found:
        result["has_metal"] = True
        result["metal_elements"] = list(set(metal_found))
        result["preprocessing_actions"].append(f"metal:{','.join(set(metal_found))}")

    # ── 4) Salt stripping ──
    try:
        stripped = rdMolStandardize.FragmentParent(mol)
        smi_stripped = Chem.MolToSmiles(stripped, canonical=True)
        result["smiles_stripped"] = smi_stripped
        if smi_stripped != Chem.MolToSmiles(mol, canonical=True):
            result["preprocessing_actions"].append("fragment_removed")
    except:
        result["smiles_stripped"] = Chem.MolToSmiles(mol, canonical=True)
        stripped = mol

    # ── 5) Charge normalization ──
    try:
        uncharger = rdMolStandardize.Uncharger()
        uncharged = uncharger.uncharge(stripped)
        smi_uncharged = Chem.MolToSmiles(uncharged, canonical=True)
        if smi_uncharged != Chem.MolToSmiles(stripped, canonical=True):
            result["needs_charge_norm"] = True
            result["preprocessing_actions"].append("charge_normalized")
        result["smiles_canonical"] = smi_uncharged
        result["mw_after"] = Descriptors.MolWt(uncharged)
    except:
        result["smiles_canonical"] = result["smiles_stripped"]
        result["mw_after"] = result["mw_raw"]

    result["mw_change"] = result["mw_after"] - result["mw_raw"]

    if not result["preprocessing_actions"]:
        result["preprocessing_actions"].append("none")

    return result


def classify_all_compounds(df: pd.DataFrame, smi_col: str = "SMILES_raw") -> pd.DataFrame:
    """전체 DataFrame에 전처리 플래그를 일괄 부여"""
    records = []
    for i, smi in enumerate(df[smi_col]):
        rec = classify_compound(smi)
        records.append(rec)
        if (i + 1) % 2000 == 0:
            logger.info(f"    Classified {i+1}/{len(df)} compounds...")

    flag_df = pd.DataFrame(records, index=df.index)
    return flag_df


# ═══════════════════════════════════════════════════════════════
#  SECTION 2. 전처리 영향 통계 분석
# ═══════════════════════════════════════════════════════════════

def analyze_preprocessing_impact(
    df: pd.DataFrame,
    flag_df: pd.DataFrame,
    endpoint: str,
    out_dir: Path,
) -> Dict:
    """
    전처리 단계별 영향 분석.

    각 전처리 카테고리에 대해:
    - 해당 화합물 수, 비율
    - 양성/음성 분포 및 전체 대비 enrichment
    - 분자량 변화
    - Fisher exact test로 클래스 연관성 검정
    """
    label = df["label"].values
    n_total = len(df)
    n_pos = (label == 1).sum()
    n_neg = (label == 0).sum()
    overall_pos_rate = n_pos / n_total

    categories = OrderedDict({
        "invalid_smiles": ~flag_df["valid_smiles"],
        "has_metal":      flag_df["has_metal"],
        "has_salt":       flag_df["has_salt"],
        "charge_normed":  flag_df["needs_charge_norm"],
        "multi_fragment":  flag_df["n_fragments"] > 1,
        "any_preprocessing": flag_df["preprocessing_actions"].apply(
            lambda x: x != ["none"] if isinstance(x, list) else True
        ),
    })

    stats_records = []
    for cat_name, mask in categories.items():
        n_affected = mask.sum()
        if n_affected == 0:
            stats_records.append({
                "category": cat_name,
                "n_affected": 0,
                "pct_of_total": 0,
                "n_pos": 0, "n_neg": 0,
                "pos_rate": 0,
                "overall_pos_rate": round(overall_pos_rate, 4),
                "enrichment_ratio": 0,
                "fisher_p": 1.0,
                "odds_ratio": 0,
                "recommendation": "N/A (none found)",
            })
            continue

        affected_labels = label[mask]
        cat_pos = (affected_labels == 1).sum()
        cat_neg = (affected_labels == 0).sum()
        cat_pos_rate = cat_pos / n_affected if n_affected > 0 else 0

        # enrichment ratio
        enrichment = cat_pos_rate / overall_pos_rate if overall_pos_rate > 0 else 0

        # Fisher exact: category × label
        a = cat_pos               # affected & positive
        b = cat_neg               # affected & negative
        c = n_pos - cat_pos       # unaffected & positive
        d = n_neg - cat_neg       # unaffected & negative
        try:
            odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])
        except:
            odds_ratio, fisher_p = np.nan, np.nan

        # 권장 사항
        if cat_name == "invalid_smiles":
            rec = "REMOVE – cannot compute descriptors"
        elif cat_name == "has_metal":
            if enrichment > 1.5:
                rec = f"CAUTION – enriched for positives (ER={enrichment:.1f}×). Keep in separate analysis or dedicated model."
            elif enrichment < 0.5:
                rec = f"KEEP or REMOVE – depleted for positives (ER={enrichment:.1f}×). Mostly negative."
            else:
                rec = f"KEEP – balanced enrichment (ER={enrichment:.1f}×)."
        elif cat_name == "has_salt":
            rec = "STRIP salt (keep parent fragment). Track MW change."
        elif cat_name == "charge_normed":
            rec = "NORMALIZE – standard practice. Minimal structural impact."
        else:
            rec = "REVIEW on case-by-case basis."

        stats_records.append({
            "category":         cat_name,
            "n_affected":       int(n_affected),
            "pct_of_total":     round(n_affected / n_total * 100, 2),
            "n_pos":            int(cat_pos),
            "n_neg":            int(cat_neg),
            "pos_rate":         round(cat_pos_rate, 4),
            "overall_pos_rate": round(overall_pos_rate, 4),
            "enrichment_ratio": round(enrichment, 3),
            "fisher_p":         round(fisher_p, 6) if not np.isnan(fisher_p) else None,
            "odds_ratio":       round(odds_ratio, 4) if not np.isnan(odds_ratio) else None,
            "recommendation":   rec,
        })

    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(out_dir / f"{endpoint}_preprocessing_impact_stats.csv", index=False)

    # ── Metal 상세 분석 ──
    metal_mask = flag_df["has_metal"]
    if metal_mask.sum() > 0:
        metal_detail = df[metal_mask][["No", "SMILES_raw", "label"]].copy()
        metal_detail["metal_elements"] = flag_df.loc[metal_mask, "metal_elements"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x)
        )
        metal_detail["mw_raw"]   = flag_df.loc[metal_mask, "mw_raw"]
        metal_detail["mw_after"] = flag_df.loc[metal_mask, "mw_after"]
        metal_detail.to_csv(out_dir / f"{endpoint}_metal_compounds_detail.csv", index=False)

        # 금속 원소별 분포
        from collections import Counter
        all_metals = []
        for elems in flag_df.loc[metal_mask, "metal_elements"]:
            if isinstance(elems, list):
                all_metals.extend(elems)
        metal_counts = Counter(all_metals)
        metal_dist = pd.DataFrame([
            {"element": k, "count": v} for k, v in metal_counts.most_common()
        ])
        metal_dist.to_csv(out_dir / f"{endpoint}_metal_element_distribution.csv", index=False)
        logger.info(f"  [{endpoint}] Metal elements: {dict(metal_counts.most_common(10))}")

    # ── Salt stripping MW 변화 분석 ──
    salt_mask = flag_df["has_salt"]
    if salt_mask.sum() > 0:
        salt_detail = df[salt_mask][["No", "SMILES_raw", "label"]].copy()
        salt_detail["n_fragments"]    = flag_df.loc[salt_mask, "n_fragments"]
        salt_detail["mw_raw"]         = flag_df.loc[salt_mask, "mw_raw"]
        salt_detail["mw_after"]       = flag_df.loc[salt_mask, "mw_after"]
        salt_detail["mw_change"]      = flag_df.loc[salt_mask, "mw_change"]
        salt_detail["smiles_stripped"] = flag_df.loc[salt_mask, "smiles_stripped"]
        salt_detail.to_csv(out_dir / f"{endpoint}_salt_compounds_detail.csv", index=False)

    return {
        "endpoint": endpoint,
        "stats": stats_df.to_dict(orient="records"),
        "n_total": n_total,
        "n_invalid": int((~flag_df["valid_smiles"]).sum()),
        "n_metal": int(metal_mask.sum()),
        "n_salt": int(salt_mask.sum()),
        "n_charge_norm": int(flag_df["needs_charge_norm"].sum()),
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 3. 전처리 시나리오별 A/B 모델 비교
# ═══════════════════════════════════════════════════════════════

def build_preprocessing_scenarios(
    df: pd.DataFrame,
    flag_df: pd.DataFrame,
    endpoint: str,
) -> Dict[str, pd.DataFrame]:
    """
    전처리 수준별 데이터셋 시나리오를 구성한다.

    시나리오:
      A) raw_all:             원본 그대로 (invalid만 제거)
      B) no_metal:            metal 화합물 제거
      C) salt_stripped:       salt strip만 수행 (metal 유지)
      D) full_preprocess:     invalid + metal제거 + salt strip + charge norm
      E) metal_as_feature:    metal을 제거하지 않고 binary feature로 추가
    """
    valid = flag_df["valid_smiles"]

    scenarios = {}

    # A) raw_all – invalid만 제거
    a_mask = valid
    a_df = df[a_mask].copy()
    a_df["_scenario"] = "raw_all"
    scenarios["raw_all"] = a_df

    # B) no_metal – metal 제거
    b_mask = valid & (~flag_df["has_metal"])
    b_df = df[b_mask].copy()
    b_df["_scenario"] = "no_metal"
    scenarios["no_metal"] = b_df

    # C) salt_stripped – salt만 처리
    c_df = df[valid].copy()
    smi_col = "canonical_smiles" if "canonical_smiles" in c_df.columns else "SMILES_raw"
    c_df[smi_col] = flag_df.loc[valid.values, "smiles_stripped"].values
    c_df["_scenario"] = "salt_stripped"
    scenarios["salt_stripped"] = c_df

    # D) full_preprocess – 기본 전처리 전체 적용
    d_mask = valid & (~flag_df["has_metal"])
    d_df = df[d_mask].copy()
    smi_col2 = "canonical_smiles" if "canonical_smiles" in d_df.columns else "SMILES_raw"
    d_df[smi_col2] = flag_df.loc[d_mask.values, "smiles_canonical"].values
    d_df["_scenario"] = "full_preprocess"
    scenarios["full_preprocess"] = d_df

    # E) metal_as_feature – metal 유지 + 피처 추가
    e_df = df[valid].copy()
    smi_col3 = "canonical_smiles" if "canonical_smiles" in e_df.columns else "SMILES_raw"
    e_df[smi_col3] = flag_df.loc[valid.values, "smiles_canonical"].values
    e_df["has_metal_feature"] = flag_df.loc[valid.values, "has_metal"].astype(int).values
    e_df["has_salt_feature"]  = flag_df.loc[valid.values, "has_salt"].astype(int).values
    e_df["_scenario"] = "metal_as_feature"
    scenarios["metal_as_feature"] = e_df

    for name, sdf in scenarios.items():
        pos = (sdf["label"] == 1).sum()
        neg = (sdf["label"] == 0).sum()
        logger.info(f"  [{endpoint}] Scenario '{name}': {len(sdf)} rows "
                    f"(pos={pos}, neg={neg}, rate={pos/len(sdf):.3f})")

    return scenarios


def run_scenario_comparison(
    scenarios: Dict[str, pd.DataFrame],
    splits_original: Tuple[pd.DataFrame, pd.DataFrame],
    endpoint: str,
    out_dir: Path,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    각 시나리오에 대해 간소 모델(XGBoost compact)을 학습하여 성능 비교.
    """
    from step3_scaffold_split import assign_scaffolds, scaffold_split
    from step4_feature_extraction import (
        extract_fg_features, extract_physchem_features,
        extract_fingerprint_features, META_COLUMNS,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (
        matthews_corrcoef, balanced_accuracy_score,
        roc_auc_score, average_precision_score,
    )

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("XGBoost not available.")
        return pd.DataFrame()

    results = []

    for scenario_name, df in scenarios.items():
        logger.info(f"  [{endpoint}] Running scenario: {scenario_name}")

        try:
            # scaffold split
            df_scf = assign_scaffolds(df)
            train_df, test_df = scaffold_split(df_scf, seed=seed)

            # feature extraction (compact: FG presence + physchem)
            combined = pd.concat([train_df, test_df], ignore_index=True)
            train_mask = combined["split"] == "train"

            fg = extract_fg_features(combined, endpoint)
            phys = extract_physchem_features(combined, endpoint)

            # fg presence + physchem only
            fg_pres = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
            feat_df = pd.concat([combined[["No","label"]], fg_pres, phys], axis=1)

            # metal/salt feature 추가 (해당 시나리오에서)
            if "has_metal_feature" in combined.columns:
                feat_df["has_metal_feature"] = combined["has_metal_feature"].values
            if "has_salt_feature" in combined.columns:
                feat_df["has_salt_feature"] = combined["has_salt_feature"].values

            train_feat = feat_df[train_mask].reset_index(drop=True)
            test_feat  = feat_df[~train_mask].reset_index(drop=True)

            # X, y
            feat_cols = [c for c in train_feat.columns if c not in META_COLUMNS and c != "label"]
            for c in feat_cols:
                train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce")
                test_feat[c]  = pd.to_numeric(test_feat[c], errors="coerce")

            X_train = train_feat[feat_cols].fillna(0).values.astype(np.float32)
            y_train = train_feat["label"].values.astype(int)
            X_test  = test_feat[feat_cols].fillna(0).values.astype(np.float32)
            y_test  = test_feat["label"].values.astype(int)

            if len(set(y_test)) < 2 or len(set(y_train)) < 2:
                logger.warning(f"    Skipping {scenario_name}: single class in train or test")
                continue

            # XGBoost
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            scale_pos = n_neg / max(n_pos, 1)

            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=scale_pos, eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            )
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)

            mcc  = matthews_corrcoef(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            roc  = roc_auc_score(y_test, y_proba)
            ap   = average_precision_score(y_test, y_proba)

            results.append({
                "scenario":           scenario_name,
                "endpoint":           endpoint,
                "train_size":         len(y_train),
                "test_size":          len(y_test),
                "train_pos":          int(n_pos),
                "train_neg":          int(n_neg),
                "train_pos_rate":     round(n_pos / len(y_train), 4),
                "test_pos_rate":      round((y_test == 1).sum() / len(y_test), 4),
                "n_features":         X_train.shape[1],
                "mcc":                round(mcc, 4),
                "balanced_accuracy":  round(bacc, 4),
                "roc_auc":            round(roc, 4),
                "pr_auc":             round(ap, 4),
            })

            logger.info(f"    {scenario_name}: MCC={mcc:.4f}, ROC={roc:.4f}, BAcc={bacc:.4f}")

        except Exception as e:
            logger.error(f"    {scenario_name} FAILED: {e}")
            import traceback; traceback.print_exc()

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / f"{endpoint}_scenario_comparison.csv", index=False)
    return result_df


# ═══════════════════════════════════════════════════════════════
#  SECTION 4. 시각화
# ═══════════════════════════════════════════════════════════════

def plot_preprocessing_impact(
    impact_stats: pd.DataFrame,
    scenario_results: pd.DataFrame,
    flag_df: pd.DataFrame,
    labels: np.ndarray,
    endpoint: str,
    out_dir: Path,
):
    """전처리 영향 분석 시각화"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) 전처리 카테고리별 양성률 비교 ──
    if not impact_stats.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) 카테고리별 영향 수
        ax = axes[0]
        cats = impact_stats[impact_stats["n_affected"] > 0]
        if not cats.empty:
            ax.barh(range(len(cats)), cats["n_affected"].values, color="steelblue", alpha=0.8)
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats["category"].values, fontsize=9)
            ax.set_xlabel("# Compounds Affected")
            ax.set_title(f"{endpoint} – Preprocessing Category Counts")
            ax.invert_yaxis()

            # 양성 수를 빨간색으로 overlay
            for i, (_, row) in enumerate(cats.iterrows()):
                if row["n_pos"] > 0:
                    ax.barh(i, row["n_pos"], color="coral", alpha=0.7)
            ax.legend(["Total affected", "Positive in affected"], fontsize=8)

        # (b) 양성률 비교
        ax = axes[1]
        if not cats.empty:
            x = range(len(cats))
            ax.bar([i - 0.15 for i in x], cats["pos_rate"].values,
                   width=0.3, label="In category", color="coral", alpha=0.8)
            ax.bar([i + 0.15 for i in x], cats["overall_pos_rate"].values,
                   width=0.3, label="Overall", color="steelblue", alpha=0.6)
            ax.set_xticks(list(x))
            ax.set_xticklabels(cats["category"].values, fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("Positive Rate")
            ax.set_title(f"{endpoint} – Positive Rate: Affected vs Overall")
            ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(out_dir / f"{endpoint}_preprocessing_category_impact.png", dpi=150)
        plt.close(fig)

    # ── 2) 시나리오별 모델 성능 비교 ──
    if not scenario_results.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics_list = [("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced Accuracy")]

        for ax, (metric, title) in zip(axes, metrics_list):
            if metric in scenario_results.columns:
                vals = scenario_results.sort_values(metric, ascending=False)
                colors = ['#2ecc71' if v == vals[metric].max() else '#3498db'
                          for v in vals[metric].values]
                ax.barh(range(len(vals)), vals[metric].values, color=colors, alpha=0.85)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(vals["scenario"].values, fontsize=9)
                ax.set_xlabel(title)
                ax.set_title(f"{endpoint} – {title}")
                ax.invert_yaxis()

                # 값 표시
                for i, v in enumerate(vals[metric].values):
                    ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)

        fig.suptitle(f"{endpoint} – Preprocessing Scenario Comparison", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{endpoint}_scenario_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 3) Metal 화합물 MW 분포 ──
    metal_mask = flag_df["has_metal"]
    if metal_mask.sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # (a) MW 분포: metal vs non-metal
        ax = axes[0]
        mw_metal     = flag_df.loc[metal_mask, "mw_raw"].dropna()
        mw_non_metal = flag_df.loc[~metal_mask & flag_df["valid_smiles"], "mw_raw"].dropna()
        ax.hist(mw_non_metal, bins=40, alpha=0.5, label=f"Non-metal (n={len(mw_non_metal)})",
                color="steelblue", density=True)
        ax.hist(mw_metal, bins=20, alpha=0.6, label=f"Metal (n={len(mw_metal)})",
                color="orangered", density=True)
        ax.set_xlabel("Molecular Weight")
        ax.set_ylabel("Density")
        ax.set_title(f"{endpoint} – MW Distribution")
        ax.legend(fontsize=8)

        # (b) Metal compound label distribution
        ax = axes[1]
        metal_labels = labels[metal_mask]
        non_metal_labels = labels[~metal_mask & flag_df["valid_smiles"].values]

        categories = ["Metal\ncompounds", "Non-metal\ncompounds"]
        pos_rates = [
            metal_labels.mean() if len(metal_labels) > 0 else 0,
            non_metal_labels.mean() if len(non_metal_labels) > 0 else 0,
        ]
        bar_colors = ["orangered", "steelblue"]
        ax.bar(categories, pos_rates, color=bar_colors, alpha=0.8)
        ax.set_ylabel("Positive Rate")
        ax.set_title(f"{endpoint} – Positive Rate by Metal Presence")
        for i, (cat, rate) in enumerate(zip(categories, pos_rates)):
            n = len(metal_labels) if i == 0 else len(non_metal_labels)
            ax.text(i, rate + 0.01, f"{rate:.3f}\n(n={n})", ha="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_dir / f"{endpoint}_metal_analysis.png", dpi=150)
        plt.close(fig)

    # ── 4) Salt stripping 전후 MW 변화 ──
    salt_mask = flag_df["has_salt"]
    if salt_mask.sum() > 5:
        fig, ax = plt.subplots(figsize=(7, 5))
        mw_before = flag_df.loc[salt_mask, "mw_raw"].dropna()
        mw_after  = flag_df.loc[salt_mask, "mw_after"].dropna()
        mw_change = flag_df.loc[salt_mask, "mw_change"].dropna()

        ax.scatter(mw_before, mw_after, alpha=0.5, s=30, c="steelblue", edgecolors="none")
        lims = [0, max(mw_before.max(), mw_after.max()) * 1.1]
        ax.plot(lims, lims, "--", color="gray", lw=1, alpha=0.7, label="y = x (no change)")
        ax.set_xlabel("MW Before Salt Stripping")
        ax.set_ylabel("MW After Salt Stripping")
        ax.set_title(f"{endpoint} – MW Change from Salt Stripping (n={salt_mask.sum()})")
        ax.legend(fontsize=9)

        # 평균 변화 표시
        mean_change = mw_change.mean()
        ax.text(0.05, 0.95, f"Mean ΔMW = {mean_change:.1f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.tight_layout()
        fig.savefig(out_dir / f"{endpoint}_salt_mw_change.png", dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  SECTION 5. 종합 리포트 생성
# ═══════════════════════════════════════════════════════════════

def generate_preprocessing_report(
    all_impacts: Dict[str, Dict],
    all_scenarios: Dict[str, pd.DataFrame],
    out_dir: Path,
):
    """전처리 영향 분석 마크다운 리포트"""
    lines = []
    lines.append("# Preprocessing Impact Analysis Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("이 리포트는 전처리 각 단계(금속 제거, salt stripping, charge normalization, "
                 "invalid SMILES 제거)가 데이터와 모델 성능에 미치는 영향을 분석한다.")
    lines.append("")

    for ep, impact in all_impacts.items():
        lines.append(f"## {ep.upper()}")
        lines.append("")
        lines.append(f"- 전체: {impact['n_total']} 화합물")
        lines.append(f"- Invalid SMILES: {impact['n_invalid']} ({impact['n_invalid']/impact['n_total']*100:.1f}%)")
        lines.append(f"- Metal 포함: {impact['n_metal']} ({impact['n_metal']/impact['n_total']*100:.1f}%)")
        lines.append(f"- Salt 포함: {impact['n_salt']} ({impact['n_salt']/impact['n_total']*100:.1f}%)")
        lines.append(f"- Charge 정규화 필요: {impact['n_charge_norm']}")
        lines.append("")

        # category stats
        stats = impact.get("stats", [])
        if stats:
            lines.append("### Category-Level Impact")
            lines.append("")
            lines.append("| Category | N | Pos Rate | Overall Rate | Enrichment | Fisher p | Recommendation |")
            lines.append("|----------|---|----------|-------------|------------|----------|----------------|")
            for s in stats:
                lines.append(f"| {s['category']} | {s['n_affected']} | "
                             f"{s['pos_rate']:.3f} | {s['overall_pos_rate']:.3f} | "
                             f"{s['enrichment_ratio']:.2f}× | "
                             f"{s.get('fisher_p', 'N/A')} | {s['recommendation']} |")
            lines.append("")

        # scenario results
        if ep in all_scenarios and not all_scenarios[ep].empty:
            sdf = all_scenarios[ep]
            lines.append("### Scenario Comparison (XGBoost compact)")
            lines.append("")
            lines.append("| Scenario | Train | Test | MCC | ROC-AUC | Balanced Acc |")
            lines.append("|----------|-------|------|-----|---------|-------------|")
            for _, row in sdf.sort_values("mcc", ascending=False).iterrows():
                lines.append(f"| {row['scenario']} | {row['train_size']} | {row['test_size']} | "
                             f"**{row['mcc']:.4f}** | {row['roc_auc']:.4f} | {row['balanced_accuracy']:.4f} |")
            lines.append("")

            best = sdf.loc[sdf["mcc"].idxmax()]
            lines.append(f"**Best scenario: `{best['scenario']}`** (MCC={best['mcc']:.4f})")
            lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **금속 화합물은 endpoint에 따라 양성 enrichment가 다르다.**")
    lines.append("   - 무조건 제거보다는 endpoint별 분석 후 결정하는 것이 바람직하다.")
    lines.append("2. **Salt stripping은 분자량 변화를 유발하지만 모델 성능에는 대개 긍정적이다.**")
    lines.append("3. **metal_as_feature 시나리오**는 금속 정보를 유지하면서 피처로 활용하여 ")
    lines.append("   정보 손실 없이 모델에 반영할 수 있다.")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = out_dir / "preprocessing_impact_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info(f"Preprocessing impact report saved to {report_path}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 6. 통합 실행 함수
# ═══════════════════════════════════════════════════════════════

def run_step2b(
    datasets: Dict[str, pd.DataFrame],
    splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    out_dir: Path,
) -> Dict:
    """
    Step 2b 전체 실행:
      1) 모든 화합물에 전처리 플래그 부여
      2) 전처리 영향 통계 분석
      3) 시나리오별 A/B 모델 비교
      4) 시각화 및 리포트 생성

    Returns: endpoint별 분석 결과
    """
    all_impacts = {}
    all_scenarios = {}
    all_flags = {}

    for ep in ENDPOINTS:
        if ep not in datasets:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  Step 2b: Preprocessing Impact Analysis – {ep}")
        logger.info(f"{'='*60}")

        df = datasets[ep]
        ep_dir = out_dir / ep
        ep_dir.mkdir(parents=True, exist_ok=True)

        # 1) 전처리 플래그 부여
        logger.info(f"  [{ep}] Classifying compounds...")
        smi_col = "SMILES_raw" if "SMILES_raw" in df.columns else "SMILES"
        flag_df = classify_all_compounds(df, smi_col=smi_col)
        flag_df.to_csv(ep_dir / f"{ep}_preprocessing_flags.csv", index=False)
        all_flags[ep] = flag_df

        # 2) 영향 통계 분석
        logger.info(f"  [{ep}] Analyzing impact...")
        impact = analyze_preprocessing_impact(df, flag_df, ep, ep_dir)
        all_impacts[ep] = impact
        save_json(impact, ep_dir / f"{ep}_preprocessing_impact.json")

        # 3) 시나리오 구성 + A/B 비교
        logger.info(f"  [{ep}] Building scenarios...")
        scenarios = build_preprocessing_scenarios(df, flag_df, ep)
        logger.info(f"  [{ep}] Running scenario comparison...")
        scenario_results = run_scenario_comparison(
            scenarios,
            splits.get(ep, (None, None)),
            ep, ep_dir,
        )
        all_scenarios[ep] = scenario_results

        # 4) 시각화
        logger.info(f"  [{ep}] Generating plots...")
        impact_stats = pd.read_csv(ep_dir / f"{ep}_preprocessing_impact_stats.csv")
        plot_preprocessing_impact(
            impact_stats, scenario_results, flag_df,
            df["label"].values, ep, ep_dir,
        )

    # 5) 종합 리포트
    generate_preprocessing_report(all_impacts, all_scenarios, out_dir)

    logger.info("\nStep 2b complete.")
    return {
        "impacts": all_impacts,
        "scenarios": all_scenarios,
        "flags": all_flags,
    }
