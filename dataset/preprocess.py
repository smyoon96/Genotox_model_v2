import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# =========================
# 1. 기본 유틸
# =========================
def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def to_canonical_smiles_keep_salts(smiles: str):
    """
    Canonical-only:
    - 염 / 멀티컴포넌트 유지
    - 전체 입력 구조를 canonical SMILES로 변환
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def to_standardized_smiles(smiles: str):
    """
    Standardized:
    - cleanup
    - fragment parent (salt 제거)
    - uncharge
    - canonical SMILES
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    try:
        # 1) 기본 cleanup
        clean_mol = rdMolStandardize.Cleanup(mol)

        # 2) salt / small fragment 제거 -> parent 추출
        parent_mol = rdMolStandardize.FragmentParent(clean_mol)

        # 3) 중성화
        uncharger = rdMolStandardize.Uncharger()
        parent_mol = uncharger.uncharge(parent_mol)

        # 4) canonical smiles
        std_smiles = Chem.MolToSmiles(parent_mol, canonical=True)
        return std_smiles

    except Exception:
        return None


def is_valid_smiles(smiles: str):
    return mol_from_smiles(smiles) is not None


# =========================
# 2. 데이터셋 전처리 함수
# =========================
def build_preprocessing_views(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    label_col: str = "label",
    keep_extra_cols: bool = True,
):
    """
    입력 df에서 Raw / Canonical-only / Standardized 3개 버전을 생성
    """

    work = df.copy()

    # 원본 보존
    work["raw_smiles"] = work[smiles_col].astype(str)

    # 최소 유효성 검사
    work["raw_valid"] = work["raw_smiles"].apply(is_valid_smiles)

    # Canonical-only (salts retained)
    work["canonical_smiles"] = work["raw_smiles"].apply(to_canonical_smiles_keep_salts)

    # Standardized (desalted)
    work["standardized_smiles"] = work["raw_smiles"].apply(to_standardized_smiles)

    # 각 버전 유효성
    work["canonical_valid"] = work["canonical_smiles"].notna()
    work["standardized_valid"] = work["standardized_smiles"].notna()

    # 중복 확인용
    work["canonical_dup"] = work["canonical_smiles"].duplicated(keep=False)
    work["standardized_dup"] = work["standardized_smiles"].duplicated(keep=False)

    # 라벨 충돌 확인용
    canon_conflict = (
        work.dropna(subset=["canonical_smiles"])
        .groupby("canonical_smiles")[label_col]
        .nunique()
        .rename("canonical_label_nunique")
        .reset_index()
    )
    canon_conflict["canonical_label_conflict"] = canon_conflict["canonical_label_nunique"] > 1

    std_conflict = (
        work.dropna(subset=["standardized_smiles"])
        .groupby("standardized_smiles")[label_col]
        .nunique()
        .rename("standardized_label_nunique")
        .reset_index()
    )
    std_conflict["standardized_label_conflict"] = std_conflict["standardized_label_nunique"] > 1

    work = work.merge(canon_conflict, on="canonical_smiles", how="left")
    work = work.merge(std_conflict, on="standardized_smiles", how="left")

    work["canonical_label_conflict"] = work["canonical_label_conflict"].fillna(False)
    work["standardized_label_conflict"] = work["standardized_label_conflict"].fillna(False)

    # 보기 좋은 컬럼 정리
    base_cols = [c for c in df.columns]
    new_cols = [
        "raw_smiles",
        "raw_valid",
        "canonical_smiles",
        "canonical_valid",
        "canonical_dup",
        "canonical_label_conflict",
        "standardized_smiles",
        "standardized_valid",
        "standardized_dup",
        "standardized_label_conflict",
    ]

    if keep_extra_cols:
        result = work[base_cols + [c for c in new_cols if c not in base_cols]]
    else:
        result = work[[smiles_col, label_col] + new_cols]

    return result


# =========================
# 3. 학습용 버전 추출 함수
# =========================
def make_training_view(
    processed_df: pd.DataFrame,
    version: str,
    label_col: str = "label",
    drop_invalid: bool = True,
    drop_duplicates: bool = False,
    drop_conflicts: bool = False,
):
    """
    version:
        - 'raw'
        - 'canonical'
        - 'standardized'
    """

    df = processed_df.copy()

    if version == "raw":
        smiles_col = "raw_smiles"
        valid_col = "raw_valid"
        dup_col = None
        conflict_col = None

    elif version == "canonical":
        smiles_col = "canonical_smiles"
        valid_col = "canonical_valid"
        dup_col = "canonical_dup"
        conflict_col = "canonical_label_conflict"

    elif version == "standardized":
        smiles_col = "standardized_smiles"
        valid_col = "standardized_valid"
        dup_col = "standardized_dup"
        conflict_col = "standardized_label_conflict"

    else:
        raise ValueError("version must be one of: raw, canonical, standardized")

    if drop_invalid:
        df = df[df[valid_col]].copy()

    df = df.rename(columns={smiles_col: "SMILES_MODEL_INPUT"})

    if drop_duplicates and dup_col is not None:
        df = df.drop_duplicates(subset=["SMILES_MODEL_INPUT"])

    if drop_conflicts and conflict_col is not None:
        df = df[~df[conflict_col]].copy()

    df = df.dropna(subset=["SMILES_MODEL_INPUT", label_col]).copy()

    return df


# =========================
# 4. 요약 함수
# =========================
def summarize_preprocessing(processed_df: pd.DataFrame, label_col: str = "label", name: str = "dataset"):
    def safe_pos_ratio(series):
        if len(series) == 0:
            return 0.0
        return float(series.mean())

    summary = {
        "dataset": name,
        "n_total": len(processed_df),
        "raw_valid_n": int(processed_df["raw_valid"].sum()),
        "canonical_valid_n": int(processed_df["canonical_valid"].sum()),
        "standardized_valid_n": int(processed_df["standardized_valid"].sum()),
        "canonical_unique": int(processed_df["canonical_smiles"].nunique(dropna=True)),
        "standardized_unique": int(processed_df["standardized_smiles"].nunique(dropna=True)),
        "canonical_conflicts": int(processed_df["canonical_label_conflict"].sum()),
        "standardized_conflicts": int(processed_df["standardized_label_conflict"].sum()),
        "positive_ratio": safe_pos_ratio(processed_df[label_col]),
    }
    return pd.DataFrame([summary])


# =========================
# 5. 파일별 실행
# =========================
def load_dataset(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def save_all_versions(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(input_path))[0]
    df = load_dataset(input_path)

    processed = build_preprocessing_views(df, smiles_col="SMILES", label_col="label")
    processed.to_csv(os.path.join(output_dir, f"{name}_preprocessing_all.csv"), index=False)

    raw_train = make_training_view(processed, version="raw", drop_invalid=True, drop_duplicates=False, drop_conflicts=False)
    canonical_train = make_training_view(processed, version="canonical", drop_invalid=True, drop_duplicates=False, drop_conflicts=False)
    standardized_train = make_training_view(processed, version="standardized", drop_invalid=True, drop_duplicates=False, drop_conflicts=False)

    raw_train.to_csv(os.path.join(output_dir, f"{name}_raw.csv"), index=False)
    canonical_train.to_csv(os.path.join(output_dir, f"{name}_canonical_only.csv"), index=False)
    standardized_train.to_csv(os.path.join(output_dir, f"{name}_standardized.csv"), index=False)

    summary = summarize_preprocessing(processed, label_col="label", name=name)
    summary.to_csv(os.path.join(output_dir, f"{name}_summary.csv"), index=False)

    return processed, summary


if __name__ == "__main__":
    input_files = [
        "/mnt/data/ames_combine.xlsx",
        "/mnt/data/invitro_pre.csv",
        "/mnt/data/invivo_pre.csv",
    ]

    output_dir = "/mnt/data/preprocessed_outputs"

    all_summaries = []
    for path in input_files:
        processed_df, summary_df = save_all_versions(path, output_dir)
        all_summaries.append(summary_df)

    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_summary.to_csv(os.path.join(output_dir, "all_dataset_summary.csv"), index=False)

    print(final_summary)