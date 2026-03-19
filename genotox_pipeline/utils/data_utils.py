"""
data_utils.py — 데이터 탐색 · 로딩 · 스키마 정규화
==================================================
"""
import os, re, glob, zipfile, logging, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# 1. Merge key 정규화
# ─────────────────────────────────────────────────────────────────────

def normalize_merge_key(series: pd.Series) -> pd.Series:
    """
    No 컬럼을 robust하게 정규화: 123, 123.0, '123' → '123'
    """
    def _norm(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        # float → int string
        if re.match(r'^-?\d+\.0+$', s):
            s = s.split('.')[0]
        return s
    return series.map(_norm)


# ─────────────────────────────────────────────────────────────────────
# 2. 파일 자동 탐색
# ─────────────────────────────────────────────────────────────────────

def _search_patterns(data_dir: Path, stem: str):
    """endpoint stem에 맞는 train/test CSV를 우선순위로 탐색"""
    candidates = [
        # 우선순위 a
        (f"{stem}_recommended_union_train.csv", f"{stem}_recommended_union_test.csv"),
        # 우선순위 b
        (f"{stem}_train_set_scaffold80_20.csv", f"{stem}_test_set_scaffold80_20.csv"),
    ]
    for trn, tst in candidates:
        tp = data_dir / trn
        ep = data_dir / tst
        if tp.exists() and ep.exists():
            return tp, ep
    # glob fallback
    for trn, tst in candidates:
        trn_g = list(data_dir.glob(f"*{stem}*train*.csv"))
        tst_g = list(data_dir.glob(f"*{stem}*test*.csv"))
        if trn_g and tst_g:
            return trn_g[0], tst_g[0]
    return None, None


def discover_split_files(endpoint: str) -> Tuple[Optional[Path], Optional[Path]]:
    """endpoint별 train/test 파일을 자동 탐색"""
    ep_cfg = cfg.ENDPOINTS[endpoint]
    raw_stem = ep_cfg["raw_stem"]
    data_dir = cfg.DATA_DIR

    # 여러 stem 변형을 시도
    for stem in [raw_stem, ep_cfg.get("fg_stem", raw_stem)]:
        tp, ep = _search_patterns(data_dir, stem)
        if tp:
            logger.info(f"[{endpoint}] train={tp.name}, test={ep.name}")
            return tp, ep

    # 더 넓은 검색
    all_csv = list(data_dir.glob("*.csv"))
    train_files = [f for f in all_csv if raw_stem in f.stem.lower() and "train" in f.stem.lower()]
    test_files  = [f for f in all_csv if raw_stem in f.stem.lower() and "test" in f.stem.lower()]
    if train_files and test_files:
        logger.info(f"[{endpoint}] fallback: train={train_files[0].name}, test={test_files[0].name}")
        return train_files[0], test_files[0]

    logger.warning(f"[{endpoint}] split files NOT found in {data_dir}")
    return None, None


# ─────────────────────────────────────────────────────────────────────
# 3. fg_descriptor_analysis 탐색
# ─────────────────────────────────────────────────────────────────────

def ensure_fg_dir() -> Optional[Path]:
    """fg_descriptor_analysis 폴더를 확보 (ZIP이면 압축해제)"""
    if cfg.FG_DIR.is_dir():
        return cfg.FG_DIR
    if cfg.FG_ZIP.exists():
        logger.info(f"Extracting {cfg.FG_ZIP} ...")
        with zipfile.ZipFile(cfg.FG_ZIP, 'r') as z:
            z.extractall(cfg.DATA_DIR)
        if cfg.FG_DIR.is_dir():
            return cfg.FG_DIR
        # ZIP 내부 폴더명이 다를 수 있음
        extracted = [d for d in cfg.DATA_DIR.iterdir() if d.is_dir() and "fg_descriptor" in d.name.lower()]
        if extracted:
            return extracted[0]
    logger.warning("fg_descriptor_analysis not found")
    return None


def find_fg_files(fg_dir: Path, endpoint: str) -> Dict[str, Optional[Path]]:
    """endpoint별 canonical_analysis.csv, preprocessed_views.csv 탐색"""
    fg_stem = cfg.ENDPOINTS[endpoint]["fg_stem"]
    result = {"canonical": None, "views": None}

    for f in fg_dir.rglob("*.csv"):
        fname = f.name.lower()
        if fg_stem.lower() in fname or endpoint.lower() in fname:
            if "canonical_analysis" in fname:
                result["canonical"] = f
            elif "preprocessed_views" in fname or "views" in fname:
                result["views"] = f

    # 못 찾으면 더 넓게
    if not result["canonical"]:
        cands = list(fg_dir.rglob("*canonical_analysis*.csv"))
        for c in cands:
            if endpoint in c.name.lower() or fg_stem in c.name.lower():
                result["canonical"] = c
                break

    return result


# ─────────────────────────────────────────────────────────────────────
# 4. 안전한 CSV 읽기
# ─────────────────────────────────────────────────────────────────────

def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """mixed dtype warning 방지 + No 정규화"""
    kwargs.setdefault("low_memory", False)
    df = pd.read_csv(path, **kwargs)
    if "No" in df.columns:
        df["No"] = normalize_merge_key(df["No"])
    return df


# ─────────────────────────────────────────────────────────────────────
# 5. 컬럼 alias 통일
# ─────────────────────────────────────────────────────────────────────

SMILES_ALIASES = ["SMILES", "smiles", "canonical_smiles", "Canonical_SMILES",
                  "standardized_smiles", "Standardized_SMILES", "smi"]

def unify_smiles_col(df: pd.DataFrame) -> pd.DataFrame:
    """SMILES 컬럼 alias를 canonical_smiles로 통일"""
    existing = [c for c in SMILES_ALIASES if c in df.columns]
    if not existing:
        logger.warning("No SMILES column found")
        return df
    primary = existing[0]
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = df[primary]
    return df


def clean_merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """merge 잔여 suffix (_x, _y) 정리"""
    cols_to_drop = []
    base_seen = set()
    for col in df.columns:
        for suf in ("_x", "_y"):
            if col.endswith(suf):
                base = col[:-len(suf)]
                if base in df.columns:
                    cols_to_drop.append(col)
                elif base not in base_seen:
                    df.rename(columns={col: base}, inplace=True)
                    base_seen.add(base)
                else:
                    cols_to_drop.append(col)
    if cols_to_drop:
        logger.info(f"Dropping merge duplicates: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────
# 6. 메타/feature 컬럼 분리
# ─────────────────────────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """메타 컬럼과 feature 후보 컬럼 분리"""
    meta = []
    feature_candidates = []
    excluded = []

    for col in df.columns:
        # 메타 컬럼
        if col in cfg.META_COLS or col.lower() in [m.lower() for m in cfg.META_COLS]:
            meta.append(col)
            continue
        # 제외 패턴
        if any(pat in col for pat in cfg.EXCLUDE_PATTERNS):
            excluded.append(col)
            continue
        # string/object는 feature 후보에서 제외 (scaffold_group_type 제외)
        if df[col].dtype == "object" and col != "scaffold_group_type":
            if df[col].nunique() > 50:  # high cardinality string → 메타
                meta.append(col)
                continue
        feature_candidates.append(col)

    return {"meta": meta, "features": feature_candidates, "excluded": excluded}


# ─────────────────────────────────────────────────────────────────────
# 7. Dirty file 감지
# ─────────────────────────────────────────────────────────────────────

def check_broadfp_dirty(path: Path) -> bool:
    """broadfp CSV가 오염됐는지 간단 체크"""
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path, nrows=5, low_memory=False)
        if "label" not in df.columns:
            return True
        if df.shape[1] < 10:
            return True
        return False
    except Exception as e:
        logger.warning(f"Dirty file detected: {path} → {e}")
        return True
