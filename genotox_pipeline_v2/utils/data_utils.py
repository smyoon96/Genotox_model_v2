"""
utils/data_utils.py — 데이터 로딩 / 전처리 유틸리티
====================================================
step1, step2, step5, step6, step8에서 공통으로 사용.
"""
import sys, hashlib, logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

logger = logging.getLogger("data_utils")


# ──────────────────────────────────────────────
#  파일 탐색
# ──────────────────────────────────────────────

def discover_split_files(endpoint: str) -> Tuple[Optional[Path], Optional[Path]]:
    """endpoint의 train/test 파일 경로 반환. 없으면 (None, None)."""
    candidates = cfg.RAW_FILE_CANDIDATES.get(endpoint, [f"{endpoint}.csv"])
    for fname in candidates:
        for ext in ["", ".csv", ".xlsx"]:
            base = fname.replace(".csv", "").replace(".xlsx", "")
            for suffix in [".csv", ".xlsx"]:
                p = cfg.DATA_DIR / (base + suffix)
                if p.exists():
                    # train/test 분리 파일 탐색
                    tr = cfg.DATA_DIR / f"{base}_train{suffix}"
                    te = cfg.DATA_DIR / f"{base}_test{suffix}"
                    if tr.exists() and te.exists():
                        return tr, te
                    # 단일 파일이면 None 반환 (step1 감사용으로 단일 파일 경로 반환)
                    return p, p
    return None, None


def ensure_fg_dir() -> Optional[Path]:
    """fg_descriptor_analysis 디렉토리 탐색. 없으면 None."""
    candidates = [
        cfg.DATA_DIR / "fg_descriptor_analysis",
        cfg.PROJECT_ROOT / "fg_descriptor_analysis",
        cfg.DATA_DIR / "fg_analysis",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            return d
    return None


def find_fg_files(fg_dir: Path, endpoint: str) -> Dict:
    """endpoint용 FG 분석 파일 탐색."""
    result = {"canonical": None, "views": None}
    if fg_dir is None:
        return result
    for fname in fg_dir.glob(f"*{endpoint}*"):
        if "canonical" in fname.name.lower() or "analysis" in fname.name.lower():
            result["canonical"] = fname
        elif "view" in fname.name.lower() or "preprocess" in fname.name.lower():
            result["views"] = fname
    # fallback: 첫 번째 파일
    all_files = list(fg_dir.glob(f"*{endpoint}*"))
    if all_files and result["canonical"] is None:
        result["canonical"] = all_files[0]
    return result


# ──────────────────────────────────────────────
#  파일 읽기
# ──────────────────────────────────────────────

def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """CSV / Excel 안전 읽기. encoding 오류 자동 처리."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl", **kwargs)
    for enc in ["utf-8-sig", "utf-8", "cp949", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"Cannot read file: {path}")


# ──────────────────────────────────────────────
#  SMILES 컬럼 통일
# ──────────────────────────────────────────────

def unify_smiles_col(df: pd.DataFrame) -> pd.DataFrame:
    """여러 SMILES 컬럼명 후보를 'canonical_smiles'로 통일."""
    df = df.copy()
    for alias in cfg.COL_MAP_SMILES:
        if alias in df.columns and alias != "canonical_smiles":
            if "canonical_smiles" not in df.columns:
                df = df.rename(columns={alias: "canonical_smiles"})
            break
    return df


# ──────────────────────────────────────────────
#  Merge 후처리
# ──────────────────────────────────────────────

def normalize_merge_key(series: pd.Series) -> pd.Series:
    """merge key 정규화: 문자열 변환 + strip."""
    return series.astype(str).str.strip()


def clean_merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """merge 후 생긴 _x/_y 중복 컬럼 정리."""
    df = df.copy()
    for col in df.columns:
        if col.endswith("_x"):
            base = col[:-2]
            col_y = base + "_y"
            if col_y in df.columns:
                # _x 우선 유지, _y 제거
                df = df.drop(columns=[col_y])
            df = df.rename(columns={col: base})
    # _ca, _vw suffix 제거
    rename_map = {}
    for col in df.columns:
        for suf in ("_ca", "_vw"):
            if col.endswith(suf):
                base = col[:-len(suf)]
                if base not in df.columns:
                    rename_map[col] = base
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def safe_bool_mask(series: pd.Series) -> pd.Series:
    """'valid', 'true', '1' 등을 bool로 변환."""
    if series.dtype == bool:
        return series
    s = series.astype(str).str.lower().str.strip()
    return s.isin(["true", "1", "yes", "valid", "t"])


# ──────────────────────────────────────────────
#  컬럼 분류
# ──────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    df 컬럼을 meta / features / excluded 로 분류.
    """
    meta, features, excluded = [], [], []
    for col in df.columns:
        if col in cfg.META_COLS:
            meta.append(col)
        elif col.startswith("fp_") or col.endswith("_present") or col.startswith("bb_"):
            features.append(col)
        elif col.endswith("_x") or col.endswith("_y") or col.startswith("__"):
            excluded.append(col)
        elif df[col].dtype == "object" and df[col].nunique() > 100:
            excluded.append(col)   # 고 카디널리티 문자열
        elif col.startswith(cfg.QM_PREFIX) or col.startswith(cfg.EXT_QM_PREFIX):
            features.append(col)
        else:
            # 나머지 numeric → features 후보
            try:
                pd.to_numeric(df[col])
                features.append(col)
            except (ValueError, TypeError):
                excluded.append(col)
    return {"meta": meta, "features": features, "excluded": excluded}


# ──────────────────────────────────────────────
#  broadfp dirty check
# ──────────────────────────────────────────────

def check_broadfp_dirty(path: Path) -> bool:
    """broadfp 파일이 없거나 비어 있으면 True (= 재생성 필요)."""
    path = Path(path)
    if not path.exists():
        return True
    if path.stat().st_size < 100:
        return True
    try:
        df = pd.read_csv(path, nrows=2, low_memory=False)
        return len(df.columns) < 5
    except Exception:
        return True
