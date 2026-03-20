"""
config.py – 유전독성 예측모델 파이프라인 전역 설정
==================================================
모든 경로, 시드, 하이퍼파라미터 범위, 컬럼 매핑 등을 한 곳에서 관리한다.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# ──────────────────────────────────────────────
#  프로젝트 루트 – 환경변수 또는 기본값
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get(
    "GENOTOX_ROOT",
    Path(__file__).resolve().parent   # 스크립트가 위치한 폴더
))
DATA_DIR   = PROJECT_ROOT / "data"
RUNS_DIR   = PROJECT_ROOT / "runs"
LATEST_TXT = PROJECT_ROOT / "LATEST_RUN_PATH.txt"

# ──────────────────────────────────────────────
#  글로벌 시드
# ──────────────────────────────────────────────
GLOBAL_SEED = 42

# ──────────────────────────────────────────────
#  Endpoint 정의
# ──────────────────────────────────────────────
ENDPOINTS = ["ames", "invitro", "invivo"]

ENDPOINT_DISPLAY = {
    "ames":    "Ames",
    "invitro": "In vitro chromosome aberration",
    "invivo":  "In vivo micronucleus",
}

# ──────────────────────────────────────────────
#  컬럼 매핑 후보 → canonical 컬럼명
# ──────────────────────────────────────────────
CANONICAL_COLUMNS = [
    "No", "label", "SMILES_raw", "canonical_smiles",
    "standardized_smiles", "murcko_scaffold",
    "scaffold_group", "scaffold_group_type",
    "endpoint", "source_dataset",
]

COL_MAP_ID     = ["No", "id", "compound_id", "ID"]
COL_MAP_SMILES = ["SMILES", "canonical_smiles", "standardized_smiles",
                  "analysis_smiles", "smiles"]
COL_MAP_LABEL  = ["label", "target", "y", "class", "Label", "TARGET"]
COL_MAP_SCAFFOLD = ["murcko_scaffold", "scaffold_group",
                    "scaffold_group_type", "murcko_scaffold_y"]

# ──────────────────────────────────────────────
#  파일 탐색 후보
# ──────────────────────────────────────────────
RAW_FILE_CANDIDATES = {
    "ames":    ["ames_combine.xlsx", "ames_combine.csv", "ames_pre.csv"],
    "invitro": ["invitro_pre.csv", "invitro_combine.csv"],
    "invivo":  ["invivo_pre.csv", "invivo_combine.csv"],
}

# ──────────────────────────────────────────────
#  Scaffold split
# ──────────────────────────────────────────────
SPLIT_RATIO = 0.8        # train 비율
SPLIT_SEED  = GLOBAL_SEED

# ──────────────────────────────────────────────
#  Fingerprint
# ──────────────────────────────────────────────
FP_RADIUS = 2
FP_NBITS  = 256
FP_MODES  = ["all", "selected"]
FP_TOPK_CANDIDATES = [32, 64, 128]

# ──────────────────────────────────────────────
#  Under-sampling 전략
# ──────────────────────────────────────────────
UNDERSAMPLING_STRATEGIES = [
    "none",
    "coverage_under",
    "hard_negative_enriched",
    "hybrid_negative_pool",
]

# ──────────────────────────────────────────────
#  모델 – Hyperparameter 탐색 공간
# ──────────────────────────────────────────────
XGB_PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0, 0.1, 0.3, 0.5],
    "reg_alpha":        [0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

LOGISTIC_PARAM_DIST = {
    "C":           [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "l1_ratio":    [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "max_iter":    [2000],
    "penalty":     ["elasticnet"],
    "solver":      ["saga"],
}

N_RANDOM_SEARCH_ITER = 30
CV_FOLDS = 5

# ──────────────────────────────────────────────
#  Threshold tuning
# ──────────────────────────────────────────────
SPECIFICITY_FLOOR = {
    "ames":    0.70,
    "invitro": 0.75,
    "invivo":  0.80,
}

# ──────────────────────────────────────────────
#  Tuning metric
# ──────────────────────────────────────────────
TUNING_METRIC = {
    "ames":    "average_precision",
    "invitro": "average_precision",
    "invivo":  "average_precision",
}

# ──────────────────────────────────────────────
#  Feature modes (실험군)
# ──────────────────────────────────────────────
FEATURE_MODES = ["compact", "broad_fp"]  # broad_tabular removed: not in main execution path

# ──────────────────────────────────────────────
#  Resampling strategies
# ──────────────────────────────────────────────
RESAMPLING_STRATEGIES = [
    "none",
    "class_weight_only",
    "coverage_under",
    "hybrid_negative_pool",
]

# ──────────────────────────────────────────────
#  Functional group SMARTS 정의
# ──────────────────────────────────────────────
FG_SMARTS = {
    # --- common FG ---
    "alcohol":                "[OX2H]",
    "ether":                  "[OD2]([#6])[#6]",
    "aldehyde":               "[CX3H1](=O)[#6]",
    "ketone":                 "[#6][CX3](=O)[#6]",
    "carboxylic_acid":        "[CX3](=O)[OX2H1]",
    "ester":                  "[#6][CX3](=O)[OX2H0][#6]",
    "amide":                  "[NX3][CX3](=[OX1])[#6]",
    "primary_amine":          "[NX3;H2;!$(NC=O)]",
    "secondary_amine":        "[NX3;H1;!$(NC=O)]",
    "tertiary_amine":         "[NX3;H0;!$(NC=O);!$(N=*)]",
    "nitro":                  "[NX3](=O)=O",
    "nitroso":                "[NX2]=O",
    "halide_F":               "[F]",
    "halide_Cl":              "[Cl]",
    "halide_Br":              "[Br]",
    "halide_I":               "[I]",
    "sulfide":                "[#16X2]([#6])[#6]",
    "sulfoxide":              "[#16X3](=[OX1])([#6])[#6]",
    "sulfone":                "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "phosphate":              "[PX4](=O)([OX2])([OX2])[OX2]",
    "cyano":                  "[CX2]#[NX1]",
    # --- genotox-relevant alerts ---
    "epoxide":                "C1OC1",
    "nitro_aromatic":         "[cR1][NX3](=O)=O",
    "aromatic_amine":         "[cR1][NX3;H2]",
    "aniline":                "c1ccc(N)cc1",
    "aromatic_amine_sec":     "[cR1][NX3;H1;!$(NC=O)]",
    "n_nitroso":              "[NX3][NX2]=O",
    "hydrazine_like":         "[NX3][NX3]",
    "azo":                    "[NX2]=[NX2]",
    "organometal_like":       "[#6]~[#50,#33,#51,#82,#80,#48,#24,#29]",
    "alpha_beta_unsat_carb":  "[CX3](=O)[CX3]=[CX3]",
    "acyl_halide":            "[CX3](=[OX1])[F,Cl,Br,I]",
    "michael_acceptor":       "[CX3](=[OX1])/[CX3]=[CX3]",
    "azide":                  "[NX1]~[NX2]~[NX1]",
    "isocyanate":             "[NX2]=C=O",
    "sulfonyl_halide":        "[#16X4](=[OX1])(=[OX1])[F,Cl,Br,I]",
}

# ──────────────────────────────────────────────
#  High-confidence alert families
# ──────────────────────────────────────────────
ALERT_FAMILIES = {
    "ames": {
        "positive": ["n_nitroso", "nitro_aromatic", "epoxide", "hydrazine_like"],
        "negative": [],
    },
    "invitro": {
        "positive": ["aromatic_amine", "aniline", "aromatic_amine_sec",
                      "alpha_beta_unsat_carb"],
        "negative": [],
    },
    "invivo": {
        "positive": ["epoxide", "organometal_like"],
        "negative": [],
    },
}

# ──────────────────────────────────────────────
#  유틸리티
# ──────────────────────────────────────────────
def make_run_dir(tag: str = "") -> Path:
    """타임스탬프 기반 실행 디렉토리 생성"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    LATEST_TXT.write_text(str(run_dir))
    return run_dir


def save_json(obj, path: Path):
    """JSON 저장 헬퍼"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
