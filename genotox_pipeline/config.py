"""
Genotox Pipeline — Global Configuration
========================================
모든 경로, endpoint 설정, 모델/전략 옵션을 한 곳에서 관리한다.
사용자 환경에 맞게 PROJECT_ROOT만 수정하면 동작한다.
"""
import os, json, pathlib, platform

# ── 경로 설정 ──────────────────────────────────────────────────────────
if platform.system() == "Windows":
    PROJECT_ROOT = pathlib.Path(r"C:\Users\Administrator\Documents\GitHub\Genotox_model\model")
else:
    PROJECT_ROOT = pathlib.Path(os.environ.get("GENOTOX_ROOT", "."))

DATA_DIR        = PROJECT_ROOT / "data"
FG_DIR          = DATA_DIR / "fg_descriptor_analysis"
FG_ZIP          = DATA_DIR / "fg_descriptor_analysis.zip"
OUTPUT_DIR      = PROJECT_ROOT / "output"
LOG_DIR         = OUTPUT_DIR / "logs"
ARTIFACT_DIR    = OUTPUT_DIR / "artifacts"
BROADFP_DIR     = OUTPUT_DIR / "broadfp"
SUMMARY_DIR     = OUTPUT_DIR / "summary"

for d in [OUTPUT_DIR, LOG_DIR, ARTIFACT_DIR, BROADFP_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Endpoint 정의 ─────────────────────────────────────────────────────
ENDPOINTS = {
    "ames": {
        "raw_stem": "ames",
        "fg_stem": "ames_combine",
        "label_col": "label",
        "positive_label": 1,
    },
    "invitro": {
        "raw_stem": "invitro",
        "fg_stem": "invitro_pre",
        "label_col": "label",
        "positive_label": 1,
    },
    "invivo": {
        "raw_stem": "invivo",
        "fg_stem": "invivo_pre",
        "label_col": "label",
        "positive_label": 1,
    },
}

# ── 메타 / 제외 컬럼 ──────────────────────────────────────────────────
META_COLS = [
    "No", "label", "scaffold_group", "scaffold_group_type",
    "canonical_smiles", "standardized_smiles", "SMILES",
    "split_source", "murcko_scaffold",
]
EXCLUDE_PATTERNS = [
    "__bootstrap__", "__internal__", "__aug__",
    "_x", "_y",  # merge 잔여 suffix
]

# ── Morgan Fingerprint ────────────────────────────────────────────────
FP_RADIUS = 2
FP_NBITS  = 256

# ── Feature Selection (fold 내부) ────────────────────────────────────
FP_SELECT_K       = 128          # 상위 k bits
FP_PREVALENCE_MIN = 0.01         # 최소 prevalence
FP_VARIANCE_MIN   = 0.005        # 최소 variance

# ── Imbalance 전략 ────────────────────────────────────────────────────
IMBALANCE_STRATEGIES = ["none", "alert_bootstrap", "smotenc", "hybrid"]

# ── 모델 설정 ─────────────────────────────────────────────────────────
MODELS = ["logreg", "xgb"]

LOGREG_PARAM_DIST = {
    "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "l1_ratio": [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0],
}

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2],
}

N_RANDOM_SEARCH = 40
CV_FOLDS = 5
RANDOM_SEED = 42

# ── Threshold tuning ─────────────────────────────────────────────────
THRESHOLD_METRIC = {
    "ames": "mcc",
    "invitro": "mcc",
    "invivo": "balanced_accuracy",
}
SPECIFICITY_FLOOR = {"invivo": 0.5}

# ── Shortlist (최우선 재현 대상) ──────────────────────────────────────
SHORTLIST = [
    {"endpoint": "ames",    "strategy": "none",            "model": "xgb"},
    {"endpoint": "invitro", "strategy": "alert_bootstrap", "model": "xgb"},
    {"endpoint": "invitro", "strategy": "none",            "model": "logreg"},  # compact benchmark
    {"endpoint": "invivo",  "strategy": "alert_bootstrap", "model": "xgb"},
]

# ── Compact Feature Sets (endpoint별) ────────────────────────────────
COMPACT_FEATURES = {
    "ames": [
        "bb_n_genotox_alerts", "nitro_aromatic", "n_nitroso",
        "epoxide", "hydrazine_like", "aromatic_ring_count",
        "fused_ring_like", "heteroaromatic_ring", "fraction_csp3",
        "rot_bonds", "ester", "sulfone",
        # QM electronic (Ames: electrophilic reactivity 핵심)
        "qm_electrophilicity_proxy", "qm_LUMO_proxy", "qm_gap_proxy",
        "qm_gasteiger_max_charge", "qm_chemical_softness_proxy",
    ],
    "invitro": [
        "bb_n_genotox_alerts", "aromatic_amine", "primary_aromatic_amine",
        "aniline", "alpha_beta_unsat_carbonyl", "halogen",
        "MW", "logP", "rot_bonds", "fraction_csp3",
        # QM electronic (invitro: DNA intercalation + charge transfer)
        "qm_HOMO_proxy", "qm_gap_proxy", "qm_electrophilicity_proxy",
        "qm_gasteiger_max_abs_charge", "qm_chemical_softness_proxy",
    ],
    "invivo": [
        "epoxide", "organometal_like", "alcohol", "fused_ring_like",
        "MW", "rot_bonds", "logP", "TPSA",
        # QM electronic (invivo: 대사 활성화 + 생체이용률)
        "qm_gap_proxy", "qm_MolMR", "qm_LabuteASA",
        "qm_chemical_softness_proxy", "qm_electrophilicity_proxy",
    ],
}

# ── QM block (optional external) ─────────────────────────────────────
QM_FILE = DATA_DIR / "qm_descriptors.csv"
QM_EXT_COLS = [
    "HOMO", "LUMO", "gap", "dipole_moment", "polarizability",
    "max_atomic_charge", "min_atomic_charge", "electrophilicity_index",
    "hardness", "softness",
]

# ── RDKit 전자적 기술자 (SMILES에서 직접 계산) ──────────────────────
COMPUTE_ELECTRONIC = True          # True면 SMILES → 전자적 descriptor 자동 계산
QM_PREFIX = "qm_"                  # computed electronic descriptor prefix
EXT_QM_PREFIX = "ext_qm_"         # external QM file descriptor prefix

# 핵심 전자적 descriptor (endpoint별 중요도 높은 것)
QM_KEY_DESCRIPTORS = [
    "qm_HOMO_proxy", "qm_LUMO_proxy", "qm_gap_proxy",
    "qm_electrophilicity_proxy", "qm_chemical_hardness_proxy",
    "qm_chemical_softness_proxy", "qm_chemical_potential_proxy",
    "qm_gasteiger_max_charge", "qm_gasteiger_min_charge",
    "qm_gasteiger_charge_range", "qm_gasteiger_max_abs_charge",
    "qm_MaxPartialCharge", "qm_MinPartialCharge",
    "qm_MolMR", "qm_LabuteASA", "qm_PEOE_VSA_sum",
    "qm_NumValenceElectrons",
    "qm_Chi0n", "qm_Chi1n", "qm_Kappa1", "qm_Kappa2",
]

# ── 물리화학적 + QM 시각화용 성질 목록 ───────────────────────────────
PROPERTY_VIZ_LIST = [
    "MW", "logP", "TPSA", "rot_bonds", "fraction_csp3",
    "qm_HOMO_proxy", "qm_LUMO_proxy", "qm_gap_proxy",
    "qm_electrophilicity_proxy", "qm_chemical_softness_proxy",
    "qm_gasteiger_max_charge", "qm_gasteiger_min_charge",
    "qm_MolMR", "qm_LabuteASA",
    "qm_NumValenceElectrons", "qm_Chi1n",
]


def save_run_config(run_dir: pathlib.Path, extra: dict = None):
    """현재 config를 JSON으로 저장"""
    cfg = {
        "project_root": str(PROJECT_ROOT),
        "fp_radius": FP_RADIUS,
        "fp_nbits": FP_NBITS,
        "fp_select_k": FP_SELECT_K,
        "cv_folds": CV_FOLDS,
        "n_random_search": N_RANDOM_SEARCH,
        "random_seed": RANDOM_SEED,
        "shortlist": SHORTLIST,
    }
    if extra:
        cfg.update(extra)
    out = run_dir / "run_config.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return out
