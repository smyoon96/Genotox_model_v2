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
ENDPOINTS = ["ames", "invitro", "invivo", "invitro_sampling", "invivo_sampling"]

ENDPOINT_DISPLAY = {
    "ames":              "Ames",
    "invitro":           "In vitro chromosome aberration",
    "invivo":            "In vivo micronucleus",
    "invitro_sampling":  "In vitro CA (Neg-Sampled)",
    "invivo_sampling":   "In vivo MN (Neg-Sampled)",
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
    "ames":    ["ames.xlsx", "ames.csv"],
    "invitro": ["invitro.xlsx","invitro.csv"],
    "invivo":  ["invivo.xlsx","invivo.csv"],
    "invitro_sampling" :["invitro_sampling.xlsx", "invitro_sampling.csv"],
    "invivo_sampling" :["invivo_sampling.xlsx", "invivo_sampling.csv"],
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
FP_NBITS  = 1024          # default (increased from 256)
FP_BITS_CANDIDATES = [256, 512, 1024, 2048]  # 실험 비교용
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
#  모델 – Hyperparameter 탐색 공간 (8 models)
#  genotox_pipeline.py의 HP_GRIDS 및 tune_model()이 이 dict를 참조
# ──────────────────────────────────────────────
HP_GRIDS = {
    "xgb": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.3, 0.5],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    },
    "lgbm": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 5, 7, 9, -1],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "num_leaves":       [15, 31, 63, 127],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples":[5, 10, 20],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    },
    "rf": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [5, 10, 15, 20, None],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2", 0.5, 0.8],
    },
    "svm": {
        "clf__C":           [0.1, 1.0, 10.0],
        "clf__kernel":      ["rbf"],
        "clf__gamma":       ["scale", 0.01],
    },
    "logistic": {
        "clf__C":           [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty":     ["l1", "l2", "elasticnet"],
        "clf__solver":      ["saga"],
        "clf__l1_ratio":    [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    },
    "ann": {
        "clf__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "clf__alpha":              [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.005, 0.01],
        "clf__batch_size":         [32, 64, 128],
    },
    "dnn": {
        "clf__hidden_layer_sizes": [(256, 128, 64), (256, 128, 64, 32),
                                    (512, 256, 128, 64), (128, 64, 32, 16)],
        "clf__alpha":              [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.005, 0.01],
        "clf__batch_size":         [32, 64, 128],
    },
    "gnn": {
        "hidden_dim":   [32, 64, 128],
        "n_layers":     [2, 3, 4],
        "dropout":      [0.1, 0.2, 0.3],
        "lr":           [0.0005, 0.001, 0.005],
    },
}

# 하이퍼파라미터 탐색 설정
HP_N_ITER = 10       # RandomizedSearchCV iterations per model
HP_CV_FOLDS = 5      # GroupKFold folds for HP search

# ──────────────────────────────────────────────
#  Threshold tuning
# ──────────────────────────────────────────────
SPECIFICITY_FLOOR = {
    "ames":              0.70,
    "invitro":           0.75,
    "invivo":            0.80,
    "invitro_sampling":  0.75,
    "invivo_sampling":   0.80,
}

# ──────────────────────────────────────────────
#  Tuning metric
# ──────────────────────────────────────────────
TUNING_METRIC = {
    "ames":    "average_precision",
    "invitro": "average_precision",
    "invivo":  "average_precision",
    "invitro_sampling" : "average_precision",
    "invivo_sampling" : "average_precision",
}

# ──────────────────────────────────────────────
#  Feature modes (실험군)
# ──────────────────────────────────────────────
FEATURE_MODES = ["compact", "broad_fp256", "broad_fp512", "broad_fp1024", "broad_fp2048"]

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
    # ═══════════════════════════════════════════════════════
    #  Benigni/Bossa Structural Alerts for Genotoxic Carcinogenicity
    #  SA1–SA28 (ToxTree implementation, Benigni & Bossa 2008;
    #  updated Benigni, Bossa & Tcheremenskaia 2013)
    #  Ref: EUR 23241 EN, Chem Rev 2011;111:2507
    # ═══════════════════════════════════════════════════════

    # --- SA1: Acyl halides ---
    "bb_sa1_acyl_halide":            "[CX3](=[OX1])[F,Cl,Br,I]",
    # --- SA2: Alkyl esters of phosphonic/sulfonic acids (alkylating) ---
    "bb_sa2_alkyl_ester_sulfo":      "[SX4](=[OX1])(=[OX1])[OX2][CX4]",
    "bb_sa2b_alkyl_ester_phospho":   "[PX4](=[OX1])([OX2][CX4])([OX2])[OX2]",
    # --- SA3: Aliphatic N-nitroso ---
    "bb_sa3_aliphatic_nnitroso":     "[NX3;H0;!$(Nc)]([CX4])[NX2]=O",
    # --- SA4: Aromatic N-nitroso ---
    "bb_sa4_aromatic_nnitroso":      "[NX3;H0;$(Nc)][NX2]=O",
    # --- SA5: Aromatic nitroso ---
    "bb_sa5_aromatic_nitroso":       "[c][NX2]=O",
    # --- SA6: Unsubstituted heteroatom-bonded heteroatom (hydrazine) ---
    "bb_sa6_hydrazine":              "[NX3;!$(NC=O)][NX3;!$(NC=O)]",
    # --- SA7: Aliphatic halide (alkyl halide, potential alkylating agent) ---
    "bb_sa7_aliphatic_halide":       "[CX4][F,Cl,Br,I]",
    # --- SA8: α,β-unsaturated aldehydes (Michael-type) ---
    "bb_sa8_ab_unsat_aldehyde":      "[CX3H1](=O)/[CX3]=[CX3]",
    # --- SA9: α,β-unsaturated alkoxy (vinyl ether Michael acceptor) ---
    "bb_sa9_ab_unsat_alkoxy":        "[CX3](=[OX1])/[CX3]=[CX3][OX2]",
    # --- SA10: Aromatic mono-/diazo ---
    "bb_sa10_azo":                   "[NX2]=[NX2]",
    # --- SA11: Aromatic N-oxide ---
    "bb_sa11_aromatic_noxide":       "[nX3]([OX1])",
    # --- SA12: Aromatic nitro ---
    "bb_sa12_aromatic_nitro":        "[c][NX3+](=O)[O-]",
    "bb_sa12b_aromatic_nitro_alt":   "[c][N](=O)=O",
    # --- SA13: Aromatic ring N-oxide ---
    "bb_sa13_ring_noxide":           "[n+]([OX1-])",
    # --- SA14: Azide / triazene ---
    "bb_sa14_azide":                 "[NX1]~[NX2]~[NX1]",
    "bb_sa14b_triazene":             "[NX3][NX2]=[NX2]",
    # --- SA15: Aziridine / Azetidine (strained ring N-heterocycles) ---
    "bb_sa15_aziridine":             "C1NC1",
    "bb_sa15b_azetidine":            "C1NCC1",
    # --- SA16: Carbamate (potential nitroso via metabolic activation) ---
    "bb_sa16_carbamate":             "[NX3][CX3](=[OX1])[OX2]",
    # --- SA17: Epoxide ---
    "bb_sa17_epoxide":               "C1OC1",
    # --- SA18: Isocyanate / isothiocyanate ---
    "bb_sa18_isocyanate":            "[NX2]=C=O",
    "bb_sa18b_isothiocyanate":       "[NX2]=C=S",
    # --- SA19: Nitrogen mustard (bis(2-chloroethyl)amine) ---
    "bb_sa19_nitrogen_mustard":      "ClCCN(CCCl)",
    "bb_sa19b_sulfur_mustard":       "ClCCSCCCl",
    # --- SA20: N-hydroxylamine ---
    "bb_sa20_nhydroxylamine":        "[NX3;H1,H2][OX2H]",
    # --- SA21: Propiolactone / propiosultone (strained ring esters) ---
    "bb_sa21_propiolactone":         "C1CC(=O)O1",
    "bb_sa21b_propiosultone":        "C1CCS(=O)(=O)O1",
    "bb_sa21c_butyrolactone":        "C1CCC(=O)O1",
    # --- SA22: Primary aromatic amine (including heterocyclic) ---
    "bb_sa22_aromatic_amine_pri":    "[c][NX3;H2;!$(NC=O)]",
    # --- SA23: Secondary aromatic amine ---
    "bb_sa23_aromatic_amine_sec":    "[c][NX3;H1;!$(NC=O)]",
    # --- SA24: Aliphatic N-nitro ---
    "bb_sa24_aliphatic_nnitro":      "[CX4][NX3][NX3+](=O)[O-]",
    # --- SA25: Aromatic nitrile ---
    "bb_sa25_aromatic_nitrile":      "[c][CX2]#[NX1]",
    # --- SA26: Sulfonyl halide ---
    "bb_sa26_sulfonyl_halide":       "[#16X4](=[OX1])(=[OX1])[F,Cl,Br,I]",
    # --- SA27: Thiocarbonyl ---
    "bb_sa27_thiocarbonyl":          "[CX3]=[SX1]",
    # --- SA28: α-halo carbonyl ---
    "bb_sa28_alpha_halo_carbonyl":   "[CX4;H1,H2]([F,Cl,Br,I])[CX3](=O)",

    # ═══════════════════════════════════════════════════════
    #  Benigni/Bossa Non-Genotoxic Carcinogenicity Alerts
    #  SA29–SA31 (updated 2013)
    # ═══════════════════════════════════════════════════════
    "bb_sa29_coumarin":              "c1cc2OC(=O)Cc2cc1",  # coumarin skeleton
    "bb_sa30_phorbol_like":          "OCC1=CC(=O)C2CC1C1OC1C2",  # simplified phorbol
    "bb_sa31_biphenyl":              "c1ccc(-c2ccccc2)cc1",  # biphenyl core

    # ═══════════════════════════════════════════════════════
    #  Kazius et al. (2005) Ames Mutagenicity Alerts
    #  J. Chem. Inf. Model. 45:508
    #  8 toxicophores not fully covered above
    # ═══════════════════════════════════════════════════════
    "kz_polycyclic_aromatic_3":      "c1ccc2c(c1)ccc1ccccc12",  # 3-ring PAH
    "kz_polycyclic_aromatic_4":      "c1ccc2c(c1)cc1ccc3ccccc3c1c2",  # 4-ring PAH
    "kz_unsat_aldehyde":             "[CX3H1](=O)[#6]=[#6]",
    "kz_diazonium":                  "[NX2+]#[NX1]",
    "kz_aromatic_hydroxylamine":     "[c][NX3]([OX2H])",
    "kz_aromatic_n_acylhydroxamic":  "[c][NX3]([OX2])[CX3]=O",
    "kz_quinone":                    "O=[#6]1[#6]=,:[#6][#6](=O)[#6]=,:[#6]1",
    "kz_bay_region_pah":             "c1cc2ccc3cccc4ccc(c1)c2c34",  # bay-region

    # ═══════════════════════════════════════════════════════
    #  ISS Ames Test Alerts (Toxtree ISSCAN module)
    #  Structural Alerts for in vitro mutagenicity
    # ═══════════════════════════════════════════════════════
    "iss_aromatic_diazo":            "c[NX2]=[NX2]c",
    "iss_haloalkene":                "[CX3]([F,Cl,Br,I])=[CX3]",
    "iss_aldehyde":                  "[CX3H1](=O)",
    "iss_michael_acceptor_enone":    "[#6][CX3](=O)[CX3]=[CX3]",
    "iss_alkyl_nitrite":             "[CX4][OX2][NX2]=O",
    "iss_nitrosamine_any":           "[NX3][NX2]=O",
    "iss_dioxolane":                 "C1OCOC1",
    "iss_aminoazo_dye":              "c1ccc([NX3])cc1/[NX2]=[NX2]/c1ccccc1",

    # ═══════════════════════════════════════════════════════
    #  ISSMIC — In Vivo Micronucleus Specific Alerts
    #  Benigni & Bossa 2011 (MN-specific substructures)
    # ═══════════════════════════════════════════════════════
    "mn_aneuploidogen_colchicine":   "c1cc(OC)c(OC)c(OC)c1",  # trimethoxyphenyl
    "mn_thiol":                      "[SX2H]",
    "mn_disulfide":                  "[SX2][SX2]",
    "mn_heavy_metal_org":            "[#6]~[#50,#33,#51,#82,#80,#48,#24,#29,#34]",
    "mn_steroid_like":               "[C]1CC[C]2[C]([C]1)CC[C]1[C]2CCC2CCCC[C]12",

    # ═══════════════════════════════════════════════════════
    #  Metal-Related Functional Groups & Organometallics
    #  금속 함유 화합물의 유전독성 관련 구조 경고
    #  Ref: IARC Group 1/2A metals, Benigni 2008 organometal SA
    # ═══════════════════════════════════════════════════════
    # -- Organometallic bonds (carbon-metal) --
    "met_organotin":                 "[#6]~[#50]",          # 유기주석 (tributyltin 등)
    "met_organoarsenic":             "[#6]~[#33]",          # 유기비소
    "met_organoantimony":            "[#6]~[#51]",          # 유기안티몬
    "met_organolead":                "[#6]~[#82]",          # 유기납
    "met_organomercury":             "[#6]~[#80]",          # 유기수은
    "met_organocadmium":             "[#6]~[#48]",          # 유기카드뮴
    "met_organochromium":            "[#6]~[#24]",          # 유기크롬
    "met_organocopper":              "[#6]~[#29]",          # 유기구리
    "met_organoselenium":            "[#6]~[#34]",          # 유기셀레늄
    "met_organonickel":              "[#6]~[#28]",          # 유기니켈
    "met_organoplatinum":            "[#6]~[#78]",          # 유기백금 (cisplatin 등)
    "met_organocobalt":              "[#6]~[#27]",          # 유기코발트
    # -- Inorganic metal atoms (presence as feature) --
    "met_has_chromium":              "[#24]",               # Cr (IARC Group 1)
    "met_has_nickel":                "[#28]",               # Ni (IARC Group 1)
    "met_has_cadmium":               "[#48]",               # Cd (IARC Group 1)
    "met_has_arsenic":               "[#33]",               # As (IARC Group 1)
    "met_has_beryllium":             "[#4]",                # Be (IARC Group 1)
    "met_has_cobalt":                "[#27]",               # Co (IARC Group 2B)
    "met_has_lead":                  "[#82]",               # Pb (IARC Group 2A)
    "met_has_mercury":               "[#80]",               # Hg
    "met_has_vanadium":              "[#23]",               # V (산화적 DNA 손상)
    "met_has_manganese":             "[#25]",               # Mn
    # -- Metal coordination patterns --
    "met_metal_halide":              "[#3,#11,#12,#13,#19,#20,#26,#27,#28,#29,#30]~[F,Cl,Br,I]",
    "met_metal_oxide":               "[#24,#25,#28,#33,#48,#80,#82]~[OX1,OX2]",
    "met_metal_sulfide":             "[#24,#28,#29,#30,#33,#48,#80,#82]~[#16]",
    # -- Chelation-prone patterns --
    "met_dithiocarbamate":           "[SX2][CX3](=[SX1])[NX3]",   # 금속 킬레이트 (Zn, Cu 등)
    "met_hydroxamic_acid":           "[NX3][CX3](=O)[OX2H]",      # Fe3+ 킬레이트
    "met_edta_like":                 "[NX3](CC(=O)[OX2H])CC(=O)[OX2H]",  # 아미노폴리카르복시산 킬레이트
    # -- Specific toxic metal compounds --
    "met_arsenate":                  "[As](=O)([OX2])([OX2])[OX2]",  # 비산염
    "met_arsenite":                  "[As]([OX2])([OX2])[OX2]",       # 아비산염
    "met_chromate":                  "[Cr](=O)(=O)([OX2])[OX2]",     # 크롬산염 (Cr6+)

    # ═══════════════════════════════════════════════════════
    #  Common Functional Groups (non-alert, descriptor용)
    # ═══════════════════════════════════════════════════════
    "fg_alcohol":                    "[OX2H]",
    "fg_ether":                      "[OD2]([#6])[#6]",
    "fg_aldehyde":                   "[CX3H1](=O)[#6]",
    "fg_ketone":                     "[#6][CX3](=O)[#6]",
    "fg_carboxylic_acid":            "[CX3](=O)[OX2H1]",
    "fg_ester":                      "[#6][CX3](=O)[OX2H0][#6]",
    "fg_amide":                      "[NX3][CX3](=[OX1])[#6]",
    "fg_primary_amine":              "[NX3;H2;!$(NC=O)]",
    "fg_secondary_amine":            "[NX3;H1;!$(NC=O)]",
    "fg_tertiary_amine":             "[NX3;H0;!$(NC=O);!$(N=*)]",
    "fg_nitro":                      "[NX3](=O)=O",
    "fg_halide_F":                   "[F]",
    "fg_halide_Cl":                  "[Cl]",
    "fg_halide_Br":                  "[Br]",
    "fg_halide_I":                   "[I]",
    "fg_sulfide":                    "[#16X2]([#6])[#6]",
    "fg_sulfoxide":                  "[#16X3](=[OX1])([#6])[#6]",
    "fg_sulfone":                    "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "fg_phosphate":                  "[PX4](=O)([OX2])([OX2])[OX2]",
    "fg_cyano":                      "[CX2]#[NX1]",
    "fg_alkene":                     "[CX3]=[CX3]",
    "fg_alkyne":                     "[CX2]#[CX2]",
    "fg_phenol":                     "[OX2H]c1ccccc1",
    "fg_sulfonic_acid":              "[SX4](=[OX1])(=[OX1])([OX2H])",
    "fg_nitrile":                    "[NX1]#[CX2]",
    "fg_urea":                       "[NX3][CX3](=[OX1])[NX3]",
    "fg_thiourea":                   "[NX3][CX3](=[SX1])[NX3]",
    "fg_guanidine":                  "[NX3][CX3](=[NX2])[NX3]",
    "fg_lactam":                     "C1CC(=O)N1",
    "fg_lactone":                    "C1CC(=O)O1",
    "fg_anhydride":                  "[CX3](=O)[OX2][CX3](=O)",
    "fg_sulfonamide":                "[NX3][SX4](=[OX1])(=[OX1])",
}

# ──────────────────────────────────────────────
#  High-confidence alert families
# ──────────────────────────────────────────────
ALERT_FAMILIES = {
    "ames": {
        "positive": [
            "bb_sa3_aliphatic_nnitroso", "bb_sa4_aromatic_nnitroso",
            "bb_sa5_aromatic_nitroso", "bb_sa12_aromatic_nitro",
            "bb_sa17_epoxide", "bb_sa6_hydrazine",
            "bb_sa22_aromatic_amine_pri", "bb_sa23_aromatic_amine_sec",
            "bb_sa8_ab_unsat_aldehyde", "bb_sa14_azide",
            "bb_sa15_aziridine", "bb_sa1_acyl_halide",
            "bb_sa7_aliphatic_halide", "bb_sa28_alpha_halo_carbonyl",
            "bb_sa19_nitrogen_mustard", "bb_sa10_azo",
            "kz_polycyclic_aromatic_3", "kz_quinone",
            "iss_nitrosamine_any", "iss_haloalkene",
            "met_chromate", "met_organoarsenic",  # Cr6+/As are Ames-positive
        ],
        "negative": [],
    },
    "invitro": {
        "positive": [
            "bb_sa22_aromatic_amine_pri", "bb_sa23_aromatic_amine_sec",
            "bb_sa8_ab_unsat_aldehyde", "iss_michael_acceptor_enone",
            "bb_sa17_epoxide", "bb_sa12_aromatic_nitro",
            "bb_sa6_hydrazine", "bb_sa10_azo",
            "kz_polycyclic_aromatic_3", "kz_quinone",
            "mn_heavy_metal_org",
            "met_has_chromium", "met_has_nickel", "met_has_cadmium",  # clastogenic metals
        ],
        "negative": [],
    },
    "invivo": {
        "positive": [
            "bb_sa17_epoxide", "mn_heavy_metal_org",
            "bb_sa12_aromatic_nitro", "bb_sa22_aromatic_amine_pri",
            "bb_sa6_hydrazine", "bb_sa7_aliphatic_halide",
            "bb_sa19_nitrogen_mustard", "mn_aneuploidogen_colchicine",
            "kz_polycyclic_aromatic_3",
            "met_has_chromium", "met_has_nickel", "met_has_arsenic",  # in vivo MN inducers
            "met_has_cadmium", "met_has_lead",
        ],
        "negative": [],
    },
    # Sampling endpoints inherit alerts from parent
    "invitro_sampling": {
        "positive": [
            "bb_sa22_aromatic_amine_pri", "bb_sa23_aromatic_amine_sec",
            "bb_sa8_ab_unsat_aldehyde", "iss_michael_acceptor_enone",
            "bb_sa17_epoxide", "bb_sa12_aromatic_nitro",
            "bb_sa6_hydrazine", "bb_sa10_azo",
            "kz_polycyclic_aromatic_3", "kz_quinone",
            "mn_heavy_metal_org",
            "met_has_chromium", "met_has_nickel", "met_has_cadmium",
        ],
        "negative": [],
    },
    "invivo_sampling": {
        "positive": [
            "bb_sa17_epoxide", "mn_heavy_metal_org",
            "bb_sa12_aromatic_nitro", "bb_sa22_aromatic_amine_pri",
            "bb_sa6_hydrazine", "bb_sa7_aliphatic_halide",
            "bb_sa19_nitrogen_mustard", "mn_aneuploidogen_colchicine",
            "kz_polycyclic_aromatic_3",
            "met_has_chromium", "met_has_nickel", "met_has_arsenic",
            "met_has_cadmium", "met_has_lead",
        ],
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


# ──────────────────────────────────────────────
#  디렉토리 상수 (step 스크립트 공용)
# ──────────────────────────────────────────────
LOG_DIR      = PROJECT_ROOT / "logs"
OUTPUT_DIR   = RUNS_DIR                   # alias for compatibility
SUMMARY_DIR  = PROJECT_ROOT / "summary"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
BROADFP_DIR  = PROJECT_ROOT / "broadfp"

for _d in [LOG_DIR, SUMMARY_DIR, ARTIFACT_DIR, BROADFP_DIR, RUNS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
#  학습 공통 설정 (step3_train 참조)
# ──────────────────────────────────────────────
RANDOM_SEED     = GLOBAL_SEED
CV_FOLDS        = 5
FP_SELECT_K     = 128          # MI 기반 FP bit 선택 수
N_RANDOM_SEARCH = 10           # RandomizedSearchCV iterations

# threshold 선택 기준 (endpoint별)
THRESHOLD_METRIC = {ep: "mcc" for ep in ENDPOINTS}

# 컬럼 제외 패턴
EXCLUDE_PATTERNS = [
    "_analysis_smiles", "scaffold", "split", "endpoint",
    "source_dataset", "No", "label", "__",
]

# 기본 실험 목록 (shortlist)
MODELS = ["xgb", "lgbm", "rf", "svm", "logistic", "ann"]
IMBALANCE_STRATEGIES = UNDERSAMPLING_STRATEGIES

SHORTLIST = [
    {"endpoint": "ames",    "model": "xgb",      "strategy": "none"},
    {"endpoint": "ames",    "model": "lgbm",     "strategy": "none"},
    {"endpoint": "ames",    "model": "rf",       "strategy": "none"},
    {"endpoint": "invitro", "model": "xgb",      "strategy": "none"},
    {"endpoint": "invitro", "model": "lgbm",     "strategy": "none"},
    {"endpoint": "invivo",  "model": "xgb",      "strategy": "none"},
    {"endpoint": "invivo",  "model": "lgbm",     "strategy": "none"},
    {"endpoint": "invivo",  "model": "rf",       "strategy": "none"},
]

# ──────────────────────────────────────────────
#  유틸리티 (호환성 alias)
# ──────────────────────────────────────────────
def save_run_config(out_dir: Path, extra: dict = None):
    """실행 설정 JSON 저장 (run_all.py 호환)"""
    import json
    from datetime import datetime
    cfg_data = {
        "project_root": str(PROJECT_ROOT),
        "data_dir":     str(DATA_DIR),
        "runs_dir":     str(RUNS_DIR),
        "global_seed":  GLOBAL_SEED,
        "endpoints":    ENDPOINTS,
        "timestamp":    datetime.now().isoformat(),
    }
    if extra:
        cfg_data.update(extra)
    out = Path(out_dir) / "run_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False, default=str)


# ──────────────────────────────────────────────
#  QM / 전자적 기술자 설정
# ──────────────────────────────────────────────
COMPUTE_ELECTRONIC = False          # RDKit 전자적 기술자 계산 여부
QM_PREFIX          = "qm_"         # 계산된 QM 기술자 컬럼 prefix
EXT_QM_PREFIX      = "extqm_"      # 외부 QM 파일 merge 시 prefix
QM_FILE            = DATA_DIR / "qm_descriptors.csv"   # 외부 QM 파일 (없으면 skip)

# ──────────────────────────────────────────────
#  메타 컬럼 (feature에서 제외)
# ──────────────────────────────────────────────
META_COLS = {
    "No", "label", "endpoint", "split", "split_source",
    "scaffold_group", "scaffold_group_type", "murcko_scaffold",
    "SMILES", "SMILES_raw", "canonical_smiles", "standardized_smiles",
    "analysis_smiles", "_analysis_smiles", "smiles",
    "source_dataset", "domain",
}
