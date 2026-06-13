"""
utils/electronic_descriptors.py -- 전자적/양자화학적 기술자 계산
================================================================
외부 QM 소프트웨어 없이 파이프라인 내에서 직접 계산.

계산 레벨:
  Level 1 (RDKit 기본, 항상 가능)
    ├─ Gasteiger 부분 전하 통계 (Max/Min/Mean/Std)
    ├─ EState 지수 (전기토폴로지적 상태)
    ├─ Chi 연결성 지수 (χ0~χ4, 전자적 위상)
    ├─ Kappa 형상 지수 (κ1~κ3)
    ├─ BCUT 고유값 기반 기술자 (전하 가중)
    ├─ Hall-Kier alpha, Balaban J
    └─ Molar Refractivity (분극율 근사)

  Level 2 (Mordred, pip install mordred)
    ├─ 확장 전자 기술자 (~200개)
    ├─ HOMO/LUMO 에너지 근사 (AcidBase 기반)
    ├─ 분자 분극율
    └─ 전자 밀도 분포 기술자

  Level 3 (xtb, pip install xtb-python)  ← 가장 정확한 반경험적 QM
    ├─ HOMO/LUMO energy (GFN2-xTB)
    ├─ HOMO-LUMO gap
    ├─ 이온화 에너지 (IP) 근사
    ├─ 전자 친화도 (EA) 근사
    ├─ 쌍극자 모멘트
    └─ Mulliken / CM5 partial charges

설치:
    pip install mordred          # Level 2
    pip install xtb-python       # Level 3 (Windows에서 conda 권장)
    conda install -c conda-forge xtb-python

사용:
    from utils.electronic_descriptors import extract_electronic_features
    elec_df = extract_electronic_features(df, smiles_col="SMILES")
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

logger = logging.getLogger("electronic_descriptors")

# ── 레벨별 가용성 확인 ────────────────────────────────────────────────
_HAS_RDKIT  = False
_HAS_MORDRED = False
_HAS_XTB    = False

try:
    from rdkit import Chem
    from rdkit.Chem import (Descriptors, AllChem, rdMolDescriptors,
                             rdPartialCharges, EState)
    from rdkit.Chem.EState import EState_VSA
    _HAS_RDKIT = True
except ImportError:
    pass

try:
    from mordred import Calculator, descriptors as mordred_descs
    _HAS_MORDRED = True
except ImportError:
    pass

try:
    from xtb.interface import Calculator as XTBCalculator
    from xtb.libxtb import VERBOSITY_MUTED
    _HAS_XTB = True
except ImportError:
    try:
        import xtb
        _HAS_XTB = True
    except ImportError:
        pass


def _mol_from_smiles(smi: str):
    """SMILES → RDKit mol. 실패 시 None."""
    if not _HAS_RDKIT or not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  Level 1: RDKit 전자적 기술자 (항상 가능)
# ═══════════════════════════════════════════════════════

def _calc_rdkit_electronic(mol) -> dict:
    """
    단일 분자의 RDKit 전자적 기술자 계산.

    Gasteiger 부분 전하:
      - 원자별 전하 분포 → 독성 관련 반응 부위 근사
      - DNA 친전자성 반응과 관련 (Ames test)

    EState (전기토폴로지적 상태):
      - Hall & Kier (1991) - 원자의 전자적 환경 인코딩
      - 기존 FP에 없는 국소 전자 정보 제공

    Chi 연결성 지수:
      - Randić (1975) - 전자 연결성의 위상적 표현
      - 분자 전자 구조의 크기/분지화 인코딩

    BCUT:
      - 전하 가중 인접 행렬의 고유값
      - 분자 전자 다양성 지수로 AD 계산에 유용
    """
    if mol is None:
        return {}

    desc = {}
    try:
        # ── Gasteiger 부분 전하 ─────────────────────────
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
                   for a in mol.GetAtoms()]
        charges = [c for c in charges if not np.isnan(c) and not np.isinf(c)]
        if charges:
            desc["qm_gasteiger_max"]    = max(charges)
            desc["qm_gasteiger_min"]    = min(charges)
            desc["qm_gasteiger_mean"]   = float(np.mean(charges))
            desc["qm_gasteiger_std"]    = float(np.std(charges))
            desc["qm_gasteiger_absmax"] = max(abs(c) for c in charges)
            # 양/음 전하 원자 비율 (반응 부위 다양성)
            desc["qm_frac_pos_charge"]  = sum(c > 0.1 for c in charges) / max(len(charges), 1)
            desc["qm_frac_neg_charge"]  = sum(c < -0.1 for c in charges) / max(len(charges), 1)

        # ── EState 지수 ──────────────────────────────────
        try:
            estate_vals = EState.EStateIndices(mol)
            estate_vals = [v for v in estate_vals
                           if not np.isnan(v) and not np.isinf(v)]
            if estate_vals:
                desc["qm_estate_max"]  = max(estate_vals)
                desc["qm_estate_min"]  = min(estate_vals)
                desc["qm_estate_mean"] = float(np.mean(estate_vals))
                desc["qm_estate_sum"]  = float(np.sum(estate_vals))
        except Exception:
            pass

        # ── RDKit 표준 전자 기술자 ────────────────────────
        rdkit_elec = {
            "qm_MaxPartialCharge":    Descriptors.MaxPartialCharge,
            "qm_MinPartialCharge":    Descriptors.MinPartialCharge,
            "qm_MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
            "qm_MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
            "qm_NumRadicalElectrons": Descriptors.NumRadicalElectrons,
        }
        for name, fn in rdkit_elec.items():
            try:
                v = fn(mol)
                if v is not None and not np.isnan(float(v)):
                    desc[name] = float(v)
            except Exception:
                pass

        # ── Chi 연결성 지수 (전자적 위상) ─────────────────
        chi_fns = {
            "qm_Chi0":  Descriptors.Chi0,
            "qm_Chi1":  Descriptors.Chi1,
            "qm_Chi2n": Descriptors.Chi2n, "qm_Chi2v": Descriptors.Chi2v,
            "qm_Chi3n": Descriptors.Chi3n, "qm_Chi3v": Descriptors.Chi3v,
            "qm_Chi4n": Descriptors.Chi4n, "qm_Chi4v": Descriptors.Chi4v,
        }
        for name, fn in chi_fns.items():
            try:
                v = fn(mol)
                if v is not None and not np.isnan(float(v)):
                    desc[name] = float(v)
            except Exception:
                pass

        # ── Kappa 형상 지수 ──────────────────────────────
        kappa_fns = {
            "qm_Kappa1": Descriptors.Kappa1,
            "qm_Kappa2": Descriptors.Kappa2,
            "qm_Kappa3": Descriptors.Kappa3,
        }
        for name, fn in kappa_fns.items():
            try:
                v = fn(mol)
                if v is not None and not np.isnan(float(v)):
                    desc[name] = float(v)
            except Exception:
                pass

        # ── BCUT 고유값 기반 기술자 ───────────────────────
        bcut_names = [
            "qm_BCUT2D_MWHI", "qm_BCUT2D_MWLOW",
            "qm_BCUT2D_CHGHI", "qm_BCUT2D_CHGLO",
            "qm_BCUT2D_LOGPHI", "qm_BCUT2D_LOGPLOW",
            "qm_BCUT2D_MRHI", "qm_BCUT2D_MRLOW",
        ]
        bcut_rdkit_names = [n.replace("qm_", "") for n in bcut_names]
        for qm_name, rdkit_name in zip(bcut_names, bcut_rdkit_names):
            try:
                fn = getattr(Descriptors, rdkit_name, None)
                if fn:
                    v = fn(mol)
                    if v is not None and not np.isnan(float(v)):
                        desc[qm_name] = float(v)
            except Exception:
                pass

        # ── 기타 전자 관련 기술자 ─────────────────────────
        misc_fns = {
            "qm_BalabanJ":    Descriptors.BalabanJ,    # 위상적 전자 지수
            "qm_HallKierAlpha": Descriptors.HallKierAlpha,
            "qm_Ipc":         Descriptors.Ipc,         # 정보 내용
            "qm_MolMR":       Descriptors.MolMR,       # 분자 굴절 (분극율 근사)
            "qm_SMR_VSA1":    Descriptors.SMR_VSA1,
            "qm_SMR_VSA2":    Descriptors.SMR_VSA2,
            "qm_SlogP_VSA1":  Descriptors.SlogP_VSA1,
            "qm_SlogP_VSA2":  Descriptors.SlogP_VSA2,
            "qm_PEOE_VSA1":   Descriptors.PEOE_VSA1,   # 부분 전하 기반 VSA
            "qm_PEOE_VSA2":   Descriptors.PEOE_VSA2,
            "qm_PEOE_VSA3":   Descriptors.PEOE_VSA3,
        }
        for name, fn in misc_fns.items():
            try:
                v = fn(mol)
                if v is not None and not np.isnan(float(v)):
                    desc[name] = float(v)
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"  RDKit electronic calc failed: {e}")

    return desc


# ═══════════════════════════════════════════════════════
#  Level 2: Mordred 전자 기술자 (pip install mordred)
# ═══════════════════════════════════════════════════════

_MORDRED_CALC = None

def _get_mordred_calc():
    """Mordred calculator 캐싱 (초기화 비용 절감)."""
    global _MORDRED_CALC
    if _MORDRED_CALC is None and _HAS_MORDRED:
        try:
            # 전자/위상 관련 descriptor만 선택 (속도 최적화)
            from mordred import (
                AcidBase, Autocorrelation, BurdenMatrix,
                EState as mEState, InformationContent,
                MoeType, TopologicalCharge,
            )
            _MORDRED_CALC = Calculator([
                AcidBase, Autocorrelation, BurdenMatrix,
                mEState, InformationContent, MoeType, TopologicalCharge,
            ], ignore_3D=True)
        except Exception as e:
            logger.warning(f"Mordred calculator init failed: {e}")
    return _MORDRED_CALC


def _calc_mordred_electronic(mol) -> dict:
    """Mordred 전자 기술자 계산."""
    if not _HAS_MORDRED or mol is None:
        return {}
    calc = _get_mordred_calc()
    if calc is None:
        return {}
    try:
        result = calc(mol)
        desc = {}
        for name, val in result.items():
            key = f"qm_mrd_{name}"
            if val is not None and not isinstance(val, Exception):
                try:
                    fval = float(val)
                    if not np.isnan(fval) and not np.isinf(fval):
                        desc[key] = fval
                except (TypeError, ValueError):
                    pass
        return desc
    except Exception as e:
        logger.debug(f"  Mordred calc failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════
#  Level 3: GFN2-xTB 반경험적 QM (pip install xtb-python)
#  HOMO/LUMO energy, gap, dipole, IP, EA
# ═══════════════════════════════════════════════════════

def _calc_xtb_electronic(mol) -> dict:
    """
    GFN2-xTB (Grimme 2019) 반경험적 QM 계산.

    독성 관련 핵심 지표:
      - HOMO energy: 전자 공여 능력 (친핵성)
      - LUMO energy: 전자 수용 능력 (친전자성 → DNA 결합 위험)
      - HOMO-LUMO gap: 화학적 반응성 (작을수록 반응성 높음)
      - 쌍극자 모멘트: 막 투과성, 단백질 결합 친화도

    계산 시간: ~0.1~1초/분자 (분자 크기에 따라)
    """
    if not _HAS_XTB or mol is None:
        return {}

    try:
        from rdkit.Chem import AllChem
        import numpy as np

        # 3D 구조 생성 (xTB는 3D 필요)
        mol_h = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if result != 0:
            # 3D 생성 실패 → SMILES 기반 2D 근사
            AllChem.Compute2DCoords(mol_h)

        AllChem.MMFFOptimizeMolecule(mol_h)

        # 원자 좌표 및 원자번호 추출
        conf = mol_h.GetConformer()
        numbers = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()])
        positions = conf.GetPositions() * 1.8897259886  # Å → bohr

        # xTB 계산
        try:
            from xtb.interface import Calculator, Param
            from xtb.libxtb import VERBOSITY_MUTED
            calc = Calculator(Param.GFN2xTB, numbers, positions)
            calc.set_verbosity(VERBOSITY_MUTED)
            res = calc.singlepoint()

            desc = {}
            desc["qm_xtb_total_energy"]  = float(res.get_energy())    # hartree
            desc["qm_xtb_homo_energy"]   = float(res.get_homo_energy())  # eV
            desc["qm_xtb_lumo_energy"]   = float(res.get_lumo_energy())  # eV
            desc["qm_xtb_homo_lumo_gap"] = desc["qm_xtb_lumo_energy"] - desc["qm_xtb_homo_energy"]
            dipole = res.get_dipole()
            desc["qm_xtb_dipole"] = float(np.linalg.norm(dipole))
            # IP ≈ -HOMO (Koopmans' theorem), EA ≈ -LUMO
            desc["qm_xtb_ip_approx"]  = -desc["qm_xtb_homo_energy"]
            desc["qm_xtb_ea_approx"]  = -desc["qm_xtb_lumo_energy"]
            # 화학적 퍼텐셜 μ = (HOMO+LUMO)/2, 화학적 강도 η = (LUMO-HOMO)/2
            desc["qm_xtb_chem_potential"] = (desc["qm_xtb_homo_energy"] + desc["qm_xtb_lumo_energy"]) / 2
            desc["qm_xtb_hardness"]       = desc["qm_xtb_homo_lumo_gap"] / 2
            desc["qm_xtb_softness"]       = 1.0 / max(desc["qm_xtb_hardness"], 1e-6)
            desc["qm_xtb_electrophilicity"] = desc["qm_xtb_chem_potential"] ** 2 / (2 * max(desc["qm_xtb_hardness"], 1e-6))

            # Mulliken 전하
            try:
                charges = res.get_charges()
                desc["qm_xtb_charge_max"] = float(np.max(charges))
                desc["qm_xtb_charge_min"] = float(np.min(charges))
                desc["qm_xtb_charge_std"] = float(np.std(charges))
            except Exception:
                pass

            return desc

        except Exception:
            # xtb-python 구버전 API
            pass

    except Exception as e:
        logger.debug(f"  xTB calc failed: {e}")

    return {}


# ═══════════════════════════════════════════════════════
#  통합 함수
# ═══════════════════════════════════════════════════════

def extract_electronic_features(
    df: pd.DataFrame,
    smiles_col: str = None,
    levels: List[str] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    분자별 전자적/양자화학적 기술자 계산 후 DataFrame 반환.

    Parameters
    ----------
    df        : SMILES 컬럼 포함 DataFrame
    smiles_col: SMILES 컬럼명 (None이면 자동 탐색)
    levels    : ["rdkit", "mordred", "xtb"] 중 선택
                None이면 가용한 것 모두 사용
    n_jobs    : 병렬 처리 수 (xtb는 느리므로 >1 권장)

    Returns
    -------
    pd.DataFrame: qm_* prefix 컬럼들, df와 같은 index
    """
    from pipeline_v2_core import find_smi

    if smiles_col is None:
        try:
            smiles_col = find_smi(df)
        except Exception:
            smiles_col = next((c for c in df.columns
                               if "smiles" in c.lower()), None)
    if smiles_col is None or smiles_col not in df.columns:
        logger.warning("SMILES column not found -- returning empty electronic features")
        return pd.DataFrame(index=df.index)

    # 가용 레벨 결정
    if levels is None:
        levels = []
        if _HAS_RDKIT:   levels.append("rdkit")
        if _HAS_MORDRED: levels.append("mordred")
        if _HAS_XTB:     levels.append("xtb")

    if not levels:
        logger.warning("No QM library available. Install: pip install mordred xtb-python")
        return pd.DataFrame(index=df.index)

    logger.info(f"  Electronic descriptors: levels={levels} n={len(df)}")
    available_levels = {
        "rdkit":   _HAS_RDKIT,
        "mordred": _HAS_MORDRED,
        "xtb":     _HAS_XTB,
    }
    for lv in levels:
        if not available_levels.get(lv, False):
            logger.warning(f"  Level '{lv}' not available -- skipping")

    smiles_list = df[smiles_col].tolist()

    def _calc_one(smi: str) -> dict:
        mol = _mol_from_smiles(str(smi))
        row = {}
        if "rdkit" in levels and _HAS_RDKIT:
            row.update(_calc_rdkit_electronic(mol))
        if "mordred" in levels and _HAS_MORDRED:
            row.update(_calc_mordred_electronic(mol))
        if "xtb" in levels and _HAS_XTB:
            row.update(_calc_xtb_electronic(mol))
        return row

    # 병렬 처리 (xtb 사용 시 n_jobs>1 권장)
    from utils.progress import pbar
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            records = Parallel(n_jobs=n_jobs)(
                delayed(_calc_one)(smi)
                for smi in pbar(smiles_list,
                                desc="  Electronic features", leave=False)
            )
        except ImportError:
            records = [_calc_one(smi)
                       for smi in pbar(smiles_list,
                                       desc="  Electronic features", leave=False)]
    else:
        records = [_calc_one(smi)
                   for smi in pbar(smiles_list,
                                   desc="  Electronic features", leave=False)]

    elec_df = pd.DataFrame(records, index=df.index)

    # 수치 변환 + NaN 처리
    for c in elec_df.columns:
        elec_df[c] = pd.to_numeric(elec_df[c], errors="coerce")
    # column-wise median fill (global 0 fill보다 나음)
    for c in elec_df.columns:
        med = elec_df[c].median()
        elec_df[c] = elec_df[c].fillna(med if not np.isnan(med) else 0.0)

    n_rdkit  = sum(1 for c in elec_df.columns if c.startswith("qm_") and "mrd" not in c and "xtb" not in c)
    n_mrd    = sum(1 for c in elec_df.columns if "mrd" in c)
    n_xtb    = sum(1 for c in elec_df.columns if "xtb" in c)
    logger.info(f"  Electronic features: "
                f"RDKit={n_rdkit} | Mordred={n_mrd} | xTB={n_xtb} | "
                f"Total={len(elec_df.columns)}")

    return elec_df


def get_available_levels() -> dict:
    """설치된 QM 라이브러리 레벨 반환."""
    return {
        "rdkit":   _HAS_RDKIT,
        "mordred": _HAS_MORDRED,
        "xtb":     _HAS_XTB,
    }


def install_guide() -> str:
    """미설치 라이브러리 설치 안내."""
    lines = ["QM descriptor 라이브러리 설치 안내:"]
    if not _HAS_RDKIT:
        lines.append("  RDKit:   conda install -c conda-forge rdkit")
    if not _HAS_MORDRED:
        lines.append("  Mordred: pip install mordred")
    if not _HAS_XTB:
        lines.append("  xTB:     conda install -c conda-forge xtb-python  (권장)")
        lines.append("           또는: pip install xtb-python")
    if _HAS_RDKIT and _HAS_MORDRED and _HAS_XTB:
        lines.append("  모든 레벨 설치됨 [OK]")
    return "\n".join(lines)
