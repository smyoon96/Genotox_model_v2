"""
step4c_delta_descriptors.py — Preprocessing-Aware & Extended Descriptors
=========================================================================
전처리 전후 차이(delta)를 descriptor로 변환 + 확장 descriptor 계산.

핵심 아이디어: "전처리가 무엇을 바꿨는지" 자체가 예측에 유용한 정보.

Usage:
    from step4c_delta_descriptors import (
        extract_delta_features,
        extract_extended_descriptors,
        extract_electronic_descriptors,
    )
"""

import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors,
    Fragments, Crippen, MolSurf, EState,
    GraphDescriptors,
)
from rdkit import DataStructs

logger = logging.getLogger(__name__)

# ─── Metal detection regex (43 elements) ───
METAL_REGEX = re.compile(
    r'\[(Na|K|Li|Ca|Mg|Fe|Cu|Zn|Mn|Co|Ni|Cr|Cd|Hg|Pb|As|Sb|Bi|Sn|'
    r'Ti|V|Mo|W|Pt|Pd|Au|Ag|Al|Ba|Sr|Se|Te|Nd|La|Ce|Cs|Rh|Nb|Zr|'
    r'Ga|In|Tl|Be)[+\-\d@H]*\]'
)

TRANSITION_METALS = {
    'Sn','Zn','Cu','Fe','Cr','Hg','Co','Ni','Ru','Pd','Ti',
    'Pb','Mn','Rh','Pt','Ir','Ag','Au','Cd','Mo','Zr','V',
    'Nb','W','Ta','Sc','Hf','Re','Os'
}
METALLOIDS = {'B','Sb','Bi','As','Se','Te','Ge','Si'}


# =============================================
#  1. Delta Features (preprocessing-aware)
# =============================================

def _compute_morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    """Morgan FP를 numpy array로 반환. 실패 시 None."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def _tanimoto(fp1, fp2) -> float:
    """두 numpy FP 간 Tanimoto. None이면 NaN."""
    if fp1 is None or fp2 is None:
        return np.nan
    on1 = fp1.sum()
    on2 = fp2.sum()
    common = (fp1 & fp2).sum()
    denom = on1 + on2 - common
    return common / denom if denom > 0 else 1.0


def _detect_metals(smi: str) -> List[str]:
    """SMILES에서 금속 원소 목록 추출."""
    if pd.isna(smi):
        return []
    return METAL_REGEX.findall(str(smi))


def _classify_bonding(smi: str) -> str:
    """
    Ionic salt / organometallic / organic / inorganic 분류.
    Multi-fragment SMILES 기준.
    """
    if pd.isna(smi):
        return 'unknown'
    smi = str(smi)
    metals = _detect_metals(smi)
    if not metals:
        return 'organic'
    
    # Check if contains carbon
    if 'C' not in smi and 'c' not in smi:
        return 'inorganic'
    
    # Check multi-fragment
    fragments = smi.split('.')
    if len(fragments) == 1:
        # Single fragment with metal = organometallic candidate
        return 'organometallic'
    
    # Multi-fragment: check if metal is in a carbon-containing fragment
    for frag in fragments:
        frag_metals = _detect_metals(frag)
        if frag_metals and ('C' in frag or 'c' in frag):
            return 'organometallic'
    
    return 'ionic_salt'


def _longest_carbon_chain(mol) -> int:
    """최장 탄소 사슬 길이 (근사치 — 선형 경로 기반)."""
    if mol is None:
        return 0
    try:
        from rdkit.Chem import rdmolops
        # Get carbon-only subgraph
        carbon_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        if not carbon_atoms:
            return 0
        
        # Find longest path among carbon atoms
        max_path = 0
        adj = Chem.GetAdjacencyMatrix(mol)
        for start in carbon_atoms[:min(len(carbon_atoms), 50)]:  # limit for speed
            visited = {start}
            queue = [(start, 1)]
            while queue:
                node, depth = queue.pop(0)
                max_path = max(max_path, depth)
                for neighbor in range(adj.shape[0]):
                    if adj[node, neighbor] and neighbor not in visited and neighbor in carbon_atoms:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return max_path
    except Exception:
        return 0


def extract_delta_features(
    df: pd.DataFrame,
    raw_smi_col: str = 'SMILES_raw',
    pre_smi_col: str = '_analysis_smiles',
    std_smi_col: str = 'standardized_smiles',
    endpoint: str = 'ames',
) -> pd.DataFrame:
    """
    전처리 전후 차이(delta)를 descriptor로 추출.
    
    Features:
        - self_T_std: 원본 vs 표준화 Tanimoto
        - self_T_pre: 원본 vs 전처리 Tanimoto
        - delta_MW: MW 변화
        - delta_LogP: LogP 변화
        - metal_detected_raw: 원본에 금속 존재 여부
        - metal_removed: 전처리 후 금속 소실 여부
        - metal_element_*: 원소별 존재 여부 (one-hot)
        - bonding_type_*: 결합 유형 (one-hot)
        - n_fragments_raw: 원본 fragment 수
        - n_fragments_pre: 전처리 후 fragment 수
        - counterion_present: counterion 존재 여부
        - longest_chain: 최장 탄소 사슬
        - tautomer_changed: 표준화에서 tautomer 변환 여부
    """
    records = []
    
    # Determine available columns
    has_raw = raw_smi_col in df.columns
    has_std = std_smi_col in df.columns
    has_pre = pre_smi_col in df.columns
    
    if not has_raw:
        # Fallback: try other column names
        for candidate in ['SMILES', 'SMILES_raw', 'smiles']:
            if candidate in df.columns:
                raw_smi_col = candidate
                has_raw = True
                break
    
    if not has_pre:
        for candidate in ['canonical_smiles', 'analysis_smiles']:
            if candidate in df.columns:
                pre_smi_col = candidate
                has_pre = True
                break
    
    for idx, row in df.iterrows():
        rec = {}
        
        raw_smi = str(row.get(raw_smi_col, '')) if has_raw else ''
        std_smi = str(row.get(std_smi_col, '')) if has_std else ''
        pre_smi = str(row.get(pre_smi_col, '')) if has_pre else ''
        
        # ── Self-Tanimoto ──
        if has_raw and has_std and raw_smi and std_smi:
            fp_raw = _compute_morgan_fp(raw_smi)
            fp_std = _compute_morgan_fp(std_smi)
            rec['self_T_std'] = _tanimoto(fp_raw, fp_std)
        else:
            fp_raw = _compute_morgan_fp(raw_smi) if raw_smi else None
            rec['self_T_std'] = np.nan
        
        if has_raw and has_pre and raw_smi and pre_smi:
            fp_pre = _compute_morgan_fp(pre_smi)
            rec['self_T_pre'] = _tanimoto(fp_raw, fp_pre)
        else:
            rec['self_T_pre'] = np.nan
        
        # ── MW / LogP delta ──
        mol_raw = Chem.MolFromSmiles(raw_smi) if raw_smi else None
        mol_pre = Chem.MolFromSmiles(pre_smi) if pre_smi else None
        
        mw_raw = Descriptors.MolWt(mol_raw) if mol_raw else np.nan
        mw_pre = Descriptors.MolWt(mol_pre) if mol_pre else np.nan
        logp_raw = Descriptors.MolLogP(mol_raw) if mol_raw else np.nan
        logp_pre = Descriptors.MolLogP(mol_pre) if mol_pre else np.nan
        
        rec['MW_raw'] = mw_raw
        rec['delta_MW'] = mw_raw - mw_pre if not (np.isnan(mw_raw) or np.isnan(mw_pre)) else np.nan
        rec['delta_LogP'] = logp_raw - logp_pre if not (np.isnan(logp_raw) or np.isnan(logp_pre)) else np.nan
        
        # ── Metal features ──
        metals_raw = _detect_metals(raw_smi)
        metals_pre = _detect_metals(pre_smi)
        
        rec['metal_detected_raw'] = 1 if metals_raw else 0
        rec['metal_removed'] = 1 if metals_raw and not metals_pre else 0
        rec['n_metals_raw'] = len(metals_raw)
        
        # One-hot for key elements
        for elem in ['Sn', 'Cr', 'Ni', 'Sb', 'Se', 'B', 'Zn', 'Cu', 'Fe', 'Pb', 'Hg', 'As', 'Cd']:
            rec[f'has_{elem}'] = 1 if elem in metals_raw else 0
        
        # Metal category
        has_tm = any(m in TRANSITION_METALS for m in metals_raw)
        has_met = any(m in METALLOIDS for m in metals_raw)
        rec['has_transition_metal'] = 1 if has_tm else 0
        rec['has_metalloid'] = 1 if has_met else 0
        
        # ── Bonding type ──
        bt = _classify_bonding(raw_smi)
        rec['is_organic'] = 1 if bt == 'organic' else 0
        rec['is_ionic_salt'] = 1 if bt == 'ionic_salt' else 0
        rec['is_organometallic'] = 1 if bt == 'organometallic' else 0
        rec['is_inorganic'] = 1 if bt == 'inorganic' else 0
        
        # ── Fragment features ──
        rec['n_fragments_raw'] = len(raw_smi.split('.')) if raw_smi else 0
        rec['n_fragments_pre'] = len(pre_smi.split('.')) if pre_smi else 0
        rec['fragments_removed'] = rec['n_fragments_raw'] - rec['n_fragments_pre']
        
        # ── Chain length ──
        rec['longest_chain'] = _longest_carbon_chain(mol_pre if mol_pre else mol_raw)
        
        # ── Tautomer change flag ──
        rec['tautomer_changed'] = 1 if (rec['self_T_std'] is not np.nan and 
                                         not np.isnan(rec.get('self_T_std', np.nan)) and 
                                         rec['self_T_std'] < 1.0) else 0
        
        # ── Severe distortion flags ──
        st = rec.get('self_T_pre', np.nan)
        rec['severe_distortion'] = 1 if (not np.isnan(st) and st < 0.5) else 0
        rec['moderate_distortion'] = 1 if (not np.isnan(st) and st < 0.8) else 0
        
        records.append(rec)
    
    delta_df = pd.DataFrame(records, index=df.index)
    logger.info(f"  [{endpoint}] Delta features: {delta_df.shape[1]} columns "
                f"(metal: {delta_df['metal_detected_raw'].sum()}, "
                f"severe_distortion: {delta_df['severe_distortion'].sum()})")
    return delta_df


# =============================================
#  2. Extended RDKit Descriptors (200+)
# =============================================

def extract_extended_descriptors(df: pd.DataFrame, endpoint: str = 'ames') -> pd.DataFrame:
    """
    RDKit의 전체 2D descriptor set (~200개) 추출.
    기존 physchem 13개를 포함하여 확장.
    """
    try:
        from pipeline_v2_core import find_smi
        smi_col = find_smi(df)
    except ImportError:
        smi_col = next((c for c in ['_analysis_smiles', 'canonical_smiles', 'SMILES']
                        if c in df.columns), df.columns[0])
    
    # RDKit descriptor list
    desc_list = Descriptors.descList  # [(name, func), ...]
    desc_names = [name for name, _ in desc_list]
    
    records = []
    for smi in df[smi_col]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            records.append({f"rdkit_{name}": np.nan for name in desc_names})
            continue
        
        rec = {}
        for name, func in desc_list:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rec[f"rdkit_{name}"] = func(mol)
            except Exception:
                rec[f"rdkit_{name}"] = np.nan
        records.append(rec)
    
    ext_df = pd.DataFrame(records, index=df.index)
    
    # Drop constant and all-NaN columns
    ext_df = ext_df.dropna(axis=1, how='all')
    nunique = ext_df.nunique()
    ext_df = ext_df.loc[:, nunique > 1]
    
    logger.info(f"  [{endpoint}] Extended RDKit descriptors: {ext_df.shape[1]} columns")
    return ext_df


# =============================================
#  3. Electronic / Pseudo-QM Descriptors
# =============================================

def extract_electronic_descriptors(df: pd.DataFrame, endpoint: str = 'ames') -> pd.DataFrame:
    """
    RDKit 기반 전자적 descriptor 추출.
    True HOMO/LUMO는 xTB/DFT가 필요하지만, RDKit로 근사 가능한 것들:
    
    - Gasteiger charges (min, max, mean, range)
    - EState indices
    - Chi/Kappa connectivity indices
    - Wildman-Crippen MR
    - LabuteASA
    - PEOE_VSA bins
    """
    try:
        from pipeline_v2_core import find_smi
        smi_col = find_smi(df)
    except ImportError:
        smi_col = next((c for c in ['_analysis_smiles', 'canonical_smiles', 'SMILES']
                        if c in df.columns), df.columns[0])
    
    records = []
    for smi in df[smi_col]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        rec = {}
        
        if mol is None:
            rec = {k: np.nan for k in [
                'gasteiger_min', 'gasteiger_max', 'gasteiger_mean', 'gasteiger_range',
                'gasteiger_abs_max', 'estate_min', 'estate_max', 'estate_range',
                'labute_asa', 'crippen_mr',
            ]}
            records.append(rec)
            continue
        
        try:
            # Gasteiger charges
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                q = atom.GetDoubleProp('_GasteigerCharge')
                if not np.isnan(q) and not np.isinf(q):
                    charges.append(q)
            
            if charges:
                rec['gasteiger_min'] = min(charges)
                rec['gasteiger_max'] = max(charges)
                rec['gasteiger_mean'] = np.mean(charges)
                rec['gasteiger_range'] = max(charges) - min(charges)
                rec['gasteiger_abs_max'] = max(abs(c) for c in charges)
            else:
                for k in ['gasteiger_min', 'gasteiger_max', 'gasteiger_mean',
                           'gasteiger_range', 'gasteiger_abs_max']:
                    rec[k] = np.nan
            
            # EState
            from rdkit.Chem.EState import EState as ES
            estates = ES.EStateIndices(mol)
            if len(estates) > 0:
                rec['estate_min'] = min(estates)
                rec['estate_max'] = max(estates)
                rec['estate_range'] = max(estates) - min(estates)
            else:
                rec['estate_min'] = rec['estate_max'] = rec['estate_range'] = np.nan
            
            # Surface/Shape
            rec['labute_asa'] = rdMolDescriptors.CalcLabuteASA(mol)
            rec['crippen_mr'] = Crippen.MolMR(mol)
            
            # Chi indices
            rec['chi0v'] = GraphDescriptors.Chi0v(mol)
            rec['chi1v'] = GraphDescriptors.Chi1v(mol)
            rec['chi2v'] = GraphDescriptors.Chi2v(mol)
            rec['chi3v'] = GraphDescriptors.Chi3v(mol)
            rec['chi0n'] = GraphDescriptors.Chi0n(mol)
            rec['chi1n'] = GraphDescriptors.Chi1n(mol)
            
            # Kappa indices
            rec['kappa1'] = GraphDescriptors.Kappa1(mol)
            rec['kappa2'] = GraphDescriptors.Kappa2(mol)
            rec['kappa3'] = GraphDescriptors.Kappa3(mol)
            
            # Hall-Kier alpha
            rec['hall_kier_alpha'] = GraphDescriptors.HallKierAlpha(mol)
            
            # PEOE_VSA (Partial Equalization of Orbital Electronegativities)
            peoe = MolSurf.PEOE_VSA_(mol)
            for i, val in enumerate(peoe):
                rec[f'peoe_vsa_{i}'] = val
            
            # SMR_VSA
            smr = MolSurf.SMR_VSA_(mol)
            for i, val in enumerate(smr):
                rec[f'smr_vsa_{i}'] = val
            
            # SlogP_VSA
            slogp = MolSurf.SlogP_VSA_(mol)
            for i, val in enumerate(slogp):
                rec[f'slogp_vsa_{i}'] = val
            
        except Exception as e:
            logger.debug(f"Electronic descriptor error: {e}")
            pass
        
        records.append(rec)
    
    elec_df = pd.DataFrame(records, index=df.index)
    elec_df = elec_df.dropna(axis=1, how='all')
    
    logger.info(f"  [{endpoint}] Electronic descriptors: {elec_df.shape[1]} columns")
    return elec_df


# =============================================
#  4. Mordred Descriptors (optional, 1600+)
# =============================================

def extract_mordred_descriptors(
    df: pd.DataFrame, endpoint: str = 'ames', ignore_3d: bool = True
) -> pd.DataFrame:
    """Mordred descriptor 추출 (설치되어 있을 경우)."""
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        logger.info(f"  [{endpoint}] Mordred not installed. Skipping.")
        return pd.DataFrame(index=df.index)
    
    try:
        from pipeline_v2_core import find_smi
        smi_col = find_smi(df)
    except ImportError:
        smi_col = next((c for c in ['_analysis_smiles', 'canonical_smiles', 'SMILES']
                        if c in df.columns), df.columns[0])
    
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    
    mols = []
    for smi in df[smi_col]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        mols.append(mol)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calc.pandas(mols, quiet=True, nproc=1)
    
    # Convert to numeric, drop errors
    result = result.apply(pd.to_numeric, errors='coerce')
    result.index = df.index
    result.columns = [f"mord_{c}" for c in result.columns]
    
    # Drop constant / all-NaN
    result = result.dropna(axis=1, how='all')
    nunique = result.nunique()
    result = result.loc[:, nunique > 1]
    
    logger.info(f"  [{endpoint}] Mordred descriptors: {result.shape[1]} columns")
    return result
