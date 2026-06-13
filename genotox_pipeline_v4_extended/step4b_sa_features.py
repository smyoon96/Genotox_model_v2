"""
step4b_sa_features.py — Benigni/Bossa Structural Alert (SA) Feature Extraction
================================================================================
51개 SA pattern을 feature로 변환. 구조정의_재정의 notebook에서 추출.

Usage:
    from step4b_sa_features import extract_sa_features
    sa_df = extract_sa_features(df, endpoint="ames")
"""

import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from typing import Dict

logger = logging.getLogger(__name__)

# ─── SA SMARTS patterns (from 구조정의_재정의-Copy1.ipynb) ───
# 일부 SA는 list (OR logic: 모든 sub-pattern 매치 수 중 min 사용)
SA_PATTERNS: Dict[str, object] = {
    'SA1': '[!$([OH1,SH1])]C(=O)[Br,Cl,F,I]',
    'SA2': [
        'S([!$([OH1,SH1])])(=O)(=O)O([$(C([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])),$(C([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])C([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])),$([CH2]c1ccccc1)])',
        'P(=O)([!$([OH1,SH1])])(O([$(C([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])),$([CH2]c1ccccc1)]))(O([$(C([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])([#1,Cl,Br,I,F])),$([CH2]c1ccccc1)]))',
    ],
    'SA3': '[NX3;!$([NX3][CX3]=[OX1,SX1]);!$([NX3][#7]);!$([NX3]c)](=O)',
    'SA4': 'c[NX3;!$([NX3][CX3]=[OX1,SX1]);!$([NX3][#7])](=O)',
    'SA5': 'c[NX2;!$([NX2][CX3]=[OX1,SX1])](=O)',
    'SA6': '[NX3H2,NX3H1,NX2H1,NX2H0]-[NX3H2,NX3H1,NX2H1,NX2H0]',
    'SA7': '[CX4;!$([CX4]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I])][F,Cl,Br,I]',
    'SA8': '[CX3H1](=O)',
    'SA9': '[CX3H1](=O)[CX4H2][CX3H1](=O)',
    'SA10': '[NX2]=[NX2]',
    'SA11': '[#6]=[#6]-[#6]=[#8]',
    'SA12': 'c[N+](=O)[O-]',
    'SA13': 'O=[CH]c',
    'SA14': '[NX1]=[NX2]=[NX1]',
    'SA15': 'C1[NX3H1]C1',
    'SA16': '[SX2][CX4]',
    'SA17': 'C1OC1',
    'SA18': '[OH1,SH1][CX3]=[OX1]',
    'SA19': '[NX3]([CH2][Cl,Br,I])[CH2][Cl,Br,I]',
    'SA20': '[SX2][SX2]',
    'SA21': '[NH2]c1cccc2ccccc12',
    'SA22': '[NH2]c1ccccc1',
    'SA23': '[NH1;!$([NH1]C=O)]c1ccccc1',
    'SA24': 'c1cc2c(cc1)cc1ccc3cccc4cccc2c1c34',
    'SA25': 'c1ccc2c(c1)cc1ccc3ccccc3c1c2',
    'SA26': 'c1ccc2c(c1)Cc1ccccc1C2',
    'SA27': 'C=[CH]c',
    'SA28': '[CX4;$([CH2]),$(C([#1])([#1]))][CX3](=[OX1])[F,Cl,Br,I]',
    'SA29': '[$(O=Cc1ccc(cc1)[N+](=O)[O-]),$(O=Cc1cccc(c1)[N+](=O)[O-]),$(O=Cc1ccccc1[N+](=O)[O-])]',
    'SA30': 'c1ccc2c(c1)[nX3]c1ccccc1[nX3]2',
    'SA31': 'a1aaa2a(a1)aaa1a(aaa3aaaaa31)a2',
    'SA31b': 'a1aaa2a(a1)aaa1aaaa3aaaaa3a1a2',
    'SA31c': 'a1aaa2a(a1)aaa1a(aaa3aaaa4aaaaa43)a1a2',
    'SA40': '[CX3](=O)[OX2][CX3](=O)',
    'SA41': '[#16X2H0;!$([#16X2H0]c)]',
    'SA42': '[NX3;$([NX3H2]),$([NX3H1]),$([NX3H0])]c1c(cccc1)[OH]',
    'SA43': 'c1ccc(cc1)N=Nc1ccccc1',
    'SA44': '[CH2]=[CH][CH]=[CH2]',
    'SA45': 'O=C1C=CC(=O)C=C1',
    'SA46': 'O=C1C(=O)c2ccccc2C1=O',
    'SA47': 'c1ccc(cc1)c1ccccc1',
    'SA48': '[N;X3;H2,H1;!$([N][!#6]);!$([N]*~[#7,#8,#15,#16])]',
    'SA49': '[#6]S(=O)(=O)[#8]',
    'SA50': 'c1cc([nH]c1)c1ccc[nH]1',
    'SA51': '[CH2]=[CH2]',
    'SA52': 'c1ccccc1-c1ccccn1',
    'SA53': '[CX3H0](=O)([#6])[#6]',
    'SA54': 'c1ccncc1',
    'SA55': 'N#[CX2]',
    'SA56': 'O=N',
    'SA57': 'O=[NX3](~[OX1])c',
}

# Pre-compile patterns
_SA_COMPILED = {}

def _compile_sa():
    """SA SMARTS를 Mol 객체로 컴파일 (모듈 로드 시 1회)"""
    global _SA_COMPILED
    for name, pat in SA_PATTERNS.items():
        if isinstance(pat, list):
            _SA_COMPILED[name] = [Chem.MolFromSmarts(p) for p in pat]
        else:
            _SA_COMPILED[name] = Chem.MolFromSmarts(pat)

_compile_sa()


def _match_sa(mol, name: str) -> int:
    """SA pattern 매칭 수 반환. list SA는 sub-pattern별 min."""
    pat = _SA_COMPILED.get(name)
    if pat is None or mol is None:
        return 0
    try:
        if isinstance(pat, list):
            counts = []
            for p in pat:
                if p is not None:
                    counts.append(len(mol.GetSubstructMatches(p, maxMatches=100)))
                else:
                    counts.append(0)
            return min(counts) if counts else 0
        else:
            return len(mol.GetSubstructMatches(pat, maxMatches=100))
    except Exception:
        return 0


def extract_sa_features(df: pd.DataFrame, endpoint: str = "ames") -> pd.DataFrame:
    """
    51개 SA structural alert pattern을 feature로 추출.
    
    Returns:
        DataFrame with columns: SA1_count, SA1_hit, ..., SA57_count, SA57_hit,
                                sa_total_hits, sa_total_count
    """
    try:
        from pipeline_v2_core import find_smi
        smi_col = find_smi(df)
    except ImportError:
        smi_col = next((c for c in ['_analysis_smiles', 'canonical_smiles', 'SMILES', 'smiles']
                        if c in df.columns), df.columns[0])
    
    mols = df[smi_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
    
    records = []
    for mol in mols:
        row = {}
        for sa_name in SA_PATTERNS:
            count = _match_sa(mol, sa_name)
            row[f"{sa_name}_count"] = count
            row[f"{sa_name}_hit"] = 1 if count > 0 else 0
        records.append(row)
    
    sa_df = pd.DataFrame(records, index=df.index)
    
    # Aggregate
    hit_cols = [c for c in sa_df.columns if c.endswith('_hit')]
    count_cols = [c for c in sa_df.columns if c.endswith('_count')]
    sa_df['sa_total_hits'] = sa_df[hit_cols].sum(axis=1)
    sa_df['sa_total_count'] = sa_df[count_cols].sum(axis=1)
    
    logger.info(f"  [{endpoint}] SA features: {sa_df.shape[1]} columns "
                f"(mean hits/mol: {sa_df['sa_total_hits'].mean():.2f})")
    return sa_df
