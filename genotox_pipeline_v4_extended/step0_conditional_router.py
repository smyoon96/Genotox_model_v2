"""
step0_conditional_router.py — Substance-Type Conditional Preprocessing
=======================================================================
물질 유형에 따라 전처리 경로를 분기하는 라우터.

기존 파이프라인 (baseline): 모든 물질에 동일한 전처리 적용
제안 파이프라인 (conditional): 물질 유형별 분기
    ├─ organic_single    → 표준 파이프라인 (현행 그대로)
    ├─ ionic_salt         → salt stripping + counterion feature 보존
    ├─ organometallic     → 제거 안 함, metal-aware feature 인코딩
    └─ multi_component    → fragment-ensemble 인코딩

Usage:
    from step0_conditional_router import ConditionalRouter
    router = ConditionalRouter()
    df = router.route_and_preprocess(df)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# rdMolStandardize: RDKit 버전에 따라 import 경로가 다름
try:
    from rdkit.Chem import rdMolStandardize
except ImportError:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        rdMolStandardize = None

logger = logging.getLogger(__name__)

# ─── Metal detection ───
METAL_REGEX = re.compile(
    r'\[(Na|K|Li|Ca|Mg|Fe|Cu|Zn|Mn|Co|Ni|Cr|Cd|Hg|Pb|As|Sb|Bi|Sn|'
    r'Ti|V|Mo|W|Pt|Pd|Au|Ag|Al|Ba|Sr|Se|Te|Nd|La|Ce|Cs|Rh|Nb|Zr|'
    r'Ga|In|Tl|Be)[+\-\d@H]*\]'
)


class ConditionalRouter:
    """물질 유형별 조건부 전처리 라우터."""
    
    def __init__(self, strategy: str = 'conditional'):
        """
        Args:
            strategy: 'baseline' (기존 전처리) or 'conditional' (조건부 분기)
        """
        self.strategy = strategy
        if rdMolStandardize is not None:
            self._lfc = rdMolStandardize.LargestFragmentChooser()
            self._uc = rdMolStandardize.Uncharger()
        else:
            self._lfc = None
            self._uc = None
    
    # ─── Classification ───
    
    @staticmethod
    def classify(smi: str) -> str:
        """
        SMILES를 4가지 유형으로 분류.
        
        Returns:
            'organic_single' | 'ionic_salt' | 'organometallic' | 'multi_component'
        """
        if pd.isna(smi) or not str(smi).strip():
            return 'unknown'
        smi = str(smi).strip()
        
        fragments = smi.split('.')
        metals = METAL_REGEX.findall(smi)
        has_carbon = 'C' in smi or 'c' in smi
        
        # No metal, single fragment → organic single
        if not metals and len(fragments) == 1:
            return 'organic_single'
        
        # No metal, multi fragment → multi_component (organic mixture)
        if not metals and len(fragments) > 1:
            return 'multi_component'
        
        # Has metal
        if metals:
            if not has_carbon:
                return 'organometallic'  # inorganic but treat as special
            
            # Check if metal is in a carbon-containing fragment (M-C bond)
            for frag in fragments:
                frag_metals = METAL_REGEX.findall(frag)
                if frag_metals and ('C' in frag or 'c' in frag):
                    return 'organometallic'
            
            return 'ionic_salt'
        
        return 'organic_single'
    
    # ─── Preprocessing routes ───
    
    def _preprocess_standard(self, smi: str) -> Tuple[str, Dict]:
        """표준 전처리: salt strip + neutralize + canonicalize."""
        meta = {'route': 'standard'}
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi, {**meta, 'error': 'parse_fail'}
            if self._lfc is not None:
                mol = self._lfc.choose(mol)
            if self._uc is not None:
                mol = self._uc.uncharge(mol)
            result = Chem.MolToSmiles(mol, canonical=True)
            return result, meta
        except Exception as e:
            return smi, {**meta, 'error': str(e)}
    
    def _preprocess_ionic_salt(self, smi: str) -> Tuple[str, Dict]:
        """
        Ionic salt: salt strip하되 counterion 정보를 metadata로 보존.
        """
        meta = {'route': 'ionic_salt'}
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi, {**meta, 'error': 'parse_fail'}
            
            fragments = smi.split('.')
            
            # Identify counterions (metal-containing fragments without C)
            counterions = []
            organic_frags = []
            for frag in fragments:
                frag_metals = METAL_REGEX.findall(frag)
                has_c = 'C' in frag or 'c' in frag
                if frag_metals and not has_c:
                    counterions.append(frag)
                else:
                    organic_frags.append(frag)
            
            # Process organic part
            organic_smi = '.'.join(organic_frags) if organic_frags else smi
            organic_mol = Chem.MolFromSmiles(organic_smi)
            if organic_mol:
                if self._lfc is not None:
                    organic_mol = self._lfc.choose(organic_mol)
                if self._uc is not None:
                    organic_mol = self._uc.uncharge(organic_mol)
                result_smi = Chem.MolToSmiles(organic_mol, canonical=True)
            else:
                result_smi = organic_smi
            
            # Preserve counterion info
            meta['counterions'] = counterions
            meta['counterion_elements'] = list(set(
                m for ci in counterions for m in METAL_REGEX.findall(ci)
            ))
            meta['n_counterions'] = len(counterions)
            
            # Compute counterion MW
            ci_mw = 0
            for ci in counterions:
                ci_mol = Chem.MolFromSmiles(ci)
                if ci_mol:
                    ci_mw += Descriptors.MolWt(ci_mol)
            meta['counterion_mw'] = ci_mw
            
            return result_smi, meta
            
        except Exception as e:
            return smi, {**meta, 'error': str(e)}
    
    def _preprocess_organometallic(self, smi: str) -> Tuple[str, Dict]:
        """
        Organometallic: 제거하지 않음. Metal-aware feature 인코딩.
        Canonicalize만 수행.
        """
        meta = {'route': 'organometallic'}
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi, {**meta, 'error': 'parse_fail'}
            
            # Canonicalize only (no salt strip, no metal removal)
            result_smi = Chem.MolToSmiles(mol, canonical=True)
            
            # Extract metal-aware features
            metals = METAL_REGEX.findall(smi)
            meta['metal_elements'] = list(set(metals))
            meta['n_metal_atoms'] = len(metals)
            
            # Alkyl chain analysis on metal center
            meta['metal_alkyl_info'] = self._analyze_metal_alkyl(mol, metals)
            
            return result_smi, meta
            
        except Exception as e:
            return smi, {**meta, 'error': str(e)}
    
    def _preprocess_multi_component(self, smi: str) -> Tuple[str, Dict]:
        """
        Multi-component: fragment-ensemble 인코딩.
        각 fragment의 descriptor를 개별 계산 후 통합.
        """
        meta = {'route': 'multi_component'}
        try:
            fragments = smi.split('.')
            meta['n_fragments'] = len(fragments)
            
            # Find largest fragment (standard approach)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi, {**meta, 'error': 'parse_fail'}
            
            if self._lfc is not None:
                largest = self._lfc.choose(mol)
            else:
                largest = mol
            if self._uc is not None:
                largest = self._uc.uncharge(largest)
            result_smi = Chem.MolToSmiles(largest, canonical=True)
            
            # Fragment ensemble: compute descriptors for all fragments
            frag_mws = []
            frag_logps = []
            for frag in fragments:
                frag_mol = Chem.MolFromSmiles(frag)
                if frag_mol:
                    frag_mws.append(Descriptors.MolWt(frag_mol))
                    frag_logps.append(Descriptors.MolLogP(frag_mol))
            
            meta['total_mw'] = sum(frag_mws)
            meta['largest_frag_mw_ratio'] = (max(frag_mws) / sum(frag_mws) 
                                              if frag_mws else 1.0)
            meta['frag_logp_range'] = (max(frag_logps) - min(frag_logps)
                                       if len(frag_logps) > 1 else 0)
            meta['frag_mw_std'] = np.std(frag_mws) if len(frag_mws) > 1 else 0
            
            return result_smi, meta
            
        except Exception as e:
            return smi, {**meta, 'error': str(e)}
    
    def _analyze_metal_alkyl(self, mol, metals: List[str]) -> Dict:
        """Organometallic의 metal 주변 alkyl chain 분석."""
        info = {'max_chain': 0, 'min_chain': 0, 'n_alkyl': 0}
        try:
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in metals:
                    chains = []
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 6:
                            # BFS to count carbon chain from this neighbor
                            chain_len = self._bfs_chain(mol, neighbor.GetIdx(), atom.GetIdx())
                            chains.append(chain_len)
                    
                    if chains:
                        info['max_chain'] = max(chains)
                        info['min_chain'] = min(chains)
                        info['n_alkyl'] = len(chains)
                    break  # First metal atom
        except Exception:
            pass
        return info
    
    @staticmethod
    def _bfs_chain(mol, start_idx: int, exclude_idx: int) -> int:
        """BFS로 carbon chain 길이 계산."""
        visited = {start_idx, exclude_idx}
        queue = [start_idx]
        length = 0
        while queue:
            next_queue = []
            for idx in queue:
                atom = mol.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    nidx = neighbor.GetIdx()
                    if nidx not in visited and neighbor.GetAtomicNum() == 6:
                        visited.add(nidx)
                        next_queue.append(nidx)
            if next_queue:
                length += 1
            queue = next_queue
        return length + 1  # Include start atom
    
    # ─── Main routing ───
    
    def route_and_preprocess(
        self,
        df: pd.DataFrame,
        smi_col: str = 'SMILES',
        output_col: str = '_analysis_smiles',
    ) -> pd.DataFrame:
        """
        전체 DataFrame에 조건부 전처리 적용.
        
        Returns:
            DataFrame with added columns:
                - _analysis_smiles: 전처리 결과 SMILES
                - _substance_type: 분류 결과
                - _route_*: route-specific metadata
        """
        df = df.copy()
        
        results = []
        for idx, row in df.iterrows():
            raw_smi = str(row.get(smi_col, ''))
            
            # Classify
            stype = self.classify(raw_smi)
            
            if self.strategy == 'baseline':
                # 모든 물질에 표준 전처리
                processed_smi, meta = self._preprocess_standard(raw_smi)
            elif self.strategy == 'conditional':
                # 물질 유형별 분기
                if stype == 'organic_single':
                    processed_smi, meta = self._preprocess_standard(raw_smi)
                elif stype == 'ionic_salt':
                    processed_smi, meta = self._preprocess_ionic_salt(raw_smi)
                elif stype == 'organometallic':
                    processed_smi, meta = self._preprocess_organometallic(raw_smi)
                elif stype == 'multi_component':
                    processed_smi, meta = self._preprocess_multi_component(raw_smi)
                else:
                    processed_smi, meta = self._preprocess_standard(raw_smi)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            results.append({
                'idx': idx,
                'processed_smi': processed_smi,
                'substance_type': stype,
                **{f'route_{k}': v for k, v in meta.items() 
                   if not isinstance(v, (list, dict))},
            })
        
        result_df = pd.DataFrame(results).set_index('idx')
        
        df[output_col] = result_df['processed_smi']
        df['_substance_type'] = result_df['substance_type']
        
        # Add route metadata columns
        for col in result_df.columns:
            if col.startswith('route_') and col not in ['route_route']:
                df[f'_{col}'] = result_df[col]
        
        # Summary
        type_counts = df['_substance_type'].value_counts()
        logger.info(f"Conditional routing ({self.strategy}):")
        for stype, count in type_counts.items():
            logger.info(f"  {stype}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def get_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Route metadata를 numeric feature로 변환.
        조건부 전처리에서만 의미 있는 feature.
        """
        features = {}
        
        # Substance type one-hot
        for stype in ['organic_single', 'ionic_salt', 'organometallic', 'multi_component']:
            features[f'type_{stype}'] = (df['_substance_type'] == stype).astype(int)
        
        # Route-specific numeric features
        for col in df.columns:
            if col.startswith('_route_') and col != '_route_route':
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.notna().any():
                    features[col.replace('_route_', 'rt_')] = numeric_col
        
        return pd.DataFrame(features, index=df.index).fillna(0)


# ─── Comparison Framework ───

def run_comparison(
    df: pd.DataFrame,
    smi_col: str = 'SMILES',
    label_col: str = 'label',
) -> Dict:
    """
    Baseline vs Conditional preprocessing 비교.
    
    Returns:
        dict with classification results and type-specific metrics.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        matthews_corrcoef, balanced_accuracy_score,
        f1_score, roc_auc_score,
    )
    
    results = {}
    
    for strategy in ['baseline', 'conditional']:
        router = ConditionalRouter(strategy=strategy)
        processed = router.route_and_preprocess(df, smi_col=smi_col)
        
        # Extract features
        from step4c_delta_descriptors import (
            extract_delta_features, extract_electronic_descriptors,
        )
        from step4b_sa_features import extract_sa_features
        
        # Build feature matrix
        delta_df = extract_delta_features(processed, raw_smi_col=smi_col)
        elec_df = extract_electronic_descriptors(processed)
        sa_df = extract_sa_features(processed)
        route_df = router.get_route_features(processed)
        
        # Morgan FP
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        smi_col_proc = '_analysis_smiles'
        fp_matrix = np.zeros((len(processed), 1024), dtype=np.int8)
        for i, smi in enumerate(processed[smi_col_proc]):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    arr = np.zeros(1024, dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fp_matrix[i] = arr
            except:
                pass
        fp_df = pd.DataFrame(fp_matrix, 
                             columns=[f'fp_{i:04d}' for i in range(1024)],
                             index=processed.index)
        
        # Combine
        X = pd.concat([fp_df, delta_df, elec_df, sa_df, route_df], axis=1)
        X = X.select_dtypes(include=[np.number]).fillna(0)
        y = processed[label_col].astype(int)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {'mcc': [], 'bacc': [], 'f1': [], 'auc': []}
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=42
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
            metrics['bacc'].append(balanced_accuracy_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            try:
                metrics['auc'].append(roc_auc_score(y_test, y_proba))
            except:
                metrics['auc'].append(np.nan)
        
        results[strategy] = {
            'mean_mcc': np.mean(metrics['mcc']),
            'std_mcc': np.std(metrics['mcc']),
            'mean_bacc': np.mean(metrics['bacc']),
            'mean_f1': np.mean(metrics['f1']),
            'mean_auc': np.nanmean(metrics['auc']),
            'n_features': X.shape[1],
            'type_distribution': processed['_substance_type'].value_counts().to_dict(),
        }
        
        # Subset analysis: metal compounds only
        metal_mask = processed['_substance_type'].isin(['ionic_salt', 'organometallic'])
        if metal_mask.sum() > 10:
            metal_metrics = {'mcc': [], 'bacc': []}
            for train_idx, test_idx in skf.split(X, y):
                test_metal = [i for i in test_idx if metal_mask.iloc[i]]
                if len(test_metal) > 0:
                    y_pred_m = clf.predict(X.iloc[test_metal])
                    y_test_m = y.iloc[test_metal]
                    if len(set(y_test_m)) > 1:
                        metal_metrics['mcc'].append(matthews_corrcoef(y_test_m, y_pred_m))
                        metal_metrics['bacc'].append(balanced_accuracy_score(y_test_m, y_pred_m))
            
            if metal_metrics['mcc']:
                results[strategy]['metal_subset_mcc'] = np.mean(metal_metrics['mcc'])
                results[strategy]['metal_subset_bacc'] = np.mean(metal_metrics['bacc'])
    
    return results
