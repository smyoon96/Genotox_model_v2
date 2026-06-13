"""
run_strategy_comparison.py — Baseline vs Conditional Preprocessing 비교 실행
=============================================================================

Usage:
    python run_strategy_comparison.py --data_dir ./data --endpoint ames
    python run_strategy_comparison.py --data_dir ./data --endpoint all
"""

import sys
import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('comparison')

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))


def load_data(data_dir: Path, endpoint: str) -> pd.DataFrame:
    """데이터 로드. csv/xlsx 자동 탐색."""
    for ext in ['csv', 'xlsx']:
        path = data_dir / f"{endpoint}.{ext}"
        if path.exists():
            if ext == 'csv':
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            logger.info(f"Loaded {path}: {df.shape}")
            return df
    raise FileNotFoundError(f"No data file for {endpoint} in {data_dir}")


def run_single_endpoint(df: pd.DataFrame, endpoint: str, output_dir: Path):
    """단일 endpoint에 대한 baseline vs conditional 비교."""
    from step0_conditional_router import ConditionalRouter
    from step4b_sa_features import extract_sa_features
    from step4c_delta_descriptors import (
        extract_delta_features,
        extract_extended_descriptors,
        extract_electronic_descriptors,
    )
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        matthews_corrcoef, balanced_accuracy_score,
        f1_score, roc_auc_score, classification_report,
    )
    
    # Detect SMILES column
    smi_col = next((c for c in ['SMILES', 'canonical_smiles', 'smiles'] if c in df.columns), None)
    label_col = next((c for c in ['label', 'Label', 'TARGET', 'target'] if c in df.columns), None)
    
    if smi_col is None or label_col is None:
        logger.error(f"Cannot find SMILES ({smi_col}) or label ({label_col}) column")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Endpoint: {endpoint} | n={len(df)} | pos={df[label_col].sum()} "
                f"({df[label_col].mean()*100:.1f}%)")
    logger.info(f"{'='*60}")
    
    all_results = {}
    
    for strategy in ['baseline', 'conditional']:
        logger.info(f"\n--- Strategy: {strategy} ---")
        
        # 1. Route and preprocess
        router = ConditionalRouter(strategy=strategy)
        processed = router.route_and_preprocess(df, smi_col=smi_col)
        
        # 2. Feature extraction
        logger.info("Extracting features...")
        
        # 2a. Morgan FP
        proc_smi = '_analysis_smiles'
        fp_bits = 1024
        fp_matrix = np.zeros((len(processed), fp_bits), dtype=np.int8)
        for i, smi in enumerate(processed[proc_smi]):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_bits)
                    arr = np.zeros(fp_bits, dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fp_matrix[i] = arr
            except:
                pass
        fp_df = pd.DataFrame(fp_matrix, columns=[f'fp_{i:04d}' for i in range(fp_bits)],
                             index=processed.index)
        
        # 2b. SA features
        sa_df = extract_sa_features(processed, endpoint)
        
        # 2c. Delta features (preprocessing-aware)
        delta_df = extract_delta_features(
            processed, raw_smi_col=smi_col, pre_smi_col=proc_smi, endpoint=endpoint
        )
        
        # 2d. Electronic descriptors
        elec_df = extract_electronic_descriptors(processed, endpoint)
        
        # 2e. Extended RDKit descriptors
        ext_df = extract_extended_descriptors(processed, endpoint)
        
        # 2f. Route features (conditional only에서 유의미)
        route_df = router.get_route_features(processed)
        
        # 3. Combine features
        feature_blocks = {
            'fp': fp_df,
            'sa': sa_df,
            'delta': delta_df,
            'electronic': elec_df,
            'extended': ext_df,
            'route': route_df,
        }
        
        X_full = pd.concat(feature_blocks.values(), axis=1)
        X_full = X_full.select_dtypes(include=[np.number])
        
        # Remove constant columns
        nunique = X_full.nunique()
        X_full = X_full.loc[:, nunique > 1]
        X_full = X_full.fillna(0).replace([np.inf, -np.inf], 0)
        
        y = processed[label_col].astype(int)
        
        logger.info(f"Feature matrix: {X_full.shape}")
        for block_name, block_df in feature_blocks.items():
            n_cols = len([c for c in block_df.columns if c in X_full.columns])
            logger.info(f"  {block_name}: {n_cols} features")
        
        # 4. Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y)):
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42
            )
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            fold_result = {
                'fold': fold,
                'mcc': matthews_corrcoef(y_test, y_pred),
                'bacc': balanced_accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
            try:
                fold_result['auc'] = roc_auc_score(y_test, y_proba)
            except:
                fold_result['auc'] = np.nan
            
            # Subset analysis
            if '_substance_type' in processed.columns:
                test_types = processed.iloc[test_idx]['_substance_type']
                for stype in ['organic_single', 'ionic_salt', 'organometallic']:
                    mask = test_types == stype
                    if mask.sum() > 5 and len(set(y_test[mask])) > 1:
                        fold_result[f'{stype}_mcc'] = matthews_corrcoef(
                            y_test[mask], y_pred[mask.values]
                        )
                        fold_result[f'{stype}_n'] = mask.sum()
            
            fold_metrics.append(fold_result)
        
        fold_df = pd.DataFrame(fold_metrics)
        
        # Summary
        summary = {
            'strategy': strategy,
            'endpoint': endpoint,
            'n': len(df),
            'n_features': X_full.shape[1],
        }
        for metric in ['mcc', 'bacc', 'f1', 'auc']:
            vals = fold_df[metric].dropna()
            summary[f'{metric}_mean'] = vals.mean()
            summary[f'{metric}_std'] = vals.std()
        
        # Subset summaries
        for stype in ['organic_single', 'ionic_salt', 'organometallic']:
            col = f'{stype}_mcc'
            if col in fold_df.columns:
                vals = fold_df[col].dropna()
                if len(vals) > 0:
                    summary[f'{stype}_mcc_mean'] = vals.mean()
                    summary[f'{stype}_n_mean'] = fold_df[f'{stype}_n'].mean() if f'{stype}_n' in fold_df.columns else 0
        
        # Type distribution
        if '_substance_type' in processed.columns:
            summary['type_dist'] = processed['_substance_type'].value_counts().to_dict()
        
        all_results[strategy] = summary
        
        logger.info(f"\n{strategy} results:")
        logger.info(f"  MCC:  {summary['mcc_mean']:.4f} ± {summary['mcc_std']:.4f}")
        logger.info(f"  BACC: {summary['bacc_mean']:.4f}")
        logger.info(f"  F1:   {summary['f1_mean']:.4f}")
        logger.info(f"  AUC:  {summary['auc_mean']:.4f}")
        
        # Save fold details
        fold_df.to_csv(output_dir / f"{endpoint}_{strategy}_folds.csv", index=False)
    
    # 5. Comparison summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON: {endpoint}")
    logger.info(f"{'='*60}")
    
    comparison = pd.DataFrame([all_results['baseline'], all_results['conditional']])
    comparison = comparison.set_index('strategy')
    
    for metric in ['mcc', 'bacc', 'f1', 'auc']:
        bl = all_results['baseline'][f'{metric}_mean']
        cd = all_results['conditional'][f'{metric}_mean']
        diff = cd - bl
        logger.info(f"  {metric.upper():4s}: baseline={bl:.4f}  conditional={cd:.4f}  "
                     f"Δ={diff:+.4f} {'↑' if diff > 0 else '↓' if diff < 0 else '='}")
    
    # Subset comparison
    for stype in ['ionic_salt', 'organometallic']:
        bl_key = f'{stype}_mcc_mean'
        if bl_key in all_results['baseline'] and bl_key in all_results['conditional']:
            bl = all_results['baseline'][bl_key]
            cd = all_results['conditional'][bl_key]
            n = all_results['conditional'].get(f'{stype}_n_mean', 0)
            logger.info(f"  {stype} subset (n≈{n:.0f}): "
                         f"baseline={bl:.4f}  conditional={cd:.4f}  Δ={cd-bl:+.4f}")
    
    comparison.to_csv(output_dir / f"{endpoint}_comparison.csv")
    logger.info(f"\nResults saved to {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--endpoint', type=str, default='ames',
                        choices=['ames', 'invitro', 'invivo', 'all'])
    parser.add_argument('--output_dir', type=str, default='./comparison_results')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    endpoints = ['ames', 'invitro', 'invivo'] if args.endpoint == 'all' else [args.endpoint]
    
    all_endpoint_results = {}
    for ep in endpoints:
        try:
            df = load_data(data_dir, ep)
            results = run_single_endpoint(df, ep, output_dir)
            all_endpoint_results[ep] = results
        except Exception as e:
            logger.error(f"Failed for {ep}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    if all_endpoint_results:
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL SUMMARY")
        logger.info(f"{'='*60}")
        
        rows = []
        for ep, res in all_endpoint_results.items():
            for strategy in ['baseline', 'conditional']:
                if strategy in res:
                    rows.append({
                        'endpoint': ep,
                        'strategy': strategy,
                        'mcc': res[strategy]['mcc_mean'],
                        'bacc': res[strategy]['bacc_mean'],
                        'f1': res[strategy]['f1_mean'],
                        'n_features': res[strategy]['n_features'],
                    })
        
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_dir / 'overall_summary.csv', index=False)
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
