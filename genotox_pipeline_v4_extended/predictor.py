"""
predictor.py — 독립 예측 모듈 (Genotoxicity Multi-Endpoint Predictor)
=======================================================================
다른 환경에서 models/ 폴더와 함께 배포하여 사용.

필요 패키지:
    pip install numpy pandas scikit-learn xgboost lightgbm rdkit joblib

사용법 (CLI):
    python predictor.py --models-dir ./models --input compounds.csv --output results.csv

사용법 (Python):
    from predictor import GenotoxPredictor
    pred = GenotoxPredictor("./models")
    results = pred.predict_smiles(["CCO", "c1ccccc1N"])
    print(results)

입력: SMILES 컬럼이 있는 CSV 또는 SMILES 리스트
출력: 각 endpoint별 확률 + 예측 + 신뢰도 등급 + conformal 확실성
"""

import sys, json, logging, argparse, warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
lg = logging.getLogger("predictor")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    lg.error("RDKit is required: pip install rdkit")


# ═══════════════════════════════════════════════════════
#  Feature 추출 (자체 내장 — 외부 의존성 없음)
# ═══════════════════════════════════════════════════════

def _mol_from_smiles(smi: str):
    if not HAS_RDKIT or pd.isna(smi):
        return None
    return Chem.MolFromSmiles(str(smi))


def _extract_physchem(mol) -> dict:
    """기본 물리화학 descriptor 13종."""
    if mol is None:
        return {k: 0.0 for k in [
            "MW","LogP","TPSA","HBD","HBA","RotatableBonds","FractionCSP3",
            "AromaticRingCount","RingCount","HeteroatomCount","HeavyAtomCount",
            "FormalCharge","NumValenceElectrons"]}
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeteroatomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FormalCharge": Descriptors.FormalCharge(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
    }


def _extract_morgan_fp(mol, n_bits=1024) -> np.ndarray:
    """Morgan fingerprint."""
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    gen = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(gen, arr)
    return arr


def _compute_ad_similarity(fp_query: np.ndarray, fp_ref: np.ndarray) -> float:
    """Tanimoto 유사도 기반 AD: 학습 데이터 내 최근접 유사도."""
    if fp_ref is None or len(fp_ref) == 0:
        return 0.0
    # Jaccard distance → Tanimoto similarity
    intersection = np.minimum(fp_query, fp_ref).sum(axis=1)
    union = np.maximum(fp_query, fp_ref).sum(axis=1)
    tani = np.divide(intersection, union, where=union > 0,
                     out=np.zeros(len(fp_ref)))
    return float(tani.max())


# ═══════════════════════════════════════════════════════
#  Predictor 클래스
# ═══════════════════════════════════════════════════════

class GenotoxPredictor:
    """
    유전독성 다중 endpoint 예측기.

    Args:
        models_dir: 모델 아티팩트 디렉토리 (final_deploy.py 출력)
        pipeline_dir: 파이프라인 코드 디렉토리 (고급 feature 추출용, optional)
    """

    def __init__(self, models_dir: Union[str, Path],
                 pipeline_dir: Union[str, Path] = None):
        self.models_dir = Path(models_dir)
        self.endpoints = {}
        self._pipeline_available = False

        # 파이프라인 코드 import 시도 (고급 feature용)
        if pipeline_dir:
            sys.path.insert(0, str(Path(pipeline_dir)))
        try:
            from step4_feature_extraction import (
                extract_fg_features, extract_physchem_features,
                extract_fingerprint_features, select_fingerprint_bits,
            )
            from step4b_sa_features import extract_sa_features
            from step4c_delta_descriptors import (
                extract_delta_features, extract_extended_descriptors,
                extract_electronic_descriptors,
            )
            self._pipeline_available = True
            self._extract_fg = extract_fg_features
            self._extract_ph = extract_physchem_features
            self._extract_fp = extract_fingerprint_features
            self._extract_sa = extract_sa_features
            self._extract_delta = extract_delta_features
            self._extract_extended = extract_extended_descriptors
            self._extract_electronic = extract_electronic_descriptors
            self._select_fp = select_fingerprint_bits
            lg.info("Pipeline modules loaded — full feature extraction available")
        except ImportError as e:
            lg.info(f"Pipeline modules not found ({e}) — using built-in features")

        # 모델 로드
        self._load_models()

    def _load_models(self):
        for ep_dir in sorted(self.models_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "metadata.json"
            model_path = ep_dir / "model.joblib"
            fcols_path = ep_dir / "feature_columns.json"

            if not all(p.exists() for p in [meta_path, model_path, fcols_path]):
                continue

            with open(meta_path) as f:
                meta = json.load(f)
            with open(fcols_path) as f:
                fcols = json.load(f)

            model = joblib.load(model_path)

            # Conformal calibration
            conf_path = ep_dir / "conformal.json"
            conformal = None
            if conf_path.exists():
                with open(conf_path) as f:
                    conformal = json.load(f)

            # AD reference
            ad_path = ep_dir / "ad_reference.npz"
            ad_ref = None
            if ad_path.exists():
                ad_ref = np.load(ad_path)["fingerprints"]

            ep = meta["endpoint"]
            self.endpoints[ep] = {
                "model": model,
                "meta": meta,
                "feature_cols": fcols,
                "conformal": conformal,
                "ad_ref": ad_ref,
            }
            lg.info(f"  Loaded: {ep} ({meta['model_type']}, "
                    f"{len(fcols)} features, t={meta['optimal_threshold']:.3f})")

        lg.info(f"  Total: {len(self.endpoints)} endpoints ready")

    def predict_smiles(self, smiles_list: List[str],
                        alpha: float = 0.05) -> pd.DataFrame:
        """
        SMILES 리스트로 예측 수행.

        Returns:
            DataFrame with columns per endpoint:
            {ep}_proba, {ep}_pred, {ep}_confidence, {ep}_conformal
        """
        if not HAS_RDKIT:
            raise ImportError("RDKit required")

        results = pd.DataFrame({"SMILES": smiles_list})
        mols = [_mol_from_smiles(s) for s in smiles_list]

        for ep, info in self.endpoints.items():
            meta = info["meta"]
            model = info["model"]
            fcols = info["feature_cols"]

            # ── Feature 추출 ──
            feat_df = self._build_features(smiles_list, mols, ep, info)
            feat_df = feat_df.reindex(columns=fcols, fill_value=0)
            X = feat_df.values.astype(np.float32)

            # ── 예측 ──
            proba = model.predict_proba(X)[:, 1]
            threshold = meta["optimal_threshold"]
            pred = (proba >= threshold).astype(int)

            # ── AD 신뢰도 ──
            confidence = []
            fp_1024 = np.array([_extract_morgan_fp(m, 1024) for m in mols])
            for i in range(len(smiles_list)):
                sim = _compute_ad_similarity(fp_1024[i], info["ad_ref"])
                if sim >= 0.6:
                    confidence.append("HIGH")
                elif sim >= 0.4:
                    confidence.append("MEDIUM")
                else:
                    confidence.append("LOW")

            # ── Conformal 확실성 ──
            conformal_labels = []
            if info["conformal"]:
                alpha_key = str(alpha)
                conf_data = info["conformal"]["alphas"].get(alpha_key)
                if conf_data:
                    q = conf_data["quantile"]
                    p0 = 1 - proba  # P(class=0)
                    p1 = proba      # P(class=1)
                    for j in range(len(smiles_list)):
                        s0 = (1 - p0[j]) <= q
                        s1 = (1 - p1[j]) <= q
                        if s0 and s1:
                            conformal_labels.append("UNCERTAIN")
                        elif not s0 and not s1:
                            conformal_labels.append("EMPTY")
                        else:
                            conformal_labels.append("CERTAIN")
                else:
                    conformal_labels = ["N/A"] * len(smiles_list)
            else:
                conformal_labels = ["N/A"] * len(smiles_list)

            # ── 종합 판정 ──
            predictions = []
            for j in range(len(smiles_list)):
                if pred[j] == 1:
                    predictions.append("Positive")
                else:
                    predictions.append("Negative")

            results[f"{ep}_proba"] = np.round(proba, 4)
            results[f"{ep}_pred"] = predictions
            results[f"{ep}_confidence"] = confidence
            results[f"{ep}_conformal"] = conformal_labels
            results[f"{ep}_threshold"] = threshold

        return results

    def _build_features(self, smiles_list, mols, ep, info):
        """Feature 추출 — 파이프라인 코드 있으면 사용, 없으면 내장 기능."""
        meta = info["meta"]
        feature_mode = meta["feature_mode"]

        if self._pipeline_available:
            return self._build_features_pipeline(smiles_list, ep, meta)
        else:
            return self._build_features_builtin(smiles_list, mols, info)

    def _build_features_pipeline(self, smiles_list, ep, meta):
        """파이프라인 모듈로 full feature 추출."""
        df = pd.DataFrame({
            "SMILES": smiles_list,
            "_analysis_smiles": smiles_list,
            "label": 0,
            "endpoint": ep,
        })

        fg = self._extract_fg(df, ep)
        ph = self._extract_ph(df, ep)

        try:
            sa = self._extract_sa(df, ep)
        except Exception:
            sa = pd.DataFrame(index=df.index)

        feature_mode = meta["feature_mode"]

        if feature_mode == "extended":
            try:
                delta = self._extract_delta(df, raw_smi_col="SMILES",
                                             pre_smi_col="SMILES", endpoint=ep)
            except Exception:
                delta = pd.DataFrame(index=df.index)
            try:
                elec = self._extract_electronic(df, ep)
            except Exception:
                elec = pd.DataFrame(index=df.index)
            feat = pd.concat([fg, ph, sa, delta, elec], axis=1)

        elif feature_mode.startswith("broad_fp"):
            nbits = int(feature_mode.replace("broad_fp", ""))
            fp = self._extract_fp(df, ep, n_bits=nbits)
            feat = pd.concat([fg, ph, sa, fp], axis=1)

        else:
            feat = pd.concat([fg, ph, sa], axis=1)

        for c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        return feat.fillna(0)

    def _build_features_builtin(self, smiles_list, mols, info):
        """내장 기능으로 기본 feature만 추출 (파이프라인 없을 때)."""
        fcols = info["feature_cols"]
        records = []

        for mol in mols:
            rec = _extract_physchem(mol)
            # Morgan FP (가장 큰 FP 크기 감지)
            fp_cols = [c for c in fcols if c.startswith("fp_")]
            if fp_cols:
                max_bit = max(int(c.split("_")[1]) for c in fp_cols) + 1
                fp_size = max(max_bit, 1024)
                fp = _extract_morgan_fp(mol, fp_size)
                for c in fp_cols:
                    bit = int(c.split("_")[1])
                    rec[c] = int(fp[bit]) if bit < fp_size else 0
            records.append(rec)

        df = pd.DataFrame(records)
        lg.warning(f"  Built-in features: {len(df.columns)} cols "
                   f"(full pipeline recommended for accuracy)")
        return df

    def predict_csv(self, input_path: str, output_path: str = None,
                     alpha: float = 0.05) -> pd.DataFrame:
        """CSV 파일 입력 → 예측 → CSV 출력."""
        df = pd.read_csv(input_path)
        smi_col = None
        for c in df.columns:
            if c.strip().upper() == "SMILES":
                smi_col = c
                break
        if smi_col is None:
            raise ValueError("CSV must have a 'SMILES' column")

        smiles = df[smi_col].astype(str).tolist()
        results = self.predict_smiles(smiles, alpha=alpha)

        # 원본 컬럼 보존
        for c in df.columns:
            if c != smi_col and c not in results.columns:
                results[c] = df[c].values

        if output_path:
            results.to_csv(output_path, index=False)
            lg.info(f"  Saved: {output_path} ({len(results)} compounds)")

        return results

    def summary(self) -> str:
        """모델 요약 문자열."""
        lines = ["Genotoxicity Multi-Endpoint Predictor", "=" * 40]
        for ep, info in self.endpoints.items():
            meta = info["meta"]
            conf = info["conformal"]
            c05 = conf["alphas"]["0.05"] if conf else {}
            lines.append(
                f"  {ep}: {meta['model_type']} | "
                f"t={meta['optimal_threshold']:.3f} | "
                f"MCC={c05.get('mcc_certain', '?')} "
                f"(coverage={c05.get('coverage', '?')})"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Genotoxicity Predictor — predict new compounds")
    p.add_argument("--models-dir", required=True,
                   help="모델 아티팩트 디렉토리 (final_deploy.py 출력)")
    p.add_argument("--input", required=True,
                   help="입력 CSV (SMILES 컬럼 필수)")
    p.add_argument("--output", default="predictions.csv",
                   help="출력 CSV (기본: predictions.csv)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Conformal alpha (기본: 0.05)")
    p.add_argument("--pipeline-dir", default=None,
                   help="파이프라인 코드 디렉토리 (고급 feature용)")
    args = p.parse_args()

    pred = GenotoxPredictor(args.models_dir, args.pipeline_dir)
    print(pred.summary())
    print()
    results = pred.predict_csv(args.input, args.output, alpha=args.alpha)
    print(f"\nPredicted {len(results)} compounds → {args.output}")
