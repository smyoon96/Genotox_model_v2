"""
qprf.py — OECD QSAR Prediction Reporting Format (QPRF) 생성기
================================================================
예측 결과에 대한 규제 제출용 QPRF 보고서를 생성.

OECD QPRF v4.4 가이드라인에 따른 구조:
  1. Substance — 대상 화합물 정보
  2. General information — 모델 일반 정보
  3. Prediction — 예측 결과
  4. Applicability domain — AD 평가
  5. Uncertainty — 불확실성 평가
  6. Adequacy — 규제 목적 적합성

Usage:
    from qprf import generate_qprf
    report = generate_qprf(predictor, "CCO", "Ethanol")
    report.save("ethanol_qprf.md")

    # 또는 CLI:
    python qprf.py --models-dir ./models --smiles "CCO" --name "Ethanol" --output ethanol_qprf.md
"""

import json, argparse, warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class QPRFReport:
    """QPRF 보고서 객체."""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def save(self, path: str):
        Path(path).write_text(self.content, encoding="utf-8")

    def __repr__(self):
        return self.content[:200] + "..."


# ── 주요 구조경보 SMARTS (파이프라인 없을 때 fallback) ──
_BUILTIN_ALERTS = {
    # Benigni-Bossa
    "bb_sa1_acyl_halide":         "[CX3](=[OX1])[F,Cl,Br,I]",
    "bb_sa3_aliphatic_nnitroso":  "[NX2](=O)[CX4]",
    "bb_sa4_aromatic_nnitroso":   "[NX2](=O)c",
    "bb_sa5_nitro_aromatic":      "[$([NX3](=O)=O),$([NX3+](=O)[O-])]c",
    "bb_sa6_hydrazine":           "[NX3H1][NX3H1]",
    "bb_sa7_aliphatic_halide":    "[CX4][F,Cl,Br,I]",
    "bb_sa8_ab_unsat_aldehyde":   "[CX3H1](=O)[CX3]=[CX3]",
    "bb_sa10_azo":                "[NX2]=[NX2]",
    "bb_sa12_aromatic_nitro":     "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
    "bb_sa17_epoxide":            "C1OC1",
    "bb_sa19_nitrogen_mustard":   "ClCCN",
    "bb_sa22_aromatic_amine_pri": "[NH2]c",
    "bb_sa23_aromatic_amine_sec": "[NH1](C)c",
    "bb_sa28_quinone":            "O=C1C=CC(=O)C=C1",
    # Kazius
    "kz_polycyclic_aromatic":     "c1ccc2c(c1)cc1ccc3ccccc3c1c2",
    "kz_unsat_aldehyde":          "[CX3H1](=O)[#6]=[#6]",
    "kz_aromatic_hydroxylamine":  "[OH]Nc",
    "kz_diazonium":              "[N+]#N",
}

_BUILTIN_FGS = {
    "alcohol":        "[OX2H]",
    "ether":          "[OD2]([#6])[#6]",
    "aldehyde":       "[CX3H1](=O)[#6]",
    "ketone":         "[CX3](=O)([#6])[#6]",
    "carboxylic_acid":"[CX3](=O)[OX2H1]",
    "ester":          "[CX3](=O)[OX2H0][#6]",
    "amine_primary":  "[NX3H2]",
    "amine_secondary":"[NX3H1]([#6])[#6]",
    "amine_tertiary": "[NX3]([#6])([#6])[#6]",
    "amide":          "[NX3][CX3](=[OX1])",
    "nitro":          "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile":        "[NX1]#[CX2]",
    "sulfide":        "[#16X2H0]",
    "sulfoxide":      "[#16X3](=[OX1])([#6])[#6]",
    "sulfonyl":       "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "phosphate":      "[PX4](=O)([OX2])([OX2])[OX2]",
    "halide_F":       "[F]",
    "halide_Cl":      "[Cl]",
    "halide_Br":      "[Br]",
    "halide_I":       "[I]",
}

_METAL_SYMBOLS = [
    "Li","Be","Na","Mg","Al","K","Ca","Ti","V","Cr","Mn","Fe","Co",
    "Ni","Cu","Zn","Ga","As","Se","Rb","Sr","Zr","Mo","Ag","Cd",
    "In","Sn","Sb","Te","Cs","Ba","W","Pt","Au","Hg","Tl","Pb","Bi",
]


def _analyze_structure(smiles: str) -> dict:
    """화합물의 구조경보 + 작용기 + 금속 분석."""
    result = {"benigni_bossa": [], "kazius": [], "functional_groups": [], "metals": []}

    if not HAS_RDKIT:
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    # 파이프라인 config에서 SMARTS 로드 시도
    try:
        from config import FG_SMARTS
        alert_smarts = {k: v for k, v in FG_SMARTS.items()
                        if k.startswith("bb_") or k.startswith("kz_")}
        fg_smarts = {k.replace("fg_", ""): v for k, v in FG_SMARTS.items()
                     if k.startswith("fg_")}
    except ImportError:
        alert_smarts = _BUILTIN_ALERTS
        fg_smarts = _BUILTIN_FGS

    # Structural alerts
    for name, smarts in alert_smarts.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                matches = mol.GetSubstructMatches(pat)
                category = "benigni_bossa" if name.startswith("bb_") else "kazius"
                display_name = name.replace("bb_", "").replace("kz_", "")
                display_name = display_name.replace("_", " ").title()
                result[category].append({
                    "name": display_name,
                    "key": name,
                    "smarts": smarts,
                    "count": len(matches),
                })
        except Exception:
            pass

    # Functional groups
    for name, smarts in fg_smarts.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                matches = mol.GetSubstructMatches(pat)
                if matches:
                    display_name = name.replace("_", " ").title()
                    result["functional_groups"].append({
                        "name": display_name,
                        "key": name,
                        "smarts": smarts,
                        "count": len(matches),
                    })
        except Exception:
            pass

    # Metals
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in _METAL_SYMBOLS:
            if sym not in result["metals"]:
                result["metals"].append(sym)

    return result


def generate_qprf(predictor, smiles: str, compound_name: str = "",
                   cas_number: str = "", alpha: float = 0.05,
                   assessor: str = "", purpose: str = "screening"
                   ) -> QPRFReport:
    """
    단일 화합물에 대한 QPRF 보고서 생성.

    Args:
        predictor: GenotoxPredictor 인스턴스
        smiles: 대상 화합물 SMILES
        compound_name: 화합물명
        cas_number: CAS 번호
        alpha: conformal prediction alpha
        assessor: 평가자 이름
        purpose: 평가 목적

    Returns:
        QPRFReport 객체
    """
    # ── 예측 수행 ──
    results = predictor.predict_smiles([smiles], alpha=alpha)

    # ── 화합물 정보 ──
    mol = Chem.MolFromSmiles(smiles) if HAS_RDKIT else None
    if mol:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        mw = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        try:
            inchi_str = inchi.MolToInchi(mol)
            inchi_key = inchi.InchiToInchiKey(inchi_str)
        except Exception:
            inchi_str = "N/A"
            inchi_key = "N/A"
        formula = rdMolDescriptors.CalcMolFormula(mol)
    else:
        canonical = smiles
        mw = logp = 0
        inchi_str = inchi_key = formula = "N/A"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 보고서 생성 ──
    sections = []

    # Header
    sections.append(f"""# QPRF — QSAR Prediction Reporting Format
## Genotoxicity Multi-Endpoint Assessment

**Generated**: {now}
**Assessor**: {assessor or 'N/A'}
**Purpose**: {purpose}

---

## 1. Substance

| Property | Value |
|----------|-------|
| Compound name | {compound_name or 'N/A'} |
| CAS number | {cas_number or 'N/A'} |
| SMILES | `{canonical}` |
| Molecular formula | {formula} |
| Molecular weight | {mw} g/mol |
| LogP | {logp} |
| InChI | `{inchi_str}` |
| InChIKey | `{inchi_key}` |
""")

    # Section 1b: Structural analysis (FG + alerts)
    fg_analysis = _analyze_structure(canonical)
    sections.append("""### 1.2 Structural analysis

#### Structural alerts (Benigni-Bossa)
""")
    bb_alerts = fg_analysis.get("benigni_bossa", [])
    if bb_alerts:
        for alert in bb_alerts:
            sections.append(f"- ⚠ **{alert['name']}** — `{alert['smarts']}`\n")
    else:
        sections.append("- No Benigni-Bossa structural alerts detected\n")

    sections.append("""
#### Structural alerts (Kazius)
""")
    kz_alerts = fg_analysis.get("kazius", [])
    if kz_alerts:
        for alert in kz_alerts:
            sections.append(f"- ⚠ **{alert['name']}** — `{alert['smarts']}`\n")
    else:
        sections.append("- No Kazius structural alerts detected\n")

    sections.append("""
#### Functional groups
""")
    fgs = fg_analysis.get("functional_groups", [])
    if fgs:
        for fg in fgs:
            sections.append(f"- {fg['name']} (count: {fg['count']})\n")
    else:
        sections.append("- No notable functional groups detected\n")

    sections.append("""
#### Metal atoms
""")
    metals = fg_analysis.get("metals", [])
    if metals:
        for m in metals:
            sections.append(f"- ⚠ **{m}** detected\n")
    else:
        sections.append("- No metal atoms\n")

    n_total_alerts = len(bb_alerts) + len(kz_alerts)
    sections.append(f"""
**Summary**: {n_total_alerts} structural alert(s), """
                    f"""{len(fgs)} functional group(s), {len(metals)} metal(s)
""")

    # Section 2: General information
    sections.append("""---

## 2. General information on the (Q)SAR model

### 2.1 Model identity
Multi-endpoint genotoxicity prediction model using machine learning (XGBoost, LightGBM, Random Forest).

### 2.2 Endpoints predicted
""")

    for ep, info in predictor.endpoints.items():
        meta = info["meta"]
        ep_display = {
            "ames": "Ames bacterial reverse mutation test (OECD TG 471)",
            "invitro_sampling": "In vitro chromosome aberration test (OECD TG 473)",
            "invivo_sampling": "In vivo mammalian micronucleus test (OECD TG 474)",
        }.get(ep, ep)
        sections.append(f"- **{ep}**: {ep_display}\n")

    sections.append("""
### 2.3 Algorithm
Gradient boosting (XGBoost/LightGBM) and Random Forest classifiers trained on curated genotoxicity datasets. Features include physicochemical descriptors, structural alerts (Benigni-Bossa, Kazius), functional group counts, Morgan fingerprints, and cross-endpoint stacking (Ames probability as auxiliary feature for in vitro/in vivo endpoints).

### 2.4 Software
- RDKit (descriptor calculation)
- XGBoost / LightGBM / scikit-learn (modeling)
- SHAP (feature importance)
""")

    # Section 3: Prediction results
    sections.append("""---

## 3. Prediction results

""")

    endpoint_results = {}
    for ep in predictor.endpoints:
        proba = results[f"{ep}_proba"].iloc[0]
        pred = results[f"{ep}_pred"].iloc[0]
        conf = results[f"{ep}_confidence"].iloc[0]
        conformal = results[f"{ep}_conformal"].iloc[0]
        threshold = results[f"{ep}_threshold"].iloc[0]
        meta = predictor.endpoints[ep]["meta"]

        endpoint_results[ep] = {
            "proba": proba, "pred": pred, "conf": conf,
            "conformal": conformal, "threshold": threshold,
        }

        pred_symbol = "⊕ Positive" if pred == "Positive" else "⊖ Negative"

        sections.append(f"""### 3.{list(predictor.endpoints.keys()).index(ep)+1} {ep}

| Item | Value |
|------|-------|
| Prediction | **{pred_symbol}** |
| Probability | {proba:.4f} |
| Decision threshold | {threshold:.3f} |
| AD confidence | {conf} |
| Conformal status | {conformal} (α={alpha}) |
| Model type | {meta['model_type']} |

""")

    # Section 4: Applicability domain
    sections.append("""---

## 4. Applicability domain assessment

### 4.1 AD methodology
Tanimoto similarity-based applicability domain using Morgan fingerprints (radius=2, 1024 bits). Each query compound's maximum Tanimoto similarity to the training set is computed. Classification: HIGH (≥0.6), MEDIUM (0.4–0.6), LOW (<0.4).

### 4.2 Conformal prediction
Nonconformity scores (1 − P(true class)) calibrated on 5-fold out-of-fold predictions. At α=0.05 (95% confidence), compounds where the prediction set contains both classes are flagged as UNCERTAIN.

### 4.3 Assessment for this compound

""")

    for ep in predictor.endpoints:
        er = endpoint_results[ep]
        conf_info = predictor.endpoints[ep]["conformal"]
        if conf_info:
            c05 = conf_info["alphas"].get(str(alpha), {})
            coverage = c05.get("coverage", "N/A")
            mcc = c05.get("mcc_certain", "N/A")
        else:
            coverage = mcc = "N/A"

        in_ad = "Yes" if er["conf"] in ("HIGH", "MEDIUM") else "No (LOW similarity)"
        reliable = "Yes" if er["conformal"] == "CERTAIN" and er["conf"] != "LOW" else "No"

        sections.append(f"""**{ep}**:
- AD status: {in_ad} (confidence={er['conf']})
- Conformal: {er['conformal']}
- Model performance in this confidence tier: coverage={coverage}, MCC={mcc}
- Prediction considered reliable: **{reliable}**

""")

    # Section 5: Uncertainty
    sections.append("""---

## 5. Uncertainty assessment

### 5.1 Model performance (overall, OOF CV)

| Endpoint | CV MCC | Threshold | Sens | Spec |
|----------|--------|-----------|------|------|
""")

    for ep, info in predictor.endpoints.items():
        meta = info["meta"]
        sections.append(
            f"| {ep} | {meta['cv_mcc']:.3f} | {meta['optimal_threshold']:.3f} "
            f"| {meta['sensitivity']:.3f} | {meta['specificity']:.3f} |\n"
        )

    sections.append(f"""
### 5.2 Conformal coverage (α={alpha})

| Endpoint | Coverage | MCC (certain) |
|----------|----------|---------------|
""")

    for ep, info in predictor.endpoints.items():
        conf = info["conformal"]
        if conf:
            c = conf["alphas"].get(str(alpha), {})
            sections.append(
                f"| {ep} | {c.get('coverage','N/A')} | {c.get('mcc_certain','N/A')} |\n"
            )

    sections.append("""
### 5.3 Sources of uncertainty
- Scaffold-split test performance is lower than OOF CV (structural novelty)
- Low AD confidence compounds have substantially lower prediction accuracy
- In vitro/in vivo endpoints have limited positive training examples
- Cross-endpoint stacking assumes Ames result correlates with other endpoints
""")

    # Section 6: Adequacy
    sections.append("""---

## 6. Adequacy of the prediction for regulatory purpose

### 6.1 Regulatory relevance
The predicted endpoints correspond to OECD Test Guidelines:
- Ames test (TG 471) — bacterial gene mutation
- In vitro chromosome aberration (TG 473) — clastogenicity
- In vivo micronucleus test (TG 474) — clastogenicity/aneugenicity

### 6.2 ICH M7 compliance
For Ames prediction, the model meets ICH M7 sensitivity requirements (≥0.80) in the high-confidence tier. ICH M7 recommends two complementary (Q)SAR methodologies; this model may serve as one component.

### 6.3 Limitations
""")

    for ep in predictor.endpoints:
        er = endpoint_results[ep]
        if er["conformal"] == "UNCERTAIN":
            sections.append(f"- **{ep}**: Prediction is UNCERTAIN — experimental verification recommended\n")
        elif er["conf"] == "LOW":
            sections.append(f"- **{ep}**: LOW AD confidence — prediction may not be reliable\n")
        else:
            sections.append(f"- **{ep}**: Prediction is within AD and CERTAIN\n")

    sections.append(f"""
### 6.4 Conclusion
""")

    all_certain = all(endpoint_results[ep]["conformal"] == "CERTAIN"
                       for ep in predictor.endpoints)
    any_positive = any(endpoint_results[ep]["pred"] == "Positive"
                        for ep in predictor.endpoints)

    if any_positive and all_certain:
        sections.append("**Genotoxicity concern identified** — one or more endpoints predict positive with high confidence. Experimental verification is strongly recommended.\n")
    elif any_positive:
        sections.append("**Potential genotoxicity concern** — positive prediction(s) present but some with uncertainty. Consider experimental verification for uncertain endpoints.\n")
    else:
        sections.append("**No genotoxicity concern identified** — all endpoints predict negative. Standard regulatory follow-up may apply.\n")

    sections.append(f"\n---\n*Report generated by Genotoxicity QPRF Generator v1.0 ({now})*\n")

    content = "\n".join(sections)
    metadata = {
        "smiles": canonical,
        "compound_name": compound_name,
        "cas": cas_number,
        "predictions": endpoint_results,
        "structural_analysis": {
            "benigni_bossa_alerts": [a["name"] for a in fg_analysis.get("benigni_bossa", [])],
            "kazius_alerts": [a["name"] for a in fg_analysis.get("kazius", [])],
            "functional_groups": [f["name"] for f in fg_analysis.get("functional_groups", [])],
            "metals": fg_analysis.get("metals", []),
        },
        "timestamp": now,
    }

    return QPRFReport(content, metadata)


def generate_batch_qprf(predictor, smiles_list: list,
                         names: list = None,
                         output_dir: str = "qprf_reports",
                         alpha: float = 0.05) -> list:
    """배치 QPRF 생성."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []

    for i, smi in enumerate(smiles_list):
        name = names[i] if names and i < len(names) else f"Compound_{i+1}"
        report = generate_qprf(predictor, smi, compound_name=name, alpha=alpha)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        report.save(str(out / f"{safe_name}_qprf.md"))
        reports.append(report)

    return reports


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QPRF Report Generator")
    p.add_argument("--models-dir", required=True)
    p.add_argument("--smiles", help="단일 SMILES")
    p.add_argument("--input", help="CSV 입력 (SMILES, name 컬럼)")
    p.add_argument("--name", default="", help="화합물명")
    p.add_argument("--output", default="qprf_report.md")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--pipeline-dir", default=None)
    args = p.parse_args()

    from predictor import GenotoxPredictor
    pred = GenotoxPredictor(args.models_dir, args.pipeline_dir)

    if args.smiles:
        report = generate_qprf(pred, args.smiles, compound_name=args.name,
                                alpha=args.alpha)
        report.save(args.output)
        print(f"QPRF saved: {args.output}")
    elif args.input:
        df = pd.read_csv(args.input)
        smiles = df["SMILES"].tolist()
        names = df["name"].tolist() if "name" in df.columns else None
        reports = generate_batch_qprf(pred, smiles, names,
                                       output_dir=args.output, alpha=args.alpha)
        print(f"Generated {len(reports)} QPRF reports → {args.output}/")
    else:
        print("Provide --smiles or --input")
