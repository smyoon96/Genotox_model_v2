# QPRF — QSAR Prediction Reporting Format
## Genotoxicity Multi-Endpoint Assessment

**Generated**: 2026-06-02 14:47
**Assessor**: N/A
**Purpose**: screening

---

## 1. Substance

| Property | Value |
|----------|-------|
| Compound name | Aniline |
| CAS number | N/A |
| SMILES | `Nc1ccccc1` |
| Molecular formula | C6H7N |
| Molecular weight | 93.13 g/mol |
| LogP | 1.27 |
| InChI | `InChI=1S/C6H7N/c7-6-4-2-1-3-5-6/h1-5H,7H2` |
| InChIKey | `PAYRUJLWNCNPSJ-UHFFFAOYSA-N` |

### 1.2 Structural analysis

#### Structural alerts (Benigni-Bossa)

- ⚠ **Sa22 Aromatic Amine Pri** — `[c][NX3;H2;!$(NC=O)]`


#### Structural alerts (Kazius)

- No Kazius structural alerts detected


#### Functional groups

- Primary Amine (count: 1)


#### Metal atoms

- No metal atoms


**Summary**: 1 structural alert(s), 1 functional group(s), 0 metal(s)

---

## 2. General information on the (Q)SAR model

### 2.1 Model identity
Multi-endpoint genotoxicity prediction model using machine learning (XGBoost, LightGBM, Random Forest).

### 2.2 Endpoints predicted

- **ames**: Ames bacterial reverse mutation test (OECD TG 471)

- **invitro_sampling**: In vitro chromosome aberration test (OECD TG 473)

- **invivo_sampling**: In vivo mammalian micronucleus test (OECD TG 474)


### 2.3 Algorithm
Gradient boosting (XGBoost/LightGBM) and Random Forest classifiers trained on curated genotoxicity datasets. Features include physicochemical descriptors, structural alerts (Benigni-Bossa, Kazius), functional group counts, Morgan fingerprints, and cross-endpoint stacking (Ames probability as auxiliary feature for in vitro/in vivo endpoints).

### 2.4 Software
- RDKit (descriptor calculation)
- XGBoost / LightGBM / scikit-learn (modeling)
- SHAP (feature importance)

---

## 3. Prediction results


### 3.1 ames

| Item | Value |
|------|-------|
| Prediction | **⊕ Positive** |
| Probability | 0.6937 |
| Decision threshold | 0.230 |
| AD confidence | HIGH |
| Conformal status | UNCERTAIN (α=0.05) |
| Model type | xgb |


### 3.2 invitro_sampling

| Item | Value |
|------|-------|
| Prediction | **⊕ Positive** |
| Probability | 0.8917 |
| Decision threshold | 0.495 |
| AD confidence | LOW |
| Conformal status | CERTAIN (α=0.05) |
| Model type | rf |


### 3.3 invivo_sampling

| Item | Value |
|------|-------|
| Prediction | **⊕ Positive** |
| Probability | 0.9869 |
| Decision threshold | 0.265 |
| AD confidence | MEDIUM |
| Conformal status | CERTAIN (α=0.05) |
| Model type | xgb |


---

## 4. Applicability domain assessment

### 4.1 AD methodology
Tanimoto similarity-based applicability domain using Morgan fingerprints (radius=2, 1024 bits). Each query compound's maximum Tanimoto similarity to the training set is computed. Classification: HIGH (≥0.6), MEDIUM (0.4–0.6), LOW (<0.4).

### 4.2 Conformal prediction
Nonconformity scores (1 − P(true class)) calibrated on 5-fold out-of-fold predictions. At α=0.05 (95% confidence), compounds where the prediction set contains both classes are flagged as UNCERTAIN.

### 4.3 Assessment for this compound


**ames**:
- AD status: Yes (confidence=HIGH)
- Conformal: UNCERTAIN
- Model performance in this confidence tier: coverage=0.8316, MCC=0.8543
- Prediction considered reliable: **No**


**invitro_sampling**:
- AD status: No (LOW similarity) (confidence=LOW)
- Conformal: CERTAIN
- Model performance in this confidence tier: coverage=0.3359, MCC=0.7116
- Prediction considered reliable: **No**


**invivo_sampling**:
- AD status: Yes (confidence=MEDIUM)
- Conformal: CERTAIN
- Model performance in this confidence tier: coverage=0.5974, MCC=0.7665
- Prediction considered reliable: **Yes**


---

## 5. Uncertainty assessment

### 5.1 Model performance (overall, OOF CV)

| Endpoint | CV MCC | Threshold | Sens | Spec |
|----------|--------|-----------|------|------|

| ames | 0.743 | 0.230 | 0.930 | 0.813 |

| invitro_sampling | 0.502 | 0.495 | 0.852 | 0.653 |

| invivo_sampling | 0.545 | 0.265 | 0.792 | 0.779 |


### 5.2 Conformal coverage (α=0.05)

| Endpoint | Coverage | MCC (certain) |
|----------|----------|---------------|

| ames | 0.8316 | 0.8543 |

| invitro_sampling | 0.3359 | 0.7116 |

| invivo_sampling | 0.5974 | 0.7665 |


### 5.3 Sources of uncertainty
- Scaffold-split test performance is lower than OOF CV (structural novelty)
- Low AD confidence compounds have substantially lower prediction accuracy
- In vitro/in vivo endpoints have limited positive training examples
- Cross-endpoint stacking assumes Ames result correlates with other endpoints

---

## 6. Adequacy of the prediction for regulatory purpose

### 6.1 Regulatory relevance
The predicted endpoints correspond to OECD Test Guidelines:
- Ames test (TG 471) — bacterial gene mutation
- In vitro chromosome aberration (TG 473) — clastogenicity
- In vivo micronucleus test (TG 474) — clastogenicity/aneugenicity

### 6.2 ICH M7 compliance
For Ames prediction, the model meets ICH M7 sensitivity requirements (≥0.80) in the high-confidence tier. ICH M7 recommends two complementary (Q)SAR methodologies; this model may serve as one component.

### 6.3 Limitations

- **ames**: Prediction is UNCERTAIN — experimental verification recommended

- **invitro_sampling**: LOW AD confidence — prediction may not be reliable

- **invivo_sampling**: Prediction is within AD and CERTAIN


### 6.4 Conclusion

**Potential genotoxicity concern** — positive prediction(s) present but some with uncertainty. Consider experimental verification for uncertain endpoints.


---
*Report generated by Genotoxicity QPRF Generator v1.0 (2026-06-02 14:47)*
